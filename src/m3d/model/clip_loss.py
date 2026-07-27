"""Distributed contrastive objective for M3D-CLIP.

The model in :mod:`m3d.model.clip` deliberately returns only local image/text
representations.  This module owns every distributed collective used by the
CLIP objective, making label offsets, gradient-preserving gathers and retrieval
metrics explicit and testable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


class CLIPLossError(RuntimeError):
    """Raised when a contrastive batch violates the distributed contract."""


class _AllGatherWithGrad(torch.autograd.Function):
    """All-gather equal-shaped tensors while preserving gradients.

    Backward sums the gradient contribution for the local shard across ranks.
    This is the behaviour required when every rank computes a loss involving
    all gathered features.
    """

    @staticmethod
    def forward(ctx: Any, tensor: Tensor, group: Any) -> tuple[Tensor, ...]:
        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        outputs = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(outputs, tensor.contiguous(), group=group)
        return tuple(outputs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        local_grad = grad_outputs[ctx.rank].contiguous()
        dist.all_reduce(local_grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return local_grad, None


def _distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _validate_features(image_features: Tensor, text_features: Tensor) -> None:
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise CLIPLossError(
            "image_features and text_features must both have shape [B, C]."
        )
    if image_features.shape != text_features.shape:
        raise CLIPLossError(
            "Image/text features must have identical local shapes; got "
            f"{tuple(image_features.shape)} and {tuple(text_features.shape)}."
        )
    if image_features.shape[0] == 0:
        raise CLIPLossError("A contrastive batch cannot be empty.")
    if image_features.device != text_features.device:
        raise CLIPLossError("Image/text features must be on the same device.")
    if not image_features.is_floating_point() or not text_features.is_floating_point():
        raise CLIPLossError("Contrastive features must be floating-point tensors.")


def _assert_equal_local_batch_size(local_batch: int, *, group: Any = None) -> None:
    if not _distributed_ready():
        return
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    value = torch.tensor([local_batch], device=device, dtype=torch.int64)
    minimum = value.clone()
    maximum = value.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    if int(minimum.item()) != int(maximum.item()):
        raise CLIPLossError(
            "Gradient-preserving CLIP all-gather requires equal local batch sizes "
            f"on every rank; observed min={int(minimum.item())}, max={int(maximum.item())}."
        )


def gather_features(
    image_features: Tensor,
    text_features: Tensor,
    *,
    gather_with_grad: bool = True,
    group: Any = None,
) -> tuple[Tensor, Tensor]:
    """Gather image/text features in rank order.

    When ``gather_with_grad`` is false, the local rank's gathered slice is
    replaced with the original tensor so local gradients are retained while
    remote slices are treated as constants.
    """

    _validate_features(image_features, text_features)
    if not _distributed_ready() or dist.get_world_size(group) == 1:
        return image_features, text_features

    local_batch = int(image_features.shape[0])
    _assert_equal_local_batch_size(local_batch, group=group)
    rank = dist.get_rank(group)

    if gather_with_grad:
        all_images = torch.cat(_AllGatherWithGrad.apply(image_features, group), dim=0)
        all_texts = torch.cat(_AllGatherWithGrad.apply(text_features, group), dim=0)
        return all_images, all_texts

    image_parts = [torch.empty_like(image_features) for _ in range(dist.get_world_size(group))]
    text_parts = [torch.empty_like(text_features) for _ in range(dist.get_world_size(group))]
    dist.all_gather(image_parts, image_features.detach(), group=group)
    dist.all_gather(text_parts, text_features.detach(), group=group)
    image_parts[rank] = image_features
    text_parts[rank] = text_features
    return torch.cat(image_parts, dim=0), torch.cat(text_parts, dim=0)


@dataclass(frozen=True, slots=True)
class CLIPLossOutput:
    loss: Tensor
    image_loss: Tensor
    text_loss: Tensor
    logits_per_image: Tensor
    logits_per_text: Tensor
    labels: Tensor
    local_batch_size: int
    global_batch_size: int

    def detached_metrics(self) -> dict[str, Tensor]:
        with torch.no_grad():
            image_accuracy = (self.logits_per_image.argmax(dim=-1) == self.labels).float().mean()
            text_accuracy = (self.logits_per_text.argmax(dim=-1) == self.labels).float().mean()
        return {
            "loss/clip": self.loss.detach(),
            "loss/image_to_text": self.image_loss.detach(),
            "loss/text_to_image": self.text_loss.detach(),
            "accuracy/image_to_text": image_accuracy,
            "accuracy/text_to_image": text_accuracy,
            "count/local_batch": torch.tensor(self.local_batch_size, device=self.loss.device),
            "count/global_batch": torch.tensor(self.global_batch_size, device=self.loss.device),
        }


class DistributedCLIPLoss(nn.Module):
    """Symmetric CLIP cross-entropy with explicit distributed semantics."""

    def __init__(
        self,
        *,
        local_loss: bool = False,
        gather_with_grad: bool = True,
        label_smoothing: float = 0.0,
        group: Any = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        self.local_loss = bool(local_loss)
        self.gather_with_grad = bool(gather_with_grad)
        self.label_smoothing = float(label_smoothing)
        self.group = group

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor | float,
    ) -> CLIPLossOutput:
        _validate_features(image_features, text_features)
        all_images, all_texts = gather_features(
            image_features,
            text_features,
            gather_with_grad=self.gather_with_grad,
            group=self.group,
        )
        local_batch = int(image_features.shape[0])
        global_batch = int(all_images.shape[0])
        rank = dist.get_rank(self.group) if _distributed_ready() else 0

        scale = torch.as_tensor(logit_scale, device=image_features.device, dtype=image_features.dtype)
        if scale.numel() != 1 or not torch.isfinite(scale).all():
            raise CLIPLossError("logit_scale must be one finite scalar.")

        if self.local_loss and global_batch != local_batch:
            logits_per_image = scale * image_features @ all_texts.transpose(0, 1)
            logits_per_text = scale * text_features @ all_images.transpose(0, 1)
            offset = rank * local_batch
            labels = torch.arange(local_batch, device=image_features.device) + offset
        else:
            logits_per_image = scale * all_images @ all_texts.transpose(0, 1)
            logits_per_text = logits_per_image.transpose(0, 1)
            labels = torch.arange(global_batch, device=image_features.device)

        image_loss = F.cross_entropy(
            logits_per_image.float(), labels, label_smoothing=self.label_smoothing
        )
        text_loss = F.cross_entropy(
            logits_per_text.float(), labels, label_smoothing=self.label_smoothing
        )
        loss = 0.5 * (image_loss + text_loss)
        return CLIPLossOutput(
            loss=loss,
            image_loss=image_loss,
            text_loss=text_loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            labels=labels,
            local_batch_size=local_batch,
            global_batch_size=global_batch,
        )


def retrieval_recall_at_k(
    image_features: Tensor,
    text_features: Tensor,
    *,
    ks: Sequence[int] = (1, 5, 10),
    chunk_size: int = 1024,
) -> dict[str, float]:
    """Compute paired image↔text Recall@K without materialising huge matrices."""

    _validate_features(image_features, text_features)
    count = int(image_features.shape[0])
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    safe_ks = sorted({max(1, min(int(k), count)) for k in ks})
    results: dict[str, float] = {}

    def directional(query: Tensor, candidates: Tensor, prefix: str) -> None:
        hits = {k: 0 for k in safe_ks}
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            scores = query[start:stop] @ candidates.transpose(0, 1)
            top = scores.topk(max(safe_ks), dim=1, largest=True, sorted=True).indices
            targets = torch.arange(start, stop, device=top.device).unsqueeze(1)
            for k in safe_ks:
                hits[k] += int((top[:, :k] == targets).any(dim=1).sum().item())
        for requested in ks:
            safe = max(1, min(int(requested), count))
            results[f"{prefix}/R@{requested}"] = hits[safe] / count

    directional(image_features, text_features, "image_to_text")
    directional(text_features, image_features, "text_to_image")
    return results


def topk_indices_chunked(
    query_features: Tensor,
    candidate_features: Tensor,
    *,
    k: int,
    chunk_size: int = 1024,
) -> Tensor:
    _validate_features(query_features, candidate_features)
    if query_features.shape[0] != candidate_features.shape[0]:
        raise CLIPLossError("Top-k paired retrieval expects equal query/candidate counts.")
    safe_k = min(max(1, int(k)), int(candidate_features.shape[0]))
    rows: list[Tensor] = []
    for start in range(0, query_features.shape[0], chunk_size):
        scores = query_features[start : start + chunk_size] @ candidate_features.transpose(0, 1)
        rows.append(scores.topk(safe_k, dim=1).indices.cpu())
    return torch.cat(rows, dim=0)


def _run_self_test() -> dict[str, Any]:
    torch.manual_seed(7)
    image_leaf = torch.randn(4, 8, requires_grad=True)
    text_leaf = torch.randn(4, 8, requires_grad=True)
    image = F.normalize(image_leaf, dim=-1)
    text = F.normalize(text_leaf, dim=-1)
    objective = DistributedCLIPLoss(local_loss=False)
    output = objective(image, text, torch.tensor(2.5, requires_grad=True))
    output.loss.backward()
    recalls = retrieval_recall_at_k(image.detach(), image.detach(), ks=(1, 5, 10))
    top = topk_indices_chunked(image.detach(), image.detach(), k=1000)
    assert output.logits_per_image.shape == (4, 4)
    assert torch.isfinite(output.loss)
    assert recalls["image_to_text/R@1"] == 1.0
    assert top.shape == (4, 4)
    return {
        "status": "passed",
        "loss_is_finite": True,
        "image_gradient_is_finite": bool(torch.isfinite(image_leaf.grad).all()),
        "small_dataset_top1000_safe": True,
        "retrieval_identity_r1": recalls["image_to_text/R@1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
