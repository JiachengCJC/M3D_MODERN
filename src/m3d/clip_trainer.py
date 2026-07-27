"""Training loop for modernised M3D-CLIP.

The loop is intentionally smaller than the multimodal-language trainer because
M3D-CLIP has one homogeneous task.  It still provides DDP, BF16 autocast,
gradient accumulation, fused AdamW, cosine warmup, exact sampler epochs,
validation retrieval and atomic resume checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from m3d.data.clip_data import CLIPBatch, M3DCLIPDataset
from m3d.model.clip import M3DCLIP
from m3d.model.clip_loss import DistributedCLIPLoss, retrieval_recall_at_k


class CLIPTrainerError(RuntimeError):
    pass


@dataclass(slots=True)
class CLIPTrainingConfig:
    epochs: int = 100
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1.0e-8
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    log_every_steps: int = 10
    eval_every_steps: int = 1000
    save_every_steps: int = 1000
    keep_last_n: int = 2
    output_dir: Path = Path("outputs/m3d-clip")
    use_bf16: bool = True
    allow_tf32: bool = True
    fused_adamw: bool = True
    local_loss: bool = False
    gather_with_grad: bool = True
    label_smoothing: float = 0.0
    seed: int = 42


@dataclass(frozen=True, slots=True)
class CLIPTrainerState:
    epoch: int
    microbatch_in_epoch: int
    optimizer_step: int
    best_mean_recall: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _distributed_ready() else 0


def _world_size() -> int:
    return dist.get_world_size() if _distributed_ready() else 1


def _is_main() -> bool:
    return _rank() == 0


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def build_clip_optimizer(model: M3DCLIP, config: CLIPTrainingConfig) -> torch.optim.Optimizer:
    no_decay_names = set(model.no_weight_decay_parameter_names())
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or name in no_decay_names:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay, "group_name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
    ]
    fused = bool(config.fused_adamw and torch.cuda.is_available())
    kwargs: dict[str, Any] = {
        "lr": config.learning_rate,
        "betas": config.betas,
        "eps": config.epsilon,
    }
    if "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        kwargs["fused"] = fused
    return torch.optim.AdamW(groups, **kwargs)


class WarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = int(math.ceil(self.total_steps * warmup_ratio))
        self.step_count = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self._apply()

    def _scale(self) -> float:
        if self.warmup_steps > 0 and self.step_count < self.warmup_steps:
            return self.step_count / self.warmup_steps
        progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply(self) -> None:
        scale = self._scale()
        for base, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base * scale

    def step(self) -> None:
        self.step_count += 1
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {"total_steps": self.total_steps, "warmup_steps": self.warmup_steps, "step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["total_steps"]) != self.total_steps:
            raise CLIPTrainerError("Scheduler total_steps changed across resume.")
        self.step_count = int(state["step_count"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self._apply()


class M3DCLIPTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: Any,
        train_dataset: M3DCLIPDataset,
        train_sampler: Any,
        validation_loader: Any | None,
        validation_dataset: M3DCLIPDataset | None,
        validation_sampler: Any | None,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosine,
        config: CLIPTrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.validation_loader = validation_loader
        self.validation_dataset = validation_dataset
        self.validation_sampler = validation_sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.objective = DistributedCLIPLoss(
            local_loss=config.local_loss,
            gather_with_grad=config.gather_with_grad,
            label_smoothing=config.label_smoothing,
        )
        self.state = CLIPTrainerState(0, 0, 0, float("-inf"))
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.use_bf16 and self.device.type == "cuda")

    def _save(self, *, tag: str | None = None) -> Path:
        step = self.state.optimizer_step
        name = tag or f"checkpoint-step-{step:08d}"
        final = self.config.output_dir / name
        temp = self.config.output_dir / f".{name}.incomplete-{os.getpid()}"
        if _is_main():
            shutil.rmtree(temp, ignore_errors=True)
            temp.mkdir(parents=True)
            payload = {
                "model": _unwrap(self.model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "trainer_state": self.state.to_dict(),
                "rng": _capture_rng(),
            }
            torch.save(payload, temp / "training_state.pt")
            _atomic_json(temp / "COMPLETED.json", {"status": "complete", "optimizer_step": step})
            if final.exists():
                shutil.rmtree(final)
            os.replace(temp, final)
            _atomic_json(self.config.output_dir / "latest.json", {"checkpoint": final.name})
            checkpoints = sorted(self.config.output_dir.glob("checkpoint-step-*"))
            for old in checkpoints[: max(0, len(checkpoints) - self.config.keep_last_n)]:
                shutil.rmtree(old, ignore_errors=True)
        if _distributed_ready():
            dist.barrier()
        return final

    def resume(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        if path.name == "latest.json":
            path = path.parent / json.loads(path.read_text())["checkpoint"]
        elif path.is_dir() and (path / "latest.json").is_file():
            path = path / json.loads((path / "latest.json").read_text())["checkpoint"]
        payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
        _unwrap(self.model).load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.state = CLIPTrainerState(**payload["trainer_state"])
        _restore_rng(payload["rng"])
        if _distributed_ready():
            dist.barrier()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.validation_loader is None:
            return {}
        self.model.eval()
        local_images: list[Tensor] = []
        local_texts: list[Tensor] = []
        for batch in self.validation_loader:
            batch = batch.to(self.device)
            with self._autocast():
                output = self.model(batch.images, batch.input_ids, batch.attention_mask)
            local_images.append(output.image_features.float().cpu())
            local_texts.append(output.text_features.float().cpu())
        image = torch.cat(local_images) if local_images else torch.empty((0, _unwrap(self.model).config.projection_dim))
        text = torch.cat(local_texts) if local_texts else torch.empty_like(image)
        gathered: list[Any] | None = [None] * _world_size() if _is_main() else None
        if _distributed_ready():
            dist.gather_object((image, text), gathered, dst=0)
        else:
            gathered = [(image, text)]
        metrics: dict[str, float] = {}
        if _is_main() and gathered is not None:
            all_image = torch.cat([item[0] for item in gathered], dim=0)
            all_text = torch.cat([item[1] for item in gathered], dim=0)
            metrics = retrieval_recall_at_k(all_image, all_text)
            metrics["mean_recall"] = sum(metrics.values()) / len(metrics)
            _atomic_json(self.config.output_dir / f"eval-step-{self.state.optimizer_step:08d}.json", metrics)
        if _distributed_ready():
            objects = [metrics]
            dist.broadcast_object_list(objects, src=0)
            metrics = objects[0]
        self.model.train()
        return metrics

    def train(self) -> dict[str, Any]:
        torch.backends.cuda.matmul.allow_tf32 = self.config.allow_tf32
        self.optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.config.epochs):
            self.train_dataset.set_epoch(epoch)
            if hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            running_loss = 0.0
            running_count = 0
            skip = self.state.microbatch_in_epoch if epoch == start_epoch else 0
            for microbatch_index, batch in enumerate(self.train_loader):
                if microbatch_index < skip:
                    continue
                batch = batch.to(self.device)
                update = ((microbatch_index + 1) % self.config.gradient_accumulation_steps == 0) or (microbatch_index + 1 == len(self.train_loader))
                sync_context = self.model.no_sync() if isinstance(self.model, DistributedDataParallel) and not update else torch.enable_grad()
                with sync_context:
                    with self._autocast():
                        output = self.model(batch.images, batch.input_ids, batch.attention_mask)
                        losses = self.objective(output.image_features, output.text_features, output.logit_scale)
                        loss = losses.loss / self.config.gradient_accumulation_steps
                    if not torch.isfinite(loss):
                        raise CLIPTrainerError("Non-finite CLIP loss.")
                    loss.backward()
                running_loss += float(losses.loss.detach())
                running_count += 1
                self.state = CLIPTrainerState(epoch, microbatch_index + 1, self.state.optimizer_step, self.state.best_mean_recall)
                if not update:
                    continue
                torch.nn.utils.clip_grad_norm_(_unwrap(self.model).parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.state = CLIPTrainerState(epoch, microbatch_index + 1, self.state.optimizer_step + 1, self.state.best_mean_recall)
                if _is_main() and self.state.optimizer_step % self.config.log_every_steps == 0:
                    print(json.dumps({"step": self.state.optimizer_step, "epoch": epoch, "loss": running_loss / max(1, running_count), "lr": self.optimizer.param_groups[0]["lr"]}, sort_keys=True), flush=True)
                    running_loss = 0.0
                    running_count = 0
                if self.validation_loader is not None and self.state.optimizer_step % self.config.eval_every_steps == 0:
                    metrics = self.evaluate()
                    best = max(self.state.best_mean_recall, metrics.get("mean_recall", float("-inf")))
                    self.state = CLIPTrainerState(epoch, microbatch_index + 1, self.state.optimizer_step, best)
                    if metrics.get("mean_recall") == best:
                        self._save(tag="best")
                if self.state.optimizer_step % self.config.save_every_steps == 0:
                    self._save()
            self.state = CLIPTrainerState(epoch + 1, 0, self.state.optimizer_step, self.state.best_mean_recall)
        final = self._save(tag="final")
        result = {"status": "complete", "optimizer_steps": self.state.optimizer_step, "elapsed_seconds": time.time() - start_time, "final_checkpoint": str(final), "best_mean_recall": self.state.best_mean_recall}
        if _is_main():
            _atomic_json(self.config.output_dir / "training_result.json", result)
        return result


def _run_self_test() -> dict[str, Any]:
    parameter = nn.Parameter(torch.tensor([1.0]))
    fake = nn.Module()
    fake.register_parameter("weight", parameter)
    optimizer = torch.optim.AdamW(fake.parameters(), lr=1.0)
    scheduler = WarmupCosine(optimizer, total_steps=4, warmup_ratio=0.25)
    values = []
    for _ in range(4):
        optimizer.step(); scheduler.step(); values.append(optimizer.param_groups[0]["lr"])
    assert values[-1] == 0.0
    return {"status": "passed", "scheduler_steps": 4, "final_lr": values[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
