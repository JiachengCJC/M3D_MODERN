"""Distributed retrieval evaluation for M3D-CLIP.

The evaluator produces the original ``test_ir.csv`` (image→text) and
``test_tr.csv`` (text→image) files, while safely clamping requested top-k to the
actual test-set size.  ``--csv-top-k 1000`` therefore works even for a tiny
logic test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import yaml

from m3d.data.clip_data import CLIPDataConfig, build_clip_dataloader
from m3d.model.clip import M3DCLIPConfig, build_m3d_clip, load_m3d_clip_checkpoint
from m3d.model.clip_loss import retrieval_recall_at_k, topk_indices_chunked


class CLIPEvaluationError(RuntimeError):
    pass


def _init() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1")); local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise CLIPEvaluationError("CUDA is required.")
    torch.cuda.set_device(local)
    return rank, world, local, torch.device("cuda", local)


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if (path / "latest.json").is_file():
        latest = json.loads((path / "latest.json").read_text(encoding="utf-8"))["checkpoint"]
        return path / latest / "training_state.pt"
    if (path / "training_state.pt").is_file():
        return path / "training_state.pt"
    for candidate in (path / "model.safetensors", path / "model_params.bin", path / "pytorch_model.bin"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve M3D-CLIP checkpoint under {path}")


def _load_weights(model: Any, checkpoint: Path) -> dict[str, Any]:
    if checkpoint.name == "training_state.pt":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        return {"path": str(checkpoint), "format": "trainer_state"}
    return load_m3d_clip_checkpoint(model, checkpoint, strict=True)


def _write_topk_csv(
    path: Path,
    indices: torch.Tensor,
    sample_ids: list[str],
    *,
    direction: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_index", "query_sample_id", f"top{indices.shape[1]}_indices", f"top{indices.shape[1]}_sample_ids", "direction"])
        for row in range(indices.shape[0]):
            values = indices[row].tolist()
            writer.writerow([row, sample_ids[row], json.dumps(values), json.dumps([sample_ids[index] for index in values]), direction])


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank, world, _, device = _init()
    config_path = Path(args.config).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    try:
        from transformers import AutoTokenizer
    except Exception as error:
        raise CLIPEvaluationError("transformers is required.") from error
    model_values = dict(payload["model"])
    tokenizer_path = Path(args.tokenizer).expanduser().resolve() if args.tokenizer else model_values["language_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    data_config = CLIPDataConfig.from_mapping(payload["data"], base_dir=base_dir)
    dataset, loader, _ = build_clip_dataloader(data_config, tokenizer, split=args.split, rank=rank, world_size=world)
    model, report = build_m3d_clip(M3DCLIPConfig(**model_values), cache_dir=args.cache_dir, local_files_only=args.local_files_only, torch_dtype=torch.bfloat16)
    checkpoint = _resolve_checkpoint(Path(args.checkpoint).expanduser().resolve())
    checkpoint_report = _load_weights(model, checkpoint)
    model.to(device).eval()
    image_parts: list[torch.Tensor] = []
    text_parts: list[torch.Tensor] = []
    ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            ids.extend(batch.sample_ids)
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch.images, batch.input_ids, batch.attention_mask)
            image_parts.append(output.image_features.float().cpu())
            text_parts.append(output.text_features.float().cpu())
    local = (torch.cat(image_parts), torch.cat(text_parts), ids) if image_parts else (torch.empty((0, model.config.projection_dim)), torch.empty((0, model.config.projection_dim)), [])
    gathered: list[Any] | None = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local, gathered, dst=0)
    else:
        gathered = [local]
    result: dict[str, Any] = {}
    if rank == 0 and gathered is not None:
        images = torch.cat([item[0] for item in gathered])
        texts = torch.cat([item[1] for item in gathered])
        sample_ids = sum([item[2] for item in gathered], [])
        if len(set(sample_ids)) != len(dataset) or len(sample_ids) != len(dataset):
            raise CLIPEvaluationError("Evaluation sample IDs are duplicated or missing.")
        metrics = retrieval_recall_at_k(images, texts, ks=(1, 5, 10), chunk_size=args.chunk_size)
        image_top = topk_indices_chunked(images, texts, k=args.csv_top_k, chunk_size=args.chunk_size)
        text_top = topk_indices_chunked(texts, images, k=args.csv_top_k, chunk_size=args.chunk_size)
        output_dir = Path(args.output_dir).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
        _write_topk_csv(output_dir / "test_ir.csv", image_top, sample_ids, direction="image_to_text")
        _write_topk_csv(output_dir / "test_tr.csv", text_top, sample_ids, direction="text_to_image")
        torch.save({"image_features": images, "text_features": texts, "sample_ids": sample_ids}, output_dir / "retrieval_features.pt")
        result = {"status": "passed", "split": args.split, "sample_count": len(sample_ids), "requested_csv_top_k": args.csv_top_k, "effective_csv_top_k": image_top.shape[1], "metrics": metrics, "model": report.to_dict(), "checkpoint": checkpoint_report}
        (output_dir / "retrieval_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "COMPLETED.json").write_text(json.dumps({"status": "complete"}, indent=2), encoding="utf-8")
    return result


def _self_test() -> dict[str, Any]:
    image = torch.eye(3)
    indices = topk_indices_chunked(image, image, k=1000)
    assert indices.shape == (3, 3)
    return {"status": "passed", "top1000_clamped_to": 3, "identity_r1": retrieval_recall_at_k(image, image)["image_to_text/R@1"]}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m3d_clip_pretrain.yaml")
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--tokenizer")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="outputs/m3d-clip-evaluation")
    parser.add_argument("--csv-top-k", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2, sort_keys=True)); return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")
    try:
        result = run(args)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
