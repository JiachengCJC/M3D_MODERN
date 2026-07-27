"""Torchrun entry point for M3D-CLIP pretraining."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from m3d.clip_trainer import CLIPTrainingConfig, M3DCLIPTrainer, WarmupCosine, build_clip_optimizer
from m3d.data.clip_data import CLIPDataConfig, build_clip_dataloader
from m3d.model.clip import M3DCLIPConfig, build_m3d_clip, load_m3d_clip_checkpoint


class CLIPEntrypointError(RuntimeError):
    pass


def _set_nested(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cursor = mapping
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def load_yaml(path: Path, overrides: list[str]) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CLIPEntrypointError("CLIP config must be a YAML mapping.")
    result = copy.deepcopy(payload)
    for item in overrides:
        if "=" not in item:
            raise CLIPEntrypointError(f"Override must be key=value: {item}")
        key, raw = item.split("=", 1)
        _set_nested(result, key, yaml.safe_load(raw))
    return result


def _init_distributed(timeout_seconds: int) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        import datetime
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=timeout_seconds))
    if not torch.cuda.is_available():
        raise CLIPEntrypointError("M3D-CLIP training requires CUDA.")
    torch.cuda.set_device(local)
    return rank, world, local, torch.device("cuda", local)


def _seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def _model_config(values: Mapping[str, Any]) -> M3DCLIPConfig:
    return M3DCLIPConfig(**dict(values))


def _training_config(values: Mapping[str, Any], *, base_dir: Path) -> CLIPTrainingConfig:
    output = Path(str(values.get("output_dir", "../outputs/m3d-clip"))).expanduser()
    if not output.is_absolute():
        output = (base_dir / output).resolve()
    betas = values.get("betas", (0.9, 0.98))
    return CLIPTrainingConfig(
        epochs=int(values.get("epochs", 100)),
        gradient_accumulation_steps=int(values.get("gradient_accumulation_steps", 1)),
        learning_rate=float(values.get("learning_rate", 1e-4)),
        weight_decay=float(values.get("weight_decay", 0.1)),
        betas=(float(betas[0]), float(betas[1])),
        epsilon=float(values.get("epsilon", 1e-8)),
        warmup_ratio=float(values.get("warmup_ratio", 0.03)),
        max_grad_norm=float(values.get("max_grad_norm", 1.0)),
        log_every_steps=int(values.get("log_every_steps", 10)),
        eval_every_steps=int(values.get("eval_every_steps", 1000)),
        save_every_steps=int(values.get("save_every_steps", 1000)),
        keep_last_n=int(values.get("keep_last_n", 2)),
        output_dir=output,
        use_bf16=bool(values.get("use_bf16", True)),
        allow_tf32=bool(values.get("allow_tf32", True)),
        fused_adamw=bool(values.get("fused_adamw", True)),
        local_loss=bool(values.get("local_loss", False)),
        gather_with_grad=bool(values.get("gather_with_grad", True)),
        label_smoothing=float(values.get("label_smoothing", 0.0)),
        seed=int(values.get("seed", 42)),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    payload = load_yaml(config_path, args.override)
    rank, world, local, device = _init_distributed(int(payload.get("distributed", {}).get("timeout_seconds", 1800)))
    base_dir = config_path.parent
    training = _training_config(payload["training"], base_dir=base_dir)
    _seed_everything(training.seed, rank)
    try:
        from transformers import AutoTokenizer
    except Exception as error:
        raise CLIPEntrypointError("transformers is required for M3D-CLIP training.") from error
    model_values = dict(payload["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_values["language_model_name_or_path"],
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    data_config = CLIPDataConfig.from_mapping(payload["data"], base_dir=base_dir)
    train_dataset, train_loader, train_sampler = build_clip_dataloader(data_config, tokenizer, split="train", rank=rank, world_size=world)
    validation_dataset, validation_loader, validation_sampler = build_clip_dataloader(data_config, tokenizer, split="validation", rank=rank, world_size=world)

    # Every rank uses the same model initialisation seed before DDP broadcast.
    torch.manual_seed(training.seed)
    model, build_report = build_m3d_clip(
        _model_config(model_values),
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
    )
    pretrained = model_values.get("pretrained_model")
    checkpoint_report = None
    if pretrained:
        path = Path(str(pretrained)).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        checkpoint_report = load_m3d_clip_checkpoint(model, path)
    model.to(device)
    wrapped: torch.nn.Module = model
    if world > 1:
        wrapped = DistributedDataParallel(model, device_ids=[local], output_device=local, gradient_as_bucket_view=True, static_graph=True)
    optimizer = build_clip_optimizer(model, training)
    updates_per_epoch = math.ceil(len(train_loader) / training.gradient_accumulation_steps)
    scheduler = WarmupCosine(optimizer, total_steps=updates_per_epoch * training.epochs, warmup_ratio=training.warmup_ratio)
    trainer = M3DCLIPTrainer(
        model=wrapped,
        train_loader=train_loader,
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        validation_loader=validation_loader,
        validation_dataset=validation_dataset,
        validation_sampler=validation_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training,
        device=device,
    )
    if rank == 0:
        training.output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(training.output_dir / "tokenizer")
        (training.output_dir / "resolved_clip_config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        (training.output_dir / "startup_report.json").write_text(json.dumps({"model": build_report.to_dict(), "checkpoint": checkpoint_report, "world_size": world}, indent=2, sort_keys=True), encoding="utf-8")
    if args.resume_from:
        trainer.resume(args.resume_from)
    if args.startup_only:
        return {"status": "startup_validated", "world_size": world}
    return trainer.train()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m3d_clip_pretrain.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume-from")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _self_test() -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text("model: {hidden_size: 8}\ntraining: {epochs: 2}\n", encoding="utf-8")
        payload = load_yaml(path, ["training.epochs=3", "model.require_flash_sdpa=false"])
        assert payload["training"]["epochs"] == 3
        return {"status": "passed", "override_count": 2}


def main() -> None:
    args = _parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2, sort_keys=True)); return
    try:
        result = run(args)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
