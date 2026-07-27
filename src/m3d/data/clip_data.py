"""Dataset, deterministic augmentation and batching for M3D-CLIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


class CLIPDataError(RuntimeError):
    pass


@dataclass(slots=True)
class CLIPDataConfig:
    data_root: Path
    caption_json: Path
    image_size: tuple[int, int, int] = (32, 256, 256)
    max_text_length: int = 128
    per_device_batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    seed: int = 42
    train_augment: bool = True
    verify_intensity_range: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, base_dir: Path) -> "CLIPDataConfig":
        def resolve(value: str) -> Path:
            path = Path(value).expanduser()
            return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()

        return cls(
            data_root=resolve(str(values["data_root"])),
            caption_json=resolve(str(values["caption_json"])),
            image_size=tuple(int(x) for x in values.get("image_size", (32, 256, 256))),
            max_text_length=int(values.get("max_text_length", 128)),
            per_device_batch_size=int(values.get("per_device_batch_size", 8)),
            num_workers=int(values.get("num_workers", 8)),
            pin_memory=bool(values.get("pin_memory", True)),
            persistent_workers=bool(values.get("persistent_workers", True)),
            prefetch_factor=int(values.get("prefetch_factor", 4)),
            seed=int(values.get("seed", 42)),
            train_augment=bool(values.get("train_augment", True)),
            verify_intensity_range=bool(values.get("verify_intensity_range", True)),
        )


@dataclass(frozen=True, slots=True)
class CLIPRecord:
    sample_id: str
    image_path: Path
    text_path: Path | None
    inline_text: str | None


@dataclass(frozen=True, slots=True)
class CLIPBatch:
    images: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    sample_ids: tuple[str, ...]
    texts: tuple[str, ...]

    def to(self, device: torch.device, *, non_blocking: bool = True) -> "CLIPBatch":
        return CLIPBatch(
            images=self.images.to(device, non_blocking=non_blocking),
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
            sample_ids=self.sample_ids,
            texts=self.texts,
        )


def _normalise_split_name(split: str) -> str:
    aliases = {"val": "validation", "valid": "validation", "hard_test": "test"}
    return aliases.get(split.strip().lower(), split.strip().lower())


def load_clip_records(config: CLIPDataConfig, split: str) -> list[CLIPRecord]:
    split = _normalise_split_name(split)
    payload = json.loads(config.caption_json.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or split not in payload:
        raise CLIPDataError(
            f"Caption JSON {config.caption_json} has no split {split!r}; available={list(payload) if isinstance(payload, Mapping) else type(payload)}"
        )
    rows = payload[split]
    if not isinstance(rows, list):
        raise CLIPDataError(f"Split {split!r} must be a JSON list.")
    records: list[CLIPRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or "image" not in row:
            raise CLIPDataError(f"Malformed row {index} in split {split!r}.")
        image_path = (config.data_root / str(row["image"])).resolve()
        inline_text = None
        text_path = None
        if "text" in row:
            candidate = (config.data_root / str(row["text"])).resolve()
            if candidate.is_file():
                text_path = candidate
            else:
                inline_text = str(row["text"])
        elif "caption" in row:
            inline_text = str(row["caption"])
        else:
            raise CLIPDataError(f"Row {index} has neither text nor caption.")
        sample_id = str(row.get("id") or row.get("sample_id") or f"{split}-{index:08d}")
        records.append(CLIPRecord(sample_id, image_path, text_path, inline_text))
    if not records:
        raise CLIPDataError(f"Split {split!r} is empty.")
    return records


def _load_volume(path: Path, expected_size: Sequence[int], verify_range: bool) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except Exception as error:
            raise CLIPDataError("nibabel is required for NIfTI CLIP inputs.") from error
        xyz = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
        array = np.transpose(xyz, (2, 1, 0))[None]
    else:
        raise CLIPDataError(f"Unsupported image format: {path}")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[None]
    expected = (1, *tuple(int(x) for x in expected_size))
    if tuple(array.shape) != expected:
        raise CLIPDataError(f"Expected image shape {expected}, got {tuple(array.shape)} at {path}")
    if not np.isfinite(array).all():
        raise CLIPDataError(f"Image contains NaN/Inf: {path}")
    if verify_range and (float(array.min()) < -1e-4 or float(array.max()) > 1.0001):
        raise CLIPDataError(
            f"Expected pre-normalised [0,1] data, got range [{array.min()}, {array.max()}] at {path}"
        )
    return torch.from_numpy(np.ascontiguousarray(array))


def _augment(image: Tensor, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = image
    if torch.rand((), generator=generator).item() < 0.5:
        k = int(torch.randint(0, 4, (), generator=generator).item())
        result = torch.rot90(result, k=k, dims=(-2, -1))
    for axis in (-3, -2, -1):
        if torch.rand((), generator=generator).item() < 0.1:
            result = torch.flip(result, dims=(axis,))
    if torch.rand((), generator=generator).item() < 0.5:
        scale = 1.0 + (torch.rand((), generator=generator).item() * 0.2 - 0.1)
        result = result * scale
    if torch.rand((), generator=generator).item() < 0.5:
        shift = torch.rand((), generator=generator).item() * 0.2 - 0.1
        result = result + shift
    return result.clamp_(0.0, 1.0).contiguous()


class M3DCLIPDataset(Dataset[dict[str, Any]]):
    def __init__(self, config: CLIPDataConfig, tokenizer: Any, *, split: str) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.split = _normalise_split_name(split)
        self.records = load_clip_records(config, self.split)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = _load_volume(
            record.image_path,
            self.config.image_size,
            self.config.verify_intensity_range,
        )
        if self.split == "train" and self.config.train_augment:
            digest = hashlib.blake2b(
                f"{self.config.seed}:{self.epoch}:{index}:{record.sample_id}".encode(),
                digest_size=8,
            ).digest()
            image = _augment(image, seed=int.from_bytes(digest, "little"))
        text = (
            record.text_path.read_text(encoding="utf-8", errors="strict")
            if record.text_path is not None
            else str(record.inline_text)
        ).strip()
        if not text:
            raise CLIPDataError(f"Empty text for sample {record.sample_id}")
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_text_length,
            add_special_tokens=True,
            padding=False,
        )
        return {
            "sample_id": record.sample_id,
            "image": image.float(),
            "text": text,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.bool),
        }


class CLIPCollator:
    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> CLIPBatch:
        if not samples:
            raise CLIPDataError("Cannot collate an empty sample list.")
        images = torch.stack([sample["image"] for sample in samples], dim=0).contiguous()
        max_len = max(int(sample["input_ids"].numel()) for sample in samples)
        if self.pad_to_multiple_of > 1:
            max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids = torch.full((len(samples), max_len), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(samples), max_len), dtype=torch.bool)
        for row, sample in enumerate(samples):
            length = int(sample["input_ids"].numel())
            input_ids[row, :length] = sample["input_ids"]
            attention[row, :length] = sample["attention_mask"]
        return CLIPBatch(
            images=images,
            input_ids=input_ids,
            attention_mask=attention,
            sample_ids=tuple(str(sample["sample_id"]) for sample in samples),
            texts=tuple(str(sample["text"]) for sample in samples),
        )


class EqualBatchDistributedSampler(Sampler[int]):
    """Shuffle globally and truncate so every rank has equal full batches."""

    def __init__(
        self,
        dataset_size: int,
        *,
        rank: int,
        world_size: int,
        per_device_batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size = int(per_device_batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        global_batch = self.world_size * self.batch_size
        self.usable_size = (self.dataset_size // global_batch) * global_batch
        if self.usable_size == 0:
            raise CLIPDataError(
                f"Dataset size {dataset_size} is smaller than global batch size {global_batch}."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.usable_size // self.world_size

    def __iter__(self) -> Iterable[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = (
            torch.randperm(self.dataset_size, generator=generator).tolist()
            if self.shuffle
            else list(range(self.dataset_size))
        )[: self.usable_size]
        return iter(indices[self.rank : self.usable_size : self.world_size])


class ExactEvaluationSampler(Sampler[int]):
    def __init__(self, dataset_size: int, *, rank: int, world_size: int) -> None:
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __len__(self) -> int:
        return max(0, (self.dataset_size - self.rank + self.world_size - 1) // self.world_size)

    def __iter__(self) -> Iterable[int]:
        return iter(range(self.rank, self.dataset_size, self.world_size))


def build_clip_dataloader(
    config: CLIPDataConfig,
    tokenizer: Any,
    *,
    split: str,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[M3DCLIPDataset, DataLoader[CLIPBatch], Sampler[int]]:
    dataset = M3DCLIPDataset(config, tokenizer, split=split)
    training = _normalise_split_name(split) == "train"
    sampler: Sampler[int]
    if training:
        sampler = EqualBatchDistributedSampler(
            len(dataset),
            rank=rank,
            world_size=world_size,
            per_device_batch_size=config.per_device_batch_size,
            seed=config.seed,
            shuffle=True,
        )
    else:
        sampler = ExactEvaluationSampler(len(dataset), rank=rank, world_size=world_size)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.per_device_batch_size,
        "sampler": sampler,
        "drop_last": training,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "collate_fn": CLIPCollator(getattr(tokenizer, "pad_token_id", 0) or 0),
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["prefetch_factor"] = config.prefetch_factor
    return dataset, DataLoader(**kwargs), sampler


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        ids = [1] + [2 + (ord(ch) % 20) for ch in text[:10]] + [2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _run_self_test() -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = []
        for index in range(4):
            np.save(root / f"image-{index}.npy", np.zeros((1, 2, 4, 4), np.float32))
            (root / f"text-{index}.txt").write_text(f"sample {index}", encoding="utf-8")
            rows.append({"image": f"image-{index}.npy", "text": f"text-{index}.txt"})
        (root / "data.json").write_text(json.dumps({"train": rows, "validation": rows, "test": rows}), encoding="utf-8")
        config = CLIPDataConfig(root, root / "data.json", image_size=(2, 4, 4), per_device_batch_size=2, num_workers=0)
        dataset, loader, sampler = build_clip_dataloader(config, _TinyTokenizer(), split="train")
        batch = next(iter(loader))
        assert batch.images.shape == (2, 1, 2, 4, 4)
        assert batch.input_ids.shape[1] % 8 == 0
        assert len(list(iter(sampler))) == 4
        return {"status": "passed", "batch_shape": list(batch.images.shape), "dynamic_padding_multiple": 8}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
