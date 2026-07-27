"""Volume and text I/O for the M3D data pipeline.

The original M3D datasets assume preprocessed ``.npy`` volumes with shape
``[C, D, H, W]`` and image intensities already normalized to ``[0, 1]``.  This
module preserves that default contract while making failures explicit and
centralising all file access.

Key policies
------------
* No hidden spatial resize, crop, resample, or min-max normalization occurs
  here.  Training must never silently change the geometry of an existing M3D
  dataset.
* Images and masks leave this module as contiguous CPU ``torch.float32``
  tensors with shape ``[C, D, H, W]``.
* A segmentation mask is selected explicitly by label ID or channel.  An
  all-zero binary mask is valid and is never interpreted as "no segmentation
  task".
* ``.npy`` is the reproducibility path.  ``.nii`` and ``.nii.gz`` are also
  supported for user datasets, with an explicit NIfTI axis conversion from
  nibabel's ``[X, Y, Z]`` array order to M3D's ``[D, H, W] = [Z, Y, X]``.
* Optional node-local caching uses an atomic copy guarded by a Linux file lock,
  so multiple DataLoader workers and distributed ranks cannot publish a
  partially copied file.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import shutil
import stat
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from m3d.config import ExperimentConfig


LOGGER = logging.getLogger(__name__)


class DataIOError(RuntimeError):
    """Raised when a source file cannot satisfy the M3D data contract."""


class VolumeKind(str, Enum):
    IMAGE = "image"
    MASK = "mask"


class VolumeFormat(str, Enum):
    NPY = "npy"
    NIFTI = "nifti"

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> "VolumeFormat":
        name = Path(path).name.lower()
        if name.endswith(".npy"):
            return cls.NPY
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            return cls.NIFTI
        raise DataIOError(
            f"Unsupported volume format for {path!s}; expected .npy, .nii, or .nii.gz"
        )


@dataclass(frozen=True, slots=True)
class VolumeGeometry:
    """Geometry metadata retained for diagnostics and future export."""

    source_shape: tuple[int, ...]
    canonical_shape_xyz: tuple[int, int, int] | None = None
    spacing_xyz: tuple[float, float, float] | None = None
    affine: np.ndarray | None = field(default=None, repr=False, compare=False)
    orientation: tuple[str, str, str] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "source_shape": list(self.source_shape),
            "canonical_shape_xyz": (
                None if self.canonical_shape_xyz is None else list(self.canonical_shape_xyz)
            ),
            "spacing_xyz": None if self.spacing_xyz is None else list(self.spacing_xyz),
            "affine": None if self.affine is None else self.affine.tolist(),
            "orientation": None if self.orientation is None else list(self.orientation),
        }


@dataclass(frozen=True, slots=True)
class LoadedVolume:
    """One validated volume and the exact source used to read it."""

    tensor: Tensor
    source_path: Path
    resolved_path: Path
    volume_format: VolumeFormat
    kind: VolumeKind
    geometry: VolumeGeometry
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tensor.device.type != "cpu":
            raise DataIOError("LoadedVolume tensors must remain on CPU")
        if self.tensor.dtype != torch.float32:
            raise DataIOError("LoadedVolume tensors must use torch.float32")
        if self.tensor.ndim != 4:
            raise DataIOError(
                f"LoadedVolume tensor must have shape [C,D,H,W], got {tuple(self.tensor.shape)}"
            )
        if not self.tensor.is_contiguous():
            raise DataIOError("LoadedVolume tensor must be contiguous")
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "resolved_path", Path(self.resolved_path))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class MaskSelection:
    """How a stored mask is converted into one binary target channel.

    Exactly one of ``label_id`` and ``channel_index`` may be supplied.  If both
    are ``None``, the source must already contain one binary channel.
    """

    label_id: int | float | None = None
    channel_index: int | None = None
    nonzero_is_foreground: bool = False

    def __post_init__(self) -> None:
        selected = sum(
            value is not None for value in (self.label_id, self.channel_index)
        )
        if selected > 1:
            raise ValueError("label_id and channel_index are mutually exclusive")
        if self.channel_index is not None and self.channel_index < 0:
            raise ValueError("channel_index cannot be negative")
        if self.nonzero_is_foreground and selected:
            raise ValueError(
                "nonzero_is_foreground cannot be combined with label_id or channel_index"
            )


@dataclass(frozen=True, slots=True)
class VolumeReaderOptions:
    """Immutable I/O policy shared by all task datasets."""

    expected_image_channels: int = 1
    expected_spatial_shape: tuple[int, int, int] = (32, 256, 256)
    image_range: tuple[float, float] = (0.0, 1.0)
    image_range_tolerance: float = 1.0e-4
    enforce_image_range: bool = True
    clamp_tiny_range_overshoot: bool = True
    use_npy_mmap: bool = True
    nifti_to_closest_canonical: bool = True
    nifti_xyz_to_dhw: bool = True
    reject_nonfinite: bool = True

    def __post_init__(self) -> None:
        if self.expected_image_channels <= 0:
            raise ValueError("expected_image_channels must be positive")
        if len(self.expected_spatial_shape) != 3:
            raise ValueError("expected_spatial_shape must contain D, H, and W")
        if any(value <= 0 for value in self.expected_spatial_shape):
            raise ValueError("expected_spatial_shape values must be positive")
        low, high = self.image_range
        if not low < high:
            raise ValueError("image_range must satisfy low < high")
        if self.image_range_tolerance < 0:
            raise ValueError("image_range_tolerance cannot be negative")


@dataclass(frozen=True, slots=True)
class LocalCacheOptions:
    """Policy for optional on-demand copies to PBS node-local storage."""

    source_root: Path
    cache_root: Path | None = None
    enabled: bool = False
    verify_size: bool = True
    preserve_mtime: bool = True
    lock_timeout_seconds: float = 300.0
    lock_poll_seconds: float = 0.05

    def __post_init__(self) -> None:
        source_root = Path(self.source_root).expanduser().resolve()
        cache_root = (
            None
            if self.cache_root is None
            else Path(self.cache_root).expanduser().resolve()
        )
        if self.enabled and cache_root is None:
            raise ValueError("cache_root is required when local caching is enabled")
        if cache_root is not None and cache_root == source_root:
            raise ValueError("cache_root must differ from source_root")
        if self.lock_timeout_seconds <= 0 or self.lock_poll_seconds <= 0:
            raise ValueError("cache lock timings must be positive")
        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "cache_root", cache_root)


class NodeLocalFileCache:
    """Resolve source paths and atomically copy them to node-local storage."""

    def __init__(self, options: LocalCacheOptions) -> None:
        self.options = options
        if options.enabled:
            assert options.cache_root is not None
            options.cache_root.mkdir(parents=True, exist_ok=True)

    def source_path(self, path: str | os.PathLike[str]) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.options.source_root / candidate
        candidate = candidate.resolve()
        _assert_below_root(candidate, self.options.source_root, label="dataset source")
        return candidate

    def resolve(self, path: str | os.PathLike[str]) -> tuple[Path, Path]:
        """Return ``(source_path, readable_path)``.

        With caching disabled both paths are identical.  With caching enabled,
        the second path is an atomic node-local copy preserving the relative
        dataset layout.
        """

        source = self.source_path(path)
        _assert_regular_readable_file(source)
        if not self.options.enabled:
            return source, source

        assert self.options.cache_root is not None
        relative = source.relative_to(self.options.source_root)
        destination = (self.options.cache_root / relative).resolve()
        _assert_below_root(destination, self.options.cache_root, label="local cache")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if self._is_valid_cached_copy(source, destination):
            return source, destination

        lock_path = destination.with_name(destination.name + ".lock")
        with _exclusive_file_lock(
            lock_path,
            timeout_seconds=self.options.lock_timeout_seconds,
            poll_seconds=self.options.lock_poll_seconds,
        ):
            if not self._is_valid_cached_copy(source, destination):
                self._copy_atomic(source, destination)

        if not self._is_valid_cached_copy(source, destination):
            raise DataIOError(
                f"Local cache validation failed after copy: {source} -> {destination}"
            )
        return source, destination

    def _is_valid_cached_copy(self, source: Path, destination: Path) -> bool:
        if not destination.is_file():
            return False
        if self.options.verify_size:
            try:
                if destination.stat().st_size != source.stat().st_size:
                    return False
            except OSError:
                return False
        return True

    def _copy_atomic(self, source: Path, destination: Path) -> None:
        temporary = destination.with_name(
            f".{destination.name}.tmp-rank{os.environ.get('RANK', 'na')}-pid{os.getpid()}"
        )
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        try:
            if self.options.preserve_mtime:
                shutil.copy2(source, temporary)
            else:
                shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
            raise


class VolumeReader:
    """Read validated M3D images and segmentation targets."""

    def __init__(
        self,
        options: VolumeReaderOptions,
        file_cache: NodeLocalFileCache,
    ) -> None:
        self.options = options
        self.file_cache = file_cache

    def load_image(self, path: str | os.PathLike[str]) -> LoadedVolume:
        source, resolved = self.file_cache.resolve(path)
        array, geometry, metadata = self._read_array(resolved)
        tensor = self._normalise_layout(
            array,
            kind=VolumeKind.IMAGE,
            source_path=source,
        )
        self._validate_image(tensor, source)
        return LoadedVolume(
            tensor=tensor,
            source_path=source,
            resolved_path=resolved,
            volume_format=VolumeFormat.from_path(source),
            kind=VolumeKind.IMAGE,
            geometry=geometry,
            metadata=metadata,
        )

    def load_mask(
        self,
        path: str | os.PathLike[str],
        *,
        selection: MaskSelection | None = None,
    ) -> LoadedVolume:
        source, resolved = self.file_cache.resolve(path)
        array, geometry, metadata = self._read_array(resolved)
        tensor = self._normalise_layout(
            array,
            kind=VolumeKind.MASK,
            source_path=source,
            permit_multiple_channels=True,
        )
        tensor = self._select_binary_mask(
            tensor,
            source_path=source,
            selection=selection or MaskSelection(),
        )
        self._validate_spatial_shape(tensor, source, kind=VolumeKind.MASK)
        return LoadedVolume(
            tensor=tensor,
            source_path=source,
            resolved_path=resolved,
            volume_format=VolumeFormat.from_path(source),
            kind=VolumeKind.MASK,
            geometry=geometry,
            metadata=metadata,
        )

    def _read_array(
        self,
        path: Path,
    ) -> tuple[np.ndarray, VolumeGeometry, dict[str, Any]]:
        volume_format = VolumeFormat.from_path(path)
        if volume_format is VolumeFormat.NPY:
            mmap_mode = "r" if self.options.use_npy_mmap else None
            try:
                array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
            except Exception as exc:
                raise DataIOError(f"Failed to load NumPy volume {path}: {exc}") from exc
            geometry = VolumeGeometry(source_shape=tuple(int(v) for v in array.shape))
            metadata = {
                "npy_dtype": str(array.dtype),
                "npy_fortran_order": bool(array.flags.f_contiguous),
                "npy_memory_mapped": isinstance(array, np.memmap),
            }
            return np.asarray(array), geometry, metadata

        try:
            import nibabel as nib
        except ImportError as exc:
            raise DataIOError(
                "NIfTI input requires nibabel. Install the reviewed "
                "requirements.txt before reading .nii or .nii.gz files."
            ) from exc

        try:
            image = nib.load(str(path))
            if self.options.nifti_to_closest_canonical:
                image = nib.as_closest_canonical(image)
            array = np.asanyarray(image.dataobj)
        except Exception as exc:
            raise DataIOError(f"Failed to load NIfTI volume {path}: {exc}") from exc

        affine = np.asarray(image.affine, dtype=np.float64)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        orientation = tuple(str(value) for value in nib.aff2axcodes(affine))
        geometry = VolumeGeometry(
            source_shape=tuple(int(value) for value in array.shape),
            canonical_shape_xyz=tuple(int(value) for value in array.shape[:3]),
            spacing_xyz=spacing,
            affine=affine,
            orientation=orientation,
        )
        metadata = {
            "nifti_dtype": str(array.dtype),
            "nifti_canonicalised": self.options.nifti_to_closest_canonical,
            "nifti_xyz_to_dhw": self.options.nifti_xyz_to_dhw,
        }

        if self.options.nifti_xyz_to_dhw:
            array = _nifti_xyz_to_channel_first_dhw(array, path)
        return np.asarray(array), geometry, metadata

    def _normalise_layout(
        self,
        array: np.ndarray,
        *,
        kind: VolumeKind,
        source_path: Path,
        permit_multiple_channels: bool = False,
    ) -> Tensor:
        if array.dtype == np.dtype("O"):
            raise DataIOError(f"Object arrays are forbidden: {source_path}")

        if array.ndim == 3:
            array = array[np.newaxis, ...]
        elif array.ndim == 4:
            # M3D default is [C,D,H,W].  A singleton trailing channel is also
            # accepted for external NIfTI/NumPy exports and moved explicitly.
            if array.shape[0] not in (1, self.options.expected_image_channels):
                if array.shape[-1] == 1:
                    array = np.moveaxis(array, -1, 0)
                elif not permit_multiple_channels:
                    raise DataIOError(
                        f"Expected channel-first image [C,D,H,W] with "
                        f"C={self.options.expected_image_channels}, got {array.shape} "
                        f"from {source_path}"
                    )
        else:
            raise DataIOError(
                f"{kind.value.capitalize()} must be 3D or 4D, got shape "
                f"{array.shape} from {source_path}"
            )

        try:
            contiguous = np.array(array, dtype=np.float32, order="C", copy=True)
        except (TypeError, ValueError, MemoryError) as exc:
            raise DataIOError(
                f"Cannot convert {source_path} to contiguous float32: {exc}"
            ) from exc

        tensor = torch.from_numpy(contiguous)
        if tensor.ndim != 4:
            raise DataIOError(
                f"Internal layout conversion failed for {source_path}: {tensor.shape}"
            )
        if self.options.reject_nonfinite and not bool(torch.isfinite(tensor).all()):
            bad_count = int((~torch.isfinite(tensor)).sum().item())
            raise DataIOError(
                f"{source_path} contains {bad_count} NaN/Inf values"
            )
        return tensor

    def _validate_image(self, tensor: Tensor, path: Path) -> None:
        if tensor.shape[0] != self.options.expected_image_channels:
            raise DataIOError(
                f"Image {path} has {tensor.shape[0]} channels; expected "
                f"{self.options.expected_image_channels}"
            )
        self._validate_spatial_shape(tensor, path, kind=VolumeKind.IMAGE)

        if not self.options.enforce_image_range:
            return
        low, high = self.options.image_range
        tolerance = self.options.image_range_tolerance
        actual_low = float(tensor.amin().item())
        actual_high = float(tensor.amax().item())
        if actual_low < low - tolerance or actual_high > high + tolerance:
            raise DataIOError(
                f"Image {path} is outside the expected normalized range "
                f"[{low}, {high}]: min={actual_low:.7g}, max={actual_high:.7g}. "
                "M3D expects preprocessing to happen before training; this "
                "loader does not silently min-max normalize volumes."
            )
        if self.options.clamp_tiny_range_overshoot and (
            actual_low < low or actual_high > high
        ):
            tensor.clamp_(min=low, max=high)

    def _validate_spatial_shape(
        self,
        tensor: Tensor,
        path: Path,
        *,
        kind: VolumeKind,
    ) -> None:
        spatial = tuple(int(value) for value in tensor.shape[-3:])
        if spatial != self.options.expected_spatial_shape:
            raise DataIOError(
                f"{kind.value.capitalize()} {path} has spatial shape {spatial}; "
                f"expected {self.options.expected_spatial_shape}. No resize is "
                "performed inside training I/O because image and mask geometry "
                "must be prepared consistently beforehand."
            )

    def _select_binary_mask(
        self,
        tensor: Tensor,
        *,
        source_path: Path,
        selection: MaskSelection,
    ) -> Tensor:
        if selection.label_id is not None:
            target = tensor.eq(float(selection.label_id))
        elif selection.channel_index is not None:
            index = selection.channel_index
            if index >= tensor.shape[0]:
                raise DataIOError(
                    f"Mask channel {index} does not exist in {source_path}; "
                    f"available channels={tensor.shape[0]}"
                )
            channel = tensor[index : index + 1]
            target = _require_binary(channel, source_path)
        elif selection.nonzero_is_foreground:
            if tensor.shape[0] != 1:
                raise DataIOError(
                    "nonzero_is_foreground requires a single-channel mask; "
                    f"got {tensor.shape[0]} channels in {source_path}"
                )
            target = tensor.ne(0)
        else:
            if tensor.shape[0] != 1:
                raise DataIOError(
                    f"Mask {source_path} has {tensor.shape[0]} channels. Supply "
                    "MaskSelection(channel_index=...) or a label_id explicitly."
                )
            target = _require_binary(tensor, source_path)

        # Equality/selection returns bool.  Float32 is the model contract.
        result = target.to(dtype=torch.float32).contiguous()
        if result.shape[0] != 1:
            raise DataIOError(
                f"Binary target from {source_path} must contain one channel, "
                f"got {tuple(result.shape)}"
            )
        # No foreground-count condition belongs here.  All-zero is valid.
        return result


def build_volume_reader(config: ExperimentConfig) -> VolumeReader:
    """Construct the shared reader from a validated experiment configuration."""

    main = config.model.main_vision
    seg = config.model.seg_vision
    if seg.enabled and (
        main.image_channels != seg.image_channels
        or main.image_size != seg.image_size
    ):
        raise DataIOError(
            "The two image encoders remain independent, but this M3D training "
            "pipeline feeds the same preprocessed volume to both. Their input "
            "channel count and image_size must therefore match."
        )

    source_root = Path(config.data.paths.data_root)
    cache_root = (
        None
        if config.data.local_cache_root is None
        else Path(config.data.local_cache_root)
    )
    cache_options = LocalCacheOptions(
        source_root=source_root,
        cache_root=cache_root,
        enabled=cache_root is not None,
    )
    reader_options = VolumeReaderOptions(
        expected_image_channels=main.image_channels,
        expected_spatial_shape=main.image_size,
        # Original M3D comments explicitly state that stored arrays are 0-1.
        image_range=(0.0, 1.0),
        enforce_image_range=True,
    )
    return VolumeReader(reader_options, NodeLocalFileCache(cache_options))


def read_utf8_text(
    path: str | os.PathLike[str],
    *,
    file_cache: NodeLocalFileCache,
    strip: bool = True,
    reject_empty: bool = True,
) -> str:
    """Read one UTF-8 text file through the same optional local cache."""

    source, resolved = file_cache.resolve(path)
    try:
        raw = resolved.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataIOError(f"Text file is not valid UTF-8: {source}") from exc
    except OSError as exc:
        raise DataIOError(f"Failed to read text file {source}: {exc}") from exc

    if "\x00" in raw:
        raise DataIOError(f"Text file contains NUL bytes: {source}")
    text = raw.strip() if strip else raw
    if reject_empty and not text:
        raise DataIOError(f"Text file is empty: {source}")
    return text


def write_volume_manifest(
    records: Sequence[LoadedVolume],
    destination: str | os.PathLike[str],
) -> None:
    """Atomically write lightweight I/O diagnostics for reproducibility."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "source_path": str(record.source_path),
            "resolved_path": str(record.resolved_path),
            "format": record.volume_format.value,
            "kind": record.kind.value,
            "shape": list(record.tensor.shape),
            "dtype": str(record.tensor.dtype),
            "min": float(record.tensor.amin().item()),
            "max": float(record.tensor.amax().item()),
            "geometry": record.geometry.to_jsonable(),
            "metadata": dict(record.metadata),
        }
        for record in records
    ]
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _nifti_xyz_to_channel_first_dhw(array: np.ndarray, path: Path) -> np.ndarray:
    """Convert nibabel array order to channel-first M3D order.

    nibabel exposes 3D volumes as ``[X,Y,Z]``.  M3D names its tensor dimensions
    ``[D,H,W]`` and uses the common medical deep-learning mapping ``D=Z``,
    ``H=Y``, ``W=X``.
    """

    if array.ndim == 3:
        return np.transpose(array, (2, 1, 0))[np.newaxis, ...]
    if array.ndim == 4:
        # NIfTI channels/time are conventionally last: [X,Y,Z,C].
        return np.transpose(array, (3, 2, 1, 0))
    raise DataIOError(
        f"NIfTI {path} must have shape [X,Y,Z] or [X,Y,Z,C], got {array.shape}"
    )


def _require_binary(tensor: Tensor, path: Path) -> Tensor:
    valid = torch.logical_or(tensor.eq(0), tensor.eq(1))
    if not bool(valid.all()):
        values = torch.unique(tensor)
        preview = values[:16].tolist()
        suffix = "..." if values.numel() > 16 else ""
        raise DataIOError(
            f"Mask {path} is not binary; unique values include {preview}{suffix}. "
            "Select a label_id explicitly for multi-class masks."
        )
    return tensor.to(dtype=torch.bool)


def _assert_below_root(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DataIOError(f"{label} path escapes configured root: {path}") from exc


def _assert_regular_readable_file(path: Path) -> None:
    try:
        information = path.stat()
    except FileNotFoundError as exc:
        raise DataIOError(f"Dataset file does not exist: {path}") from exc
    except OSError as exc:
        raise DataIOError(f"Cannot stat dataset file {path}: {exc}") from exc
    if not stat.S_ISREG(information.st_mode):
        raise DataIOError(f"Dataset path is not a regular file: {path}")
    if not os.access(path, os.R_OK):
        raise DataIOError(f"Dataset file is not readable: {path}")


@contextlib.contextmanager
def _exclusive_file_lock(
    lock_path: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    start = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start >= timeout_seconds:
                    raise DataIOError(
                        f"Timed out waiting for cache lock {lock_path} after "
                        f"{timeout_seconds:.1f}s"
                    )
                time.sleep(poll_seconds)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Persist a rename on Linux filesystems where directory fsync is allowed."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "DataIOError",
    "LoadedVolume",
    "LocalCacheOptions",
    "MaskSelection",
    "NodeLocalFileCache",
    "VolumeFormat",
    "VolumeGeometry",
    "VolumeKind",
    "VolumeReader",
    "VolumeReaderOptions",
    "build_volume_reader",
    "read_utf8_text",
    "write_volume_manifest",
]
