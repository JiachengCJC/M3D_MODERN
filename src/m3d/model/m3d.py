"""Complete task-routed M3D model for M3D-Modernized.

This module is the point where every previously implemented component becomes
one trainable multimodal model.  The execution graph is deliberately explicit:

Text task
---------
``images -> Main 3D ViT -> MM projector -> Phi-3 -> language objective``

Segmentation task
-----------------
``images -> Main 3D ViT -> MM projector -> Phi-3``
``Phi-3 [SEG] state -> segmentation projector``
``images -> independent SegVol 3D ViT -> prompt encoder -> mask decoder``
``language objective + Dice + BCE``

The two image encoders share only the Python implementation in ``vit3d.py``.
They are separate modules, parameters, gradients, optimizer states and
checkpoint entries.  Task routing uses :class:`m3d.data.schema.TaskName`; a
segmentation target is never inspected to decide whether SegVol should run.  A
valid all-zero target therefore executes the complete segmentation graph.

Compatibility names
-------------------
The outer module preserves the important original M3D names:

* ``vision_tower`` for the Main 3D ViT wrapper;
* ``mm_projector`` for the Main-ViT-to-Phi-3 projector;
* ``seg_projector`` for the Phi-3-to-SegVol prompt projector; and
* ``seg_module`` for the complete SegVol module.

The language model is wrapped as ``language_model`` because its modern wrapper
owns selective-logit loss computation, visual-token validation and PEFT/LoRA.
Published component checkpoints are loaded by the component-aware helpers in
``checkpoint.py`` rather than by relying on one monolithic legacy key layout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from m3d.config import (
    ExperimentConfig,
    OptimizationConfig,
    ProjectorConfig,
    SegmentationConfig,
    VisionEncoderConfig,
)
from m3d.data.schema import M3DBatch, TaskName
from m3d.model.checkpoint import (
    CheckpointSource,
    EmbeddingLoadReport,
    ModuleLoadReport,
    load_input_embeddings_from_projector_checkpoint,
    load_main_vision_checkpoint,
    load_projector_checkpoint,
    load_segmentation_module_checkpoint,
    load_segmentation_vision_checkpoint,
    read_checkpoint,
)
from m3d.model.language import (
    LanguageModelBuildReport,
    LanguageModelOutput,
    LogitsMode,
    M3DLanguageModel,
    build_language_model,
)
from m3d.model.loss import M3DLoss, M3DLossOutput, build_m3d_loss
from m3d.model.projector import (
    ProjectorOutput,
    SpatialPoolingProjector,
    build_multimodal_projector,
    validate_visual_token_contract,
)
from m3d.model.segmentation_prompt import (
    SegmentationPromptOutput,
    SegmentationPromptProjector,
    build_segmentation_prompt_projector,
    validate_segmentation_prompt_contract,
)
from m3d.model.segvol import SegVol, SegVolOutput, build_segvol_module
from m3d.model.vit3d import (
    ViT3DEncoder,
    ViT3DTower,
    VisionEncoderOutput,
    assert_independent_encoders,
    build_main_vision_tower,
)


class M3DConfigurationError(ValueError):
    """Raised when independently valid M3D components cannot be connected."""


class M3DInputError(ValueError):
    """Raised when a runtime batch violates the complete-model contract."""


class M3DExecutionError(RuntimeError):
    """Raised when one component returns an invalid result."""


@dataclass(frozen=True, slots=True)
class M3DPretrainedLoadReport:
    """Auditable reports for all optional published component checkpoints."""

    main_vision: ModuleLoadReport | None = None
    multimodal_projector: ModuleLoadReport | None = None
    projector_input_embeddings: EmbeddingLoadReport | None = None
    segmentation_module: ModuleLoadReport | None = None
    segmentation_vision_override: ModuleLoadReport | None = None

    def to_dict(self) -> dict[str, Any]:
        def module_report(value: ModuleLoadReport | None) -> dict[str, Any] | None:
            return None if value is None else value.to_dict()

        embedding = self.projector_input_embeddings
        return {
            "main_vision": module_report(self.main_vision),
            "multimodal_projector": module_report(self.multimodal_projector),
            "projector_input_embeddings": (
                None if embedding is None else dataclasses.asdict(embedding)
            ),
            "segmentation_module": module_report(self.segmentation_module),
            "segmentation_vision_override": module_report(
                self.segmentation_vision_override
            ),
        }


@dataclass(frozen=True, slots=True)
class M3DParameterSummary:
    """Parameter counts before distributed wrapping or optimizer construction."""

    total: int
    trainable: int
    main_vision: int
    multimodal_projector: int
    language_model: int
    segmentation_projector: int
    segmentation_module: int
    shared_image_encoder_parameters: int
    shared_image_encoder_storages: int

    @property
    def frozen(self) -> int:
        return self.total - self.trainable

    @property
    def trainable_fraction(self) -> float:
        return self.trainable / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "frozen": self.frozen,
            "trainable_fraction": self.trainable_fraction,
        }


@dataclass(frozen=True, slots=True)
class M3DBuildReport:
    """Serializable complete-model construction report."""

    language: LanguageModelBuildReport | None
    parameters: M3DParameterSummary
    pretrained: M3DPretrainedLoadReport
    segmentation_enabled: bool
    visual_token_count: int
    language_hidden_size: int
    segmentation_prompt_dim: int | None
    legacy_component_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": None if self.language is None else self.language.to_dict(),
            "parameters": self.parameters.to_dict(),
            "pretrained": self.pretrained.to_dict(),
            "segmentation_enabled": self.segmentation_enabled,
            "visual_token_count": self.visual_token_count,
            "language_hidden_size": self.language_hidden_size,
            "segmentation_prompt_dim": self.segmentation_prompt_dim,
            "legacy_component_names": list(self.legacy_component_names),
        }


@dataclass(frozen=True, slots=True)
class M3DModelOutput:
    """Structured output from one explicit task graph.

    ``loss_output`` is present only when labels are supplied.  For a
    segmentation task, labels and segmentation targets must be supplied
    together; this reproduces the joint language + dense-mask objective.

    Large intermediate tensors from the Main ViT and MM projector are returned
    only when ``return_intermediates=True``.  The language and segmentation
    outputs remain available because evaluators need generated logits, hidden
    states, mask logits and count statistics.
    """

    task: TaskName
    language_output: LanguageModelOutput
    segmentation_output: SegVolOutput | None
    loss_output: M3DLossOutput | None
    segmentation_prompt_output: SegmentationPromptOutput | None = None
    main_vision_output: VisionEncoderOutput | None = None
    projector_output: ProjectorOutput | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName.parse(self.task))
        has_segmentation = self.segmentation_output is not None
        if self.task.requires_segmentation_target != has_segmentation:
            raise M3DExecutionError(
                "Segmentation output presence disagrees with explicit task: "
                f"task={self.task.value}, has_segmentation={has_segmentation}."
            )
        if self.task.requires_segmentation_target != (
            self.segmentation_prompt_output is not None
        ):
            raise M3DExecutionError(
                "Segmentation prompt presence disagrees with explicit task."
            )
        if self.loss_output is not None and self.loss_output.task is not self.task:
            raise M3DExecutionError(
                "Loss task disagrees with complete-model output task."
            )

    @property
    def loss(self) -> Tensor | None:
        """Trainer-compatible access to the scalar local objective."""

        return None if self.loss_output is None else self.loss_output.total

    @property
    def logits(self) -> Tensor | None:
        """Expose language logits when requested through ``logits_mode``."""

        return self.language_output.logits

    @property
    def segmentation_logits(self) -> Tensor | None:
        return (
            None
            if self.segmentation_output is None
            else self.segmentation_output.logits
        )

    def detached_metrics(self) -> dict[str, Tensor]:
        if self.loss_output is None:
            return {}
        return self.loss_output.detached_metrics()


@dataclass(frozen=True, slots=True)
class _LanguagePathOutput:
    """Internal result shared by the text and segmentation forward graphs."""

    main_vision: VisionEncoderOutput
    projected: ProjectorOutput
    language: LanguageModelOutput


def _parameter_count(module: nn.Module | None, *, trainable_only: bool = False) -> int:
    if module is None:
        return 0
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def _storage_identity(parameter: nn.Parameter) -> tuple[Any, ...]:
    """Return a best-effort identity for detecting accidental shared storage."""

    if parameter.device.type == "meta":
        return ("meta", id(parameter))
    try:
        storage = parameter.untyped_storage()
        return (
            parameter.device.type,
            parameter.device.index,
            storage.data_ptr(),
            storage.nbytes(),
        )
    except RuntimeError:
        return ("object", id(parameter))


def _shared_parameter_counts(
    first: nn.Module,
    second: nn.Module,
) -> tuple[int, int]:
    first_parameters = tuple(first.parameters())
    second_parameters = tuple(second.parameters())
    shared_objects = len(
        {id(parameter) for parameter in first_parameters}
        & {id(parameter) for parameter in second_parameters}
    )
    first_storages = {_storage_identity(parameter) for parameter in first_parameters}
    second_storages = {_storage_identity(parameter) for parameter in second_parameters}
    shared_storages = len(first_storages & second_storages)
    return shared_objects, shared_storages


def _checkpoint_contains_input_embeddings(source: CheckpointSource) -> bool:
    suffixes = ("model.embed_tokens.weight", "embed_tokens.weight")
    return any(
        any(key == suffix or key.endswith("." + suffix) for suffix in suffixes)
        for key in source.tensors
    )


class M3DModel(nn.Module):
    """Complete M3D model with separate text and segmentation execution graphs."""

    def __init__(
        self,
        *,
        vision_tower: ViT3DTower,
        mm_projector: SpatialPoolingProjector,
        language_model: M3DLanguageModel,
        objective: M3DLoss,
        seg_projector: SegmentationPromptProjector | None = None,
        seg_module: SegVol | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(vision_tower, ViT3DTower):
            raise M3DConfigurationError("vision_tower must be a ViT3DTower.")
        if not isinstance(mm_projector, SpatialPoolingProjector):
            raise M3DConfigurationError(
                "mm_projector must be a SpatialPoolingProjector."
            )
        if not isinstance(language_model, M3DLanguageModel):
            raise M3DConfigurationError(
                "language_model must be an M3DLanguageModel."
            )
        if not isinstance(objective, M3DLoss):
            raise M3DConfigurationError("objective must be an M3DLoss.")
        if (seg_projector is None) != (seg_module is None):
            raise M3DConfigurationError(
                "seg_projector and seg_module must either both exist or both be absent."
            )

        # These names intentionally preserve the original M3D component names.
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.language_model = language_model
        self.seg_projector = seg_projector
        self.seg_module = seg_module
        self.objective = objective

        self._validate_component_contracts()

    # ------------------------------------------------------------------
    # Legacy-compatible accessors
    # ------------------------------------------------------------------

    def get_model(self) -> "M3DModel":
        """Compatibility with the old ``LamedMetaForCausalLM.get_model`` API."""

        return self

    def get_vision_tower(self) -> ViT3DTower:
        return self.vision_tower

    @property
    def seg_enable(self) -> bool:
        return self.seg_module is not None

    @property
    def main_image_encoder(self) -> ViT3DEncoder:
        return self.vision_tower.vision_tower

    @property
    def segmentation_image_encoder(self) -> ViT3DEncoder | None:
        return None if self.seg_module is None else self.seg_module.image_encoder

    @property
    def segmentation_token_id(self) -> int:
        return self.language_model.segmentation_token_id

    @property
    def language_hidden_size(self) -> int:
        return self.language_model.hidden_size

    # ------------------------------------------------------------------
    # Static validation and reports
    # ------------------------------------------------------------------

    def _validate_component_contracts(self) -> None:
        errors: list[str] = []

        if self.vision_tower.hidden_size != self.mm_projector.in_dim:
            errors.append(
                "Main ViT hidden size differs from MM projector input: "
                f"{self.vision_tower.hidden_size} != {self.mm_projector.in_dim}"
            )
        if self.mm_projector.out_dim != self.language_model.hidden_size:
            errors.append(
                "MM projector output differs from language hidden size: "
                f"{self.mm_projector.out_dim} != {self.language_model.hidden_size}"
            )
        if self.mm_projector.proj_out_num != self.language_model.visual_token_count:
            errors.append(
                "MM projector visual-token count differs from tokenizer contract: "
                f"{self.mm_projector.proj_out_num} != "
                f"{self.language_model.visual_token_count}"
            )
        if self.vision_tower.num_patches != self.mm_projector.input_token_count:
            errors.append(
                "Main ViT patch count differs from MM projector input count: "
                f"{self.vision_tower.num_patches} != "
                f"{self.mm_projector.input_token_count}"
            )

        if self.seg_enable:
            assert self.seg_projector is not None
            assert self.seg_module is not None
            if self.seg_projector.language_hidden_size != self.language_model.hidden_size:
                errors.append(
                    "Segmentation projector input differs from language hidden size: "
                    f"{self.seg_projector.language_hidden_size} != "
                    f"{self.language_model.hidden_size}"
                )
            if self.seg_projector.prompt_embed_dim != (
                self.seg_module.prompt_encoder.embed_dim
            ):
                errors.append(
                    "Segmentation projector output differs from SegVol prompt dim: "
                    f"{self.seg_projector.prompt_embed_dim} != "
                    f"{self.seg_module.prompt_encoder.embed_dim}"
                )

            assert_independent_encoders(
                self.main_image_encoder,
                self.seg_module.image_encoder,
            )
            shared_objects, shared_storages = _shared_parameter_counts(
                self.main_image_encoder,
                self.seg_module.image_encoder,
            )
            if shared_objects or shared_storages:
                errors.append(
                    "Main and SegVol image encoders must not share Parameters or "
                    f"storage: parameters={shared_objects}, storages={shared_storages}"
                )

        if errors:
            raise M3DConfigurationError(
                "Incompatible complete M3D components:\n  - "
                + "\n  - ".join(errors)
            )

    def parameter_summary(self) -> M3DParameterSummary:
        shared_parameters = 0
        shared_storages = 0
        if self.seg_module is not None:
            shared_parameters, shared_storages = _shared_parameter_counts(
                self.main_image_encoder,
                self.seg_module.image_encoder,
            )
        return M3DParameterSummary(
            total=_parameter_count(self),
            trainable=_parameter_count(self, trainable_only=True),
            main_vision=_parameter_count(self.vision_tower),
            multimodal_projector=_parameter_count(self.mm_projector),
            language_model=_parameter_count(self.language_model),
            segmentation_projector=_parameter_count(self.seg_projector),
            segmentation_module=_parameter_count(self.seg_module),
            shared_image_encoder_parameters=shared_parameters,
            shared_image_encoder_storages=shared_storages,
        )

    def no_weight_decay_parameter_names(self) -> frozenset[str]:
        """Return exact outer names for positional and class embeddings."""

        names = {
            f"vision_tower.vision_tower.{name}"
            for name in self.main_image_encoder.no_weight_decay_parameter_names()
        }
        if self.seg_module is not None:
            names.update(
                f"seg_module.image_encoder.{name}"
                for name in self.seg_module.image_encoder.no_weight_decay_parameter_names()
            )
        return frozenset(names)

    # ------------------------------------------------------------------
    # Shared Main-ViT -> Phi-3 path
    # ------------------------------------------------------------------

    def encode_images(
        self,
        images: Tensor,
        *,
        return_output: bool = False,
    ) -> Tensor | ProjectorOutput:
        """Legacy-compatible Main 3D ViT + MM projector image encoding."""

        vision_output = self.vision_tower.forward_with_output(
            images,
            output_hidden_states=False,
        )
        projected = self.mm_projector(vision_output, return_output=True)
        if not isinstance(projected, ProjectorOutput):
            raise M3DExecutionError(
                "MM projector did not return the requested structured output."
            )
        return projected if return_output else projected.projected_tokens

    def _run_language_path(
        self,
        *,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None,
        position_ids: Tensor | None,
        logits_mode: LogitsMode | str,
        use_cache: bool,
    ) -> _LanguagePathOutput:
        self._validate_common_inputs(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        main_vision_output = self.vision_tower.forward_with_output(
            images,
            output_hidden_states=False,
        )
        projected = self.mm_projector(
            main_vision_output,
            return_output=True,
        )
        if not isinstance(projected, ProjectorOutput):
            raise M3DExecutionError(
                "MM projector did not return ProjectorOutput."
            )
        language = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeddings=projected.projected_tokens,
            labels=labels,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_mode=logits_mode,
        )
        return _LanguagePathOutput(
            main_vision=main_vision_output,
            projected=projected,
            language=language,
        )

    # ------------------------------------------------------------------
    # Separate task graphs
    # ------------------------------------------------------------------

    def forward_text(
        self,
        *,
        task: TaskName | str,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
        position_ids: Tensor | None = None,
        logits_mode: LogitsMode | str = LogitsMode.NONE,
        use_cache: bool = False,
        return_intermediates: bool = False,
    ) -> M3DModelOutput:
        """Run a text-only objective without touching any SegVol module."""

        resolved_task = TaskName.parse(task)
        if resolved_task.requires_segmentation_target:
            raise M3DInputError(
                "forward_text cannot execute the segmentation task."
            )

        path = self._run_language_path(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            logits_mode=logits_mode,
            use_cache=use_cache,
        )
        loss_output = (
            None
            if labels is None
            else self.objective(
                task=resolved_task,
                language_output=path.language,
            )
        )
        return M3DModelOutput(
            task=resolved_task,
            language_output=path.language,
            segmentation_output=None,
            loss_output=loss_output,
            main_vision_output=(path.main_vision if return_intermediates else None),
            projector_output=(path.projected if return_intermediates else None),
        )

    def forward_segmentation(
        self,
        *,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
        segmentation_targets: Tensor | None = None,
        position_ids: Tensor | None = None,
        logits_mode: LogitsMode | str = LogitsMode.NONE,
        use_cache: bool = False,
        multimask_output: bool = False,
        return_intermediates: bool = False,
    ) -> M3DModelOutput:
        """Run the language graph and the independent SegVol graph.

        ``labels`` and ``segmentation_targets`` are both absent for pure
        inference, or both present for the original joint M3D training
        objective.  An all-zero target is fully valid and does not alter routing.
        """

        if self.seg_projector is None or self.seg_module is None:
            raise M3DConfigurationError(
                "Segmentation was requested but the complete model was built "
                "without seg_projector/seg_module."
            )
        if (labels is None) != (segmentation_targets is None):
            raise M3DInputError(
                "Segmentation training requires labels and segmentation_targets "
                "together.  Omit both for inference."
            )
        if labels is not None and multimask_output:
            raise M3DInputError(
                "The binary M3D segmentation objective expects one selected mask; "
                "set multimask_output=False during training."
            )
        if segmentation_targets is not None:
            self._validate_segmentation_targets(images, segmentation_targets)

        path = self._run_language_path(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            logits_mode=logits_mode,
            use_cache=use_cache,
        )

        segmentation_prompt = self.seg_projector.extract_and_project(
            last_hidden_state=path.language.last_hidden_state,
            input_ids=input_ids,
            segmentation_token_id=self.segmentation_token_id,
            attention_mask=attention_mask,
        )
        segmentation_output = self.seg_module(
            images,
            text_embedding=segmentation_prompt.prompt_embeddings,
            multimask_output=multimask_output,
            return_structured=True,
        )
        if not isinstance(segmentation_output, SegVolOutput):
            raise M3DExecutionError(
                "SegVol did not return the requested structured output."
            )

        loss_output = (
            None
            if labels is None
            else self.objective(
                task=TaskName.SEGMENTATION,
                language_output=path.language,
                segmentation_output=segmentation_output,
                segmentation_targets=segmentation_targets,
            )
        )
        return M3DModelOutput(
            task=TaskName.SEGMENTATION,
            language_output=path.language,
            segmentation_output=segmentation_output,
            loss_output=loss_output,
            segmentation_prompt_output=segmentation_prompt,
            main_vision_output=(path.main_vision if return_intermediates else None),
            projector_output=(path.projected if return_intermediates else None),
        )

    def forward(
        self,
        *,
        task: TaskName | str,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
        segmentation_targets: Tensor | None = None,
        position_ids: Tensor | None = None,
        logits_mode: LogitsMode | str = LogitsMode.NONE,
        use_cache: bool = False,
        multimask_output: bool = False,
        return_intermediates: bool = False,
    ) -> M3DModelOutput:
        """Dispatch to one graph using explicit Python task metadata.

        The branch is selected before tensor execution.  It never calls
        ``segmentation_targets.sum()`` and never creates a fake zero mask for a
        text task.  Task-homogeneous distributed sampling guarantees all ranks
        select the same branch at the same step.
        """

        resolved_task = TaskName.parse(task)
        if resolved_task.requires_segmentation_target:
            return self.forward_segmentation(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                segmentation_targets=segmentation_targets,
                position_ids=position_ids,
                logits_mode=logits_mode,
                use_cache=use_cache,
                multimask_output=multimask_output,
                return_intermediates=return_intermediates,
            )

        if segmentation_targets is not None:
            raise M3DInputError(
                f"Task {resolved_task.value!r} must not carry segmentation targets."
            )
        if multimask_output:
            raise M3DInputError(
                "multimask_output is meaningful only for the segmentation task."
            )
        return self.forward_text(
            task=resolved_task,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            logits_mode=logits_mode,
            use_cache=use_cache,
            return_intermediates=return_intermediates,
        )

    def forward_batch(
        self,
        batch: M3DBatch,
        *,
        logits_mode: LogitsMode | str = LogitsMode.NONE,
        return_intermediates: bool = False,
    ) -> M3DModelOutput:
        """Execute the exact graph represented by a validated ``M3DBatch``."""

        if not isinstance(batch, M3DBatch):
            raise TypeError(
                f"batch must be M3DBatch, got {type(batch).__name__}."
            )
        return self(
            task=batch.task,
            images=batch.images,
            input_ids=batch.text.input_ids,
            attention_mask=batch.text.attention_mask,
            labels=batch.text.labels,
            segmentation_targets=batch.segmentation_targets,
            logits_mode=logits_mode,
            use_cache=False,
            multimask_output=False,
            return_intermediates=return_intermediates,
        )

    # ------------------------------------------------------------------
    # Runtime input contracts
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_common_inputs(
        *,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None,
        position_ids: Tensor | None,
    ) -> None:
        if images.ndim != 5:
            raise M3DInputError(
                f"images must have shape [B,C,D,H,W], got {tuple(images.shape)}."
            )
        if not images.is_floating_point():
            raise M3DInputError("images must use a floating-point dtype.")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise M3DInputError(
                "input_ids must be torch.long with shape [B,S]."
            )
        if attention_mask.shape != input_ids.shape:
            raise M3DInputError(
                "attention_mask must have the same [B,S] shape as input_ids."
            )
        if attention_mask.dtype not in (torch.bool, torch.long, torch.int32):
            raise M3DInputError(
                "attention_mask must be bool or an integer 0/1 tensor."
            )
        if int(images.shape[0]) != int(input_ids.shape[0]):
            raise M3DInputError(
                "Image and text batch sizes differ: "
                f"images={int(images.shape[0])}, text={int(input_ids.shape[0])}."
            )
        if images.device != input_ids.device or input_ids.device != attention_mask.device:
            raise M3DInputError(
                "images, input_ids and attention_mask must be on the same device."
            )
        if labels is not None:
            if labels.shape != input_ids.shape or labels.dtype != torch.long:
                raise M3DInputError(
                    "labels must be torch.long with the same shape as input_ids."
                )
            if labels.device != input_ids.device:
                raise M3DInputError("labels and input_ids are on different devices.")
        if position_ids is not None:
            if position_ids.shape != input_ids.shape or position_ids.dtype != torch.long:
                raise M3DInputError(
                    "position_ids must be torch.long with the same shape as input_ids."
                )
            if position_ids.device != input_ids.device:
                raise M3DInputError(
                    "position_ids and input_ids are on different devices."
                )

    @staticmethod
    def _validate_segmentation_targets(images: Tensor, targets: Tensor) -> None:
        expected = (int(images.shape[0]), 1, *tuple(int(v) for v in images.shape[2:]))
        if tuple(targets.shape) != expected:
            raise M3DInputError(
                "segmentation_targets must match [B,1,D,H,W]: "
                f"received={tuple(targets.shape)}, expected={expected}."
            )
        if not targets.is_floating_point():
            raise M3DInputError(
                "segmentation_targets must use a floating-point dtype."
            )
        if targets.device != images.device:
            raise M3DInputError(
                "segmentation_targets and images are on different devices."
            )


# ---------------------------------------------------------------------------
# Construction and published-checkpoint loading
# ---------------------------------------------------------------------------
def _checkpoint_target_embedding(module: nn.Module) -> nn.Embedding:
    """Return the trainable embedding inside a bare or PEFT-wrapped module."""

    if isinstance(module, nn.Embedding):
        return module

    modules_to_save = getattr(module, "modules_to_save", None)
    if isinstance(modules_to_save, nn.ModuleDict):
        candidates = [
            child
            for child in modules_to_save.values()
            if isinstance(child, nn.Embedding)
        ]
        if len(candidates) == 1:
            return candidates[0]

        raise M3DConfigurationError(
            "Expected exactly one PEFT modules_to_save embedding, "
            f"but found {len(candidates)}."
        )

    raise M3DConfigurationError(
        "The projector checkpoint contains embed_tokens.weight, but the "
        "language input embedding is neither nn.Embedding nor a supported "
        f"PEFT wrapper: {type(module).__name__}."
    )

def _load_pretrained_components(
    model: M3DModel,
    config: ExperimentConfig,
    *,
    strict: bool,
) -> M3DPretrainedLoadReport:
    main_report: ModuleLoadReport | None = None
    projector_report: ModuleLoadReport | None = None
    embedding_report: EmbeddingLoadReport | None = None
    segmentation_report: ModuleLoadReport | None = None
    segmentation_override_report: ModuleLoadReport | None = None

    if config.model.main_vision.checkpoint_path is not None:
        main_report = load_main_vision_checkpoint(
            model.vision_tower,
            config.model.main_vision.checkpoint_path,
            strict=strict,
        )

    if config.model.projector.checkpoint_path is not None:
        projector_source = read_checkpoint(config.model.projector.checkpoint_path)
        projector_report = load_projector_checkpoint(
            model.mm_projector,
            projector_source,
            strict=strict,
        )
        if _checkpoint_contains_input_embeddings(projector_source):
            input_embedding = _checkpoint_target_embedding(
                model.language_model.get_input_embeddings()
            )
            added_tokens = int(
                model.language_model.tokenizer_metadata.added_token_count
            )
            if added_tokens <= 0:
                raise M3DConfigurationError(
                    "The projector checkpoint contains added-token embeddings, "
                    "but tokenizer metadata reports no added tokens."
                )
            embedding_report = load_input_embeddings_from_projector_checkpoint(
                input_embedding,
                projector_source,
                num_new_tokens=added_tokens,
            )

    if config.model.segmentation.enabled:
        if model.seg_module is None:
            raise M3DConfigurationError(
                "Configuration enables segmentation but model has no seg_module."
            )
        if config.model.segmentation.checkpoint_path is not None:
            segmentation_report = load_segmentation_module_checkpoint(
                model.seg_module,
                config.model.segmentation.checkpoint_path,
                strict=strict,
            )
        # A separately supplied seg_vision checkpoint intentionally overrides
        # only the image encoder after the full SegVol checkpoint is loaded.
        if config.model.seg_vision.checkpoint_path is not None:
            segmentation_override_report = load_segmentation_vision_checkpoint(
                model.seg_module.image_encoder,
                config.model.seg_vision.checkpoint_path,
                strict=strict,
            )

    model._validate_component_contracts()
    return M3DPretrainedLoadReport(
        main_vision=main_report,
        multimodal_projector=projector_report,
        projector_input_embeddings=embedding_report,
        segmentation_module=segmentation_report,
        segmentation_vision_override=segmentation_override_report,
    )


def _apply_training_stage_policy(model: "M3DModel", config: ExperimentConfig) -> None:
    """Apply stage-level trainability rules after checkpoint loading.

    Component-level ``freeze`` fields remain authoritative for normal LoRA and
    joint fine-tuning.  Projector pretraining is stricter: the complete model is
    frozen first, then only the multimodal projector and token embedding/output
    tables are enabled.  This reproduces the original M3D stage-1 objective
    without allocating gradients or Adam states for the full Phi-3 model.
    """

    stage = str(config.optimization.stage)
    if stage != "projector_pretrain":
        return

    model.requires_grad_(False)
    model.mm_projector.requires_grad_(True)
    input_embeddings = model.language_model.get_input_embeddings()
    output_embeddings = model.language_model.get_output_embeddings()
    input_embeddings.requires_grad_(True)
    output_embeddings.requires_grad_(True)

    if model.seg_enable:
        raise M3DConfigurationError(
            "projector_pretrain must disable the segmentation branch."
        )


def build_m3d_model(
    config: ExperimentConfig,
    tokenizer_bundle: Any,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    load_pretrained_components: bool = True,
    strict_pretrained: bool = True,
) -> tuple[M3DModel, M3DBuildReport]:
    """Build Phi-3, both independent image encoders and every connector.

    The language model is constructed first because its hidden size determines
    both the MM projector output size and segmentation-projector input size.
    Published component checkpoints are loaded only after the complete static
    contracts have been validated.
    """

    if not isinstance(config, ExperimentConfig):
        raise TypeError(
            f"config must be ExperimentConfig, got {type(config).__name__}."
        )
    config.validate()

    language_model, language_report = build_language_model(
        config,
        tokenizer_bundle,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
    vision_tower = build_main_vision_tower(config.model.main_vision)
    if config.optimization.checkpoint_main_vision:
        interval = int(
            config.model.main_vision.activation_checkpoint_every_n_layers
        )
        vision_tower.vision_tower.set_activation_checkpointing(interval or 1)
    else:
        vision_tower.vision_tower.set_activation_checkpointing(0)

    mm_projector = build_multimodal_projector(
        projector_config=config.model.projector,
        main_vision_config=config.model.main_vision,
        language_hidden_size=language_model.hidden_size,
    )
    validate_visual_token_contract(
        projector=mm_projector,
        configured_visual_token_count=tokenizer_bundle.metadata.visual_token_count,
    )

    seg_module: SegVol | None = None
    seg_projector: SegmentationPromptProjector | None = None
    if config.model.segmentation.enabled:
        seg_module = build_segvol_module(
            segmentation_config=config.model.segmentation,
            segmentation_vision_config=config.model.seg_vision,
            optimization_config=config.optimization,
        )
        seg_projector = build_segmentation_prompt_projector(
            language_hidden_size=language_model.hidden_size,
            segmentation_config=config.model.segmentation,
        )
        validate_segmentation_prompt_contract(
            projector=seg_projector,
            language_hidden_size=language_model.hidden_size,
            segvol_prompt_embed_dim=seg_module.prompt_encoder.embed_dim,
        )

    model = M3DModel(
        vision_tower=vision_tower,
        mm_projector=mm_projector,
        language_model=language_model,
        objective=build_m3d_loss(config.model.segmentation),
        seg_projector=seg_projector,
        seg_module=seg_module,
    )

    pretrained = (
        _load_pretrained_components(
            model,
            config,
            strict=strict_pretrained,
        )
        if load_pretrained_components
        else M3DPretrainedLoadReport()
    )
    _apply_training_stage_policy(model, config)
    report = M3DBuildReport(
        language=language_report,
        parameters=model.parameter_summary(),
        pretrained=pretrained,
        segmentation_enabled=model.seg_enable,
        visual_token_count=model.mm_projector.proj_out_num,
        language_hidden_size=model.language_hidden_size,
        segmentation_prompt_dim=(
            None
            if model.seg_projector is None
            else model.seg_projector.prompt_embed_dim
        ),
        legacy_component_names=tuple(
            name
            for name in (
                "vision_tower",
                "mm_projector",
                "seg_projector" if model.seg_enable else None,
                "seg_module" if model.seg_enable else None,
            )
            if name is not None
        ),
    )
    return model, report


# ---------------------------------------------------------------------------
# Dependency-free CPU self-test
# ---------------------------------------------------------------------------


class _ToyDecoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> Any:
        del attention_mask
        hidden = self.norm(inputs_embeds + torch.tanh(self.proj(inputs_embeds)))
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class _ToyCausalLM(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int) -> None:
        super().__init__()
        self.model = _ToyDecoder(hidden_size)
        self.embed_tokens = nn.Embedding(vocabulary_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocabulary_size, bias=False)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocabulary_size,
            model_type="phi3",
            use_cache=False,
            base_model_prefix="model",
        )

    def get_decoder(self) -> nn.Module:
        return self.model

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


def _tiny_components() -> tuple[M3DModel, Any]:
    hidden = 32
    image_size = (8, 16, 16)
    patch_size = (4, 8, 8)
    main_vision_config = VisionEncoderConfig(
        image_channels=1,
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
        qkv_bias=True,
        use_cls_token=True,
        attention_backend="math",
        require_flash_sdpa=False,
        activation_checkpoint_every_n_layers=0,
    )
    seg_vision_config = VisionEncoderConfig(
        image_channels=1,
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
        qkv_bias=True,
        use_cls_token=False,
        attention_backend="math",
        require_flash_sdpa=False,
        activation_checkpoint_every_n_layers=0,
    )
    projector_config = ProjectorConfig(
        layer_type="mlp",
        num_layers=2,
        pooling_type="spatial",
        pooling_size=1,
    )
    segmentation_config = SegmentationConfig(
        enabled=True,
        prompt_embed_dim=hidden,
        decoder_depth=2,
        decoder_heads=4,
        dice_loss_weight=1.0,
        bce_loss_weight=1.0,
    )
    optimisation = OptimizationConfig(
        checkpoint_language_model=False,
        checkpoint_main_vision=False,
        checkpoint_seg_vision=False,
        checkpoint_segmentation_decoder=False,
    )
    metadata = SimpleNamespace(
        tokenizer_name_or_path="toy",
        original_vocab_size=64,
        vocabulary_size=64,
        added_token_count=4,
        image_token="<im_patch>",
        image_token_id=3,
        segmentation_token="[SEG]",
        segmentation_token_id=4,
        box_start_token="<bx_start>",
        box_start_token_id=5,
        box_end_token="<bx_end>",
        box_end_token_id=6,
        pad_token_id=0,
        eos_token_id=2,
        visual_token_count=8,
    )

    language_model = M3DLanguageModel(
        _ToyCausalLM(vocabulary_size=64, hidden_size=hidden),
        tokenizer_metadata=metadata,
    )
    main_tower = build_main_vision_tower(main_vision_config)
    projector = build_multimodal_projector(
        projector_config=projector_config,
        main_vision_config=main_vision_config,
        language_hidden_size=hidden,
    )
    seg_module = build_segvol_module(
        segmentation_config=segmentation_config,
        segmentation_vision_config=seg_vision_config,
        optimization_config=optimisation,
    )
    seg_projector = build_segmentation_prompt_projector(
        language_hidden_size=hidden,
        segmentation_config=segmentation_config,
        dropout=0.0,
    )
    model = M3DModel(
        vision_tower=main_tower,
        mm_projector=projector,
        language_model=language_model,
        objective=build_m3d_loss(segmentation_config),
        seg_projector=seg_projector,
        seg_module=seg_module,
    )
    return model, metadata


def _toy_text_tensors(metadata: Any) -> tuple[Tensor, Tensor, Tensor]:
    # Eight image placeholders plus prompt/answer tokens.  Position 11 contains
    # [SEG], so NEXT_TOKEN alignment selects hidden state at position 10.
    input_ids = torch.tensor(
        [
            [
                1,
                *([metadata.image_token_id] * metadata.visual_token_count),
                10,
                11,
                metadata.segmentation_token_id,
                12,
                2,
                0,
                0,
            ],
            [
                1,
                *([metadata.image_token_id] * metadata.visual_token_count),
                13,
                14,
                metadata.segmentation_token_id,
                15,
                2,
                0,
                0,
            ],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids.ne(0)
    labels = torch.full_like(input_ids, -100)
    labels[:, 10:14] = input_ids[:, 10:14]
    return input_ids, attention_mask, labels


def run_self_test() -> Mapping[str, Any]:
    torch.manual_seed(89)
    model, metadata = _tiny_components()
    model.train()

    images = torch.randn(2, 1, 8, 16, 16)
    input_ids, attention_mask, labels = _toy_text_tensors(metadata)

    text_output = model(
        task=TaskName.CAPTION,
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_intermediates=True,
    )
    if text_output.loss is None:
        raise AssertionError("Text task did not produce a loss.")
    text_output.loss.backward()

    if model.main_image_encoder.patch_embedding.position_embeddings.grad is None:
        raise AssertionError("Text loss did not reach the Main 3D ViT.")
    assert model.seg_module is not None
    if any(parameter.grad is not None for parameter in model.seg_module.parameters()):
        raise AssertionError("Text task unexpectedly executed SegVol.")

    model.zero_grad(set_to_none=True)
    all_zero_target = torch.zeros(2, 1, 8, 16, 16)
    seg_output = model(
        task=TaskName.SEGMENTATION,
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        segmentation_targets=all_zero_target,
        return_intermediates=True,
    )
    if seg_output.loss is None or seg_output.segmentation_logits is None:
        raise AssertionError("Segmentation task did not produce loss/logits.")
    seg_output.loss.backward()

    main_gradient = model.main_image_encoder.patch_embedding.position_embeddings.grad
    seg_gradient = model.seg_module.image_encoder.patch_embedding.position_embeddings.grad
    if main_gradient is None or not torch.isfinite(main_gradient).all():
        raise AssertionError("Segmentation loss did not reach the Main 3D ViT.")
    if seg_gradient is None or not torch.isfinite(seg_gradient).all():
        raise AssertionError("Segmentation loss did not reach the SegVol 3D ViT.")

    summary = model.parameter_summary()
    if summary.shared_image_encoder_parameters != 0:
        raise AssertionError("Image encoders share Parameter objects.")
    if summary.shared_image_encoder_storages != 0:
        raise AssertionError("Image encoders share parameter storage.")
    if text_output.segmentation_output is not None:
        raise AssertionError("Text output contains a segmentation result.")
    if tuple(seg_output.segmentation_logits.shape) != (2, 1, 8, 16, 16):
        raise AssertionError("Unexpected final segmentation-logit shape.")

    malformed_routing_detected = False
    try:
        model(
            task=TaskName.CAPTION,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            segmentation_targets=all_zero_target,
        )
    except M3DInputError:
        malformed_routing_detected = True
    if not malformed_routing_detected:
        raise AssertionError("Text task accepted a segmentation target.")

    return {
        "status": "passed",
        "text_loss": float(text_output.loss.detach()),
        "segmentation_loss": float(seg_output.loss.detach()),
        "all_zero_segmentation_target_executed": True,
        "text_task_skipped_segvol": True,
        "main_vision_gradient_is_finite": True,
        "segvol_vision_gradient_is_finite": True,
        "main_visual_token_shape": list(
            text_output.projector_output.projected_tokens.shape
            if text_output.projector_output is not None
            else ()
        ),
        "segmentation_logit_shape": list(seg_output.segmentation_logits.shape),
        "shared_image_encoder_parameters": summary.shared_image_encoder_parameters,
        "shared_image_encoder_storages": summary.shared_image_encoder_storages,
        "malformed_task_routing_detected": malformed_routing_detected,
        "legacy_component_names_present": all(
            hasattr(model, name)
            for name in ("vision_tower", "mm_projector", "seg_projector", "seg_module")
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a dependency-free tiny complete-model forward/backward test.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.self_test:
        raise SystemExit("Pass --self-test to run the complete-model CPU test.")
    print(json.dumps(run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
