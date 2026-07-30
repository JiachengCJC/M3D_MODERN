#!/usr/bin/env python3
"""
Static release audit for the M3D-Modernized reconstruction.

这份脚本主要做七类检查：
1. 所有预期文件是否全部存在
2. 所有 Python 文件能否编译、解析
3. 所有 Shell/PBS 脚本是否有 Bash 语法错误
4. 所有 YAML 配置文件能否正常解析
5. 代码中有没有禁止出现的危险写法
6. M3D 是否保留两个独立的图像编码器设计
7. ASPIRE 2A PBS、joint baseline 路径和 Phi-3 LoRA 配置是否可操作

最后，它会生成类似这样的结果:
{
  "status": "passed",
  "expected_file_count": 68,
  "missing_files": [],
  "python_files_compiled": 39,
  "shell_files_checked": 13,
  "yaml_files_parsed": 4,
  "forbidden_patterns": [],
  "source_sha256": "..."
}
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


EXPECTED_FILES = (
    "scripts/00_setup_environment.sh",
    "requirements.txt",
    "scripts/01_preflight.py",
    "scripts/02_preflight.pbs",
    "src/m3d/config.py",
    "configs/m3d_joint_finetune.yaml",
    "src/m3d/runtime.py",
    "src/m3d/tokenization.py",
    "src/m3d/data/schema.py",
    "src/m3d/data/io.py",
    "src/m3d/data/transforms.py",
    "src/m3d/data/manifest.py",
    "src/m3d/data/prompt_templates.py",
    "src/m3d/data/dataset_catalog.py",
    "src/m3d/data/anatomy_catalog.py",
    "src/m3d/data/datasets.py",
    "src/m3d/data/sampler.py",
    "src/m3d/data/collator.py",
    "src/m3d/data/loader.py",
    "src/m3d/model/attention.py",
    "src/m3d/model/vit3d.py",
    "src/m3d/model/checkpoint.py",
    "src/m3d/model/projector.py",
    "src/m3d/model/segmentation_prompt.py",
    "src/m3d/model/segvol_prompt_encoder.py",
    "src/m3d/model/segvol_transformer.py",
    "src/m3d/model/segvol_mask_decoder.py",
    "src/m3d/model/segvol.py",
    "src/m3d/model/language.py",
    "src/m3d/model/loss.py",
    "src/m3d/model/m3d.py",
    "src/m3d/distributed.py",
    "src/m3d/optim.py",
    "src/m3d/scheduler.py",
    "src/m3d/checkpointing.py",
    "src/m3d/trainer.py",
    "src/m3d/train.py",
    "scripts/03_aspire2a_integration.py",
    "scripts/04_aspire2a_checkpoint_integration.py",
    "scripts/05_aspire2a_integration.pbs",
    "scripts/06_train_aspire2a.pbs",
    "src/m3d/export.py",
    "src/m3d/inference.py",
    "src/m3d/evaluate.py",
    "scripts/07_evaluate_aspire2a.pbs",
    "scripts/08_export_aspire2a.pbs",
    "scripts/09_inference_aspire2a.pbs",
    "src/m3d/model/clip.py",
    "src/m3d/model/clip_loss.py",
    "src/m3d/data/clip_data.py",
    "src/m3d/clip_trainer.py",
    "src/m3d/train_clip.py",
    "configs/m3d_clip_pretrain.yaml",
    "scripts/10_train_clip_aspire2a.pbs",
    "src/m3d/evaluate_clip.py",
    "scripts/11_evaluate_clip_aspire2a.pbs",
    "configs/m3d_projector_pretrain.yaml",
    "configs/m3d_lora_finetune.yaml",
    "configs/m3d_stage1_projector_only.yaml",
    "configs/m3d_stage2_joint_lora.yaml",
    "scripts/12_prepare_manifests_aspire2a.pbs",
    "scripts/13_release_audit.py",
    "scripts/14_local_distributed_e2e.py",
    "pyproject.toml",
    "src/m3d/__init__.py",
    "README.md",
    "TWO_STAGE_ASPIRE2A.md",
    "LICENSE",
)


@dataclass(slots=True)
class AuditResult:
    status: str
    expected_file_count: int
    missing_files: list[str]
    python_files_compiled: int
    shell_files_checked: int
    yaml_files_parsed: int
    forbidden_patterns: list[str]
    source_sha256: str


def _sha256_tree(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def audit(root: Path) -> AuditResult:
    missing = [name for name in EXPECTED_FILES if not (root / name).is_file()]
    python_files = [root / name for name in EXPECTED_FILES if name.endswith(".py") and (root / name).is_file()]
    shell_files = [root / name for name in EXPECTED_FILES if name.endswith((".sh", ".pbs")) and (root / name).is_file()]
    yaml_files = [root / name for name in EXPECTED_FILES if name.endswith((".yaml", ".yml")) and (root / name).is_file()]

    for path in python_files:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec", dont_inherit=True)
        ast.parse(source, filename=str(path))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)
    for path in yaml_files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"YAML root is not a mapping: {path}")

    forbidden: list[str] = []
    source_files = python_files
    fake_mask_pattern = re.compile(r"segmentation_targets?\.sum\(\)\s*[><=!]+")
    unsafe_load_pattern = re.compile(r"torch\.load\([^\n]*weights_only\s*=\s*False")
    # Some files intentionally load trusted, self-created trainer state containing RNG tuples.
    trusted_state_exceptions = {"src/m3d/clip_trainer.py", "src/m3d/evaluate_clip.py"}
    for path in source_files:
        relative = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cuda" and not node.args and not node.keywords:
                forbidden.append(f"hardcoded_cuda:{relative}:{getattr(node, 'lineno', '?')}")
        if fake_mask_pattern.search(source):
            forbidden.append(f"fake_zero_mask_routing:{relative}")
        if relative not in trusted_state_exceptions and unsafe_load_pattern.search(source):
            forbidden.append(f"unsafe_torch_load:{relative}")

    pbs_projects: dict[str, str] = {}
    for path in (candidate for candidate in shell_files if candidate.suffix == ".pbs"):
        relative = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8")
        project_match = re.search(r"^#PBS\s+-P\s+(\S+)\s*$", source, re.MULTILINE)
        queue_match = re.search(r"^#PBS\s+-q\s+(\S+)\s*$", source, re.MULTILINE)
        if project_match is None:
            forbidden.append(f"missing_pbs_project:{relative}")
        else:
            pbs_projects[relative] = project_match.group(1)
        if queue_match is None:
            forbidden.append(f"missing_pbs_queue:{relative}")
        elif queue_match.group(1) not in {"normal", "ai"}:
            forbidden.append(
                f"non_routing_pbs_queue:{relative}:{queue_match.group(1)}"
            )
        if re.search(r"(?<!TORCH_)NCCL_ASYNC_ERROR_HANDLING", source):
            forbidden.append(f"deprecated_nccl_async_variable:{relative}")
        python_module = source.find("module load python/")
        gcc_module = source.find("module load gcc/")
        if python_module >= 0 and gcc_module >= 0 and gcc_module < python_module:
            forbidden.append(f"gcc_loaded_before_python:{relative}")

    project_values = sorted(set(pbs_projects.values()))
    if len(project_values) > 1:
        forbidden.append(f"inconsistent_pbs_projects:{','.join(project_values)}")

    joint_path = root / "configs/m3d_joint_finetune.yaml"
    if joint_path.is_file():
        joint = yaml.safe_load(joint_path.read_text(encoding="utf-8"))
        joint_model = joint.get("model", {}) if isinstance(joint, dict) else {}
        if joint_model.get("language_model_name_or_path") != (
            "../llm_models/phi-3-mini-128k-instruct"
        ):
            forbidden.append("joint_baseline_wrong_language_model")
        projector = joint_model.get("projector", {})
        if projector.get("checkpoint_path") != (
            "../LaMed/output/LaMed-Phi3-4B-pretrain-0000/mm_projector.bin"
        ):
            forbidden.append("joint_baseline_wrong_projector_checkpoint")

    for config_name in (
        "configs/m3d_joint_finetune.yaml",
        "configs/m3d_lora_finetune.yaml",
        "configs/m3d_projector_pretrain.yaml",
        "configs/m3d_stage1_projector_only.yaml",
        "configs/m3d_stage2_joint_lora.yaml",
    ):
        config_path = root / config_name
        if not config_path.is_file():
            continue
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        targets = payload.get("model", {}).get("lora", {}).get("target_modules", [])
        if "qkv_proj" not in targets or any(
            name in targets for name in ("q_proj", "k_proj", "v_proj")
        ):
            forbidden.append(f"invalid_phi3_lora_targets:{config_name}")

    stage1_path = root / "configs/m3d_stage1_projector_only.yaml"
    if stage1_path.is_file():
        stage1 = yaml.safe_load(stage1_path.read_text(encoding="utf-8"))
        model = stage1.get("model", {})
        optimization = stage1.get("optimization", {})
        if (
            optimization.get("stage") != "projector_pretrain"
            or optimization.get("projector_pretrain_train_token_embeddings") is not False
            or model.get("lora", {}).get("enabled") is not False
            or model.get("segmentation", {}).get("enabled") is not False
            or model.get("main_vision", {}).get("freeze") is not True
            or model.get("projector", {}).get("freeze") is not False
        ):
            forbidden.append("invalid_strict_stage1_trainability_contract")

    stage2_path = root / "configs/m3d_stage2_joint_lora.yaml"
    if stage2_path.is_file():
        stage2 = yaml.safe_load(stage2_path.read_text(encoding="utf-8"))
        model = stage2.get("model", {})
        all_non_language_components_unfrozen = (
            model.get("main_vision", {}).get("freeze") is False
            and model.get("seg_vision", {}).get("freeze") is False
            and model.get("projector", {}).get("freeze") is False
            and model.get("segmentation", {}).get("freeze_prompt_encoder") is False
            and model.get("segmentation", {}).get("freeze_mask_decoder") is False
        )
        if (
            stage2.get("optimization", {}).get("stage") != "joint_finetune"
            or model.get("lora", {}).get("enabled") is not True
            or model.get("segmentation", {}).get("enabled") is not True
            or not all_non_language_components_unfrozen
        ):
            forbidden.append("invalid_stage2_joint_lora_trainability_contract")

    model_text = (root / "src/m3d/model/m3d.py").read_text(encoding="utf-8") if (root / "src/m3d/model/m3d.py").is_file() else ""
    for required in ("vision_tower", "seg_module", "assert_independent_encoders"):
        if required not in model_text:
            forbidden.append(f"missing_dual_encoder_contract:{required}")

    all_existing = [root / name for name in EXPECTED_FILES if (root / name).is_file()]
    status = "passed" if not missing and not forbidden else "failed"
    return AuditResult(
        status=status,
        expected_file_count=len(EXPECTED_FILES),
        missing_files=missing,
        python_files_compiled=len(python_files),
        shell_files_checked=len(shell_files),
        yaml_files_parsed=len(yaml_files),
        forbidden_patterns=forbidden,
        source_sha256=_sha256_tree(root, all_existing),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(Path(args.root).expanduser().resolve())
    payload = asdict(result)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if result.status == "passed" else 1)


if __name__ == "__main__":
    main()
