# ASPIRE 2A 两阶段 M3D 完整运行手册

这条路线实现以下精确训练合同：

- Stage 1：只训练 `mm_projector`。Main 3D ViT、完整 Phi-3、token
  embeddings、LM head 全部冻结；不创建 LoRA，不创建 SegVol。
- Stage 2：Phi-3 只通过 LoRA（以及 PEFT 保存的新 M3D token
  embeddings/LM head）训练；Main 3D ViT、`mm_projector`、SegVol image
  encoder、segmentation projector、prompt encoder 和 mask decoder 全部解冻。
- Stage 1 到 Stage 2 不是 resume。Stage 1 先导出
  `multimodal_projector.safetensors`，Stage 2 再把它作为初始化组件加载。
- 正式训练建议使用 DDP。FSDP2 是内存不足时的替代策略，并在正式训练前
  单独做集成测试。

下面所有命令都应在 login node 的项目根目录执行。每个 PBS 作业必须完成并且
`Exit_status = 0` 后，才能执行依赖它的下一步。

## 0. 设置路径

```bash
export M3D_ROOT=/scratch/users/nus/e1129906/M3D_MODERN
cd "$M3D_ROOT"

export STAGE1_CONFIG="$M3D_ROOT/configs/m3d_stage1_projector_only.yaml"
export STAGE2_CONFIG="$M3D_ROOT/configs/m3d_stage2_joint_lora.yaml"

export STAGE1_RUN="$M3D_ROOT/outputs/m3d-stage1-projector-only-a100"
export STAGE1_EXPORT="$M3D_ROOT/outputs/m3d-stage1-projector-export"
export STAGE1_PROJECTOR="$STAGE1_EXPORT/components/multimodal_projector/multimodal_projector.safetensors"

export STAGE2_RUN="$M3D_ROOT/outputs/m3d-stage2-joint-lora-a100"
export STAGE2_EXPORT="$M3D_ROOT/outputs/m3d-stage2-joint-lora-export"
```

## 1. 从零创建环境

```bash
cd "$M3D_ROOT"
bash scripts/00_setup_environment.sh

module load python/3.10.9
source .venv/bin/activate
python -m pip install -e . --no-deps
```

登录节点上的静态和 CPU 自测：

```bash
python scripts/13_release_audit.py \
  --root "$M3D_ROOT" \
  --output "$M3D_ROOT/release_audit.json"

PYTHONPATH="$M3D_ROOT/src" python -m m3d.model.m3d --self-test
PYTHONPATH="$M3D_ROOT/src" python -m m3d.export --self-test
```

三个命令都必须返回 0。模型自测结果必须包含：

```text
"strict_projector_stage_only_trains_mm_projector": true
```

导出自测结果必须包含：

```text
"projector_stage_export_supported": true
"projector_stage_handoff_roundtrip": true
```

## 2. 计算节点硬件/软件 preflight

```bash
PREFLIGHT_JOB=$(qsub -S /bin/bash scripts/02_preflight.pbs)
echo "$PREFLIGHT_JOB"
```

作业结束后检查：

```bash
qstat -xf "$PREFLIGHT_JOB" |
  egrep 'job_state|Exit_status|resources_used.walltime'
```

必须看到 `job_state = F` 和 `Exit_status = 0`。

## 3. Stage 1 manifests（只包含 caption）

`12_prepare_manifests_aspire2a.pbs` 现在默认只生成 task weight 大于 0
的任务，所以 Stage 1 manifest 不会包含 VQA、positioning 或 segmentation。

```bash
S1_MANIFEST_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE1_CONFIG}" \
    scripts/12_prepare_manifests_aspire2a.pbs
)
echo "$S1_MANIFEST_JOB"
```

成功后检查：

```bash
qstat -xf "$S1_MANIFEST_JOB" |
  egrep 'job_state|Exit_status|resources_used.walltime'

for split in train validation test; do
  test -s "$STAGE1_RUN/manifests/${split}.jsonl" ||
    echo "MISSING: $split"
done
```

manifest log 中的 `Counts by task` 应只有 `caption`。

## 4. Stage 1 startup-only

```bash
S1_STARTUP_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE1_CONFIG},M3D_LOCAL_FILES_ONLY=1,M3D_NUM_WORKERS=0,M3D_STARTUP_ONLY=1" \
    scripts/06_train_aspire2a.pbs
)
echo "$S1_STARTUP_JOB"
```

必须 `Exit_status = 0`。这一步会构建真实 Phi-3、ViT、projector、
optimizer 和 DDP，但不读取训练 batch。

## 5. Stage 1 两步 DDP smoke test

这一步使用独立输出目录，真实执行 forward、backward、optimizer step 和
checkpoint；不会污染正式 Stage 1 输出。

```bash
export S1_SMOKE="$M3D_ROOT/outputs/m3d-stage1-smoke-ddp"

S1_SMOKE_MANIFEST_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE1_CONFIG},M3D_MANIFEST_OUTPUT_DIR=${S1_SMOKE}/manifests" \
    scripts/12_prepare_manifests_aspire2a.pbs
)
echo "$S1_SMOKE_MANIFEST_JOB"
```

等 manifest 作业成功后：

```bash
S1_SMOKE_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE1_CONFIG},M3D_OUTPUT_DIR=${S1_SMOKE},M3D_LOCAL_FILES_ONLY=1,M3D_NUM_WORKERS=2,M3D_EPOCHS=1,M3D_STEPS_PER_EPOCH=2,M3D_GRAD_ACCUM=1,M3D_SAVE_EVERY_STEPS=1,M3D_CHECKPOINT_ASYNC=0" \
    scripts/06_train_aspire2a.pbs
)
echo "$S1_SMOKE_JOB"
```

必须 `Exit_status = 0`，并且：

```bash
test -s "$S1_SMOKE/training_result.json" &&
  echo "Stage 1 DDP smoke OK"
test -s "$S1_SMOKE/latest.json" &&
  echo "Stage 1 checkpoint OK"
```

`M3D_NUM_WORKERS=2` 会实际验证多进程 DataLoader，不是退回
`num_workers=0` 绕过问题。

## 6. 正式 Stage 1 训练

```bash
STAGE1_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE1_CONFIG},M3D_LOCAL_FILES_ONLY=1" \
    scripts/06_train_aspire2a.pbs
)
echo "$STAGE1_JOB"
```

成功判据：

```bash
qstat -xf "$STAGE1_JOB" |
  egrep 'job_state|Exit_status|resources_used.walltime'
test -s "$STAGE1_RUN/training_result.json" &&
  test -s "$STAGE1_RUN/latest.json" &&
  echo "Stage 1 training OK"
```

如果作业被调度器中断，只能用同一个 Stage 1 配置和同一种 distributed
strategy 续训：

```bash
qsub -S /bin/bash \
  -v "M3D_CONFIG=${STAGE1_CONFIG},M3D_RESUME_FROM=latest,M3D_LOCAL_FILES_ONLY=1" \
  scripts/06_train_aspire2a.pbs
```

## 7. 导出 Stage 1 projector，作为 Stage 2 初始化

Stage 1 没有 LoRA 和 SegVol，所以必须使用 `M3D_EXPORT_FORMAT=bundle`。

```bash
S1_EXPORT_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE1_CONFIG},M3D_CHECKPOINT=${STAGE1_RUN},M3D_OUTPUT_DIR=${STAGE1_EXPORT},M3D_EXPORT_FORMAT=bundle,M3D_STRATEGY=ddp,M3D_LOCAL_FILES_ONLY=1" \
    scripts/08_export_aspire2a.pbs
)
echo "$S1_EXPORT_JOB"
```

成功判据：

```bash
qstat -xf "$S1_EXPORT_JOB" |
  egrep 'job_state|Exit_status|resources_used.walltime'
test -s "$STAGE1_EXPORT/COMPLETED.json" &&
  test -s "$STAGE1_PROJECTOR" &&
  echo "Stage 1 projector handoff OK"
```

Stage 2 配置默认读取的就是上面的固定路径。如果改变
`STAGE1_EXPORT`，也必须同步修改
`configs/m3d_stage2_joint_lora.yaml` 中的
`model.projector.checkpoint_path`。

## 8. Stage 2 manifests（全部六个任务）

```bash
S2_MANIFEST_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG}" \
    scripts/12_prepare_manifests_aspire2a.pbs
)
echo "$S2_MANIFEST_JOB"
```

成功后应在 `$STAGE2_RUN/manifests` 看到 train、validation、test 三个
JSONL。log 中应列出 caption、VQA、positioning 和 segmentation。

## 9. Stage 2 分布式与 checkpoint 测试矩阵

这些是小规模集成测试，不是正式训练。每一个都必须 `Exit_status = 0`。

DDP runtime：

```bash
S2_DDP_TEST=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_LOCAL_FILES_ONLY=1" \
    scripts/05_aspire2a_integration.pbs
)
echo "$S2_DDP_TEST"
```

FSDP2 runtime：

```bash
S2_FSDP2_TEST=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_INTEGRATION_STRATEGY=fsdp2,M3D_LOCAL_FILES_ONLY=1" \
    scripts/05_aspire2a_integration.pbs
)
echo "$S2_FSDP2_TEST"
```

DDP 同步 checkpoint：

```bash
S2_CKPT_SYNC_TEST=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_INTEGRATION_SUITE=checkpoint,M3D_CHECKPOINT_MODE=sync,M3D_LOCAL_FILES_ONLY=1" \
    scripts/05_aspire2a_integration.pbs
)
echo "$S2_CKPT_SYNC_TEST"
```

DDP 异步 checkpoint：

```bash
S2_CKPT_ASYNC_TEST=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_INTEGRATION_SUITE=checkpoint,M3D_CHECKPOINT_MODE=async,M3D_LOCAL_FILES_ONLY=1" \
    scripts/05_aspire2a_integration.pbs
)
echo "$S2_CKPT_ASYNC_TEST"
```

FSDP2 同步 checkpoint：

```bash
S2_FSDP2_CKPT_TEST=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_INTEGRATION_SUITE=checkpoint,M3D_INTEGRATION_STRATEGY=fsdp2,M3D_CHECKPOINT_MODE=sync,M3D_LOCAL_FILES_ONLY=1" \
    scripts/05_aspire2a_integration.pbs
)
echo "$S2_FSDP2_CKPT_TEST"
```

checkpoint 报告中的以下值都应为 `true`：

- `model_restore_exact_by_rank`
- `optimizer_restore_exact_by_rank`
- `scheduler_restore_exact_by_rank`
- `data_cursor_restore_exact_by_rank`
- `rng_exact_by_rank`
- replay model/optimizer 的 `allclose`

## 10. Stage 2 startup-only 和正式训练

```bash
S2_STARTUP_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_LOCAL_FILES_ONLY=1,M3D_NUM_WORKERS=0,M3D_STARTUP_ONLY=1" \
    scripts/06_train_aspire2a.pbs
)
echo "$S2_STARTUP_JOB"
```

成功后提交正式 DDP：

```bash
STAGE2_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_LOCAL_FILES_ONLY=1" \
    scripts/06_train_aspire2a.pbs
)
echo "$STAGE2_JOB"
```

如果 DDP 显存不足，使用 FSDP2 从头开始，并使用不同输出目录：

```bash
export STAGE2_FSDP2_RUN="$M3D_ROOT/outputs/m3d-stage2-joint-lora-fsdp2-a100"

# 先为新的 output_dir 生成相同任务的 manifests。
qsub -S /bin/bash \
  -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_MANIFEST_OUTPUT_DIR=${STAGE2_FSDP2_RUN}/manifests" \
  scripts/12_prepare_manifests_aspire2a.pbs

# 上一个 manifest 作业成功后再提交。
qsub -S /bin/bash \
  -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_OUTPUT_DIR=${STAGE2_FSDP2_RUN},M3D_STRATEGY=fsdp2,M3D_LOCAL_FILES_ONLY=1" \
  scripts/06_train_aspire2a.pbs
```

不要用 FSDP2 resume DDP checkpoint，也不要用 DDP resume FSDP2 checkpoint。
同策略的中断续训：

```bash
qsub -S /bin/bash \
  -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_RESUME_FROM=latest,M3D_LOCAL_FILES_ONLY=1" \
  scripts/06_train_aspire2a.pbs
```

## 11. 导出最终 Stage 2 模型

`all` 同时产生完整 M3D bundle、PEFT adapter 和 LoRA-merged Phi-3。

```bash
S2_EXPORT_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_CHECKPOINT=${STAGE2_RUN},M3D_OUTPUT_DIR=${STAGE2_EXPORT},M3D_EXPORT_FORMAT=all,M3D_STRATEGY=ddp,M3D_LOCAL_FILES_ONLY=1" \
    scripts/08_export_aspire2a.pbs
)
echo "$S2_EXPORT_JOB"
```

如果正式训练用的是 FSDP2，把 checkpoint 路径改为 FSDP2 输出，并设置
`M3D_STRATEGY=fsdp2`。成功判据：

```bash
test -s "$STAGE2_EXPORT/COMPLETED.json" &&
  test -s "$STAGE2_EXPORT/export_manifest.json" &&
  test -d "$STAGE2_EXPORT/language_adapter" &&
  test -d "$STAGE2_EXPORT/language_merged" &&
  echo "Stage 2 export OK"
```

## 12. 评估

```bash
S2_EVAL_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_CONFIG=${STAGE2_CONFIG},M3D_EXPORT_DIR=${STAGE2_EXPORT},M3D_LOCAL_FILES_ONLY=1,M3D_NUM_WORKERS=4" \
    scripts/07_evaluate_aspire2a.pbs
)
echo "$S2_EVAL_JOB"
```

这里故意使用 `M3D_NUM_WORKERS=4`，会验证多进程 DataLoader。若要诊断与
worker 无关的模型错误，可以临时改为 0，但不能把 0 当成多 worker 已通过。

## 13. 单病例推理

支持 `.npy`、`.nii`、`.nii.gz`。输入必须已经是：

- shape `[C,D,H,W] = [1,32,256,256]`
- floating point
- intensity 范围 `[0,1]`

脚本不会自动 resize 或 normalize。

```bash
export IMAGE=/absolute/path/to/ct.npy
export QUESTION_FILE=/absolute/path/to/question.txt

S2_INFER_JOB=$(
  qsub -S /bin/bash \
    -v "M3D_EXPORT_DIR=${STAGE2_EXPORT},M3D_IMAGE=${IMAGE},M3D_QUESTION_FILE=${QUESTION_FILE},M3D_MODE=auto,M3D_LOCAL_FILES_ONLY=1" \
    scripts/09_inference_aspire2a.pbs
)
echo "$S2_INFER_JOB"
```

文本问题可用 `M3D_MODE=text`；明确要求生成 mask 时可用
`M3D_MODE=segmentation`。

## 14. 通用作业检查

提交成功只代表 PBS 接受了作业，不代表程序成功。对每个 job ID 都执行：

```bash
qstat -xf JOB_ID |
  egrep 'job_state|Exit_status|resources_used.walltime|comment'
```

唯一成功条件是：

```text
job_state = F
Exit_status = 0
```

详细日志分别位于：

- `logs/manifests/<job-id>/manifest.log`
- `logs/integration/<job-id>/integration.log`
- `logs/training/<job-id>/train.log`
- `logs/export/<job-id>/export.log`
- `logs/evaluation/<job-id>/evaluation.log`
- `logs/inference/<job-id>/inference.log`
