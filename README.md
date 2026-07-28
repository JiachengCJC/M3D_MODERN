# M3D-Modernized

A from-scratch, execution-order reconstruction of the original **M3D** project,
modernised for PyTorch 2.6 and NSCC ASPIRE 2A A100 GPUs.

The central architectural constraint is preserved throughout the repository:

> **M3D uses two independent 3D image encoders.**
>
> The Main 3D ViT supplies visual tokens to Phi-3. The SegVol 3D ViT supplies
> spatial features to the segmentation decoder. They share a Python class, but
> never share parameters, gradients, optimizer state, checkpoint storage or
> FSDP groups.

## What is reproduced

- M3D-CLIP image-text contrastive pretraining.
- Main 3D ViT initialisation from M3D-CLIP.
- Phi-3 multimodal projector pretraining.
- Phi-3 LoRA fine-tuning for caption, VQA and positioning.
- Joint language + 3D segmentation fine-tuning.
- SegVol prompt encoder, two-way transformer and mask decoder.
- Caption, VQA, positioning, segmentation and retrieval evaluation.
- Single-case `.npy`, `.nii` and `.nii.gz` inference.
- DDP primary training path and FSDP2 memory fallback.
- Distributed checkpoint save, exact resume and portable export.

## Modernisation decisions

- PyTorch native `scaled_dot_product_attention` with Flash-SDPA required on A100.
- BF16 compute and TF32 matrix multiplication.
- Activation checkpointing independently for Phi-3, Main ViT, SegVol ViT and decoder.
- Task-homogeneous global batches: every rank executes the same conditional graph.
- No fake zero segmentation masks for text tasks.
- An all-zero target remains a valid segmentation sample.
- Dynamic text padding with configured sequence buckets.
- Fused AdamW on the DDP+A100 path.
- FSDP2 bottom-up wrapping as the OOM fallback.
- Exact sampler/scheduler/RNG checkpoint restoration.
- Selective LM-head computation only at supervised answer positions.
- No separate `flash-attn` or DeepSpeed dependency.

## Environment

The pinned ASPIRE 2A software contract is:

```text
Python       3.10.9
PyTorch      2.6.0 + CUDA 11.8
TorchVision  0.21.0 + CUDA 11.8
Transformers 4.52.4
PEFT         0.15.2
MONAI        1.4.0
NumPy        1.26.4
```

Create the environment:

```bash
cd /scratch/users/industry/theiahealth/theiahth/M3D-modernized
bash scripts/00_setup_environment.sh
module load python/3.10.9
source .venv/bin/activate
python -m pip install -e . --no-deps
```

All PBS files currently charge project `58001002`. Before submitting, confirm
that it appears in your ASPIRE 2A project list. If your allocation uses another
project ID, update every `#PBS -P` line consistently or override it with
`qsub -P <project-id>`.

Run the preflight PBS job:

```bash
qsub scripts/02_preflight.pbs
```

`qsub` returns as soon as PBS accepts a job. Do not run a dependent step merely
because the preceding `qsub` command returned successfully: wait for the job to
leave `qstat`, then require its log/report status to be `passed`.

## Data contract

Training volumes must already be preprocessed to:

```text
shape: [1, 32, 256, 256]
dtype: float32-compatible
intensity range: [0, 1]
```

The runtime does not silently resize or normalise an incompatible volume.
Manifest construction validates paths and converts the original M3D metadata
sources into deterministic task records.

```bash
qsub scripts/12_prepare_manifests_aspire2a.pbs
```

The default joint config expects the original data layout under `Data/data`.
Edit only the paths in the YAML when your dataset is stored elsewhere.

## Recommended execution order

### 1. M3D-CLIP pretraining

Configuration:

```text
configs/m3d_clip_pretrain.yaml
```

Submit:

```bash
qsub scripts/10_train_clip_aspire2a.pbs
```

Evaluate retrieval and create the original-style `test_ir.csv` and
`test_tr.csv` files:

```bash
qsub \
  -v M3D_CHECKPOINT=/absolute/path/to/m3d-clip-output \
  scripts/11_evaluate_clip_aspire2a.pbs
```

`M3D_CSV_TOP_K=1000` is safe even when the test set contains fewer than 1000
cases; the evaluator clamps `k` to the actual candidate count.

### 2. Projector pretraining

```text
configs/m3d_projector_pretrain.yaml
```

This stage freezes the full language model and Main ViT, then trains only:

```text
MM Projector
Input token embeddings
LM head/output embeddings
```

Submit through the formal training PBS with a config override:

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/m3d_projector_pretrain.yaml \
  scripts/06_train_aspire2a.pbs
```

### 3. LoRA language fine-tuning

```text
configs/m3d_lora_finetune.yaml
```

This stage disables SegVol, freezes Main ViT and the pretrained projector, and
trains Phi-3 LoRA plus the saved embedding/LM-head modules.

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/m3d_lora_finetune.yaml \
  scripts/06_train_aspire2a.pbs
```

### 4. Joint language + segmentation fine-tuning

```text
configs/m3d_joint_finetune.yaml
```

```bash
qsub scripts/06_train_aspire2a.pbs
```

The two execution graphs are:

```text
Text tasks:
Main 3D ViT → MM Projector → Phi-3 → language loss

Segmentation task:
Main 3D ViT → MM Projector → Phi-3 → language loss
                                  ↓
                             [SEG] prompt
                                  ↓
Independent SegVol 3D ViT → Prompt Encoder → Mask Decoder → Dice + BCE
```

## Correctness-first integration tests

Run these before a long training job:

```bash
# DDP forward/backward: one caption step and one segmentation step
qsub scripts/05_aspire2a_integration.pbs

# DDP exact checkpoint replay
qsub \
  -v M3D_INTEGRATION_SUITE=checkpoint \
  scripts/05_aspire2a_integration.pbs

# FSDP2 forward/backward
qsub \
  -v M3D_INTEGRATION_STRATEGY=fsdp2 \
  scripts/05_aspire2a_integration.pbs

# FSDP2 exact checkpoint replay
qsub \
  -v M3D_INTEGRATION_SUITE=checkpoint,M3D_INTEGRATION_STRATEGY=fsdp2 \
  scripts/05_aspire2a_integration.pbs
```

The tests verify real A100 Flash-SDPA, NCCL, fused AdamW, both conditional
execution graphs, gradient ownership, DCP save/load, sampler replay and rank RNG.

## Resume training

Resume a specific complete checkpoint:

```bash
qsub \
  -v M3D_RESUME_FROM=/absolute/path/checkpoint-step-00001000 \
  scripts/06_train_aspire2a.pbs
```

Resume the latest complete checkpoint under the configured output directory:

```bash
qsub -v M3D_RESUME_FROM=latest scripts/06_train_aspire2a.pbs
```

Only checkpoint directories containing `COMPLETED.json` are eligible.
Checkpoint saves occur at gradient-accumulation boundaries.

## DDP and FSDP2

DDP is the default and fastest path when one full model replica fits per A100:

```bash
qsub scripts/06_train_aspire2a.pbs
```

Use FSDP2 when DDP still runs out of memory:

```bash
qsub -v M3D_STRATEGY=fsdp2 scripts/06_train_aspire2a.pbs
```

FSDP2 separately shards Main ViT, Phi-3 layers, SegVol ViT and the segmentation
decoder. A text task does not all-gather SegVol ViT layers.

## Export

Export the latest training checkpoint:

```bash
qsub \
  -v M3D_CHECKPOINT=/absolute/path/to/training-output \
  scripts/08_export_aspire2a.pbs
```

Available formats:

```text
bundle   complete M3D safetensors + components + tokenizer
adapter  bundle + PEFT LoRA adapter
merged   bundle + LoRA-merged Phi-3
all      all formats
```

The portable export retains both independent image encoders.

## Single-case inference

Create a question file and submit:

```bash
qsub \
  -v M3D_EXPORT_DIR=/absolute/path/to/export,\
M3D_IMAGE=/absolute/path/to/ct.nii.gz,\
M3D_QUESTION_FILE=/absolute/path/to/question.txt,\
M3D_MODE=auto \
  scripts/09_inference_aspire2a.pbs
```

`auto` runs SegVol only when the generated answer contains `[SEG]`.
`segmentation` requires `[SEG]`; `text` always skips SegVol.

## Evaluation

Evaluate a portable export:

```bash
qsub \
  -v M3D_EXPORT_DIR=/absolute/path/to/export \
  scripts/07_evaluate_aspire2a.pbs
```

Evaluate only segmentation:

```bash
qsub \
  -v M3D_EXPORT_DIR=/absolute/path/to/export,M3D_TASKS=segmentation \
  scripts/07_evaluate_aspire2a.pbs
```

Metrics include caption/VQA text metrics, 3D box IoU, hard and soft segmentation
metrics, `[SEG]` trigger rate, and optional M3D-CLIP retrieval Recall@K.

## Release audit

Run after editing any core file:

```bash
python scripts/13_release_audit.py --root . --output release_audit.json
```

The audit checks:

- all 65 planned files exist;
- every Python file compiles;
- every PBS/shell file passes `bash -n`;
- every YAML file parses;
- dual-encoder contracts remain present;
- prohibited hard-coded `.cuda()` and mask-sum task routing are absent.
- PBS project IDs are consistent and only submit-capable routing queues are used;
- the joint release-component paths and Phi-3 fused LoRA targets are correct.

## Local distributed end-to-end validation

Before submitting the GPU jobs, a two-process CPU system can exercise the real
training entry point, DDP, FSDP2, LoRA, both independent 3D encoders, SegVol,
distributed-checkpoint save/load/replay, full training, and
`resume_from=latest` with a generated tiny Phi-3 model and synthetic volumes:

```bash
python scripts/14_local_distributed_e2e.py --require-pinned
```

The command writes a consolidated
`local_distributed_e2e_report.json` and individual command logs below
`outputs/local-distributed-e2e/<timestamp>/`. It intentionally uses CPU BF16
and Gloo. Because PyTorch 2.6 CPU composable FSDP cannot safely reshard between
forward and backward, only this explicitly gated local mode delays resharding
until backward; the CUDA path uses the configured post-forward resharding.
CUDA 11.8, NCCL, A100 Flash-SDPA, PBS, and cluster-fabric validation remain
mandatory in `02_preflight.pbs` and `05_aspire2a_integration.pbs`.

## Package installation

After the environment is created (this is already included in the setup
sequence above):

```bash
pip install -e . --no-deps
```

Console commands:

```text
m3d-train
m3d-evaluate
m3d-export
m3d-inference
m3d-clip-train
m3d-clip-evaluate
```

## Repository layout

```text
configs/        CLIP, projector, LoRA and joint-stage YAML files
scripts/        environment, PBS, integration and audit commands
src/m3d/data/   deterministic manifests, datasets, samplers and collators
src/m3d/model/  two ViTs, Phi-3, projector, SegVol, CLIP and losses
src/m3d/        runtime, distributed wrapping, trainer, evaluation and export
```

## Licence and attribution

This reconstruction retains the original repository's MIT licence and copyright
notice. Model weights and third-party datasets may have separate licences and
access terms; review them before redistribution or clinical/commercial use.

This software is research code and is not a medical device.
