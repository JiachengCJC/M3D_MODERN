# M3D-Modernized：中文完整使用手册

> 本 README 按照真实执行顺序编写，目标是让第一次接触本项目的人可以从零开始，在 NSCC ASPIRE 2A 上完成环境建立、数据检查、Manifest 生成、M3D-CLIP 训练、Phi-3 多模态训练、分布式集成测试、正式训练、断点续训、导出、单病例推理与评估。

---

## 目录

1. [项目定位](#1-项目定位)
2. [必须先理解的模型结构](#2-必须先理解的模型结构)
3. [本项目实现了哪些训练阶段](#3-本项目实现了哪些训练阶段)
4. [现代化改动](#4-现代化改动)
5. [推荐执行路线](#5-推荐执行路线)
6. [项目目录结构](#6-项目目录结构)
7. [第 0 步：取得代码并进入项目目录](#7-第-0-步取得代码并进入项目目录)
8. [第 1 步：准备原始数据和预训练权重](#8-第-1-步准备原始数据和预训练权重)
9. [第 2 步：检查 3D 数据格式](#9-第-2-步检查-3d-数据格式)
10. [第 3 步：建立 ASPIRE 2A Python 环境](#10-第-3-步建立-aspire-2a-python-环境)
11. [第 4 步：安装本项目为 Python package](#11-第-4-步安装本项目为-python-package)
12. [第 5 步：运行 GPU 与分布式 Preflight](#12-第-5-步运行-gpu-与分布式-preflight)
13. [第 6 步：理解和修改 YAML 配置](#13-第-6-步理解和修改-yaml-配置)
14. [第 7 步：生成 Manifest](#14-第-7-步生成-manifest)
15. [第 8 步：运行 Release Audit](#15-第-8-步运行-release-audit)
16. [第 9 步：M3D-CLIP 预训练与检索评估](#16-第-9-步m3d-clip-预训练与检索评估)
17. [第 10 步：Projector Pretraining](#17-第-10-步projector-pretraining)
18. [第 11 步：LoRA Fine-tuning](#18-第-11-步lora-fine-tuning)
19. [第 12 步：Joint Language + Segmentation Fine-tuning](#19-第-12-步joint-language--segmentation-fine-tuning)
20. [第 13 步：正式训练前必须运行的四个集成测试](#20-第-13-步正式训练前必须运行的四个集成测试)
21. [第 14 步：提交正式训练](#21-第-14-步提交正式训练)
22. [第 15 步：监控 PBS 作业和日志](#22-第-15-步监控-pbs-作业和日志)
23. [第 16 步：断点续训](#23-第-16-步断点续训)
24. [第 17 步：DDP 与 FSDP2 的选择](#24-第-17-步ddp-与-fsdp2-的选择)
25. [第 18 步：导出训练结果](#25-第-18-步导出训练结果)
26. [第 19 步：单病例推理](#26-第-19-步单病例推理)
27. [第 20 步：完整模型评估](#27-第-20-步完整模型评估)
28. [输出文件说明](#28-输出文件说明)
29. [PBS 环境变量速查表](#29-pbs-环境变量速查表)
30. [常见错误与排查](#30-常见错误与排查)
31. [从代码角度理解完整执行顺序](#31-从代码角度理解完整执行顺序)
32. [所有核心文件的职责](#32-所有核心文件的职责)
33. [当前已知边界](#33-当前已知边界)
34. [许可证与医学用途声明](#34-许可证与医学用途声明)

---

# 1. 项目定位

`M3D-Modernized` 是对原始 M3D 项目的从零重构版本，目标是在不改变核心模型语义的前提下，把训练方式升级到：

```text
PyTorch 2.6
CUDA 11.8
A100 BF16
PyTorch SDPA / Flash-SDPA
DDP
FSDP2
Distributed Checkpoint
动态文本 padding
任务均衡且任务同质的 batch
```

本项目覆盖：

```text
M3D-CLIP 图文对比学习
Main 3D ViT 初始化
Phi-3 多模态 Projector 预训练
Phi-3 LoRA 微调
Caption / VQA / Positioning
语言 + 3D segmentation 联合微调
SegVol Prompt Encoder
SegVol Two-Way Transformer
SegVol Mask Decoder
分布式断点保存与精确恢复
模型导出
单病例推理
Caption / VQA / Positioning / Segmentation / Retrieval 评估
```

本项目以 NSCC ASPIRE 2A 为主要运行环境，但 Python 代码本身不依赖 PBS；只要其他服务器提供兼容的 PyTorch、CUDA 和 A100，也可以直接使用 `torchrun` 启动。

---

# 2. 必须先理解的模型结构

## 2.1 M3D 必须保留两个独立的 3D image encoder

这是整个项目最重要的结构约束：

> **Main 3D ViT 和 SegVol 3D ViT 是两套独立参数。**

它们只共用同一个 Python 类定义，不共用：

```text
Parameter object
底层 tensor storage
Gradient
Optimizer state
Checkpoint namespace
FSDP group
```

### Encoder 1：Main 3D ViT

用途：把 CT 转成提供给 Phi-3 的视觉 token。

```text
CT [B,1,32,256,256]
    ↓
Main 3D ViT
    ↓
2048 patch tokens + 1 CLS token
    ↓
Spatial Pooling Projector
    ↓
256 visual tokens
    ↓
Phi-3
```

### Encoder 2：SegVol 3D ViT

用途：为 3D segmentation decoder 提供稠密空间特征。

```text
CT [B,1,32,256,256]
    ↓
独立 SegVol 3D ViT
    ↓
[B,768,8,16,16]
    ↓
Prompt Encoder + Mask Decoder
    ↓
3D mask logits
```

## 2.2 普通文本任务的执行图

以下任务属于文本图：

```text
caption
vqa_closed
vqa_open
vqa_yes_no
positioning
```

执行：

```text
Main 3D ViT
    ↓
MM Projector
    ↓
Phi-3
    ↓
Language Loss
```

这一类 batch 不运行：

```text
SegVol 3D ViT
SegVol Prompt Encoder
SegVol Mask Decoder
```

也不会创建假的全零 segmentation target。

## 2.3 Segmentation 任务的执行图

```text
Main 3D ViT
    ↓
MM Projector
    ↓
Phi-3
    ├── Language Loss
    └── [SEG] 对应 hidden state
              ↓
      Segmentation Projector
              ↓
      SegVol Prompt Encoder
              ↓
独立 SegVol 3D ViT → Mask Decoder
              ↓
         Dice + BCE Loss
```

任务类型由：

```python
task == "segmentation"
```

决定，而不是由：

```python
mask.sum() > 0
```

决定。

因此，全零 mask 仍然是合法 segmentation 样本，完整 segmentation graph 仍然会执行。

---

# 3. 本项目实现了哪些训练阶段

本仓库提供四类配置：

| 阶段 | 配置文件 | 主要训练内容 |
|---|---|---|
| M3D-CLIP | `configs/m3d_clip_pretrain.yaml` | 3D ViT + BERT + projection heads |
| Projector pretraining | `configs/m3d_projector_pretrain.yaml` | MM Projector、token embeddings、LM head |
| LoRA fine-tuning | `configs/m3d_lora_finetune.yaml` | Phi-3 LoRA、token embeddings、LM head |
| Joint fine-tuning | `configs/m3d_joint_finetune.yaml` | Main ViT、Phi-3 LoRA、Projector、SegVol 全路径 |

## 3.1 两种实际工作方式

### 路线 A：使用原 M3D 发布权重，直接运行现代化 joint training

这是最容易真正跑通的路线。

配置默认读取：

```text
M3D-CLIP pretrained_ViT.bin
原 M3D mm_projector.bin
原 SegVol pytorch_model.bin
Hugging Face Phi-3 base model
```

然后使用现代化训练框架继续 joint fine-tuning。

### 路线 B：重新训练每个阶段

仓库允许分别训练：

```text
CLIP
Projector
LoRA
Joint
```

但必须注意：

> 当前版本不会自动把一个阶段的 Distributed Checkpoint 转换成下一个阶段所需的原始组件文件格式。

例如：

```text
CLIP trainer 保存的是完整 CLIP training_state.pt
MLLM Main ViT loader 默认期望 pretrained_ViT.bin 兼容格式
```

同样：

```text
Projector pretraining 结束后的 Distributed Checkpoint
不能直接作为下一个 stage 的 exact resume
```

`resume_from` 仅用于同一训练阶段、同一模型布局、同一数据语义的精确续训，不应用来跨阶段转换。

因此，当前默认配置的可运行基线仍然使用原始发布组件权重。重新训练阶段用于独立实验和验证；跨阶段权重转换需要额外提取兼容组件并修改 YAML 路径。

---

# 4. 现代化改动

## 4.1 Attention

两套 3D ViT 和 SegVol Two-Way Transformer 使用：

```python
torch.nn.functional.scaled_dot_product_attention
```

A100 正式训练配置要求 Flash backend 可用；不允许静默退回普通 math attention。

Phi-3 与 BERT 也使用 Hugging Face 原生 SDPA 路径。

本项目不依赖：

```text
flash-attn package
DeepSpeed
```

## 4.2 数值精度

```text
主计算：BF16
矩阵乘法：允许 TF32
Dice/BCE 大规模 reduction：FP32
梯度归约：FSDP2 可使用 FP32
```

不使用 FP16 GradScaler。

## 4.3 Activation Checkpointing

可分别控制：

```text
Phi-3
Main 3D ViT
SegVol 3D ViT
SegVol decoder
```

默认两个 ViT 和 Phi-3 开启，segmentation decoder 默认关闭。

## 4.4 Task-homogeneous distributed batch

同一个 global microbatch 中，所有 rank 执行相同任务。

正确：

```text
rank 0：segmentation
rank 1：segmentation
```

错误：

```text
rank 0：caption
rank 1：segmentation
```

这样可以避免 DDP/FSDP2 条件执行图不同导致 deadlock 或梯度错误。

## 4.5 动态文本 padding

配置 buckets：

```text
128
256
384
512
```

一个 batch 根据最长有效序列选择最小可容纳 bucket，而不是所有样本都 padding 到 512。

## 4.6 Selective LM head

训练时只对 `labels != -100` 的答案位置计算完整 vocabulary logits。

不会为以下无监督位置生成无用 logits：

```text
256 个 image placeholders
问题文本
padding tokens
```

## 4.7 Optimizer

DDP + A100 默认要求：

```text
fused AdamW
```

FSDP2 使用兼容 DTensor 的保守 AdamW 路径。

## 4.8 Checkpoint

Joint M3D 使用 PyTorch Distributed Checkpoint，保存：

```text
模型
optimizer
scheduler
sampler cursor
manifest fingerprint
RNG
训练进度
```

只有带有 `COMPLETED.json` 的 checkpoint 才是可恢复 checkpoint。

---

# 5. 推荐执行路线

第一次运行时，建议严格按以下顺序：

```text
0. 解压项目并进入根目录
1. 放置 Data、原 M3D 权重和 SegVol 权重
2. 检查图像与 mask shape/range
3. 建立 Python 环境
4. pip install -e . --no-deps
5. 提交 preflight PBS
6. 修改 YAML 中的数据和权重路径
7. 生成对应 stage 的 manifests
8. 运行 release audit
9. 运行 DDP runtime integration
10. 运行 DDP checkpoint integration
11. 必要时运行 FSDP2 两个 integration
12. 提交 startup-only
13. 提交正式训练
14. 监控日志和 GPU
15. 从完整 checkpoint 断点续训
16. 导出 portable model
17. 单病例推理
18. 完整评估
```

对于只想尽快验证 joint M3D 的用户：

```text
环境 → 数据 → published checkpoints → joint manifests
→ DDP integration → startup-only → joint training
```

M3D-CLIP、Projector、LoRA 的独立重训不是运行 joint baseline 的强制前置步骤。

---

# 6. 项目目录结构

推荐根目录：

```text
/scratch/users/industry/theiahealth/theiahth/M3D-modernized
```

完整结构示例：

```text
M3D-modernized/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── m3d_clip_pretrain.yaml
│   ├── m3d_projector_pretrain.yaml
│   ├── m3d_lora_finetune.yaml
│   └── m3d_joint_finetune.yaml
│
├── scripts/
│   ├── 00_setup_environment.sh
│   ├── 01_preflight.py
│   ├── 02_preflight.pbs
│   ├── 03_aspire2a_integration.py
│   ├── 04_aspire2a_checkpoint_integration.py
│   ├── 05_aspire2a_integration.pbs
│   ├── 06_train_aspire2a.pbs
│   ├── 07_evaluate_aspire2a.pbs
│   ├── 08_export_aspire2a.pbs
│   ├── 09_inference_aspire2a.pbs
│   ├── 10_train_clip_aspire2a.pbs
│   ├── 11_evaluate_clip_aspire2a.pbs
│   ├── 12_prepare_manifests_aspire2a.pbs
│   └── 13_release_audit.py
│
├── src/m3d/
│   ├── data/
│   ├── model/
│   ├── train.py
│   ├── trainer.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── export.py
│   └── ...
│
├── Data/
│   └── data/
│       ├── M3D_Cap_npy/
│       ├── M3D-VQA/
│       ├── M3D_Seg_npy/
│       └── M3D_RefSeg_npy/
│
├── LaMed/
│   ├── pretrained_model/
│   │   ├── M3D-CLIP/pretrained_ViT.bin
│   │   └── SegVol/pytorch_model.bin
│   └── output/
│       └── LaMed-Phi3-4B-pretrain-0000/mm_projector.bin
│
├── outputs/
├── logs/
└── .venv/
```

`Data/` 和 `LaMed/` 可以放在其他位置，但必须修改 YAML 中的路径。

---

# 7. 第 0 步：取得代码并进入项目目录

## 7.1 解压完整项目

```bash
cd /scratch/users/industry/theiahealth/theiahth
unzip M3D-modernized-complete.zip
cd M3D-modernized
```

确认当前目录：

```bash
pwd
```

预期：

```text
/scratch/users/industry/theiahealth/theiahth/M3D-modernized
```

确认核心文件存在：

```bash
ls configs
ls scripts
ls src/m3d
```

## 7.2 每次提交 PBS 前必须从项目根目录执行

PBS 文件使用：

```bash
PBS_O_WORKDIR
```

作为项目根目录。

因此推荐：

```bash
cd /scratch/users/industry/theiahealth/theiahth/M3D-modernized
qsub scripts/...
```

不要在其他目录提交后假设 PBS 会自动找到项目。

---

# 8. 第 1 步：准备原始数据和预训练权重

## 8.1 默认 joint 配置需要的权重

### Main 3D ViT

```text
LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin
```

YAML：

```yaml
model:
  main_vision:
    checkpoint_path: ../LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin
```

### MM Projector

```text
LaMed/output/LaMed-Phi3-4B-pretrain-0000/mm_projector.bin
```

YAML：

```yaml
model:
  projector:
    checkpoint_path: ../LaMed/output/LaMed-Phi3-4B-pretrain-0000/mm_projector.bin
```

### SegVol

```text
LaMed/pretrained_model/SegVol/pytorch_model.bin
```

YAML：

```yaml
model:
  segmentation:
    checkpoint_path: ../LaMed/pretrained_model/SegVol/pytorch_model.bin
```

### Phi-3

默认：

```yaml
language_model_name_or_path: microsoft/Phi-3-mini-4k-instruct
```

模型可以：

```text
从 Hugging Face 下载
从共享 Hugging Face cache 读取
从本地目录读取
```

离线环境建议提前把模型下载到：

```text
<cache-root>/huggingface
```

## 8.2 为什么 YAML 中是 `../Data` 和 `../LaMed`

所有相对路径均相对于 YAML 文件所在目录解析。

例如 YAML 位于：

```text
M3D-modernized/configs/m3d_joint_finetune.yaml
```

所以：

```text
../Data/data
```

解析为：

```text
M3D-modernized/Data/data
```

不是相对于你执行命令时的 shell 当前目录。

## 8.3 快速检查权重路径

```bash
ls -lh LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin
ls -lh LaMed/output/LaMed-Phi3-4B-pretrain-0000/mm_projector.bin
ls -lh LaMed/pretrained_model/SegVol/pytorch_model.bin
```

任何一个文件不存在，joint model startup 会失败。

## 8.4 默认数据结构

```text
Data/data/
├── M3D_Cap_npy/
│   ├── M3D_Cap.json
│   └── ... image/text files ...
├── M3D-VQA/
│   ├── M3D_VQA_train.csv
│   ├── M3D_VQA_val.csv
│   ├── M3D_VQA_test.csv
│   └── M3D_VQA_yn_train.csv
├── M3D_Seg_npy/
│   ├── 0001/0001.json
│   ├── 0002/0002.json
│   └── ...
└── M3D_RefSeg_npy/
    ├── M3D_RefSeg.csv
    └── M3D_RefSeg_test.csv
```

## 8.5 Caption JSON 需要的字段

每条记录至少需要：

```text
image
text
```

并按 split 分组，例如：

```json
{
  "train": [
    {
      "image": "M3D_Cap_npy/Case001/ct.npy",
      "text": "M3D_Cap_npy/Case001/report.txt"
    }
  ],
  "validation": [],
  "test": []
}
```

## 8.6 VQA CSV 需要的字段

Open VQA 至少需要：

```text
Image Path
Question
Answer
```

Closed VQA 额外需要：

```text
Choice A
Choice B
Choice C
Choice D
Answer Choice
```

可选：

```text
Question Type
```

## 8.7 RefSeg CSV 需要的字段

```text
Image
Mask
Mask_ID
Question
Answer
```

`Mask_ID` 用于从多类别 label volume 中选择一个类别并转换成二值 target。

## 8.8 M3D segmentation metadata

程序搜索：

```text
M3D_Seg_npy/*/[0-9][0-9][0-9][0-9].json
```

并要求 JSON 文件名与父目录 tag 一致，例如：

```text
M3D_Seg_npy/0001/0001.json
```

JSON 中使用：

```text
train
或
test
```

每条记录至少有：

```text
image
label 或 mask
```

---

# 9. 第 2 步：检查 3D 数据格式

## 9.1 图像输入契约

正式训练要求：

```text
逻辑 shape：[C,D,H,W]
C = 1
D = 32
H = 256
W = 256
```

允许存储为：

```text
[D,H,W]
[1,D,H,W]
[D,H,W,1]
```

但转换后必须成为：

```text
[1,32,256,256]
```

## 9.2 图像数值范围

默认要求：

```text
float32-compatible
所有数值有限
min >= 0
max <= 1
```

训练 loader 不会静默执行：

```text
min-max normalization
windowing
resize
crop
spacing resampling
```

这些必须在数据准备阶段完成。

## 9.3 Mask 契约

Mask 可以是：

```text
单通道二值 mask
多类别 label volume
多通道 mask
```

Manifest 中的 label ID 或 channel selection 会把它转换成：

```text
[1,32,256,256]
值为 0 或 1
```

全零 target 合法。

## 9.4 NPY 快速检查

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

path = Path("/absolute/path/to/ct.npy")
arr = np.load(path, mmap_mode="r", allow_pickle=False)
print("path:", path)
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("min:", float(np.min(arr)))
print("max:", float(np.max(arr)))
print("finite:", bool(np.isfinite(arr).all()))
PY
```

预期：

```text
shape: (32,256,256) 或 (1,32,256,256)
min: 0.0 左右
max: 1.0 左右
finite: True
```

## 9.5 检查 image 与 mask 是否同尺寸

```bash
python - <<'PY'
import numpy as np

image = np.load("/path/to/image.npy", mmap_mode="r")
mask = np.load("/path/to/mask.npy", mmap_mode="r")

image_spatial = image.shape[-3:]
mask_spatial = mask.shape[-3:]

print("image:", image.shape)
print("mask:", mask.shape)
print("same spatial shape:", image_spatial == mask_spatial)
print("mask unique sample:", np.unique(mask)[:30])
PY
```

## 9.6 NIfTI 输入

Nibabel 读取 NIfTI 时原始 array order 通常为：

```text
[X,Y,Z]
```

项目会 canonicalise，然后转换为模型内部：

```text
[C,D,H,W]
```

即：

```text
D = Z
H = Y
W = X
```

不要根据 ITK-SNAP 显示的 axial/coronal/sagittal 平面直接推断 numpy 轴顺序。

## 9.7 检查 NIfTI geometry

```bash
python - <<'PY'
import nibabel as nib
import numpy as np

path = "/path/to/ct.nii.gz"
img = nib.load(path)
canonical = nib.as_closest_canonical(img)

print("source shape:", img.shape)
print("canonical shape:", canonical.shape)
print("spacing:", canonical.header.get_zooms()[:3])
print("orientation:", nib.aff2axcodes(canonical.affine))
print("dtype:", canonical.get_data_dtype())
arr = np.asanyarray(canonical.dataobj)
print("min/max:", float(arr.min()), float(arr.max()))
PY
```

## 9.8 输入已经是目标 shape 时 Resize 会怎样

本训练框架不会在 loader 内调用 Resize。

如果你自己的预处理流程中调用 MONAI：

```python
Resize(spatial_size=[32,256,256])
```

而输入已经是：

```text
[1,32,256,256]
```

空间 shape 不变，但仍可能执行插值计算。对 segmentation mask 必须使用 nearest interpolation，不能使用 trilinear。

---

# 10. 第 3 步：建立 ASPIRE 2A Python 环境

## 10.1 在 login node 执行

```bash
cd /scratch/users/industry/theiahealth/theiahth/M3D-modernized
bash scripts/00_setup_environment.sh
```

脚本会：

```text
module reset
加载 PrgEnv-gnu/8.3.3
加载 gcc/11.4.0-nscc
加载 python/3.10.9
加载 cuda/11.8.0
加载 cmake/3.31.3
加载 ninja/1.11.1
建立 .venv
安装 PyTorch 2.6.0 cu118
安装 requirements.txt
运行 pip check
记录环境 lock
```

## 10.2 默认环境位置

```text
M3D-modernized/.venv
```

## 10.3 使用自定义环境目录

```bash
M3D_VENV_DIR=/scratch/users/industry/theiahealth/theiahth/envs/m3d-modern \
M3D_CACHE_ROOT=/scratch/users/industry/theiahealth/theiahth/.cache/m3d-modern \
bash scripts/00_setup_environment.sh
```

后续 PBS 必须传同一个 `M3D_VENV_DIR`。

## 10.4 每次手工运行 Python 前激活环境

```bash
module reset
module load PrgEnv-gnu/8.3.3
module load gcc/11.4.0-nscc
module load python/3.10.9
module load cuda/11.8.0
module load cmake/3.31.3
module load ninja/1.11.1

source .venv/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

注意：

> 你在 login shell 中执行过 `module load cuda/11.8.0`，并不代表 PBS compute job 会自动继承可靠的 module stack。

所以每个 PBS 都会重新 `module reset` 并加载固定版本。

## 10.5 不要额外安装 flash-attn 或 DeepSpeed

本项目使用：

```text
PyTorch native SDPA
DDP
FSDP2
```

不要另外引入不同 CUDA/NCCL/cuDNN stack，避免符号冲突。

---

# 11. 第 4 步：安装本项目为 Python package

环境建立后：

```bash
source .venv/bin/activate
pip install -e . --no-deps
```

`--no-deps` 的原因是依赖已经由固定版本的 `requirements.txt` 安装，避免 editable install 再次修改版本。

安装后可使用：

```text
m3d-train
m3d-evaluate
m3d-export
m3d-inference
m3d-clip-train
m3d-clip-evaluate
```

也可以始终使用：

```bash
python -m m3d.train
python -m m3d.evaluate
python -m m3d.export
python -m m3d.inference
```

验证安装：

```bash
python - <<'PY'
import m3d
print(m3d.__version__)
PY
```

---

# 12. 第 5 步：运行 GPU 与分布式 Preflight

## 12.1 提交

```bash
cd /scratch/users/industry/theiahealth/theiahth/M3D-modernized
qsub scripts/02_preflight.pbs
```

## 12.2 查看作业状态

```bash
qstat -u "$USER"
```

常见状态：

```text
Q：排队
R：运行
E：退出处理中
C：完成
```

## 12.3 查看实时日志

先找到 job ID，例如：

```text
14990000.pbs101
```

然后：

```bash
tail -f logs/preflight/14990000.pbs101/preflight.log
```

## 12.4 Preflight 检查内容

```text
两张 GPU 是否可见
每个 rank 是否绑定正确 GPU
GPU 是否为 A100
BF16 是否支持
PyTorch/CUDA 版本
NCCL process group
跨 rank collective
Flash-SDPA
CUDA forward/backward
```

## 12.5 成功输出

```text
logs/preflight/<JOBID>/
├── preflight.log
├── gpu.csv
└── preflight_report.json
```

只有 preflight 通过后再加载 Phi-3 和完整 M3D。

---

# 13. 第 6 步：理解和修改 YAML 配置

## 13.1 修改前先复制配置

不建议直接反复修改基准 YAML。

```bash
cp configs/m3d_joint_finetune.yaml configs/my_joint_run.yaml
```

之后使用：

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/my_joint_run.yaml \
  scripts/06_train_aspire2a.pbs
```

## 13.2 最先修改的路径

```yaml
data:
  paths:
    data_root: ../Data/data
```

以及：

```yaml
model:
  main_vision:
    checkpoint_path: ...
  projector:
    checkpoint_path: ...
  segmentation:
    checkpoint_path: ...
```

## 13.3 Main ViT 配置

```yaml
image_size: [32,256,256]
patch_size: [4,16,16]
hidden_size: 768
depth: 12
num_heads: 12
mlp_dim: 3072
use_cls_token: true
```

Patch grid：

```text
32/4 = 8
256/16 = 16
256/16 = 16

8×16×16 = 2048 patch tokens
```

加 CLS 后：

```text
2049 tokens
```

## 13.4 SegVol ViT 配置

几何结构相同，但：

```yaml
use_cls_token: false
```

输出直接恢复为：

```text
[B,768,8,16,16]
```

## 13.5 Projector 配置

```yaml
pooling_type: spatial
pooling_size: 2
num_layers: 2
```

Token 数：

```text
2048
→ 3D 2×2×2 average pooling
→ 256
```

Tokenizer 中必须恰好有 256 个 `<im_patch>`。

## 13.6 Task weights

Joint 默认：

```yaml
task_weights:
  caption: 1.0
  vqa_closed: 1.0
  vqa_open: 1.0
  vqa_yes_no: 0.5
  positioning: 1.0
  segmentation: 2.0
```

Sampler 使用的相对 score 大致为：

```text
weight × dataset_size^temperature_alpha
```

然后转换成每个 epoch 的确定性 task quota。

`segmentation: 2.0` 表示 segmentation 被提高采样权重，不表示每一步同时包含两种任务。

## 13.7 有效 global batch size

公式：

```text
per_device_batch_size
× world_size
× gradient_accumulation_steps
```

Joint 默认：

```text
1 × 2 × 4 = 8
```

Projector/LoRA 默认：

```text
1 × 2 × 8 = 16
```

CLIP 默认：

```text
8 × 2 × 1 = 16
```

增加 gradient accumulation 不会降低单个 sample 的 activation memory，但会增大有效 batch。

## 13.8 Dynamic padding

```yaml
dynamic_padding: true
pad_to_multiple_of: 8
sequence_length_buckets: [128,256,384,512]
```

如果 batch 最长文本为 117 token，则选择 128。

## 13.9 DDP 配置

```yaml
strategy: ddp
find_unused_parameters: true
static_graph: false
```

Joint training 必须允许 unused parameters，因为 text step 会跳过 SegVol。

## 13.10 FSDP2 配置

```yaml
strategy: fsdp2
reshard_after_forward: true
cpu_offload: false
mixed_precision: bf16
```

只有 DDP OOM 时再切换。

## 13.11 Checkpoint 配置

```yaml
save_every_steps: 1000
keep_last_n: 2
asynchronous: true
```

保存频率单位是：

```text
optimizer update
```

不是 microbatch。

## 13.12 Output directory

每个训练阶段必须使用不同 output directory。

不要让 projector、LoRA 和 joint 共用一个目录。

---

# 14. 第 7 步：生成 Manifest

## 14.1 Manifest 的作用

Manifest 把原始 JSON/CSV 转成确定性的 JSONL records，明确记录：

```text
record ID
split
task
prompt variant
image path
mask path
label ID
question
answer
source dataset
metadata
```

训练不会在 worker 中临时猜测任务类型。

## 14.2 Manifest 输出位置

默认：

```text
<checkpoint.output_dir>/manifests/
```

例如 joint：

```text
outputs/m3d-phi3-joint-modernized-a100/manifests/
├── train.jsonl
├── train.jsonl.summary.json
├── validation.jsonl
├── validation.jsonl.summary.json
├── test.jsonl
└── test.jsonl.summary.json
```

因为每个 config 的 output directory 不同，所以每个 stage 需要自己的 manifests。

## 14.3 Joint manifest：推荐 PBS

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/m3d_joint_finetune.yaml \
  scripts/12_prepare_manifests_aspire2a.pbs
```

默认包含：

```text
caption
vqa_closed
vqa_open
vqa_yes_no
positioning
segmentation
```

## 14.4 Joint manifest：直接 Python

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python -m m3d.data.manifest \
  --config configs/m3d_joint_finetune.yaml \
  --splits train validation test
```

## 14.5 Projector stage manifest

Projector stage 只训练 caption，因此必须跳过其他 task：

```bash
python -m m3d.data.manifest \
  --config configs/m3d_projector_pretrain.yaml \
  --splits train validation test \
  --skip-vqa-closed \
  --skip-vqa-open \
  --skip-vqa-yes-no \
  --skip-positioning \
  --skip-generated-segmentation \
  --skip-refseg
```

不能使用包含 segmentation records 的 manifest，因为 projector config 中 segmentation branch 被禁用。

## 14.6 LoRA stage manifest

LoRA stage允许：

```text
caption
vqa_closed
vqa_open
vqa_yes_no
positioning
```

但禁用 segmentation。

```bash
python -m m3d.data.manifest \
  --config configs/m3d_lora_finetune.yaml \
  --splits train validation test \
  --skip-generated-segmentation \
  --skip-refseg
```

## 14.7 仅生成 train

```bash
python -m m3d.data.manifest \
  --config configs/m3d_joint_finetune.yaml \
  --splits train
```

## 14.8 暂时跳过文件检查

```bash
--no-verify-files
```

只用于调试 metadata。正式训练前必须进行真实文件检查。

## 14.9 Split overlap

Manifest 默认拒绝同一 task 的同一 image 同时出现在多个 split。

只有你明确确认原始数据设计允许时，才使用：

```bash
--allow-split-overlap
```

这可能造成数据泄漏，不能为了绕过错误随便开启。

## 14.10 查看 summary

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/m3d-phi3-joint-modernized-a100/manifests/train.jsonl.summary.json")
summary = json.loads(path.read_text())
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
```

重点确认：

```text
record_count
counts_by_task
fingerprint
split
```

## 14.11 Manifest 发生变化后不能精确恢复旧 checkpoint

Checkpoint 保存 manifest fingerprint。

以下任何变化都会让 exact resume 失败：

```text
增加/删除 records
改变 question/answer
改变 image path
改变 mask path
改变 task
重新划分 split
```

---

# 15. 第 8 步：运行 Release Audit

每次修改核心代码或 YAML 后：

```bash
source .venv/bin/activate
python scripts/13_release_audit.py \
  --root . \
  --output release_audit.json
```

检查：

```text
所有计划文件存在
所有 Python 可以 compile
所有 PBS/shell 通过 bash -n
所有 YAML 可以 parse
双 image encoder 契约仍存在
禁止的 .cuda() 模式
禁止使用 mask.sum() 进行 task routing
```

成功应显示：

```text
status: passed
```

---

# 16. 第 9 步：M3D-CLIP 预训练与检索评估

## 16.1 什么时候需要运行 CLIP

Joint 默认使用原始发布的：

```text
pretrained_ViT.bin
```

所以只想跑 joint baseline 时，CLIP 不是强制步骤。

以下情况运行 CLIP：

```text
你想复现 M3D-CLIP 对比学习
你想比较新的图像 encoder 训练方法
你想重新生成 image/text retrieval features
你想测试现代 SDPA/Flash-SDPA 对 CLIP 的影响
```

## 16.2 CLIP 配置

```text
configs/m3d_clip_pretrain.yaml
```

默认：

```text
image encoder：3D ViT
text encoder：bert-base-uncased
projection dimension：768
per-device batch：8
world size：2
effective batch：16
epochs：100
```

## 16.3 Startup-only

```bash
qsub \
  -v M3D_STARTUP_ONLY=1 \
  scripts/10_train_clip_aspire2a.pbs
```

只检查模型、数据、DDP 和 optimizer，不进入长训练。

## 16.4 正式 CLIP 训练

```bash
qsub scripts/10_train_clip_aspire2a.pbs
```

自定义输出目录：

```bash
qsub \
  -v M3D_OUTPUT_DIR=/scratch/.../outputs/my-clip-run \
  scripts/10_train_clip_aspire2a.pbs
```

快速测试：

```bash
qsub \
  -v M3D_EPOCHS=1,M3D_BATCH_SIZE=2,M3D_NUM_WORKERS=2 \
  scripts/10_train_clip_aspire2a.pbs
```

## 16.5 CLIP checkpoint

输出类似：

```text
outputs/m3d-clip-modernized-a100/
├── checkpoint-step-00001000/
│   ├── training_state.pt
│   └── COMPLETED.json
├── best/
├── final/
├── latest.json
└── training_result.json
```

`training_state.pt` 包含：

```text
model
optimizer
scheduler
trainer state
RNG
```

## 16.6 CLIP 断点续训

```bash
qsub \
  -v M3D_RESUME_FROM=/absolute/path/to/clip-output \
  scripts/10_train_clip_aspire2a.pbs
```

或具体 checkpoint 目录。

## 16.7 CLIP retrieval evaluation

```bash
qsub \
  -v M3D_CHECKPOINT=/absolute/path/to/clip-output \
  scripts/11_evaluate_clip_aspire2a.pbs
```

输出：

```text
retrieval_metrics.json
retrieval_features.pt
test_ir.csv
test_tr.csv
COMPLETED.json
```

## 16.8 `text_ir` 与 `text_tr` 的含义

```text
test_ir.csv：image → text retrieval
```

对每张 image，列出最相似的 text candidates。

```text
test_tr.csv：text → image retrieval
```

对每段 text，列出最相似的 image candidates。

## 16.9 Top 1000

默认：

```text
M3D_CSV_TOP_K=1000
```

即使测试集小于 1000，程序也会自动执行：

```text
effective_k = min(1000, test_size)
```

不会出现 `torch.topk k > size`。

## 16.10 新 CLIP encoder 与 MLLM 的连接限制

CLIP trainer 的 checkpoint 保存完整 CLIP state，而 Main ViT loader 默认读取原发布 `pretrained_ViT.bin` 兼容格式。

当前仓库没有额外提供自动的：

```text
CLIP training_state.pt
→ pretrained_ViT.bin
```

转换命令。

因此不要直接把 `training_state.pt` 填入 `main_vision.checkpoint_path` 并假设会自动识别。

---

# 17. 第 10 步：Projector Pretraining

## 17.1 配置

```text
configs/m3d_projector_pretrain.yaml
```

## 17.2 训练参数

该 stage 会先冻结完整模型，然后只开启：

```text
MM Projector
Phi-3 input token embeddings
Phi-3 LM head/output embeddings
```

不训练：

```text
Main 3D ViT
Phi-3 decoder 主体
SegVol
```

## 17.3 先生成 projector manifests

见前面的 Projector Manifest 命令。

确认：

```text
counts_by_task 中只能有 caption
```

## 17.4 Startup-only

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/m3d_projector_pretrain.yaml,M3D_STARTUP_ONLY=1 \
  scripts/06_train_aspire2a.pbs
```

## 17.5 正式训练

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/m3d_projector_pretrain.yaml \
  scripts/06_train_aspire2a.pbs
```

## 17.6 默认有效 batch

```text
per-device = 1
world size = 2
grad accumulation = 8
有效 global batch = 16
```

## 17.7 不要跨 stage 使用 exact resume

Projector stage 的 checkpoint 只能恢复同一个 projector stage。

不能直接：

```text
Projector checkpoint
--resume-from
LoRA config
```

因为 exact resume 会检查模型和训练语义 fingerprint。

---

# 18. 第 11 步：LoRA Fine-tuning

## 18.1 配置

```text
configs/m3d_lora_finetune.yaml
```

## 18.2 默认训练模块

```text
Phi-3 LoRA
Input embeddings
LM head
```

冻结：

```text
Main 3D ViT
MM Projector
SegVol
```

## 18.3 LoRA target modules

```text
q_proj
k_proj
v_proj
o_proj
gate_up_proj
down_proj
```

默认：

```text
rank = 16
alpha = 32
dropout = 0.05
```

## 18.4 先生成 LoRA manifests

必须跳过 segmentation records。

见前面的 LoRA Manifest 命令。

## 18.5 Startup-only

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/m3d_lora_finetune.yaml,M3D_STARTUP_ONLY=1 \
  scripts/06_train_aspire2a.pbs
```

## 18.6 正式训练

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/m3d_lora_finetune.yaml \
  scripts/06_train_aspire2a.pbs
```

## 18.7 为什么输出目录没有 `model_with_lora.bin`

现代训练保存的是 Distributed Checkpoint：

```text
checkpoint-step-xxxxxxxx/dcp/
```

这不是原项目的单个 `model_with_lora.bin`。

训练后使用 export：

```bash
qsub \
  -v M3D_CHECKPOINT=/path/to/training-output,M3D_EXPORT_FORMAT=adapter \
  scripts/08_export_aspire2a.pbs
```

生成标准：

```text
language_adapter/adapter_model.safetensors
language_adapter/adapter_config.json
```

---

# 19. 第 12 步：Joint Language + Segmentation Fine-tuning

## 19.1 配置

```text
configs/m3d_joint_finetune.yaml
```

## 19.2 默认可训练模块

```text
Main 3D ViT
MM Projector
Phi-3 LoRA
Input embeddings
LM head
Segmentation Projector
SegVol 3D ViT
SegVol Prompt Encoder
SegVol Mask Decoder
```

## 19.3 默认模块学习率

```text
language model          1e-5
Main ViT                 5e-6
SegVol ViT               5e-6
MM Projector             5e-5
Segmentation Projector   5e-5
Segmentation Decoder     1e-5
Token embeddings         5e-5
```

## 19.4 Segmentation loss

```text
total loss
= language loss
+ 1.0 × Dice loss
+ 1.0 × BCEWithLogits loss
```

## 19.5 全零 mask

全零 target：

```text
仍是 segmentation task
仍运行 SegVol
仍计算 BCE
原 M3D Dice 公式得到 Dice loss = 1
```

不会被错误路由成文本任务。

## 19.6 先生成 joint manifests

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/m3d_joint_finetune.yaml \
  scripts/12_prepare_manifests_aspire2a.pbs
```

## 19.7 不要直接提交 120 小时训练

先完成下一节的四个 integration tests。

---

# 20. 第 13 步：正式训练前必须运行的四个集成测试

## 20.1 DDP runtime integration

```bash
qsub scripts/05_aspire2a_integration.pbs
```

执行：

```text
1 个 caption optimizer update
1 个 segmentation optimizer update
```

验证：

```text
真实 tokenizer
真实 Phi-3
真实 checkpoints
双 A100
NCCL
Flash-SDPA
fused AdamW
Caption 是否跳过 SegVol
Segmentation 是否训练第二套 ViT
```

## 20.2 DDP checkpoint integration

```bash
qsub \
  -v M3D_INTEGRATION_SUITE=checkpoint \
  scripts/05_aspire2a_integration.pbs
```

验证：

```text
训练一步
保存 checkpoint
再训练一步扰动状态
加载 checkpoint
重放下一 batch
模型/optimizer/scheduler/sampler/RNG 精确恢复
```

## 20.3 FSDP2 runtime integration

```bash
qsub \
  -v M3D_INTEGRATION_STRATEGY=fsdp2 \
  scripts/05_aspire2a_integration.pbs
```

## 20.4 FSDP2 checkpoint integration

```bash
qsub \
  -v M3D_INTEGRATION_SUITE=checkpoint,M3D_INTEGRATION_STRATEGY=fsdp2 \
  scripts/05_aspire2a_integration.pbs
```

## 20.5 推荐顺序

```text
1. DDP runtime
2. DDP checkpoint
3. FSDP2 runtime
4. FSDP2 checkpoint
```

即使你最终只用 DDP，也至少运行前两个。

## 20.6 Integration 日志

```text
logs/integration/<JOBID>/integration.log
logs/integration/<JOBID>/gpu.csv
```

## 20.7 成功条件

报告中应看到：

```text
status = passed
completed_optimizer_steps = 2
committed_microbatches = 2
共享 image encoder parameters = 0
共享 image encoder storages = 0
```

Checkpoint integration 还应看到：

```text
model restore exact = true
optimizer restore exact = true
scheduler restore exact = true
sampler restore exact = true
RNG restore exact = true
```

---

# 21. 第 14 步：提交正式训练

## 21.1 Startup-only

```bash
qsub \
  -v M3D_STARTUP_ONLY=1 \
  scripts/06_train_aspire2a.pbs
```

Startup-only 会完整执行：

```text
配置读取
CUDA/NCCL 初始化
Tokenizer
Manifest/DataLoader
Phi-3
LoRA
Main ViT
SegVol ViT
Checkpoint load
DDP/FSDP2 wrap
Optimizer
Scheduler
Checkpoint manager
```

但不消费训练 batch。

## 21.2 正式 DDP 训练

```bash
qsub scripts/06_train_aspire2a.pbs
```

默认资源：

```text
queue：glong
node：1
GPU：2
CPU：64
RAM：440 GB
walltime：120 小时
```

## 21.3 指定配置

```bash
qsub \
  -v M3D_CONFIG=/absolute/path/configs/my_joint_run.yaml \
  scripts/06_train_aspire2a.pbs
```

## 21.4 指定输出目录

```bash
qsub \
  -v M3D_OUTPUT_DIR=/scratch/.../outputs/my-joint-run \
  scripts/06_train_aspire2a.pbs
```

Fresh run 时，输出目录中若已有 checkpoint 或 `training_result.json`，PBS 默认拒绝覆盖。

## 21.5 常用覆盖项

```bash
qsub \
  -v M3D_EPOCHS=3,M3D_GRAD_ACCUM=8,M3D_NUM_WORKERS=8,M3D_SAVE_EVERY_STEPS=500 \
  scripts/06_train_aspire2a.pbs
```

## 21.6 使用 override file

建立：

```text
my_overrides.txt
```

内容：

```text
optimization.epochs=3
optimization.gradient_accumulation_steps=8
data.num_workers=8
checkpoint.save_every_steps=500
logging.log_every_steps=5
```

提交：

```bash
qsub \
  -v M3D_OVERRIDES_FILE=/absolute/path/my_overrides.txt \
  scripts/06_train_aspire2a.pbs
```

使用文件比在 `qsub -v` 中传复杂 list 更安全。

## 21.7 离线模式

当 Phi-3/tokenizer 已缓存：

```bash
qsub \
  -v M3D_LOCAL_FILES_ONLY=1 \
  scripts/06_train_aspire2a.pbs
```

## 21.8 Node-local data cache

默认关闭。

启用：

```bash
qsub \
  -v M3D_NODE_LOCAL_CACHE=1 \
  scripts/06_train_aspire2a.pbs
```

数据会在首次访问时原子复制到 `$TMPDIR`。

注意：

```text
没有自动 eviction
TMPDIR 必须有足够容量
```

---

# 22. 第 15 步：监控 PBS 作业和日志

## 22.1 查看排队状态

```bash
qstat -u "$USER"
```

## 22.2 查看原因

```bash
qstat -f <JOBID> | less
```

## 22.3 实时训练日志

```bash
tail -f logs/training/<JOBID>/train.log
```

## 22.4 GPU 使用率

```bash
tail -f logs/training/<JOBID>/gpu.csv
```

## 22.5 训练日志目录

```text
logs/training/<JOBID>/
├── train.log
├── gpu.csv
├── command.txt
├── environment.txt
├── modules.txt
└── pip-freeze.txt
```

## 22.6 TensorBoard

默认：

```text
logs/tensorboard/<experiment-name>
```

在可访问节点运行：

```bash
tensorboard \
  --logdir logs/tensorboard \
  --port 6006 \
  --bind_all
```

然后通过 SSH tunnel 访问。

## 22.7 Queue running job limit

错误：

```text
Not Running: User has reached queue glong running job limit
```

含义不是代码错误，而是你已经达到该 queue 的并发运行上限。

处理：

```text
等待当前 glong 作业结束
删除不需要的排队作业
改用允许的其他 queue（必须符合项目政策）
```

删除作业：

```bash
qdel <JOBID>
```

---

# 23. 第 16 步：断点续训

## 23.1 Checkpoint 目录

```text
checkpoint-step-00001000/
├── COMPLETED.json
├── trainer_state.json
├── resolved_config.json
├── dcp/
└── rank_state/
```

只有 `COMPLETED.json` 存在才可恢复。

## 23.2 恢复指定 checkpoint

```bash
qsub \
  -v M3D_RESUME_FROM=/absolute/path/checkpoint-step-00001000 \
  scripts/06_train_aspire2a.pbs
```

## 23.3 恢复最新 checkpoint

```bash
qsub \
  -v M3D_RESUME_FROM=latest \
  scripts/06_train_aspire2a.pbs
```

PBS 会从配置对应的 output directory 读取 `latest.json`。

## 23.4 Exact resume 要求

必须保持一致：

```text
world size
per-device batch size
gradient accumulation
模型参数布局
LoRA target modules
optimizer groups
manifest fingerprint
task weights
sampler seed
训练计划
```

## 23.5 不能跨 world size 恢复

例如：

```text
原训练：2 GPUs
恢复：4 GPUs
```

当前 exact sampler schedule 不支持这种变化。

## 23.6 不能从 accumulation window 中间恢复

Checkpoint 只允许保存于：

```text
optimizer.step
scheduler.step
zero_grad
```

之后。

否则未提交的 `.grad` 不在普通 checkpoint 中。

## 23.7 不要把 stage 切换当成 resume

```text
Projector stage checkpoint
≠
LoRA stage exact resume checkpoint
```

跨阶段需要权重初始化/转换，不是恢复同一训练状态。

---

# 24. 第 17 步：DDP 与 FSDP2 的选择

## 24.1 DDP

优点：

```text
最快
通信逻辑简单
fused AdamW
checkpoint 和评估更直接
```

缺点：

```text
每张 GPU 保存完整模型参数
每张 GPU 保存完整 optimizer states
```

默认：

```bash
qsub scripts/06_train_aspire2a.pbs
```

## 24.2 FSDP2

```bash
qsub \
  -v M3D_STRATEGY=fsdp2 \
  scripts/06_train_aspire2a.pbs
```

优点：

```text
参数分片
梯度分片
optimizer state 分片
降低单卡模型状态显存
```

缺点：

```text
更多通信
复杂度更高
可能较慢
评估要求各 rank forward 次数一致
```

## 24.3 什么时候切 FSDP2

推荐顺序：

```text
1. batch size = 1
2. BF16
3. Main ViT activation checkpointing
4. SegVol ViT activation checkpointing
5. Phi-3 gradient checkpointing
6. 确认没有保存全部 hidden states
7. DDP 仍 OOM
8. 切 FSDP2
```

当前默认配置已经完成前六项。

## 24.4 Gradient accumulation 不能解决所有 OOM

增加 accumulation：

```text
减少每次 optimizer update 的单卡 batch 数需求
增大有效 batch
```

但不会减少一个单病例 forward 的 activation memory。

如果 batch size 已是 1，仍在单个 segmentation forward OOM，应该使用 FSDP2、activation checkpointing 或进一步模型级优化。

## 24.5 冻结模块是否能减少 OOM

冻结参数可减少：

```text
gradient
optimizer states
某些 activation 保存
```

但只要 forward 仍需要该模块，它的临时 activation 和参数仍会占显存。

如果不想牺牲任何训练模块，优先使用 FSDP2，而不是冻结 SegVol。

---

# 25. 第 18 步：导出训练结果

## 25.1 为什么需要 export

训练 checkpoint 是分布式格式，不适合直接作为最终推理发布目录。

Export 会生成：

```text
完整 M3D safetensors
独立非语言 components
LoRA adapter
可选 LoRA-merged Phi-3
tokenizer
manifest
hash
```

## 25.2 导出最新训练结果

```bash
qsub \
  -v M3D_CHECKPOINT=/absolute/path/to/training-output \
  scripts/08_export_aspire2a.pbs
```

`M3D_CHECKPOINT` 可以是：

```text
训练 output directory
latest.json
具体 checkpoint-step directory
```

## 25.3 指定格式

```bash
qsub \
  -v M3D_CHECKPOINT=/path/to/output,M3D_EXPORT_FORMAT=all \
  scripts/08_export_aspire2a.pbs
```

格式：

| 格式 | 内容 |
|---|---|
| `bundle` | 完整 M3D safetensors + components + tokenizer |
| `adapter` | bundle + PEFT adapter |
| `merged` | bundle + LoRA-merged Phi-3 |
| `all` | 全部 |

## 25.4 指定输出目录

```bash
qsub \
  -v M3D_CHECKPOINT=/path/to/output,M3D_OUTPUT_DIR=/path/to/export \
  scripts/08_export_aspire2a.pbs
```

## 25.5 覆盖旧 export

```bash
M3D_OVERWRITE=1
```

仍然采用 staging + atomic rename，不会先删除旧 export 再开始长时间导出。

## 25.6 导出目录

```text
m3d-export/
├── COMPLETED.json
├── export_manifest.json
├── resolved_config.json
├── tokenizer/
├── m3d_model/
├── components/
│   ├── main_vision/
│   ├── multimodal_projector/
│   ├── segmentation_projector/
│   └── segvol/
├── language_adapter/
└── language_merged/
```

确认两个 image encoder 都存在：

```text
components/main_vision
components/segvol
```

## 25.7 Shard size

默认：

```text
4GB
```

修改：

```bash
M3D_MAX_SHARD_SIZE=8GB
```

---

# 26. 第 19 步：单病例推理

## 26.1 准备问题文件

```bash
cat > question.txt <<'EOF2'
Please identify and segment the coronary stenosis.
EOF2
```

推荐问题文件而不是直接通过 `qsub -v` 传长句，因为长句可能包含空格、逗号和引号。

## 26.2 自动模式

```bash
qsub \
  -v M3D_EXPORT_DIR=/absolute/path/to/export,M3D_IMAGE=/absolute/path/to/ct.nii.gz,M3D_QUESTION_FILE=/absolute/path/to/question.txt,M3D_MODE=auto \
  scripts/09_inference_aspire2a.pbs
```

`auto`：

```text
生成 [SEG] → 运行 SegVol
不生成 [SEG] → 只输出文字
```

## 26.3 强制 segmentation

```bash
M3D_MODE=segmentation
```

若模型未生成 `[SEG]`，作业失败。不会使用全零 prompt 伪造 mask。

## 26.4 纯文本

```bash
M3D_MODE=text
```

永远跳过 SegVol。

## 26.5 输入要求

支持：

```text
.npy
.nii
.nii.gz
```

仍要求：

```text
shape = [1,32,256,256] 逻辑布局
range = [0,1]
```

推理不会自动预处理。

## 26.6 输出

```text
inference-<JOBID>/
├── COMPLETED.json
├── question.txt
├── result.json
├── build_report.json
├── mask.nii.gz 或 mask.npy
└── probability.nii.gz 或 probability.npy
```

没有触发 segmentation 时，不会生成 mask 文件。

## 26.7 Mask threshold

默认：

```text
0.5
```

修改：

```bash
M3D_MASK_THRESHOLD=0.4
```

## 26.8 确定性生成

默认：

```text
do_sample = false
num_beams = 1
seed = 42
```

采样：

```bash
qsub \
  -v M3D_EXPORT_DIR=...,M3D_IMAGE=...,M3D_QUESTION_FILE=...,M3D_DO_SAMPLE=1,M3D_TEMPERATURE=0.7,M3D_TOP_P=0.9 \
  scripts/09_inference_aspire2a.pbs
```

不要同时设置：

```text
do_sample=true
num_beams>1
```

## 26.9 NIfTI 输出只显示一个平面的常见原因

先检查实际 mask shape：

```bash
python - <<'PY'
import nibabel as nib
img = nib.load("mask.nii.gz")
print(img.shape)
print(img.header.get_zooms())
print(nib.aff2axcodes(img.affine))
PY
```

正常应为三维 volume，而不是单张 2D slice。

若只有一个轴尺寸为 1，问题通常来自上游数据 shape 或保存时 transpose，不是 ITK-SNAP 本身。

---

# 27. 第 20 步：完整模型评估

## 27.1 从 portable export 评估

```bash
qsub \
  -v M3D_EXPORT_DIR=/absolute/path/to/export \
  scripts/07_evaluate_aspire2a.pbs
```

默认：

```text
source = export
split = test
strategy = ddp
tasks = all
```

## 27.2 只评估 segmentation

```bash
qsub \
  -v M3D_EXPORT_DIR=/path/to/export,M3D_TASKS=segmentation \
  scripts/07_evaluate_aspire2a.pbs
```

## 27.3 多个任务

建立：

```text
eval_tasks.txt
```

内容：

```text
caption
vqa_closed
vqa_open
segmentation
```

提交：

```bash
qsub \
  -v M3D_EXPORT_DIR=/path/to/export,M3D_TASKS_FILE=/path/to/eval_tasks.txt \
  scripts/07_evaluate_aspire2a.pbs
```

## 27.4 快速评估前 100 个样本

```bash
qsub \
  -v M3D_EXPORT_DIR=/path/to/export,M3D_MAX_SAMPLES_PER_TASK=100 \
  scripts/07_evaluate_aspire2a.pbs
```

这是全局 100 个，不是每个 GPU 100 个。

## 27.5 从训练 checkpoint 直接评估

DDP：

```bash
qsub \
  -v M3D_EVAL_SOURCE=checkpoint,M3D_CHECKPOINT=/path/to/training-output,M3D_STRATEGY=ddp \
  scripts/07_evaluate_aspire2a.pbs
```

FSDP2：

```bash
qsub \
  -v M3D_EVAL_SOURCE=checkpoint,M3D_CHECKPOINT=/path/to/training-output,M3D_STRATEGY=fsdp2 \
  scripts/07_evaluate_aspire2a.pbs
```

## 27.6 Caption metrics

```text
Exact Match
Token F1
ROUGE-L F1
BLEU-1
BLEU-4
METEOR
可选 BERTScore
```

## 27.7 VQA metrics

```text
Normalized Exact Match
Token F1
ROUGE-L F1
```

## 27.8 Positioning metrics

对 `<bx_start>[x1,y1,z1,x2,y2,z2]<bx_end>` 计算：

```text
box parse rate
3D IoU
IoU ≥ 0.25
IoU ≥ 0.50
normalized centre distance
```

## 27.9 Segmentation metrics

```text
[SEG] trigger rate
Hard Dice
Hard IoU
Precision
Recall
Specificity
Legacy soft Dice
Empty-target count
Empty-target correct-empty rate
Non-empty target Dice
```

## 27.10 BERTScore

```bash
qsub \
  -v M3D_EXPORT_DIR=/path/to/export,M3D_BERTSCORE=1 \
  scripts/07_evaluate_aspire2a.pbs
```

BERTScore 会额外加载文本 encoder，增加显存和时间。

## 27.11 Retrieval evaluation

完整 generative M3D 没有单独的 M3D-CLIP text encoder 接口，因此 retrieval 使用预先计算的 features：

```bash
qsub \
  -v M3D_EXPORT_DIR=/path/to/export,M3D_RETRIEVAL_IMAGE_FEATURES=/path/image_features.npy,M3D_RETRIEVAL_TEXT_FEATURES=/path/text_features.npy \
  scripts/07_evaluate_aspire2a.pbs
```

两份 features 必须：

```text
shape = [N,C]
第 i 张 image 对应第 i 段 text
```

## 27.12 评估输出

```text
evaluation-output/
├── COMPLETED.json
├── evaluation_report.json
├── metrics/
├── predictions/
└── rank_rows/
```

每个 rank 先独立写 JSONL，rank 0 最后合并，避免把所有长文本 prediction 复制到每个 GPU process。

---

# 28. 输出文件说明

## 28.1 Joint training output

```text
output_dir/
├── resolved_config.json
├── startup_report.json
├── trainer_build_report.json
├── tokenizer/
├── checkpoint-step-*/
├── latest.json
└── training_result.json
```

## 28.2 `startup_report.json`

包含：

```text
环境版本
GPU
Tokenizer IDs
Manifest fingerprint
模型参数数量
两个 image encoder 独立性
checkpoint load report
DDP/FSDP2 report
optimizer groups
scheduler plan
```

## 28.3 `training_result.json`

包含：

```text
completed optimizer steps
total optimizer steps
final epoch
final committed microbatch
elapsed time
final checkpoint
resume source
```

## 28.4 GPU telemetry

所有 PBS 都会写 GPU CSV，用于检查：

```text
是否两张 GPU 都工作
显存峰值
segmentation step 是否明显更高
checkpoint 时 GPU 是否空闲
是否长期低 utilization
```

---

# 29. PBS 环境变量速查表

## 29.1 Preflight

| 变量 | 默认值 | 用途 |
|---|---:|---|
| `M3D_VENV_DIR` | `<root>/.venv` | Python 环境 |
| `M3D_CACHE_ROOT` | `<root>/.cache` | cache 根目录 |
| `M3D_EXPECTED_GPUS` | `2` | 预期 GPU 数 |
| `M3D_OMP_NUM_THREADS` | `8` | OMP threads |
| `M3D_MKL_NUM_THREADS` | `8` | MKL threads |

## 29.2 Integration

| 变量 | 默认值 | 可选值/用途 |
|---|---:|---|
| `M3D_INTEGRATION_SUITE` | `runtime` | `runtime` / `checkpoint` |
| `M3D_INTEGRATION_STRATEGY` | `ddp` | `ddp` / `fsdp2` |
| `M3D_CHECKPOINT_MODE` | `async` | `async` / `sync` |
| `M3D_NUM_WORKERS` | `0` | 每 rank workers |
| `M3D_KEEP_CHECKPOINT` | `0` | 是否保留大型测试 DCP |
| `M3D_LOCAL_FILES_ONLY` | `0` | HF 离线模式 |

## 29.3 正式训练

| 变量 | 用途 |
|---|---|
| `M3D_CONFIG` | 指定 YAML |
| `M3D_STRATEGY` | `ddp` / `fsdp2` |
| `M3D_OUTPUT_DIR` | 覆盖 output directory |
| `M3D_RESUME_FROM` | checkpoint 路径或 `latest` |
| `M3D_STARTUP_ONLY` | 只启动验证 |
| `M3D_EPOCHS` | 覆盖 epochs |
| `M3D_GRAD_ACCUM` | 覆盖 accumulation |
| `M3D_STEPS_PER_EPOCH` | 固定每 epoch microbatches |
| `M3D_NUM_WORKERS` | 每 rank worker 数 |
| `M3D_SAVE_EVERY_STEPS` | checkpoint interval |
| `M3D_KEEP_LAST_N` | 保留 checkpoint 数量 |
| `M3D_LOG_EVERY_STEPS` | logging interval |
| `M3D_CHECKPOINT_ASYNC` | 开关异步保存 |
| `M3D_COMPILE_MODEL` | 开关 `torch.compile` |
| `M3D_NODE_LOCAL_CACHE` | 开关 node-local volume cache |
| `M3D_LOCAL_FILES_ONLY` | HF 离线加载 |
| `M3D_OVERRIDES_FILE` | 多行配置覆盖文件 |
| `M3D_ALLOW_EXISTING_OUTPUT` | 允许 fresh run 使用已有目录；不推荐 |

## 29.4 评估

| 变量 | 用途 |
|---|---|
| `M3D_EVAL_SOURCE` | `export` / `checkpoint` |
| `M3D_EXPORT_DIR` | portable export |
| `M3D_CHECKPOINT` | training checkpoint/output |
| `M3D_STRATEGY` | checkpoint source 时 `ddp` / `fsdp2` |
| `M3D_SPLIT` | `validation` / `test` |
| `M3D_TASKS` | 单个任务或 `all` |
| `M3D_TASKS_FILE` | 多任务文件 |
| `M3D_MAX_SAMPLES_PER_TASK` | 快速评估样本数 |
| `M3D_MAX_NEW_TOKENS` | 生成长度 |
| `M3D_MASK_THRESHOLD` | mask threshold |
| `M3D_BERTSCORE` | 开关 BERTScore |
| `M3D_RETRIEVAL_IMAGE_FEATURES` | image feature 文件 |
| `M3D_RETRIEVAL_TEXT_FEATURES` | text feature 文件 |
| `M3D_OVERWRITE` | 覆盖输出 |

## 29.5 导出

| 变量 | 用途 |
|---|---|
| `M3D_CHECKPOINT` | source checkpoint/output |
| `M3D_OUTPUT_DIR` | export 目录 |
| `M3D_EXPORT_FORMAT` | `bundle`/`adapter`/`merged`/`all` |
| `M3D_MAX_SHARD_SIZE` | 例如 `4GB` |
| `M3D_STRATEGY` | `ddp` / `fsdp2` |
| `M3D_OVERWRITE` | 原子替换旧 export |

## 29.6 推理

| 变量 | 用途 |
|---|---|
| `M3D_EXPORT_DIR` | portable export |
| `M3D_IMAGE` | `.npy/.nii/.nii.gz` |
| `M3D_QUESTION_FILE` | 推荐的问题文件 |
| `M3D_QUESTION` | 简短问题字符串 |
| `M3D_MODE` | `auto`/`text`/`segmentation` |
| `M3D_MAX_NEW_TOKENS` | 生成长度 |
| `M3D_MASK_THRESHOLD` | 二值化阈值 |
| `M3D_SAVE_MASK` | 保存 binary mask |
| `M3D_SAVE_PROBABILITY` | 保存 probability |
| `M3D_OUTPUT_DIR` | 输出目录 |
| `M3D_OVERWRITE` | 覆盖旧结果 |

---

# 30. 常见错误与排查

## 30.1 `Manifest does not exist`

错误示例：

```text
Manifest does not exist: .../manifests/train.jsonl
```

处理：

```text
确认使用的是哪个 config
查看该 config 的 checkpoint.output_dir
为该 config 重新生成 manifests
```

## 30.2 Manifest 中有 segmentation，但 config 禁用了 SegVol

常见于 Projector 或 LoRA stage。

处理：

```text
Projector manifest 跳过所有非 caption task
LoRA manifest 跳过 generated segmentation 和 refseg
```

## 30.3 `No available kernel` / Flash-SDPA 失败

检查：

```text
是否在 A100 compute node
PyTorch 是否是 2.6.0 cu118
输入 dtype 是否 BF16
是否错误加载了额外 CUDA/cuDNN/NCCL module
```

先重新运行：

```bash
qsub scripts/02_preflight.pbs
```

## 30.4 DDP OOM

确认：

```text
per-device batch size = 1
两个 ViT checkpointing 开启
Phi-3 gradient checkpointing 开启
没有 output_hidden_states=True
没有保留完整 logits
```

然后切：

```bash
M3D_STRATEGY=fsdp2
```

## 30.5 CPU RAM OOM

常见于：

```text
export format=all
async checkpoint staging
DataLoader workers 太多
node-local cache 太大
```

处理：

```text
减少 num_workers
使用 bundle 导出而非 all
关闭 async checkpoint 做诊断
关闭 node-local cache
```

## 30.6 `find_unused_parameters` 错误或 DDP hang

Joint task graph 是动态的。

必须：

```yaml
find_unused_parameters: true
static_graph: false
```

也必须保证同一 global step 所有 ranks 使用相同 task。

## 30.7 训练中没有 `model_with_lora.bin`

正常。

现代训练使用 Distributed Checkpoint。

使用 export 生成：

```text
adapter_model.safetensors
adapter_config.json
```

## 30.8 Checkpoint 无法恢复

检查：

```text
是否有 COMPLETED.json
world size 是否相同
manifest 是否改变
config 是否改变
LoRA targets 是否改变
output directory 是否对应正确 stage
```

## 30.9 `selected index k out of range`

新版 CLIP evaluator 已执行：

```text
k = min(requested_k, dataset_size)
```

确认你运行的是：

```text
src/m3d/evaluate_clip.py
```

而不是旧原仓库 eval script。

## 30.10 输入 intensity 超出 `[0,1]`

错误说明 loader 不会自动 normalise。

先确定你的预处理策略：

```text
CT windowing
clip
normalization
resize
```

图像和 mask 必须使用一致的空间变换。

## 30.11 Mask 只有一层

检查：

```text
原始 mask shape
模型 logits shape
保存前 transpose
NIfTI affine
```

模型最终应输出：

```text
[B,1,32,256,256]
```

## 30.12 PBS 作业排队不运行

```text
User has reached queue glong running job limit
```

是 queue 限制，不是代码错误。

## 30.13 `accelerate: command not found`

本项目正式入口不使用 `accelerate launch`。

使用：

```text
torchrun
```

PBS 脚本已经封装。

## 30.14 环境中 CUDA 已经 load，PBS 还需要 load 吗

需要。

每个 PBS job 是独立 shell，必须在脚本内重新建立 module stack。

## 30.15 `CUDA_VISIBLE_DEVICES` 显示 UUID

PBS 可能把 GPU 以 UUID 暴露：

```text
GPU-xxxxxxxx
```

这是正常的。不要手工改成 `0,1`。`torchrun` 会在 PBS 分配范围内使用本地 rank。

## 30.16 Training loss 多小才算合格

不能只看一个固定数字。

需要分别观察：

```text
language loss
Dice loss
BCE loss
validation metrics
[SEG] trigger rate
non-empty mask Dice
empty mask false-positive rate
```

总 loss 由多个尺度不同的项相加，不能直接与单一语言模型 loss 比较。

---

# 31. 从代码角度理解完整执行顺序

## 31.1 Joint training

```text
m3d.train.main
    ↓
load_config
    ↓
RuntimeContext 初始化 CUDA/NCCL
    ↓
build_tokenizer
    ↓
build_training_data_pipeline
    ├── read_manifest
    ├── DatasetCatalog
    ├── task datasets
    ├── DistributedTaskBatchSampler
    ├── M3DCollator
    └── DataLoader
    ↓
build_model_synchronously
    ↓
build_m3d_model
    ├── Phi-3 + LoRA
    ├── Main 3D ViT
    ├── MM Projector
    ├── Segmentation Projector
    └── SegVol
         ├── 独立 SegVol 3D ViT
         ├── Prompt Encoder
         └── Mask Decoder
    ↓
load published components
    ↓
prepare_distributed_model
    ├── DDP
    └── FSDP2
    ↓
build_optimizer
    ↓
build_scheduler
    ↓
CheckpointManager
    ↓
M3DTrainer.train
```

## 31.2 一个 text microbatch

```text
M3DBatch
    ↓
Main ViT
    ↓
Projector
    ↓
替换 <im_patch> embeddings
    ↓
Phi-3
    ↓
Selective LM head
    ↓
Language loss
    ↓
Backward
```

## 31.3 一个 segmentation microbatch

```text
M3DBatch
    ├── image
    ├── text
    └── real segmentation target

image → Main ViT → Projector → Phi-3
                              ├── Language loss
                              └── [SEG] hidden
                                         ↓
                               Segmentation Projector
                                         ↓
image → 独立 SegVol ViT → Prompt Encoder → Mask Decoder
                                         ↓
                                  Dice + BCE
```

## 31.4 Inference

```text
load portable export
    ↓
load image
    ↓
Main ViT + Projector，只运行一次
    ↓
Phi-3 generation
    ↓
完整序列 replay 获取 [SEG] 前一位置 hidden state
    ↓
只有生成 [SEG] 的 rows 才运行 SegVol
    ↓
保存 mask/probability
```

---

# 32. 所有核心文件的职责

## 32.1 根目录

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 固定 Python 依赖版本 |
| `pyproject.toml` | editable install 和 console scripts |
| `LICENSE` | MIT License 与 attribution |
| `README.md` | 当前中文完整手册 |

## 32.2 配置

| 文件 | 用途 |
|---|---|
| `m3d_clip_pretrain.yaml` | CLIP 预训练 |
| `m3d_projector_pretrain.yaml` | Projector stage |
| `m3d_lora_finetune.yaml` | LoRA 文本任务 stage |
| `m3d_joint_finetune.yaml` | 语言 + segmentation joint stage |

## 32.3 数据层

| 文件 | 用途 |
|---|---|
| `schema.py` | Task、Sample、Batch 类型 |
| `io.py` | NPY/NIfTI 读取、shape/range 验证 |
| `transforms.py` | 同步 3D augmentation |
| `manifest.py` | 原 metadata → JSONL manifest |
| `prompt_templates.py` | Caption/VQA/Box/Seg prompt |
| `dataset_catalog.py` | Task 与执行图契约 |
| `anatomy_catalog.py` | Anatomy/class metadata |
| `datasets.py` | 每个任务 Dataset |
| `sampler.py` | Task-balanced homogeneous schedule |
| `collator.py` | Stack image/mask、动态 padding |
| `loader.py` | 完整 DataLoader pipeline |
| `clip_data.py` | CLIP Dataset 与 loader |

## 32.4 模型层

| 文件 | 用途 |
|---|---|
| `attention.py` | SDPA/Flash attention |
| `vit3d.py` | 3D ViT |
| `checkpoint.py` | 原 M3D/SegVol 权重兼容加载 |
| `projector.py` | 2048 → 256 visual token projector |
| `language.py` | Phi-3、LoRA、visual embedding replacement |
| `segmentation_prompt.py` | `[SEG]` hidden → prompt embedding |
| `segvol_prompt_encoder.py` | Sparse/dense prompt |
| `segvol_transformer.py` | Two-Way Transformer |
| `segvol_mask_decoder.py` | 3D mask decoder |
| `segvol.py` | 完整独立 SegVol branch |
| `loss.py` | Language + Dice + BCE |
| `m3d.py` | 完整 M3D task routing |
| `clip.py` | M3D-CLIP 双编码器 |
| `clip_loss.py` | 分布式 CLIP loss 与 retrieval |

## 32.5 训练基础设施

| 文件 | 用途 |
|---|---|
| `config.py` | YAML 解析与验证 |
| `runtime.py` | Device、seed、distributed runtime |
| `distributed.py` | DDP/FSDP2 wrapping |
| `optim.py` | 按模块分组的 AdamW |
| `scheduler.py` | Warmup + cosine |
| `checkpointing.py` | DCP save/load/exact resume |
| `trainer.py` | Joint training loop |
| `train.py` | Joint training entry |
| `clip_trainer.py` | CLIP training loop |
| `train_clip.py` | CLIP entry |
| `export.py` | Portable export |
| `inference.py` | 单病例 inference |
| `evaluate.py` | Generative M3D evaluation |
| `evaluate_clip.py` | CLIP retrieval evaluation |

## 32.6 PBS 与工具脚本

| 文件 | 用途 |
|---|---|
| `00_setup_environment.sh` | 建立环境 |
| `01_preflight.py` | GPU/NCCL/Flash 测试 |
| `02_preflight.pbs` | 提交 preflight |
| `03_aspire2a_integration.py` | 两任务真实训练测试 |
| `04_aspire2a_checkpoint_integration.py` | 精确恢复测试 |
| `05_aspire2a_integration.pbs` | 集成测试 PBS |
| `06_train_aspire2a.pbs` | 正式 joint/projector/LoRA 训练 |
| `07_evaluate_aspire2a.pbs` | 完整评估 |
| `08_export_aspire2a.pbs` | 模型导出 |
| `09_inference_aspire2a.pbs` | 单病例推理 |
| `10_train_clip_aspire2a.pbs` | CLIP 训练 |
| `11_evaluate_clip_aspire2a.pbs` | CLIP 评估 |
| `12_prepare_manifests_aspire2a.pbs` | 默认 joint manifest 生成 |
| `13_release_audit.py` | 全项目静态审计 |

---

# 33. 当前已知边界

## 33.1 当前容器测试与 ASPIRE 2A 实测不同

仓库内 self-tests 覆盖：

```text
Python syntax
CPU 数值一致性
shape contract
checkpoint key
小模型 forward/backward
sampler resume
loss normalization
```

以下必须在 ASPIRE 2A 实际运行：

```text
双 A100
NCCL
Flash-SDPA
真实 Phi-3 4B
真实 published checkpoints
FSDP2 DTensor
异步 DCP
完整 32×256×256 segmentation forward
```

这就是四个 integration jobs 存在的原因。

## 33.2 不自动预处理原始 CT

本项目训练代码不是原始 DICOM/NIfTI preprocessing pipeline。

必须先准备一致的：

```text
orientation
spacing
crop
resize
intensity normalization
image-mask alignment
```

## 33.3 不自动跨训练阶段转换组件权重

当前提供每个 stage 的训练入口，但不提供所有 stage 之间的自动 component extraction/conversion。

不要把 exact resume 当作 stage transition。

## 33.4 非临床软件

本项目用于研究和工程验证，不是医疗器械，不应直接用于临床诊断或治疗决策。

---

# 34. 许可证与医学用途声明

本重构保留原项目的 MIT License 与原版权声明。

以下内容可能拥有各自独立的许可证和访问条款：

```text
Phi-3 模型权重
BERT 模型权重
M3D 数据集
SegVol 权重
原 M3D 发布权重
第三方评价模型
```

在重新发布、商业化、临床研究或跨机构传输前，请分别确认数据和模型授权。

> 本代码仅供研究使用，不构成医学建议，也不是经过监管机构批准的医疗器械。

---

# 最短可运行命令清单

下面是一条使用原发布组件权重的 joint baseline 最短路线：

```bash
# 1. 进入项目
cd /scratch/users/industry/theiahealth/theiahth/M3D-modernized

# 2. 建环境
bash scripts/00_setup_environment.sh

# 3. 安装 package
module load python/3.10.9
source .venv/bin/activate
python -m pip install -e . --no-deps
# 4. Preflight
qsub scripts/02_preflight.pbs

# 5. 生成 joint manifests
qsub scripts/12_prepare_manifests_aspire2a.pbs

# 6. Release audit
python scripts/13_release_audit.py --root . --output release_audit.json

# 7. DDP runtime integration
qsub scripts/05_aspire2a_integration.pbs

# 8. DDP checkpoint integration
qsub -v M3D_INTEGRATION_SUITE=checkpoint scripts/05_aspire2a_integration.pbs

# 9. Startup-only
qsub -v M3D_STARTUP_ONLY=1 scripts/06_train_aspire2a.pbs

# 10. 正式训练
qsub scripts/06_train_aspire2a.pbs

# 11. 断点续训
qsub -v M3D_RESUME_FROM=latest scripts/06_train_aspire2a.pbs

# 12. 导出
qsub -v M3D_CHECKPOINT=/absolute/path/to/training-output scripts/08_export_aspire2a.pbs

# 13. 推理
qsub -v M3D_EXPORT_DIR=/absolute/path/to/export,M3D_IMAGE=/absolute/path/to/ct.nii.gz,M3D_QUESTION_FILE=/absolute/path/to/question.txt scripts/09_inference_aspire2a.pbs

# 14. 评估
qsub -v M3D_EXPORT_DIR=/absolute/path/to/export scripts/07_evaluate_aspire2a.pbs
```
