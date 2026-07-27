# Manifest smoke-test data

This fixture exercises every manifest builder on one training example per
source:

- M3D caption
- open and closed M3D VQA
- yes/no M3D VQA
- referring segmentation
- Decathlon positioning and generated segmentation

Run it from the repository root in the project Python 3.10 environment:

```bash
python -m m3d.data.manifest \
  --config configs/m3d_joint_finetune.yaml \
  --set "data.paths.data_root=${PWD}/dummy_data/manifest_smoke" \
  --splits train \
  --output-dir /tmp/m3d-manifest-smoke-output
```

The expected training manifest contains 11 logical records:

- caption: 1
- vqa_closed: 1
- vqa_open: 1
- vqa_yes_no: 1
- positioning: 4
- segmentation: 3

Use `--splits train` for this smoke test. The current manifest implementation
maps both validation and test to the same RefSeg/Decathlon test rows, so asking
for both splits together correctly triggers its overlap guard.
