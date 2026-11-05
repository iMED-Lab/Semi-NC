# Semi-NC

## Environment

Recommended environment: Python 3.9, PyTorch 1.13, and a CUDA-capable GPU.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test]"
```

## Data

Labeled CSV:

```csv
image,mask_0,mask_1,patient_id
images/0001.png,masks/cn/0001.png,masks/lc/0001.png,P001
```

Unlabeled CSV:

```csv
image
images/1001.png
```

Paths may be absolute or relative to the CSV file. By default, `mask_0` is the
corneal-nerve mask and `mask_1` is the Langerhans-cell mask.
`patient_id` is required when running the patient-cluster bootstrap.

## Training

Pretrain the teacher:

```bash
python scripts/pretrain_teacher.py --labeled-manifest data/labeled.csv
```

Train the student:

```bash
python scripts/train_student.py \
  --labeled-manifest data/labeled.csv \
  --unlabeled-manifest data/unlabeled.csv \
  --teacher checkpoints/teacher.pt
```

Resume training with `--resume checkpoints/student.pt`. Experiment settings are
defined in `configs/semi_nc.yaml`. Use the same `--seed` value for the teacher
and student of each run; repeat with five different seeds for the paper protocol.

## Evaluation

```bash
python scripts/evaluate.py \
  --manifest data/test.csv \
  --checkpoint checkpoints/student.pt \
  --output results/evaluation.json
```

Aggregate five independent runs:

```bash
python scripts/aggregate_runs.py \
  results/run1.json results/run2.json results/run3.json \
  results/run4.json results/run5.json
```

For a paired patient-cluster bootstrap against a baseline:

```bash
python scripts/paired_bootstrap.py \
  --ours results/ours.json --baseline results/baseline.json
```

## Verification

```bash
python scripts/smoke.py
python -m pytest -q
```
