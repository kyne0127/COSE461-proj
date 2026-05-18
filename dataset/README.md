# Image Evaluation Dataset

This directory is for the image-only Phase 2 experiment.

The evaluator reads a manifest and runs B1~B5+Ours on static images. It does
not execute robot actions.

## Directory Layout

Recommended layout:

```text
dataset/
├── manifest.example.jsonl
├── manifest.jsonl              # create this after image collection
└── images/
    ├── s1_001_t0.png
    ├── s1_001_c1.png
    ├── s1_001_c2.png
    └── ...
```

## Manifest Fields

Each JSONL row is one evaluated checkpoint.

Required fields:

- `id`: unique sample id
- `scenario`: `S1`~`S6`
- `task`: natural language instruction
- `initial_img`: t0 image path, relative to manifest file or absolute
- `c1_img`: C1 checkpoint image path
- `c2_img`: C2 checkpoint image path
- `checkpoint`: checkpoint to score for this row, `C1` or `C2`
- `gold_state`: expected grounding state
- `gold_decision`: `CONTINUE`, `ASK`, or `STOP`
- `target_label`: full target object label used by B3
- `destination_label`: full destination label used by B3

Optional fields are preserved as sample metadata.

## Validate Without Models

This command checks manifest schema only. It does not open images or load
AmbRes/OpenAI models.

```bash
python src/evaluate.py dataset/manifest.example.jsonl --validate-only
```

## Run Evaluation

AmbRes-based methods require the real model environment.

```bash
python src/evaluate.py dataset/manifest.jsonl \
  --methods b1 b2 b3 b4 ours \
  --model-type finetune \
  --adapter-ckpt <AMBRES_CKPT> \
  --predictions-csv results/predictions.csv \
  --metrics-json results/metrics.json
```

B5 additionally requires `openai` and `OPENAI_API_KEY`.

```bash
OPENAI_API_KEY=... python src/evaluate.py dataset/manifest.jsonl \
  --methods b5 \
  --llm-model gpt-4o
```
