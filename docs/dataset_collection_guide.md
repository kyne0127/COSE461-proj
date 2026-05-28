# Dataset Collection Guide — Evaluation Dataset (S1~S7)

> This guide covers image dataset collection for evaluating B1~B5+Ours pipeline methods.
> Not for SmolVLA training demos — **evaluation images only**.

---

## 1. Physical Objects

| Role | Object | Count |
|------|--------|-------|
| Target | Paper cup (identical) | 2 |
| Target | Cube (identical) | 2 |
| Destination | Red box | 1 |
| Destination | Yellow box | 1 |

---

## 2. Recording Tool

```bash
python scripts/record_scenario.py \
    --scenario <S1~S7> \
    --task "<task description>" \
    --target-label <cup|cube> \
    --destination-label <"red box"|"yellow box"|box> \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl \
    --append-manifest
```

| Key | Action |
|-----|--------|
| `SPACE` | Capture t0 (initial scene) + start recording |
| `1` | Capture C1 checkpoint |
| `2` | Capture C2 checkpoint |
| `r` | Restart current trial |
| `q` | Save and quit |

---

## 3. Scenario Setup

### Common Initial State (t0)

All scenarios start with a **clear initial scene**:
1 target + 1 destination, placed without overlap.

---

### S1 — No change

- **Gold**: `CLEAR` → `CONTINUE`
- **Checkpoint**: C1

```
t0  cup x1 + red box x1
    ↓  (no intervention)
C1  cup x1 + red box x1  (identical)
C2  cup x1 + red box x1  (identical)
```

**Cup (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S1 --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

**Cube (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S1 --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

---

### S2 — target 1 → 2 (add identical)

- **Gold**: `AMBIGUOUS_TARGET` → `ASK`
- **Checkpoint**: C1

```
t0  cup x1 (pos A) + red box x1
    ↓  [Before C1: add identical cup at pos B]
C1  cup x2 + red box x1
C2  cup x2 + red box x1
```

> Place the two cups far enough apart to appear as separate objects.

**Cup (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S2 --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

**Cube (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S2 --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

---

### S3 — target 1 → 0 (removed)

- **Gold**: `INVALID_TARGET` → `STOP`
- **Checkpoint**: C1

```
t0  cup x1 + red box x1
    ↓  [Before C1: remove cup from scene]
C1  cup x0 + red box x1
C2  cup x0 + red box x1  ← same scene as C1
```

**Cup (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S3 --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

**Cube (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S3 --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

---

### S4 — target moved

- **Gold**: `AMBIGUOUS_TARGET` → `ASK`
- **Checkpoint**: C1

```
t0  cup x1 (pos A) + red box x1
    ↓  [Before C1: move cup from pos A to pos B]
C1  cup x1 (pos B) + red box x1
C2  cup x1 (pos B) + red box x1
```

> Move the cup far enough — at least across half the frame. Small movements may be classified as CLEAR.

**Cup (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S4 --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

**Cube (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S4 --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

---

### S5 — destination 1 → 2 (add identical)

- **Gold**: `AMBIGUOUS_DESTINATION` → `ASK`
- **Checkpoint**: C2

```
t0  cup x1 + red box x1
C1  cup x1 + red box x1  (no change at C1)
    ↓  [Before C2: add yellow box]
C2  cup x1 + red box x1 + yellow box x1
```

> Task must say **"box"** (no color). If "red box" is specified, adding yellow box does not create ambiguity.

**Cup (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S5 --task "pick the cup and put it in the box" \
    --target-label cup --destination-label box \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

**Cube (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S5 --task "pick the cube and put it in the box" \
    --target-label cube --destination-label box \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

---

### S6 — destination 1 → 0 (removed)

- **Gold**: `INVALID_DESTINATION` → `STOP`
- **Checkpoint**: C2

```
t0  cup x1 + red box x1
C1  cup x1 + red box x1  (no change at C1)
    ↓  [Before C2: remove red box from scene]
C2  cup x1 + red box x0
```

**Cup (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S6 --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

**Cube (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S6 --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

---

### S7 — destination moved

- **Gold**: `AMBIGUOUS_DESTINATION` → `ASK`
- **Checkpoint**: C2

```
t0  cup x1 + red box x1 (pos A)
C1  cup x1 + red box x1 (pos A)  (no change at C1)
    ↓  [Before C2: move red box from pos A to pos B]
C2  cup x1 + red box x1 (pos B)
```

> Move the box far enough — at least across half the frame.

**Cup (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S7 --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

**Cube (5 trials)**
```bash
python scripts/record_scenario.py \
    --scenario S7 --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --out-dir data-evaluation \
    --manifest-path dataset/manifest_eval.jsonl --append-manifest
```

---

## 4. Trial Plan

**10 trials per scenario (cup×5 + cube×5) = 70 total**

| Scenario | Gold State | Cup | Cube | Est. Time |
|----------|-----------|-----|------|-----------|
| S1 | CLEAR | 5 | 5 | 30 min |
| S2 | AMBIGUOUS_TARGET | 5 | 5 | 50 min |
| S3 | INVALID_TARGET | 5 | 5 | 40 min |
| S4 | AMBIGUOUS_TARGET | 5 | 5 | 40 min |
| S5 | AMBIGUOUS_DESTINATION | 5 | 5 | 50 min |
| S6 | INVALID_DESTINATION | 5 | 5 | 40 min |
| S7 | AMBIGUOUS_DESTINATION | 5 | 5 | 40 min |
| **Total** | | **35** | **35** | **~5 hrs** |

> Change object positions on every trial for meaningful visual variation.

---

## 5. Notes

- **S5 task wording**: must use `"box"` (no color). `"red box"` invalidates the S5 gold label.
- **S4 movement distance**: move target far enough — small displacements may be classified as CLEAR.
- **S7 movement distance**: same as S4 but for destination.
- **Position variation**: change object placement every trial.
- **Lighting/background**: keep consistent; minor variation is acceptable.

---

## 6. Post-collection Evaluation

```bash
# Validate manifest
python src/evaluate.py dataset/manifest_eval.jsonl --validate-only --check-images

# Run full evaluation
python src/evaluate.py dataset/manifest_eval.jsonl \
    --methods b1 b2 b3 b4 ours \
    --model-type finetune \
    --adapter-ckpt nFwD6qtf9T8dkJaQXU9vkW \
    --predictions-csv logs/predictions_real.csv \
    --metrics-json logs/metrics_real.json
```
