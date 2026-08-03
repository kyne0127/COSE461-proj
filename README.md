# AmbResVLM — Execution-Aware Ambiguity Handling for Language-Guided Robotic Manipulation

COSE461 project. AmbResVLM judges whether a scene is ambiguous at the moment a command is issued, but it can't
detect scene changes that invalidate an initially-clear grounding *during* execution. This project stores the
initial grounding G₀ in memory and re-invokes AmbResVLM at execution checkpoints, classifying any mismatch
between G₀ and Gₜ as a grounding-state transition (CONTINUE / ASK / STOP).

See [docs/proposal.md](docs/proposal.md) for the full research proposal and [docs/PROGRESS.md](docs/PROGRESS.md)
for the current status.

## Layout

- `module/` — desktop ↔ GPU-server remote training/inference system (gRPC), robot connector, model plugins
  (ACT, Diffusion, TD-MPC2, Pi0, SmolVLA, custom). See [docs/project_spec.md](docs/project_spec.md).
- `src/` — grounding-state pipeline: `memory_pipeline/` (episode memory, checkpoint monitors, runtime FSM),
  `checkpoint_monitors/` (baseline ablations B1–B5), `trigger/` (scene-change trigger signals), `finetune/`,
  `baselines/`, `evaluate.py`.
- `scripts/` — dataset collection/annotation, Grounding DINO detection, manifest building, evaluation and
  visualization CLIs.
- `dataset/` — manifests (`manifest.jsonl`, `manifest_eval.jsonl`, `manifest_train.jsonl`) and CVAT
  annotation tooling. Underlying images/videos are hosted on the Hugging Face Hub (see below), not in git.
- `docs/` — proposal, architecture, dataset/finetuning guides, experiment results.
- `tests/` — unit + integration tests (`run_tests.sh`, `run_integration.sh`).

## Data & checkpoints

Raw captures and evaluation data are too large for git and live on the Hugging Face Hub instead:

- [`kyne0127/vla-evaluation`](https://huggingface.co/datasets/kyne0127/vla-evaluation) — evaluation scene captures
- [`kyne0127/ambres-training`](https://huggingface.co/datasets/kyne0127/ambres-training) — training data for the
  grounding-state classifier

Local directories such as `data/`, `data-evaluation*/`, `data-training*/` and `checkpoints/` are gitignored
working copies of the above (or local model checkpoints) — pull them via `huggingface_hub` rather than committing
them.

## Setup

See [docs/setup_and_workflow.md](docs/setup_and_workflow.md) (local setup) and
[docs/runpod_connection.md](docs/runpod_connection.md) (remote GPU training on RunPod).

```bash
./run_tests.sh          # unit tests
./run_integration.sh    # integration tests
```
