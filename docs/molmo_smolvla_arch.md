# Molmo server + SmolVLA desktop split

Date: 2026-05-14
Last updated: 2026-05-14 (implementation complete)

## Goal

Run Molmo (AmbRes) on a server and SmolVLA (action model) on the desktop (VRAM 8GB).
The server handles ambiguity resolution; the desktop handles low-latency action generation.

---

## Architecture overview

```
Desktop (RTX 3060 / 8 GB VRAM)          GPU Server (RunPod)
────────────────────────────────         ───────────────────────────
InferencePipeline                        GenericInferenceService
  │                                        │
  ├─ GenericClient ──── gRPC ────────────► AmbResHandler (Molmo 7B)
  │    query / respond                      session-based multi-turn
  │
  └─ SmolVLAModel (local)
       predict_action()  ◄── 30 Hz ──── RobotConnector
       bfloat16, ~8 GB                    SO-ARM100/101
```

Network latency only affects episode_start (one-shot AmbRes call).
The 30 Hz action loop runs entirely on the desktop.

---

## Implemented files

### New files

| File | Purpose |
|------|---------|
| `module/models/smolvla/model.py` | SmolVLA local model wrapper |
| `module/models/smolvla/__init__.py` | Package export |
| `module/config/models/smolvla.yaml` | SmolVLA model config (bfloat16, 8 GB) |
| `module/config/pipelines/ambres_smolvla.yaml` | Pipeline config: AmbRes on server + SmolVLA local |

### Modified files

| File | Change |
|------|--------|
| `module/utils/registry.py` | Added `"smolvla"` entry to `_BUILTIN_MODELS` |
| `module/desktop/pipeline.py` | Added `local_model_*` fields to `PipelineConfig`; local model path in `connect()` and action loop |
| `scripts/run_desktop.py` | Pipeline mode prints `local_model_type`; added `smolvla` to train choices |

---

## Runtime states — design vs implementation

### INIT
**Design:** Load SmolVLA, connect to Molmo server.

**Implementation (`InferencePipeline.connect()`):**
```python
# Always: connect GenericClient to server (for AmbRes)
self._generic = GenericClient(host, port, ...)
self._generic.connect()

# local_model_type set → load SmolVLA locally, skip InferenceClient
if self._cfg.local_model_type:
    self._local_model = ModelRegistry.build(
        self._cfg.local_model_type,   # "smolvla"
        self._cfg.local_model_config, # {model_id, precision, ...}
    )
    self._local_model.load_checkpoint(self._cfg.local_model_checkpoint or "")
    # empty checkpoint_path → SmolVLAPolicy.from_pretrained("lerobot/smolvla")
else:
    # legacy path: server-side action model via InferenceClient
    self._infer = InferenceClient(...)
```

### EPISODE_START
**Design:** SmolVLA reset(), capture observation, send image + task_text to Molmo.

**Implementation (`InferencePipeline.run()` episode loop):**
```python
session_id = f"ep_{ep}_{uuid.uuid4().hex[:6]}"
context = {"task_text": task_text}

# 1. Reset SmolVLA action buffer
if self._local_model is not None:
    self._local_model.reset()

# 2. Capture observation
obs = self._robot.get_observation()

# 3. Run episode_start handlers (AmbRes query)
for step in self._episode_start_steps:
    self._run_step(step, obs, context, session_id)
```

`_run_step` builds the gRPC payload via `input_map` and calls:
```python
self._generic.infer(
    handler_id="ambres",
    method="query",
    payload={"task_description": context["task_text"]},
    images=obs.images,
    session_id=session_id,
)
```
Result `task_objects` is joined (`join_list` transform) and stored as `context["task_text"]`.

### CLARIFY (conditional)
**Design:** If `task_ambiguous == True`, ask user; send response to Molmo; update task_text.

**Implementation (`_run_step` clarification flow):**
```python
if step.clarify_on and result.get("task_ambiguous"):
    prompt = result.get("clarifying_question", "Please clarify:")
    print(f"\n[pipeline:ambres] {prompt}")
    user_response = input("> ").strip()

    clarify_result = self._generic.infer(
        handler_id="ambres",
        method="respond",
        payload={"response": user_response},
        session_id=session_id,  # same session → multi-turn Molmo dialogue
    )
    # task_objects from respond() also joined → context["task_text"]
    self._apply_output(step, clarify_result, context)
```

### ACTION_LOOP
**Design:** Run SmolVLA with task_text + local observation; send actions.

**Implementation (30 Hz loop in `run()`):**
```python
for step_idx in range(max_steps):
    t0 = time.perf_counter()
    obs = self._robot.get_observation()

    if self._local_model is not None:
        obs_obj = Observation(
            images=obs.images,
            state=obs.state,
            task_text=context.get("task_text", ""),
            episode_id=ep,
            step=step_idx,
        )
        action = self._local_model.predict_action(obs_obj)
    else:
        # legacy: server-side action model
        action = self._infer.get_action(model_id=..., ...)

    self._robot.send_action(action)

    elapsed = time.perf_counter() - t0
    if dt - elapsed > 0:
        time.sleep(dt - elapsed)
```

### EPISODE_END
**Design:** Stop loop, reset buffers for next episode.

**Implementation:** Loop exits after `max_episode_steps` (default 500 = ~16 s at 30 Hz).
`SmolVLAModel.reset()` is called at the top of the next episode, which clears `_action_buf`
and calls `policy.reset()` if the policy exposes it.

---

## SmolVLA model — `module/models/smolvla/model.py`

**Class:** `SmolVLAModel(BaseLeRobotModel)`, registered as `"smolvla"`.

**Architecture (upstream):**
- SigLIP vision encoder + SmolLM2 language backbone
- ~500M parameters
- Flow-matching action head
- HuggingFace checkpoint: `lerobot/smolvla`

**Inference path:**
```
Observation.images  (HWC uint8 or float32)
  → permute (CHW) → bfloat16 tensor
  + Observation.state  → bfloat16 tensor
  + Observation.task_text → ["task string"]
  → SmolVLAPolicy.select_action(batch)
  → actions (1, action_horizon, action_dim) or (1, action_dim)
  → squeeze → float32 numpy → robot
```

**Chunk execution buffer:**
```python
# Only re-runs the model when buffer is exhausted
if self._action_buf is None or self._buf_idx >= self._action_horizon:
    self._action_buf = self._run_vla(observation)  # shape (action_horizon, action_dim)
    self._buf_idx = 0

action = self._action_buf[self._buf_idx]
self._buf_idx += 1
```

**Key config options:**

| Key | Default | Notes |
|-----|---------|-------|
| `model_id` | `"lerobot/smolvla"` | HuggingFace ID or local path |
| `device` | `"cuda"` | |
| `precision` | `"bfloat16"` | Halves VRAM vs float32 |
| `action_horizon` | `1` | Steps to execute per model call |
| `use_amp` | `true` | `torch.amp.autocast` |
| `lr` | `1e-4` | Used only if `train_step()` is called |

---

## PipelineConfig changes — `module/desktop/pipeline.py`

Three new fields added to `PipelineConfig`:

```python
action_model_id: str = ""           # server model (legacy path)
local_model_type: str = ""          # "smolvla" → local path
local_model_checkpoint: str = ""    # empty = HuggingFace download
local_model_config: Dict[str, Any] = field(default_factory=dict)
```

`action_model_id` is now optional (was required before).
Setting `local_model_type` skips `InferenceClient` entirely.

---

## Pipeline config — `module/config/pipelines/ambres_smolvla.yaml`

```yaml
local_model_type: smolvla
local_model_checkpoint: ""          # HF download
local_model_config:
  model_id: "lerobot/smolvla"
  device: cuda
  precision: bfloat16
  action_horizon: 1
  use_amp: true

fps: 30.0
max_episode_steps: 500              # ~16 s

pre_handlers:
  - handler_id: ambres
    method: query
    trigger: episode_start
    input_map:
      task_text: task_description
    output_map:
      task_objects: task_text
    output_transform: join_list     # ["banana", "drawer"] → "banana drawer"
    clarify_on: task_ambiguous
    clarify_prompt_key: clarifying_question
    clarify_method: respond
```

---

## Model registry — `module/utils/registry.py`

```python
_BUILTIN_MODELS = {
    ...
    "smolvla": "module.models.smolvla.model.SmolVLAModel",  # added
    ...
}
```

`ModelRegistry.auto_discover()` imports the module, executing `@ModelRegistry.register("smolvla")`.

---

## Open decisions — resolved

| Decision | Resolution |
|----------|-----------|
| SmolVLA checkpoint id | `lerobot/smolvla` (HuggingFace); overridable via `local_model_checkpoint` |
| Precision / VRAM | `bfloat16` by default; configurable per `precision` key |
| task_text formatting | `join_list` (space-separated): `["banana", "drawer"] → "banana drawer"` |
| Clarification UI | Terminal `input()` in `_run_step`; same mechanism as legacy Pi0 pipeline |
| session_id mapping | `f"ep_{episode_index}_{uuid4().hex[:6]}"` — unique per episode, consistent across AmbRes query/respond calls |

---

## How to run

```bash
# 1. Open SSH tunnel to server (keep terminal open)
python scripts/open_tunnel.py --env .env.runpod --auto-reconnect

# 2. Verify server connection (separate terminal)
python scripts/check_connection.py

# 3. Run AmbRes + SmolVLA pipeline
python scripts/run_desktop.py \
  pipeline \
  --pipeline-config module/config/pipelines/ambres_smolvla.yaml \
  --task "pick up the banana"

# Or with automatic tunnel bootstrap:
python scripts/run_desktop_with_tunnel.py \
  --env .env.runpod \
  pipeline \
  --pipeline-config module/config/pipelines/ambres_smolvla.yaml \
  --task "pick up the banana" \
  --n-episodes 3
```

---

## Data flow — code-level

```
RobotConnector.get_observation()
  └─ RawObservation { images: Dict[str, ndarray], state: ndarray }
        │
        ▼ (episode_start)
GenericClient.infer("ambres", "query", payload, images, session_id)
  └─ Server: AmbResHandler → Molmo 7B
  └─ Returns: { task_objects: [...], task_ambiguous: bool, clarifying_question: str }
        │
        ▼ (if task_ambiguous)
GenericClient.infer("ambres", "respond", {"response": user_input}, session_id)
  └─ Returns: { task_objects: [...] }  → joined → context["task_text"]
        │
        ▼ (action loop, 30 Hz)
SmolVLAModel.predict_action(Observation(images, state, task_text))
  └─ SmolVLAPolicy.select_action(batch)  [local GPU, bfloat16]
  └─ Returns: ndarray (action_dim,)
        │
        ▼
RobotConnector.send_action(action)
```

---

## Risks / notes

- Molmo requires large GPU memory; keep it server-side. AmbRes handler is auto-loaded from `module/config/server.yaml` at server start.
- Network latency affects only the one-shot ambiguity resolution at episode start, not the action loop.
- `session_id` is consistent across query and respond within the same episode (Molmo multi-turn dialogue requires this).
- SmolVLA first run downloads weights from HuggingFace. Pre-cache with: `SmolVLAPolicy.from_pretrained("lerobot/smolvla")`.
- `bfloat16` reduces VRAM by ~50% vs float32. If RTX 3060 VRAM is insufficient, reduce `action_horizon` or switch cameras to lower resolution.
