# Molmo server + SmolVLA desktop split (design notes)

Date: 2026-05-14

## Goal
Run Molmo (AmbRes) on a server and SmolVLA (action model) on the desktop (VRAM 8GB). The server handles ambiguity resolution; the desktop handles low-latency action generation.

## Current codebase fit
- AmbRes handler already runs on server via GenericInferenceService.
- Desktop pipeline can call server handlers before action loop.
- Action model is invoked locally in the control loop (recommended).

Relevant modules:
- module/models/ambres/handler.py
- module/desktop/pipeline.py
- module/config/pipelines/ambres_pi0.yaml

## Recommended control location
Run the control process locally on the desktop:
- Low-latency action loop.
- Safe fallback if network stalls.
- Server only provides Molmo outputs (task objects, ambiguity flag, clarifying question).

## Suggested runtime states (desktop)
1) INIT
   - Load SmolVLA and set eval mode.
   - Connect to Molmo server.

2) EPISODE_START
   - Call SmolVLA reset().
   - Capture observation.
   - Send image + task text to Molmo (query).

3) CLARIFY (conditional)
   - If task_ambiguous == True, ask user.
   - Send user response to Molmo (respond).
   - Update task_text.

4) ACTION_LOOP
   - Run SmolVLA with task_text + local observation.
   - Send actions to robot.

5) EPISODE_END
   - Stop loop, reset buffers for next episode.

## SmolVLA behavior while Molmo runs
- Keep model loaded and idle.
- Do not generate actions until task_text is finalized.
- After task_text update, reset() before starting the loop.

## Data flow (high level)
Desktop -> Molmo server:
- image, task_description, session_id

Molmo server -> Desktop:
- task_objects, task_ambiguous, clarifying_question

Desktop -> SmolVLA:
- images + state + task_text

SmolVLA -> Desktop:
- action tensor

## Open decisions
- SmolVLA checkpoint id and precision (affects VRAM).
- Final task_text formatting (join_list vs first-object, etc.).
- Clarification UI (terminal input vs GUI).

## Risks / notes
- Molmo requires large GPU memory; keep it server-side.
- Network latency affects only ambiguity resolution, not action loop.
- Ensure session_id mapping is consistent per episode.
