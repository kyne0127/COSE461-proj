"""
src.memory_pipeline.runtime_fsm
=================================
Parallel FSM for execution-aware ambiguity handling.

Two independent FSMs run in parallel:

  Decision FSM  — GDino-driven, bottom-up
    Continuously tracks intent-parsed objects (specific + generic labels).
    Detects object-level changes including same-category additions.
    Resolves ambiguity via AmbRes + EpisodeMemory context.
    States: MONITORING → CHECKING → CONTINUE/ASK/STOP

  Phase FSM  — Decision-driven, top-down gate
    Controls whether the robot may move.
    Transitions are driven entirely by Decision FSM output.
    States: MOVE ↔ STOP

Trigger mechanism (object-level, not whole-scene):
  specific labels  = intent-parsed target/destination
  generic labels   = core noun extracted from specific label
                     ("red cup" → "cup", "sprite bottle" → "bottle")

  Each frame:
    GDino.detect(specific) + GDino.detect(generic)
    detect_changes_enhanced(ref, curr)
      → standard changes  (appear/disappear/move/count on specific label)
      → same-category     (generic count ↑ while specific count unchanged)
    Any change → Decision FSM enters CHECKING → AmbRes → Decision
    Decision → Phase FSM update

State interactions:
  Decision   Phase
  MONITORING  MOVE    (normal execution)
  CHECKING    STOP    (pause while evaluating)
  ASKING      STOP    (pause while waiting for user)
  STOPPED     STOP    (terminal — hard stop)
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_ROOT.parent
_AMBRES_ROOT = _REPO_ROOT.parent / "AmbRes"
for _p in (_SRC_ROOT, _REPO_ROOT, _AMBRES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from memory_pipeline.context_builder import build_memory_context, inject_context
from memory_pipeline.episode_memory import CheckpointRecord, EpisodeMemory, ObjectState
from memory_pipeline.scene_monitor import SceneMonitor
from monitoring.consistency_monitor import (
    Decision, GroundingState,
    check_grounding,
    check_grounding_ensemble,
    get_checkpoint_detections_from_frame,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FSM state enums
# ---------------------------------------------------------------------------

class DecisionState(Enum):
    MONITORING = auto()   # tracking objects; last result was CONTINUE (or t0)
    CHECKING   = auto()   # change detected; AmbRes in progress
    ASKING     = auto()   # ASK decision; waiting for user response
    STOPPED    = auto()   # STOP decision; terminal


class PhaseState(Enum):
    MOVE = auto()   # robot may execute actions
    STOP = auto()   # robot must halt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generalize_label(label: str) -> str:
    """Extract core noun: 'red cup' → 'cup', 'sprite bottle' → 'bottle'."""
    words = label.strip().split()
    return words[-1] if len(words) > 1 else label


def _infer_checkpoint_type(changes: list[str]) -> str:
    """Infer C1 (target check) or C2 (destination check) from change list."""
    has_target = any(c.startswith("target") for c in changes)
    has_dest   = any(c.startswith("destination") for c in changes)
    if has_dest and not has_target:
        return "C2"
    return "C1"   # target change or mixed → target check first


def _to_pil(frame: np.ndarray) -> Image.Image:
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(frame)


def _to_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _open_image(src: np.ndarray | str | Path) -> tuple[Image.Image, np.ndarray]:
    if isinstance(src, (str, Path)):
        pil = Image.open(src).convert("RGB")
        arr = _to_float32(np.asarray(pil))
    else:
        arr = _to_float32(src)
        pil = _to_pil(src)
    return pil, arr


def load_video_frames(video_path: str | Path) -> list[np.ndarray]:
    """Load all frames from a video file as uint8 RGB arrays."""
    path = str(video_path)
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    except ImportError:
        pass
    try:
        import imageio
        return [np.asarray(f) for f in imageio.get_reader(path)]
    except ImportError:
        pass
    raise ImportError(
        "Video loading requires 'cv2' or 'imageio'. "
        "Install: pip install opencv-python  OR  pip install imageio[ffmpeg]"
    )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FSMStepResult:
    """Result of one trigger-fired check."""
    frame_idx:       int
    decision_state:  DecisionState
    phase_state:     PhaseState
    grounding_state: GroundingState
    decision:        Decision
    checkpoint_type: str              # "C1" | "C2"
    changes:         list[str]
    memory_context:  str = ""
    question:        str = ""
    user_response:   str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx":       self.frame_idx,
            "decision_state":  self.decision_state.name,
            "phase_state":     self.phase_state.name,
            "grounding_state": self.grounding_state.value,
            "decision":        self.decision.value,
            "checkpoint_type": self.checkpoint_type,
            "changes":         self.changes,
            "memory_context":  self.memory_context,
            "question":        self.question,
            "user_response":   self.user_response,
        }


@dataclass
class MemoryFSMResult:
    """Full episode result."""
    final_decision_state: DecisionState
    final_phase_state:    PhaseState
    memory:               EpisodeMemory
    triggered_steps:      list[FSMStepResult] = field(default_factory=list)
    stop_reason:          str = ""
    total_frames:         int = 0

    @property
    def c1(self) -> FSMStepResult | None:
        """First trigger that performed a target (C1) check."""
        return next((s for s in self.triggered_steps if s.checkpoint_type == "C1"), None)

    @property
    def c2(self) -> FSMStepResult | None:
        """First trigger that performed a destination (C2) check."""
        return next((s for s in self.triggered_steps if s.checkpoint_type == "C2"), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_decision_state": self.final_decision_state.name,
            "final_phase_state":    self.final_phase_state.name,
            "stop_reason":          self.stop_reason,
            "total_frames":         self.total_frames,
            "triggered_steps":      [s.to_dict() for s in self.triggered_steps],
            "memory":               self.memory.to_dict(),
        }


# ---------------------------------------------------------------------------
# MemoryFSM
# ---------------------------------------------------------------------------

class MemoryFSM:
    """
    Parallel FSM for execution-aware ambiguity handling.

    Usage — full video:
        fsm = MemoryFSM(task, target_label, dest_label, handler=h, gdino=g)
        result = fsm.run(frames, frame_t0=initial_frame)

    Usage — frame-by-frame:
        fsm.initialize(frame_t0)
        for idx, frame in enumerate(frames):
            step_result = fsm.step(frame, frame_idx=idx)
            # phase_state tells robot: MOVE or STOP
            robot.set_enabled(fsm.phase_state == PhaseState.MOVE)
            if fsm.decision_state == DecisionState.STOPPED:
                break
        result = fsm.result()

    Args:
        task_description:   Original user task string.
        target_label:       Specific GDino label for target (from intent parsing).
        destination_label:  Specific GDino label for destination.
        handler:            AmbRes handler instance.
        gdino:              GroundingDINO instance.
        threshold:          Pixel distance for check_grounding().
        use_memory_context: False → ablation (no memory injection).
        user_response_fn:   (question, g0) → str. Called on ASK.
        session_prefix:     AmbRes session ID prefix.
        min_frames_gap:     Min frames between consecutive trigger evaluations.
    """

    def __init__(
        self,
        task_description: str,
        target_label: str,
        destination_label: str,
        *,
        handler: Any,
        gdino: Any,
        threshold: float = 50.0,
        use_memory_context: bool = True,
        user_response_fn: Callable[[str, dict], str] | None = None,
        session_prefix: str = "fsm",
        min_frames_gap: int = 1,
    ) -> None:
        self._task    = task_description
        self._target  = target_label
        self._dest    = destination_label
        self._target_generic = _generalize_label(target_label)
        self._dest_generic   = _generalize_label(destination_label)

        self._handler   = handler
        self._gdino     = gdino
        self._threshold = threshold
        self._use_mem   = use_memory_context
        self._user_fn   = user_response_fn or (lambda q, g: "")
        self._prefix    = session_prefix
        self._min_gap   = min_frames_gap

        self._monitor = SceneMonitor(gdino, move_threshold_px=threshold)
        self._memory  = EpisodeMemory(task_description)

        # Parallel FSM states
        self._decision = DecisionState.MONITORING
        self._phase    = PhaseState.STOP     # STOP until initialize() called

        self._g0:              dict[str, Any] | None = None
        self._image_shape:     list[int] = []
        self._ref_record:      CheckpointRecord | None = None
        self._gdino_t0_det:    dict[str, Any] = {}   # stored at initialize() for ensemble
        self._frame_count   = 0
        self._last_trigger  = -999
        self._trigger_count = 0
        self._triggered_steps: list[FSMStepResult] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def decision_state(self) -> DecisionState:
        return self._decision

    @property
    def phase_state(self) -> PhaseState:
        return self._phase

    def initialize(self, frame_t0: np.ndarray | str | Path) -> None:
        """
        Process t0 frame: build G₀, seed EpisodeMemory, enter MONITORING/MOVE.
        Must be called before step().
        """
        pil_t0, _ = _open_image(frame_t0)
        W, H = pil_t0.size
        self._image_shape = [H, W]

        specific_det = self._gdino.detect(pil_t0, [self._target, self._dest])
        generic_det  = self._gdino.detect(pil_t0, [self._target_generic, self._dest_generic])
        self._gdino_t0_det = specific_det   # saved for check_grounding_ensemble at trigger time

        t0_record = self._monitor.snapshot_with_generic(
            specific_det, generic_det,
            self._target, self._dest,
            self._target_generic, self._dest_generic,
            "t0",
        )
        t0_record.decision_state = DecisionState.MONITORING.name
        t0_record.phase_state    = PhaseState.MOVE.name

        self._memory.record(t0_record)
        self._ref_record = t0_record

        self._g0 = {
            "target": {
                "label":  self._target,
                "coord":  t0_record.target.coord,
                "coords": t0_record.target.coords,
            },
            "destination": {
                "label":  self._dest,
                "coord":  t0_record.destination.coord,
                "coords": t0_record.destination.coords,
            },
            "image_shape": self._image_shape,
        }

        self._decision = DecisionState.MONITORING
        self._phase    = PhaseState.MOVE

        logger.info(
            "FSM init: target=%s@%s  dest=%s@%s  "
            "[generic: target=%s, dest=%s]",
            self._target, t0_record.target.coord,
            self._dest,   t0_record.destination.coord,
            self._target_generic, self._dest_generic,
        )

    def step(
        self,
        frame: np.ndarray | str | Path,
        frame_idx: int | None = None,
    ) -> FSMStepResult | None:
        """
        Process one frame.

        Returns FSMStepResult when a trigger fires (change detected + AmbRes called).
        Returns None while monitoring with no change.

        After each call, check:
          fsm.phase_state     → MOVE / STOP for robot control
          fsm.decision_state  → current resolution state
        """
        if self._decision == DecisionState.STOPPED:
            return None

        idx = frame_idx if frame_idx is not None else self._frame_count
        self._frame_count += 1

        pil, arr = _open_image(frame)

        # ── 1. Detect with specific + generic labels ──────────────────────
        specific_det = self._gdino.detect(pil, [self._target, self._dest])
        generic_det  = self._gdino.detect(pil, [self._target_generic, self._dest_generic])

        curr_record = self._monitor.snapshot_with_generic(
            specific_det, generic_det,
            self._target, self._dest,
            self._target_generic, self._dest_generic,
            f"f{idx}",
        )

        # ── 2. Enhanced change detection (specific + same-category) ───────
        changes = self._monitor.detect_changes_enhanced(self._ref_record, curr_record)

        if not changes:
            return None

        # ── 3. Cooldown guard ─────────────────────────────────────────────
        if idx - self._last_trigger < self._min_gap:
            logger.debug("FSM: change at f%d within cooldown — skipped", idx)
            return None

        # ── 4. Decision FSM → CHECKING; Phase FSM → STOP ──────────────────
        self._decision = DecisionState.CHECKING
        self._phase    = PhaseState.STOP

        # ── 5. Infer checkpoint type from nature of change ─────────────────
        ck_type = _infer_checkpoint_type(changes)

        # ── 6. AmbRes + memory context → grounding check ──────────────────
        step_result = self._run_check(
            arr, specific_det, curr_record, changes, idx, ck_type
        )
        self._triggered_steps.append(step_result)
        self._last_trigger = idx
        self._ref_record   = curr_record

        # ── 7. Update parallel FSM states from Decision ───────────────────
        if step_result.decision == Decision.CONTINUE:
            self._decision = DecisionState.MONITORING
            self._phase    = PhaseState.MOVE
        elif step_result.decision == Decision.ASK:
            self._decision = DecisionState.ASKING
            self._phase    = PhaseState.STOP
            if step_result.user_response:
                # User responded — resolution complete, resume
                self._decision = DecisionState.MONITORING
                self._phase    = PhaseState.MOVE
        elif step_result.decision == Decision.STOP:
            self._decision = DecisionState.STOPPED
            self._phase    = PhaseState.STOP

        return step_result

    def run(
        self,
        frames: list[np.ndarray | str | Path],
        frame_t0: np.ndarray | str | Path | None = None,
    ) -> MemoryFSMResult:
        """Process a full frame sequence."""
        if not frames:
            return self._build_result(0)

        t0 = frame_t0 if frame_t0 is not None else frames[0]
        step_frames = frames if frame_t0 is not None else frames[1:]

        self.initialize(t0)
        for idx, frame in enumerate(step_frames):
            self.step(frame, frame_idx=idx)
            if self._decision == DecisionState.STOPPED:
                break

        return self._build_result(len(step_frames))

    def result(self, total_frames: int | None = None) -> MemoryFSMResult:
        return self._build_result(total_frames or self._frame_count)

    # ------------------------------------------------------------------ #
    # Internal: checkpoint execution
    # ------------------------------------------------------------------ #

    def _run_check(
        self,
        frame_arr: np.ndarray,
        curr_det: dict[str, Any],
        curr_record: CheckpointRecord,
        changes: list[str],
        frame_idx: int,
        ck_type: str,
    ) -> FSMStepResult:
        self._trigger_count += 1
        session_id = f"{self._prefix}_t{self._trigger_count}_f{frame_idx}"

        # ── Memory context ─────────────────────────────────────────────────
        context_str = (
            build_memory_context(self._memory, ck_type) if self._use_mem else ""
        )
        task_with_context = inject_context(self._task, context_str)

        # ── AmbRes: re-extract labels with memory-conditioned task ─────────
        try:
            tgt, dst, vlm_question = self._extract_labels(frame_arr, task_with_context, session_id)
            # Guard against hallucinated labels (memory context can confuse the VLM
            # into returning unrelated objects like 'banana').  Fall back when there
            # is no word overlap with the original task labels.
            def _has_word_overlap(a: str, b: str) -> bool:
                return bool(set(a.lower().split()) & set(b.lower().split()))

            if not _has_word_overlap(tgt, self._target):
                logger.warning(
                    "FSM trigger%d: VLM returned label '%s' has no overlap with '%s' "
                    "— reverting to original",
                    self._trigger_count, tgt, self._target,
                )
                tgt = self._target
                vlm_question = ""
            if not _has_word_overlap(dst, self._dest):
                logger.warning(
                    "FSM trigger%d: VLM returned dest '%s' has no overlap with '%s' "
                    "— reverting to original",
                    self._trigger_count, dst, self._dest,
                )
                dst = self._dest
        except RuntimeError as exc:
            logger.warning(
                "AmbRes failed (trigger %d, f%d): %s — using original labels",
                self._trigger_count, frame_idx, exc,
            )
            tgt, dst, vlm_question = self._target, self._dest, ""

        # ── Re-detect if labels changed ────────────────────────────────────
        pil = _to_pil(frame_arr)
        if tgt != self._target or dst != self._dest:
            curr_det = self._gdino.detect(pil, [tgt, dst])

        # ── Grounding check (heavy — trigger fired, worth calling Molmo) ──────
        # Molmo detection on the triggered frame (on-demand, not every frame).
        # Then run ensemble: GDino(t0) + GDino(ck) + Molmo(ck) → decision.
        try:
            molmo_det = get_checkpoint_detections_from_frame(
                frame_arr, self._g0, self._handler,
                session_id + "_det",
            )
        except Exception as exc:
            logger.warning("FSM trigger%d: Molmo detect failed (%s) — falling back to GDino only", self._trigger_count, exc)
            molmo_det = curr_det

        state, decision = check_grounding_ensemble(
            self._g0, molmo_det, self._gdino_t0_det, curr_det,
            ck_type, self._threshold,
        )
        logger.info(
            "FSM trigger%d (f%d, %s): state=%s  decision=%s  changes=%s",
            self._trigger_count, frame_idx, ck_type,
            state.value, decision.value, changes,
        )

        # ── ASK path ───────────────────────────────────────────────────────
        question = ""
        user_response = ""
        if decision == Decision.ASK:
            role  = "target" if ck_type == "C1" else "destination"
            label = (self._g0 or {}).get(role, {}).get("label", role)
            if vlm_question:
                # Use the VLM's own context-aware question (memory-conditioned)
                question = f"[{ck_type}] {vlm_question}"
                logger.info("FSM trigger%d: using VLM question: %s",
                            self._trigger_count, vlm_question)
            else:
                # Fallback template with change context
                change_desc = "; ".join(changes[:2]) if changes else ""
                if change_desc:
                    question = (
                        f"[{ck_type}] 씬 변화({change_desc})가 감지됐습니다. "
                        + (f"어떤 '{label}'을(를) 집어야 하나요?" if role == "target"
                           else f"어떤 '{label}'에 놓아야 하나요?")
                    )
                else:
                    question = (
                        f"[{ck_type}] 어떤 '{label}'을(를) 집어야 하나요?"
                        if role == "target"
                        else f"[{ck_type}] 어떤 '{label}'에 놓아야 하나요?"
                    )
                logger.info("FSM trigger%d: VLM returned no question; using fallback",
                            self._trigger_count)
            user_response = self._user_fn(question, self._g0)
            if user_response:
                self._g0 = self._clarify_g0(
                    pil, frame_arr, user_response, session_id
                )

        # ── Record in EpisodeMemory ────────────────────────────────────────
        resolved_decision = (
            DecisionState.ASKING   if decision == Decision.ASK   else
            DecisionState.STOPPED  if decision == Decision.STOP  else
            DecisionState.MONITORING
        )
        resolved_phase = (
            PhaseState.STOP if decision in (Decision.ASK, Decision.STOP)
            else PhaseState.MOVE
        )

        curr_record.step            = f"trigger_{self._trigger_count}"
        curr_record.decision        = decision.value
        curr_record.grounding_state = state.value
        curr_record.user_response   = user_response
        curr_record.changes         = changes
        curr_record.decision_state  = resolved_decision.name
        curr_record.phase_state     = resolved_phase.name
        self._memory.record(curr_record)

        return FSMStepResult(
            frame_idx=frame_idx,
            decision_state=resolved_decision,
            phase_state=resolved_phase,
            grounding_state=state,
            decision=decision,
            checkpoint_type=ck_type,
            changes=changes,
            memory_context=context_str,
            question=question,
            user_response=user_response,
        )

    def _clarify_g0(
        self,
        pil: Image.Image,
        frame_arr: np.ndarray,
        user_response: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Rebuild G₀ after user clarification."""
        clarify_ctx = inject_context(
            self._task,
            build_memory_context(self._memory, "clarify") if self._use_mem else "",
        )
        try:
            tgt_cl, dst_cl, _ = self._extract_labels(
                frame_arr,
                inject_context(clarify_ctx, f'User clarification: "{user_response}"'),
                f"{session_id}_clarify",
            )
            det_cl = self._gdino.detect(pil, [tgt_cl, dst_cl])
            return {
                "target":      {"label": tgt_cl, "coord": self._first_coord(det_cl, tgt_cl), "coords": self._all_coords(det_cl, tgt_cl)},
                "destination": {"label": dst_cl, "coord": self._first_coord(det_cl, dst_cl), "coords": self._all_coords(det_cl, dst_cl)},
                "image_shape": self._image_shape,
            }
        except RuntimeError as exc:
            logger.warning("Clarification AmbRes failed: %s — G₀ unchanged", exc)
            return self._g0

    def _build_result(self, total_frames: int) -> MemoryFSMResult:
        stop_reason = ""
        if self._decision == DecisionState.STOPPED and self._triggered_steps:
            last = self._triggered_steps[-1]
            stop_reason = f"{last.checkpoint_type} {last.grounding_state.value} → STOP"
        return MemoryFSMResult(
            final_decision_state=self._decision,
            final_phase_state=self._phase,
            memory=self._memory,
            triggered_steps=list(self._triggered_steps),
            stop_reason=stop_reason,
            total_frames=total_frames,
        )

    def _extract_labels(
        self,
        frame_arr: np.ndarray,
        task_with_context: str,
        session_id: str,
    ) -> tuple[str, str, str]:
        """Returns (target_label, destination_label, vlm_clarifying_question)."""
        from extraction.role_parser import parse_roles

        def _unwrap(r: dict, step: str) -> dict:
            if not r.get("success"):
                raise RuntimeError(f"AmbRes {step} failed: {r.get('error')}")
            return {k: v for k, v in r.items() if k != "success"}

        _unwrap(self._handler.handle("reset", {}, [], session_id), "reset")
        step1 = _unwrap(
            self._handler.handle(
                "query", {"task_description": task_with_context}, [frame_arr], session_id
            ), "query",
        )
        vlm_question: str = step1.get("clarifying_question", "") or ""
        step2 = _unwrap(
            self._handler.handle("respond", {"response": ""}, [], session_id),
            "respond",
        )
        obj_list = (
            step2.get("object_list") or step2.get("task_objects") or
            step1.get("object_list") or step1.get("task_objects") or []
        )
        roles = parse_roles(obj_list, task_with_context)
        return roles["target"], roles["destination"], vlm_question

    @staticmethod
    def _first_coord(det: dict, label: str) -> list[int] | None:
        raw = det.get(label) or []
        valid = [c for c in raw if isinstance(c, (list, tuple)) and len(c) == 2]
        return [int(valid[0][0]), int(valid[0][1])] if valid else None

    @staticmethod
    def _all_coords(det: dict, label: str) -> list[list[int]]:
        raw = det.get(label) or []
        return [[int(c[0]), int(c[1])] for c in raw
                if isinstance(c, (list, tuple)) and len(c) == 2]
