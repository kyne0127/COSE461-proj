"""
src.trigger.state_machine
==========================
FSM-based AmbRes pipeline with trigger-driven checkpoint detection.

Replaces the fixed-step C1/C2 design in src/pipeline.py with an
event-driven loop.  The machine advances through phases only when
a TriggerSignal fires, then calls the existing check_grounding /
extract_g0 logic from the monitoring and extraction modules.

State diagram:
                   start()
    IDLE ──────────────────► GROUNDING
                                 │  g0 extracted
                                 ▼
                           PRE_PICK ◄── reset after ASK
                                 │  c1_trigger fires
                                 ▼
                          CHECKING_C1
                                 │  CONTINUE / ASK resolved
                                 ▼
                            PICKING
                                 │  c2_trigger fires
                                 ▼
                          CHECKING_C2
                                 │  CONTINUE / ASK resolved
                                 ▼
                           PRE_PLACE
                                 │  (no further trigger)
                                 ▼
                            COMPLETE
                  (any STOP decision)
                            STOPPED

ASK path: the FSM calls `user_response_fn(question, g0)`, waits for
the response, runs a rolling G₀ update (extract_g0 on checkpoint image),
then returns to the current monitoring phase with updated G₀.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from monitoring.consistency_monitor import Decision, GroundingState, check_grounding
from trigger.signals import RobotState, TriggerSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------

class Phase(Enum):
    IDLE         = auto()   # before start()
    GROUNDING    = auto()   # G₀ extraction in progress
    PRE_PICK     = auto()   # monitoring for C1 trigger
    CHECKING_C1  = auto()   # AmbResVLM check at C1
    PICKING      = auto()   # between C1 and C2 trigger
    CHECKING_C2  = auto()   # AmbResVLM check at C2
    PRE_PLACE    = auto()   # between C2 check and COMPLETE
    COMPLETE     = auto()   # pipeline succeeded
    STOPPED      = auto()   # hard STOP issued


# ---------------------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """One phase transition event."""
    from_phase:    Phase
    to_phase:      Phase
    trigger_score: float = 0.0
    decision:      Decision | None = None
    state_label:   GroundingState | None = None
    user_response: str = ""
    question:      str = ""


# ---------------------------------------------------------------------------
# State machine result
# ---------------------------------------------------------------------------

@dataclass
class FSMResult:
    """Full execution record of one episode."""
    final_phase:  Phase
    g0_initial:   dict[str, Any] | None
    g0_final:     dict[str, Any] | None
    transitions:  list[Transition] = field(default_factory=list)
    stop_reason:  str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_phase": self.final_phase.name,
            "g0_initial":  self.g0_initial,
            "g0_final":    self.g0_final,
            "stop_reason": self.stop_reason,
            "transitions": [
                {
                    "from":      t.from_phase.name,
                    "to":        t.to_phase.name,
                    "trigger_score": t.trigger_score,
                    "decision":  t.decision.value if t.decision else None,
                    "state":     t.state_label.value if t.state_label else None,
                    "question":  t.question,
                    "user_response": t.user_response,
                }
                for t in self.transitions
            ],
        }


# ---------------------------------------------------------------------------
# AmbResStateMachine
# ---------------------------------------------------------------------------

class AmbResStateMachine:
    """
    Trigger-driven AmbRes FSM for one robot episode.

    Typical usage:
        fsm = AmbResStateMachine(
            image_t0=...,
            image_c1=...,
            image_c2=...,
            task_description=...,
            c1_trigger=DINOv2Trigger(monitor, phase="pre_pick"),
            c2_trigger=CompositeTrigger([ForceTrigger(5.0), StepTrigger(400)]),
            handler=ambres_handler,
            threshold=50.0,
        )
        for robot_state in robot_step_generator():
            result = fsm.step(robot_state)
            if result is not None:   # episode ended
                break

    Args:
        image_t0:         Path to frame at t₀ (used for G₀ extraction).
        image_c1:         Path to frame at C1 (used for consistency check).
        image_c2:         Path to frame at C2 (used for consistency check).
        task_description: Natural-language task instruction.
        c1_trigger:       Signal that fires when C1 check should be invoked.
        c2_trigger:       Signal that fires when C2 check should be invoked.
        handler:          Pre-built AmbResHandler.
        threshold:        Pixel distance threshold for check_grounding().
        user_response_fn: Called on ASK decisions.
                          Signature: (question: str, g0: dict) -> str.
                          Return "" to skip rolling G₀ update.
        max_triggers:     Max AmbResVLM calls allowed per episode (§4.4).
        session_prefix:   Prefix for AmbRes session IDs.
    """

    def __init__(
        self,
        image_t0: Any,
        image_c1: Any,
        image_c2: Any,
        task_description: str,
        c1_trigger: TriggerSignal,
        c2_trigger: TriggerSignal,
        handler: Any,
        *,
        threshold: float = 50.0,
        user_response_fn: Callable[[str, dict], str] | None = None,
        max_triggers: int = 3,
        session_prefix: str = "fsm",
    ) -> None:
        self._image_t0 = image_t0
        self._image_c1 = image_c1
        self._image_c2 = image_c2
        self._task = task_description
        self._c1_trigger = c1_trigger
        self._c2_trigger = c2_trigger
        self._handler = handler
        self._threshold = threshold
        self._user_response_fn = user_response_fn or (lambda q, g: "")
        self._max_triggers = max_triggers
        self._session_prefix = session_prefix

        self._phase = Phase.IDLE
        self._g0: dict[str, Any] | None = None
        self._trigger_count = 0
        self._transitions: list[Transition] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def phase(self) -> Phase:
        return self._phase

    def start(self) -> None:
        """Extract G₀ and move to PRE_PICK.  Call before the action loop."""
        self._phase = Phase.GROUNDING
        self._g0 = self._extract_g0(self._image_t0, session_id=f"{self._session_prefix}_t0")
        self._transition(Phase.PRE_PICK, Transition(Phase.GROUNDING, Phase.PRE_PICK))
        logger.info("FSM: G₀ extracted — entering PRE_PICK")

    def step(self, state: RobotState) -> FSMResult | None:
        """
        Advance the FSM by one step.

        Args:
            state: Current robot + sensor snapshot.

        Returns:
            FSMResult when the episode ends (COMPLETE or STOPPED), else None.
        """
        if self._phase == Phase.PRE_PICK:
            if self._c1_trigger.check(state):
                return self._run_check("C1", state)

        elif self._phase == Phase.PICKING:
            if self._c2_trigger.check(state):
                return self._run_check("C2", state)

        elif self._phase in (Phase.COMPLETE, Phase.STOPPED):
            return self._result()

        return None

    # ------------------------------------------------------------------ #
    # Internal: checkpoint execution
    # ------------------------------------------------------------------ #

    def _run_check(self, checkpoint: str, state: RobotState) -> FSMResult | None:
        """Execute AmbRes consistency check at C1 or C2."""
        if self._trigger_count >= self._max_triggers:
            logger.warning(
                "FSM: max_triggers (%d) exceeded — falling back to CONTINUE",
                self._max_triggers,
            )
            next_phase = Phase.PICKING if checkpoint == "C1" else Phase.PRE_PLACE
            self._transition(next_phase, Transition(self._phase, next_phase))
            return None

        checking_phase = Phase.CHECKING_C1 if checkpoint == "C1" else Phase.CHECKING_C2
        self._transition(checking_phase, Transition(self._phase, checking_phase))
        self._trigger_count += 1

        image = self._image_c1 if checkpoint == "C1" else self._image_c2
        session_id = f"{self._session_prefix}_{checkpoint.lower()}"

        det = self._get_checkpoint_detections(image, session_id)
        state_label, decision = check_grounding(
            self._g0, det, checkpoint, self._threshold
        )
        logger.info("FSM %s: state=%s decision=%s", checkpoint, state_label, decision)

        question = ""
        user_response = ""

        if decision == Decision.STOP:
            t = Transition(
                checking_phase, Phase.STOPPED,
                decision=decision, state_label=state_label,
            )
            self._transition(Phase.STOPPED, t)
            self._transitions[-1].decision = decision
            return self._result(stop_reason=f"{checkpoint} {state_label.value} → STOP")

        if decision == Decision.ASK:
            question = self._build_question(checkpoint, state_label)
            user_response = self._user_response_fn(question, self._g0)
            if user_response:
                self._g0 = self._extract_g0(
                    image,
                    user_response=user_response,
                    session_id=f"{self._session_prefix}_{checkpoint.lower()}_update",
                )
                logger.info("FSM: G₀ updated after user response at %s", checkpoint)

        next_phase = Phase.PICKING if checkpoint == "C1" else Phase.PRE_PLACE
        t = Transition(
            checking_phase, next_phase,
            decision=decision, state_label=state_label,
            question=question, user_response=user_response,
        )
        self._transition(next_phase, t)

        if next_phase == Phase.PRE_PLACE:
            self._transition(Phase.COMPLETE, Transition(Phase.PRE_PLACE, Phase.COMPLETE))
            return self._result()

        return None

    # ------------------------------------------------------------------ #
    # Internal: helpers
    # ------------------------------------------------------------------ #

    def _transition(self, new_phase: Phase, record: Transition) -> None:
        self._phase = new_phase
        self._transitions.append(record)

    def _result(self, stop_reason: str = "") -> FSMResult:
        return FSMResult(
            final_phase=self._phase,
            g0_initial=None,    # not stored separately; g0 starts at initial value
            g0_final=self._g0,
            transitions=list(self._transitions),
            stop_reason=stop_reason,
        )

    def _build_question(self, checkpoint: str, state: GroundingState) -> str:
        if self._g0 is None:
            return f"[{checkpoint} {state.value}] Please clarify the task."
        if checkpoint == "C1":
            label = (self._g0.get("target") or {}).get("label", "target object")
            return f"[C1 {state.value}] 어떤 '{label}'을(를) 집어야 하나요?"
        else:
            label = (self._g0.get("destination") or {}).get("label", "destination")
            return f"[C2 {state.value}] 어떤 '{label}'에 놓아야 하나요?"

    def _extract_g0(
        self,
        image_path: Any,
        user_response: str = "",
        session_id: str = "fsm_g0",
    ) -> dict[str, Any]:
        """Run AmbRes extract_g0 (initial) or rolling update (with user_response)."""
        from extraction.ambres_g0_extractor import (
            _first_coord,
            _load_image,
            _object_list_from_step,
            _unwrap_handler_result,
        )
        from extraction.role_parser import parse_roles

        image_arr, image_shape = _load_image(image_path)
        tensors = [image_arr]

        _unwrap_handler_result(
            self._handler.handle("reset", {}, [], session_id), "reset"
        )
        step1 = _unwrap_handler_result(
            self._handler.handle(
                "query", {"task_description": self._task}, tensors, session_id
            ),
            "query",
        )
        step2 = _unwrap_handler_result(
            self._handler.handle(
                "respond", {"response": user_response}, [], session_id
            ),
            "respond",
        )

        object_list = _object_list_from_step(step2) or _object_list_from_step(step1)
        roles = parse_roles(object_list, self._task)
        labels = [roles["target"], roles["destination"]]

        detections = _unwrap_handler_result(
            self._handler.handle("detect", {"objects": labels}, tensors, session_id),
            "detect",
        )["detections"]

        return {
            "target": {
                "label": roles["target"],
                "coord": _first_coord(detections, roles["target"]),
            },
            "destination": {
                "label": roles["destination"],
                "coord": _first_coord(detections, roles["destination"]),
            },
            "image_shape": image_shape,
        }

    def _get_checkpoint_detections(
        self, image_path: Any, session_id: str
    ) -> dict[str, list]:
        """Re-detect G₀ objects in the checkpoint frame."""
        from monitoring.consistency_monitor import get_checkpoint_detections

        return get_checkpoint_detections(
            image_path, self._g0, self._handler, session_id=session_id
        )
