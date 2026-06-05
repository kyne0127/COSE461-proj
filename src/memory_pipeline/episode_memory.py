"""
src.memory_pipeline.episode_memory
====================================
Episode-scoped memory for dynamic ambiguity tracking.

One EpisodeMemory is created at t₀ and accumulates CheckpointRecords as
the robot moves through its states.  This accumulated history is what
separates the memory-augmented pipeline from static single-shot AmbRes:
each AmbRes call receives prior observations and decisions as context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Per-object snapshot at one timestep
# ---------------------------------------------------------------------------

@dataclass
class ObjectState:
    """Detected state of one object label at a single timestep."""
    label:    str
    coord:    list[int] | None      # [x, y] of highest-confidence detection
    coords:   list[list[int]]       # all detected instances
    detected: bool

    @classmethod
    def from_detection(cls, label: str, detections: dict[str, Any]) -> "ObjectState":
        raw = detections.get(label) or []
        valid = [c for c in raw if isinstance(c, (list, tuple)) and len(c) == 2]
        return cls(
            label=label,
            coord=[int(valid[0][0]), int(valid[0][1])] if valid else None,
            coords=[[int(c[0]), int(c[1])] for c in valid],
            detected=bool(valid),
        )

    def instance_count(self) -> int:
        return len(self.coords)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label":    self.label,
            "coord":    self.coord,
            "coords":   self.coords,
            "detected": self.detected,
        }


# ---------------------------------------------------------------------------
# Full record of one checkpoint
# ---------------------------------------------------------------------------

@dataclass
class CheckpointRecord:
    """Everything observed and decided at one checkpoint."""
    step:            str                 # "t0" | "trigger_N" | ...
    target:          ObjectState         # specific-label detections
    destination:     ObjectState         # specific-label detections
    decision:        str = ""            # "CONTINUE" | "ASK" | "STOP" | "" (t0)
    grounding_state: str = ""            # GroundingState.value
    explanation:     str = ""            # AmbRes STEP2 ambiguity reasoning
    user_response:   str = ""            # clarification from user (ASK path)
    trigger_score:   float = 0.0         # change score that fired the trigger
    changes:         list[str] = field(default_factory=list)
    # Generic-label detections for same-category object detection
    target_generic:  ObjectState | None = None
    dest_generic:    ObjectState | None = None
    # Parallel FSM states at time of recording
    decision_state:  str = ""            # DecisionState.name
    phase_state:     str = ""            # PhaseState.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "step":            self.step,
            "target":          self.target.to_dict(),
            "destination":     self.destination.to_dict(),
            "decision":        self.decision,
            "grounding_state": self.grounding_state,
            "explanation":     self.explanation,
            "user_response":   self.user_response,
            "trigger_score":   self.trigger_score,
            "changes":         self.changes,
            "target_generic":  self.target_generic.to_dict() if self.target_generic else None,
            "dest_generic":    self.dest_generic.to_dict()   if self.dest_generic   else None,
            "decision_state":  self.decision_state,
            "phase_state":     self.phase_state,
        }


# ---------------------------------------------------------------------------
# Episode-level memory
# ---------------------------------------------------------------------------

class EpisodeMemory:
    """
    Accumulates object states and decisions across one robot episode.

    Usage:
        memory = EpisodeMemory(task)
        memory.record(t0_record)
        # ... robot executes ...
        memory.record(c1_record)  # after C1 check
        memory.record(c2_record)  # after C2 check
    """

    def __init__(self, task_description: str) -> None:
        self.task = task_description
        self._records: list[CheckpointRecord] = []

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def record(self, rec: CheckpointRecord) -> None:
        self._records.append(rec)

    def update_last_decision(
        self,
        decision: str,
        grounding_state: str = "",
        user_response: str = "",
    ) -> None:
        """Set decision fields on the most recently recorded checkpoint."""
        if self._records:
            last = self._records[-1]
            last.decision = decision
            last.grounding_state = grounding_state
            last.user_response = user_response

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    @property
    def records(self) -> list[CheckpointRecord]:
        return list(self._records)

    def get(self, step: str) -> CheckpointRecord | None:
        for r in reversed(self._records):
            if r.step == step:
                return r
        return None

    def last_record(self) -> CheckpointRecord | None:
        return self._records[-1] if self._records else None

    def t0(self) -> CheckpointRecord | None:
        return self.get("t0")

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "task":    self.task,
            "records": [r.to_dict() for r in self._records],
        }
