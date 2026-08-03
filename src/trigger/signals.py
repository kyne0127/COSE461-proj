"""
src.trigger.signals
===================
Trigger signal abstractions for the FSM-based AmbRes pipeline.

Each TriggerSignal answers one question: "has the condition to invoke
AmbResVLM been met?"  Concrete signals range from simple step-count
thresholds (for debugging / fallback) to physics-based cues
(joint goal reached, gripper force spike) and vision-based cues
(DINOv2 scene-change score).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Robot state carrier
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    """Snapshot of robot + sensor state at one timestep."""
    step:          int                    = 0
    joints:        list[float]            = field(default_factory=list)
    gripper_force: float                  = 0.0
    gripper_width: float                  = 0.0
    frame:         Any                    = None   # np.ndarray (H,W,3), uint8 or float


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TriggerSignal(ABC):
    """
    A stateful condition that fires at most once per episode unless reset().

    Implementers must:
      - Return True from check() when the condition is first met.
      - Remain True on subsequent check() calls until reset().
      - Be idempotent after firing (no repeated side-effects).
    """

    def __init__(self) -> None:
        self._fired = False

    @abstractmethod
    def _condition(self, state: RobotState) -> bool:
        """Return True when the trigger condition is satisfied."""

    def check(self, state: RobotState) -> bool:
        if self._fired:
            return True
        result = self._condition(state)
        if result:
            self._fired = True
        return result

    def reset(self) -> None:
        self._fired = False

    @property
    def fired(self) -> bool:
        return self._fired


# ---------------------------------------------------------------------------
# Concrete signals
# ---------------------------------------------------------------------------

class StepTrigger(TriggerSignal):
    """Fires when `state.step >= target_step`.  Fallback for environments
    without physical sensors; mirrors the existing fixed-checkpoint design."""

    def __init__(self, target_step: int) -> None:
        super().__init__()
        self.target_step = target_step

    def _condition(self, state: RobotState) -> bool:
        return state.step >= self.target_step

    def __repr__(self) -> str:
        return f"StepTrigger(target={self.target_step})"


class JointGoalTrigger(TriggerSignal):
    """Fires when the robot's joint configuration is within `tol` radians
    of a target configuration on all joints.

    Useful for detecting when the end-effector has reached the pre-grasp
    pose (C1 trigger) in a real-robot deployment.
    """

    def __init__(self, goal_joints: list[float], tol: float = 0.05) -> None:
        super().__init__()
        self.goal_joints = goal_joints
        self.tol = tol

    def _condition(self, state: RobotState) -> bool:
        if len(state.joints) != len(self.goal_joints):
            return False
        return all(
            abs(j - g) <= self.tol
            for j, g in zip(state.joints, self.goal_joints)
        )

    def __repr__(self) -> str:
        return f"JointGoalTrigger(tol={self.tol})"


class ForceTrigger(TriggerSignal):
    """Fires when `state.gripper_force` exceeds `threshold`.

    Detects a successful grasp event (C1→C2 transition) without relying
    on fixed step counts.
    """

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def _condition(self, state: RobotState) -> bool:
        return state.gripper_force >= self.threshold

    def __repr__(self) -> str:
        return f"ForceTrigger(threshold={self.threshold})"


class DINOv2Trigger(TriggerSignal):
    """Fires when DINOv2-based scene-change score exceeds `threshold`.

    Delegates score computation to an external DINOv2Monitor instance so
    that feature storage and phase logic live in one place.  The trigger
    simply wraps the monitor's compute_score() call.

    Args:
        monitor:   DINOv2Monitor instance (already initialised with G₀ features).
        phase:     Which monitoring phase to pass to compute_score().
        threshold: Score above which the trigger fires (default 0.15).
    """

    def __init__(
        self,
        monitor: Any,
        phase: Literal["pre_pick", "pre_place"],
        threshold: float = 0.15,
    ) -> None:
        super().__init__()
        self._monitor = monitor
        self._phase = phase
        self._threshold = threshold
        self._last_score: float = 0.0

    def _condition(self, state: RobotState) -> bool:
        if state.frame is None:
            return False
        self._last_score = self._monitor.compute_score(state.frame, self._phase)
        return self._last_score > self._threshold

    @property
    def last_score(self) -> float:
        return self._last_score

    def __repr__(self) -> str:
        return (
            f"DINOv2Trigger(phase={self._phase!r}, "
            f"threshold={self._threshold}, last={self._last_score:.4f})"
        )


class CompositeTrigger(TriggerSignal):
    """Combines multiple TriggerSignals with AND / OR logic.

    Useful for requiring both a joint-goal AND a minimum step count, or
    for falling back to a step-count trigger when sensor data is absent.
    """

    def __init__(
        self,
        signals: list[TriggerSignal],
        mode: Literal["any", "all"] = "any",
    ) -> None:
        super().__init__()
        self._signals = signals
        self._mode = mode

    def _condition(self, state: RobotState) -> bool:
        results = [s.check(state) for s in self._signals]
        return any(results) if self._mode == "any" else all(results)

    def reset(self) -> None:
        super().reset()
        for s in self._signals:
            s.reset()

    def __repr__(self) -> str:
        return f"CompositeTrigger(mode={self._mode!r}, n={len(self._signals)})"
