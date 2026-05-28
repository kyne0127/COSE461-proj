"""src.trigger — trigger-based FSM for AmbRes pipeline."""
from trigger.dinov2_monitor import DINOv2Monitor, coord_to_bbox
from trigger.signals import (
    CompositeTrigger,
    DINOv2Trigger,
    ForceTrigger,
    JointGoalTrigger,
    RobotState,
    StepTrigger,
    TriggerSignal,
)
from trigger.state_machine import AmbResStateMachine, FSMResult, Phase, Transition

__all__ = [
    "RobotState",
    "TriggerSignal",
    "StepTrigger",
    "JointGoalTrigger",
    "ForceTrigger",
    "DINOv2Trigger",
    "CompositeTrigger",
    "DINOv2Monitor",
    "coord_to_bbox",
    "Phase",
    "Transition",
    "FSMResult",
    "AmbResStateMachine",
]
