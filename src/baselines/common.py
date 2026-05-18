"""Common result types for baseline methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from monitoring.consistency_monitor import Decision

BaselineDecision = Decision


@dataclass
class BaselineResult:
    """Serializable result returned by all baseline methods."""

    method: str
    decision: Decision
    reason: str = ""
    question: str = ""
    raw_output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "decision": self.decision.value,
            "reason": self.reason,
            "question": self.question,
            "raw_output": self.raw_output,
            "metadata": self.metadata,
        }
