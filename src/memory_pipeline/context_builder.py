"""
src.memory_pipeline.context_builder
=====================================
Serializes EpisodeMemory into a natural-language context string that is
injected into AmbRes task_description at each checkpoint call.

This is the bridge between the memory module and the VLM reasoning step.
Without this, AmbRes sees the same static task description every time;
with this, it receives a chronological record of what was observed and
decided, enabling dynamic re-grounding.
"""
from __future__ import annotations

from memory_pipeline.episode_memory import CheckpointRecord, EpisodeMemory


# ---------------------------------------------------------------------------
# Internal formatters
# ---------------------------------------------------------------------------

def _format_object(obj_state, role: str) -> str:
    n = obj_state.instance_count()
    if not obj_state.detected:
        return f"{role} '{obj_state.label}': NOT DETECTED"
    if n == 1:
        return f"{role} '{obj_state.label}': detected at {obj_state.coord}"
    return (
        f"{role} '{obj_state.label}': {n} instances detected at "
        + ", ".join(str(c) for c in obj_state.coords)
    )


def _format_record(rec: CheckpointRecord) -> str:
    lines = [f"[{rec.step}]"]
    lines.append("  " + _format_object(rec.target, "target"))
    lines.append("  " + _format_object(rec.destination, "destination"))

    if rec.changes:
        lines.append("  changes: " + "; ".join(rec.changes))

    if rec.decision:
        line = f"  decision: {rec.decision}"
        if rec.grounding_state:
            line += f"  (state: {rec.grounding_state})"
        lines.append(line)

    if rec.user_response:
        lines.append(f'  user clarified: "{rec.user_response}"')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_memory_context(memory: EpisodeMemory, current_step: str) -> str:
    """
    Build a natural-language summary of all prior observations.

    The context is injected into the AmbRes task_description so the VLM
    can generate clarifying questions that reference the specific scene
    changes observed (e.g. count increase, position shift) rather than
    generic "어떤 X을 집을까요?" questions.

    Args:
        memory:       Accumulated EpisodeMemory (may be empty at t0).
        current_step: Checkpoint currently being evaluated ("C1" or "C2").

    Returns:
        Multi-line context string, or "" if no prior records exist.
    """
    records = memory.records
    if not records:
        return ""

    lines = [
        "[Episode Memory — prior observations]",
    ]
    for rec in records:
        lines.append(_format_record(rec))

    lines.append(f"[Now evaluating: {current_step}]")

    # Collect changes across all records to surface them explicitly
    all_changes: list[str] = []
    for rec in records:
        if rec.changes:
            all_changes.extend(rec.changes)

    if all_changes:
        lines.append(
            "[Scene changes detected: " + "; ".join(all_changes) + "]"
        )
        lines.append(
            "[If asking a clarifying question, reference the specific changes above "
            "rather than asking a generic question]"
        )

    return "\n".join(lines)


def inject_context(task: str, context: str) -> str:
    """
    Append memory context to a task description for AmbRes.

    If context is empty (first call at t0), the task is returned unchanged.
    """
    if not context:
        return task
    return f"{task}\n\n{context}"
