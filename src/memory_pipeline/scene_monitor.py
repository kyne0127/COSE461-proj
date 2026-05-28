"""
src.memory_pipeline.scene_monitor
====================================
GDino-based object state monitor.

Provides discrete snapshots of object presence/position at each checkpoint.
In the full real-robot deployment this would run as a demon at 5-10 Hz;
for offline evaluation it is called once per checkpoint image.

Detects:
  - Object appearance / disappearance
  - Instance count changes (one cup → two cups)
  - Position shifts beyond a threshold (object moved)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image

from memory_pipeline.episode_memory import CheckpointRecord, ObjectState


def _euclidean(a: list[int], b: list[int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class SceneMonitor:
    """
    Wraps GroundingDINO to produce CheckpointRecords for EpisodeMemory.

    Args:
        gdino:             GroundingDINO instance (module.models.ambres.gdino).
        move_threshold_px: Pixel distance above which a position shift is
                           reported as a "moved" change event.
    """

    def __init__(self, gdino: Any, move_threshold_px: float = 50.0) -> None:
        self._gdino = gdino
        self._move_thr = move_threshold_px

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(
        self,
        image_path: str | Path,
        target_label: str,
        destination_label: str,
        step: str,
    ) -> CheckpointRecord:
        """
        Detect both objects and return an unpopulated CheckpointRecord.

        decision / grounding_state / user_response are left empty;
        the pipeline fills them after the AmbRes call.
        """
        pil_image = Image.open(image_path).convert("RGB")
        detections = self._gdino.detect(pil_image, [target_label, destination_label])

        return CheckpointRecord(
            step=step,
            target=ObjectState.from_detection(target_label, detections),
            destination=ObjectState.from_detection(destination_label, detections),
        )

    def snapshot_from_detections(
        self,
        detections: dict[str, Any],
        target_label: str,
        destination_label: str,
        step: str,
    ) -> CheckpointRecord:
        """Build a CheckpointRecord from already-computed detections."""
        return CheckpointRecord(
            step=step,
            target=ObjectState.from_detection(target_label, detections),
            destination=ObjectState.from_detection(destination_label, detections),
        )

    # ------------------------------------------------------------------ #
    # Change detection
    # ------------------------------------------------------------------ #

    def snapshot_with_generic(
        self,
        specific_detections: dict[str, Any],
        generic_detections: dict[str, Any],
        target_label: str,
        destination_label: str,
        target_generic_label: str,
        dest_generic_label: str,
        step: str,
    ) -> CheckpointRecord:
        """Build a CheckpointRecord with both specific and generic detections."""
        record = self.snapshot_from_detections(
            specific_detections, target_label, destination_label, step
        )
        record.target_generic = ObjectState.from_detection(target_generic_label, generic_detections)
        record.dest_generic   = ObjectState.from_detection(dest_generic_label,   generic_detections)
        return record

    def detect_changes_enhanced(
        self,
        prev: CheckpointRecord,
        curr: CheckpointRecord,
    ) -> list[str]:
        """detect_changes() + same-category object addition detection.

        Fires when a generic-label count increases while the specific-label
        count stays the same — i.e., a same-kind-but-differently-named object
        appeared in the scene.
        """
        changes = self.detect_changes(prev, curr)

        for role, spec_prev, spec_curr, gen_prev, gen_curr in (
            ("target",
             prev.target,      curr.target,
             prev.target_generic, curr.target_generic),
            ("destination",
             prev.destination, curr.destination,
             prev.dest_generic,  curr.dest_generic),
        ):
            if gen_prev is None or gen_curr is None:
                continue
            # Skip if generic label == specific label (no additional info)
            if gen_curr.label == spec_curr.label:
                continue

            prev_generic_n  = gen_prev.instance_count()
            curr_generic_n  = gen_curr.instance_count()
            curr_specific_n = spec_curr.instance_count()
            prev_specific_n = spec_prev.instance_count()

            if curr_generic_n > prev_generic_n and curr_specific_n == prev_specific_n:
                changes.append(
                    f"{role}: same-category '{gen_curr.label}' count "
                    f"{prev_generic_n}→{curr_generic_n} "
                    f"('{spec_curr.label}' count unchanged at {curr_specific_n})"
                )

        return changes

    def detect_changes(
        self,
        prev: CheckpointRecord,
        curr: CheckpointRecord,
    ) -> list[str]:
        """
        Return a human-readable list of object-state changes between two snapshots.

        Detects: appearance, disappearance, count change, position shift.
        """
        changes: list[str] = []
        changes.extend(self._object_changes(prev.target, curr.target, "target"))
        changes.extend(self._object_changes(prev.destination, curr.destination, "destination"))
        return changes

    def _object_changes(
        self,
        prev: ObjectState,
        curr: ObjectState,
        role: str,
    ) -> list[str]:
        changes: list[str] = []
        label = curr.label

        # Appearance / disappearance
        if prev.detected and not curr.detected:
            changes.append(f"{role} '{label}' disappeared")
            return changes
        if not prev.detected and curr.detected:
            changes.append(f"{role} '{label}' appeared")
            return changes
        if not prev.detected and not curr.detected:
            return changes

        # Count change
        prev_n = prev.instance_count()
        curr_n = curr.instance_count()
        if curr_n > prev_n:
            changes.append(
                f"{role} '{label}' count increased {prev_n} → {curr_n}"
            )
        elif curr_n < prev_n:
            changes.append(
                f"{role} '{label}' count decreased {prev_n} → {curr_n}"
            )

        # Position shift (single instance only, to avoid matching ambiguity)
        if prev_n == 1 and curr_n == 1 and prev.coord and curr.coord:
            dist = _euclidean(prev.coord, curr.coord)
            if dist > self._move_thr:
                changes.append(
                    f"{role} '{label}' moved {dist:.0f}px "
                    f"({prev.coord} → {curr.coord})"
                )

        return changes
