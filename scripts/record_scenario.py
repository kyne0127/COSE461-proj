#!/usr/bin/env python3
"""
scripts/record_scenario.py
==========================
Record t0/C1/C2 still captures for S1~S6 scenarios.

Controls:
  SPACE  ->  Capture t0 (initial scene) + start recording
  1      ->  Capture C1 checkpoint
  2      ->  Capture C2 checkpoint
  r      ->  Restart current trial
  q      ->  Save and quit

Usage:
  python scripts/record_scenario.py \\
      --scenario S2 \\
      --task "pick the cup and put it in the red box" \\
      --target-label cup --destination-label "red box"

Output structure:
  <out-dir>/
    S2/
      trial_001/
        video_global.mp4
        t0.png
        c1.png
        c2.png
        meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from module.desktop.zed_connector import ZEDCapture

# ── Scenario metadata ─────────────────────────────────────────────────────────
SCENARIO_META: dict[str, dict] = {
    "S1": {
        "gold_state":    "CLEAR",
        "gold_decision": "CONTINUE",
        "checkpoint":    "C1",
        "desc":          "No change",
        "intervention":  "(none — keep the scene as is)",
        "object_states": {
            "t0": {"target": 1, "destination": 1},
            "c1": {"target": 1, "destination": 1},
            "c2": {"target": 1, "destination": 1},
        },
    },
    "S2": {
        "gold_state":    "AMBIGUOUS_TARGET",
        "gold_decision": "ASK",
        "checkpoint":    "C1",
        "desc":          "target 1 -> 2 (add identical)",
        "intervention":  "* Before C1: add an identical target to the scene",
        "object_states": {
            "t0": {"target": 1, "destination": 1},
            "c1": {"target": 2, "destination": 1},
            "c2": {"target": 2, "destination": 1},
        },
    },
    "S3": {
        "gold_state":    "INVALID_TARGET",
        "gold_decision": "STOP",
        "checkpoint":    "C1",
        "desc":          "target 1 -> 0 (removed)",
        "intervention":  "* Before C1: remove the target from the scene",
        "object_states": {
            "t0": {"target": 1, "destination": 1},
            "c1": {"target": 0, "destination": 1},
            "c2": {"target": 0, "destination": 1},
        },
    },
    "S4": {
        "gold_state":    "AMBIGUOUS_TARGET",
        "gold_decision": "ASK",
        "checkpoint":    "C1",
        "desc":          "target moved to new position",
        "intervention":  "* Before C1: move the target to a different position",
        "object_states": {
            "t0": {"target": 1, "destination": 1},
            "c1": {"target": 1, "destination": 1, "note": "target moved"},
            "c2": {"target": 1, "destination": 1, "note": "target moved"},
        },
    },
    "S5": {
        "gold_state":    "AMBIGUOUS_DESTINATION",
        "gold_decision": "ASK",
        "checkpoint":    "C2",
        "desc":          "destination 1 -> 2 (add identical)",
        "intervention":  "* Before C2: add an identical destination to the scene",
        "object_states": {
            "t0": {"target": 1, "destination": 1},
            "c1": {"target": 1, "destination": 1},
            "c2": {"target": 1, "destination": 2},
        },
    },
    "S6": {
        "gold_state":    "INVALID_DESTINATION",
        "gold_decision": "STOP",
        "checkpoint":    "C2",
        "desc":          "destination 1 -> 0 (removed)",
        "intervention":  "* Before C2: remove the destination from the scene",
        "object_states": {
            "t0": {"target": 1, "destination": 1},
            "c1": {"target": 1, "destination": 1},
            "c2": {"target": 1, "destination": 0},
        },
    },
    "S7": {
        "gold_state":    "AMBIGUOUS_DESTINATION",
        "gold_decision": "ASK",
        "checkpoint":    "C2",
        "desc":          "destination moved to new position",
        "intervention":  "* Before C2: move the destination to a different position",
        "object_states": {
            "t0": {"target": 1, "destination": 1},
            "c1": {"target": 1, "destination": 1},
            "c2": {"target": 1, "destination": 1, "note": "destination moved"},
        },
    },
}

_STATE_HINT = {
    "IDLE":    ("SPACE", "Capture t0 (initial scene)"),
    "T0_DONE": ("1",     "Capture C1 checkpoint"),
    "C1_DONE": ("2",     "Capture C2 checkpoint"),
    "C2_DONE": ("q",     "Save and quit"),
}

_STATE_TO_TIMESTEP = {
    "IDLE":    "t0",
    "T0_DONE": "c1",
    "C1_DONE": "c2",
    "C2_DONE": "c2",
}


def _fmt_objects(obj_state: dict, target_label: str, dest_label: str) -> str:
    parts = []
    for key, val in obj_state.items():
        if key == "note":
            parts.append(f"({val})")
        elif key == "target":
            parts.append(f"{target_label}x{val}")
        elif key == "destination":
            parts.append(f"{dest_label}x{val}")
        else:
            parts.append(f"{key}x{val}")
    return "  ".join(parts)
_STATE_COLOR = {
    "IDLE":    (160, 160, 160),
    "T0_DONE": (50,  200, 255),
    "C1_DONE": (50,  255, 120),
    "C2_DONE": (50,  120, 255),
}


# ── Overlay rendering ─────────────────────────────────────────────────────────

def _put(img: np.ndarray, text: str, y: int, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(bgr: np.ndarray, state: str, scenario: str, trial: int,
                 meta: dict, stills: set, recording: bool,
                 target_label: str = "target", dest_label: str = "destination") -> np.ndarray:
    h, w = bgr.shape[:2]
    canvas = bgr.copy()

    bar = canvas.copy()
    cv2.rectangle(bar, (0, 0), (w, 160), (0, 0, 0), -1)
    canvas = cv2.addWeighted(bar, 0.55, canvas, 0.45, 0)

    col = _STATE_COLOR.get(state, (255, 255, 255))
    next_key, next_hint = _STATE_HINT.get(state, ("", ""))
    timestep = _STATE_TO_TIMESTEP.get(state, "t0")
    obj_state = meta.get("object_states", {}).get(timestep, {})
    obj_str = _fmt_objects(obj_state, target_label, dest_label)

    _put(canvas, f"[{scenario}] Trial {trial:03d}  |  {meta['desc']}", 24, col, 0.65, 2)
    _put(canvas, meta["intervention"],                                  48, (255, 220, 80), 0.55)
    _put(canvas, f"State: {state}  |  Scene: {obj_str}",               72, col, 0.55)
    _put(canvas, f"Next:  [{next_key}] {next_hint}",                    96, (200, 200, 200), 0.52)

    icons = (
        f"t0={'[OK]' if 't0' in stills else '[ ]'}  "
        f"C1={'[OK]' if 'c1' in stills else '[ ]'}  "
        f"C2={'[OK]' if 'c2' in stills else '[ ]'}"
    )
    rec_str = "  * REC" if recording else ""
    _put(canvas, icons + rec_str, 120, (180, 255, 180) if recording else (180, 180, 180), 0.52)
    _put(canvas,
         f"t0:{_fmt_objects(meta['object_states']['t0'], target_label, dest_label)}"
         f"  ->  c1:{_fmt_objects(meta['object_states']['c1'], target_label, dest_label)}"
         f"  ->  c2:{_fmt_objects(meta['object_states']['c2'], target_label, dest_label)}",
         142, (140, 140, 140), 0.45)

    _put(canvas, "[r] restart    [q] save and quit",
         h - 10, (140, 140, 140), 0.48)

    # Intervention reminder banner
    if state == "T0_DONE" and scenario not in ("S1",) and scenario != "S4":
        cv2.rectangle(canvas, (0, h - 50), (w, h - 25), (0, 0, 180), -1)
        _put(canvas, f"  {meta['intervention']}", h - 33, (255, 255, 100), 0.55)
    elif state == "C1_DONE" and scenario == "S4":
        cv2.rectangle(canvas, (0, h - 50), (w, h - 25), (0, 0, 180), -1)
        _put(canvas, f"  {meta['intervention']}", h - 33, (255, 255, 100), 0.55)

    return canvas


# ── Main loop ─────────────────────────────────────────────────────────────────

def _next_trial_number(out_root: Path) -> int:
    existing = sorted(out_root.glob("trial_*"))
    return len(existing) + 1


def run(args: argparse.Namespace) -> None:
    meta      = SCENARIO_META[args.scenario]
    out_root  = Path(args.out_dir) / args.scenario
    trial     = args.trial if args.trial else _next_trial_number(out_root)
    trial_dir = out_root / f"trial_{trial:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Scenario : {args.scenario}  Trial {trial:03d}")
    print(f"  Desc     : {meta['desc']}")
    print(f"  Action   : {meta['intervention']}")
    print(f"  Task     : {args.task}")
    print(f"  Save to  : {trial_dir}")
    print(f"{'='*60}\n")

    zed = ZEDCapture(resolution=args.resolution, fps=30, depth_mode="PERFORMANCE")

    video_path = str(trial_dir / "video_global.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    w_vid, h_vid = (1280, 720) if args.resolution == "HD720" else (672, 376)
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w_vid, h_vid))

    state     = "IDLE"
    stills: set[str] = set()
    recording = False

    cv2.namedWindow("record_scenario", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("record_scenario", max(w_vid, 800), max(h_vid + 20, 450))

    try:
        while True:
            rgb, _ = zed.capture_synchronized()
            bgr    = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if recording:
                writer.write(bgr)

            display = draw_overlay(bgr.copy(), state, args.scenario, trial,
                                   meta, stills, recording,
                                   args.target_label, args.destination_label)
            cv2.imshow("record_scenario", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):
                stills.clear()
                state     = "IDLE"
                recording = False
                writer.release()
                writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w_vid, h_vid))
                print("[r] Trial restarted — stills and video cleared")

            elif key == ord(' ') and state == "IDLE":
                cv2.imwrite(str(trial_dir / "t0.png"), bgr)
                stills.add("t0")
                state     = "T0_DONE"
                recording = True
                print("[SPACE] t0 saved -> t0.png  (recording started)")

            elif key == ord('1') and state == "T0_DONE":
                cv2.imwrite(str(trial_dir / "c1.png"), bgr)
                stills.add("c1")
                state = "C1_DONE"
                print("[1] C1 saved -> c1.png")

            elif key == ord('2') and state == "C1_DONE":
                cv2.imwrite(str(trial_dir / "c2.png"), bgr)
                stills.add("c2")
                state = "C2_DONE"
                print("[2] C2 saved -> c2.png")

    finally:
        writer.release()
        zed.release()
        cv2.destroyAllWindows()

    # ── Save metadata ─────────────────────────────────────────────────────────
    entry = {
        "id":                f"{args.scenario.lower()}_trial_{trial:03d}",
        "scenario":          args.scenario,
        "task":              args.task,
        "initial_img":       str(trial_dir / "t0.png") if "t0" in stills else "",
        "c1_img":            str(trial_dir / "c1.png") if "c1" in stills else "",
        "c2_img":            str(trial_dir / "c2.png") if "c2" in stills else "",
        "checkpoint":        meta["checkpoint"],
        "gold_state":        meta["gold_state"],
        "gold_decision":     meta["gold_decision"],
        "target_label":      args.target_label,
        "destination_label": args.destination_label,
        "object_states":     meta["object_states"],
        "video":             video_path,
        "notes":             meta["intervention"],
        "recorded_at":       datetime.now().isoformat(timespec="seconds"),
    }

    with open(trial_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    if args.append_manifest and len(stills) >= 2:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Appended to manifest -> {manifest_path}")

    print(f"\n{'='*60}")
    print(f"  Saved  Trial {trial:03d}")
    print(f"  Video : {video_path}")
    print(f"  Stills: t0={'O' if 't0' in stills else '-'}  "
          f"C1={'O' if 'c1' in stills else '-'}  "
          f"C2={'O' if 'c2' in stills else '-'}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scenario-based recording (S1~S6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Intervention guide:
  S1  no change
  S2  before C1: add identical target
  S3  before C1: remove target
  S4  before C1: move target to different position
  S5  before C2: add identical destination
  S6  before C2: remove destination
  S7  before C2: move destination to different position

Example:
  python scripts/record_scenario.py \\
      --scenario S2 \\
      --task "pick the cup and put it in the red box" \\
      --target-label cup --destination-label "red box" \\
      --append-manifest
        """,
    )
    p.add_argument("--scenario",          required=True, choices=list(SCENARIO_META),
                   metavar="{" + ",".join(SCENARIO_META) + "}")
    p.add_argument("--task",              required=True)
    p.add_argument("--target-label",      default="cup")
    p.add_argument("--destination-label", default="box")
    p.add_argument("--trial",             type=int, default=None,
                   help="Trial number (auto-increments if omitted)")
    p.add_argument("--resolution",        default="VGA", choices=["VGA", "HD720"])
    p.add_argument("--out-dir",           default=str(ROOT / "data-evaluation"),
                   help="Output root directory (default: data-evaluation/)")
    p.add_argument("--manifest-path",     default=str(ROOT / "dataset" / "manifest.jsonl"),
                   help="Manifest file to append to (default: dataset/manifest.jsonl)")
    p.add_argument("--append-manifest",   action="store_true",
                   help="Append entry to manifest file")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
