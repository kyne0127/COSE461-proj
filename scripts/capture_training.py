#!/usr/bin/env python3
"""
scripts/capture_training.py
============================
Scene capture script for AmbRes/Molmo finetuning.

Guides through object combinations (cup/cube x red box/yellow box)
and captures images systematically for training data collection.

Controls:
  SPACE  ->  Capture current scene (repeat for same combo)
  n      ->  Next combo
  p      ->  Previous combo
  r      ->  Undo last capture
  q      ->  Save and quit

Output structure:
  data-training/
    <combo_id>/
      img_001.png
      img_002.png
      ...
      meta.json

Usage:
  python scripts/capture_training.py
  python scripts/capture_training.py --out-dir /path/to/output --target-per-combo 10
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

# ── 씬 조합 정의 ──────────────────────────────────────────────────────────────
# ambiguous: False = CLEAR, "target" = AMBIGUOUS_TARGET,
#            "destination" = AMBIGUOUS_DESTINATION, "invalid" = INVALID_TARGET
COMBOS: list[dict] = [
    # ── Clear scenes ─────────────────────────────────────────────────────────
    {
        "id": "cup1_redbox1",
        "desc": "cup x1 + red box x1",
        "setup": "Place 1 cup and 1 red box without overlap",
        "task": "pick the cup and put it in the red box",
        "target_label": "cup",
        "destination_label": "red box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    {
        "id": "cup1_yellowbox1",
        "desc": "cup x1 + yellow box x1",
        "setup": "Place 1 cup and 1 yellow box without overlap",
        "task": "pick the cup and put it in the yellow box",
        "target_label": "cup",
        "destination_label": "yellow box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    {
        "id": "cube1_redbox1",
        "desc": "cube x1 + red box x1",
        "setup": "Place 1 cube and 1 red box without overlap",
        "task": "pick the cube and put it in the red box",
        "target_label": "cube",
        "destination_label": "red box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    {
        "id": "cube1_yellowbox1",
        "desc": "cube x1 + yellow box x1",
        "setup": "Place 1 cube and 1 yellow box without overlap",
        "task": "pick the cube and put it in the yellow box",
        "target_label": "cube",
        "destination_label": "yellow box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    # ── Ambiguous target ──────────────────────────────────────────────────────
    {
        "id": "cup2_redbox1",
        "desc": "cup x2 + red box x1",
        "setup": "Place 2 identical cups (well separated) and 1 red box",
        "task": "pick the cup and put it in the red box",
        "target_label": "cup",
        "destination_label": "red box",
        "ambiguous": "target",
        "gold_state": "AMBIGUOUS_TARGET",
    },
    {
        "id": "cup2_yellowbox1",
        "desc": "cup x2 + yellow box x1",
        "setup": "Place 2 identical cups (well separated) and 1 yellow box",
        "task": "pick the cup and put it in the yellow box",
        "target_label": "cup",
        "destination_label": "yellow box",
        "ambiguous": "target",
        "gold_state": "AMBIGUOUS_TARGET",
    },
    {
        "id": "cube2_redbox1",
        "desc": "cube x2 + red box x1",
        "setup": "Place 2 identical cubes (well separated) and 1 red box",
        "task": "pick the cube and put it in the red box",
        "target_label": "cube",
        "destination_label": "red box",
        "ambiguous": "target",
        "gold_state": "AMBIGUOUS_TARGET",
    },
    {
        "id": "cube2_yellowbox1",
        "desc": "cube x2 + yellow box x1",
        "setup": "Place 2 identical cubes (well separated) and 1 yellow box",
        "task": "pick the cube and put it in the yellow box",
        "target_label": "cube",
        "destination_label": "yellow box",
        "ambiguous": "target",
        "gold_state": "AMBIGUOUS_TARGET",
    },
    # ── Ambiguous destination ─────────────────────────────────────────────────
    {
        "id": "cup1_twobox",
        "desc": "cup x1 + red box + yellow box",
        "setup": "Place 1 cup, 1 red box, 1 yellow box  [task has no color]",
        "task": "pick the cup and put it in the box",
        "target_label": "cup",
        "destination_label": "box",
        "ambiguous": "destination",
        "gold_state": "AMBIGUOUS_DESTINATION",
    },
    {
        "id": "cube1_twobox",
        "desc": "cube x1 + red box + yellow box",
        "setup": "Place 1 cube, 1 red box, 1 yellow box  [task has no color]",
        "task": "pick the cube and put it in the box",
        "target_label": "cube",
        "destination_label": "box",
        "ambiguous": "destination",
        "gold_state": "AMBIGUOUS_DESTINATION",
    },
    # ── Distractor ────────────────────────────────────────────────────────────
    {
        "id": "cup1_redbox1_cube_distractor",
        "desc": "cup x1 + red box x1 + cube (distractor)",
        "setup": "Place 1 cup, 1 red box, 1 cube (unrelated object)",
        "task": "pick the cup and put it in the red box",
        "target_label": "cup",
        "destination_label": "red box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    {
        "id": "cube1_redbox1_cup_distractor",
        "desc": "cube x1 + red box x1 + cup (distractor)",
        "setup": "Place 1 cube, 1 red box, 1 cup (unrelated object)",
        "task": "pick the cube and put it in the red box",
        "target_label": "cube",
        "destination_label": "red box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    # ── Invalid target ────────────────────────────────────────────────────────
    {
        "id": "noobj_redbox1",
        "desc": "red box only  (no target)",
        "setup": "Place only 1 red box. Remove all cups and cubes from scene.",
        "task": "pick the cup and put it in the red box",
        "target_label": "cup",
        "destination_label": "red box",
        "ambiguous": "invalid",
        "gold_state": "INVALID_TARGET",
    },
    {
        "id": "noobj_yellowbox1",
        "desc": "yellow box only  (no target)",
        "setup": "Place only 1 yellow box. Remove all cups and cubes from scene.",
        "task": "pick the cube and put it in the yellow box",
        "target_label": "cube",
        "destination_label": "yellow box",
        "ambiguous": "invalid",
        "gold_state": "INVALID_TARGET",
    },
]

_GOLD_COLOR = {
    "CLEAR":                  (50,  220,  80),
    "AMBIGUOUS_TARGET":       (50,  200, 255),
    "AMBIGUOUS_DESTINATION":  (50,  150, 255),
    "INVALID_TARGET":         (80,   80, 255),
}


# ── 오버레이 렌더링 ────────────────────────────────────────────────────────────

def _put(img: np.ndarray, text: str, y: int, color=(220, 220, 220), scale=0.55, thick=1):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(bgr: np.ndarray, combo: dict, combo_idx: int,
                 total: int, captured: int, target_per: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    canvas = bgr.copy()

    bar = canvas.copy()
    cv2.rectangle(bar, (0, 0), (w, 155), (0, 0, 0), -1)
    canvas = cv2.addWeighted(bar, 0.6, canvas, 0.4, 0)

    state_col = _GOLD_COLOR.get(combo["gold_state"], (220, 220, 220))
    progress_col = (50, 255, 120) if captured >= target_per else (200, 200, 200)

    _put(canvas, f"[{combo_idx+1}/{total}] {combo['desc']}", 24, state_col, 0.65, 2)
    _put(canvas, f"  Setup: {combo['setup']}", 48, (255, 230, 80), 0.55)
    _put(canvas, f"  Task : {combo['task']}", 70, (200, 200, 200), 0.52)
    _put(canvas, f"  Gold : {combo['gold_state']}", 92, state_col, 0.52)
    _put(canvas, f"  Captured: {captured}/{target_per}", 114, progress_col, 0.55)
    _put(canvas,
         "[SPACE] capture  [n] next  [p] prev  [r] undo last  [q] quit",
         138, (160, 160, 160), 0.48)

    # 진행 바
    bar_w = int(w * captured / max(target_per, 1))
    cv2.rectangle(canvas, (0, h - 6), (bar_w, h), state_col, -1)

    return canvas


# ── 메인 ─────────────────────────────────────────────────────────────────────

def _next_img_index(combo_dir: Path) -> int:
    existing = sorted(combo_dir.glob("img_*.png"))
    return len(existing) + 1


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    zed = ZEDCapture(resolution=args.resolution, fps=30, depth_mode="PERFORMANCE")

    combo_idx = 0
    total = len(COMBOS)

    cv2.namedWindow("capture_training", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("capture_training", 900, 530)

    print(f"\n{total} combos  |  target {args.target_per_combo} images per combo")
    print("SPACE=capture  n=next  p=prev  r=undo last  q=quit\n")

    try:
        while True:
            combo = COMBOS[combo_idx]
            combo_dir = out_root / combo["id"]
            combo_dir.mkdir(parents=True, exist_ok=True)

            rgb, _ = zed.capture_synchronized()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            captured = _next_img_index(combo_dir) - 1
            display = draw_overlay(bgr.copy(), combo, combo_idx, total,
                                   captured, args.target_per_combo)
            cv2.imshow("capture_training", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('n'):
                if combo_idx < total - 1:
                    combo_idx += 1
                    print(f"-> [{combo_idx+1}/{total}] {COMBOS[combo_idx]['desc']}")
                else:
                    print("Already at last combo.")

            elif key == ord('p'):
                if combo_idx > 0:
                    combo_idx -= 1
                    print(f"<- [{combo_idx+1}/{total}] {COMBOS[combo_idx]['desc']}")
                else:
                    print("Already at first combo.")

            elif key == ord('r'):
                imgs = sorted(combo_dir.glob("img_*.png"))
                if imgs:
                    imgs[-1].unlink()
                    print(f"  [r] removed: {imgs[-1].name}  (remaining: {len(imgs)-1})")
                else:
                    print("  No images to remove.")

            elif key == ord(' '):
                idx = _next_img_index(combo_dir)
                img_path = combo_dir / f"img_{idx:03d}.png"
                cv2.imwrite(str(img_path), bgr)
                print(f"  [SPACE] saved -> {img_path.name}  "
                      f"({idx}/{args.target_per_combo})")

                # update meta.json
                meta = {
                    "combo_id":          combo["id"],
                    "desc":              combo["desc"],
                    "task":              combo["task"],
                    "target_label":      combo["target_label"],
                    "destination_label": combo["destination_label"],
                    "gold_state":        combo["gold_state"],
                    "ambiguous":         combo["ambiguous"],
                    "image_count":       idx,
                    "updated_at":        datetime.now().isoformat(timespec="seconds"),
                }
                with open(combo_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                if idx >= args.target_per_combo:
                    print(f"  Combo complete! Press [n] to move to next combo.\n")

    finally:
        zed.release()
        cv2.destroyAllWindows()

    # summary
    print(f"\n{'='*55}")
    print(f"  Capture Summary")
    print(f"{'='*55}")
    total_imgs = 0
    for c in COMBOS:
        d = out_root / c["id"]
        count = len(list(d.glob("img_*.png"))) if d.exists() else 0
        total_imgs += count
        status = "done" if count >= args.target_per_combo else f"{count}/{args.target_per_combo}"
        print(f"  {status:>6}  {c['id']}")
    print(f"{'='*55}")
    print(f"  Total: {total_imgs} images  ->  {out_root}")
    print(f"\nNext: annotate coordinates with annotate_coords.py")
    print(f"  python scripts/annotate_coords.py --dir {out_root} \\")
    print(f"      --target-label cup --destination-label box")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AmbRes training data capture (guided scene combos)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Combo list (14 total):
  Clear           : cup+redbox, cup+yellowbox, cube+redbox, cube+yellowbox
  AmbiguousTarget : cup x2+redbox, cup x2+yellowbox, cube x2+redbox, cube x2+yellowbox
  AmbiguousDest   : cup+twobox, cube+twobox
  Distractor      : cup+redbox+cube, cube+redbox+cup
  InvalidTarget   : noobj+redbox, noobj+yellowbox
        """,
    )
    p.add_argument("--out-dir",           default=str(ROOT / "data-training"),
                   help="Output root directory (default: data-training/)")
    p.add_argument("--resolution",        default="VGA", choices=["VGA", "HD720"])
    p.add_argument("--target-per-combo",  type=int, default=10,
                   help="Target images per combo (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
