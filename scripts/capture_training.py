#!/usr/bin/env python3
"""
scripts/capture_training.py
============================
AmbRes/Molmo finetuning 학습 데이터 촬영 스크립트.

물체 조합(cup/cube × red box/yellow box)별로 씬 구성을 안내하며
이미지를 체계적으로 촬영한다.

조작법:
  SPACE  →  현재 씬 촬영 (같은 조합 반복 촬영 가능)
  n      →  다음 조합으로 이동
  p      →  이전 조합으로 이동
  r      →  마지막 촬영 취소
  q      →  저장 후 종료

출력 구조:
  data-training/
    <combo_id>/
      img_001.png
      img_002.png
      ...
      meta.json   (label 정보)

사용 예:
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
        "desc": "컵 1개 + 빨간박스 1개",
        "setup": "컵 1개, 빨간박스 1개를 겹치지 않게 배치",
        "task": "pick the cup and put it in the red box",
        "target_label": "cup",
        "destination_label": "red box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    {
        "id": "cup1_yellowbox1",
        "desc": "컵 1개 + 노란박스 1개",
        "setup": "컵 1개, 노란박스 1개를 겹치지 않게 배치",
        "task": "pick the cup and put it in the yellow box",
        "target_label": "cup",
        "destination_label": "yellow box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    {
        "id": "cube1_redbox1",
        "desc": "큐브 1개 + 빨간박스 1개",
        "setup": "큐브 1개, 빨간박스 1개를 겹치지 않게 배치",
        "task": "pick the cube and put it in the red box",
        "target_label": "cube",
        "destination_label": "red box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    {
        "id": "cube1_yellowbox1",
        "desc": "큐브 1개 + 노란박스 1개",
        "setup": "큐브 1개, 노란박스 1개를 겹치지 않게 배치",
        "task": "pick the cube and put it in the yellow box",
        "target_label": "cube",
        "destination_label": "yellow box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    # ── Ambiguous target ──────────────────────────────────────────────────────
    {
        "id": "cup2_redbox1",
        "desc": "컵 2개 + 빨간박스 1개",
        "setup": "동일한 컵 2개(서로 충분히 떨어진 위치), 빨간박스 1개 배치",
        "task": "pick the cup and put it in the red box",
        "target_label": "cup",
        "destination_label": "red box",
        "ambiguous": "target",
        "gold_state": "AMBIGUOUS_TARGET",
    },
    {
        "id": "cup2_yellowbox1",
        "desc": "컵 2개 + 노란박스 1개",
        "setup": "동일한 컵 2개(서로 충분히 떨어진 위치), 노란박스 1개 배치",
        "task": "pick the cup and put it in the yellow box",
        "target_label": "cup",
        "destination_label": "yellow box",
        "ambiguous": "target",
        "gold_state": "AMBIGUOUS_TARGET",
    },
    {
        "id": "cube2_redbox1",
        "desc": "큐브 2개 + 빨간박스 1개",
        "setup": "동일한 큐브 2개(서로 충분히 떨어진 위치), 빨간박스 1개 배치",
        "task": "pick the cube and put it in the red box",
        "target_label": "cube",
        "destination_label": "red box",
        "ambiguous": "target",
        "gold_state": "AMBIGUOUS_TARGET",
    },
    {
        "id": "cube2_yellowbox1",
        "desc": "큐브 2개 + 노란박스 1개",
        "setup": "동일한 큐브 2개(서로 충분히 떨어진 위치), 노란박스 1개 배치",
        "task": "pick the cube and put it in the yellow box",
        "target_label": "cube",
        "destination_label": "yellow box",
        "ambiguous": "target",
        "gold_state": "AMBIGUOUS_TARGET",
    },
    # ── Ambiguous destination ─────────────────────────────────────────────────
    {
        "id": "cup1_twobox",
        "desc": "컵 1개 + 빨간박스 + 노란박스",
        "setup": "컵 1개, 빨간박스 1개, 노란박스 1개 배치 (task에 색 없음)",
        "task": "pick the cup and put it in the box",
        "target_label": "cup",
        "destination_label": "box",
        "ambiguous": "destination",
        "gold_state": "AMBIGUOUS_DESTINATION",
    },
    {
        "id": "cube1_twobox",
        "desc": "큐브 1개 + 빨간박스 + 노란박스",
        "setup": "큐브 1개, 빨간박스 1개, 노란박스 1개 배치 (task에 색 없음)",
        "task": "pick the cube and put it in the box",
        "target_label": "cube",
        "destination_label": "box",
        "ambiguous": "destination",
        "gold_state": "AMBIGUOUS_DESTINATION",
    },
    # ── Distractor ────────────────────────────────────────────────────────────
    {
        "id": "cup1_redbox1_cube_distractor",
        "desc": "컵 1개 + 빨간박스 1개 + 큐브(distractor)",
        "setup": "컵 1개, 빨간박스 1개, 큐브 1개(무관 물체) 배치",
        "task": "pick the cup and put it in the red box",
        "target_label": "cup",
        "destination_label": "red box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    {
        "id": "cube1_redbox1_cup_distractor",
        "desc": "큐브 1개 + 빨간박스 1개 + 컵(distractor)",
        "setup": "큐브 1개, 빨간박스 1개, 컵 1개(무관 물체) 배치",
        "task": "pick the cube and put it in the red box",
        "target_label": "cube",
        "destination_label": "red box",
        "ambiguous": False,
        "gold_state": "CLEAR",
    },
    # ── Invalid target ────────────────────────────────────────────────────────
    {
        "id": "noobj_redbox1",
        "desc": "빨간박스만 (target 없음)",
        "setup": "빨간박스 1개만 배치. 컵/큐브는 씬에서 제거",
        "task": "pick the cup and put it in the red box",
        "target_label": "cup",
        "destination_label": "red box",
        "ambiguous": "invalid",
        "gold_state": "INVALID_TARGET",
    },
    {
        "id": "noobj_yellowbox1",
        "desc": "노란박스만 (target 없음)",
        "setup": "노란박스 1개만 배치. 컵/큐브는 씬에서 제거",
        "task": "pick the cube and put it in the yellow box",
        "target_label": "cube",
        "destination_label": "yellow box",
        "ambiguous": "invalid",
        "gold_state": "INVALID_TARGET",
    },
]

# 상태별 색상
_GOLD_COLOR = {
    "CLEAR":                  (50,  220,  80),   # 초록
    "AMBIGUOUS_TARGET":       (50,  200, 255),   # 노란-주황
    "AMBIGUOUS_DESTINATION":  (50,  150, 255),   # 주황
    "INVALID_TARGET":         (80,   80, 255),   # 빨강
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
    _put(canvas, f"  세팅: {combo['setup']}", 48, (255, 230, 80), 0.55)
    _put(canvas, f"  Task: {combo['task']}", 70, (200, 200, 200), 0.52)
    _put(canvas, f"  Gold: {combo['gold_state']}", 92, state_col, 0.52)
    _put(canvas, f"  촬영: {captured}/{target_per}장", 114, progress_col, 0.55)
    _put(canvas,
         "[SPACE] 촬영  [n] 다음  [p] 이전  [r] 마지막 취소  [q] 종료",
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

    print(f"\n총 {total}개 조합  |  조합당 목표 {args.target_per_combo}장")
    print("SPACE=촬영  n=다음  p=이전  r=마지막 취소  q=종료\n")

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
                    print(f"→ [{combo_idx+1}/{total}] {COMBOS[combo_idx]['desc']}")
                else:
                    print("마지막 조합입니다.")

            elif key == ord('p'):
                if combo_idx > 0:
                    combo_idx -= 1
                    print(f"← [{combo_idx+1}/{total}] {COMBOS[combo_idx]['desc']}")
                else:
                    print("첫 번째 조합입니다.")

            elif key == ord('r'):
                imgs = sorted(combo_dir.glob("img_*.png"))
                if imgs:
                    imgs[-1].unlink()
                    print(f"  [r] 삭제: {imgs[-1].name}  (남은: {len(imgs)-1}장)")
                else:
                    print("  삭제할 이미지가 없습니다.")

            elif key == ord(' '):
                idx = _next_img_index(combo_dir)
                img_path = combo_dir / f"img_{idx:03d}.png"
                cv2.imwrite(str(img_path), bgr)
                print(f"  [SPACE] 저장 → {img_path.name}  "
                      f"(누적: {idx}/{args.target_per_combo})")

                # meta.json 갱신
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
                    print(f"  ✓ 조합 완료! [n]으로 다음 조합으로 이동하세요.\n")

    finally:
        zed.release()
        cv2.destroyAllWindows()

    # 완료 요약
    print(f"\n{'='*55}")
    print(f"  촬영 완료 요약")
    print(f"{'='*55}")
    total_imgs = 0
    for c in COMBOS:
        d = out_root / c["id"]
        count = len(list(d.glob("img_*.png"))) if d.exists() else 0
        total_imgs += count
        status = "✓" if count >= args.target_per_combo else f"{count}/{args.target_per_combo}"
        print(f"  {status:>8}  {c['id']}")
    print(f"{'='*55}")
    print(f"  총 {total_imgs}장  →  {out_root}")
    print(f"\n다음 단계: annotate_coords.py 로 좌표 annotation")
    print(f"  python scripts/annotate_coords.py --dir {out_root} \\")
    print(f"      --target-label cup --destination-label box")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AmbRes 학습 데이터 촬영 (조합별 씬 안내)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
조합 목록 (총 14개):
  Clear           : cup+redbox, cup+yellowbox, cube+redbox, cube+yellowbox
  AmbiguousTarget : cup×2+redbox, cup×2+yellowbox, cube×2+redbox, cube×2+yellowbox
  AmbiguousDest   : cup+twobox, cube+twobox
  Distractor      : cup+redbox+cube, cube+redbox+cup
  InvalidTarget   : noobj+redbox, noobj+yellowbox
        """,
    )
    p.add_argument("--out-dir",           default=str(ROOT / "data-training"),
                   help="출력 루트 디렉터리 (기본: data-training/)")
    p.add_argument("--resolution",        default="VGA", choices=["VGA", "HD720"])
    p.add_argument("--target-per-combo",  type=int, default=10,
                   help="조합당 목표 촬영 수 (기본: 10)")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
