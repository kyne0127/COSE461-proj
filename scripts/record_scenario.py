#!/usr/bin/env python3
"""
scripts/record_scenario.py
==========================
6개 시나리오(S1~S6)의 영상 기록 + t₀/C1/C2 스틸 캡처 스크립트.

조작법:
  SPACE  →  t₀  초기 장면 캡처 (기록 시작)
  1      →  C1  pre-pick checkpoint 캡처
  2      →  C2  pre-place checkpoint 캡처
  r      →  현재 trial 재시작
  q      →  저장 후 종료

사용 예:
  python scripts/record_scenario.py \\
      --scenario S2 \\
      --task "pick the cup and put it into the box" \\
      --target-label cup --destination-label box

출력 구조:
  data_eval/
    S2/
      trial_001/
        video_global.mp4   (전체 영상)
        t0.png             (초기 장면)
        c1.png             (C1 checkpoint)
        c2.png             (C2 checkpoint)
        meta.json          (manifest entry)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from module.desktop.zed_connector import ZEDCapture

# ── 시나리오 메타데이터 ────────────────────────────────────────────────────────
SCENARIO_META: dict[str, dict] = {
    "S1": {
        "gold_state":   "CLEAR",
        "gold_decision": "CONTINUE",
        "checkpoint":   "C1",
        "desc":         "변화 없음 (clear continuation)",
        "intervention": "(없음 — 장면을 그대로 유지)",
    },
    "S2": {
        "gold_state":   "AMBIGUOUS_TARGET",
        "gold_decision": "ASK",
        "checkpoint":   "C1",
        "desc":         "동일 target 추가",
        "intervention": "★ C1 전: 동일한 컵을 씬에 추가하세요",
    },
    "S3": {
        "gold_state":   "INVALID_TARGET",
        "gold_decision": "STOP",
        "checkpoint":   "C1",
        "desc":         "target 제거",
        "intervention": "★ C1 전: 컵을 씬에서 제거하세요",
    },
    "S4": {
        "gold_state":   "AMBIGUOUS_DESTINATION",
        "gold_decision": "ASK",
        "checkpoint":   "C2",
        "desc":         "destination 추가",
        "intervention": "★ C2 전: 동일한 박스를 씬에 추가하세요",
    },
    "S5": {
        "gold_state":   "CLEAR",
        "gold_decision": "CONTINUE",
        "checkpoint":   "C1",
        "desc":         "무관 물체 추가 (distractor)",
        "intervention": "★ C1 전: 태스크와 무관한 물체를 씬에 추가하세요",
    },
    "S6": {
        "gold_state":   "AMBIGUOUS_TARGET",
        "gold_decision": "ASK",
        "checkpoint":   "C1",
        "desc":         "target 위치 이동",
        "intervention": "★ C1 전: 컵을 다른 위치로 이동하세요",
    },
}

# 상태별 (다음 키, 안내 문구)
_STATE_HINT = {
    "IDLE":     ("SPACE", "t₀ 초기 장면 캡처"),
    "T0_DONE":  ("1",     "C1 checkpoint 캡처"),
    "C1_DONE":  ("2",     "C2 checkpoint 캡처"),
    "C2_DONE":  ("q",     "저장 후 종료"),
}
_STATE_COLOR = {
    "IDLE":    (160, 160, 160),
    "T0_DONE": (50,  200, 255),
    "C1_DONE": (50,  255, 120),
    "C2_DONE": (50,  120, 255),
}


# ── 오버레이 렌더링 ────────────────────────────────────────────────────────────

def _put(img: np.ndarray, text: str, y: int, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(bgr: np.ndarray, state: str, scenario: str, trial: int,
                 meta: dict, stills: set, recording: bool) -> np.ndarray:
    h, w = bgr.shape[:2]
    canvas = bgr.copy()

    # 반투명 상단 바
    bar = canvas.copy()
    cv2.rectangle(bar, (0, 0), (w, 140), (0, 0, 0), -1)
    canvas = cv2.addWeighted(bar, 0.55, canvas, 0.45, 0)

    col = _STATE_COLOR.get(state, (255, 255, 255))
    next_key, next_hint = _STATE_HINT.get(state, ("", ""))

    _put(canvas, f"[{scenario}] Trial {trial:03d}  |  {meta['desc']}", 24,  col,  0.65, 2)
    _put(canvas, meta["intervention"],                                  50,  (255, 220, 80), 0.58)
    _put(canvas, f"State: {state}",                                     76,  col,  0.58)
    _put(canvas, f"Next:  [{next_key}] {next_hint}",                    100, (200, 200, 200), 0.55)

    # 스틸 체크
    icons = (
        f"t0={'[OK]' if 't0' in stills else '[ ]'}  "
        f"C1={'[OK]' if 'c1' in stills else '[ ]'}  "
        f"C2={'[OK]' if 'c2' in stills else '[ ]'}"
    )
    rec_str = "  ● REC" if recording else ""
    _put(canvas, icons + rec_str, 124, (180, 255, 180) if recording else (180, 180, 180), 0.52)

    # 하단 힌트
    _put(canvas, "[r] 재시작    [q] 저장 후 종료",
         h - 10, (140, 140, 140), 0.48)

    # 개입 시점 강조 배너
    if state == "T0_DONE" and scenario not in ("S1",) and scenario != "S4":
        cv2.rectangle(canvas, (0, h - 50), (w, h - 25), (0, 0, 180), -1)
        _put(canvas, f"  {meta['intervention']}", h - 33, (255, 255, 100), 0.55)
    elif state == "C1_DONE" and scenario == "S4":
        cv2.rectangle(canvas, (0, h - 50), (w, h - 25), (0, 0, 180), -1)
        _put(canvas, f"  {meta['intervention']}", h - 33, (255, 255, 100), 0.55)

    return canvas


# ── 메인 루프 ──────────────────────────────────────────────────────────────────

def _next_trial_number(out_root: Path) -> int:
    existing = sorted(out_root.glob("trial_*"))
    return len(existing) + 1


def run(args: argparse.Namespace) -> None:
    meta   = SCENARIO_META[args.scenario]
    out_root = ROOT / "data_eval" / args.scenario
    trial  = args.trial if args.trial else _next_trial_number(out_root)
    trial_dir = out_root / f"trial_{trial:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  시나리오: {args.scenario}  Trial {trial:03d}")
    print(f"  설명:     {meta['desc']}")
    print(f"  개입:     {meta['intervention']}")
    print(f"  Task:     {args.task}")
    print(f"  저장:     {trial_dir}")
    print(f"{'='*60}\n")

    zed = ZEDCapture(resolution=args.resolution, fps=30, depth_mode="PERFORMANCE")

    video_path = str(trial_dir / "video_global.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_vid, h_vid = (1280, 720) if args.resolution == "HD720" else (672, 376)
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w_vid, h_vid))

    state   = "IDLE"
    stills: set[str] = set()
    recording = False

    cv2.namedWindow("record_scenario", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("record_scenario", max(w_vid, 800), max(h_vid + 20, 450))

    try:
        while True:
            rgb, _ = zed.capture_synchronized()          # uint8 RGB [H, W, 3]
            bgr    = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if recording:
                writer.write(bgr)

            display = draw_overlay(bgr.copy(), state, args.scenario, trial,
                                   meta, stills, recording)
            cv2.imshow("record_scenario", display)
            key = cv2.waitKey(1) & 0xFF

            # ── 키 처리 ────────────────────────────────────────────────────
            if key == ord('q'):
                break

            elif key == ord('r'):
                stills.clear()
                state     = "IDLE"
                recording = False
                writer.release()
                writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w_vid, h_vid))
                print("[r] Trial 재시작 — 스틸/영상 초기화")

            elif key == ord(' ') and state == "IDLE":
                cv2.imwrite(str(trial_dir / "t0.png"), bgr)
                stills.add("t0")
                state     = "T0_DONE"
                recording = True
                print(f"[SPACE] t₀ 저장 → t0.png  (기록 시작)")

            elif key == ord('1') and state == "T0_DONE":
                cv2.imwrite(str(trial_dir / "c1.png"), bgr)
                stills.add("c1")
                state = "C1_DONE"
                print(f"[1] C1 저장 → c1.png")

            elif key == ord('2') and state == "C1_DONE":
                cv2.imwrite(str(trial_dir / "c2.png"), bgr)
                stills.add("c2")
                state = "C2_DONE"
                print(f"[2] C2 저장 → c2.png")

    finally:
        writer.release()
        zed.release()
        cv2.destroyAllWindows()

    # ── 메타데이터 저장 ────────────────────────────────────────────────────────
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
        "video":             video_path,
        "notes":             meta["intervention"],
        "recorded_at":       datetime.now().isoformat(timespec="seconds"),
    }

    with open(trial_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    if args.append_manifest and len(stills) >= 2:
        manifest_path = ROOT / "dataset" / "manifest.jsonl"
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"manifest.jsonl 에 추가 완료")

    print(f"\n{'='*60}")
    print(f"  저장 완료  Trial {trial:03d}")
    print(f"  영상:  {video_path}")
    print(f"  스틸:  t0={'O' if 't0' in stills else '-'}  "
          f"C1={'O' if 'c1' in stills else '-'}  "
          f"C2={'O' if 'c2' in stills else '-'}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="시나리오별 영상 기록 (S1~S6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
시나리오별 개입 지시:
  S1  변화 없음
  S2  C1 전: 동일한 컵 추가
  S3  C1 전: 컵 제거
  S4  C2 전: 동일한 박스 추가
  S5  C1 전: 무관 물체 추가
  S6  C1 전: 컵을 다른 위치로 이동

예시:
  python scripts/record_scenario.py \\
      --scenario S2 \\
      --task "pick the cup and put it into the box" \\
      --target-label cup --destination-label box --append-manifest
        """,
    )
    p.add_argument("--scenario",          required=True, choices=list(SCENARIO_META))
    p.add_argument("--task",              required=True)
    p.add_argument("--target-label",      default="cup")
    p.add_argument("--destination-label", default="box")
    p.add_argument("--trial",             type=int, default=None,
                   help="trial 번호 (생략 시 자동 증가)")
    p.add_argument("--resolution",        default="VGA", choices=["VGA", "HD720"])
    p.add_argument("--append-manifest",   action="store_true",
                   help="dataset/manifest.jsonl 에 자동 추가")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
