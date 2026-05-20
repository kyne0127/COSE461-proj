#!/usr/bin/env python3
"""
AmbRes 데이터 레이블링 스크립트
물체: 큐브×2, 빨간상자, 노란상자, 종이컵×2
역할: 큐브/종이컵 = target  |  빨간상자/노란상자 = destination

실행:
  python scripts/collect_and_label.py --mode train
  python scripts/collect_and_label.py --mode test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/AmbRes")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ────────────────────────────────────────────────────────────────────────────
# 씬 프리셋 정의
# ────────────────────────────────────────────────────────────────────────────
#
# obj_list     : 씬에 실제로 있는 물체 이름 (구체적)
# ambiguity_map: 모호한 물체 → 구체적 인스턴스 목록
#   - 큐브 2개  → {"cube":      ["left cube",      "right cube"]}
#   - 컵 2개    → {"paper cup": ["left paper cup", "right paper cup"]}
#   - 상자 2개  → {"box":       ["red box",        "yellow box"]}

def _s(targets, boxes):
    """(target_list, box_list) → (obj_list, ambiguity_map) 자동 생성."""
    obj_list = targets + boxes
    amb = {}
    if len([t for t in targets if "cube" in t]) >= 2:
        cubes = [t for t in targets if "cube" in t]
        amb["cube"] = cubes
    if len([t for t in targets if "cup" in t]) >= 2:
        cups = [t for t in targets if "cup" in t]
        amb["paper cup"] = cups
    if len(boxes) >= 2:
        amb["box"] = boxes
    return obj_list, amb

SCENES = []

# ── 타겟: 큐브 1개 ──────────────────────────────────────────────────────────
T1 = ["cube"]
T2 = ["left cube", "right cube"]
P1 = ["paper cup"]
P2 = ["left paper cup", "right paper cup"]

for targets, t_label in [
    (T1, "1큐브"),
    (T2, "2큐브"),
    (P1, "1종이컵"),
    (P2, "2종이컵"),
    (T1 + P1, "1큐브+1종이컵"),
    (T2 + P1, "2큐브+1종이컵"),
    (T1 + P2, "1큐브+2종이컵"),
    (T2 + P2, "2큐브+2종이컵"),
]:
    for boxes, b_label in [
        (["red box"],              "빨간상자"),
        (["yellow box"],           "노란상자"),
        (["red box", "yellow box"],"빨간상자+노란상자"),
    ]:
        obj_list, amb = _s(targets, boxes)
        amb_note = ""
        if "cube" in amb:      amb_note += " [큐브모호]"
        if "paper cup" in amb: amb_note += " [컵모호]"
        if "box" in amb:       amb_note += " [상자모호]"
        if not amb_note:       amb_note = " [명확]"
        SCENES.append({
            "desc":          f"{t_label} + {b_label}{amb_note}",
            "obj_list":      obj_list,
            "ambiguity_map": amb,
        })

# ────────────────────────────────────────────────────────────────────────────
# 태스크 자동 생성
# ────────────────────────────────────────────────────────────────────────────

TASK_TEMPLATES = [
    "Put the {tgt} next to the {box}.",
    "Place the {tgt} inside the {box}.",
    "Move the {tgt} to the {box}.",
    "Pick up the {tgt} and place it near the {box}.",
]

def generate_tasks(obj_list: list[str], amb_map: dict) -> list[str]:
    """obj_list + ambiguity_map → 태스크 목록 자동 생성."""
    # target 플레이스홀더: ambiguity_map 키 우선, 없으면 obj 이름 그대로
    tgt_phs: list[str] = []
    seen = set()
    for obj in obj_list:
        if "cube" in obj:
            ph = "cube" if "cube" in amb_map else obj
        elif "cup" in obj:
            ph = "paper cup" if "paper cup" in amb_map else obj
        else:
            continue
        if ph not in seen:
            tgt_phs.append(ph)
            seen.add(ph)

    # destination 플레이스홀더
    box_phs: list[str] = []
    seen = set()
    for obj in obj_list:
        if "box" in obj:
            ph = "box" if "box" in amb_map else obj
            if ph not in seen:
                box_phs.append(ph)
                seen.add(ph)

    tasks = []
    for tgt in tgt_phs:
        for box in box_phs:
            for tmpl in TASK_TEMPLATES:
                tasks.append(tmpl.replace("{tgt}", "{" + tgt + "}").replace("{box}", "{" + box + "}"))
    return tasks


# ────────────────────────────────────────────────────────────────────────────
# 레이블링 루프
# ────────────────────────────────────────────────────────────────────────────

def show_scenes() -> None:
    print("\n  번호  씬 구성")
    print("  " + "─" * 55)
    for i, s in enumerate(SCENES, 1):
        print(f"  {i:2d})  {s['desc']}")
    print("  " + "─" * 55)

def select_scene() -> dict:
    show_scenes()
    while True:
        raw = input(f"\n  씬 번호 입력 (1~{len(SCENES)}, s=건너뜀): ").strip().lower()
        if raw == "s":
            return {}
        try:
            n = int(raw)
            if 1 <= n <= len(SCENES):
                return SCENES[n - 1]
        except ValueError:
            pass
        print(f"  ✗ 1~{len(SCENES)} 또는 s 입력")


def label_one(img_path: Path, img_index: int, total: int, raw_file: Path) -> bool:
    """이미지 1장 레이블링. 저장 성공 시 True."""
    # 이미지 정보
    try:
        from PIL import Image as PILImage
        with PILImage.open(img_path) as im:
            w, h = im.size
        img_info = f"{w}×{h}"
    except Exception:
        img_info = "크기 불명"

    print(f"\n{'═'*60}")
    print(f"  [{img_index}/{total}]  {img_path.name}  ({img_info})")
    print(f"  경로: {img_path.resolve()}")
    print(f"{'═'*60}")

    # 씬 선택
    scene = select_scene()
    if not scene:
        print("  건너뜀.")
        return False

    obj_list = scene["obj_list"]
    amb_map  = scene["ambiguity_map"]
    tasks    = generate_tasks(obj_list, amb_map)

    print(f"\n  obj_list      : {obj_list}")
    print(f"  ambiguity_map : {amb_map}")
    print(f"  태스크 {len(tasks)}개 자동 생성:")
    for t in tasks:
        print(f"    - {t}")

    confirm = input("\n  저장할까요? (Enter = 저장, r = 씬 재선택, s = 건너뜀): ").strip().lower()
    if confirm == "s":
        return False
    if confirm == "r":
        return label_one(img_path, img_index, total, raw_file)

    sample = {
        "id":            img_path.stem,
        "obj_list":      obj_list,
        "tasks":         tasks,
        "ambiguity_map": amb_map,
    }
    with open(raw_file, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return True


# ────────────────────────────────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AmbRes 데이터 레이블링")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    args = parser.parse_args()

    from ambres import ASSETS_DIR, DATA_DIR

    img_dir  = DATA_DIR.get_dir("real", args.mode)
    raw_file = ASSETS_DIR.get_dir("real", args.mode) / "data_raw.jsonl"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.touch()

    # 완료된 ID
    done_ids: set[str] = set()
    with open(raw_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    imgs = sorted(
        list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.jpg"))
    )
    remaining = [p for p in imgs if p.stem not in done_ids]

    print(f"\n{'='*60}")
    print(f"  AmbRes 레이블링  ({args.mode})")
    print(f"  이미지 경로: {img_dir}")
    print(f"  레이블 파일: {raw_file}")
    print(f"  전체 {len(imgs)}장  완료 {len(done_ids)}장  남은 {len(remaining)}장")
    print(f"  물체: 큐브×2 / 빨간상자 / 노란상자 / 종이컵×2")
    print(f"  역할: 큐브·종이컵=target  /  상자=destination")
    print(f"{'='*60}")
    print("  Ctrl+C = 중단 (재실행 시 이어서 진행)\n")

    if not remaining:
        print("  모두 레이블링 완료.")
        return

    labeled = 0
    try:
        for i, img_path in enumerate(remaining, 1):
            if label_one(img_path, i + len(done_ids), len(imgs), raw_file):
                labeled += 1
                print(f"  ✓ 저장 완료  (누적 {len(done_ids) + labeled}/{len(imgs)})")
    except (KeyboardInterrupt, EOFError):
        print("\n\n  중단됨. 재실행하면 이어서 진행합니다.")

    print(f"\n  이번 세션 {labeled}장 완료.")
    if args.mode == "train":
        print(f"\n  다음: python scripts/collect_and_label.py --mode test")
    else:
        print(f"\n  다음 단계:")
        print(f"  1. 샘플 생성: cd /workspace/AmbRes && python scripts/make_samples.py --env real")
        print(f"  2. 학습     : HF_HOME=/workspace/hf_cache python scripts/train.py --env real")


if __name__ == "__main__":
    main()
