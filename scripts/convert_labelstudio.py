#!/usr/bin/env python3
"""
Label Studio JSON export → AmbRes data_raw.jsonl 변환 스크립트

사용법:
  python scripts/convert_labelstudio.py \
    --input  project-1-at-2026-05-21-11-18-3a51227e.json \
    --output /workspace/AmbRes/assets/data/real/train/data_raw.jsonl \
    --mode   train
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/AmbRes")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ────────────────────────────────────────────────────────────────────────────
# 태스크 자동 생성 (collect_and_label.py와 동일)
# ────────────────────────────────────────────────────────────────────────────

TASK_TEMPLATES_PLACE = [
    "Put the {tgt} next to the {box}.",
    "Place the {tgt} inside the {box}.",
    "Move the {tgt} to the {box}.",
    "Pick up the {tgt} and place it near the {box}.",
]
TASK_TEMPLATES_PICK = [
    "Pick up the {tgt}.",
    "Grasp the {tgt}.",
    "Bring me the {tgt}.",
]


def generate_tasks(obj_list: list[str], amb_map: dict) -> list[str]:
    tgt_phs, seen = [], set()
    for obj in obj_list:
        if "cube" in obj:
            ph = "cube" if "cube" in amb_map else obj
        elif "cup" in obj:
            ph = "paper cup" if "paper cup" in amb_map else obj
        else:
            continue
        if ph not in seen:
            tgt_phs.append(ph); seen.add(ph)

    box_phs, seen = [], set()
    for obj in obj_list:
        if "box" in obj:
            ph = "box" if "box" in amb_map else obj
            if ph not in seen:
                box_phs.append(ph); seen.add(ph)

    tasks = []
    if box_phs:
        for tgt in tgt_phs:
            for box in box_phs:
                for tmpl in TASK_TEMPLATES_PLACE:
                    tasks.append(
                        tmpl.replace("{tgt}", "{"+tgt+"}").replace("{box}", "{"+box+"}")
                    )
    else:
        for tgt in tgt_phs:
            for tmpl in TASK_TEMPLATES_PICK:
                tasks.append(tmpl.replace("{tgt}", "{"+tgt+"}"))
    return tasks


# ────────────────────────────────────────────────────────────────────────────
# Label Studio 좌표 변환
# ────────────────────────────────────────────────────────────────────────────

def pct_to_px(x_pct: float, y_pct: float, w: int, h: int) -> list[int]:
    """Label Studio % 좌표 → 픽셀 좌표."""
    return [int(x_pct / 100 * w), int(y_pct / 100 * h)]


def extract_image_id(file_upload: str) -> str:
    """
    '3c321cfd-20260520_125334_scene001.jpeg'
    → '20260520_125334_scene001'
    """
    name = Path(file_upload).stem          # remove .jpeg
    # UUID prefix 제거 (8hex-... 패턴)
    m = re.match(r'^[0-9a-f]+-(.+)$', name)
    return m.group(1) if m else name


# ────────────────────────────────────────────────────────────────────────────
# 메인 변환
# ────────────────────────────────────────────────────────────────────────────

def convert(input_path: str, output_path: str, mode: str) -> None:
    with open(input_path) as f:
        tasks = json.load(f)

    records = []

    for task in tasks:
        img_id = extract_image_id(task["file_upload"])

        # 어노테이션이 없는 태스크 건너뜀
        if not task.get("annotations"):
            print(f"  [skip] {img_id} — 어노테이션 없음")
            continue

        result = task["annotations"][0]["result"]

        # label → [(x_px, y_px), ...] 수집
        raw: dict[str, list[list[int]]] = {}
        for ann in result:
            v = ann["value"]
            label = v["keypointlabels"][0]
            w = ann["original_width"]
            h = ann["original_height"]
            pt = pct_to_px(v["x"], v["y"], w, h)
            raw.setdefault(label, []).append(pt)

        # ── obj_list / ambiguity_map / grounding 구성 ─────────────────────
        obj_list: list[str] = []
        ambiguity_map: dict[str, list[str]] = {}
        grounding: dict[str, list[list[int]]] = {}

        # cube 처리 (x좌표 정렬 → left/right)
        cubes = sorted(raw.get("cube", []), key=lambda p: p[0])
        if len(cubes) == 2:
            obj_list += ["left cube", "right cube"]
            ambiguity_map["cube"] = ["left cube", "right cube"]
            grounding["left cube"]  = [cubes[0]]
            grounding["right cube"] = [cubes[1]]
        elif len(cubes) == 1:
            obj_list.append("cube")
            grounding["cube"] = cubes

        # paper cup 처리 (x좌표 정렬 → left/right)
        cups = sorted(raw.get("paper cup", []), key=lambda p: p[0])
        if len(cups) == 2:
            obj_list += ["left paper cup", "right paper cup"]
            ambiguity_map["paper cup"] = ["left paper cup", "right paper cup"]
            grounding["left paper cup"]  = [cups[0]]
            grounding["right paper cup"] = [cups[1]]
        elif len(cups) == 1:
            obj_list.append("paper cup")
            grounding["paper cup"] = cups

        # box 처리
        has_red    = "red box"    in raw
        has_yellow = "yellow box" in raw
        if has_red:
            obj_list.append("red box")
            grounding["red box"] = raw["red box"]
        if has_yellow:
            obj_list.append("yellow box")
            grounding["yellow box"] = raw["yellow box"]
        if has_red and has_yellow:
            ambiguity_map["box"] = ["red box", "yellow box"]

        # tasks 생성
        tasks_list = generate_tasks(obj_list, ambiguity_map)

        if not tasks_list:
            print(f"  [warn] {img_id} — task 없음 (obj_list={obj_list})")

        record = {
            "id":            img_id,
            "obj_list":      obj_list,
            "tasks":         tasks_list,
            "ambiguity_map": ambiguity_map,
            "grounding":     grounding,
        }
        records.append(record)
        print(f"  ✓ {img_id}  obj={obj_list}  tasks={len(tasks_list)}")

    # 저장
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n  총 {len(records)}개 → {out}")


# ────────────────────────────────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Label Studio JSON → data_raw.jsonl")
    parser.add_argument("--input",  required=True, help="Label Studio export JSON 경로")
    parser.add_argument("--output", required=True, help="출력 data_raw.jsonl 경로")
    parser.add_argument("--mode",   choices=["train", "test"], default="train")
    args = parser.parse_args()

    print(f"\n  변환 시작: {args.input} → {args.output}\n")
    convert(args.input, args.output, args.mode)


if __name__ == "__main__":
    main()
