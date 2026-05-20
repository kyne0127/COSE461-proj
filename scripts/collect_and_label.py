#!/usr/bin/env python3
"""
AmbRes 재학습 데이터 수집 + 레이블링 스크립트
물체 세트: 큐브, 빨간 상자, 노란 상자, 종이컵

실행:
  python scripts/collect_and_label.py --mode train
  python scripts/collect_and_label.py --mode test

각 씬마다:
  1. ZED/USB 카메라로 이미지 캡처
  2. 씬에 어떤 물체가 있는지 선택
  3. 태스크 선택
  4. 자동으로 data_raw.jsonl에 저장
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, "/workspace/AmbRes")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ────────────────────────────────────────────────────────────────────────────
# 물체 세트 정의
# ────────────────────────────────────────────────────────────────────────────

OBJECTS = {
    "1": "cube",
    "2": "red box",
    "3": "yellow box",
    "4": "paper cup",
}

# 태스크 템플릿: {object} 플레이스홀더 사용
TASK_TEMPLATES = [
    "Put the {obj1} next to the {obj2}.",
    "Move the {obj1} on top of the {obj2}.",
    "Place the {obj1} inside the {obj2}.",
    "Move the {obj1} to the left of the {obj2}.",
    "Move the {obj1} to the right of the {obj2}.",
    "Pick up the {obj1} and place it near the {obj2}.",
]

# 씬에 있는 물체 조합 예시 (레이블링 시 선택)
SCENE_PRESETS = {
    "A": {
        "desc": "큐브 + 빨간 상자 + 노란 상자 + 종이컵 (4개 모두)",
        "obj_list": ["cube", "red box", "yellow box", "paper cup"],
        "ambiguity_map": {"box": ["red box", "yellow box"]},
    },
    "B": {
        "desc": "큐브 + 빨간 상자 + 종이컵",
        "obj_list": ["cube", "red box", "paper cup"],
        "ambiguity_map": {},
    },
    "C": {
        "desc": "큐브 + 노란 상자 + 종이컵",
        "obj_list": ["cube", "yellow box", "paper cup"],
        "ambiguity_map": {},
    },
    "D": {
        "desc": "빨간 상자 + 노란 상자 + 종이컵",
        "obj_list": ["red box", "yellow box", "paper cup"],
        "ambiguity_map": {"box": ["red box", "yellow box"]},
    },
    "E": {
        "desc": "큐브 + 빨간 상자 + 노란 상자",
        "obj_list": ["cube", "red box", "yellow box"],
        "ambiguity_map": {"box": ["red box", "yellow box"]},
    },
    "F": {
        "desc": "큐브 + 종이컵 (2개)",
        "obj_list": ["cube", "paper cup"],
        "ambiguity_map": {},
    },
    "G": {
        "desc": "직접 입력",
        "obj_list": None,
        "ambiguity_map": None,
    },
}


# ────────────────────────────────────────────────────────────────────────────
# 카메라 캡처
# ────────────────────────────────────────────────────────────────────────────

def capture_image(source: str, camera_index: int, img_save_path: Path) -> bool:
    """카메라에서 이미지 1장 캡처 후 저장. 성공하면 True."""
    import cv2
    import numpy as np
    from PIL import Image

    if source == "zed":
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from module.desktop.zed_connector import ZEDCapture
            cam = ZEDCapture(resolution="VGA", fps=30)
            for _ in range(5):  # warmup
                cam.capture_synchronized()
            rgb, _ = cam.capture_synchronized()
            cam.release()
        except Exception as e:
            print(f"  [오류] ZED 캡처 실패: {e}")
            return False
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"  [오류] USB 카메라 index={camera_index} 열기 실패")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(5):  # warmup
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("  [오류] 프레임 캡처 실패")
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_save_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_save_path), bgr)
    h, w = rgb.shape[:2]
    print(f"  ✓ 저장: {img_save_path}  ({w}×{h})")
    return True


# ────────────────────────────────────────────────────────────────────────────
# 씬 레이블링
# ────────────────────────────────────────────────────────────────────────────

def select_scene_preset() -> dict:
    print("\n  씬 구성 선택:")
    for key, preset in SCENE_PRESETS.items():
        print(f"    {key}) {preset['desc']}")
    while True:
        ans = input("  선택 (A~G): ").strip().upper()
        if ans in SCENE_PRESETS:
            preset = SCENE_PRESETS[ans]
            if preset["obj_list"] is not None:
                return dict(preset)
            # G: 직접 입력
            return direct_input()
        print("  ✗ A~G 중 선택하세요.")


def direct_input() -> dict:
    print("\n  물체 번호를 입력하세요:")
    for k, v in OBJECTS.items():
        print(f"    {k}) {v}")
    print("  (복수 선택: 1,2,3 또는 all)")
    while True:
        raw = input("  > ").strip().lower()
        if raw == "all":
            selected = list(OBJECTS.values())
            break
        nums = [x.strip() for x in raw.split(",")]
        if all(n in OBJECTS for n in nums):
            selected = [OBJECTS[n] for n in nums]
            break
        print("  ✗ 올바른 번호를 입력하세요.")

    # ambiguity_map 자동 생성
    amb_map = {}
    if "red box" in selected and "yellow box" in selected:
        amb_map["box"] = ["red box", "yellow box"]

    print(f"  obj_list      : {selected}")
    print(f"  ambiguity_map : {amb_map}")
    return {"obj_list": selected, "ambiguity_map": amb_map}


def select_tasks(obj_list: list[str], amb_map: dict) -> list[str]:
    """태스크 템플릿을 obj_list의 물체 쌍으로 채워 선택하도록 합니다."""
    from itertools import combinations

    # 사용 가능한 (obj1, obj2) 쌍 생성
    pairs = list(combinations(obj_list, 2))
    # ambiguous 물체는 'box' 형태 플레이스홀더로도 추가
    if amb_map:
        for amb_key in amb_map:
            for other in obj_list:
                if other not in amb_map.get(amb_key, []):
                    pairs.append((amb_key, other))
                    pairs.append((other, amb_key))

    # 중복 제거
    pairs = list(dict.fromkeys(pairs))

    # 태스크 후보 생성
    candidates = []
    for o1, o2 in pairs[:6]:  # 너무 많으면 잘라냄
        for tmpl in TASK_TEMPLATES[:3]:
            t = tmpl.replace("{obj1}", "{" + o1 + "}").replace("{obj2}", "{" + o2 + "}")
            candidates.append(t)

    print("\n  태스크 선택 (복수 선택 가능, 쉼표 구분 또는 all):")
    for i, t in enumerate(candidates, 1):
        print(f"    {i:2d}) {t}")

    while True:
        raw = input("  > ").strip().lower()
        if raw == "all":
            return candidates
        if raw == "s":
            return []
        try:
            nums = [int(x.strip()) for x in raw.split(",")]
            if all(1 <= n <= len(candidates) for n in nums):
                return [candidates[n - 1] for n in nums]
        except ValueError:
            pass
        print(f"  ✗ 1~{len(candidates)} 중 선택하세요. (s = 건너뜀)")


# ────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AmbRes 데이터 수집 + 레이블링")
    parser.add_argument("--mode",   choices=["train", "test"], default="train")
    parser.add_argument("--source", choices=["zed", "usb", "skip"], default="zed",
                        help="카메라 소스 (skip = 카메라 없이 레이블만)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--n-scenes", type=int, default=0,
                        help="촬영할 씬 수 (0 = Ctrl+C까지 무한)")
    args = parser.parse_args()

    from ambres import ASSETS_DIR, DATA_DIR

    img_dir  = DATA_DIR.get_dir("real", args.mode)
    raw_file = ASSETS_DIR.get_dir("real", args.mode) / "data_raw.jsonl"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.touch()

    # 완료된 ID 로드
    done_ids: set[str] = set()
    with open(raw_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    print(f"\n{'='*60}")
    print(f"  AmbRes 데이터 수집  ({args.mode})")
    print(f"  이미지 저장: {img_dir}")
    print(f"  레이블 파일: {raw_file}")
    print(f"  완료된 씬  : {len(done_ids)}개")
    print(f"  물체 세트  : 큐브 / 빨간 상자 / 노란 상자 / 종이컵")
    print(f"{'='*60}")
    print("  Ctrl+C 로 중단, 재실행 시 이어서 진행\n")

    scene_count = 0
    try:
        while args.n_scenes == 0 or scene_count < args.n_scenes:
            scene_count += 1
            img_id = str(uuid.uuid4()).replace("-", "")[:22]
            img_path = img_dir / f"{img_id}.jpeg"

            print(f"\n{'─'*60}")
            print(f"  씬 #{scene_count}  ID={img_id}")
            print(f"{'─'*60}")

            # 1. 카메라 캡처
            if args.source != "skip":
                input(f"\n  [1] 씬을 구성하고 Enter 입력 (캡처)... ")
                ok = capture_image(args.source, args.camera_index, img_path)
                if not ok:
                    print("  캡처 실패 — 이 씬 건너뜀")
                    continue
            else:
                print(f"\n  [1] 카메라 skip 모드 — 이미지 없이 레이블만 생성")
                img_path.touch()  # 빈 파일 생성

            # 2. 씬 구성 선택
            print("\n  [2] 이 씬의 물체 구성을 선택하세요:")
            preset = select_scene_preset()
            obj_list   = preset["obj_list"]
            amb_map    = preset["ambiguity_map"]
            print(f"  obj_list      : {obj_list}")
            print(f"  ambiguity_map : {amb_map}")

            # 3. 태스크 선택
            print("\n  [3] 이 씬에 적합한 태스크를 선택하세요 (all 또는 번호):")
            tasks = select_tasks(obj_list, amb_map)
            if not tasks:
                print("  태스크 없음 — 이 씬 건너뜀")
                img_path.unlink(missing_ok=True)
                continue
            print(f"  선택된 태스크 {len(tasks)}개")

            # 4. 저장
            sample = {
                "id": img_id,
                "obj_list": obj_list,
                "tasks": tasks,
                "ambiguity_map": amb_map,
            }
            with open(raw_file, "a") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            print(f"\n  ✓ 저장 완료  (누적: {len(done_ids) + scene_count}개)")

            ans = input("\n  계속하시겠습니까? (Enter = 계속, q = 종료): ").strip().lower()
            if ans == "q":
                break

    except KeyboardInterrupt:
        print("\n\n  중단됨.")

    total = len(done_ids) + scene_count
    print(f"\n  총 {total}개 씬 완료.")
    print(f"\n  다음 단계:")
    print(f"  1. 반대 모드도 수집: python scripts/collect_and_label.py --mode test")
    print(f"  2. 샘플 생성: cd /workspace/AmbRes && python scripts/make_samples.py --env real")
    print(f"  3. 학습: HF_HOME=/workspace/hf_cache python scripts/train.py --env real")


if __name__ == "__main__":
    main()
