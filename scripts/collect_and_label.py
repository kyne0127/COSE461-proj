#!/usr/bin/env python3
"""
AmbRes 데이터 레이블링 스크립트 (그라운딩 검증 포함)
물체: 큐브×2, 빨간상자, 노란상자, 종이컵×2
역할: 큐브/종이컵 = target  |  빨간상자/노란상자 = destination

실행:
  python scripts/collect_and_label.py --mode train
  python scripts/collect_and_label.py --mode train --host localhost --port 50051
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/AmbRes")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ────────────────────────────────────────────────────────────────────────────
# obj_list / ambiguity_map 생성
# ────────────────────────────────────────────────────────────────────────────

def _s(targets, boxes):
    obj_list = targets + boxes
    amb = {}
    if len([t for t in targets if "cube" in t]) >= 2:
        amb["cube"] = [t for t in targets if "cube" in t]
    if len([t for t in targets if "cup" in t]) >= 2:
        amb["paper cup"] = [t for t in targets if "cup" in t]
    if len(boxes) >= 2:
        amb["box"] = boxes
    return obj_list, amb


# ────────────────────────────────────────────────────────────────────────────
# 태스크 자동 생성
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
    tgt_phs: list[str] = []
    seen: set[str] = set()
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

    box_phs: list[str] = []
    seen = set()
    for obj in obj_list:
        if "box" in obj:
            ph = "box" if "box" in amb_map else obj
            if ph not in seen:
                box_phs.append(ph)
                seen.add(ph)

    tasks: list[str] = []
    if box_phs:
        for tgt in tgt_phs:
            for box in box_phs:
                for tmpl in TASK_TEMPLATES_PLACE:
                    tasks.append(
                        tmpl.replace("{tgt}", "{" + tgt + "}")
                            .replace("{box}", "{" + box + "}")
                    )
    else:
        for tgt in tgt_phs:
            for tmpl in TASK_TEMPLATES_PICK:
                tasks.append(tmpl.replace("{tgt}", "{" + tgt + "}"))
    return tasks


# ────────────────────────────────────────────────────────────────────────────
# 씬 선택 (터미널 입력)
# ────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str, choices: list[str]) -> str:
    while True:
        raw = input(prompt).strip().lower()
        if raw in choices:
            return raw
        print(f"  ✗ {'/'.join(choices)} 중 입력하세요.")


def select_scene() -> dict:
    print()
    n_cube = _ask("  1) 큐브 개수    (0 / 1 / 2 / s=건너뜀): ", ["0", "1", "2", "s"])
    if n_cube == "s":
        return {}

    has_red = _ask("  2) 빨간 상자    (y / n): ", ["y", "n"])
    has_yel = _ask("  3) 노란 상자    (y / n): ", ["y", "n"])
    n_cup   = _ask("  4) 종이컵 개수  (0 / 1 / 2): ", ["0", "1", "2"])

    targets: list[str] = []
    if n_cube == "2":
        targets += ["left cube", "right cube"]
    elif n_cube == "1":
        targets += ["cube"]
    if n_cup == "2":
        targets += ["left paper cup", "right paper cup"]
    elif n_cup == "1":
        targets += ["paper cup"]

    boxes: list[str] = []
    if has_red == "y":
        boxes.append("red box")
    if has_yel == "y":
        boxes.append("yellow box")

    obj_list, amb_map = _s(targets, boxes)

    desc_parts = []
    if n_cube != "0": desc_parts.append(f"큐브×{n_cube}")
    if has_red == "y": desc_parts.append("빨간상자")
    if has_yel == "y": desc_parts.append("노란상자")
    if n_cup != "0": desc_parts.append(f"종이컵×{n_cup}")

    return {"desc": " + ".join(desc_parts), "obj_list": obj_list, "ambiguity_map": amb_map}


# ────────────────────────────────────────────────────────────────────────────
# 그라운딩 검증 (서버 detect → annotated 이미지 저장)
# ────────────────────────────────────────────────────────────────────────────

COLORS = ["#FF4444", "#44BB44", "#4488FF", "#FFAA00", "#FF44FF", "#00CCCC"]


def verify_grounding(
    img_path: Path,
    obj_list: list[str],
    host: str = "localhost",
    port: int = 50051,
) -> tuple[dict, Path] | None:
    """
    서버 detect 메서드로 obj_list 각 물체의 픽셀 좌표 감지.
    annotated 이미지 저장 후 (detections, annotated_path) 반환.
    실패 시 None 반환.
    """
    import numpy as np
    from PIL import Image as PILImage, ImageDraw

    try:
        from module.desktop.generic_client import GenericClient
    except ImportError:
        print("  [경고] GenericClient import 실패 — 그라운딩 건너뜀")
        return None

    try:
        img = PILImage.open(img_path).convert("RGB")
        img_arr = np.array(img)

        with GenericClient(host=host, port=port, timeout=60.0) as client:
            result = client.infer(
                handler_id="ambres",
                method="detect",
                payload={"objects": obj_list},
                images={"image": img_arr},
                session_id="labeling",
            )
        detections: dict = result.get("detections", {})
    except Exception as e:
        print(f"  [경고] 서버 감지 실패: {e}")
        return None

    # ── Annotated 이미지 생성 ────────────────────────────────────────────────
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = img.size
    r = max(8, w // 55)  # 점 반지름

    for i, (obj, coords) in enumerate(detections.items()):
        color = COLORS[i % len(COLORS)]
        valid = [c for c in coords if c and len(c) >= 2]
        for j, pt in enumerate(valid):
            x, y = int(pt[0]), int(pt[1])
            # 외곽 원
            draw.ellipse([x - r*2, y - r*2, x + r*2, y + r*2],
                         outline=color, width=3)
            # 중심 점
            draw.ellipse([x - r//2, y - r//2, x + r//2, y + r//2],
                         fill=color)
            # 레이블
            label = f"{obj}[{j}]" if len(valid) > 1 else obj
            draw.text((x + r*2 + 4, y - r), label, fill=color)

    ann_path = img_path.parent / f"annotated_{img_path.stem}.png"
    annotated.save(ann_path)

    return detections, ann_path


def print_detections(detections: dict) -> None:
    """감지 결과를 터미널에 보기 좋게 출력."""
    print()
    print(f"  {'물체':<22}  {'개수':>4}  좌표")
    print(f"  {'─'*22}  {'─'*4}  {'─'*30}")
    for obj, coords in detections.items():
        valid = [c for c in coords if c and len(c) >= 2]
        cnt = len(valid)
        flag = "⚠ 복수!" if cnt > 1 else ("✓" if cnt == 1 else "✗ 없음")
        coord_str = "  ".join(f"({int(c[0])},{int(c[1])})" for c in valid[:4])
        if len(valid) > 4:
            coord_str += f"  ...+{len(valid)-4}"
        print(f"  {obj:<22}  {cnt:>2}개 {flag:<6}  {coord_str}")
    print()


# ────────────────────────────────────────────────────────────────────────────
# 이미지 1장 레이블링
# ────────────────────────────────────────────────────────────────────────────

def label_one(
    img_path: Path,
    img_index: int,
    total: int,
    raw_file: Path,
    host: str,
    port: int,
) -> bool:
    try:
        from PIL import Image as PILImage
        with PILImage.open(img_path) as im:
            w, h = im.size
        img_info = f"{w}×{h}"
    except Exception:
        img_info = "크기 불명"

    print(f"\n{'═'*65}")
    print(f"  [{img_index}/{total}]  {img_path.name}  ({img_info})")
    print(f"  경로: {img_path.resolve()}")
    print(f"{'═'*65}")

    while True:
        # ── 1. 씬 선택 ────────────────────────────────────────────────────
        scene = select_scene()
        if not scene:
            print("  건너뜀.")
            return False

        obj_list = scene["obj_list"]
        amb_map  = scene["ambiguity_map"]
        tasks    = generate_tasks(obj_list, amb_map)

        print(f"\n  obj_list      : {obj_list}")
        print(f"  ambiguity_map : {amb_map}")
        print(f"  태스크 {len(tasks)}개 자동 생성")

        # ── 2. 그라운딩 검증 ───────────────────────────────────────────────
        print("\n  그라운딩 검증 중 (서버 detect)...")
        grounding = verify_grounding(img_path, obj_list, host=host, port=port)

        if grounding:
            detections, ann_path = grounding
            print(f"  annotated 이미지: {ann_path}")
            print_detections(detections)

            # 미감지 물체 경고
            missing = [o for o in obj_list if o not in detections or
                       not [c for c in detections[o] if c and len(c) >= 2]]
            if missing:
                print(f"  ⚠ 감지 실패 물체: {missing}")

            action = input(
                "  그라운딩 확인:\n"
                "    Enter = 저장  /  r = 재감지  /  n = 씬 재선택  /  s = 건너뜀\n"
                "  > "
            ).strip().lower()
        else:
            print("  그라운딩 서버 연결 실패 (좌표 없이 저장 가능)")
            action = input(
                "    Enter = 저장  /  n = 씬 재선택  /  s = 건너뜀\n"
                "  > "
            ).strip().lower()

        if action == "s":
            return False
        if action == "n":
            continue  # 씬 재선택
        if action == "r":
            # 재감지만: 씬 변경 없이 grounding 재실행
            print("\n  재감지 중...")
            grounding = verify_grounding(img_path, obj_list, host=host, port=port)
            if grounding:
                detections, ann_path = grounding
                print(f"  annotated 이미지: {ann_path}")
                print_detections(detections)
            action2 = input(
                "    Enter = 저장  /  n = 씬 재선택  /  s = 건너뜀\n"
                "  > "
            ).strip().lower()
            if action2 == "s":
                return False
            if action2 == "n":
                continue

        # ── 3. 저장 ───────────────────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description="AmbRes 데이터 레이블링 (그라운딩 검증)")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--host", default="localhost", help="gRPC 서버 호스트")
    parser.add_argument("--port", type=int, default=50051, help="gRPC 서버 포트")
    args = parser.parse_args()

    from ambres import ASSETS_DIR, DATA_DIR

    img_dir  = DATA_DIR.get_dir("real", args.mode)
    raw_file = ASSETS_DIR.get_dir("real", args.mode) / "data_raw.jsonl"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.touch()

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

    print(f"\n{'='*65}")
    print(f"  AmbRes 레이블링 + 그라운딩 검증  ({args.mode})")
    print(f"  이미지 경로 : {img_dir}")
    print(f"  레이블 파일 : {raw_file}")
    print(f"  gRPC 서버   : {args.host}:{args.port}")
    print(f"  전체 {len(imgs)}장  완료 {len(done_ids)}장  남은 {len(remaining)}장")
    print(f"  물체: 큐브×2 / 빨간상자 / 노란상자 / 종이컵×2")
    print(f"{'='*65}")
    print("  Ctrl+C = 중단 (재실행 시 이어서 진행)\n")

    if not remaining:
        print("  모두 레이블링 완료.")
        return

    labeled = 0
    try:
        for i, img_path in enumerate(remaining, 1):
            if label_one(img_path, i + len(done_ids), len(imgs), raw_file,
                         host=args.host, port=args.port):
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
