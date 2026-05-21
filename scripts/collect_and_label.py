#!/usr/bin/env python3
"""
AmbRes 데이터 레이블링 + 그라운딩 검증 스크립트
물체: 큐브×2, 빨간상자, 노란상자, 종이컵×2

실행:
  python scripts/collect_and_label.py --mode train           # 신규 레이블링
  python scripts/collect_and_label.py --mode train --verify  # 기존 레이블 그라운딩 검증
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/AmbRes")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ────────────────────────────────────────────────────────────────────────────
# obj_list / ambiguity_map 생성 헬퍼
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
                    tasks.append(tmpl.replace("{tgt}", "{"+tgt+"}").replace("{box}", "{"+box+"}"))
    else:
        for tgt in tgt_phs:
            for tmpl in TASK_TEMPLATES_PICK:
                tasks.append(tmpl.replace("{tgt}", "{"+tgt+"}"))
    return tasks


# ────────────────────────────────────────────────────────────────────────────
# 씬 선택 (신규 레이블링 시)
# ────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str, choices: list[str]) -> str:
    while True:
        raw = input(prompt).strip().lower()
        if raw in choices:
            return raw
        print(f"  ✗ {'/'.join(choices)} 중 입력하세요.")


def select_scene() -> dict:
    print()
    n_cube  = _ask("  1) 큐브 개수    (0/1/2/s=건너뜀): ", ["0","1","2","s"])
    if n_cube == "s": return {}
    has_red = _ask("  2) 빨간 상자    (y/n): ", ["y","n"])
    has_yel = _ask("  3) 노란 상자    (y/n): ", ["y","n"])
    n_cup   = _ask("  4) 종이컵 개수  (0/1/2): ", ["0","1","2"])

    targets = []
    if n_cube == "2": targets += ["left cube", "right cube"]
    elif n_cube == "1": targets += ["cube"]
    if n_cup == "2": targets += ["left paper cup", "right paper cup"]
    elif n_cup == "1": targets += ["paper cup"]

    boxes = []
    if has_red == "y": boxes.append("red box")
    if has_yel == "y": boxes.append("yellow box")

    obj_list, amb_map = _s(targets, boxes)
    desc_parts = []
    if n_cube != "0": desc_parts.append(f"큐브×{n_cube}")
    if has_red == "y": desc_parts.append("빨간상자")
    if has_yel == "y": desc_parts.append("노란상자")
    if n_cup != "0": desc_parts.append(f"종이컵×{n_cup}")
    return {"desc": " + ".join(desc_parts), "obj_list": obj_list, "ambiguity_map": amb_map}


# ────────────────────────────────────────────────────────────────────────────
# 그라운딩 검증
# ────────────────────────────────────────────────────────────────────────────

COLORS = ["#FF4444", "#44BB44", "#4488FF", "#FFAA00", "#FF44FF", "#00CCCC"]


def verify_grounding(img_path: Path, obj_list: list[str],
                     host: str = "localhost", port: int = 50051):
    """detect 호출 → annotated 이미지 저장 → (detections, ann_path) 반환."""
    import numpy as np
    from PIL import Image as PILImage, ImageDraw

    try:
        from module.desktop.generic_client import GenericClient
    except ImportError:
        print("  [경고] GenericClient 없음 — 그라운딩 건너뜀")
        return None

    try:
        img = PILImage.open(img_path).convert("RGB")
        with GenericClient(host=host, port=port, timeout=60.0) as client:
            result = client.infer(
                handler_id="ambres",
                method="detect",
                payload={"objects": obj_list},
                images={"image": np.array(img)},
                session_id="labeling",
            )
        detections: dict = result.get("detections", {})
    except Exception as e:
        print(f"  [경고] 서버 감지 실패: {e}")
        return None

    # annotated 이미지 생성
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = img.size
    r = max(8, w // 55)
    for i, (obj, coords) in enumerate(detections.items()):
        color = COLORS[i % len(COLORS)]
        valid = [c for c in coords if c and len(c) >= 2]
        for j, pt in enumerate(valid):
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse([x-r*2, y-r*2, x+r*2, y+r*2], outline=color, width=3)
            draw.ellipse([x-r//2, y-r//2, x+r//2, y+r//2], fill=color)
            label = f"{obj}[{j}]" if len(valid) > 1 else obj
            draw.text((x+r*2+4, y-r), label, fill=color)

    ann_path = img_path.parent / f"annotated_{img_path.stem}.png"
    annotated.save(ann_path)
    return detections, ann_path


def show_detections(detections: dict, obj_list: list[str]) -> None:
    print()
    print(f"  {'물체':<24}  {'개수':>4}  좌표")
    print(f"  {'─'*24}  {'─'*4}  {'─'*36}")
    for obj in obj_list:
        coords = detections.get(obj, [])
        valid  = [c for c in coords if c and len(c) >= 2]
        cnt    = len(valid)
        flag   = "⚠ 복수!" if cnt > 1 else ("✓" if cnt == 1 else "✗ 미감지")
        coord_str = "  ".join(f"[{i}]({int(c[0])},{int(c[1])})" for i, c in enumerate(valid[:4]))
        if len(valid) > 4: coord_str += f" +{len(valid)-4}"
        print(f"  {obj:<24}  {cnt:>2}개 {flag:<7}  {coord_str}")
    print()


def correct_detections(detections: dict, obj_list: list[str]) -> dict:
    """각 물체의 감지 좌표를 사용자가 수정. 수정된 grounding dict 반환."""
    print("  ── 좌표 수정 ─────────────────────────────────────────────")
    print("  각 물체에 대해 올바른 좌표를 선택하세요.")
    print("  Enter=전부유지  숫자=인덱스선택  d=직접입력  x=삭제\n")

    corrected: dict = {}
    for obj in obj_list:
        coords = detections.get(obj, [])
        valid  = [c for c in coords if c and len(c) >= 2]

        coord_str = "  ".join(f"[{i}]({int(c[0])},{int(c[1])})" for i, c in enumerate(valid))
        flag = "⚠ 복수!" if len(valid) > 1 else ("✓" if len(valid) == 1 else "✗ 미감지")
        print(f"  {obj}  →  {len(valid)}개 {flag}  {coord_str}")

        raw = input("    > ").strip().lower()

        if raw == "" :
            corrected[obj] = [[int(c[0]), int(c[1])] for c in valid]
        elif raw == "x":
            corrected[obj] = []
            print(f"    → {obj}: 삭제됨")
        elif raw == "d":
            coords_in = input("    좌표 입력 (x,y 형식, 여러 개는 세미콜론으로: 342,212;477,191): ").strip()
            pts = []
            for part in coords_in.split(";"):
                part = part.strip()
                if "," in part:
                    x, y = part.split(",", 1)
                    pts.append([int(x.strip()), int(y.strip())])
            corrected[obj] = pts
            print(f"    → {obj}: {pts}")
        else:
            # 인덱스 선택 (쉼표로 여러 개 가능: "0,1")
            try:
                indices = [int(i.strip()) for i in raw.split(",") if i.strip().isdigit()]
                chosen = [[int(valid[i][0]), int(valid[i][1])] for i in indices if i < len(valid)]
                corrected[obj] = chosen
                print(f"    → {obj}: {chosen}")
            except Exception:
                corrected[obj] = [[int(c[0]), int(c[1])] for c in valid]
                print(f"    → 입력 오류, 원본 유지")
        print()

    return corrected


# ────────────────────────────────────────────────────────────────────────────
# 이미지 1장 처리 (기존 레이블 사용 + 그라운딩 검증)
# ────────────────────────────────────────────────────────────────────────────

def process_one(img_path: Path, img_index: int, total: int,
                existing: dict | None, raw_file: Path,
                host: str, port: int) -> bool:
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

    # 기존 레이블 사용 or 신규 선택
    if existing:
        obj_list = existing["obj_list"]
        amb_map  = existing["ambiguity_map"]
        tasks    = existing.get("tasks") or generate_tasks(obj_list, amb_map)
        print(f"\n  기존 레이블:")
        print(f"    obj_list      : {obj_list}")
        print(f"    ambiguity_map : {amb_map}")
        need_scene = False
    else:
        need_scene = True

    while True:
        if need_scene:
            scene = select_scene()
            if not scene:
                return False
            obj_list = scene["obj_list"]
            amb_map  = scene["ambiguity_map"]
            tasks    = generate_tasks(obj_list, amb_map)
            print(f"\n  obj_list      : {obj_list}")
            print(f"  ambiguity_map : {amb_map}")
            print(f"  태스크 {len(tasks)}개 자동 생성")
            need_scene = False

        # 그라운딩 검증
        print("\n  그라운딩 검증 중...")
        result = verify_grounding(img_path, obj_list, host=host, port=port)

        grounding: dict = {}
        if result:
            detections, ann_path = result
            print(f"  annotated: {ann_path}")
            show_detections(detections, obj_list)
            missing = [o for o in obj_list
                       if not [c for c in detections.get(o, []) if c and len(c) >= 2]]
            if missing:
                print(f"  ⚠ 미감지: {missing}")
        else:
            detections = {}
            print("  (그라운딩 없이 진행)")

        action = input(
            "  확인: Enter=저장  c=좌표수정후저장  r=재감지  n=씬재선택  s=건너뜀\n  > "
        ).strip().lower()

        if action == "s":
            return False
        if action == "r":
            continue
        if action == "n":
            need_scene = True
            continue
        if action == "c":
            grounding = correct_detections(detections, obj_list)
        elif detections:
            # Enter: 자동 감지 결과 그대로 저장
            grounding = {
                obj: [[int(c[0]), int(c[1])] for c in detections.get(obj, [])
                      if c and len(c) >= 2]
                for obj in obj_list
            }

        # 저장
        sample = {
            "id":            img_path.stem,
            "obj_list":      obj_list,
            "tasks":         tasks,
            "ambiguity_map": amb_map,
        }
        if grounding:
            sample["grounding"] = grounding
        _rewrite_or_append(raw_file, sample)
        return True


def _rewrite_or_append(raw_file: Path, sample: dict) -> None:
    """동일 id가 있으면 덮어쓰고, 없으면 추가."""
    lines = []
    replaced = False
    if raw_file.exists():
        with open(raw_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("id") == sample["id"]:
                        lines.append(json.dumps(sample, ensure_ascii=False))
                        replaced = True
                    else:
                        lines.append(line)
                except Exception:
                    lines.append(line)
    if not replaced:
        lines.append(json.dumps(sample, ensure_ascii=False))
    with open(raw_file, "w") as f:
        f.write("\n".join(lines) + "\n")


# ────────────────────────────────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AmbRes 레이블링 + 그라운딩 검증")
    parser.add_argument("--mode",   choices=["train", "test"], default="train")
    parser.add_argument("--verify", action="store_true",
                        help="기존 레이블에 대해 그라운딩 검증만 실행")
    parser.add_argument("--host",   default="localhost")
    parser.add_argument("--port",   type=int, default=50051)
    args = parser.parse_args()

    from ambres import ASSETS_DIR, DATA_DIR

    img_dir  = DATA_DIR.get_dir("real", args.mode)
    raw_file = ASSETS_DIR.get_dir("real", args.mode) / "data_raw.jsonl"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.touch()

    # 기존 레이블 로드
    existing_map: dict[str, dict] = {}
    with open(raw_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    existing_map[e["id"]] = e
                except Exception:
                    pass

    imgs = sorted(
        list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.jpg"))
    )

    if args.verify:
        # 검증 모드: 기존 레이블이 있는 이미지만
        targets = [p for p in imgs if p.stem in existing_map]
        mode_label = "그라운딩 검증"
    else:
        # 신규 모드: 레이블 없는 이미지만
        targets = [p for p in imgs if p.stem not in existing_map]
        mode_label = "신규 레이블링"

    print(f"\n{'='*65}")
    print(f"  AmbRes {mode_label}  ({args.mode})")
    print(f"  이미지 경로 : {img_dir}")
    print(f"  레이블 파일 : {raw_file}")
    print(f"  gRPC 서버   : {args.host}:{args.port}")
    print(f"  전체 {len(imgs)}장  기존 레이블 {len(existing_map)}장  대상 {len(targets)}장")
    print(f"{'='*65}")
    print("  Ctrl+C = 중단\n")

    if not targets:
        print("  처리할 이미지 없음.")
        return

    done = 0
    try:
        for i, img_path in enumerate(targets, 1):
            existing = existing_map.get(img_path.stem)
            if process_one(img_path, i, len(targets), existing, raw_file,
                           host=args.host, port=args.port):
                done += 1
                print(f"  ✓ 완료  ({done}/{len(targets)})")
    except (KeyboardInterrupt, EOFError):
        print("\n\n  중단됨.")

    print(f"\n  이번 세션 {done}장 완료.")
    if not args.verify:
        print(f"\n  검증: python scripts/collect_and_label.py --mode {args.mode} --verify")
    else:
        print(f"\n  다음 단계:")
        print(f"  1. 샘플 생성: cd /workspace/AmbRes && python scripts/make_samples.py --env real")
        print(f"  2. 학습     : HF_HOME=/workspace/hf_cache python scripts/train.py --env real")


if __name__ == "__main__":
    main()
