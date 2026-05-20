#!/usr/bin/env python3
"""
CLI 기반 AmbRes 원시 데이터 레이블링 스크립트.

각 이미지에 대해 obj_list / tasks / ambiguity_map 을 입력하면
/workspace/AmbRes/assets/data/real/{mode}/data_raw.jsonl 에 누적 저장합니다.
중단 후 재실행하면 이미 레이블링된 이미지를 건너뜁니다.

사용법:
  python scripts/label_ambres_data.py --mode train
  python scripts/label_ambres_data.py --mode test  --image-dir ~/datasets/ambres/real/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/AmbRes")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ────────────────────────────────────────────────────────────────────────────
# 입력 헬퍼
# ────────────────────────────────────────────────────────────────────────────

def _safe_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt


def _input_str_list(prompt: str) -> list[str]:
    while True:
        raw = _safe_input(prompt)
        if raw:
            return [o.strip() for o in raw.split(",") if o.strip()]
        print("  ✗ 비어 있습니다. 다시 입력하세요.")


def _input_json(prompt: str, default=None):
    while True:
        raw = _safe_input(prompt)
        if not raw and default is not None:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("  ✗ JSON 형식 오류. 다시 입력하세요.")


# ────────────────────────────────────────────────────────────────────────────
# 검증
# ────────────────────────────────────────────────────────────────────────────

def _validate(sample: dict) -> list[str]:
    """레이블 일관성 검사. 오류 메시지 목록 반환 (비어 있으면 OK)."""
    errors = []
    obj_set = set(sample["obj_list"])
    amb_map: dict = sample["ambiguity_map"]

    for amb_key, instances in amb_map.items():
        for inst in instances:
            if inst not in obj_set:
                errors.append(
                    f"ambiguity_map['{amb_key}'] 의 '{inst}'이(가) obj_list에 없습니다."
                )

    import re
    for task in sample["tasks"]:
        placeholders = re.findall(r"\{([^}]+)\}", task)
        for ph in placeholders:
            if ph not in obj_set and ph not in amb_map:
                errors.append(
                    f"task '{task}' 의 플레이스홀더 '{{{ph}}}'이(가) "
                    f"obj_list 또는 ambiguity_map에 없습니다."
                )
    return errors


# ────────────────────────────────────────────────────────────────────────────
# 레이블링 루프
# ────────────────────────────────────────────────────────────────────────────

def label_image(img_id: str, img_path: Path, out_file: Path) -> bool:
    """이미지 1장 레이블링. 성공하면 True, 건너뜀이면 False."""
    print(f"\n{'─'*60}")
    print(f"  이미지 : {img_path}")
    print(f"  ID     : {img_id}")
    print(f"{'─'*60}")
    print("  (s = 이 이미지 건너뜀,  q = 전체 중단)")

    try:
        obj_list = _input_str_list(
            "\n  obj_list  (쉼표 구분, 예: red mug, blue mug, sprite bottle)\n  > "
        )

        print("\n  tasks  — pick-and-place 태스크를 JSON 배열로 입력합니다.")
        print('  플레이스홀더 예시: "Put the {mug} next to the {bottle}."')
        print('  예시 입력: ["Put the {mug} next to the {bottle}."]')
        tasks = _input_json("  > ", default=[])

        print("\n  ambiguity_map  — 씬에 같은 종류가 여러 개 있는 물체만 등록합니다.")
        print('  예시: {"mug": ["red mug", "blue mug"]}')
        print('  없으면 {} 입력')
        ambiguity_map = _input_json("  > ", default={})

    except KeyboardInterrupt:
        ans = _safe_input("\n  (s = 건너뜀, q = 전체 중단, Enter = 계속): ")
        if ans.lower() == "q":
            raise
        return False

    if not obj_list or _safe_input("  앞의 입력 중 's'를 치셨다면 건너뜁니다. 저장할까요? (y/n, 기본=y): ").lower() == "n":
        return False

    sample = {
        "id": img_id,
        "obj_list": obj_list,
        "tasks": tasks,
        "ambiguity_map": ambiguity_map,
    }

    # 검증
    errors = _validate(sample)
    if errors:
        print("\n  ⚠ 검증 오류:")
        for e in errors:
            print(f"    - {e}")
        if _safe_input("  그래도 저장하시겠습니까? (y/n, 기본=n): ").lower() != "y":
            print("  건너뜀.")
            return False

    with open(out_file, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  ✓ 저장 완료")
    return True


# ────────────────────────────────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AmbRes 원시 데이터 레이블링 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--env",  choices=["real", "sim"], default="real")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="이미지 디렉터리 (기본: ~/datasets/ambres/{env}/{mode})",
    )
    args = parser.parse_args()

    from ambres import ASSETS_DIR, DATA_DIR

    raw_file = ASSETS_DIR.get_dir(args.env, args.mode) / "data_raw.jsonl"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.touch()

    img_dir = (
        Path(args.image_dir).expanduser()
        if args.image_dir
        else DATA_DIR.get_dir(args.env, args.mode)
    )
    img_dir.mkdir(parents=True, exist_ok=True)

    # 이미 레이블링된 id
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
        list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.png"))
    )
    remaining = [p for p in imgs if p.stem not in done_ids]

    print(f"\n{'='*60}")
    print(f"  env={args.env}  mode={args.mode}")
    print(f"  이미지 디렉터리: {img_dir}")
    print(f"  출력 파일      : {raw_file}")
    print(f"  총 {len(imgs)}장  |  완료 {len(done_ids)}장  |  남은 {len(remaining)}장")
    print(f"{'='*60}")

    if not remaining:
        print("\n  모든 이미지가 이미 레이블링되어 있습니다.")
        return

    labeled = 0
    try:
        for i, img_path in enumerate(remaining, 1):
            print(f"\n[{i}/{len(remaining)}]", end="")
            if label_image(img_path.stem, img_path, raw_file):
                labeled += 1
    except KeyboardInterrupt:
        print("\n\n  중단됨. 재실행하면 이어서 진행합니다.")

    print(f"\n  이번 세션에서 {labeled}장 레이블링 완료.")
    print(f"  다음 단계: python /workspace/AmbRes/scripts/make_samples.py --env {args.env}")


if __name__ == "__main__":
    main()
