#!/usr/bin/env python3
"""
AmbRes 파인튠 모델 평가 스크립트 (gRPC 서버 사용)

테스트셋 이미지 + data.json으로 정확도 측정.

사용법:
  python scripts/eval_ambres.py
  python scripts/eval_ambres.py --scene scene040        # 특정 씬만
  python scripts/eval_ambres.py --max-per-scene 3       # 씬당 최대 3개 태스크
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from module.desktop.generic_client import GenericClient


IMG_DIR   = Path("/root/datasets/ambres/real/test")
DATA_JSON = Path("/workspace/AmbRes/assets/data/real/test/data.json")
ADAPTER   = "AB5siP8DA78aA78wR5Y8Mw"


def load_image_as_float32(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return (np.array(img) / 255.0).astype(np.float32)


def run_eval(args):
    client = GenericClient(host="localhost", port=50051)
    client.connect()

    # 핸들러 로드 (이미 서버에 로드돼 있으면 no-op)
    ok = client.load_handler("ambres", config={
        "model_type":   "finetune",
        "adapter_ckpt": ADAPTER,
        "use_dino":     True,
        "use_detection": False,
        "save_dir":     "/tmp/eval_captures",
    })
    print(f"load_handler: {ok}")

    with open(DATA_JSON) as f:
        all_samples = json.load(f)

    # 씬별로 그룹핑
    scenes: dict[str, list] = {}
    for s in all_samples:
        scenes.setdefault(s["id"], []).append(s)

    if args.scene:
        target = [k for k in scenes if args.scene in k]
        if not target:
            print(f"씬 '{args.scene}' 을 찾을 수 없습니다.")
            return
        scenes = {k: scenes[k] for k in target}

    tp = tn = fp = fn = 0
    results = []

    for scene_id, samples in scenes.items():
        img_path = IMG_DIR / f"{scene_id}.jpeg"
        if not img_path.exists():
            print(f"[skip] 이미지 없음: {img_path}")
            continue

        img_arr = load_image_as_float32(img_path)

        # amb/non-amb 각각 최대 N개
        amb_samples     = [s for s in samples if s["ambiguity_bool"]][:args.max_per_scene]
        nonamb_samples  = [s for s in samples if not s["ambiguity_bool"]][:args.max_per_scene]
        selected = amb_samples + nonamb_samples

        print(f"\n{'─'*60}")
        print(f"씬: {scene_id}  (amb={len(amb_samples)}, non-amb={len(nonamb_samples)})")

        for s in selected:
            task     = s["task_description"]
            expected = s["ambiguity_bool"]

            # 세션 초기화
            client.infer(handler_id="ambres", method="reset", session_id="eval")
            t0 = time.time()
            resp = client.infer(
                handler_id="ambres",
                method="query",
                payload={"task_description": task},
                images={"cam": img_arr},
                session_id="eval",
            )
            elapsed = time.time() - t0

            if not resp.get("success"):
                print(f"  [ERR] {task[:55]}")
                continue

            predicted = resp["task_ambiguous"]
            correct   = (predicted == expected)

            if expected and predicted:      tp += 1
            elif not expected and not predicted: tn += 1
            elif not expected and predicted: fp += 1
            elif expected and not predicted: fn += 1

            mark = "✓" if correct else "✗"
            print(f"  {mark} [{elapsed:.1f}s]  "
                  f"exp={'AMB' if expected else 'OK '}  "
                  f"got={'AMB' if predicted else 'OK '}  "
                  f"| {task[:55]}")
            if not correct:
                print(f"      objects={resp.get('task_objects')}  "
                      f"q={resp.get('clarifying_question','')[:60]}")
            results.append({
                "scene": scene_id, "task": task,
                "expected": expected, "predicted": predicted,
                "correct": correct,
            })

    total = tp + tn + fp + fn
    if total == 0:
        print("\n결과 없음")
        return

    acc  = (tp + tn) / total * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) else float("nan")
    rec  = tp / (tp + fn) * 100 if (tp + fn) else float("nan")

    print(f"\n{'═'*60}")
    print(f"결과:  총 {total}개  정확도 {acc:.1f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Precision={prec:.1f}%  Recall={rec:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",          default="",  help="특정 씬 ID (부분 매칭)")
    p.add_argument("--max-per-scene",  type=int, default=2,
                   help="씬당 amb/non-amb 각각 최대 몇 개 테스트할지 (기본 2)")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
