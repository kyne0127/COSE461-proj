#!/usr/bin/env python3
"""
scripts/build_eval_manifest.py

HuggingFace vla-evaluation의 모든 60개 eval trial에 대해
GPU 서버 경로 기준 manifest.jsonl을 생성합니다.

로컬 실행:
    python scripts/build_eval_manifest.py \
        --image-dir /tmp/gdino_all_imgs \
        --out manifest_gpu.jsonl
"""

import argparse
import json
import shutil
from pathlib import Path

REPO_ID = "kyne0127/vla-evaluation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="/tmp/gdino_all_imgs",
                        help="GPU 서버의 이미지 캐시 디렉토리")
    parser.add_argument("--out", default="manifest_gpu.jsonl")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("pip install huggingface_hub")
        return

    # HF에서 manifest.jsonl 다운로드
    cached = hf_hub_download(REPO_ID, repo_type="dataset", filename="dataset/manifest.jsonl")
    hf_entries = [json.loads(l) for l in open(cached, encoding="utf-8") if l.strip()]
    print(f"HF manifest: {len(hf_entries)}개 항목")

    img_dir = Path(args.image_dir)
    out_entries = []
    missing = []

    for e in hf_entries:
        sid = e["id"]  # e.g. "s1_trial_001"

        # GPU 서버의 이미지 경로로 재매핑
        t0  = img_dir / f"eval_{sid}_t0.png"
        c1  = img_dir / f"eval_{sid}_c1.png"
        c2  = img_dir / f"eval_{sid}_c2.png"

        entry = {
            "id":                e["id"],
            "scenario":          e["scenario"],
            "task":              e["task"],
            "initial_img":       str(t0),
            "c1_img":            str(c1),
            "c2_img":            str(c2),
            "checkpoint":        e["checkpoint"],
            "gold_state":        e["gold_state"],
            "gold_decision":     e["gold_decision"],
            "target_label":      e["target_label"],
            "destination_label": e["destination_label"],
        }
        if e.get("notes"):
            entry["notes"] = e["notes"]

        out_entries.append(entry)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in out_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 분포 요약
    from collections import Counter
    dec_dist = Counter(e["gold_decision"] for e in out_entries)
    ckpt_dist = Counter(e["checkpoint"] for e in out_entries)
    print(f"\n생성 완료: {out_path}  ({len(out_entries)}개)")
    print(f"gold_decision: {dict(dec_dist)}")
    print(f"checkpoint:    {dict(ckpt_dist)}")
    print(f"\n예시 (첫 항목):")
    print(json.dumps(out_entries[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
