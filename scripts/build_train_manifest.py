#!/usr/bin/env python3
"""
scripts/build_train_manifest.py

kyne0127/ambres-training 데이터셋의 training 이미지를 GPU 서버에 다운받고
visualize_detector_sequence.py용 로컬 경로 manifest를 생성합니다.

- s3_trial_006 중복 항목을 자동 제거합니다.
- HF 경로 (data/SN/trial_XXX/*.png) → 로컬 경로 (/tmp/train_imgs/SN_trial_XXX_*.png) 로 재매핑

GPU 서버 실행:
    HF_HOME=/workspace/hf_cache python scripts/build_train_manifest.py \
        --img-dir /tmp/train_imgs \
        --out     /tmp/manifest_train_gpu.jsonl
"""

import argparse
import json
import re
from pathlib import Path

REPO_ID = "kyne0127/ambres-training"
TRAIN_MANIFEST = "manifest_train.jsonl"


def hf_path_from_manifest(manifest_img_path: str) -> str:
    """
    manifest의 img 경로 (data-training-v2/SN/trial_XXX/t0.png)
    → HF 실제 파일 경로 (data/SN/trial_XXX/t0.png)
    """
    p = manifest_img_path.replace("data-training-v2/", "data/")
    # 혹시 이미 data/ 로 시작하면 그대로
    return p


def local_filename(entry_id: str, frame: str) -> str:
    """
    s1_trial_001, t0 → train_s1_trial_001_t0.png
    """
    return f"train_{entry_id}_{frame}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="/tmp/train_imgs",
                        help="이미지 저장 디렉토리")
    parser.add_argument("--out", default="/tmp/manifest_train_gpu.jsonl",
                        help="출력 manifest 경로")
    parser.add_argument("--skip-download", action="store_true",
                        help="이미지가 이미 있으면 다운로드 건너뜀")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("pip install huggingface_hub"); return

    img_dir = Path(args.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    # manifest 로드
    cached = hf_hub_download(REPO_ID, repo_type="dataset", filename=TRAIN_MANIFEST)
    raw_entries = [json.loads(l) for l in open(cached, encoding="utf-8") if l.strip()]
    print(f"원본 항목: {len(raw_entries)}개")

    # 중복 제거: id 기준 첫 번째만 유지
    seen_ids: set[str] = set()
    entries = []
    for e in raw_entries:
        if e["id"] in seen_ids:
            print(f"  [중복 제거] {e['id']}")
            continue
        seen_ids.add(e["id"])
        entries.append(e)
    print(f"중복 제거 후: {len(entries)}개")

    # 이미지 다운로드 + 로컬 경로 재매핑
    out_entries = []
    for i, e in enumerate(entries):
        sid = e["id"]
        frame_map = {
            "t0": e["initial_img"],
            "c1": e["c1_img"],
            "c2": e["c2_img"],
        }
        local_paths = {}
        for frame, manifest_path in frame_map.items():
            hf_path = hf_path_from_manifest(manifest_path)
            local_file = img_dir / local_filename(sid, frame)

            if not local_file.exists() or not args.skip_download:
                try:
                    dl = hf_hub_download(
                        REPO_ID, repo_type="dataset",
                        filename=hf_path,
                        local_dir=str(img_dir),
                        local_dir_use_symlinks=False,
                    )
                    # hf_hub_download가 서브디렉토리에 저장하면 rename
                    dl_path = Path(dl)
                    if dl_path != local_file:
                        local_file.parent.mkdir(parents=True, exist_ok=True)
                        dl_path.rename(local_file)
                except Exception as ex:
                    print(f"  [경고] {sid} {frame} 다운로드 실패: {ex}")
                    local_file = Path(hf_path)  # fallback: HF 캐시 경로 사용
            local_paths[frame] = str(local_file)

        out_entry = {
            "id":                sid,
            "scenario":          e["scenario"],
            "task":              e["task"],
            "initial_img":       local_paths["t0"],
            "c1_img":            local_paths["c1"],
            "c2_img":            local_paths["c2"],
            "checkpoint":        e["checkpoint"],
            "gold_state":        e["gold_state"],
            "gold_decision":     e["gold_decision"],
            "target_label":      e["target_label"],
            "destination_label": e["destination_label"],
        }
        if e.get("notes"):
            out_entry["notes"] = e["notes"]
        out_entries.append(out_entry)

        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{len(entries)}")

    # 저장
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in out_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    from collections import Counter
    gd = Counter(e["gold_decision"] for e in out_entries)
    sc = Counter(e["scenario"] for e in out_entries)
    ck = Counter(e["checkpoint"] for e in out_entries)
    print(f"\n생성 완료: {out_path}  ({len(out_entries)}개)")
    print(f"gold_decision : {dict(gd)}")
    print(f"scenario      : {dict(sorted(sc.items()))}")
    print(f"checkpoint    : {dict(ck)}")
    print(f"\n예시:\n{json.dumps(out_entries[0], indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
