#!/usr/bin/env python3
"""
scripts/prepare_annotation_images.py

manifest_eval.jsonl의 t0 이미지를 CVAT 업로드용 flat 디렉토리로 복사합니다.
파일명에 sample_id가 포함되어 어노테이션 후 매핑이 쉽습니다.

GPU 서버 실행:
    python /workspace/COSE461-proj/scripts/prepare_annotation_images.py
    python /workspace/COSE461-proj/scripts/prepare_annotation_images.py --also-checkpoints
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",  default=None)
    parser.add_argument("--out-dir",   default="/tmp/cvat_upload")
    parser.add_argument("--also-checkpoints", action="store_true",
                        help="t0 외에 c1/c2도 포함 (어노테이션 량 3배)")
    parser.add_argument("--repo-id",   default="kyne0127/vla-evaluation")
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parent.parent
    manifest  = args.manifest or str(proj_root / "dataset" / "manifest_eval.jsonl")
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    # 파일명 충돌 방지를 위해 sample_id 기반 이름 사용
    keys = ["initial_img"]
    if args.also_checkpoints:
        keys += ["c1_img", "c2_img"]

    key_suffix = {"initial_img": "t0", "c1_img": "c1", "c2_img": "c2"}

    copied, skipped = 0, 0
    index_rows = []

    for entry in entries:
        sid = entry["id"]
        for key in keys:
            src_path = entry.get(key, "")
            if not src_path:
                continue

            # 로컬 경로 우선, 없으면 HF 다운로드
            img_path = _resolve_path(src_path, args.repo_id)
            if img_path is None:
                print(f"  [skip] {sid}/{key_suffix[key]}: 이미지 없음")
                skipped += 1
                continue

            suffix = Path(img_path).suffix
            dst_name = f"{sid}__{key_suffix[key]}{suffix}"
            dst_path = out_dir / dst_name
            shutil.copy2(img_path, dst_path)
            copied += 1

            index_rows.append({
                "filename": dst_name,
                "id": sid,
                "image_key": key_suffix[key],
                "scenario": entry.get("scenario"),
                "task": entry.get("task"),
                "target_label": entry.get("target_label"),
                "destination_label": entry.get("destination_label"),
            })

    # CVAT 업로드 후 어노테이션 매핑에 쓸 인덱스 파일 저장
    index_path = out_dir / "image_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_rows, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {copied}장 복사, {skipped}장 스킵")
    print(f"출력 디렉토리: {out_dir}")
    print(f"인덱스 파일:   {index_path}")
    print(f"\nCVAT 업로드 방법:")
    print(f"  1. https://app.cvat.ai 접속")
    print(f"  2. Tasks → + Create task")
    print(f"  3. Labels 탭: cup / red box / cube / box 추가 (Type: rectangle)")
    print(f"  4. Files: {out_dir}/*.png 전체 드래그 업로드")
    print(f"  5. Submit → 어노테이션 시작")


def _resolve_path(manifest_path: str, repo_id: str) -> str | None:
    if os.path.exists(manifest_path):
        return manifest_path

    m = re.search(r'(data[-_]evaluation/.+|data[-_]training/.+|images/.+)',
                  manifest_path)
    if not m:
        return None

    rel = m.group(1).replace("data_evaluation", "data-evaluation")

    # GPU 서버의 vla-evaluation 디렉토리 직접 확인
    server_base = Path("/workspace/COSE461-proj/dataset/vla-evaluation")
    candidate = server_base / rel
    if candidate.exists():
        return str(candidate)

    # HF 캐시에서 다운로드
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel)
        return local
    except Exception as e:
        print(f"  [warn] HF download 실패 ({rel}): {e}")
        return None


if __name__ == "__main__":
    main()
