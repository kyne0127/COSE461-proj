#!/usr/bin/env python3
"""
scripts/download_cvat_images.py

manifest_eval.jsonl의 t0 이미지를 HuggingFace에서 로컬로 다운로드합니다.
파일명에 sample_id가 포함되어 CVAT 업로드 후 매핑이 쉽습니다.

로컬 실행:
    pip install huggingface_hub
    python scripts/download_cvat_images.py
    python scripts/download_cvat_images.py --out-dir C:/tmp/cvat_upload
"""

import json
import re
import shutil
from pathlib import Path

REPO_ID = "kyne0127/vla-evaluation"


def hf_path(manifest_path: str) -> str | None:
    """로컬 경로에서 HuggingFace 파일 경로 추출."""
    m = re.search(r'(data[-_]eval(?:uation)?/.+)', manifest_path)
    if not m:
        return None
    return m.group(1).replace("data_eval/", "data-evaluation/").replace("data_evaluation/", "data-evaluation/")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out-dir",  default="C:/tmp/cvat_upload")
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parent.parent
    manifest  = args.manifest or str(proj_root / "dataset" / "manifest_eval.jsonl")
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub가 설치되지 않았습니다.")
        print("  pip install huggingface_hub")
        return

    index_rows = []
    ok, fail = 0, 0

    for entry in entries:
        sid = entry["id"]
        src = entry.get("initial_img", "")
        rel = hf_path(src)
        if rel is None:
            print(f"  [skip] {sid}: 경로 파싱 실패 ({src})")
            fail += 1
            continue

        suffix = Path(rel).suffix or ".png"
        dst_name = f"{sid}__t0{suffix}"
        dst_path = out_dir / dst_name

        if dst_path.exists():
            print(f"  [exist] {dst_name}")
            ok += 1
        else:
            try:
                cached = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=rel,
                )
                shutil.copy2(cached, dst_path)
                print(f"  [ok] {dst_name}")
                ok += 1
            except Exception as e:
                print(f"  [fail] {sid}: {e}")
                fail += 1
                continue

        index_rows.append({
            "filename":          dst_name,
            "id":                sid,
            "scenario":          entry.get("scenario"),
            "task":              entry.get("task"),
            "target_label":      entry.get("target_label"),
            "destination_label": entry.get("destination_label"),
        })

    index_path = out_dir / "image_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_rows, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {ok}장 다운로드, {fail}장 실패")
    print(f"출력 디렉토리: {out_dir}")
    print(f"\nCVAT 업로드 방법:")
    print(f"  1. https://app.cvat.ai → Tasks → + Create task")
    print(f"  2. Labels 탭: cup / red box / cube / box  (Type: rectangle)")
    print(f"  3. Files: {out_dir}/*.png 전체 선택 후 드래그 업로드")
    print(f"  4. Submit → 어노테이션 시작")


if __name__ == "__main__":
    main()
