#!/usr/bin/env python3
"""
trial별 t0/c1/c2 이미지를 가로로 합쳐 하나의 이미지로 저장합니다.

사용법:
    python scripts/combine_trial_vis.py \
        --src-dir /tmp/vis_combined_raw \
        --out-dir /workspace/vis_combined \
        --scenarios S1 S2 S3
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CHECKPOINT_ORDER = ["t0", "c1", "c2"]
LABELS = {"t0": "T0 (initial)", "c1": "C1 (pre-pick)", "c2": "C2 (pre-place)"}


def _font(size=13):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            from PIL import ImageFont
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    from PIL import ImageFont
    return ImageFont.load_default()


def combine_trial(images: dict[str, Path], out_path: Path, trial_id: str) -> None:
    """t0/c1/c2 이미지를 가로로 붙여 저장."""
    panels = []
    for ckpt in CHECKPOINT_ORDER:
        if ckpt not in images:
            continue
        img = Image.open(images[ckpt]).convert("RGB")
        panels.append((ckpt, img))

    if not panels:
        return

    W, H = panels[0][1].size
    BAR = 28
    combined = Image.new("RGB", (W * len(panels), H + BAR), (20, 20, 20))
    draw = ImageDraw.Draw(combined)
    font = _font(13)

    for i, (ckpt, img) in enumerate(panels):
        combined.paste(img, (i * W, BAR))
        label = LABELS.get(ckpt, ckpt.upper())
        draw.text((i * W + 6, 6), label, fill="white", font=font)

    # trial ID in center
    tid_w = draw.textlength(trial_id, font=font) if hasattr(draw, "textlength") else len(trial_id) * 7
    draw.text((W * len(panels) // 2 - tid_w // 2, 6), trial_id, fill="#AAAAAA", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    print(f"  saved → {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir",   required=True)
    parser.add_argument("--out-dir",   required=True)
    parser.add_argument("--scenarios", nargs="+", default=None)
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 파일명 패턴: {trial_id}_{scenario}_{ckpt}.png
    # 예: s1_trial_001_S1_t0.png
    pattern = re.compile(r"^(.+)_(S\d+)_(t0|c1|c2)\.png$")

    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for f in sorted(src.glob("*.png")):
        m = pattern.match(f.name)
        if not m:
            continue
        trial_id, scenario, ckpt = m.group(1), m.group(2), m.group(3)
        if args.scenarios and scenario not in args.scenarios:
            continue
        grouped.setdefault((trial_id, scenario), {})[ckpt] = f

    print(f"{len(grouped)} trials found in {src}")
    for (trial_id, scenario), images in sorted(grouped.items()):
        out_name = f"{trial_id}_{scenario}_combined.png"
        combine_trial(images, out / out_name, f"{trial_id} ({scenario})")

    print(f"\n완료 → {out}")


if __name__ == "__main__":
    main()
