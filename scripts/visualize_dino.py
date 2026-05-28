#!/usr/bin/env python3
"""
DINO 검출 결과를 이미지에 시각화합니다.

기본 사용:
    python scripts/visualize_dino.py <image> --objects cup "red box" --out /tmp/vis.png

여러 threshold 비교 (나란히 출력):
    python scripts/visualize_dino.py <image> --objects cup "red box" \
        --thresholds 0.10 0.20 0.35 --out /tmp/vis_compare.png

데이터셋 전체 trial 일괄 시각화:
    python scripts/visualize_dino.py --manifest dataset/hf_manifest.jsonl \
        --scenarios S1 S2 --box-threshold 0.20 --out-dir /tmp/dino_vis/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
_REPO = Path(__file__).resolve().parent.parent
_AMBRES = _REPO.parent / "AmbRes"
for p in (_SRC, _REPO, _AMBRES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    "#FF4444", "#44BB44", "#4488FF", "#FFAA00",
    "#FF44FF", "#00CCCC", "#FF8844", "#88FF44",
]


def _font(size: int = 13):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_detections(
    pil_img: Image.Image,
    raw_dets: list[dict],
    objects: list[str],
    threshold_label: str = "",
) -> Image.Image:
    """raw_dets: detect_raw() 반환값. 모든 bbox를 색+레이블+score로 그림."""
    obj_color = {obj: PALETTE[i % len(PALETTE)] for i, obj in enumerate(objects)}
    unmatched_color = "#888888"

    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = _font(13)
    font_small = _font(11)

    # 헤더: threshold 표시
    if threshold_label:
        W, H = img.size
        bar_h = 22
        new = Image.new("RGB", (W, H + bar_h), (30, 30, 30))
        new.paste(img, (0, bar_h))
        d2 = ImageDraw.Draw(new)
        d2.text((6, 4), threshold_label, fill="white", font=font)
        img = new
        draw = ImageDraw.Draw(img)

    for det in raw_dets:
        x1, y1, x2, y2 = det["box"]
        if threshold_label:
            y1 += 22; y2 += 22  # 헤더 offset
        score = det["score"]
        dino_lbl = det["dino_label"]
        matched = det["matched"]

        color = obj_color.get(matched, unmatched_color) if matched else unmatched_color

        # bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # 중심 점
        cx, cy = det["center"]
        if threshold_label:
            cy += 22
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        # 레이블 태그
        tag = f"{dino_lbl} {score:.2f}"
        if matched and matched != dino_lbl:
            tag += f" →{matched}"
        tw = draw.textlength(tag, font=font_small) if hasattr(draw, "textlength") else len(tag) * 6
        draw.rectangle([x1, y1 - 16, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), tag, fill="black", font=font_small)

    # 우측 상단: 객체별 감지 수 요약
    counts = {obj: sum(1 for d in raw_dets if d["matched"] == obj) for obj in objects}
    summary = "  ".join(f"{obj}: {n}" for obj, n in counts.items())
    W_img, _ = img.size
    sw = draw.textlength(summary, font=font) if hasattr(draw, "textlength") else len(summary) * 7
    draw.rectangle([W_img - sw - 8, 4, W_img - 2, 20], fill=(30, 30, 30, 180))
    draw.text((W_img - sw - 6, 4), summary, fill="white", font=font)

    return img


def visualize_single(
    image_path: str | Path,
    objects: list[str],
    box_threshold: float,
    text_threshold: float,
    out_path: str | Path,
) -> None:
    from module.models.ambres.dino_detector import DINODetector

    pil = Image.open(image_path).convert("RGB")
    detector = DINODetector(device="cuda", box_threshold=box_threshold, text_threshold=text_threshold)
    raw = detector.detect_raw(pil, objects)

    label = "box={:.2f} text={:.2f}  |  {}".format(
        box_threshold, text_threshold, Path(image_path).name
    )
    annotated = _draw_detections(pil, raw, objects, threshold_label=label)
    annotated.save(out_path)
    print(f"saved → {out_path}  ({len(raw)} detections)")


def visualize_compare(
    image_path: str | Path,
    objects: list[str],
    thresholds: list[float],
    out_path: str | Path,
) -> None:
    """여러 threshold 결과를 가로로 나란히 비교."""
    from module.models.ambres.dino_detector import DINODetector

    pil = Image.open(image_path).convert("RGB")
    panels = []
    for thr in thresholds:
        detector = DINODetector(device="cuda", box_threshold=thr, text_threshold=thr * 0.75)
        raw = detector.detect_raw(pil, objects)
        label = "box={:.2f} text={:.2f}  ({} dets)".format(thr, thr * 0.75, len(raw))
        panels.append(_draw_detections(pil, raw, objects, threshold_label=label))

    W = panels[0].width
    H = panels[0].height
    combined = Image.new("RGB", (W * len(panels), H), (20, 20, 20))
    for i, panel in enumerate(panels):
        combined.paste(panel, (i * W, 0))
    combined.save(out_path)
    print(f"saved → {out_path}  ({len(thresholds)} panels)")


def visualize_manifest(
    manifest_path: str | Path,
    objects_override: list[str] | None,
    scenarios: list[str] | None,
    box_threshold: float,
    text_threshold: float,
    out_dir: Path,
    checkpoints: list[str],
) -> None:
    """manifest의 각 trial에 대해 t0/c1/c2 이미지를 시각화."""
    import json
    from module.models.ambres.dino_detector import DINODetector

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(manifest_path)
    base_dir = manifest.parent
    samples = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                samples.append(json.loads(line))

    if scenarios:
        samples = [s for s in samples if s.get("scenario") in scenarios]

    detector = DINODetector(device="cuda", box_threshold=box_threshold, text_threshold=text_threshold)
    print(f"Loaded DINODetector (box={box_threshold}, text={text_threshold})")
    print(f"Processing {len(samples)} trials → {out_dir}\n")

    ckpt_key = {"t0": "initial_img", "c1": "c1_img", "c2": "c2_img"}

    for sample in samples:
        sid = sample.get("id", "?")
        scenario = sample.get("scenario", "?")
        objects = objects_override or [
            sample.get("target_label", ""),
            sample.get("destination_label", ""),
        ]
        objects = [o for o in objects if o]

        for ckpt in checkpoints:
            img_rel = sample.get(ckpt_key[ckpt])
            if not img_rel:
                continue
            img_path = (base_dir / img_rel) if not Path(img_rel).is_absolute() else Path(img_rel)
            if not img_path.exists():
                print(f"  [SKIP] {img_path} not found")
                continue

            pil = Image.open(img_path).convert("RGB")
            raw = detector.detect_raw(pil, objects)
            label = "[{}] {} {}  box={:.2f}  |  objects: {}".format(
                scenario, sid, ckpt.upper(), box_threshold, ", ".join(objects)
            )
            annotated = _draw_detections(pil, raw, objects, threshold_label=label)
            out_name = "{}_{}_{}.png".format(sid.replace("/", "_"), scenario, ckpt)
            out_path = out_dir / out_name
            annotated.save(out_path)

        n_by_obj = {
            obj: sum(1 for r in detector.detect_raw(Image.open(
                (base_dir / sample.get(ckpt_key["t0"])).resolve()
            ).convert("RGB"), [obj]) if r["matched"] == obj)
            for obj in objects
        } if objects else {}
        print(f"  {sid} ({scenario}): objects={objects}")

    print(f"\n완료 → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DINO bbox detections on images.")
    # 단일 이미지 모드
    parser.add_argument("image", nargs="?", default="", help="이미지 경로")
    parser.add_argument("--objects", nargs="+", default=[], help="검출할 객체 이름")
    parser.add_argument("--out", default="", help="출력 이미지 경로 (단일 이미지)")

    # 비교 모드
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        help="비교할 box threshold 값들 (지정 시 나란히 비교 이미지 생성)"
    )

    # manifest 모드
    parser.add_argument("--manifest", default="", help="manifest.jsonl 경로")
    parser.add_argument("--scenarios", nargs="+", help="시각화할 시나리오 (예: S1 S2)")
    parser.add_argument("--checkpoints", nargs="+", default=["t0", "c1"],
                        choices=["t0", "c1", "c2"], help="시각화할 체크포인트")
    parser.add_argument("--out-dir", default="/tmp/dino_vis", help="출력 디렉터리 (manifest 모드)")

    # 공통 threshold
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.15)

    args = parser.parse_args()

    if args.manifest:
        visualize_manifest(
            args.manifest,
            objects_override=args.objects or None,
            scenarios=args.scenarios,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            out_dir=Path(args.out_dir),
            checkpoints=args.checkpoints,
        )
    elif args.image:
        if not args.objects:
            parser.error("--objects 가 필요합니다.")
        if args.thresholds:
            out = args.out or "/tmp/dino_compare.png"
            visualize_compare(args.image, args.objects, args.thresholds, out)
        else:
            out = args.out or "/tmp/dino_vis.png"
            visualize_single(
                args.image, args.objects,
                args.box_threshold, args.text_threshold, out
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
