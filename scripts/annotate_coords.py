#!/usr/bin/env python3
"""
scripts/annotate_coords.py
==========================
이미지에서 target / destination 좌표를 클릭으로 annotate하는 GUI 툴.

조작법:
  마우스 왼쪽 클릭  →  현재 레이블(target 또는 destination) 좌표 지정
  마우스 오른쪽 클릭 → 마지막 포인트 취소
  TAB               →  레이블 전환 (target ↔ destination)
  ENTER             →  현재 이미지 저장 후 다음 이미지
  r                 →  현재 이미지 annotation 재시작
  q                 →  저장 후 종료

출력:
  1. annotations/<image_stem>.json  — G₀ 포맷 + Molmo training 포맷
  2. annotations/manifest_annot.jsonl — 전체 annotation 목록

사용 예:
  # 단일 이미지
  python scripts/annotate_coords.py --images data_eval/S2/trial_001/t0.png \\
      --target-label cup --destination-label box

  # 디렉터리 일괄
  python scripts/annotate_coords.py --dir data_eval/S2 --target-label cup --destination-label box

  # manifest.jsonl의 initial_img 일괄
  python scripts/annotate_coords.py --from-manifest dataset/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── 색상 팔레트 ─────────────────────────────────────────────────────────────────
_COLORS = {
    "target":      (50,  200, 255),   # 노란-주황 (BGR)
    "destination": (50,  255,  80),   # 초록 (BGR)
}
_RADIUS = 8


# ── 데이터 클래스 ────────────────────────────────────────────────────────────────
class Annotation:
    def __init__(self, image_path: Path, target_label: str, destination_label: str):
        self.image_path    = image_path
        self.target_label  = target_label
        self.dest_label    = destination_label
        self.points: dict[str, list[tuple[int, int]]] = {
            "target":      [],
            "destination": [],
        }
        self.active_role = "target"   # 현재 클릭 레이블

    def add(self, x: int, y: int):
        self.points[self.active_role].append((x, y))

    def pop(self):
        pts = self.points[self.active_role]
        if pts:
            pts.pop()

    def reset(self):
        self.points = {"target": [], "destination": []}
        self.active_role = "target"

    def toggle_role(self):
        self.active_role = "destination" if self.active_role == "target" else "target"

    def label_of(self, role: str) -> str:
        return self.target_label if role == "target" else self.dest_label

    def to_dict(self, image_shape: tuple[int, int]) -> dict:
        """G₀ 포맷 + Molmo training 포맷으로 직렬화."""
        h, w = image_shape

        def coords_entry(role: str) -> dict:
            pts = self.points[role]
            if not pts:
                return {"label": self.label_of(role), "coord": None, "all_coords": []}
            # 첫 번째 클릭 = primary coord (G₀ extractor 기준)
            primary = list(pts[0])
            return {
                "label":      self.label_of(role),
                "coord":      primary,
                "all_coords": [list(p) for p in pts],
            }

        def molmo_point_tokens(role: str) -> list[dict]:
            """Molmo fine-tuning 포맷: x/y를 0-100 정규화."""
            pts = self.points[role]
            label = self.label_of(role)
            tokens = []
            for px, py in pts:
                tokens.append({
                    "label": label,
                    "x_norm": round(px / w * 100, 2),
                    "y_norm": round(py / h * 100, 2),
                    "x_px":  px,
                    "y_px":  py,
                })
            return tokens

        return {
            # ── G₀ 포맷 (consistency_monitor / pipeline 입력) ──
            "g0": {
                "target":      coords_entry("target"),
                "destination": coords_entry("destination"),
                "image_shape": [h, w],
            },
            # ── Molmo training 포맷 ──
            "molmo_training": {
                "target":      molmo_point_tokens("target"),
                "destination": molmo_point_tokens("destination"),
                "prompt_target":      f"Point to the {self.target_label}.",
                "prompt_destination": f"Point to the {self.dest_label}.",
            },
            # ── 메타 ──
            "image_path": str(self.image_path),
            "target_label":      self.target_label,
            "destination_label": self.dest_label,
        }


# ── 렌더링 ────────────────────────────────────────────────────────────────────

def render(bgr: np.ndarray, annot: Annotation) -> np.ndarray:
    canvas = bgr.copy()
    h, w   = canvas.shape[:2]

    # 이미 찍힌 포인트 렌더링
    for role, pts in annot.points.items():
        color = _COLORS[role]
        label = annot.label_of(role)
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (x, y), _RADIUS, color, 2)
            cv2.circle(canvas, (x, y), 3, color, -1)
            cv2.putText(canvas, f"{label}[{i}] ({x},{y})",
                        (x + _RADIUS + 3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 상단 오버레이
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

    active_col = _COLORS[annot.active_role]
    active_lbl = annot.label_of(annot.active_role)

    lines = [
        f"파일: {annot.image_path.name}",
        f"현재 레이블: {annot.active_role.upper()} = '{active_lbl}'  "
        f"(TAB으로 전환)",
        f"target={len(annot.points['target'])}pt  "
        f"destination={len(annot.points['destination'])}pt  |  "
        f"[ENTER] 저장·다음   [r] 초기화   [우클릭] 취소   [q] 종료",
    ]
    for i, ln in enumerate(lines):
        col = active_col if i == 1 else (220, 220, 220)
        cv2.putText(canvas, ln, (8, 22 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    return canvas


# ── 이미지 단위 annotation 루프 ────────────────────────────────────────────────

def annotate_image(image_path: Path, target_label: str, dest_label: str,
                   out_dir: Path, existing: Optional[dict] = None) -> Optional[dict]:
    """
    단일 이미지를 annotation.
    Returns: annotation dict (저장 완료 시) or None (건너뜀 / q 종료).
    """
    bgr_orig = cv2.imread(str(image_path))
    if bgr_orig is None:
        print(f"[WARN] 이미지 열기 실패: {image_path}")
        return None

    h, w = bgr_orig.shape[:2]
    annot = Annotation(image_path, target_label, dest_label)

    # 기존 annotation 복원
    if existing:
        g0 = existing.get("g0", {})
        for role in ("target", "destination"):
            coords = g0.get(role, {}).get("all_coords", [])
            for c in coords:
                if c:
                    annot.points[role].append(tuple(c))

    win = "annotate_coords"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    scale = min(1280 / w, 800 / h, 1.5)
    cv2.resizeWindow(win, int(w * scale), int(h * scale))

    result: Optional[dict] = None
    quit_all = False

    def on_mouse(event, x, y, flags, _):
        nonlocal annot
        # 실제 이미지 좌표로 역변환
        win_w = cv2.getWindowImageRect(win)[2]
        win_h = cv2.getWindowImageRect(win)[3]
        px = int(x * w / win_w)
        py = int(y * h / win_h)
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))

        if event == cv2.EVENT_LBUTTONDOWN:
            annot.add(px, py)
            print(f"  [{annot.active_role}] ({px}, {py})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            annot.pop()

    cv2.setMouseCallback(win, on_mouse)

    while True:
        frame = render(bgr_orig, annot)
        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            quit_all = True
            break
        elif key == 13:   # ENTER
            result = annot.to_dict((h, w))
            break
        elif key == ord('r'):
            annot.reset()
        elif key == 9:    # TAB
            annot.toggle_role()
            print(f"  → 레이블 전환: {annot.active_role} ({annot.label_of(annot.active_role)})")

    cv2.destroyAllWindows()

    if result:
        out_path = out_dir / f"{image_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  저장 → {out_path}")

    return None if quit_all else result


# ── 이미지 목록 수집 ──────────────────────────────────────────────────────────

def collect_images(args: argparse.Namespace) -> list[tuple[Path, str, str]]:
    """(image_path, target_label, destination_label) 리스트 반환."""
    items: list[tuple[Path, str, str]] = []

    if args.from_manifest:
        manifest = Path(args.from_manifest)
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                img = Path(row["initial_img"])
                if img.exists():
                    items.append((img, row["target_label"], row["destination_label"]))
                else:
                    print(f"[WARN] 이미지 없음: {img}")
        return items

    tgt  = args.target_label
    dest = args.destination_label

    if args.images:
        for p in args.images:
            items.append((Path(p), tgt, dest))
    elif args.dir:
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in sorted(Path(args.dir).rglob(ext)):
                if "annotated" not in p.name:
                    items.append((p, tgt, dest))

    return items


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    items = collect_images(args)
    if not items:
        print("annotation 대상 이미지가 없습니다.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_annot.jsonl"

    print(f"\n총 {len(items)}개 이미지 annotation 시작")
    print(f"출력 디렉터리: {out_dir}")
    print(f"조작: 왼쪽 클릭=포인트 추가 | TAB=레이블 전환 | ENTER=저장·다음 | r=초기화 | q=종료\n")

    completed = 0
    for i, (img_path, tgt, dest) in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {img_path.name}  target={tgt}  dest={dest}")

        # 기존 annotation 로드 (재개 지원)
        existing_path = out_dir / f"{img_path.stem}.json"
        existing = None
        if existing_path.exists() and not args.force:
            with open(existing_path, encoding="utf-8") as f:
                existing = json.load(f)
            print(f"  기존 annotation 로드 (r 키로 초기화 가능)")

        result = annotate_image(img_path, tgt, dest, out_dir, existing)

        if result is None and not (existing_path.exists() and not args.force):
            print("종료 (q)")
            break

        if result:
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "image": str(img_path),
                    "annotation": str(out_dir / f"{img_path.stem}.json"),
                    "target_label": tgt,
                    "destination_label": dest,
                }, ensure_ascii=False) + "\n")
            completed += 1

    print(f"\n완료: {completed}/{len(items)} 이미지 annotation")
    print(f"결과: {out_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="object 좌표 annotation GUI (G₀ + Molmo training 포맷 출력)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
출력 JSON 포맷 (annotations/<stem>.json):
  {
    "g0": {
      "target":      {"label": "cup", "coord": [x, y], "all_coords": [[x,y], ...]},
      "destination": {"label": "box", "coord": [x, y], "all_coords": [[x,y], ...]},
      "image_shape": [H, W]
    },
    "molmo_training": {
      "target":      [{"label":"cup", "x_norm":45.2, "y_norm":67.8, "x_px":289, "y_px":254}],
      "destination": [...],
      "prompt_target":      "Point to the cup.",
      "prompt_destination": "Point to the box."
    }
  }
        """,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images",        nargs="+", help="이미지 파일 경로 목록")
    src.add_argument("--dir",           help="이미지 디렉터리 (재귀 탐색)")
    src.add_argument("--from-manifest", help="dataset/manifest.jsonl 경로")

    p.add_argument("--target-label",      default="cup",
                   help="target 레이블 (--images/--dir 사용 시)")
    p.add_argument("--destination-label", default="box",
                   help="destination 레이블 (--images/--dir 사용 시)")
    p.add_argument("--out-dir",           default="annotations",
                   help="annotation JSON 저장 디렉터리 (기본: annotations/)")
    p.add_argument("--force",             action="store_true",
                   help="기존 annotation 무시하고 재annotation")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
