#!/usr/bin/env python3
"""
Fine-tuning 데이터셋 annotation 시각화 스크립트.

train.jsonl / val.jsonl의 저장된 DINO 좌표를 이미지에 그려서
사람이 검수할 수 있도록 annotated 이미지 + HTML 리뷰 페이지를 생성한다.

색상 규칙:
  빨강 원   — checkpoint에서 탐지된 target (실제 학습에 사용되는 좌표)
  파랑 원   — checkpoint에서 탐지된 destination
  빨강 점선 — t0 기준(G0) target 위치
  파랑 점선 — t0 기준(G0) destination 위치

Usage:
    python3 scripts/annotate_finetune_dataset.py
    python3 scripts/annotate_finetune_dataset.py --jsonl dataset/finetune/train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def draw_annotation(
    img: Image.Image,
    g0_tgt: list[int] | None,
    g0_dst: list[int] | None,
    ck_tgt: list[list[int]],
    ck_dst: list[list[int]],
    target_label: str,
    dest_label: str,
) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)

    R_G0 = 10   # G0 기준 원 반지름
    R_CK = 8    # checkpoint 탐지 원 반지름

    # G0 기준 위치 (점선 효과: 외곽 + 내부 빈 원)
    if g0_tgt:
        x, y = g0_tgt
        draw.ellipse([x-R_G0, y-R_G0, x+R_G0, y+R_G0], outline=(255, 80, 80), width=1)
        draw.ellipse([x-R_G0-3, y-R_G0-3, x+R_G0+3, y+R_G0+3], outline=(255, 80, 80), width=1)
        draw.text((x+R_G0+3, y-8), f"G0:{target_label}", fill=(255, 80, 80))

    if g0_dst:
        x, y = g0_dst
        draw.ellipse([x-R_G0, y-R_G0, x+R_G0, y+R_G0], outline=(80, 80, 255), width=1)
        draw.ellipse([x-R_G0-3, y-R_G0-3, x+R_G0+3, y+R_G0+3], outline=(80, 80, 255), width=1)
        draw.text((x+R_G0+3, y-8), f"G0:{dest_label}", fill=(80, 80, 255))

    # Checkpoint 탐지 좌표 (굵은 원)
    for i, (x, y) in enumerate(ck_tgt):
        draw.ellipse([x-R_CK, y-R_CK, x+R_CK, y+R_CK], outline=(255, 0, 0), width=3)
        draw.text((x+R_CK+3, y-8), f"{target_label}[{i}]", fill=(255, 0, 0))

    for i, (x, y) in enumerate(ck_dst):
        draw.ellipse([x-R_CK, y-R_CK, x+R_CK, y+R_CK], outline=(0, 0, 255), width=3)
        draw.text((x+R_CK+3, y-8), f"{dest_label}[{i}]", fill=(0, 0, 255))

    # 탐지 없음 표시
    if not ck_tgt:
        draw.text((5, 5), f"NO {target_label} DETECTED", fill=(255, 0, 0))
    if not ck_dst:
        draw.text((5, 20), f"NO {dest_label} DETECTED", fill=(0, 0, 255))

    return img


def make_label_bar(width: int, example: dict) -> Image.Image:
    bar = Image.new("RGB", (width, 36), (30, 30, 30))
    draw = ImageDraw.Draw(bar)

    decision_color = {
        "CONTINUE": (80, 200, 80),
        "ASK":      (255, 200, 0),
        "STOP":     (255, 60, 60),
    }
    gold_state = example.get("gold_state", "?")
    ambiguity = example.get("ambiguity_bool", False)
    stop_bool = example.get("stop_bool", False)
    decision = "STOP" if stop_bool else ("ASK" if ambiguity else "CONTINUE")
    color = decision_color.get(decision, (200, 200, 200))

    q = example.get("clarifying_question", "")
    text = f"[{decision}] {gold_state}  |  {example['id']}  |  q: {q[:60]}"
    draw.rectangle([0, 0, width, 36], fill=(30, 30, 30))
    draw.text((6, 10), text, fill=color)
    return bar


def annotate_jsonl(jsonl_path: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    records = []

    for ex in examples:
        img_path = Path(ex["image"])
        if not img_path.exists():
            print(f"  [SKIP] 이미지 없음: {img_path}")
            continue

        meta = ex.get("meta", {})
        g0_tgt = meta.get("g0_target_coord")
        g0_dst = meta.get("g0_dest_coord")
        ck_tgt = meta.get("ck_target_coords", [])
        ck_dst = meta.get("ck_dest_coords", [])

        img = Image.open(img_path).convert("RGB")
        annotated = draw_annotation(
            img, g0_tgt, g0_dst, ck_tgt, ck_dst,
            ex.get("target_label", "target"),
            ex.get("destination_label", "dest"),
        )
        bar = make_label_bar(img.width, ex)

        combined = Image.new("RGB", (img.width, img.height + bar.height))
        combined.paste(annotated, (0, 0))
        combined.paste(bar, (0, img.height))

        # _aug (flip/dark/bright/noise) 포함한 이름 그대로 저장
        out_path = out_dir / f"{ex['id']}.png"
        combined.save(out_path)

        # 문제 탐지: 좌표가 없는 경우
        issue = None
        gold = ex.get("gold_state", "")
        if not ck_tgt and gold not in ("INVALID_TARGET",):
            issue = f"target '{ex.get('target_label')}' NOT detected"
        elif not ck_dst and gold not in ("INVALID_DESTINATION",):
            issue = f"dest '{ex.get('destination_label')}' NOT detected"
        elif len(ck_tgt) > 1 and ex.get("gold_state") == "CLEAR":
            issue = f"CLEAR인데 target {len(ck_tgt)}개 탐지됨"
        elif len(ck_tgt) == 1 and ex.get("gold_state") == "AMBIGUOUS_TARGET" and "moved" not in ex.get("clarifying_question", ""):
            issue = f"AMBIGUOUS_TARGET인데 target 1개만 탐지 (moved 케이스인지 확인)"

        records.append({
            "id": ex["id"],
            "gold_state": ex.get("gold_state"),
            "decision": "STOP" if ex.get("stop_bool") else ("ASK" if ex.get("ambiguity_bool") else "CONTINUE"),
            "n_target": len(ck_tgt),
            "n_dest": len(ck_dst),
            "clarifying_question": ex.get("clarifying_question", ""),
            "issue": issue,
            "annotated_img": str(out_path.relative_to(out_dir.parent)),
        })

    return records


def make_html(records: list[dict], out_dir: Path, split_name: str) -> None:
    issues = [r for r in records if r["issue"]]
    normal = [r for r in records if not r["issue"]]

    rows = ""
    for r in issues + normal:
        bg = "#3a1a1a" if r["issue"] else "#1a1a1a"
        decision_color = {"CONTINUE": "#50c850", "ASK": "#ffc800", "STOP": "#ff3c3c"}.get(r["decision"], "#ccc")
        rows += f"""
        <tr style="background:{bg}">
          <td><img src="{r['annotated_img']}" width="336" loading="lazy"></td>
          <td style="color:{decision_color};font-weight:bold">{r['decision']}</td>
          <td>{r['gold_state']}</td>
          <td>{r['n_target']}</td>
          <td>{r['n_dest']}</td>
          <td>{r['clarifying_question']}</td>
          <td style="color:#ff6060">{r['issue'] or ''}</td>
          <td style="font-size:10px;color:#888">{r['id']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Dataset Review — {split_name}</title>
<style>
  body {{ background:#111; color:#ccc; font-family:monospace; font-size:12px; }}
  table {{ border-collapse:collapse; width:100%; }}
  th {{ background:#222; padding:6px; text-align:left; position:sticky; top:0; }}
  td {{ padding:4px 8px; border-bottom:1px solid #333; vertical-align:middle; }}
  .summary {{ background:#222; padding:12px; margin-bottom:16px; border-radius:4px; }}
</style></head><body>
<div class="summary">
  <b>{split_name}</b> — 총 {len(records)}개
  | 이슈: <span style="color:#ff6060">{len(issues)}개</span>
  | CONTINUE: {sum(1 for r in records if r['decision']=='CONTINUE')}
  | ASK: {sum(1 for r in records if r['decision']=='ASK')}
  | STOP: {sum(1 for r in records if r['decision']=='STOP')}
  <br><small style="color:#888">빨강=target탐지, 파랑=dest탐지, 옅은색=G0기준위치 / 이슈 있는 항목이 위에 표시됨</small>
</div>
<table>
  <tr>
    <th>이미지</th><th>결정</th><th>gold_state</th>
    <th>#target</th><th>#dest</th><th>질문/메시지</th><th>이슈</th><th>id</th>
  </tr>
  {rows}
</table></body></html>"""

    html_path = out_dir / f"review_{split_name}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  HTML: {html_path}")
    if issues:
        print(f"  [!] 이슈 {len(issues)}개:")
        for r in issues:
            print(f"      {r['id']}: {r['issue']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+",
                        default=["dataset/finetune/train.jsonl",
                                 "dataset/finetune/val.jsonl"])
    parser.add_argument("--out-dir", default="dataset/finetune/annotated")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / args.out_dir

    for jsonl_path in args.jsonl:
        p = repo / jsonl_path if not Path(jsonl_path).is_absolute() else Path(jsonl_path)
        if not p.exists():
            print(f"[SKIP] {p} 없음")
            continue
        split = p.stem
        print(f"\n[{split}] annotation 생성 중...")
        records = annotate_jsonl(p, out_dir)
        make_html(records, out_dir, split)
        print(f"  annotated: {len(records)}개 → {out_dir}/")


if __name__ == "__main__":
    main()
