#!/usr/bin/env python3
"""
dataset/annotator.py
=====================
Inter-annotator agreement 측정용 CLI 라벨링 도구.

사용법:
  python dataset/annotator.py --name <annotator_name>
  python dataset/annotator.py --name <annotator_name> --manifest dataset/manifest.jsonl
  python dataset/annotator.py --kappa  # 기존 annotation 파일들 간 Cohen's kappa 계산

출력:
  dataset/annotations_<name>.jsonl  — 라벨링 결과
  dataset/kappa_report.json         — kappa 계산 결과 (--kappa 시)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VALID_STATES = [
    "CLEAR",
    "AMBIGUOUS_TARGET",
    "INVALID_TARGET",
    "AMBIGUOUS_DESTINATION",
    "INVALID_DESTINATION",
    "UNSAFE_OR_BLOCKED",
]

VALID_DECISIONS = ["CONTINUE", "ASK", "STOP"]

STATE_ABBREV = {
    "1": "CLEAR",
    "2": "AMBIGUOUS_TARGET",
    "3": "INVALID_TARGET",
    "4": "AMBIGUOUS_DESTINATION",
    "5": "INVALID_DESTINATION",
    "6": "UNSAFE_OR_BLOCKED",
}

DECISION_ABBREV = {
    "1": "CONTINUE",
    "2": "ASK",
    "3": "STOP",
}


# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────

def load_manifest(path: str) -> List[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_annotations(path: str) -> Dict[str, dict]:
    """id → annotation dict"""
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ann = json.loads(line)
                result[ann["id"]] = ann
    return result


def save_annotation(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────────────────────────────────
# Cohen's kappa
# ────────────────────────────────────────────────────────────────────────────

def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    assert len(labels_a) == len(labels_b), "길이 불일치"
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))

    # observed agreement
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # expected agreement
    pe = 0.0
    for cat in categories:
        p_a = labels_a.count(cat) / n
        p_b = labels_b.count(cat) / n
        pe += p_a * p_b

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def compute_kappa_report(annotation_files: List[str]) -> dict:
    """annotation 파일 2개 이상을 비교해 kappa 계산."""
    if len(annotation_files) < 2:
        return {"error": "annotation 파일이 2개 이상 필요합니다."}

    # 공통 sample id만 사용
    all_anns = []
    for f in annotation_files:
        all_anns.append(load_annotations(f))

    common_ids = set(all_anns[0].keys())
    for anns in all_anns[1:]:
        common_ids &= set(anns.keys())
    common_ids = sorted(common_ids)

    if not common_ids:
        return {"error": "공통 sample이 없습니다."}

    report = {
        "n_samples": len(common_ids),
        "annotators": [Path(f).stem for f in annotation_files],
        "pairs": [],
    }

    names = [Path(f).stem for f in annotation_files]
    for i in range(len(annotation_files)):
        for j in range(i + 1, len(annotation_files)):
            states_i = [all_anns[i][sid]["gold_state"] for sid in common_ids]
            states_j = [all_anns[j][sid]["gold_state"] for sid in common_ids]
            decisions_i = [all_anns[i][sid]["gold_decision"] for sid in common_ids]
            decisions_j = [all_anns[j][sid]["gold_decision"] for sid in common_ids]

            kappa_state = cohen_kappa(states_i, states_j)
            kappa_decision = cohen_kappa(decisions_i, decisions_j)

            # per-sample agreement detail
            details = []
            for sid in common_ids:
                details.append({
                    "id": sid,
                    "state_agree": all_anns[i][sid]["gold_state"] == all_anns[j][sid]["gold_state"],
                    f"{names[i]}_state": all_anns[i][sid]["gold_state"],
                    f"{names[j]}_state": all_anns[j][sid]["gold_state"],
                    "decision_agree": all_anns[i][sid]["gold_decision"] == all_anns[j][sid]["gold_decision"],
                    f"{names[i]}_decision": all_anns[i][sid]["gold_decision"],
                    f"{names[j]}_decision": all_anns[j][sid]["gold_decision"],
                })

            report["pairs"].append({
                "annotator_a": names[i],
                "annotator_b": names[j],
                "kappa_state": round(kappa_state, 4),
                "kappa_decision": round(kappa_decision, 4),
                "kappa_interpretation": _interpret_kappa(min(kappa_state, kappa_decision)),
                "details": details,
            })

    return report


def _interpret_kappa(k: float) -> str:
    if k < 0:      return "Poor (< 0)"
    if k < 0.20:   return "Slight (0.00–0.20)"
    if k < 0.40:   return "Fair (0.20–0.40)"
    if k < 0.60:   return "Moderate (0.40–0.60)"
    if k < 0.80:   return "Substantial (0.60–0.80)"
    return "Almost perfect (0.80–1.00)"


# ────────────────────────────────────────────────────────────────────────────
# CLI annotation loop
# ────────────────────────────────────────────────────────────────────────────

def prompt_state() -> str:
    print("\n  Grounding State:")
    for k, v in STATE_ABBREV.items():
        print(f"    {k}) {v}")
    while True:
        ans = input("  선택 (1-6 또는 전체 이름): ").strip()
        if ans in STATE_ABBREV:
            return STATE_ABBREV[ans]
        if ans.upper() in VALID_STATES:
            return ans.upper()
        print("  ※ 다시 입력하세요.")


def prompt_decision() -> str:
    print("\n  Decision:")
    for k, v in DECISION_ABBREV.items():
        print(f"    {k}) {v}")
    while True:
        ans = input("  선택 (1-3 또는 전체 이름): ").strip()
        if ans in DECISION_ABBREV:
            return DECISION_ABBREV[ans]
        if ans.upper() in VALID_DECISIONS:
            return ans.upper()
        print("  ※ 다시 입력하세요.")


def run_annotation(manifest_path: str, annotator_name: str, output_dir: str) -> str:
    samples = load_manifest(manifest_path)
    out_path = os.path.join(output_dir, f"annotations_{annotator_name}.jsonl")

    # 이미 라벨링된 항목 건너뜀
    done_ids = set()
    if os.path.exists(out_path):
        done_ids = set(load_annotations(out_path).keys())

    remaining = [s for s in samples if s["id"] not in done_ids]

    if not remaining:
        print(f"\n모든 샘플이 이미 라벨링되어 있습니다 → {out_path}")
        return out_path

    print(f"\n{'='*60}")
    print(f"  Annotator : {annotator_name}")
    print(f"  총 샘플   : {len(samples)}개  |  남은 샘플: {len(remaining)}개")
    print(f"  출력 파일 : {out_path}")
    print(f"{'='*60}")
    print("\n  각 샘플에 대해 Grounding State와 Decision을 입력하세요.")
    print("  (q 입력 시 중단, 나중에 재실행하면 이어서 진행)\n")

    for idx, sample in enumerate(remaining, 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{len(remaining)}] ID: {sample['id']}")
        print(f"  Scenario   : {sample['scenario']}")
        print(f"  Task       : {sample['task']}")
        print(f"  Checkpoint : {sample['checkpoint']}")
        print(f"  Target     : {sample['target_label']}")
        print(f"  Destination: {sample['destination_label']}")
        print(f"  Notes      : {sample.get('notes', '')}")
        print(f"\n  이미지 경로 (직접 열어서 확인):")
        img_base = os.path.join(os.path.dirname(manifest_path))
        print(f"    t0  : {os.path.join(img_base, sample['initial_img'])}")
        print(f"    c1  : {os.path.join(img_base, sample['c1_img'])}")
        if sample.get("c2_img"):
            print(f"    c2  : {os.path.join(img_base, sample['c2_img'])}")

        try:
            state = prompt_state()
            if state.lower() == "q":
                print("\n중단됨. 재실행하면 이어서 진행합니다.")
                break
            decision = prompt_decision()
            if decision.lower() == "q":
                print("\n중단됨. 재실행하면 이어서 진행합니다.")
                break
        except (KeyboardInterrupt, EOFError):
            print("\n\n중단됨. 재실행하면 이어서 진행합니다.")
            break

        note = input("  메모 (선택, 없으면 Enter): ").strip()

        record = {
            "id": sample["id"],
            "annotator": annotator_name,
            "gold_state": state,
            "gold_decision": decision,
            "note": note,
        }
        save_annotation(out_path, record)
        print(f"  ✓ 저장: {state} / {decision}")

    print(f"\n완료. 결과 파일: {out_path}")
    return out_path


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-annotator annotation & kappa 계산")
    parser.add_argument("--name", type=str, help="annotator 이름 (라벨링 모드)")
    parser.add_argument("--manifest", type=str,
                        default=str(REPO_ROOT / "dataset" / "manifest.jsonl"))
    parser.add_argument("--output-dir", type=str,
                        default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--kappa", action="store_true",
                        help="기존 annotation 파일들 간 Cohen's kappa 계산")
    parser.add_argument("--kappa-files", nargs="+",
                        help="kappa 계산할 annotation 파일 경로 (미지정 시 output-dir 내 자동 검색)")
    parser.add_argument("--kappa-output", type=str,
                        default=str(REPO_ROOT / "dataset" / "kappa_report.json"))
    args = parser.parse_args()

    if args.kappa:
        if args.kappa_files:
            files = args.kappa_files
        else:
            files = sorted(Path(args.output_dir).glob("annotations_*.jsonl"))
            files = [str(f) for f in files]
        if len(files) < 2:
            print(f"[error] kappa 계산에는 annotation 파일이 2개 이상 필요합니다.")
            print(f"  발견된 파일: {files}")
            sys.exit(1)
        print(f"\nKappa 계산 중... ({len(files)}개 파일)")
        for f in files:
            print(f"  - {f}")
        report = compute_kappa_report(files)
        with open(args.kappa_output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {args.kappa_output}")
        for pair in report.get("pairs", []):
            print(f"\n  {pair['annotator_a']} vs {pair['annotator_b']}")
            print(f"    kappa (state)    : {pair['kappa_state']}  → {pair['kappa_interpretation']}")
            print(f"    kappa (decision) : {pair['kappa_decision']}")
            for d in pair["details"]:
                state_mark = "✓" if d["state_agree"] else "✗"
                dec_mark = "✓" if d["decision_agree"] else "✗"
                print(f"    {d['id']}: state {state_mark}  decision {dec_mark}")
        return

    if not args.name:
        parser.print_help()
        sys.exit(1)

    run_annotation(args.manifest, args.name, args.output_dir)


if __name__ == "__main__":
    main()
