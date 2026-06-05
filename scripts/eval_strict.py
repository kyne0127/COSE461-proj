#!/usr/bin/env python3
"""
Strict accuracy evaluation: decision + clarifying question 동시 채점.

채점 기준:
  CONTINUE (gold=CLEAR):
    - pred=CONTINUE  AND  question 비어 있음  →  strict ✓
  ASK (gold=AMBIGUOUS_*):
    - pred=ASK  AND  question이 올바른 object를 언급  →  strict ✓
    - AMBIGUOUS_TARGET:      question에 target_label 포함
    - AMBIGUOUS_DESTINATION: question에 destination_label 포함
  STOP (gold=INVALID_*):
    - pred=STOP  AND  question이 올바른 object를 언급  →  strict ✓
    - INVALID_TARGET:      question에 target_label 포함
    - INVALID_DESTINATION: question에 destination_label 포함

Usage:
  python3 scripts/eval_strict.py
  python3 scripts/eval_strict.py --csvs logs/eval_holdout_v2_cond1.csv logs/eval_holdout_v2_fixed.csv
"""

from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO     = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "dataset/vla-evaluation-v4/manifest_eval-v4.jsonl"

# 조건 순서 (비교표 출력용)
COND_ORDER = [
    "cond2: FT image only",
    "cond4: FT + DINO, no G0",
    "cond1: FT + DINO + G0",
]


def load_labels(manifest_path: Path) -> dict[str, dict]:
    """trial_id → {target_label, destination_label, gold_state} 매핑."""
    result = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        result[e["id"]] = {
            "target_label":      e["target_label"],
            "destination_label": e["destination_label"],
            "gold_state":        e["gold_state"],
            "scenario":          e["scenario"],
        }
    return result


def is_strict_correct(row: dict, labels: dict) -> bool:
    """decision + question 동시 채점."""
    pred  = row["predicted_decision"]
    gold  = row["gold_decision"]
    q     = row["clarifying_question"].lower().strip()
    state = row["gold_state"]
    tid   = row["id"]

    info  = labels.get(tid, {})
    tgt   = info.get("target_label", "").lower()
    dst   = info.get("destination_label", "").lower()

    # decision 틀리면 바로 탈락
    if pred != gold:
        return False

    if gold == "CONTINUE":
        # question 비어있어야 함
        return q == ""

    if gold == "ASK":
        if "TARGET" in state:   # AMBIGUOUS_TARGET
            return tgt in q
        if "DESTINATION" in state:  # AMBIGUOUS_DESTINATION
            return dst in q
        return True  # 기타 ASK 상태는 decision 정답만으로 통과

    if gold == "STOP":
        if "TARGET" in state:   # INVALID_TARGET
            return tgt in q
        if "DESTINATION" in state:  # INVALID_DESTINATION
            return dst in q
        return True

    return False


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(rows: list[dict], labels: dict) -> dict:
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    results = {}
    for cond, cond_rows in by_cond.items():
        total         = len(cond_rows)
        dec_correct   = sum(r["predicted_decision"] == r["gold_decision"] for r in cond_rows)
        strict_pass   = sum(is_strict_correct(r, labels) for r in cond_rows)

        per_state = defaultdict(lambda: [0, 0, 0])  # [dec_ok, strict_ok, total]
        per_scenario = defaultdict(lambda: [0, 0, 0])
        fail_cases = []

        for r in cond_rows:
            gs = r["gold_state"]
            sc = r["scenario"]
            dec_ok = r["predicted_decision"] == r["gold_decision"]
            st_ok  = is_strict_correct(r, labels)

            per_state[gs][0]    += dec_ok
            per_state[gs][1]    += st_ok
            per_state[gs][2]    += 1
            per_scenario[sc][0] += dec_ok
            per_scenario[sc][1] += st_ok
            per_scenario[sc][2] += 1

            if dec_ok and not st_ok:
                fail_cases.append({
                    "id":       r["id"],
                    "state":    gs,
                    "pred":     r["predicted_decision"],
                    "question": r["clarifying_question"],
                    "target":   labels.get(r["id"], {}).get("target_label", "?"),
                    "dest":     labels.get(r["id"], {}).get("destination_label", "?"),
                })

        results[cond] = {
            "decision_acc":  dec_correct / total,
            "strict_acc":    strict_pass / total,
            "dec_correct":   dec_correct,
            "strict_pass":   strict_pass,
            "total":         total,
            "per_state":     {k: {"dec": v[0], "strict": v[1], "total": v[2]}
                              for k, v in sorted(per_state.items())},
            "per_scenario":  {k: {"dec": v[0], "strict": v[1], "total": v[2]}
                              for k, v in sorted(per_scenario.items())},
            "q_fail_cases":  fail_cases,  # decision ok but question wrong
        }
    return results


def print_report(results: dict):
    print("\n" + "="*70)
    print("  Strict Accuracy Report")
    print("  기준: decision 정답 AND question에 올바른 object label 포함")
    print("="*70)

    # 비교표
    print(f"\n{'Condition':<30} {'Dec Acc':>8}  {'Strict Acc':>10}  {'Drop':>6}")
    print("-"*58)
    for cond in COND_ORDER:
        if cond not in results:
            continue
        m = results[cond]
        drop = m["decision_acc"] - m["strict_acc"]
        print(f"{cond:<30} "
              f"{m['decision_acc']*100:>7.1f}%  "
              f"{m['strict_acc']*100:>9.1f}%  "
              f"{drop*100:>5.1f}%")

    # per-state 상세
    print(f"\n{'State':<30}", end="")
    for cond in COND_ORDER:
        short = cond.split()[-1] if cond in results else ""
        print(f"  {short:>12}", end="")
    print()
    print("-"*70)

    all_states = ["CLEAR", "AMBIGUOUS_TARGET", "AMBIGUOUS_DESTINATION",
                  "INVALID_TARGET", "INVALID_DESTINATION"]
    for state in all_states:
        print(f"{state:<30}", end="")
        for cond in COND_ORDER:
            if cond not in results:
                print(f"  {'?':>12}", end="")
                continue
            ps = results[cond]["per_state"].get(state, {"dec": 0, "strict": 0, "total": 0})
            cell = f"{ps['strict']}/{ps['total']} (d:{ps['dec']})"
            print(f"  {cell:>12}", end="")
        print()

    # question 실패 케이스 (decision ok but question wrong)
    print(f"\n{'─'*70}")
    print("  Decision ✓ but Question ✗ (object label 불일치)")
    print(f"{'─'*70}")
    for cond in COND_ORDER:
        if cond not in results:
            continue
        fails = results[cond]["q_fail_cases"]
        if not fails:
            continue
        print(f"\n  [{cond}] {len(fails)}건")
        for f in fails[:5]:
            print(f"    {f['id']:<25} state={f['state']}")
            print(f"      target={f['target']}, dest={f['dest']}")
            print(f"      q=\"{f['question']}\"")
        if len(fails) > 5:
            print(f"    ... 외 {len(fails)-5}건")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", default=[
        "logs/eval_holdout_v2_cond1.csv",
        "logs/eval_holdout_v2_fixed.csv",
    ])
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--out-json", default="logs/eval_strict_results.json")
    args = parser.parse_args()

    labels = load_labels(Path(args.manifest))
    print(f"Labels 로드: {len(labels)}개 trial")

    all_rows = []
    for csv_path in args.csvs:
        p = REPO / csv_path
        if not p.exists():
            print(f"[SKIP] 파일 없음: {p}")
            continue
        rows = load_csv(p)
        print(f"로드: {p.name} → {len(rows)}행")
        all_rows.extend(rows)

    if not all_rows:
        print("평가할 데이터가 없어요.")
        return

    results = evaluate(all_rows, labels)
    print_report(results)

    # 저장 (q_fail_cases는 제외하고 메트릭만)
    out = {
        k: {kk: vv for kk, vv in v.items() if kk != "q_fail_cases"}
        for k, v in results.items()
    }
    out_path = REPO / args.out_json
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
