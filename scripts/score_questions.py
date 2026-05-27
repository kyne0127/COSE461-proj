#!/usr/bin/env python3
"""
scripts/score_questions.py

생성된 ASK 질문의 관련성(Question Relevance)을 룰 기반으로 일괄 채점합니다.

사용법:
    python scripts/score_questions.py --predictions-csv logs/predictions_hf_v5.csv
    python scripts/score_questions.py --predictions-csv logs/predictions_hf_v5.csv --show-errors
    python scripts/score_questions.py --predictions-csv logs/predictions_hf_v5.csv --output-csv logs/question_scores_v5.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# ── 기대 질문 유형 정의 ────────────────────────────────────────────────────────

# Gold State → 질문이 다루어야 할 역할(role) 집합
# target_role: 픽업 대상에 대한 질문
# destination_role: 놓을 위치에 대한 질문
_ROLE_BY_STATE: dict[str, str] = {
    "AMBIGUOUS_TARGET":      "target",
    "INVALID_TARGET":        "target",
    "AMBIGUOUS_DESTINATION": "destination",
    "INVALID_DESTINATION":   "destination",
    "UNSAFE_OR_BLOCKED":     "destination",
    "INITIAL_AMBIGUOUS":     "target",  # initial ambiguity is always about target
}

# 질문 텍스트에서 target vs destination 역할 판단 키워드
_TARGET_KW   = ["집어", "pick", "잡을", "집을", "대상", "target", "[c1 ", "어떤"]
_DEST_KW     = ["놓아", "place", "놓을", "목적지", "destination", "[c2 "]


def _detect_role(question: str) -> str | None:
    """질문 텍스트에서 target / destination 역할을 추론합니다."""
    q = question.lower()
    is_target = any(kw in q for kw in _TARGET_KW)
    is_dest   = any(kw in q for kw in _DEST_KW)

    # [C1 ...] prefix → target, [C2 ...] prefix → destination (가장 신뢰도 높음)
    if "[c1 " in q:
        return "target"
    if "[c2 " in q:
        return "destination"

    if is_target and not is_dest:
        return "target"
    if is_dest and not is_target:
        return "destination"
    return None   # 구분 불가


def score_question(question: str, gold_state: str) -> dict:
    """
    Returns:
        {
          "relevant":   bool | None,  # None = 채점 불가 (빈 질문 또는 예외 state)
          "reason":     str,
          "role_guess": str | None,
          "role_expected": str | None,
        }
    """
    if not question.strip():
        return {"relevant": None, "reason": "no question generated", "role_guess": None, "role_expected": None}

    expected_role = _ROLE_BY_STATE.get(gold_state)
    if expected_role is None:
        return {"relevant": None, "reason": f"no expected role for state {gold_state}",
                "role_guess": None, "role_expected": None}

    role_guess = _detect_role(question)
    if role_guess is None:
        return {"relevant": None, "reason": "could not determine question role",
                "role_guess": None, "role_expected": expected_role}

    relevant = role_guess == expected_role
    return {
        "relevant":      relevant,
        "reason":        "role matches" if relevant else f"expected {expected_role}, got {role_guess}",
        "role_guess":    role_guess,
        "role_expected": expected_role,
    }


def score_csv(predictions_csv: Path) -> list[dict]:
    results = []
    with open(predictions_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method") != "OURS":
                continue
            if row.get("predicted_decision") != "ASK":
                continue
            score = score_question(row.get("question", ""), row.get("gold_state", ""))
            results.append({
                "sample_id":        row["sample_id"],
                "scenario":         row["scenario"],
                "checkpoint":       row["checkpoint"],
                "gold_state":       row["gold_state"],
                "gold_decision":    row["gold_decision"],
                "predicted_state":  row["predicted_state"],
                "question":         row.get("question", ""),
                **score,
            })
    return results


def print_summary(results: list[dict], show_errors: bool = False) -> None:
    scorable  = [r for r in results if r["relevant"] is not None]
    relevant  = [r for r in scorable if r["relevant"]]
    irrel     = [r for r in scorable if not r["relevant"]]
    unscored  = [r for r in results if r["relevant"] is None]

    total = len(results)
    print(f"\n{'='*60}")
    print(f"  Question Relevance Score  (OURS, ASK-only rows)")
    print(f"{'='*60}")
    print(f"  Total ASK rows         : {total}")
    print(f"  Scorable               : {len(scorable)}")
    print(f"  Relevant (correct role): {len(relevant)}")
    print(f"  Irrelevant             : {len(irrel)}")
    print(f"  Unscored (no question) : {len(unscored)}")
    if scorable:
        pct = len(relevant) / len(scorable) * 100
        print(f"\n  Question Relevance     : {pct:.1f}% ({len(relevant)}/{len(scorable)})")

    # Per-scenario breakdown
    scenarios = sorted({r["scenario"] for r in results})
    if len(scenarios) > 1:
        print(f"\n  Per-scenario:")
        for sc in scenarios:
            sc_rows = [r for r in scorable if r["scenario"] == sc]
            if not sc_rows:
                continue
            sc_rel = sum(1 for r in sc_rows if r["relevant"])
            print(f"    {sc}: {sc_rel}/{len(sc_rows)} ({sc_rel/len(sc_rows)*100:.0f}%)")

    if show_errors and irrel:
        print(f"\n  Irrelevant questions:")
        for r in irrel:
            print(f"    [{r['scenario']}] {r['sample_id']} | gold={r['gold_state']} "
                  f"predicted={r['predicted_state']} | {r['reason']}")
            print(f"      Q: {r['question'][:100]}")

    if unscored:
        print(f"\n  Unscored (empty question):")
        for r in unscored:
            print(f"    [{r['scenario']}] {r['sample_id']} | gold={r['gold_state']} | {r['reason']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Batch-score question relevance in predictions CSV.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-csv",      default="", help="CSV 출력 경로 (생략 시 콘솔만)")
    parser.add_argument("--show-errors",     action="store_true", help="잘못된 질문 상세 출력")
    parser.add_argument("--output-json",     default="", help="JSON 요약 출력 경로")
    args = parser.parse_args()

    results = score_csv(Path(args.predictions_csv))
    print_summary(results, show_errors=args.show_errors)

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id", "scenario", "checkpoint", "gold_state", "gold_decision",
            "predicted_state", "question", "relevant", "reason",
            "role_guess", "role_expected",
        ]
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved → {out}")

    if args.output_json:
        scorable = [r for r in results if r["relevant"] is not None]
        relevant = sum(1 for r in scorable if r["relevant"])
        summary = {
            "total_ask": len(results),
            "scorable": len(scorable),
            "relevant": relevant,
            "question_relevance": relevant / len(scorable) if scorable else None,
        }
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved → {args.output_json}")


if __name__ == "__main__":
    main()
