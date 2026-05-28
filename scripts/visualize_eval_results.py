#!/usr/bin/env python3
"""
scripts/visualize_eval_results.py

evaluate.py 결과(predictions.csv + metrics.json)를 시각화합니다.

로컬 실행:
    python scripts/visualize_eval_results.py \
        --predictions  /tmp/eval_results/predictions.csv \
        --metrics      /tmp/eval_results/metrics.json \
        --out-dir      /tmp/eval_results/vis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── 색상 팔레트 ────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "B1_INITIAL_ONLY":          "#5B8DB8",
    "B2_NO_MEMORY":             "#7EC8A4",
    "B3_ATTRIBUTE_AWARE_COUNT": "#F4A460",
    "B4_BINARY_ANOMALY":        "#D95F5F",
    "OURS":                     "#9B59B6",
}
DECISION_COLORS = {"CONTINUE": "#4CAF50", "ASK": "#FF9800", "STOP": "#F44336"}
GOLD_ORDER      = ["CONTINUE", "ASK", "STOP"]
METHODS_DISPLAY = {
    "B1_INITIAL_ONLY":          "B1 Initial-only",
    "B2_NO_MEMORY":             "B2 No-memory",
    "B3_ATTRIBUTE_AWARE_COUNT": "B3 Count-rule",
    "B4_BINARY_ANOMALY":        "B4 Binary-anomaly",
    "OURS":                     "Ours",
}


def _method_label(m: str) -> str:
    return METHODS_DISPLAY.get(m, m)


# ── 1. 전체 Accuracy / Miss-rate / False-alarm 막대그래프 ──────────────────────

def plot_overall_metrics(metrics: dict, out_dir: Path):
    methods  = list(metrics.keys())
    acc      = [metrics[m].get("decision_accuracy",  0) for m in methods]
    miss     = [metrics[m].get("miss_rate",          0) for m in methods]
    fa       = [metrics[m].get("false_alarm_rate",   0) for m in methods]

    x    = np.arange(len(methods))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    bars_acc  = ax.bar(x - w,   acc,  w, label="Accuracy",        color="#5B8DB8")
    bars_miss = ax.bar(x,       miss, w, label="Miss-rate",        color="#D95F5F")
    bars_fa   = ax.bar(x + w,   fa,   w, label="False-alarm-rate", color="#F4A460")

    for bar in [*bars_acc, *bars_miss, *bars_fa]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title("Overall Performance: B1–B4")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_dir / "01_overall_metrics.png", dpi=150)
    plt.close(fig)
    print("  saved: 01_overall_metrics.png")


# ── 2. 메서드별 Confusion Matrix (gold vs predicted) ──────────────────────────

def plot_confusion_matrices(df: pd.DataFrame, methods: list, out_dir: Path):
    decisions = ["CONTINUE", "ASK", "STOP"]
    n = len(methods)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    for i, method in enumerate(methods):
        ax = axes[i]
        sub = df[df["method"] == method]
        mat = np.zeros((3, 3), dtype=int)
        for gi, gold in enumerate(decisions):
            for pi, pred in enumerate(decisions):
                mat[gi, pi] = ((sub["gold_decision"] == gold) &
                               (sub["predicted_decision"] == pred)).sum()

        im = ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_xticks(range(3)); ax.set_xticklabels(decisions, rotation=15)
        ax.set_yticks(range(3)); ax.set_yticklabels(decisions)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
        ax.set_title(_method_label(method))

        for gi in range(3):
            for pi in range(3):
                v = mat[gi, pi]
                ax.text(pi, gi, str(v), ha="center", va="center",
                        color="white" if v > mat.max() * 0.6 else "black",
                        fontsize=13, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrix: Gold vs Predicted Decision", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "02_confusion_matrices.png", dpi=150)
    plt.close(fig)
    print("  saved: 02_confusion_matrices.png")


# ── 3. 시나리오별 Accuracy 히트맵 ─────────────────────────────────────────────

def plot_scenario_heatmap(df: pd.DataFrame, methods: list, out_dir: Path):
    scenarios = sorted(df["scenario"].dropna().unique())
    acc_mat   = np.zeros((len(methods), len(scenarios)))

    for mi, method in enumerate(methods):
        for si, sc in enumerate(scenarios):
            sub = df[(df["method"] == method) & (df["scenario"] == sc)]
            if len(sub):
                acc_mat[mi, si] = (sub["gold_decision"] == sub["predicted_decision"]).mean()

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.2), max(4, len(methods) * 0.9)))
    im = ax.imshow(acc_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy")

    ax.set_xticks(range(len(scenarios))); ax.set_xticklabels(scenarios)
    ax.set_yticks(range(len(methods)));   ax.set_yticklabels([_method_label(m) for m in methods])
    ax.set_title("Accuracy per Scenario × Method")

    for mi in range(len(methods)):
        for si in range(len(scenarios)):
            ax.text(si, mi, f"{acc_mat[mi, si]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")

    fig.tight_layout()
    fig.savefig(out_dir / "03_scenario_heatmap.png", dpi=150)
    plt.close(fig)
    print("  saved: 03_scenario_heatmap.png")


# ── 4. 샘플별 결정 비교 스트립 ────────────────────────────────────────────────

def plot_decision_strip(df: pd.DataFrame, methods: list, out_dir: Path):
    sample_ids = sorted(df["sample_id"].unique())
    dec2int    = {"CONTINUE": 0, "ASK": 1, "STOP": 2}

    fig, axes  = plt.subplots(len(methods) + 1, 1,
                               figsize=(max(14, len(sample_ids) * 0.25), 2.5 * (len(methods) + 1)),
                               sharex=True)

    # 첫 행: gold
    sub_gold = df[df["method"] == methods[0]].sort_values("sample_id")
    gold_ints = [dec2int.get(g, 0) for g in sub_gold["gold_decision"]]
    cmap      = plt.cm.get_cmap("RdYlGn_r", 3)
    axes[0].bar(range(len(sample_ids)), [1] * len(sample_ids),
                color=[cmap(g / 2.0) for g in gold_ints], edgecolor="none")
    axes[0].set_ylabel("Gold", fontsize=8)
    axes[0].set_yticks([])

    for mi, method in enumerate(methods):
        ax  = axes[mi + 1]
        sub = df[df["method"] == method].sort_values("sample_id")
        # 정렬 순서를 gold와 맞춤
        pred_ints = [dec2int.get(p, 0) for p in sub["predicted_decision"]]
        correct   = (sub["gold_decision"].values == sub["predicted_decision"].values)
        colors    = []
        for pi, c in zip(pred_ints, correct):
            if c:
                colors.append(cmap(pi / 2.0))
            else:
                colors.append("#888888")   # 오답: 회색

        ax.bar(range(len(sample_ids)), [1] * len(sample_ids),
               color=colors, edgecolor="none")
        ax.set_ylabel(_method_label(method), fontsize=7)
        ax.set_yticks([])

    axes[-1].set_xticks(range(len(sample_ids)))
    axes[-1].set_xticklabels(
        [s.replace("s", "S").replace("_trial_", "-T") for s in sample_ids],
        rotation=90, fontsize=6
    )

    # 범례
    patches = [
        mpatches.Patch(color=cmap(0 / 2.0), label="CONTINUE"),
        mpatches.Patch(color=cmap(1 / 2.0), label="ASK"),
        mpatches.Patch(color=cmap(2 / 2.0), label="STOP"),
        mpatches.Patch(color="#888888",      label="Wrong"),
    ]
    fig.legend(handles=patches, loc="upper right", ncol=4, fontsize=8)
    fig.suptitle("Per-Sample Decision Strip (grey = wrong prediction)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "04_decision_strip.png", dpi=150)
    plt.close(fig)
    print("  saved: 04_decision_strip.png")


# ── 5. Gold 클래스별 Precision / Recall ──────────────────────────────────────

def plot_per_class_pr(df: pd.DataFrame, methods: list, out_dir: Path):
    decisions = ["CONTINUE", "ASK", "STOP"]
    n_dec     = len(decisions)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_dec, figsize=(6 * n_dec, 5), sharey=True)

    for di, gold_cls in enumerate(decisions):
        ax  = axes[di]
        prec_vals, rec_vals, labels = [], [], []

        for method in methods:
            sub = df[df["method"] == method]
            tp  = ((sub["gold_decision"] == gold_cls) & (sub["predicted_decision"] == gold_cls)).sum()
            fp  = ((sub["gold_decision"] != gold_cls) & (sub["predicted_decision"] == gold_cls)).sum()
            fn  = ((sub["gold_decision"] == gold_cls) & (sub["predicted_decision"] != gold_cls)).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            prec_vals.append(prec)
            rec_vals.append(rec)
            labels.append(_method_label(method))

        x = np.arange(n_methods)
        w = 0.35
        b1 = ax.bar(x - w / 2, prec_vals, w, label="Precision",
                    color=[METHOD_COLORS.get(m, "#999") for m in methods], alpha=0.9)
        b2 = ax.bar(x + w / 2, rec_vals,  w, label="Recall",
                    color=[METHOD_COLORS.get(m, "#999") for m in methods], alpha=0.5,
                    edgecolor=[METHOD_COLORS.get(m, "#999") for m in methods], linewidth=1.5)

        for bar in [*b1, *b2]:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"Gold = {gold_cls}", color=DECISION_COLORS.get(gold_cls, "black"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.25)
        if di == 0:
            ax.set_ylabel("Score")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    handles = [mpatches.Patch(color="gray", alpha=0.9, label="Precision"),
               mpatches.Patch(color="gray", alpha=0.4, label="Recall")]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle("Per-Class Precision & Recall", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "05_per_class_pr.png", dpi=150)
    plt.close(fig)
    print("  saved: 05_per_class_pr.png")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="predictions.csv 경로")
    parser.add_argument("--metrics",     required=True, help="metrics.json 경로")
    parser.add_argument("--out-dir",     default="/tmp/eval_results/vis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df      = pd.read_csv(args.predictions)
    metrics = json.load(open(args.metrics, encoding="utf-8"))

    methods = list(metrics.keys())
    print(f"Methods: {methods}")
    print(f"Samples: {len(df)}")
    print(f"Saving to: {out_dir}\n")

    # 정규화: predicted_decision 컬럼명 확인
    if "predicted_decision" not in df.columns:
        if "prediction" in df.columns:
            df = df.rename(columns={"prediction": "predicted_decision"})
        else:
            raise ValueError(f"columns: {list(df.columns)}")

    plot_overall_metrics(metrics, out_dir)
    plot_confusion_matrices(df, methods, out_dir)
    plot_scenario_heatmap(df, methods, out_dir)
    plot_decision_strip(df, methods, out_dir)
    plot_per_class_pr(df, methods, out_dir)

    print("\n모든 시각화 완료.")
    print(f"결과: {out_dir}/")


if __name__ == "__main__":
    main()
