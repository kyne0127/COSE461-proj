#!/usr/bin/env python3
"""
kyne0127/vla-evaluation → dataset/hf_manifest.jsonl 변환 스크립트
meta.json의 절대경로를 로컬 상대경로로 교체
"""
import json
from pathlib import Path

HF_DIR   = Path("/workspace/COSE461-proj/dataset/vla-evaluation")
OUT_FILE = Path("/workspace/COSE461-proj/dataset/hf_manifest.jsonl")

entries = []
for meta_path in sorted(HF_DIR.glob("data-evaluation/S*/trial_*/meta.json")):
    trial_dir = meta_path.parent
    with open(meta_path) as f:
        meta = json.load(f)
    # 절대경로 → 로컬 상대경로로 교체 (dataset/ 기준)
    rel = trial_dir.relative_to(HF_DIR.parent)  # vla-evaluation/data-evaluation/S1/trial_001
    meta["initial_img"] = str(rel / "t0.png")
    meta["c1_img"]      = str(rel / "c1.png")
    meta["c2_img"]      = str(rel / "c2.png")
    entries.append(meta)

with open(OUT_FILE, "w") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Written {len(entries)} entries → {OUT_FILE}")

# 통계 출력
from collections import Counter
scenarios = Counter(e["scenario"] for e in entries)
states    = Counter(e["gold_state"] for e in entries)
decisions = Counter(e["gold_decision"] for e in entries)
print(f"Scenarios: {dict(sorted(scenarios.items()))}")
print(f"Gold States: {dict(sorted(states.items()))}")
print(f"Gold Decisions: {dict(sorted(decisions.items()))}")
