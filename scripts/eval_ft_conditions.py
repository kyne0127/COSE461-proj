#!/usr/bin/env python3
"""
3+2가지 조건 비교 평가 (val.jsonl 기준)

  cond1: FT + DINO coords    — fine-tuned + G0 + checkpoint detection coords
  cond2: FT - DINO           — fine-tuned + G0 only, no detection coords
  cond3: pre-FT + DINO coords — Molmo+AmbRes merged (no v3 FT) + same prompt
  cond4: FT + DINO, no G0    — fine-tuned, detection coords but no G0 reference
  cond5: FT + DINO, no tax   — fine-tuned, full coords but no taxonomy text

모델 로딩:
  FT model:    Molmo-7B → merge AmbRes LoRA → load v3 LoRA (checkpoint-162)
  pre-FT model: Molmo-7B → merge AmbRes LoRA (v3 LoRA 없음)

Usage:
  python3 scripts/eval_ft_conditions.py
  python3 scripts/eval_ft_conditions.py --conditions cond1 cond2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "AmbRes"))

from ambres.training.data import process_msg_list

MODEL_ID        = "allenai/Molmo-7B-D-0924"
AMBRES_CKPT     = "/workspace/AmbRes/ckpt/AB5siP8DA78aA78wR5Y8Mw/checkpoint-390"
FT_CKPT         = str(REPO / "ckpt/grounding_ft/BvFsfsf4/checkpoint-162")

HF_CACHE        = "/workspace/hf_cache"

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

_TAXONOMY = """\
Use the following grounding state taxonomy to guide your judgment:

  CLEAR             — target and destination are uniquely identifiable at expected
                      positions; the robot can safely continue.
  AMBIGUOUS_TARGET  — multiple target candidates exist or the target has moved to an
                      unexpected position; ask the user which one to pick.
  INVALID_TARGET    — the target object cannot be found in the scene; the robot must
                      stop immediately (safety-critical).
  AMBIGUOUS_DESTINATION — multiple destination candidates or destination is unclear;
                      ask the user for clarification.
  INVALID_DESTINATION — the destination is no longer visible; the robot must
                      stop immediately (safety-critical).

Decision mapping:
  CLEAR                    → respond not ambiguous (robot CONTINUES)
  INVALID_TARGET           → respond ambiguous, set stop_bool=True
  INVALID_DESTINATION      → respond ambiguous, set stop_bool=True
  Any other non-CLEAR      → respond ambiguous and ask for user clarification"""


def _fmt(c):     return f"({int(c[0])},{int(c[1])})"
def _fmtl(cs):   return ", ".join(_fmt(c) for c in cs[:5]) if cs else "none"


def prompt_full(task, tgt, dst, g0t, g0d, ckt, ckd, ck):
    """cond1 — G0 + DINO coords + taxonomy"""
    g0_t = _fmt(g0t) if g0t else "unknown"
    g0_d = _fmt(g0d) if g0d else "unknown"
    return "\n".join([
        f"Original robot task: {task}", "",
        "Initial scene (t0) — stored reference positions:",
        f'  - Target "{tgt}" was at pixel {g0_t}',
        f'  - Destination "{dst}" was at pixel {g0_d}', "",
        f"Checkpoint {ck} — GroundingDINO detection results (authoritative):",
        f'  - "{tgt}": {len(ckt)} detection(s) at [{_fmtl(ckt)}]',
        f'  - "{dst}": {len(ckd)} detection(s) at [{_fmtl(ckd)}]', "",
        "Based ONLY on the detection counts and positions above, assess whether "
        "the task grounding is valid.", "",
        _TAXONOMY,
    ])


def prompt_no_dino(task, tgt, dst, g0t, g0d, ck):
    """cond2 — G0 only, no checkpoint detection"""
    g0_t = _fmt(g0t) if g0t else "unknown"
    g0_d = _fmt(g0d) if g0d else "unknown"
    return "\n".join([
        f"Original robot task: {task}", "",
        "Initial scene (t0) — stored reference positions:",
        f'  - Target "{tgt}" was at pixel {g0_t}',
        f'  - Destination "{dst}" was at pixel {g0_d}', "",
        f"Checkpoint {ck} — No automatic detection. "
        "Examine the scene image directly.",
        "Assess whether the target and destination are still uniquely "
        "identifiable and at their expected positions.", "",
        _TAXONOMY,
    ])


def prompt_no_g0(task, tgt, dst, ckt, ckd, ck):
    """cond4 — DINO coords only, no G0 reference"""
    return "\n".join([
        f"Original robot task: {task}", "",
        f"Checkpoint {ck} — GroundingDINO detection results (authoritative):",
        f'  - "{tgt}": {len(ckt)} detection(s) at [{_fmtl(ckt)}]',
        f'  - "{dst}": {len(ckd)} detection(s) at [{_fmtl(ckd)}]', "",
        "Based ONLY on the detection counts and positions above, assess whether "
        "the task grounding is valid.", "",
        _TAXONOMY,
    ])


def prompt_no_tax(task, tgt, dst, g0t, g0d, ckt, ckd, ck):
    """cond5 — G0 + DINO coords, no taxonomy"""
    g0_t = _fmt(g0t) if g0t else "unknown"
    g0_d = _fmt(g0d) if g0d else "unknown"
    return "\n".join([
        f"Original robot task: {task}", "",
        "Initial scene (t0) — stored reference positions:",
        f'  - Target "{tgt}" was at pixel {g0_t}',
        f'  - Destination "{dst}" was at pixel {g0_d}', "",
        f"Checkpoint {ck} — GroundingDINO detection results (authoritative):",
        f'  - "{tgt}": {len(ckt)} detection(s) at [{_fmtl(ckt)}]',
        f'  - "{dst}": {len(ckd)} detection(s) at [{_fmtl(ckd)}]', "",
        "Based on the detection results, assess whether the task grounding is "
        "valid. Output JSON with task_ambiguous (bool), stop_bool (bool), "
        "and clarifying_question (string).",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_base_merged():
    """Molmo-7B + AmbRes LoRA merged → base for both FT and pre-FT."""
    import os; os.environ["HF_HOME"] = HF_CACHE
    print("  [1/2] Loading Molmo-7B base...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"  [2/2] Merging AmbRes LoRA: {AMBRES_CKPT}")
    model = PeftModel.from_pretrained(model, AMBRES_CKPT)
    model = model.merge_and_unload()
    model.eval()
    return model, processor


def add_ft_lora(model):
    """Load v3 fine-tuned LoRA on top of merged base."""
    print(f"  [+LoRA] Loading v3 LoRA: {FT_CKPT}")
    model = PeftModel.from_pretrained(model, FT_CKPT)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def infer(model, processor, img_path: str, prompt: str) -> dict:
    img = Image.open(img_path).convert("RGB")
    messages = [{"role": "user",
                 "content": f"<image>\nTASK DESCRIPTION: {prompt}"}]
    inputs = process_msg_list(messages, [img], processor)
    inputs = {k: v.to(device=model.device).unsqueeze(0) for k, v in inputs.items()}
    inputs["images"]      = inputs["images"].to(model.dtype)
    inputs["image_masks"] = inputs["image_masks"].to(model.dtype)

    gen_cfg = GenerationConfig(
        do_sample=False, max_new_tokens=256, stop_strings="<|endoftext|>",
    )
    with torch.no_grad():
        out = model.generate_from_batch(
            inputs, gen_cfg, tokenizer=processor.tokenizer,
            return_dict_in_generate=True,
        )
    tokens = out.sequences[0, inputs["input_ids"].size(1):]
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)

    # JSON 파싱
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"task_ambiguous": False, "stop_bool": False, "clarifying_question": text[:100]}


def parse_decision(result: dict) -> str:
    ambig = bool(result.get("task_ambiguous", result.get("ambiguity_bool", False)))
    stop  = bool(result.get("stop_bool", False))
    cq    = str(result.get("clarifying_question", "")).upper()
    if not ambig:
        return "CONTINUE"
    if stop or cq.startswith("STOP:"):
        return "STOP"
    return "ASK"


def gold_decision(gold_state: str) -> str:
    if gold_state == "CLEAR":
        return "CONTINUE"
    if gold_state in ("INVALID_TARGET", "INVALID_DESTINATION", "UNSAFE_OR_BLOCKED"):
        return "STOP"
    return "ASK"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(model, processor, samples, prompts, label):
    results = []
    for i, (s, prompt) in enumerate(zip(samples, prompts)):
        raw = infer(model, processor, s["image"], prompt)
        pred = parse_decision(raw)
        gold = gold_decision(s["gold_state"])
        ok   = pred == gold
        results.append({
            "id": s["id"],
            "gold_state": s["gold_state"],
            "gold_decision": gold,
            "predicted_decision": pred,
            "correct": ok,
            "raw_output": raw,
        })
        mark = "✓" if ok else "✗"
        print(f"  [{i+1:2d}/{len(samples)}] {mark} {s['id'][:38]:<38} "
              f"gold={gold:<8} pred={pred}  raw={json.dumps(raw)[:60]}")
    return results


def compute_metrics(results):
    total   = len(results)
    correct = sum(r["correct"] for r in results)
    non_cont = [r for r in results if r["gold_decision"] != "CONTINUE"]
    cont_only = [r for r in results if r["gold_decision"] == "CONTINUE"]
    miss = sum(1 for r in non_cont  if r["predicted_decision"] == "CONTINUE")
    fa   = sum(1 for r in cont_only if r["predicted_decision"] != "CONTINUE")
    per  = defaultdict(lambda: [0, 0])
    for r in results:
        per[r["gold_state"]][1] += 1
        if r["correct"]:
            per[r["gold_state"]][0] += 1
    return {
        "decision_accuracy": correct / total if total else 0,
        "miss_rate":         miss / len(non_cont) if non_cont else 0,
        "false_alarm_rate":  fa   / len(cont_only) if cont_only else 0,
        "correct": correct, "total": total,
        "per_state": {s: f"{v[0]}/{v[1]}" for s, v in sorted(per.items())},
    }


def print_summary(label, m):
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"{'─'*62}")
    print(f"  Accuracy : {m['decision_accuracy']*100:.1f}%  ({m['correct']}/{m['total']})")
    print(f"  Miss     : {m['miss_rate']*100:.1f}%")
    print(f"  FAR      : {m['false_alarm_rate']*100:.1f}%")
    for state, score in m["per_state"].items():
        print(f"    {state:<35} {score}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", default="dataset/finetune/val.jsonl")
    parser.add_argument("--conditions", nargs="+",
                        choices=["cond1","cond2","cond3","cond4","cond5"],
                        default=["cond1","cond2","cond3","cond4","cond5"])
    parser.add_argument("--out-csv",  default="logs/eval_conditions.csv")
    parser.add_argument("--out-json", default="logs/eval_conditions_metrics.json")
    args = parser.parse_args()

    samples = [json.loads(l) for l in (REPO / args.val).read_text().splitlines() if l.strip()]
    print(f"val 샘플: {len(samples)}개")
    print("gold_state:", dict(Counter(s["gold_state"] for s in samples)))

    # 프롬프트 사전 생성
    P = {k: [] for k in ["cond1","cond2","cond3","cond4","cond5"]}
    for s in samples:
        meta = s.get("meta", {})
        g0t  = meta.get("g0_target_coord")
        g0d  = meta.get("g0_dest_coord")
        ckt  = meta.get("ck_target_coords", [])
        ckd  = meta.get("ck_dest_coords", [])
        ck   = s.get("checkpoint", "C1")
        tgt  = s.get("target_label", "target")
        dst  = s.get("destination_label", "dest")
        task = s.get("task_description", "").split("\n")[0].replace("Original robot task: ", "")
        P["cond1"].append(prompt_full   (task, tgt, dst, g0t, g0d, ckt, ckd, ck))
        P["cond2"].append(prompt_no_dino(task, tgt, dst, g0t, g0d,          ck))
        P["cond3"].append(prompt_full   (task, tgt, dst, g0t, g0d, ckt, ckd, ck))  # same as cond1
        P["cond4"].append(prompt_no_g0  (task, tgt, dst,           ckt, ckd, ck))
        P["cond5"].append(prompt_no_tax (task, tgt, dst, g0t, g0d, ckt, ckd, ck))

    all_results = {}
    all_metrics = {}
    needs_ft    = [c for c in args.conditions if c != "cond3"]
    needs_preft = "cond3" in args.conditions

    # ── FT 모델 (cond1, 2, 4, 5) ─────────────────────────────────────────────
    if needs_ft:
        print("\n[FT 모델 로드]")
        model, processor = load_base_merged()
        model = add_ft_lora(model)

        for cond in ["cond1","cond2","cond4","cond5"]:
            if cond not in args.conditions:
                continue
            labels = {
                "cond1": "cond1: FT + DINO coords",
                "cond2": "cond2: FT - DINO (image only)",
                "cond4": "cond4: FT + DINO, no G0",
                "cond5": "cond5: FT + DINO, no taxonomy",
            }
            print(f"\n▶ {labels[cond]}")
            r = run_condition(model, processor, samples, P[cond], cond)
            all_results[labels[cond]] = r
            all_metrics[labels[cond]] = compute_metrics(r)
            print_summary(labels[cond], all_metrics[labels[cond]])

        del model
        import gc; gc.collect(); torch.cuda.empty_cache()
        print("\n[VRAM 정리]")

    # ── pre-FT 모델 (cond3) ───────────────────────────────────────────────────
    if needs_preft:
        print("\n[pre-FT 모델 로드] (Molmo + AmbRes merged, no v3 LoRA)")
        model, processor = load_base_merged()

        label3 = "cond3: pre-FT + DINO coords"
        print(f"\n▶ {label3}")
        r = run_condition(model, processor, samples, P["cond3"], "cond3")
        all_results[label3] = r
        all_metrics[label3] = compute_metrics(r)
        print_summary(label3, all_metrics[label3])

        del model
        import gc; gc.collect(); torch.cuda.empty_cache()

    # ── 최종 비교표 ───────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print("  최종 비교")
    print(f"{'═'*62}")
    print(f"  {'조건':<35} {'Acc':>6} {'Miss':>6} {'FAR':>6}")
    print(f"  {'─'*55}")
    for name, m in all_metrics.items():
        print(f"  {name:<35} {m['decision_accuracy']*100:>5.1f}% "
              f"{m['miss_rate']*100:>5.1f}% "
              f"{m['false_alarm_rate']*100:>5.1f}%")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    import csv
    out_csv  = REPO / args.out_csv
    out_json = REPO / args.out_json
    out_csv.parent.mkdir(exist_ok=True)

    rows = [{"condition": cond, **{k: v for k, v in r.items() if k != "raw_output"}}
            for cond, results in all_results.items() for r in results]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    out_json.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"\n저장: {out_csv}")
    print(f"메트릭: {out_json}")


if __name__ == "__main__":
    main()
