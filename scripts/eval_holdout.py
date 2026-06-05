#!/usr/bin/env python3
"""
Holdout evaluation on vla-evaluation-v4 (unseen data).

조건 정의:
  cond1: FT + DINO + G0      (full system — target condition)
  cond2: FT image only       (no DINO, no G0 — FT 기여 측정용)
  cond3: native pre-FT       (AmbRes native handler, task+image만 입력, DINO/G0/taxonomy 없음)
                              ※ cond1과 입력 형식이 다르므로 controlled 비교 불가.
                                "native pre-FT AmbRes baseline"으로 해석할 것.
  cond4: FT + DINO, no G0    (G0 없음 — DINO 기여 측정용)
  cond5: FT + DINO, no tax   (taxonomy 없음 — taxonomy 기여 측정용)

  [추후 추가 예정]
  cond3_full_prompt: pre-FT (AmbRes만, v2 LoRA 없음) + 동일 prompt_full 입력
                     → cond1과 완전히 통제된 pre-FT 비교

DINO를 inference 시 실시간 실행해서 G0 + checkpoint coords 생성.
clarifying_question도 CSV에 저장.

Usage:
  python3 scripts/eval_holdout.py
  python3 scripts/eval_holdout.py --conditions cond1 cond2 cond4
"""

from __future__ import annotations

import argparse
import csv
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

MODEL_ID     = "allenai/Molmo-7B-D-0924"
AMBRES_CKPT  = "/workspace/AmbRes/ckpt/AB5siP8DA78aA78wR5Y8Mw/checkpoint-390"
FT_CKPT_DEFAULT = str(REPO / "ckpt/grounding_ft_v2/7wzpjiVc/checkpoint-192")
HF_CACHE     = "/workspace/hf_cache"
V4_DIR       = REPO / "dataset/vla-evaluation-v4"
MANIFEST     = V4_DIR / "manifest_eval-v4.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders (동일 구조 유지)
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


def _fmt(c):   return f"({int(c[0])},{int(c[1])})"
def _fmtl(cs): return ", ".join(_fmt(c) for c in cs[:5]) if cs else "none"


def prompt_full(task, tgt, dst, g0t, g0d, ckt, ckd, ck):
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


def prompt_image_only(task, tgt, dst, ck):
    """FT + image only: G₀ 없음, DINO 없음. 순수 VLM 시각 판단.
    초기 위치 참조 없음을 명시해 'expected position' 판단을 유도하지 않음."""
    return "\n".join([
        f"Original robot task: {task}", "",
        f"Checkpoint {ck} — No detection or initial reference available.",
        "No initial positions are recorded. Examine the scene image directly.",
        f'Assess whether the target "{tgt}" and destination "{dst}" are',
        "uniquely identifiable in the current scene.", "",
        _TAXONOMY,
    ])


def prompt_no_g0(task, tgt, dst, ckt, ckd, ck):
    return "\n".join([
        f"Original robot task: {task}", "",
        "Initial scene (t0) — stored reference positions:",
        f'  - Target "{tgt}": not available (G₀ not recorded)',
        f'  - Destination "{dst}": not available (G₀ not recorded)', "",
        f"Checkpoint {ck} — GroundingDINO detection results (authoritative):",
        f'  - "{tgt}": {len(ckt)} detection(s) at [{_fmtl(ckt)}]',
        f'  - "{dst}": {len(ckd)} detection(s) at [{_fmtl(ckd)}]', "",
        "G₀ reference positions are unavailable. Assess validity based solely on "
        "detection counts.", "",
        _TAXONOMY,
    ])


def prompt_no_tax(task, tgt, dst, g0t, g0d, ckt, ckd, ck):
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
        "Based on the detection results, assess whether the task grounding is valid. "
        "Output JSON with task_ambiguous (bool), stop_bool (bool), "
        "and clarifying_question (string).",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DINO detection
# ─────────────────────────────────────────────────────────────────────────────

def load_dino():
    import os; os.environ["HF_HOME"] = HF_CACHE
    sys.path.insert(0, str(REPO))
    from module.models.ambres.gdino import GroundingDINO
    print("[DINO] 로딩...")
    dino = GroundingDINO(box_threshold=0.25, text_threshold=0.25)
    print("[DINO] 완료")
    return dino


def dino_coords(dino, img_path: str | Path, labels: list[str]) -> dict[str, list]:
    pil = Image.open(img_path).convert("RGB")
    det = dino.detect(pil, labels)
    return {lbl: [[int(c[0]), int(c[1])] for c in (det.get(lbl) or []) if len(c) == 2]
            for lbl in labels}


def precompute_coords(samples: list[dict], dino) -> list[dict]:
    """각 trial에 대해 G0 + checkpoint coords 미리 계산."""
    print(f"\n[DINO] {len(samples)}개 trial 탐지 중...")
    results = []
    for i, s in enumerate(samples):
        tgt, dst = s["target_label"], s["destination_label"]
        t0_img = s["t0_img"]
        ck_img = s["ck_img"]

        t0_det = dino_coords(dino, t0_img, [tgt, dst])
        ck_det = dino_coords(dino, ck_img, [tgt, dst])

        g0_tgt = t0_det[tgt][0] if t0_det[tgt] else None
        g0_dst = t0_det[dst][0] if t0_det[dst] else None
        ck_tgt = ck_det[tgt]
        ck_dst = ck_det[dst]

        results.append({**s,
                        "g0_tgt": g0_tgt, "g0_dst": g0_dst,
                        "ck_tgt": ck_tgt, "ck_dst": ck_dst})
        print(f"  [{i+1:2d}/{len(samples)}] {s['id']:<30} "
              f"g0_tgt={g0_tgt}  ck_tgt={len(ck_tgt)}개  ck_dst={len(ck_dst)}개")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ─────────────────────────────────────────────────────────────────────────────

def load_base_merged():
    import os; os.environ["HF_HOME"] = HF_CACHE
    print("  [1/2] Loading Molmo-7B...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    print(f"  [2/2] Merging AmbRes LoRA: {AMBRES_CKPT}")
    model = PeftModel.from_pretrained(model, AMBRES_CKPT)
    model = model.merge_and_unload()
    model.eval()
    return model, processor


def add_ft_lora(model, ft_ckpt: str):
    print(f"  [+LoRA] {ft_ckpt}")
    model = PeftModel.from_pretrained(model, ft_ckpt)
    model.eval()
    return model


def load_native_handler():
    """pre-FT 모델을 native AmbRes handle_query 방식으로 로드.
    outlines logits processor를 사용해 task_ambiguous 키를 강제 출력."""
    import os; os.environ["HF_HOME"] = HF_CACHE
    sys.path.insert(0, str(REPO / "src"))
    from extraction.ambres_g0_extractor import _make_handler
    print(f"  [native AmbRes] loading: AB5siP8DA78aA78wR5Y8Mw")
    # AmbresFineTuned: ASSETS_DIR.CKPT / adapter_ckpt 경로를 사용.
    # AB5siP8DA78aA78wR5Y8Mw 디렉토리 안에 checkpoint-390/ 가 있어야 함.
    handler = _make_handler("finetune", "AB5siP8DA78aA78wR5Y8Mw",
                            use_detection=False)
    return handler


def infer(model, processor, img_path: str, prompt: str) -> dict:
    img = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": f"<image>\nTASK DESCRIPTION: {prompt}"}]
    inputs = process_msg_list(messages, [img], processor)
    inputs = {k: v.to(device=model.device).unsqueeze(0) for k, v in inputs.items()}
    inputs["images"]      = inputs["images"].to(model.dtype)
    inputs["image_masks"] = inputs["image_masks"].to(model.dtype)
    cfg = GenerationConfig(do_sample=False, max_new_tokens=256,
                           stop_strings="<|endoftext|>")
    with torch.no_grad():
        out = model.generate_from_batch(inputs, cfg,
                                        tokenizer=processor.tokenizer,
                                        return_dict_in_generate=True)
    tokens = out.sequences[0, inputs["input_ids"].size(1):]
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"task_ambiguous": False, "stop_bool": False, "clarifying_question": text[:120]}


def parse_decision(r: dict) -> str:
    ambig = bool(r.get("task_ambiguous", r.get("ambiguity_bool", False)))
    stop  = bool(r.get("stop_bool", False))
    cq    = str(r.get("clarifying_question", "")).upper()
    if not ambig:          return "CONTINUE"
    if stop or cq.startswith("STOP:"): return "STOP"
    return "ASK"


def run_condition_native(handler, samples, label):
    """Native AmbRes handle_query 방식으로 cond3 실행.
    outlines 강제로 task_ambiguous 키 보장. STOP 개념 없음 (pre-FT 한계)."""
    from extraction.ambres_g0_extractor import _load_image, _unwrap_handler_result
    results = []
    for i, s in enumerate(samples):
        img_arr, pil_img = _load_image(s["ck_img"])
        session = f"native_{i}"
        _unwrap_handler_result(handler.handle("reset", {}, [], session), "reset")
        raw = _unwrap_handler_result(
            handler.handle("query", {"task_description": s["task"]},
                           [img_arr], session), "query")

        ambig = bool(raw.get("task_ambiguous", False))
        cq    = str(raw.get("clarifying_question", ""))
        pred  = "ASK" if ambig else "CONTINUE"
        gold  = gold_decision(s["gold_state"])
        ok    = pred == gold
        results.append({
            "id": s["id"], "scenario": s["scenario"],
            "gold_state": s["gold_state"], "gold_decision": gold,
            "predicted_decision": pred, "correct": ok,
            "clarifying_question": cq,
        })
        mark = "✓" if ok else "✗"
        print(f"  [{i+1:2d}/{len(samples)}] {mark} {s['id']:<28} "
              f"gold={gold:<8} pred={pred:<8}  q=\"{cq[:50]}\"")
    return results


def gold_decision(gs: str) -> str:
    if gs == "CLEAR": return "CONTINUE"
    if gs in ("INVALID_TARGET", "INVALID_DESTINATION", "UNSAFE_OR_BLOCKED"): return "STOP"
    return "ASK"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(model, processor, samples_with_coords, prompts, label):
    results = []
    for i, (s, prompt) in enumerate(zip(samples_with_coords, prompts)):
        raw  = infer(model, processor, s["ck_img"], prompt)
        pred = parse_decision(raw)
        gold = gold_decision(s["gold_state"])
        cq   = raw.get("clarifying_question", "")
        ok   = pred == gold
        results.append({
            "id": s["id"], "scenario": s["scenario"],
            "gold_state": s["gold_state"], "gold_decision": gold,
            "predicted_decision": pred, "correct": ok,
            "clarifying_question": cq,
        })
        mark = "✓" if ok else "✗"
        print(f"  [{i+1:2d}/{len(samples_with_coords)}] {mark} {s['id']:<28} "
              f"gold={gold:<8} pred={pred:<8}  q=\"{cq[:50]}\"")
    return results


def compute_metrics(results):
    total    = len(results)
    correct  = sum(r["correct"] for r in results)
    non_cont = [r for r in results if r["gold_decision"] != "CONTINUE"]
    cont     = [r for r in results if r["gold_decision"] == "CONTINUE"]
    miss = sum(1 for r in non_cont if r["predicted_decision"] == "CONTINUE")
    fa   = sum(1 for r in cont    if r["predicted_decision"] != "CONTINUE")
    per  = defaultdict(lambda: [0, 0])
    for r in results:
        per[r["gold_state"]][1] += 1
        if r["correct"]: per[r["gold_state"]][0] += 1
    return {
        "decision_accuracy": correct / total if total else 0,
        "miss_rate":         miss / len(non_cont) if non_cont else 0,
        "false_alarm_rate":  fa   / len(cont)     if cont     else 0,
        "correct": correct, "total": total,
        "per_state": {s: f"{v[0]}/{v[1]}" for s, v in sorted(per.items())},
    }


def print_summary(label, m):
    print(f"\n{'─'*64}")
    print(f"  {label}")
    print(f"{'─'*64}")
    print(f"  Accuracy : {m['decision_accuracy']*100:.1f}%  ({m['correct']}/{m['total']})")
    print(f"  Miss     : {m['miss_rate']*100:.1f}%")
    print(f"  FAR      : {m['false_alarm_rate']*100:.1f}%")
    for state, score in m["per_state"].items():
        print(f"    {state:<35} {score}")


def print_question_samples(all_results):
    """cond1 결과에서 시나리오별 question 샘플 출력."""
    cond1_key = next((k for k in all_results if "cond1" in k), None)
    if not cond1_key:
        return
    print(f"\n{'═'*64}")
    print("  Clarifying Question 샘플 (cond1: FT + DINO)")
    print(f"{'═'*64}")
    by_state = defaultdict(list)
    for r in all_results[cond1_key]:
        by_state[r["gold_state"]].append(r)
    for state in sorted(by_state.keys()):
        samples = by_state[state]
        print(f"\n  [{state}]")
        for r in samples[:3]:
            cq = r["clarifying_question"]
            mark = "✓" if r["correct"] else "✗"
            print(f"    {mark} {r['id']:<28} → \"{cq}\"")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--ft-ckpt",  default=FT_CKPT_DEFAULT,
                        help="FT LoRA checkpoint 경로 (default: v2 최신)")
    parser.add_argument("--conditions", nargs="+",
                        choices=["cond1","cond2","cond3","cond4","cond5"],
                        default=["cond1","cond2","cond3","cond4","cond5"])
    parser.add_argument("--out-csv",  default="logs/eval_holdout.csv")
    parser.add_argument("--out-json", default="logs/eval_holdout_metrics.json")
    args = parser.parse_args()

    # ── manifest 로드 + 경로 절대화 ───────────────────────────────────────────
    raw = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    samples = []
    for e in raw:
        ck = e.get("checkpoint", "C1")
        ck_img_rel = e["c1_img"] if ck == "C1" else e["c2_img"]
        samples.append({
            "id":            e["id"],
            "scenario":      e["scenario"],
            "task":          e["task"],
            "target_label":  e["target_label"],
            "destination_label": e["destination_label"],
            "gold_state":    e["gold_state"],
            "gold_decision": e["gold_decision"],
            "checkpoint":    ck,
            "t0_img":        str(V4_DIR / e["initial_img"]),
            "ck_img":        str(V4_DIR / ck_img_rel),
        })
    print(f"Holdout 샘플: {len(samples)}개")
    print("시나리오:", dict(Counter(s["scenario"] for s in samples)))
    print("gold_state:", dict(Counter(s["gold_state"] for s in samples)))

    # ── DINO로 coords 사전 계산 ────────────────────────────────────────────────
    dino = load_dino()
    samples = precompute_coords(samples, dino)
    del dino
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("\n[DINO 완료, VRAM 정리]")

    # ── 프롬프트 생성 ─────────────────────────────────────────────────────────
    P = {k: [] for k in ["cond1","cond2","cond3","cond4","cond5"]}
    for s in samples:
        task = s["task"]
        tgt, dst = s["target_label"], s["destination_label"]
        g0t, g0d = s["g0_tgt"], s["g0_dst"]
        ckt, ckd = s["ck_tgt"], s["ck_dst"]
        ck = s["checkpoint"]
        P["cond1"].append(prompt_full      (task, tgt, dst, g0t, g0d, ckt, ckd, ck))
        P["cond2"].append(prompt_image_only(task, tgt, dst,                    ck))
        P["cond3"].append(prompt_full      (task, tgt, dst, g0t, g0d, ckt, ckd, ck))
        P["cond4"].append(prompt_no_g0     (task, tgt, dst,           ckt, ckd, ck))
        P["cond5"].append(prompt_no_tax    (task, tgt, dst, g0t, g0d, ckt, ckd, ck))

    all_results = {}
    all_metrics = {}
    needs_ft    = [c for c in args.conditions if c != "cond3"]
    needs_preft = "cond3" in args.conditions

    # ── FT 모델 ───────────────────────────────────────────────────────────────
    if needs_ft:
        print(f"\n[FT 모델 로드] {args.ft_ckpt}")
        model, processor = load_base_merged()
        model = add_ft_lora(model, args.ft_ckpt)

        for cond, label in [("cond1","cond1: FT + DINO + G0"),
                             ("cond2","cond2: FT image only"),
                             ("cond4","cond4: FT + DINO, no G0"),
                             ("cond5","cond5: FT + DINO, no taxonomy")]:
            if cond not in args.conditions:
                continue
            print(f"\n▶ {label}")
            r = run_condition(model, processor, samples, P[cond], cond)
            all_results[label] = r
            all_metrics[label] = compute_metrics(r)
            print_summary(label, all_metrics[label])

        del model
        gc.collect(); torch.cuda.empty_cache()
        print("\n[VRAM 정리]")

    # ── pre-FT 모델 (native AmbRes handle_query) ─────────────────────────────
    if needs_preft:
        print("\n[pre-FT native 로드] outlines logits processor 사용 → task_ambiguous 강제")
        handler_native = load_native_handler()
        label3 = "cond3: pre-FT native (visual only, no STOP)"
        print(f"\n▶ {label3}")
        r = run_condition_native(handler_native, samples, label3)
        all_results[label3] = r
        all_metrics[label3] = compute_metrics(r)
        print_summary(label3, all_metrics[label3])
        del handler_native
        gc.collect(); torch.cuda.empty_cache()

    # ── Question 샘플 출력 ────────────────────────────────────────────────────
    print_question_samples(all_results)

    # ── 최종 비교표 ───────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("  최종 비교 (Holdout — vla-evaluation-v4)")
    print(f"{'═'*64}")
    print(f"  {'조건':<35} {'Acc':>6} {'Miss':>6} {'FAR':>6}")
    print(f"  {'─'*53}")
    order = ["cond2: FT image only",
             "cond4: FT + DINO, no G0",
             "cond1: FT + DINO + G0",
             "cond5: FT + DINO, no taxonomy",
             "cond3: pre-FT native (visual only, no STOP)"]
    for name in order:
        if name not in all_metrics:
            continue
        m = all_metrics[name]
        print(f"  {name:<35} {m['decision_accuracy']*100:>5.1f}% "
              f"{m['miss_rate']*100:>5.1f}% "
              f"{m['false_alarm_rate']*100:>5.1f}%")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_csv  = REPO / args.out_csv
    out_json = REPO / args.out_json
    out_csv.parent.mkdir(exist_ok=True)

    rows = [{"condition": cond, **r}
            for cond, results in all_results.items() for r in results]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    out_json.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"\n저장: {out_csv}")
    print(f"메트릭: {out_json}")


if __name__ == "__main__":
    main()
