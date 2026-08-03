#!/usr/bin/env python3
"""
Counterfactual G₀ Diagnostic Evaluation.

같은 이미지, 같은 ck coords에서 G₀만 바꿔 모델이 G₀를 실제로 사용하는지 검증.

  G₀_same    (G₀ = ck 좌표)           → 예상 decision: CONTINUE  (위치 변화 없음)
  G₀_shifted (G₀ = ck + 200px offset) → 예상 decision: ASK       (위치 변화 감지)

조건 대상: ck에서 count=1인 trial (count≠1은 G₀ 비교로 판단 불가능하므로 제외)

Memory Reliance Rate (MRR) = G₀_same→CONTINUE AND G₀_shifted→ASK 인 trial 수 / 전체 eligible

Usage:
  # v1 LoRA (학습 전 비교용)
  python3 scripts/eval_counterfactual.py --ft-ckpt ckpt/grounding_ft/BvFsfsf4/checkpoint-162

  # v2 LoRA (새 학습 결과)
  python3 scripts/eval_counterfactual.py --ft-ckpt ckpt/grounding_ft_v2/<run_id>/checkpoint-NNN

  # 두 버전 비교
  python3 scripts/eval_counterfactual.py \\
      --ft-ckpt ckpt/grounding_ft_v2/<run_id>/checkpoint-NNN \\
      --compare-ckpt ckpt/grounding_ft/BvFsfsf4/checkpoint-162
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

REPO       = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "AmbRes"))

from ambres.training.data import process_msg_list

MODEL_ID    = "allenai/Molmo-7B-D-0924"
AMBRES_CKPT = "/workspace/AmbRes/ckpt/AB5siP8DA78aA78wR5Y8Mw/checkpoint-390"
HF_CACHE    = "/workspace/hf_cache"
V4_DIR      = REPO / "dataset/vla-evaluation-v4"
MANIFEST    = V4_DIR / "manifest_eval-v4.jsonl"

# G₀ shift 크기 (px). count=1에서 "이동했다"고 판단할 수 있는 충분한 거리
SHIFT_PX = 200


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
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


def shifted_g0(coord: list[int], shift_px: int) -> list[int]:
    """이미지 안에서 최대한 멀리 이동. x축 기준 반대방향으로 shift_px."""
    cx = coord[0]
    dx = -shift_px if cx > 320 else +shift_px
    return [cx + dx, coord[1]]


# ─────────────────────────────────────────────────────────────────────────────
# DINO
# ─────────────────────────────────────────────────────────────────────────────

def load_dino():
    import os; os.environ["HF_HOME"] = HF_CACHE
    sys.path.insert(0, str(REPO))
    from module.models.ambres.gdino import GroundingDINO
    print("[DINO] 로딩...")
    dino = GroundingDINO(box_threshold=0.25, text_threshold=0.25)
    print("[DINO] 완료")
    return dino


def dino_coords(dino, img_path, labels):
    pil = Image.open(img_path).convert("RGB")
    det = dino.detect(pil, labels)
    return {lbl: [[int(c[0]), int(c[1])] for c in (det.get(lbl) or []) if len(c) == 2]
            for lbl in labels}


def precompute_dino(samples, dino):
    print(f"\n[DINO] {len(samples)}개 trial 탐지 중...")
    out = []
    for i, s in enumerate(samples):
        tgt, dst = s["target_label"], s["destination_label"]
        t0_det = dino_coords(dino, s["t0_img"], [tgt, dst])
        ck_det = dino_coords(dino, s["ck_img"], [tgt, dst])
        g0_tgt = t0_det[tgt][0] if t0_det[tgt] else None
        g0_dst = t0_det[dst][0] if t0_det[dst] else None
        ck_tgt = ck_det[tgt]
        ck_dst = ck_det[dst]
        out.append({**s, "g0_tgt": g0_tgt, "g0_dst": g0_dst,
                    "ck_tgt": ck_tgt, "ck_dst": ck_dst})
        print(f"  [{i+1:2d}/{len(samples)}] {s['id']:<28} "
              f"ck={s['checkpoint']} tgt_n={len(ck_tgt)} dst_n={len(ck_dst)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ft_ckpt: str):
    import os; os.environ["HF_HOME"] = HF_CACHE
    print(f"  [1/2] Loading Molmo-7B...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    print(f"  [2/2] Merging AmbRes LoRA...")
    model = PeftModel.from_pretrained(model, AMBRES_CKPT)
    model = model.merge_and_unload()
    print(f"  [3/3] Loading FT LoRA: {ft_ckpt}")
    model = PeftModel.from_pretrained(model, ft_ckpt)
    model.eval()
    return model, processor


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
    text   = processor.tokenizer.decode(tokens, skip_special_tokens=True)
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
    if not ambig:   return "CONTINUE"
    if stop:        return "STOP"
    return "ASK"


# ─────────────────────────────────────────────────────────────────────────────
# Counterfactual evaluation
# ─────────────────────────────────────────────────────────────────────────────

def is_eligible(s: dict) -> bool:
    """count=1인 trial만 counterfactual 대상."""
    ck = s["checkpoint"]
    if ck == "C1":
        return len(s["ck_tgt"]) == 1
    else:
        return len(s["ck_dst"]) == 1


def build_cf_prompts(s: dict, shift_px: int) -> tuple[str, str, list[int], list[int]]:
    """
    테스트 대상 role의 G₀만 변경. 비테스트 role은 ck 현재 좌표로 고정해
    DINO t0 오류에 의한 confound를 제거한다.

      C1 (target 테스트): g0_tgt 변경, g0_dst = ck_dst[0] (고정)
      C2 (dest 테스트):   g0_dst 변경, g0_tgt = ck_tgt[0] (고정)

    Returns:
        prompt_same, prompt_shifted, g0_same_coord, g0_shifted_coord
    """
    task     = s["task"]
    tgt, dst = s["target_label"], s["destination_label"]
    ck       = s["checkpoint"]
    ckt, ckd = s["ck_tgt"], s["ck_dst"]

    if ck == "C1":
        ref_coord = ckt[0]
        g0_same   = ref_coord
        g0_sh     = shifted_g0(ref_coord, shift_px)
        # 비테스트 role(dest): ck 좌표로 고정 (t0 DINO 오류 배제)
        g0_dst_fixed = ckd[0] if ckd else s["g0_dst"]
        p_same    = prompt_full(task, tgt, dst, g0_same, g0_dst_fixed, ckt, ckd, ck)
        p_shifted = prompt_full(task, tgt, dst, g0_sh,   g0_dst_fixed, ckt, ckd, ck)
    else:  # C2
        ref_coord = ckd[0]
        g0_same   = ref_coord
        g0_sh     = shifted_g0(ref_coord, shift_px)
        # 비테스트 role(target): ck 좌표로 고정
        g0_tgt_fixed = ckt[0] if ckt else s["g0_tgt"]
        p_same    = prompt_full(task, tgt, dst, g0_tgt_fixed, g0_same, ckt, ckd, ck)
        p_shifted = prompt_full(task, tgt, dst, g0_tgt_fixed, g0_sh,   ckt, ckd, ck)

    return p_same, p_shifted, g0_same, g0_sh


def run_counterfactual(model, processor, samples_with_coords: list[dict],
                       shift_px: int) -> list[dict]:
    eligible   = [s for s in samples_with_coords if is_eligible(s)]
    ineligible = [s for s in samples_with_coords if not is_eligible(s)]

    print(f"\n  Eligible (count=1): {len(eligible)}개  "
          f"Ineligible (count≠1): {len(ineligible)}개")
    if ineligible:
        skip_info = Counter(s["scenario"] for s in ineligible)
        print(f"  Skip 시나리오: {dict(skip_info)}")

    results = []
    for i, s in enumerate(eligible):
        p_same, p_shifted, g0_same, g0_sh = build_cf_prompts(s, shift_px)

        r_same    = infer(model, processor, s["ck_img"], p_same)
        r_shifted = infer(model, processor, s["ck_img"], p_shifted)

        d_same    = parse_decision(r_same)
        d_shifted = parse_decision(r_shifted)

        # G₀_same → CONTINUE, G₀_shifted → ASK の両方が正解でpassed
        pass_same    = d_same    == "CONTINUE"
        pass_shifted = d_shifted == "ASK"
        passed       = pass_same and pass_shifted

        results.append({
            "id":           s["id"],
            "scenario":     s["scenario"],
            "checkpoint":   s["checkpoint"],
            "gold_state":   s["gold_state"],
            "g0_same":      g0_same,
            "g0_shifted":   g0_sh,
            "dec_same":     d_same,
            "dec_shifted":  d_shifted,
            "pass_same":    pass_same,
            "pass_shifted": pass_shifted,
            "passed":       passed,
            "cq_same":      r_same.get("clarifying_question", ""),
            "cq_shifted":   r_shifted.get("clarifying_question", ""),
        })

        mark = "✓" if passed else "✗"
        s_mark = "✓" if pass_same    else "✗"
        sh_mark = "✓" if pass_shifted else "✗"
        print(f"  [{i+1:2d}/{len(eligible)}] {mark} {s['id']:<28} "
              f"same={s_mark}{d_same:<9} shifted={sh_mark}{d_shifted:<5} "
              f"scenario={s['scenario']}")

    return results


def compute_mrr(results: list[dict]) -> dict:
    total     = len(results)
    passed    = sum(r["passed"]       for r in results)
    same_ok   = sum(r["pass_same"]    for r in results)
    shifted_ok= sum(r["pass_shifted"] for r in results)

    by_scenario = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        by_scenario[r["scenario"]]["total"]  += 1
        by_scenario[r["scenario"]]["passed"] += r["passed"]

    by_gold = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        by_gold[r["gold_state"]]["total"]  += 1
        by_gold[r["gold_state"]]["passed"] += r["passed"]

    return {
        "mrr":            passed / total if total else 0,
        "same_acc":       same_ok   / total if total else 0,
        "shifted_acc":    shifted_ok/ total if total else 0,
        "passed": passed, "total": total,
        "by_scenario":    {k: f"{v['passed']}/{v['total']}"
                           for k, v in sorted(by_scenario.items())},
        "by_gold_state":  {k: f"{v['passed']}/{v['total']}"
                           for k, v in sorted(by_gold.items())},
    }


def print_mrr_summary(label: str, m: dict):
    print(f"\n{'═'*64}")
    print(f"  Counterfactual Diagnostic — {label}")
    print(f"{'═'*64}")
    print(f"  Memory Reliance Rate (MRR) : {m['mrr']*100:.1f}%  ({m['passed']}/{m['total']})")
    print(f"    G₀_same    → CONTINUE   : {m['same_acc']*100:.1f}%")
    print(f"    G₀_shifted → ASK        : {m['shifted_acc']*100:.1f}%")
    print(f"\n  by scenario:")
    for s, score in m["by_scenario"].items():
        print(f"    {s}  {score}")
    print(f"\n  by gold state:")
    for gs, score in m["by_gold_state"].items():
        print(f"    {gs:<35} {score}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> list[dict]:
    base_dir = manifest_path.parent

    def resolve(rel: str) -> str:
        p = Path(rel)
        return str(p if p.is_absolute() else base_dir / p)

    raw = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    samples = []
    for e in raw:
        ck = e.get("checkpoint", "C1")
        ck_img_rel = e["c1_img"] if ck == "C1" else e["c2_img"]
        samples.append({
            "id":                e["id"],
            "scenario":          e["scenario"],
            "task":              e["task"],
            "target_label":      e["target_label"],
            "destination_label": e["destination_label"],
            "gold_state":        e["gold_state"],
            "checkpoint":        ck,
            "t0_img":            resolve(e["initial_img"]),
            "ck_img":            resolve(ck_img_rel),
        })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",      default=str(MANIFEST))
    parser.add_argument("--ft-ckpt",       required=True,
                        help="평가할 FT LoRA checkpoint 경로")
    parser.add_argument("--compare-ckpt",  default=None,
                        help="비교용 FT LoRA checkpoint (선택, 있으면 두 버전 비교)")
    parser.add_argument("--out-json",      default="logs/eval_counterfactual.json")
    parser.add_argument("--shift-px",      type=int, default=SHIFT_PX,
                        help=f"G₀ shift 크기 (px), default={SHIFT_PX}")
    args = parser.parse_args()
    shift_px = args.shift_px

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    samples = load_manifest(Path(args.manifest))
    print(f"Holdout 샘플: {len(samples)}개")
    print("시나리오:", dict(Counter(s["scenario"] for s in samples)))

    # ── DINO ─────────────────────────────────────────────────────────────────
    dino = load_dino()
    samples = precompute_dino(samples, dino)
    del dino
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("\n[DINO 완료, VRAM 정리]")

    # count=1 eligible 수 미리 출력
    eligible = [s for s in samples if is_eligible(s)]
    print(f"\nEligible trials (count=1 at ck): {len(eligible)}/{len(samples)}")
    print("  by scenario:", dict(Counter(s["scenario"] for s in eligible)))
    print("  by gold_state:", dict(Counter(s["gold_state"] for s in eligible)))

    all_metrics = {}

    # ── 주 평가 모델 ──────────────────────────────────────────────────────────
    ckpts_to_eval = [(args.ft_ckpt, Path(args.ft_ckpt).parent.name)]
    if args.compare_ckpt:
        ckpts_to_eval.append((args.compare_ckpt, Path(args.compare_ckpt).parent.name + " [compare]"))

    for ft_ckpt, label in ckpts_to_eval:
        print(f"\n\n{'█'*64}")
        print(f"  모델: {label}")
        print(f"  ckpt: {ft_ckpt}")
        print(f"{'█'*64}")

        model, processor = load_model(ft_ckpt)

        print(f"\n▶ Counterfactual diagnostic (shift={shift_px}px)")
        results = run_counterfactual(model, processor, samples, shift_px)
        m = compute_mrr(results)
        print_mrr_summary(label, m)
        all_metrics[label] = {"metrics": m, "results": results}

        del model
        gc.collect(); torch.cuda.empty_cache()
        print("\n[VRAM 정리]")

    # ── 비교 요약 (두 모델인 경우) ────────────────────────────────────────────
    if len(all_metrics) == 2:
        labels = list(all_metrics.keys())
        print(f"\n{'═'*64}")
        print("  Comparison Summary")
        print(f"{'═'*64}")
        print(f"  {'Model':<35} {'MRR':>6} {'Same→CONT':>10} {'Shift→ASK':>10}")
        print(f"  {'─'*60}")
        for lbl, d in all_metrics.items():
            m = d["metrics"]
            print(f"  {lbl:<35} "
                  f"{m['mrr']*100:>5.1f}% "
                  f"{m['same_acc']*100:>9.1f}% "
                  f"{m['shifted_acc']*100:>9.1f}%")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_path = REPO / args.out_json
    out_path.parent.mkdir(exist_ok=True)
    # results 안의 bool은 json serializable
    out_path.write_text(json.dumps(
        {k: {**v, "results": v["results"]} for k, v in all_metrics.items()},
        indent=2, ensure_ascii=False,
    ))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
