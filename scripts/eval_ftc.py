#!/usr/bin/env python3
"""
Evaluation script for FT-C (episode memory variant) with Molmo validation.

FT-C was trained with gold episode memory context ("disappeared" for S3).
At inference, GDino force-matching can produce "moved Xpx" instead of "disappeared"
for S3 cases, causing the model to output ASK instead of STOP.

Fix: validate GDino detections with Molmo.
  GDino count=1 (suspected force-match) + Molmo count=0
      → trust Molmo → "target disappeared"  (correct for S3)
  GDino count=1 + Molmo count=1
      → real movement → "target moved Xpx"  (correct for S4)

Usage:
  python3 scripts/eval_ftc.py \\
      --manifest dataset/vla-evaluation-v4/manifest_eval-v4.jsonl \\
      --ft-ckpt  ckpt/ablation/FT-C/final \\
      --out-csv  logs/eval_ftc_memory.csv \\
      --out-json logs/eval_ftc_memory.json
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "AmbRes"))

from ambres.training.data import process_msg_list

MODEL_ID = "allenai/Molmo-7B-D-0924"
HF_CACHE  = "/workspace/hf_cache"

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
  CLEAR                    -> respond not ambiguous (robot CONTINUES)
  INVALID_TARGET           -> respond ambiguous, set stop_bool=True
  INVALID_DESTINATION      -> respond ambiguous, set stop_bool=True
  Any other non-CLEAR      -> respond ambiguous and ask for user clarification"""


def _fmt(c):
    return f"({int(c[0])},{int(c[1])})" if c else "unknown"


def _fmtl(cs):
    return ", ".join(f"({int(c[0])},{int(c[1])})" for c in cs[:5]) if cs else "none"


def build_memory_context(
    tgt_label, dst_label, g0t, g0d,
    ck_tgt_coords, ck_dst_coords,
    molmo_tgt_count=None, molmo_dst_count=None,
):
    """Build episode memory context with optional Molmo validation.

    When Molmo counts are provided:
      GDino count=1 (force-match suspected) + Molmo count=0
          → classify as 'disappeared' (trust Molmo's absence signal)
      GDino count=1 + Molmo count=1
          → classify as 'moved' (real movement confirmed)
    """
    changes = []

    ck_tgt_n = len(ck_tgt_coords)
    ck_dst_n = len(ck_dst_coords)

    # ── Target changes ────────────────────────────────────────────────────
    if ck_tgt_n == 0:
        changes.append(f"target '{tgt_label}' disappeared")
    elif ck_tgt_n > 1:
        changes.append(f"target '{tgt_label}' count increased 1 -> {ck_tgt_n}")
    elif ck_tgt_n == 1 and g0t and ck_tgt_coords:
        dist = ((g0t[0]-ck_tgt_coords[0][0])**2 + (g0t[1]-ck_tgt_coords[0][1])**2)**0.5
        if dist > 50:
            # Molmo validation: if Molmo says 0, GDino force-matched → disappeared
            if molmo_tgt_count is not None and molmo_tgt_count == 0:
                changes.append(f"target '{tgt_label}' disappeared")
            else:
                changes.append(f"target '{tgt_label}' moved {int(dist)}px")

    # ── Destination changes ───────────────────────────────────────────────
    if ck_dst_n == 0:
        changes.append(f"destination '{dst_label}' disappeared")
    elif ck_dst_n > 1:
        changes.append(f"destination '{dst_label}' count increased 1 -> {ck_dst_n}")
    elif ck_dst_n == 1 and g0d and ck_dst_coords:
        dist = ((g0d[0]-ck_dst_coords[0][0])**2 + (g0d[1]-ck_dst_coords[0][1])**2)**0.5
        if dist > 50:
            if molmo_dst_count is not None and molmo_dst_count == 0:
                changes.append(f"destination '{dst_label}' disappeared")
            else:
                changes.append(f"destination '{dst_label}' moved {int(dist)}px")

    if not changes:
        return ""

    # Format must match training exactly (build_dataset.py):
    # - em dash (—) in header
    # - list format [x, y] for coordinates
    # - "C1" or "C2" as checkpoint label (inferred from changes)
    ck_label = "C2" if any("destination" in c for c in changes) else "C1"
    lines = [
        "[Episode Memory — prior observations]",  # em dash
        "[t0]",
        f"  target '{tgt_label}': detected at {list(g0t) if g0t else 'unknown'}",
        f"  destination '{dst_label}': detected at {list(g0d) if g0d else 'unknown'}",
        f"[Now evaluating: {ck_label}]",
        "[Scene changes detected: " + "; ".join(changes) + "]",
    ]
    return "\n".join(lines)


def build_prompt(task, tgt, dst, g0t, g0d, ckt, ckd, ck, memory_ctx=""):
    parts = [f"Original robot task: {task}", ""]
    if memory_ctx:
        parts += [memory_ctx, ""]
    parts += [
        "Initial scene (t0) -- stored reference positions:",
        f'  - Target "{tgt}" was at pixel {_fmt(g0t)}',
        f'  - Destination "{dst}" was at pixel {_fmt(g0d)}',
        "",
        f"Checkpoint {ck} -- GroundingDINO detection results (authoritative):",
        f'  - "{tgt}": {len(ckt)} detection(s) at [{_fmtl(ckt)}]',
        f'  - "{dst}": {len(ckd)} detection(s) at [{_fmtl(ckd)}]',
        "",
        "Based on the detection results and scene history above, assess whether "
        "the task grounding is valid.",
        "",
        _TAXONOMY,
    ]
    return "\n".join(parts)


def load_dino():
    from module.models.ambres.gdino import GroundingDINO
    print("[DINO] loading...")
    dino = GroundingDINO(box_threshold=0.25, text_threshold=0.25)
    print("[DINO] ready")
    return dino


def load_handler():
    sys.path.insert(0, str(REPO / "src"))
    from extraction.ambres_g0_extractor import _make_handler
    print("[Handler] loading AmbRes (for Molmo detection)...")
    handler = _make_handler("fs_prompt", "", use_detection=False)
    print("[Handler] ready")
    return handler


def dino_coords(dino, img_path, labels):
    pil = Image.open(img_path).convert("RGB")
    det = dino.detect(pil, labels)
    return {lbl: [[int(c[0]), int(c[1])] for c in (det.get(lbl) or []) if len(c) == 2]
            for lbl in labels}


def molmo_counts(handler, img_path, g0, session_id):
    """Get Molmo detection counts for target and destination."""
    try:
        sys.path.insert(0, str(REPO / "src"))
        from monitoring.consistency_monitor import get_checkpoint_detections
        det = get_checkpoint_detections(img_path, g0, handler, session_id)
        tgt_n = len([c for c in (det.get(g0["target"]["label"]) or [])
                     if isinstance(c, (list, tuple)) and len(c) == 2])
        dst_n = len([c for c in (det.get(g0["destination"]["label"]) or [])
                     if isinstance(c, (list, tuple)) and len(c) == 2])
        return tgt_n, dst_n
    except Exception:
        return None, None


def load_model(ft_ckpt):
    os.environ["HF_HOME"] = HF_CACHE
    print("[Model] loading Molmo-7B...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    print(f"[Model] loading FT-C adapter: {ft_ckpt}")
    model = PeftModel.from_pretrained(base, ft_ckpt)
    model.eval()
    gen_cfg = GenerationConfig(
        do_sample=False, max_new_tokens=256, stop_strings="<|endoftext|>")
    print("[Model] ready")
    return model, processor, gen_cfg


def infer(model, processor, gen_cfg, img_path, prompt):
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": f"<image>\nTASK DESCRIPTION: {prompt}"}]
    inputs = process_msg_list(msgs, [img], processor)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    inputs["images"] = inputs["images"].to(model.dtype)
    inputs["image_masks"] = inputs["image_masks"].to(model.dtype)
    with torch.no_grad():
        out = model.generate_from_batch(
            inputs, gen_cfg, tokenizer=processor.tokenizer,
            return_dict_in_generate=True)
    tokens = out.sequences[0, inputs["input_ids"].size(1):]
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        d = json.loads(m.group() if m else text)
    except Exception:
        d = {"task_ambiguous": False, "stop_bool": False, "clarifying_question": text[:120]}
    ambig = bool(d.get("task_ambiguous", d.get("ambiguity_bool", False)))
    stop  = bool(d.get("stop_bool", False))
    cq    = str(d.get("clarifying_question", ""))
    if not ambig:
        return "CONTINUE", cq
    if stop or cq.upper().startswith("STOP:"):
        return "STOP", cq
    return "ASK", cq


def gold_decision(gs):
    if gs == "CLEAR":
        return "CONTINUE"
    if gs in ("INVALID_TARGET", "INVALID_DESTINATION", "UNSAFE_OR_BLOCKED"):
        return "STOP"
    return "ASK"


def compute_metrics(results):
    total = len(results)
    correct = sum(r["correct"] for r in results)
    non_cont = [r for r in results if r["gold_decision"] != "CONTINUE"]
    cont     = [r for r in results if r["gold_decision"] == "CONTINUE"]
    miss = sum(1 for r in non_cont if r["predicted_decision"] == "CONTINUE")
    fa   = sum(1 for r in cont    if r["predicted_decision"] != "CONTINUE")
    per  = defaultdict(lambda: [0, 0])
    for r in results:
        per[r["gold_state"]][1] += 1
        if r["correct"]:
            per[r["gold_state"]][0] += 1
    return {
        "decision_accuracy": round(correct / total, 4) if total else 0,
        "miss_rate":         round(miss / len(non_cont), 4) if non_cont else 0,
        "false_alarm_rate":  round(fa / len(cont), 4) if cont else 0,
        "correct": correct, "total": total,
        "per_state": {s: f"{v[0]}/{v[1]}" for s, v in sorted(per.items())},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FT-C with episode memory context injection")
    parser.add_argument(
        "--manifest", default="dataset/vla-evaluation-v4/manifest_eval-v4.jsonl")
    parser.add_argument("--ft-ckpt",  default="ckpt/ablation/FT-C/final")
    parser.add_argument("--out-csv",  default="logs/eval_ftc_memory.csv")
    parser.add_argument("--out-json", default="logs/eval_ftc_memory.json")
    args = parser.parse_args()

    V4_DIR = REPO / "dataset/vla-evaluation-v4"
    raw = [json.loads(l)
           for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    samples = []
    for e in raw:
        ck = e.get("checkpoint", "C1")
        samples.append({
            "id": e["id"], "scenario": e["scenario"], "task": e["task"],
            "target_label": e["target_label"],
            "destination_label": e["destination_label"],
            "gold_state": e["gold_state"],
            "gold_decision": e["gold_decision"],
            "checkpoint": ck,
            "t0_img": str(V4_DIR / e["initial_img"]),
            "ck_img": str(V4_DIR / (e["c1_img"] if ck == "C1" else e["c2_img"])),
        })
    print(f"Samples: {len(samples)}")
    print("Scenarios:", dict(Counter(s["scenario"] for s in samples)))

    # ── GDino precompute ─────────────────────────────────────────────────────
    dino = load_dino()
    print(f"\n[DINO] precomputing {len(samples)} samples...")
    for i, s in enumerate(samples):
        tgt, dst = s["target_label"], s["destination_label"]
        t0 = dino_coords(dino, s["t0_img"], [tgt, dst])
        ck = dino_coords(dino, s["ck_img"], [tgt, dst])
        s["g0t"] = t0[tgt][0] if t0[tgt] else None
        s["g0d"] = t0[dst][0] if t0[dst] else None
        s["ckt"] = ck[tgt]
        s["ckd"] = ck[dst]
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} done")
    del dino
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("[DINO done, VRAM cleared]")

    # ── Molmo precompute (for memory validation) ──────────────────────────────
    handler = load_handler()
    print(f"\n[Molmo] precomputing detection counts for {len(samples)} samples...")
    for i, s in enumerate(samples):
        tgt, dst = s["target_label"], s["destination_label"]
        g0 = {
            "target":      {"label": tgt, "coord": s["g0t"], "coords": [s["g0t"]] if s["g0t"] else []},
            "destination": {"label": dst, "coord": s["g0d"], "coords": [s["g0d"]] if s["g0d"] else []},
        }
        tgt_n, dst_n = molmo_counts(handler, s["ck_img"], g0, f"molmo_val_{s['id']}")
        s["molmo_tgt_n"] = tgt_n
        s["molmo_dst_n"] = dst_n
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} done")
    del handler
    gc.collect(); torch.cuda.empty_cache()
    print("[Molmo done, VRAM cleared]\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, processor, gen_cfg = load_model(args.ft_ckpt)

    results = []
    for i, s in enumerate(samples):
        tgt, dst = s["target_label"], s["destination_label"]

        # Molmo-validated coords for prompt:
        # If Molmo says 0 but GDino says 1 → force-match detected
        # → replace GDino coords with empty list so prompt count matches memory "disappeared"
        ckt_prompt = s["ckt"]
        ckd_prompt = s["ckd"]
        if s.get("molmo_tgt_n") == 0 and len(s["ckt"]) > 0:
            ckt_prompt = []  # trust Molmo absence over GDino force-match
        if s.get("molmo_dst_n") == 0 and len(s["ckd"]) > 0:
            ckd_prompt = []

        memory_ctx = build_memory_context(
            tgt, dst, s["g0t"], s["g0d"], ckt_prompt, ckd_prompt,
            molmo_tgt_count=s.get("molmo_tgt_n"),
            molmo_dst_count=s.get("molmo_dst_n"),
        )
        prompt = build_prompt(
            s["task"], tgt, dst, s["g0t"], s["g0d"],
            ckt_prompt, ckd_prompt, s["checkpoint"], memory_ctx)
        pred, cq = infer(model, processor, gen_cfg, s["ck_img"], prompt)
        gold = gold_decision(s["gold_state"])
        ok = pred == gold
        results.append({
            "id": s["id"], "scenario": s["scenario"],
            "gold_state": s["gold_state"], "gold_decision": gold,
            "predicted_decision": pred, "correct": ok,
            "clarifying_question": cq,
        })
        mark = "OK" if ok else "FAIL"
        print(f"  [{i+1:2d}/{len(samples)}] {mark} {s['id']:<28} "
              f"gold={gold:<8} pred={pred}")

    metrics = {"FT-C + memory context": compute_metrics(results)}
    print("\n" + json.dumps(metrics, indent=2))

    out_json = REPO / args.out_json
    out_csv  = REPO / args.out_csv
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out_csv}\n       {out_json}")


if __name__ == "__main__":
    main()
