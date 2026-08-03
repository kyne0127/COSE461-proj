#!/usr/bin/env python3
"""
Fine-tuning dataset builder for change-aware grounding decisions.

AmbRes output format 유지:
  ambiguity_bool: False → CONTINUE
  ambiguity_bool: True  → ASK (clarifying_question) or STOP ("STOP: ...")

Augmentation pipeline:
  Step 1: 두 데이터셋 합치기 (ambres-training + vla-evaluation)
  Step 2: C1 + C2 모두 사용 (object_states로 자동 label 생성)
  Step 3: H-flip + brightness augmentation
  Step 4: 탐지 좌표 perturbation (±5px Gaussian noise)

Usage:
    python3 src/finetune/build_dataset.py \
        --out-dir dataset/finetune \
        --aug-flip --aug-brightness --aug-coord-noise \
        --dino-threshold 0.25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
_REPO = _SRC.parent
for p in (_SRC, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy & prompt template
# ─────────────────────────────────────────────────────────────────────────────

_TAXONOMY_TEXT = """\
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


def build_memory_context(
    target_label: str,
    dest_label: str,
    g0_target_coord: list[int] | None,
    g0_dest_coord: list[int] | None,
    object_states: dict,
    checkpoint: str,
) -> str:
    """Build episode memory context from object_states metadata.

    Uses gold object_states (not GDino detections) to generate accurate
    change records — avoids force-matching noise at training time.
    """
    t0 = object_states.get("t0", {})
    ck_key = checkpoint.lower()  # "c1" or "c2"
    ck = object_states.get(ck_key, {})

    t0_tgt = t0.get("target", 1)
    t0_dst = t0.get("destination", 1)
    ck_tgt = ck.get("target", t0_tgt)
    ck_dst = ck.get("destination", t0_dst)
    note   = ck.get("note", "")

    changes = []

    # Target changes
    if ck_tgt == 0:
        changes.append(f"target '{target_label}' disappeared")
    elif ck_tgt > t0_tgt:
        changes.append(f"target '{target_label}' count increased {t0_tgt} → {ck_tgt}")
    elif "moved" in note.lower() and checkpoint == "C1":
        changes.append(f"target '{target_label}' moved to a different position")

    # Destination changes
    if ck_dst == 0:
        changes.append(f"destination '{dest_label}' disappeared")
    elif ck_dst > t0_dst:
        changes.append(f"destination '{dest_label}' count increased {t0_dst} → {ck_dst}")
    elif "moved" in note.lower() and checkpoint == "C2":
        changes.append(f"destination '{dest_label}' moved to a different position")

    if not changes:
        return ""

    lines = [
        "[Episode Memory — prior observations]",
        "[t0]",
        f"  target '{target_label}': detected at {g0_target_coord or 'unknown'}",
        f"  destination '{dest_label}': detected at {g0_dest_coord or 'unknown'}",
        f"[Now evaluating: {checkpoint}]",
        "[Scene changes detected: " + "; ".join(changes) + "]",
    ]
    return "\n".join(lines)


def build_prompt(
    task: str,
    target_label: str,
    dest_label: str,
    g0_target_coord: list[int] | None,
    g0_dest_coord: list[int] | None,
    ck_target_coords: list[list[int]],
    ck_dest_coords: list[list[int]],
    checkpoint: str,
    memory_context: str = "",
) -> str:
    def _fmt(coords):
        if not coords:
            return "none"
        return ", ".join(f"({c[0]},{c[1]})" for c in coords[:5])

    g0_t = f"({g0_target_coord[0]},{g0_target_coord[1]})" if g0_target_coord else "unknown"
    g0_d = f"({g0_dest_coord[0]},{g0_dest_coord[1]})"   if g0_dest_coord   else "unknown"

    n_t = len(ck_target_coords)
    n_d = len(ck_dest_coords)

    parts = [
        f"Original robot task: {task}",
        "",
    ]
    if memory_context:
        parts += [memory_context, ""]

    parts += [
        "Initial scene (t0) — stored reference positions:",
        f'  - Target "{target_label}" was at pixel {g0_t}',
        f'  - Destination "{dest_label}" was at pixel {g0_d}',
        "",
        f"Checkpoint {checkpoint} — GroundingDINO detection results (authoritative):",
        f'  - "{target_label}": {n_t} detection(s) at [{_fmt(ck_target_coords)}]',
        f'  - "{dest_label}": {n_d} detection(s) at [{_fmt(ck_dest_coords)}]',
        "",
        "Based on the detection results and scene history above, assess whether the "
        "task grounding is valid.",
        "",
        _TAXONOMY_TEXT,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Gold label → AmbRes output mapping
# ─────────────────────────────────────────────────────────────────────────────

_STOP_STATES = {"INVALID_TARGET", "INVALID_DESTINATION", "UNSAFE_OR_BLOCKED"}


def _clarifying_question(
    gold_state: str,
    target_label: str,
    dest_label: str,
    ck_tgt_coords: list[list[int]],
    ck_dst_coords: list[list[int]],
) -> str:
    """좌표 기반으로 상황을 세분화해 적절한 질문/메시지를 반환."""
    if gold_state == "CLEAR":
        return ""
    if gold_state == "INVALID_TARGET":
        return f"The {target_label} cannot be found in the scene."
    if gold_state == "INVALID_DESTINATION":
        return f"The {dest_label} cannot be found in the scene."
    if gold_state == "UNSAFE_OR_BLOCKED":
        return ""
    if gold_state == "AMBIGUOUS_TARGET":
        if len(ck_tgt_coords) > 1:
            return f"Which {target_label} would you like me to pick?"
        # count==1이지만 위치가 이동한 경우 (S6 등)
        return f"The {target_label} has moved. Continue with the moved target?"
    if gold_state == "AMBIGUOUS_DESTINATION":
        if len(ck_dst_coords) == 0:
            return f"I cannot locate the {dest_label}. Where should I place it?"
        if len(ck_dst_coords) > 1:
            return f"Which {dest_label} should I place it on?"
        # count==1이지만 위치가 이동한 경우 (S7 등)
        return f"The {dest_label} has moved. Continue with the moved destination?"
    return ""


def gold_to_ambres_output(
    gold_state: str,
    target_label: str,
    dest_label: str,
    ck_tgt_coords: list[list[int]],
    ck_dst_coords: list[list[int]],
) -> dict[str, Any]:
    """Map gold_state + detection coords → training output fields.

    Fields:
      ambiguity_bool:      True if robot should NOT continue as-is
      stop_bool:           True for STOP decisions
      clarifying_question: 상황별 질문/메시지; CONTINUE는 빈 문자열
    """
    return {
        "ambiguity_bool": gold_state != "CLEAR",
        "stop_bool":      gold_state in _STOP_STATES,
        "clarifying_question": _clarifying_question(
            gold_state, target_label, dest_label,
            ck_tgt_coords, ck_dst_coords,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DINO detection helper
# ─────────────────────────────────────────────────────────────────────────────

def run_dino(gdino, image_path: str | Path, labels: list[str]) -> dict[str, list]:
    from PIL import Image
    pil = Image.open(image_path).convert("RGB")
    return gdino.detect(pil, labels)


def _coords_from_manual(ann: dict, key: str) -> list[list[int]]:
    """manual annotation dict에서 좌표 리스트 추출."""
    raw = ann.get(key) or []
    return [[int(c[0]), int(c[1])] for c in raw if len(c) == 2]


# ─────────────────────────────────────────────────────────────────────────────
# Gold state inference from object_states (when not explicitly labeled)
# ─────────────────────────────────────────────────────────────────────────────

def infer_gold_state_from_counts(
    t0_target: int,
    t0_dest: int,
    ck_target: int,
    ck_dest: int,
    checkpoint: str,  # "C1" checks target, "C2" checks destination
) -> str:
    """Infer gold state from object counts at t0 and checkpoint.

    Conservative heuristic based on count changes.
    Not perfect — position data would be needed for AMBIGUOUS vs CLEAR
    when count is stable but object moved. Use only for augmentation.
    """
    if checkpoint == "C1":
        if ck_target == 0:
            return "INVALID_TARGET"
        if ck_target > t0_target:
            return "AMBIGUOUS_TARGET"
        if ck_target == t0_target:
            return "CLEAR"
        # count decreased but > 0: likely moved
        return "AMBIGUOUS_TARGET"
    else:  # C2
        if ck_dest == 0:
            return "INVALID_DESTINATION"
        if ck_dest > t0_dest:
            return "AMBIGUOUS_DESTINATION"
        return "CLEAR"


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def flip_coords(coords: list[list[int]], img_w: int) -> list[list[int]]:
    return [[img_w - 1 - c[0], c[1]] for c in coords]


def perturb_coords(
    coords: list[list[int]],
    sigma: float = 5.0,
    rng: random.Random | None = None,
) -> list[list[int]]:
    if rng is None:
        rng = random.Random()
    return [
        [int(c[0] + rng.gauss(0, sigma)), int(c[1] + rng.gauss(0, sigma))]
        for c in coords
    ]


def augment_image_brightness(image_path: str | Path, factor: float, out_path: str | Path) -> None:
    from PIL import Image, ImageEnhance
    img = Image.open(image_path).convert("RGB")
    enhancer = ImageEnhance.Brightness(img)
    enhancer.enhance(factor).save(out_path)


def flip_image(image_path: str | Path, out_path: str | Path) -> None:
    from PIL import Image
    Image.open(image_path).convert("RGB").transpose(
        Image.FLIP_LEFT_RIGHT
    ).save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Core: build one training example
# ─────────────────────────────────────────────────────────────────────────────

def build_example(
    sample_id: str,
    task: str,
    target_label: str,
    dest_label: str,
    g0_target_coord: list[int] | None,
    g0_dest_coord: list[int] | None,
    ck_image_path: str,
    ck_target_coords: list[list[int]],
    ck_dest_coords: list[list[int]],
    checkpoint: str,
    gold_state: str,
    object_states: dict | None = None,
) -> dict[str, Any]:
    memory_context = ""
    if object_states:
        memory_context = build_memory_context(
            target_label, dest_label,
            g0_target_coord, g0_dest_coord,
            object_states, checkpoint,
        )
    prompt = build_prompt(
        task, target_label, dest_label,
        g0_target_coord, g0_dest_coord,
        ck_target_coords, ck_dest_coords,
        checkpoint,
        memory_context=memory_context,
    )
    output = gold_to_ambres_output(
        gold_state, target_label, dest_label,
        ck_target_coords, ck_dest_coords,
    )
    return {
        "id": sample_id,
        "image": ck_image_path,
        "task_description": prompt,
        "ambiguity_bool": output["ambiguity_bool"],
        "stop_bool": output["stop_bool"],
        "clarifying_question": output["clarifying_question"],
        "gold_state": gold_state,
        "checkpoint": checkpoint,
        "target_label": target_label,
        "destination_label": dest_label,
        "meta": {
            "g0_target_coord": g0_target_coord,
            "g0_dest_coord": g0_dest_coord,
            "ck_target_coords": ck_target_coords,
            "ck_dest_coords": ck_dest_coords,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    manifests: list[Path],
    out_dir: Path,
    gdino,
    *,
    aug_flip: bool = False,
    aug_brightness: bool = False,
    aug_coord_noise: bool = False,
    aug_both_checkpoints: bool = True,
    seed: int = 42,
    resume: bool = False,
    manual_annotations: dict | None = None,
) -> None:
    """manual_annotations: {trial_id: {target: [[x,y],...], dest: [[x,y],...]}}
    checkpoint detection에서 DINO 대신 사용. 없는 trial은 DINO fallback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)
    rng = random.Random(seed)

    examples: list[dict[str, Any]] = []
    skipped = 0

    for manifest_path in manifests:
        base_dir = manifest_path.parent
        samples = [
            json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()
        ]
        total = len(samples)
        print(f"  Loaded {total} samples from {manifest_path.name}")

        for i, sample in enumerate(samples, 1):
            def resolve(p):
                pp = Path(p)
                return str(pp if pp.is_absolute() else base_dir / pp)

            t0_img  = resolve(sample["initial_img"])
            c1_img  = resolve(sample["c1_img"])
            c2_img  = resolve(sample["c2_img"])
            task    = sample["task"]
            tgt     = sample["target_label"]
            dst     = sample["destination_label"]
            sid     = sample["id"]

            cache_file = cache_dir / f"{sid}.json"
            if resume and cache_file.exists():
                cached = json.loads(cache_file.read_text())
                examples.extend(cached)
                skipped += 1
                print(f"    [{i}/{total}] {sid} (cached, skip)")
                continue

            # ── t0 G0 좌표 ──────────────────────────────────────────────
            print(f"    [{i}/{total}] {sid}", end=" ", flush=True)
            if gdino is not None:
                print("DINO...", end=" ", flush=True)
                t0_det = run_dino(gdino, t0_img, [tgt, dst])
                t0_tgt_coords = [c for c in (t0_det.get(tgt) or []) if len(c) == 2]
                t0_dst_coords = [c for c in (t0_det.get(dst) or []) if len(c) == 2]
                g0_target = [int(t0_tgt_coords[0][0]), int(t0_tgt_coords[0][1])] if t0_tgt_coords else None
                g0_dest   = [int(t0_dst_coords[0][0]), int(t0_dst_coords[0][1])] if t0_dst_coords else None
            else:
                g0_target = None
                g0_dest   = None

            # ── Checkpoints to process ───────────────────────────────────
            checkpoints_to_use = [
                (sample["checkpoint"], sample["gold_state"], sample["gold_decision"])
            ]
            if aug_both_checkpoints and "object_states" in sample:
                other_ck = "C2" if sample["checkpoint"] == "C1" else "C1"
                os = sample["object_states"]
                gold_state_other = infer_gold_state_from_counts(
                    t0_target=os["t0"].get("target", 0),
                    t0_dest=  os["t0"].get("destination", 0),
                    ck_target=os.get(other_ck.lower(), {}).get("target", 0),
                    ck_dest=  os.get(other_ck.lower(), {}).get("destination", 0),
                    checkpoint=other_ck,
                )
                checkpoints_to_use.append((other_ck, gold_state_other, None))

            sample_examples: list[dict[str, Any]] = []
            obj_states = sample.get("object_states")  # gold change info

            for ck, gold_state, _ in checkpoints_to_use:
                ck_img = c1_img if ck == "C1" else c2_img
                # manual annotation 우선 사용, 없으면 DINO fallback
                if manual_annotations and sid in manual_annotations:
                    ann = manual_annotations[sid]
                    ck_tgt = _coords_from_manual(ann, "target")
                    ck_dst = _coords_from_manual(ann, "dest")
                else:
                    ck_det = run_dino(gdino, ck_img, [tgt, dst])
                    ck_tgt = [[int(c[0]), int(c[1])] for c in (ck_det.get(tgt) or []) if len(c) == 2]
                    ck_dst = [[int(c[0]), int(c[1])] for c in (ck_det.get(dst) or []) if len(c) == 2]

                # ── Base example ─────────────────────────────────────────
                ex = build_example(
                    f"{sid}_{ck}", task, tgt, dst,
                    g0_target, g0_dest, ck_img,
                    ck_tgt, ck_dst, ck, gold_state,
                    object_states=obj_states,
                )
                sample_examples.append(ex)
                print(f"{ck}:{gold_state}", end=" ", flush=True)

                # ── Coord noise augmentation ─────────────────────────────
                if aug_coord_noise:
                    for ni in range(2):
                        ck_tgt_n = perturb_coords(ck_tgt, sigma=5.0, rng=rng)
                        ck_dst_n = perturb_coords(ck_dst, sigma=5.0, rng=rng)
                        g0_tgt_n = perturb_coords([g0_target], sigma=3.0, rng=rng)[0] if g0_target else None
                        g0_dst_n = perturb_coords([g0_dest],   sigma=3.0, rng=rng)[0] if g0_dest   else None
                        ex_n = build_example(
                            f"{sid}_{ck}_noise{ni}", task, tgt, dst,
                            g0_tgt_n, g0_dst_n, ck_img,
                            ck_tgt_n, ck_dst_n, ck, gold_state,
                            object_states=obj_states,
                        )
                        sample_examples.append(ex_n)

                # ── H-flip augmentation ──────────────────────────────────
                if aug_flip:
                    from PIL import Image
                    img_w = Image.open(ck_img).width
                    flip_path = img_dir / f"{sid}_{ck}_flip.jpg"
                    flip_image(ck_img, flip_path)

                    ck_tgt_f = flip_coords(ck_tgt, img_w)
                    ck_dst_f = flip_coords(ck_dst, img_w)
                    g0_tgt_f = flip_coords([g0_target], img_w)[0] if g0_target else None
                    g0_dst_f = flip_coords([g0_dest],   img_w)[0] if g0_dest   else None

                    ex_f = build_example(
                        f"{sid}_{ck}_flip", task, tgt, dst,
                        g0_tgt_f, g0_dst_f, str(flip_path),
                        ck_tgt_f, ck_dst_f, ck, gold_state,
                        object_states=obj_states,
                    )
                    sample_examples.append(ex_f)

                # ── Brightness augmentation ──────────────────────────────
                if aug_brightness:
                    for factor, suffix in [(0.7, "dark"), (1.3, "bright")]:
                        bright_path = img_dir / f"{sid}_{ck}_{suffix}.jpg"
                        augment_image_brightness(ck_img, factor, bright_path)
                        ex_b = build_example(
                            f"{sid}_{ck}_{suffix}", task, tgt, dst,
                            g0_target, g0_dest, str(bright_path),
                            ck_tgt, ck_dst, ck, gold_state,
                            object_states=obj_states,
                        )
                        sample_examples.append(ex_b)

            # ── 샘플 캐시 저장 (재시작 시 skip용) ────────────────────────
            cache_file.write_text(json.dumps(sample_examples, ensure_ascii=False))
            examples.extend(sample_examples)
            print()

    if skipped:
        print(f"\n  (캐시에서 {skipped}개 샘플 skip)")

    # ── Train/val split (90/10) ──────────────────────────────────────────────
    rng.shuffle(examples)
    split = int(len(examples) * 0.9)
    train_ex = examples[:split]
    val_ex   = examples[split:]

    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    with open(train_path, "w") as f:
        for ex in train_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for ex in val_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n총 {len(examples)}개 examples:")
    print(f"  train: {len(train_ex)} → {train_path}")
    print(f"  val:   {len(val_ex)}  → {val_path}")

    # ── Label distribution ───────────────────────────────────────────────────
    from collections import Counter
    dist = Counter(ex["gold_state"] for ex in examples)
    print("\nGold state distribution:")
    for state, cnt in sorted(dist.items()):
        print(f"  {state:<30} {cnt}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset for change-aware grounding")
    parser.add_argument(
        "--manifests", nargs="+",
        default=[
            "dataset/ambres-training/manifest_train_local.jsonl",
            "dataset/vla-evaluation/dataset/manifest.jsonl",
        ],
        help="Manifest JSONL files to use",
    )
    parser.add_argument("--out-dir", default="dataset/finetune")
    parser.add_argument("--aug-flip",          action="store_true")
    parser.add_argument("--aug-brightness",    action="store_true")
    parser.add_argument("--aug-coord-noise",   action="store_true")
    parser.add_argument("--no-both-checkpoints", action="store_true",
                        help="Only use annotated checkpoint, skip other")
    parser.add_argument("--dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-cached samples (resume interrupted run)")
    parser.add_argument("--manual-annotations", default=None,
                        help="JSON file from manual_annotate.py; overrides DINO for checkpoint coords")
    parser.add_argument("--no-dino", action="store_true",
                        help="Skip DINO loading entirely (requires --manual-annotations for all trials; G0 coords will be None)")
    args = parser.parse_args()

    # ── manual annotations 로드 ──────────────────────────────────────────────
    manual_annotations: dict | None = None
    if args.manual_annotations:
        p = Path(args.manual_annotations)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / p
        if p.exists():
            manual_annotations = json.loads(p.read_text())
            print(f"[Manual] {len(manual_annotations)}개 trial annotation 로드: {p}")
        else:
            print(f"[WARN] manual annotations 파일 없음: {p}")

    # ── DINO 로드 (--no-dino 없으면) ─────────────────────────────────────────
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    gdino = None
    if not args.no_dino:
        from module.models.ambres.gdino import GroundingDINO
        print("[DINO] 로딩 중...")
        gdino = GroundingDINO(
            box_threshold=args.dino_box_threshold,
            text_threshold=args.dino_text_threshold,
        )
        print("[DINO] 준비 완료")
    else:
        print("[DINO] skip (--no-dino)")

    manifests = [Path(p) for p in args.manifests]
    for m in manifests:
        if not m.exists():
            print(f"[WARN] manifest not found: {m}")

    manifests = [m for m in manifests if m.exists()]
    print(f"\n{len(manifests)}개 manifest 사용: {[m.name for m in manifests]}")

    build_dataset(
        manifests=manifests,
        out_dir=Path(args.out_dir),
        gdino=gdino,
        aug_flip=args.aug_flip,
        aug_brightness=args.aug_brightness,
        aug_coord_noise=args.aug_coord_noise,
        aug_both_checkpoints=not args.no_both_checkpoints,
        seed=args.seed,
        resume=args.resume,
        manual_annotations=manual_annotations,
    )


if __name__ == "__main__":
    main()
