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


def build_prompt(
    task: str,
    target_label: str,
    dest_label: str,
    g0_target_coord: list[int] | None,
    g0_dest_coord: list[int] | None,
    ck_target_coords: list[list[int]],
    ck_dest_coords: list[list[int]],
    checkpoint: str,
) -> str:
    def _fmt(coords):
        if not coords:
            return "none"
        return ", ".join(f"({c[0]},{c[1]})" for c in coords[:5])

    g0_t = f"({g0_target_coord[0]},{g0_target_coord[1]})" if g0_target_coord else "unknown"
    g0_d = f"({g0_dest_coord[0]},{g0_dest_coord[1]})"   if g0_dest_coord   else "unknown"

    n_t = len(ck_target_coords)
    n_d = len(ck_dest_coords)

    return "\n".join([
        f"Original robot task: {task}",
        "",
        "Initial scene (t0) — stored reference positions:",
        f'  - Target "{target_label}" was at pixel {g0_t}',
        f'  - Destination "{dest_label}" was at pixel {g0_d}',
        "",
        f"Checkpoint {checkpoint} — GroundingDINO detection results (authoritative):",
        f'  - "{target_label}": {n_t} detection(s) at [{_fmt(ck_target_coords)}]',
        f'  - "{dest_label}": {n_d} detection(s) at [{_fmt(ck_dest_coords)}]',
        "",
        "Based ONLY on the detection counts and positions above, assess whether the "
        "task grounding is valid.",
        "",
        _TAXONOMY_TEXT,
    ])


def build_prompt_no_g0(
    task: str,
    target_label: str,
    dest_label: str,
    ck_target_coords: list[list[int]],
    ck_dest_coords: list[list[int]],
    checkpoint: str,
) -> str:
    """G₀ reference 없이 count만으로 판단하는 프롬프트."""
    def _fmt(coords):
        if not coords:
            return "none"
        return ", ".join(f"({c[0]},{c[1]})" for c in coords[:5])

    n_t = len(ck_target_coords)
    n_d = len(ck_dest_coords)

    return "\n".join([
        f"Original robot task: {task}",
        "",
        "Initial scene (t0) — stored reference positions:",
        f'  - Target "{target_label}": not available (G₀ not recorded)',
        f'  - Destination "{dest_label}": not available (G₀ not recorded)',
        "",
        f"Checkpoint {checkpoint} — GroundingDINO detection results (authoritative):",
        f'  - "{target_label}": {n_t} detection(s) at [{_fmt(ck_target_coords)}]',
        f'  - "{dest_label}": {n_d} detection(s) at [{_fmt(ck_dest_coords)}]',
        "",
        "G₀ reference positions are unavailable. Assess validity based solely on "
        "detection counts.",
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

def gold_state_count_rule(
    checkpoint: str,
    ck_tgt_count: int,
    ck_dst_count: int,
) -> str:
    """G₀ 없을 때 count만으로 gold state 결정.

    count=0 → STOP (object absent)
    count=1 → CLEAR → CONTINUE (위치 비교 불가, 보수적 CONTINUE)
    count>1 → ASK (multiple candidates)

    Note: S4/S7 (moved object, count=1) → CLEAR here.
    Without G₀ we cannot detect displacement — this is intentional.
    """
    if checkpoint == "C1":
        if ck_tgt_count == 0:
            return "INVALID_TARGET"
        if ck_tgt_count > 1:
            return "AMBIGUOUS_TARGET"
        return "CLEAR"
    else:  # C2
        if ck_dst_count == 0:
            return "INVALID_DESTINATION"
        if ck_dst_count > 1:
            return "AMBIGUOUS_DESTINATION"
        return "CLEAR"


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
) -> dict[str, Any]:
    prompt = build_prompt(
        task, target_label, dest_label,
        g0_target_coord, g0_dest_coord,
        ck_target_coords, ck_dest_coords,
        checkpoint,
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


def build_example_no_g0(
    sample_id: str,
    task: str,
    target_label: str,
    dest_label: str,
    ck_image_path: str,
    ck_target_coords: list[list[int]],
    ck_dest_coords: list[list[int]],
    checkpoint: str,
) -> dict[str, Any]:
    """G₀ 없는 학습 샘플. gold_state는 count rule로 자동 결정."""
    gold_state = gold_state_count_rule(
        checkpoint, len(ck_target_coords), len(ck_dest_coords)
    )
    prompt = build_prompt_no_g0(
        task, target_label, dest_label,
        ck_target_coords, ck_dest_coords,
        checkpoint,
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
            "g0_target_coord": None,
            "g0_dest_coord": None,
            "ck_target_coords": ck_target_coords,
            "ck_dest_coords": ck_dest_coords,
            "no_g0": True,
        },
    }


def build_counterfactual_examples(
    base_sid: str,
    task: str,
    target_label: str,
    dest_label: str,
    g0_target_coord: list[int] | None,
    g0_dest_coord: list[int] | None,
    ck_image_path: str,
    ck_target_coords: list[list[int]],
    ck_dest_coords: list[list[int]],
    checkpoint: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """같은 이미지·같은 ck coords에서 G₀만 바꾼 반례 2개를 생성.

    C1, count=1:
      cf_same    : G₀_target = ck_target  →  CLEAR
      cf_shifted : G₀_target += large_offset  →  AMBIGUOUS_TARGET

    C2, count=1 (dest):
      cf_same    : G₀_dest = ck_dest  →  CLEAR
      cf_shifted : G₀_dest += large_offset  →  AMBIGUOUS_DESTINATION

    모델이 count=1 shortcut이 아니라 G₀ 좌표 비교를 배우도록 강제.
    """
    examples = []
    shifts = [(-200, 0), (200, 0), (0, -150), (0, 150)]

    if checkpoint == "C1" and len(ck_target_coords) == 1:
        cx, cy = ck_target_coords[0]
        shift = rng.choice(shifts)

        # Variant A: G₀ = current position → CLEAR
        ex_same = build_example(
            f"{base_sid}_cf_same", task, target_label, dest_label,
            [cx, cy], g0_dest_coord,
            ck_image_path, ck_target_coords, ck_dest_coords,
            checkpoint, "CLEAR",
        )
        ex_same["meta"]["counterfactual"] = "same"
        examples.append(ex_same)

        # Variant B: G₀ = far away → AMBIGUOUS_TARGET (moved)
        g0_shifted = [cx + shift[0], cy + shift[1]]
        ex_shifted = build_example(
            f"{base_sid}_cf_shifted", task, target_label, dest_label,
            g0_shifted, g0_dest_coord,
            ck_image_path, ck_target_coords, ck_dest_coords,
            checkpoint, "AMBIGUOUS_TARGET",
        )
        ex_shifted["meta"]["counterfactual"] = "shifted"
        examples.append(ex_shifted)

    elif checkpoint == "C2" and len(ck_dest_coords) == 1:
        cx, cy = ck_dest_coords[0]
        shift = rng.choice(shifts)

        # Variant A: G₀_dest = current position → CLEAR
        ex_same = build_example(
            f"{base_sid}_cf_same", task, target_label, dest_label,
            g0_target_coord, [cx, cy],
            ck_image_path, ck_target_coords, ck_dest_coords,
            checkpoint, "CLEAR",
        )
        ex_same["meta"]["counterfactual"] = "same"
        examples.append(ex_same)

        # Variant B: G₀_dest = far away → AMBIGUOUS_DESTINATION (moved)
        g0_shifted = [cx + shift[0], cy + shift[1]]
        ex_shifted = build_example(
            f"{base_sid}_cf_shifted", task, target_label, dest_label,
            g0_target_coord, g0_shifted,
            ck_image_path, ck_target_coords, ck_dest_coords,
            checkpoint, "AMBIGUOUS_DESTINATION",
        )
        ex_shifted["meta"]["counterfactual"] = "shifted"
        examples.append(ex_shifted)

    return examples


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
    no_g0_augment: bool = False,
    counterfactual: bool = False,
    train_ratio: float = 0.8,
    seed: int = 42,
    manual_annotations: dict | None = None,
) -> None:
    """trial 단위로 train/val split 후 train set에만 augmentation 적용.

    - train: base + aug(coord_noise, flip, brightness) + no_g0 + counterfactual
    - val:   base only (augmentation 없음, data leakage 방지)

    manual_annotations: {trial_id: {target: [[x,y],...], dest: [[x,y],...]}}
    counterfactual: 같은 이미지에서 G₀만 바꾼 반례 2개 추가 (G₀ 비교 학습 강제)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    rng = random.Random(seed)

    # ── 1. 전체 샘플 로드 ────────────────────────────────────────────────────
    all_samples: list[tuple[Path, dict]] = []
    for manifest_path in manifests:
        base_dir = manifest_path.parent
        rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
        print(f"  Loaded {len(rows)} samples from {manifest_path.name}")
        for s in rows:
            all_samples.append((base_dir, s))

    # ── 2. Trial 단위 train/val split ────────────────────────────────────────
    trial_ids = [s["id"] for _, s in all_samples]
    shuffled = trial_ids[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    train_trial_ids = set(shuffled[:n_train])
    val_trial_ids   = set(shuffled[n_train:])
    print(f"\n  Trial split: train={len(train_trial_ids)}, val={len(val_trial_ids)}")

    train_examples: list[dict[str, Any]] = []
    val_examples:   list[dict[str, Any]] = []
    total = len(all_samples)

    # ── 3. 샘플 처리 ─────────────────────────────────────────────────────────
    for i, (base_dir, sample) in enumerate(all_samples, 1):
        def resolve(p):
            pp = Path(p)
            return str(pp if pp.is_absolute() else base_dir / pp)

        t0_img = resolve(sample["initial_img"])
        c1_img = resolve(sample["c1_img"])
        c2_img = resolve(sample["c2_img"])
        task   = sample["task"]
        tgt    = sample["target_label"]
        dst    = sample["destination_label"]
        sid    = sample["id"]
        is_train = sid in train_trial_ids

        print(f"  [{i}/{total}] {sid} ({'train' if is_train else 'val'})", end=" ", flush=True)

        # ── G₀ 좌표 (t0 DINO) ────────────────────────────────────────────
        if gdino is not None:
            print("DINO...", end=" ", flush=True)
            t0_det = run_dino(gdino, t0_img, [tgt, dst])
            t0_tgt = [c for c in (t0_det.get(tgt) or []) if len(c) == 2]
            t0_dst = [c for c in (t0_det.get(dst) or []) if len(c) == 2]
            g0_target = [int(t0_tgt[0][0]), int(t0_tgt[0][1])] if t0_tgt else None
            g0_dest   = [int(t0_dst[0][0]), int(t0_dst[0][1])] if t0_dst else None
        else:
            g0_target = None
            g0_dest   = None

        # ── Primary checkpoint ───────────────────────────────────────────
        ck         = sample["checkpoint"]
        gold_state = sample["gold_state"]
        ck_img     = c1_img if ck == "C1" else c2_img

        if manual_annotations and sid in manual_annotations:
            ann    = manual_annotations[sid]
            ck_tgt = _coords_from_manual(ann, "target")
            ck_dst = _coords_from_manual(ann, "dest")
        else:
            ck_det = run_dino(gdino, ck_img, [tgt, dst])
            ck_tgt = [[int(c[0]), int(c[1])] for c in (ck_det.get(tgt) or []) if len(c) == 2]
            ck_dst = [[int(c[0]), int(c[1])] for c in (ck_det.get(dst) or []) if len(c) == 2]

        # ── Base example ─────────────────────────────────────────────────
        base_ex = build_example(
            f"{sid}_{ck}", task, tgt, dst,
            g0_target, g0_dest, ck_img,
            ck_tgt, ck_dst, ck, gold_state,
        )
        print(f"{ck}:{gold_state}", end=" ", flush=True)

        if is_train:
            train_examples.append(base_ex)

            # ── Coord noise ──────────────────────────────────────────────
            if aug_coord_noise:
                for ni in range(2):
                    ck_tgt_n = perturb_coords(ck_tgt, sigma=5.0, rng=rng)
                    ck_dst_n = perturb_coords(ck_dst, sigma=5.0, rng=rng)
                    g0_tgt_n = perturb_coords([g0_target], sigma=3.0, rng=rng)[0] if g0_target else None
                    g0_dst_n = perturb_coords([g0_dest],   sigma=3.0, rng=rng)[0] if g0_dest   else None
                    train_examples.append(build_example(
                        f"{sid}_{ck}_noise{ni}", task, tgt, dst,
                        g0_tgt_n, g0_dst_n, ck_img,
                        ck_tgt_n, ck_dst_n, ck, gold_state,
                    ))

            # ── H-flip ───────────────────────────────────────────────────
            if aug_flip:
                from PIL import Image
                img_w = Image.open(ck_img).width
                flip_path = img_dir / f"{sid}_{ck}_flip.jpg"
                flip_image(ck_img, flip_path)
                ck_tgt_f = flip_coords(ck_tgt, img_w)
                ck_dst_f = flip_coords(ck_dst, img_w)
                g0_tgt_f = flip_coords([g0_target], img_w)[0] if g0_target else None
                g0_dst_f = flip_coords([g0_dest],   img_w)[0] if g0_dest   else None
                train_examples.append(build_example(
                    f"{sid}_{ck}_flip", task, tgt, dst,
                    g0_tgt_f, g0_dst_f, str(flip_path),
                    ck_tgt_f, ck_dst_f, ck, gold_state,
                ))

            # ── Brightness ───────────────────────────────────────────────
            if aug_brightness:
                for factor, suffix in [(0.7, "dark"), (1.3, "bright")]:
                    bright_path = img_dir / f"{sid}_{ck}_{suffix}.jpg"
                    augment_image_brightness(ck_img, factor, bright_path)
                    train_examples.append(build_example(
                        f"{sid}_{ck}_{suffix}", task, tgt, dst,
                        g0_target, g0_dest, str(bright_path),
                        ck_tgt, ck_dst, ck, gold_state,
                    ))

            # ── G₀-absent ────────────────────────────────────────────────
            if no_g0_augment:
                ex_ng = build_example_no_g0(
                    f"{sid}_{ck}_nog0", task, tgt, dst,
                    ck_img, ck_tgt, ck_dst, ck,
                )
                train_examples.append(ex_ng)
                print(f"nog0:{ex_ng['gold_state']}", end=" ", flush=True)

            # ── Counterfactual G₀ pairs ───────────────────────────────────
            if counterfactual:
                cf_exs = build_counterfactual_examples(
                    f"{sid}_{ck}", task, tgt, dst,
                    g0_target, g0_dest,
                    ck_img, ck_tgt, ck_dst, ck, rng,
                )
                train_examples.extend(cf_exs)
                if cf_exs:
                    print(f"cf×{len(cf_exs)}", end=" ", flush=True)

        else:
            # val: base example only
            val_examples.append(base_ex)

        print()

    # ── 4. 저장 ──────────────────────────────────────────────────────────────
    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    rng.shuffle(train_examples)

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    all_examples = train_examples + val_examples
    print(f"\n총 {len(all_examples)}개 examples:")
    print(f"  train: {len(train_examples)} ({len(train_trial_ids)} trials) → {train_path}")
    print(f"  val:   {len(val_examples)} ({len(val_trial_ids)} trials, base only) → {val_path}")

    from collections import Counter
    dist = Counter(ex["gold_state"] for ex in all_examples)
    print("\nGold state distribution (train+val):")
    for state, cnt in sorted(dist.items()):
        print(f"  {state:<30} {cnt}")

    # trial leakage 검증
    train_tids = {e["id"].rsplit("_C", 1)[0].rsplit("_cf", 1)[0].rsplit("_nog0", 1)[0]
                  .rsplit("_noise", 1)[0].rsplit("_flip", 1)[0].rsplit("_dark", 1)[0]
                  .rsplit("_bright", 1)[0] for e in train_examples}
    val_tids   = {e["id"].rsplit("_C", 1)[0] for e in val_examples}
    overlap = train_tids & val_tids
    if overlap:
        print(f"\n[WARN] trial leakage {len(overlap)}개: {sorted(overlap)[:5]}")
    else:
        print("\n[OK] train/val trial overlap 없음")


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
    parser.add_argument("--no-g0-augment", action="store_true",
                        help="각 샘플에 G₀ 없는 버전도 추가 (count rule로 label). no-memory ablation 학습용")
    parser.add_argument("--counterfactual", action="store_true",
                        help="같은 이미지에서 G₀만 바꾼 반례 2개 추가 (G₀ 비교 학습 강제)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Trial 단위 train 비율 (default: 0.8)")
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
        no_g0_augment=args.no_g0_augment,
        counterfactual=args.counterfactual,
        train_ratio=args.train_ratio,
        seed=args.seed,
        manual_annotations=manual_annotations,
    )


if __name__ == "__main__":
    main()
