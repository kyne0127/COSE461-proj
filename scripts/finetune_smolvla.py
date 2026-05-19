#!/usr/bin/env python3
"""
scripts/finetune_smolvla.py
===========================
SmolVLA visual fine-tuning on locally collected EpisodeBuffer data.

Fine-tuning mode (visual_finetune=true in model config):
  - SigLIP vision encoder: lr=1e-5  (visual domain adaptation)
  - lm_expert action expert: lr=1e-4
  - SmolLM2 text backbone: frozen

Data format (written by DatasetWriter):
  data_root/
    episode_000000/
      states.npy          (T, state_dim)
      actions.npy         (T, action_dim)
      <cam_name>/
        000000.npy ... (H, W, C) uint8

Usage:
    python scripts/finetune_smolvla.py \\
        --data-root  data/my_robot_dataset \\
        --save-dir   checkpoints/smolvla_visual_ft \\
        --task       "pick up the banana" \\
        --steps      3000 \\
        --batch-size 4

    # Resume from a previous checkpoint:
    python scripts/finetune_smolvla.py \\
        --data-root  data/my_robot_dataset \\
        --checkpoint checkpoints/smolvla_visual_ft/step_001000 \\
        --save-dir   checkpoints/smolvla_visual_ft \\
        --task       "pick up the banana" \\
        --steps      3000
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

class EpisodeDiskDataset:
    """
    Reads episodes written by DatasetWriter and yields (obs, action_chunk) pairs.

    Each sample:
        images:  Dict[cam_name, (H, W, C) uint8 ndarray]
        state:   (state_dim,) float32
        actions: (chunk_size, action_dim) float32   ← action chunk starting at frame t
        task:    str
    """

    def __init__(
        self,
        data_root: Path,
        task: str,
        chunk_size: int = 50,
        cameras: list[str] | None = None,
    ) -> None:
        self.task       = task
        self.chunk_size = chunk_size

        # Collect all episode dirs
        ep_dirs = sorted(data_root.glob("episode_*"))
        if not ep_dirs:
            raise FileNotFoundError(f"No episode dirs found under {data_root}")

        self._samples: list[tuple] = []  # (ep_dir, frame_idx)
        for ep_dir in ep_dirs:
            actions = _load_npy(ep_dir / "actions.npy")
            T = len(actions)
            # Only include frames where a full chunk is available
            for t in range(T - chunk_size + 1):
                self._samples.append((ep_dir, t))

        if not self._samples:
            raise ValueError(
                f"No valid samples (chunk_size={chunk_size}). "
                "Collect more data or reduce chunk_size."
            )

        # Detect camera names from first episode
        if cameras is not None:
            self._cameras = cameras
        else:
            first_dir = ep_dirs[0]
            self._cameras = [
                d.name for d in first_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        print(f"[dataset] {len(ep_dirs)} episodes | {len(self._samples)} samples "
              f"| cameras: {self._cameras}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        ep_dir, t = self._samples[idx]

        images = {}
        for cam in self._cameras:
            img_path = ep_dir / cam / f"{t:06d}.npy"
            images[cam] = _load_npy(img_path)  # (H, W, C) uint8

        states  = _load_npy(ep_dir / "states.npy")
        actions = _load_npy(ep_dir / "actions.npy")

        return {
            "images": images,                               # Dict[str, (H,W,C)]
            "state":  states[t],                            # (state_dim,)
            "action": actions[t : t + self.chunk_size],     # (chunk_size, action_dim)
            "task":   self.task,
        }


def _load_npy(path: Path):
    import numpy as np
    return np.load(str(path))


def collate_fn(samples: list[dict]) -> dict:
    """Stack list of samples into a batch dict."""
    import numpy as np
    cam_names = list(samples[0]["images"].keys())
    batch = {
        "images": {
            cam: np.stack([s["images"][cam] for s in samples])  # (B, H, W, C)
            for cam in cam_names
        },
        "states":  np.stack([s["state"]  for s in samples]),    # (B, state_dim)
        "actions": np.stack([s["action"] for s in samples]),    # (B, chunk, action_dim)
        "tasks":   [s["task"] for s in samples],
    }
    return batch


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler

    from module.models.smolvla.model import SmolVLAModel
    from module.models.base_model import TrainingBatch

    data_root  = Path(args.data_root)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────
    chunk_size = args.chunk_size
    dataset = EpisodeDiskDataset(
        data_root=data_root,
        task=args.task,
        chunk_size=chunk_size,
    )
    sampler = RandomSampler(dataset, replacement=True, num_samples=args.steps * args.batch_size)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model_config = {
        "model_id":        args.base_model,
        "device":          "cuda" if torch.cuda.is_available() else "cpu",
        "precision":       "bfloat16",
        "use_amp":         True,
        "visual_finetune": True,    # SigLIP + lm_expert; text frozen
        "grad_checkpoint": True,    # saves ~2GB VRAM
        "lr":              args.lr,
        "lr_vision":       args.lr_vision,
        "weight_decay":    args.weight_decay,
    }
    model = SmolVLAModel(model_config)
    checkpoint = args.checkpoint or ""
    model.load_checkpoint(checkpoint)

    # ── Training ───────────────────────────────────────────────────────────
    step       = 0
    t_start    = time.perf_counter()
    loss_accum = 0.0

    print(f"\n[finetune] Starting — {args.steps} steps, batch_size={args.batch_size}")
    print(f"[finetune] Save every {args.save_every} steps → {save_dir}\n")

    for batch_raw in loader:
        tb = TrainingBatch(
            images  = batch_raw["images"],
            states  = batch_raw["states"],
            actions = batch_raw["actions"],
            extra   = {"task_texts": batch_raw["tasks"]},
        )
        loss_dict = model.train_step(tb)
        loss_accum += loss_dict.get("loss", 0.0)
        step += 1

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t_start
            avg_loss = loss_accum / args.log_every
            loss_accum = 0.0
            print(
                f"  step {step:5d}/{args.steps}"
                f"  loss={avg_loss:.4f}"
                f"  {elapsed:.0f}s"
            )

        if step % args.save_every == 0 or step == args.steps:
            ckpt_dir = save_dir / f"step_{step:06d}"
            model.save_checkpoint(ckpt_dir)
            print(f"  [saved] {ckpt_dir}")

        if step >= args.steps:
            break

    print(f"\n[finetune] Done. Final checkpoint: {save_dir / f'step_{step:06d}'}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLA (visual domain adaptation) on local episode data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-root",    required=True,
                        help="Root directory of collected episodes (DatasetWriter output)")
    parser.add_argument("--save-dir",     required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--task",         required=True,
                        help="Task description string (same as used during collection)")
    parser.add_argument("--base-model",   default="lerobot/smolvla_base",
                        help="HF model ID or local path for base checkpoint")
    parser.add_argument("--checkpoint",   default=None,
                        help="Resume from a saved checkpoint (overrides --base-model)")
    parser.add_argument("--steps",        type=int,   default=3000)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--chunk-size",   type=int,   default=50,
                        help="Action chunk length (must match SmolVLA n_action_steps=50)")
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="lm_expert learning rate")
    parser.add_argument("--lr-vision",    type=float, default=1e-5,
                        help="SigLIP vision encoder learning rate (keep 10x lower than --lr)")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-every",    type=int,   default=50)
    parser.add_argument("--save-every",   type=int,   default=500)
    parser.add_argument("--num-workers",  type=int,   default=2)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
