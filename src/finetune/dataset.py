"""
Fine-tuning dataset for change-aware grounding decisions.

Loads train.jsonl built by build_dataset.py and converts to
Molmo conversation format using AmbRes's process_msg_list utility.

Conversation structure (single turn):
  User:      <image> TASK DESCRIPTION: {structured_prompt}
  Assistant: {"task_ambiguous": ..., "stop_bool": ..., "clarifying_question": "..."}

User tokens → labels=-100 (not trained)
Assistant tokens → labels=token_ids (trained)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

_AMBRES = Path(__file__).resolve().parents[3] / "AmbRes"
if str(_AMBRES) not in sys.path:
    sys.path.insert(0, str(_AMBRES))

from ambres.training.data import process_msg_list


class GroundingDecisionDataset(Dataset):
    """Dataset for change-aware grounding decision fine-tuning.

    Each example:
        image:            checkpoint image (PIL)
        task_description: structured prompt (G0 + DINO + taxonomy)
        label:            {"task_ambiguous": bool, "stop_bool": bool,
                           "clarifying_question": str}
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        processor: AutoProcessor,
        image_base_dir: str | Path | None = None,
    ) -> None:
        self.processor = processor
        self.image_base_dir = Path(image_base_dir) if image_base_dir else None

        raw = []
        for line in Path(jsonl_path).read_text().splitlines():
            if line.strip():
                raw.append(json.loads(line))
        self.data = raw
        print(f"  Loaded {len(self.data)} examples from {Path(jsonl_path).name}")

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_image(self, img_path: str) -> Path:
        p = Path(img_path)
        if p.is_absolute() and p.exists():
            return p
        if self.image_base_dir is not None:
            candidate = self.image_base_dir / p
            if candidate.exists():
                return candidate
        # fallback: relative to repo root
        repo = Path(__file__).resolve().parents[2]
        return repo / p

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, str]]:
        task_description = example["task_description"]
        label = {
            "task_ambiguous": example["ambiguity_bool"],
            "stop_bool":      example["stop_bool"],
            "clarifying_question": example.get("clarifying_question", ""),
        }
        assistant_text = json.dumps(label, ensure_ascii=False)

        return [
            {"role": "user",      "content": f"<image>\nTASK DESCRIPTION: {task_description}"},
            {"role": "assistant", "content": assistant_text},
        ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.data[idx]
        img_path = self._resolve_image(example["image"])
        image = Image.open(img_path).convert("RGB")
        messages = self._build_messages(example)

        data_dict = process_msg_list(
            messages, [image], self.processor, append_eos=True
        )
        return data_dict


class DataCollator:
    """Pads a batch of GroundingDecisionDataset items to same length."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids_list    = [item["input_ids"]      for item in batch]
        labels_list       = [item["labels"]          for item in batch]
        attn_list         = [item.get("attention_mask",
                               torch.ones_like(item["input_ids"])) for item in batch]
        images_list       = [item["images"]          for item in batch]
        img_idx_list      = [item["image_input_idx"] for item in batch]
        img_mask_list     = [item["image_masks"]     for item in batch]

        max_len = max(t.size(0) for t in input_ids_list)

        def pad_right(tensors, pad_val, dtype=torch.long):
            out = torch.full((len(tensors), max_len), pad_val, dtype=dtype)
            for i, t in enumerate(tensors):
                out[i, :t.size(0)] = t
            return out

        input_ids      = pad_right(input_ids_list, self.pad_token_id)
        labels         = pad_right(labels_list,   -100)
        attention_mask = pad_right(attn_list,      0)

        # Images are already (C, H, W) float tensors — stack directly
        images        = torch.stack(images_list)
        image_input_idx = torch.stack(img_idx_list)
        image_masks    = torch.stack(img_mask_list)

        return {
            "input_ids":       input_ids,
            "labels":          labels,
            "attention_mask":  attention_mask,
            "images":          images,
            "image_input_idx": image_input_idx,
            "image_masks":     image_masks,
        }
