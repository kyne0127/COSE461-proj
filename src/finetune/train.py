"""
Fine-tuning script for change-aware grounding decisions.

Pipeline:
  1. Load base Molmo-7B
  2. Merge AmbRes LoRA adapter (preserves scene understanding)
  3. Apply new LoRA for our task (change-aware grounding)
  4. Train on dataset/finetune/train.jsonl
  5. Save new adapter to ckpt/grounding_ft/<run_id>/

Usage:
    python3 src/finetune/train.py \
        --ambres-ckpt AmbRes/ckpt/AB5siP8DA78aA78wR5Y8Mw/checkpoint-390 \
        --train-jsonl dataset/finetune/train.jsonl \
        --val-jsonl   dataset/finetune/val.jsonl \
        --out-dir     ckpt/grounding_ft

    # With all augmentations
    python3 src/finetune/train.py \
        --ambres-ckpt AmbRes/ckpt/AB5siP8DA78aA78wR5Y8Mw/checkpoint-390 \
        --train-jsonl dataset/finetune/train.jsonl \
        --val-jsonl   dataset/finetune/val.jsonl \
        --epochs 3 \
        --lr 5e-5 \
        --lora-r 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments

_REPO  = Path(__file__).resolve().parents[2]
_SRC   = _REPO / "src"
_AMBRES = _REPO.parent / "AmbRes"
for p in (_SRC, _REPO, _AMBRES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from finetune.dataset import DataCollator, GroundingDecisionDataset

MODEL_ID = "allenai/Molmo-7B-D-0924"


def load_model_and_processor(ambres_ckpt: str | Path):
    """Load base Molmo → merge AmbRes adapter → ready for new LoRA."""
    print("[1/3] Loading base Molmo...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    print(f"[2/3] Merging AmbRes LoRA adapter: {ambres_ckpt}")
    model = PeftModel.from_pretrained(model, str(ambres_ckpt))
    model = model.merge_and_unload()
    model.train()
    print("      Merge complete.")

    return model, processor


def apply_new_lora(model, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
    """Apply fresh LoRA adapter for our grounding task."""
    print(f"[3/3] Applying new LoRA (r={lora_r}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["att_proj", "ff_proj", "attn_out", "ff_out"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Molmo for change-aware grounding")
    parser.add_argument("--ambres-ckpt", required=True,
                        help="Path to AmbRes LoRA checkpoint directory")
    parser.add_argument("--train-jsonl", default="dataset/finetune/train.jsonl")
    parser.add_argument("--val-jsonl",   default="dataset/finetune/val.jsonl")
    parser.add_argument("--out-dir",     default="ckpt/grounding_ft")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch-size",  type=int,   default=2)
    parser.add_argument("--grad-accum",  type=int,   default=4)
    parser.add_argument("--lora-r",      type=int,   default=8)
    parser.add_argument("--lora-alpha",  type=int,   default=16)
    parser.add_argument("--lora-dropout",type=float, default=0.05)
    parser.add_argument("--max-length",  type=int,   default=2048)
    parser.add_argument("--run-id",      default=None,
                        help="Resume a previous run by its ID (auto-detects latest checkpoint)")
    args = parser.parse_args()

    # ── run_id / 출력 디렉터리 결정 ──────────────────────────────────────────
    import shortuuid
    run_id  = args.run_id or shortuuid.uuid()[:8]
    out_dir = Path(args.out_dir) / run_id

    # 기존 체크포인트 자동 탐색
    resume_ckpt = None
    if out_dir.exists():
        ck_dirs = sorted(
            [d for d in out_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        if ck_dirs:
            resume_ckpt = str(ck_dirs[-1])
            print(f"[Resume] 기존 체크포인트 발견: {resume_ckpt}")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(args.ambres_ckpt)
    model = apply_new_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    print("\n[Dataset] Loading...")
    train_dataset = GroundingDecisionDataset(args.train_jsonl, processor)
    val_dataset   = GroundingDecisionDataset(args.val_jsonl,   processor)
    collator = DataCollator(pad_token_id=processor.tokenizer.pad_token_id or 0)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        run_name=f"grounding_ft_{run_id}",
        logging_dir=str(out_dir / "tb_logs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=False,
        learning_rate=args.lr,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        save_only_model=True,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(f"\n[Train] run_id={run_id}  out={out_dir}")
    print(f"        train={len(train_dataset)}  val={len(val_dataset)}")
    print(f"        epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}×{args.grad_accum}\n")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(out_dir / "final"))
    print(f"\n완료: adapter saved → {out_dir / 'final'}")

    # ── 학습 요약 저장 ────────────────────────────────────────────────────────
    summary = {
        "run_id":       run_id,
        "ambres_ckpt":  str(args.ambres_ckpt),
        "train_jsonl":  str(args.train_jsonl),
        "val_jsonl":    str(args.val_jsonl),
        "epochs":       args.epochs,
        "lr":           args.lr,
        "lora_r":       args.lora_r,
        "lora_alpha":   args.lora_alpha,
        "n_train":      len(train_dataset),
        "n_val":        len(val_dataset),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
