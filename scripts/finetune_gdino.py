#!/usr/bin/env python3
"""
scripts/finetune_gdino.py

CVAT에서 export한 COCO JSON 어노테이션으로 GroundingDINO를 파인튜닝합니다.

학습 전략:
  - Swin-T 비전 백본 + BERT 텍스트 인코더 : 동결 (freeze)
  - Text-visual cross-attention + detection head : 학습
  → VRAM ~6 GB, RTX 6000 Ada에서 약 30~60분 (100에폭)

GPU 서버 실행:
    python /workspace/COSE461-proj/scripts/finetune_gdino.py \\
        --coco-json /tmp/cvat_upload/annotations.json \\
        --image-dir /tmp/cvat_upload \\
        --out-dir   /workspace/COSE461-proj/checkpoints/gdino-ft

    # 검증셋 따로 있는 경우
    python .../finetune_gdino.py \\
        --coco-json  annotations_train.json \\
        --val-json   annotations_val.json \\
        --image-dir  /tmp/cvat_upload \\
        --out-dir    /workspace/COSE461-proj/checkpoints/gdino-ft \\
        --epochs 80 --batch-size 4 --lr 2e-5

완료 후 gdino.py / ambres.yaml에서:
    gdino_model_id: "/workspace/COSE461-proj/checkpoints/gdino-ft/best"
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    get_cosine_schedule_with_warmup,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class GDinoDataset(Dataset):
    """
    COCO JSON 어노테이션 → GroundingDINO 학습 포맷으로 변환.

    텍스트 쿼리: 이미지에 등장하는 모든 클래스를 ". "으로 연결
    예) "cup. red box."

    labels 포맷:
        boxes       : (N, 4) normalized CxCyWH tensor
        class_labels: (N,)   0-indexed (텍스트 phrase 순서)
    """

    def __init__(
        self,
        coco_path: str,
        image_dir: str,
        processor: GroundingDinoProcessor,
        max_size: int = 640,
        augment: bool = True,
    ):
        with open(coco_path, encoding="utf-8") as f:
            coco = json.load(f)

        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_size  = max_size
        self.augment   = augment

        # 카테고리 id → name 매핑
        self.cat_id2name: Dict[int, str] = {
            c["id"]: c["name"] for c in coco["categories"]
        }

        # 이미지 id → {file_name, width, height}
        img_meta: Dict[int, dict] = {
            img["id"]: img for img in coco["images"]
        }

        # 이미지별 annotations 수집
        ann_by_img: Dict[int, list] = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        self.samples: List[dict] = []
        for img_id, meta in img_meta.items():
            anns = ann_by_img.get(img_id, [])
            if not anns:
                continue  # 어노테이션 없는 이미지 건너뜀
            self.samples.append({
                "file_name": meta["file_name"],
                "width":     meta["width"],
                "height":    meta["height"],
                "anns":      anns,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[dict]:
        sample = self.samples[idx]
        img_path = self.image_dir / sample["file_name"]

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] 이미지 로드 실패 ({img_path}): {e}")
            return None

        W, H = pil_img.size
        anns  = sample["anns"]

        # ─ 텍스트 쿼리 구성: 중복 제거, 알파벳 순 정렬 ─
        classes_in_image = sorted({
            self.cat_id2name[a["category_id"]] for a in anns
        })
        text_query = ". ".join(classes_in_image) + "."

        # ─ bbox 변환: COCO [x, y, w, h] → normalized CxCyWH ─
        boxes, class_labels = [], []
        for ann in anns:
            cat_name = self.cat_id2name[ann["category_id"]]
            cls_idx  = classes_in_image.index(cat_name)

            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / W
            cy = (y + bh / 2) / H
            nw = bw / W
            nh = bh / H

            # 범위 클리핑 (어노테이션 오류 방어)
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.01, min(1.0, nw))
            nh = max(0.01, min(1.0, nh))

            boxes.append([cx, cy, nw, nh])
            class_labels.append(cls_idx)

        # ─ processor로 인코딩 ─
        encoding = self.processor(
            images=pil_img,
            text=text_query,
            return_tensors="pt",
        )

        return {
            "pixel_values":    encoding["pixel_values"].squeeze(0),
            "input_ids":       encoding["input_ids"].squeeze(0),
            "attention_mask":  encoding["attention_mask"].squeeze(0),
            "token_type_ids":  encoding.get("token_type_ids",
                                torch.zeros_like(encoding["input_ids"])).squeeze(0),
            "boxes":           torch.tensor(boxes, dtype=torch.float32),
            "class_labels":    torch.tensor(class_labels, dtype=torch.long),
            "text_query":      text_query,
        }


def collate_fn(batch: list) -> Optional[dict]:
    """None 샘플 제거 + 이미지/텍스트 패딩."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    pixel_values   = torch.stack([b["pixel_values"]   for b in batch])
    input_ids      = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence(
        [b["token_type_ids"] for b in batch], batch_first=True, padding_value=0)

    # labels: 배치 내 각 이미지별 GT boxes + class_labels
    labels = [
        {"boxes": b["boxes"], "class_labels": b["class_labels"]}
        for b in batch
    ]

    return {
        "pixel_values":   pixel_values,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels":         labels,
    }


# ── 모델 설정 ─────────────────────────────────────────────────────────────────

def freeze_backbone(model: GroundingDinoForObjectDetection) -> None:
    """
    비전 백본(Swin-T)과 텍스트 인코더(BERT)를 동결합니다.
    Cross-attention 인코더 + detection head만 학습합니다.
    → VRAM 절약 + 과적합 방지
    """
    frozen_prefixes = (
        "model.backbone",       # Swin-T
        "model.encoder.layers", # feature pyramid
        "model.text_backbone",  # BERT
    )
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[freeze] 학습 파라미터: {trainable/1e6:.1f}M / {total/1e6:.1f}M")


# ── 커스텀 손실 함수 ──────────────────────────────────────────────────────────
# 이유: transformers GroundingDINO의 내장 loss_function은 logits(256-dim)과
#       config.num_labels(2)가 불일치하여 cross_entropy에서 NaN 발생.
#       pred_boxes + logits를 직접 받아 L1 + GIoU + text focal loss를 계산함.

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (M,4) b: (N,4) xyxy → IoU (M,N)"""
    inter_x1 = torch.max(a[:, None, 0], b[None, :, 0])
    inter_y1 = torch.max(a[:, None, 1], b[None, :, 1])
    inter_x2 = torch.min(a[:, None, 2], b[None, :, 2])
    inter_y2 = torch.min(a[:, None, 3], b[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(1e-6)


def _giou(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """element-wise GIoU, both (N,4)"""
    iou = _box_iou(pred_xyxy, gt_xyxy).diag()
    encl_x1 = torch.min(pred_xyxy[:, 0], gt_xyxy[:, 0])
    encl_y1 = torch.min(pred_xyxy[:, 1], gt_xyxy[:, 1])
    encl_x2 = torch.max(pred_xyxy[:, 2], gt_xyxy[:, 2])
    encl_y2 = torch.max(pred_xyxy[:, 3], gt_xyxy[:, 3])
    encl = (encl_x2 - encl_x1).clamp(0) * (encl_y2 - encl_y1).clamp(0)
    area_p = (pred_xyxy[:, 2]-pred_xyxy[:, 0]) * (pred_xyxy[:, 3]-pred_xyxy[:, 1])
    area_g = (gt_xyxy[:, 2]-gt_xyxy[:, 0]) * (gt_xyxy[:, 3]-gt_xyxy[:, 1])
    union  = area_p + area_g - iou * (area_p + area_g - iou * encl).clamp(1e-6)
    return iou - (encl - union.clamp(1e-6)) / encl.clamp(1e-6)


def compute_detection_loss(
    pred_boxes: torch.Tensor,   # (batch, num_q, 4) CxCyWH normalized
    logits:     torch.Tensor,   # (batch, num_q, seq_len)
    labels:     List[Dict],
    device:     torch.device,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    labels[i]: {'boxes': (N,4) CxCyWH, 'class_labels': (N,) token-position index}

    각 이미지별로:
      1. IoU 기반 헝가리안-like 매칭 (greedy: GT별 최고 IoU 예측 선택)
      2. L1 + GIoU 박스 회귀 손실
      3. text-focal 손실 (매칭된 쿼리 → 해당 토큰 위치 양성)
    """
    total = torch.zeros(1, device=device, requires_grad=False)
    count = 0

    for b, lbl in enumerate(labels):
        gt_boxes = lbl["boxes"]           # (num_gt, 4)
        cls_toks = lbl["class_labels"]    # (num_gt,) token positions
        num_gt = gt_boxes.shape[0]
        if num_gt == 0:
            continue

        pb = pred_boxes[b]               # (num_q, 4)
        lg = logits[b]                   # (num_q, seq_len)
        seq_len = lg.shape[-1]

        # ─ 매칭 ─
        pb_xyxy = _cxcywh_to_xyxy(pb.detach())
        gt_xyxy = _cxcywh_to_xyxy(gt_boxes)
        iou_mat = _box_iou(pb_xyxy, gt_xyxy)      # (num_q, num_gt)
        matched_pred_idx = iou_mat.argmax(0)        # (num_gt,)

        # ─ 박스 손실 ─
        mp = pb[matched_pred_idx]                   # (num_gt, 4)
        mp_xyxy = _cxcywh_to_xyxy(mp)
        l1   = F.l1_loss(mp, gt_boxes)
        giou = (1 - _giou(mp_xyxy, gt_xyxy)).mean()
        box_loss = 5.0 * l1 + 2.0 * giou

        # ─ text focal 손실 ─
        # 패딩 위치의 -inf를 유한값으로 교체
        lg_clamped = lg.clamp(min=-1e4)             # (num_q, seq_len)

        # 양성 타겟: 매칭된 쿼리 × 해당 토큰 위치
        target = torch.zeros_like(lg_clamped)
        for i, (pi, ti) in enumerate(zip(matched_pred_idx, cls_toks)):
            ti_int = ti.item()
            if 0 <= ti_int < seq_len:
                target[pi, ti_int] = 1.0

        p = lg_clamped.sigmoid()
        ce  = F.binary_cross_entropy_with_logits(lg_clamped, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        focal = ce * ((1 - p_t).clamp(1e-6) ** gamma)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        text_loss = (alpha_t * focal).mean()

        total = total + box_loss + text_loss
        count += 1

    return total / max(count, 1)


# ── 학습 루프 ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ─ 모델 + processor 로드 ─
    base_id  = args.base_model
    processor = GroundingDinoProcessor.from_pretrained(base_id)
    model     = GroundingDinoForObjectDetection.from_pretrained(
        base_id, ignore_mismatched_sizes=True
    ).to(device)

    freeze_backbone(model)

    # ─ 데이터셋 ─
    train_ds = GDinoDataset(args.coco_json, args.image_dir, processor, augment=True)
    val_ds   = None
    if args.val_json:
        val_ds = GDinoDataset(args.val_json, args.image_dir, processor, augment=False)

    print(f"[data] train={len(train_ds)}  val={len(val_ds) if val_ds else 0}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    ) if val_ds else None

    # ─ optimizer / scheduler ─
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = total_steps // 10
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ─ 출력 디렉토리 ─
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    # ─ 학습 루프 ─
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            if batch is None:
                continue

            labels = [
                {k: v.to(device) for k, v in lbl.items()}
                for lbl in batch["labels"]
            ]
            # labels를 모델에 직접 전달하지 않음:
            # 내장 loss_function이 logits(256-dim) vs num_labels(2) 불일치로 NaN 발생.
            outputs = model(
                pixel_values   = batch["pixel_values"].to(device),
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                token_type_ids = batch["token_type_ids"].to(device),
            )

            loss = compute_detection_loss(
                outputs.pred_boxes, outputs.logits, labels, device
            )
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches  += 1

        avg_train = train_loss / max(n_batches, 1)

        # ─ 검증 ─
        avg_val = _validate(model, val_loader, device) if val_loader else None

        log = {
            "epoch":      epoch,
            "train_loss": round(avg_train, 4),
            "val_loss":   round(avg_val, 4) if avg_val is not None else None,
        }
        history.append(log)

        val_str = f"  val={avg_val:.4f}" if avg_val is not None else ""
        print(f"[epoch {epoch:3d}/{args.epochs}] train={avg_train:.4f}{val_str}")

        # ─ best 모델 저장 ─
        monitor = avg_val if avg_val is not None else avg_train
        if monitor < best_val_loss:
            best_val_loss = monitor
            _save(model, processor, out_dir / "best")
            print(f"  → best 모델 저장 (loss={monitor:.4f})")

        # ─ 주기 체크포인트 ─
        if epoch % args.save_every == 0:
            _save(model, processor, out_dir / f"epoch_{epoch:03d}")

    # ─ 최종 저장 ─
    _save(model, processor, out_dir / "final")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n학습 완료. 저장 경로: {out_dir}")
    print(f"gdino_model_id: \"{out_dir / 'best'}\"  로 설정하세요.")


def _validate(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            labels = [
                {k: v.to(device) for k, v in lbl.items()}
                for lbl in batch["labels"]
            ]
            outputs = model(
                pixel_values   = batch["pixel_values"].to(device),
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                token_type_ids = batch["token_type_ids"].to(device),
            )
            loss = compute_detection_loss(
                outputs.pred_boxes, outputs.logits, labels, device
            )
            if torch.isfinite(loss):
                total += loss.item()
                n += 1
    model.train()
    return total / max(n, 1)


def _save(model, processor, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    processor.save_pretrained(str(path))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GroundingDINO fine-tuner")
    parser.add_argument("--coco-json",   required=True,
                        help="CVAT export COCO JSON (학습용)")
    parser.add_argument("--val-json",    default=None,
                        help="검증용 COCO JSON (없으면 train loss만 모니터링)")
    parser.add_argument("--image-dir",   required=True,
                        help="이미지 디렉토리 (CVAT 업로드 때 쓴 디렉토리)")
    parser.add_argument("--out-dir",     required=True,
                        help="파인튜닝 모델 저장 경로")
    parser.add_argument("--base-model",  default="IDEA-Research/grounding-dino-base",
                        help="베이스 모델 (HF ID 또는 로컬 경로)")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--save-every",  type=int,   default=20,
                        help="N 에폭마다 중간 체크포인트 저장")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
