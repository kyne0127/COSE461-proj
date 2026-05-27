"""
Grounding DINO 기반 오픈보캐블러리 객체 검출기.

task_objects (e.g. ["cube", "red box"]) 를 받아
각 객체의 bbox 중심 픽셀 좌표와 개수를 반환한다.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

logger = logging.getLogger(__name__)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
BOX_THRESHOLD  = 0.20
TEXT_THRESHOLD = 0.15


class DINODetector:
    def __init__(self, device: str = "cuda",
                 box_threshold: float = BOX_THRESHOLD,
                 text_threshold: float = TEXT_THRESHOLD):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        logger.info("DINODetector: loading %s (box=%.2f, text=%.2f) ...",
                    MODEL_ID, box_threshold, text_threshold)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, cache_dir="/workspace/hf_cache"
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID, cache_dir="/workspace/hf_cache"
        ).to(device)
        self.model.eval()
        logger.info("DINODetector: ready")

    def _infer_with_phrase_scores(
        self,
        image: Image.Image,
        objects: List[str],
        box_threshold: float,
        text_threshold: float,
    ) -> List[Dict]:
        """DINO 추론 후 각 bbox를 phrase별 logit score 비교로 매칭.

        복합 레이블("cube box") 대신 각 phrase 토큰 span의 max score를 직접 비교해
        가장 높은 score의 phrase에 bbox를 할당한다.

        Returns:
            list of {
                "box":          [x1, y1, x2, y2] (픽셀, xyxy),
                "center":       [cx, cy],
                "score":        float,            # max score over all text tokens
                "phrase_scores": {obj: float},    # phrase별 max score
                "matched":      str | None,
            }
        """
        W, H = image.size
        text_prompt = " . ".join(objects) + " ."

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # logits: (1, num_queries, seq_len) — sigmoid 적용
        logits = outputs.logits[0].sigmoid().cpu()       # (Q, L)
        pred_boxes = outputs.pred_boxes[0].cpu()          # (Q, 4) normalized cxcywh

        # phrase별 토큰 span 찾기
        input_ids = inputs["input_ids"][0].cpu().tolist()
        phrase_spans = _find_phrase_spans(self.processor.tokenizer, input_ids, objects)

        # box_threshold 기준으로 쿼리 필터링
        max_scores = logits.max(dim=-1).values            # (Q,)
        keep_mask = max_scores > box_threshold
        kept_logits     = logits[keep_mask]               # (N, L)
        kept_boxes_norm = pred_boxes[keep_mask]           # (N, 4)
        kept_scores     = max_scores[keep_mask]           # (N,)

        # 정규화 cxcywh → 픽셀 xyxy
        cx = kept_boxes_norm[:, 0]; cy = kept_boxes_norm[:, 1]
        bw = kept_boxes_norm[:, 2]; bh = kept_boxes_norm[:, 3]
        x1 = ((cx - bw / 2) * W).numpy()
        y1 = ((cy - bh / 2) * H).numpy()
        x2 = ((cx + bw / 2) * W).numpy()
        y2 = ((cy + bh / 2) * H).numpy()

        results = []
        for i in range(len(kept_scores)):
            box   = [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])]
            score = float(kept_scores[i])
            row   = kept_logits[i].numpy()  # (L,)

            # phrase별 max score 계산
            phrase_scores: Dict[str, float] = {}
            for obj, (s, e) in phrase_spans.items():
                ps = float(row[s:e].max()) if e > s else 0.0
                if ps >= text_threshold:
                    phrase_scores[obj] = ps

            # 가장 높은 phrase에 할당 (동점이면 미할당)
            matched: str | None = None
            if phrase_scores:
                best_score = max(phrase_scores.values())
                best = [o for o, v in phrase_scores.items() if v == best_score]
                if len(best) == 1:
                    matched = best[0]

            results.append({
                "box":           box,
                "center":        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                "score":         score,
                "phrase_scores": phrase_scores,
                "matched":       matched,
            })

        return results

    def detect(
        self,
        image: Image.Image,
        objects: List[str],
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> Dict[str, List[List[int]]]:
        """
        Returns:
            {obj: [[cx, cy], ...]}  픽셀 좌표 (bbox 중심점)
            인스턴스가 없으면 빈 리스트.
        """
        box_threshold  = box_threshold  if box_threshold  is not None else self.box_threshold
        text_threshold = text_threshold if text_threshold is not None else self.text_threshold

        if not objects:
            return {}

        raw = self._infer_with_phrase_scores(image, objects, box_threshold, text_threshold)
        matched = [d for d in raw if d["matched"]]
        kept = _nms_global(matched)

        detections: Dict[str, List[List[int]]] = {obj: [] for obj in objects}
        for d in kept:
            cx, cy = int(d["center"][0]), int(d["center"][1])
            detections[d["matched"]].append([cx, cy])

        logger.debug("DINODetector: %s", {k: len(v) for k, v in detections.items()})
        return detections

    def detect_raw(
        self,
        image: Image.Image,
        objects: List[str],
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> List[Dict]:
        """bbox, score, matched_obj를 모두 반환 (시각화/디버깅용).

        Returns:
            list of {
                "box":    [x1, y1, x2, y2]  (픽셀, xyxy),
                "center": [cx, cy],
                "score":  float,
                "dino_label": str,           # best phrase or "unmatched"
                "matched":    str | None,
            }
        """
        box_threshold  = box_threshold  if box_threshold  is not None else self.box_threshold
        text_threshold = text_threshold if text_threshold is not None else self.text_threshold

        if not objects:
            return []

        raw = self._infer_with_phrase_scores(image, objects, box_threshold, text_threshold)

        matched_list  = [d for d in raw if d["matched"]]
        unmatched_list = [d for d in raw if not d["matched"]]

        kept = _nms_global(matched_list)

        result = []
        for d in kept + unmatched_list:
            result.append({
                "box":        [int(v) for v in d["box"]],
                "center":     [int(d["center"][0]), int(d["center"][1])],
                "score":      d["score"],
                "dino_label": d["matched"] if d["matched"] else "unmatched",
                "matched":    d["matched"],
            })
        return result


def _find_phrase_spans(
    tokenizer,
    input_ids: List[int],
    objects: List[str],
) -> Dict[str, tuple[int, int]]:
    """각 phrase의 토큰 span (start, end) 을 input_ids에서 찾아 반환."""
    spans: Dict[str, tuple[int, int]] = {}
    for obj in objects:
        obj_ids = tokenizer(obj, add_special_tokens=False)["input_ids"]
        n = len(obj_ids)
        for i in range(len(input_ids) - n + 1):
            if input_ids[i : i + n] == obj_ids:
                spans[obj] = (i, i + n)
                break
    return spans


def _nms_global(items: list, iou_threshold: float = 0.5) -> list:
    """클래스 무관 global NMS. 겹치는 bbox 중 score 높은 것만 유지."""
    if len(items) <= 1:
        return items

    boxes = np.array([d["box"] for d in items], dtype=float)
    scores = np.array([d["score"] for d in items])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_threshold]

    return [items[i] for i in keep]


def spatial_select(
    candidates: List[List[int]],
    direction: str,
) -> List[int] | None:
    """
    여러 bbox 중심점 중 방향 힌트에 맞는 것을 선택.

    direction: "left" | "right" | "near" | "far" | "front" | "back"
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    if direction in ("left",):
        return min(candidates, key=lambda p: p[0])
    elif direction in ("right",):
        return max(candidates, key=lambda p: p[0])
    elif direction in ("near", "front"):
        return max(candidates, key=lambda p: p[1])
    elif direction in ("far", "back"):
        return min(candidates, key=lambda p: p[1])
    return candidates[0]


def extract_direction(text: str) -> str | None:
    """사용자 응답 텍스트에서 방향 힌트 추출."""
    text = text.lower()
    for kw in ("left", "right", "near", "far", "front", "back"):
        if kw in text:
            return kw
    return None
