"""
module.models.ambres.gdino
===========================
GroundingDINO open-vocabulary object detector wrapper.

Transformers의 AutoModelForZeroShotObjectDetection 인터페이스를 사용합니다.
Molmo의 generative point 예측 대신 bounding box + confidence score 기반으로
객체를 탐지하여 오인식 문제를 개선합니다.

모델별 예상 VRAM:
    grounding-dino-tiny : ~0.8 GB
    grounding-dino-base : ~1.5 GB  (권장)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"

# top-down 로봇 장면에서 자주 오탐/미탐되는 레이블을 richer description으로 대체.
# handler.py의 detect() 호출 전에 enrich_labels()를 적용하면 별도 학습 없이
# 즉시 탐지 성능을 개선할 수 있습니다.
_LABEL_ENRICHMENT: dict[str, str] = {
    "cup":        "paper cup or disposable cup viewed from above",
    "box":        "cardboard box or plastic storage box",
    "red box":    "red cardboard box or red plastic box",
    "cube":       "small colored cube block",
    "bottle":     "bottle viewed from above",
    "mug":        "coffee mug or ceramic mug",
    "red mug":    "red coffee mug or red ceramic mug",
}


class GroundingDINO:
    """
    GroundingDINO open-vocabulary detector.

    Args:
        model_id:        HuggingFace 모델 ID
        device:          "cuda" | "cpu"
        box_threshold:   bounding box confidence 최솟값 (default 0.30)
        text_threshold:  텍스트 레이블 매칭 confidence 최솟값 (default 0.25)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        enrich_labels: bool = True,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        logger.info("GroundingDINO: loading %s on %s …", model_id, device)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)
        self._model.eval()
        self._device = device
        self._box_thr = box_threshold
        self._text_thr = text_threshold
        self._enrich = enrich_labels
        logger.info("GroundingDINO: ready  box_thr=%.2f  text_thr=%.2f  enrich=%s",
                    box_threshold, text_threshold, enrich_labels)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(
        self,
        pil_image: Image.Image,
        objects: List[str],
    ) -> Dict[str, List[List[float]]]:
        """
        Args:
            pil_image: PIL RGB 이미지
            objects:   검색할 객체 이름 목록 (e.g. ["red box", "paper cup"])

        Returns:
            {label: [[cx, cy], ...]}
              - 픽셀 좌표, confidence 내림차순 정렬
              - 탐지되지 않은 객체는 빈 리스트 반환
        """
        if not objects:
            return {}

        # enrich_labels: 짧은 generic 레이블을 richer description으로 대체하여 쿼리
        # 원래 레이블 → enriched 쿼리 매핑을 유지해야 결과를 원래 레이블로 돌려줄 수 있음
        if self._enrich:
            query_map = {o: enrich_label(o) for o in objects}
        else:
            query_map = {o: o.lower().strip() for o in objects}

        # GroundingDINO 텍스트 포맷: "red cardboard box. paper cup."
        enriched_objects = list(query_map.values())
        text_query = ". ".join(enriched_objects) + "."
        W, H = pil_image.size

        inputs = self._processor(
            images=pil_image,
            text=text_query,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self._box_thr,
            text_threshold=self._text_thr,
            target_sizes=[(H, W)],
        )[0]

        # enriched query → 원래 레이블로 역매핑하는 역방향 dict
        reverse_map = {v: k for k, v in query_map.items()}

        # {원래 레이블: [{"box":…, "score":…, "coord":…}, ...]} 중간 버킷
        buckets: Dict[str, list] = {obj: [] for obj in objects}

        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            label_str = label.lower().strip() if isinstance(label, str) else str(label)
            matched = _match_label(label_str, enriched_objects)
            if matched is None:
                continue
            original = reverse_map.get(matched, matched)
            x1, y1, x2, y2 = box.tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            buckets[original].append({
                "box":   [x1, y1, x2, y2],
                "score": float(score),
                "coord": [cx, cy],
            })
            logger.debug(
                "GroundingDINO: '%s' → query '%s' → label '%s'  score=%.3f",
                label_str, matched, original, score,
            )

        # confidence 내림차순 정렬 → NMS → 좌표만 반환
        return {
            obj: [d["coord"] for d in _nms(
                sorted(dets, key=lambda d: d["score"], reverse=True)
            )]
            for obj, dets in buckets.items()
        }

    def detect_with_scores(
        self,
        pil_image: Image.Image,
        objects: List[str],
    ) -> Dict[str, List[Dict]]:
        """
        detect()와 동일하나 confidence score + bounding box도 함께 반환합니다.

        Returns:
            {label: [{"coord": [cx, cy], "box": [x1,y1,x2,y2], "score": float}, ...]}
        """
        if not objects:
            return {}

        if self._enrich:
            query_map = {o: enrich_label(o) for o in objects}
        else:
            query_map = {o: o.lower().strip() for o in objects}

        enriched_objects = list(query_map.values())
        text_query = ". ".join(enriched_objects) + "."
        reverse_map = {v: k for k, v in query_map.items()}
        W, H = pil_image.size

        inputs = self._processor(
            images=pil_image,
            text=text_query,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self._box_thr,
            text_threshold=self._text_thr,
            target_sizes=[(H, W)],
        )[0]

        buckets: Dict[str, list] = {obj: [] for obj in objects}

        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            label_str = label.lower().strip() if isinstance(label, str) else str(label)
            matched = _match_label(label_str, enriched_objects)
            if matched is None:
                continue
            original = reverse_map.get(matched, matched)
            x1, y1, x2, y2 = box.tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            buckets[original].append({
                "coord": [cx, cy],
                "box":   [x1, y1, x2, y2],
                "score": float(score),
            })

        return {
            obj: _nms(sorted(items, key=lambda d: d["score"], reverse=True))
            for obj, items in buckets.items()
        }


# ------------------------------------------------------------------ #
# NMS 헬퍼
# ------------------------------------------------------------------ #

def _iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list, iou_thr: float = 0.5) -> list:
    """confidence 내림차순 기준 greedy NMS. 같은 레이블 내 겹치는 박스 제거."""
    keep: list = []
    for d in detections:  # 이미 score 내림차순 정렬 상태
        if not any(_iou(d["box"], k["box"]) > iou_thr for k in keep):
            keep.append(d)
    return keep


# ------------------------------------------------------------------ #
# 레이블 매칭 헬퍼
# ------------------------------------------------------------------ #

def _match_label(detected_label: str, objects: List[str]) -> Optional[str]:
    """
    GroundingDINO가 반환한 레이블을 원래 objects 목록에 매핑합니다.

    우선순위:
    1. 완전 일치 (exact match)
    2. 부분 문자열 포함 (detected_label ⊂ object 또는 반대)
    3. 단어 단위 겹침 (word-level overlap, 최고 점수 객체)
    """
    dl = detected_label.lower()

    # 1. 완전 일치
    for obj in objects:
        if dl == obj.lower():
            return obj

    # 2. 부분 문자열
    for obj in objects:
        ol = obj.lower()
        if dl in ol or ol in dl:
            return obj

    # 3. 단어 겹침 — 최고 점수 객체 반환
    dl_words = set(dl.split())
    best_score, best_obj = 0, None
    for obj in objects:
        ol_words = set(obj.lower().split())
        overlap = len(dl_words & ol_words)
        if overlap > best_score:
            best_score, best_obj = overlap, obj

    return best_obj if best_score > 0 else None


# ------------------------------------------------------------------ #
# 레이블 enrichment 헬퍼
# ------------------------------------------------------------------ #

def enrich_label(label: str) -> str:
    """
    짧은 generic 레이블을 top-down 로봇 장면에 맞는 richer description으로 변환.

    _LABEL_ENRICHMENT 테이블에 없는 레이블은 그대로 반환합니다.
    외부에서 직접 호출하여 커스텀 enrichment를 테스트할 수 있습니다.

    Args:
        label: 원래 객체 레이블 (e.g. "cup", "red box")

    Returns:
        enriched description (e.g. "paper cup or disposable cup viewed from above")
    """
    return _LABEL_ENRICHMENT.get(label.lower().strip(), label.lower().strip())
