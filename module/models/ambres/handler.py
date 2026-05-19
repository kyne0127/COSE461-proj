"""
module.models.ambres.handler
=============================
AmbRes(모호성 해소 VLM) → GenericInferenceService 핸들러.

AmbRes 패키지(ambres.ambres_model, ambres.sam2_model)를 직접 import하여
GPU 서버 단일 프로세스에서 실행합니다.
Desktop은 GenericClient를 통해 단일 gRPC 연결로 모호성 해소와 로봇 액션 생성을
모두 처리합니다.

지원 메서드:
    reset         : 세션 대화 기록 초기화
    query         : 이미지 + task_description → 모호성 판단 + 객체 추출
    respond       : 사용자 명확화 답변 → 모호성 해소 + 최종 객체 확정
    detect        : 이미지 + 객체 목록 → pixel 좌표 검출 (Molmo point 출력)
    set_image_sam : SAM2 이미지 사전 세팅
    query_mask    : pixel 좌표 points → 세그멘테이션 마스크

Config keys:
    model_type    (str):  "fs_prompt" | "finetune"  [default: "fs_prompt"]
    adapter_ckpt  (str):  finetune 체크포인트 이름 (model_type="finetune" 시 필수)
    use_detection (bool): query/respond 단계에서 Molmo detect 자동 수행 [default: False]
    use_sam       (bool): SAM2 로드 여부 [default: False]
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from module.models.base_handler import BaseHandler, HandlerRegistry

logger = logging.getLogger(__name__)


@HandlerRegistry.register("ambres")
class AmbResHandler(BaseHandler):
    """
    AmbRes VLM을 gRPC GenericInferenceService에 연결하는 핸들러.

    Molmo 7B 모델은 단일 인스턴스를 유지하고, 세션별 대화 기록(messages + images)을
    별도 dict에 보관하여 멀티턴 대화를 지원합니다.
    추론은 threading.Lock으로 직렬화합니다 (Molmo는 thread-safe하지 않음).
    """

    # ------------------------------------------------------------------ #
    # 초기화
    # ------------------------------------------------------------------ #

    def setup(self, config: Dict[str, Any]) -> None:
        model_type    = config.get("model_type", "fs_prompt")
        use_detection = config.get("use_detection", False)
        use_sam       = config.get("use_sam", False)
        save_dir      = config.get("save_dir", None)

        self._save_dir = save_dir
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            logger.info("AmbResHandler: received images will be saved to %s", save_dir)

        logger.info(
            "AmbResHandler.setup: model_type=%s  use_detection=%s  use_sam=%s",
            model_type, use_detection, use_sam,
        )

        try:
            from ambres.ambres_model import AmbresFSPrompt, AmbresFineTuned
        except ImportError as e:
            raise ImportError(
                "AmbRes 패키지를 찾을 수 없습니다. "
                "PYTHONPATH에 /workspace/AmbRes 가 포함되어 있는지 확인하세요.\n"
                f"원본 오류: {e}"
            ) from e

        if model_type == "finetune":
            adapter_ckpt = config.get("adapter_ckpt", "")
            if not adapter_ckpt:
                raise ValueError(
                    "model_type='finetune' 사용 시 'adapter_ckpt' 설정이 필요합니다."
                )
            self._model = AmbresFineTuned(
                adapter_ckpt=adapter_ckpt, use_detection=use_detection
            )
        else:
            self._model = AmbresFSPrompt(use_detection=use_detection)

        # fs_prompt: few-shot 예시를 messages에 세팅
        self._model.reset_chat()

        # SAM2 (선택)
        self._sam = None
        if use_sam:
            try:
                from ambres.sam2_model import Sam
                self._sam = Sam()
                logger.info("AmbResHandler: SAM2 loaded")
            except ImportError as e:
                logger.warning("SAM2 로드 실패 (sam2 패키지 미설치): %s", e)

        # 세션 상태: session_id → {"messages": list, "images": list[PIL.Image]}
        self._sessions: Dict[str, Dict] = {}

        # Molmo는 thread-safe하지 않으므로 추론 직렬화
        self._lock = threading.Lock()

        logger.info("AmbResHandler: ready (model_type=%s)", model_type)

    # ------------------------------------------------------------------ #
    # 세션 관리
    # ------------------------------------------------------------------ #

    def _load_session(self, session_id: str) -> None:
        """세션 상태(messages + images)를 모델에 복원."""
        if session_id and session_id in self._sessions:
            sess = self._sessions[session_id]
            self._model.messages = list(sess["messages"])
            self._model.images   = list(sess["images"])
        else:
            # 새 세션: few-shot 초기 messages 포함해서 초기화
            self._model.reset_chat()
            self._model.images = []

    def _save_session(self, session_id: str) -> None:
        """현재 모델 상태를 세션에 저장."""
        if session_id:
            self._sessions[session_id] = {
                "messages": list(self._model.messages),
                "images":   list(self._model.images),
            }

    # ------------------------------------------------------------------ #
    # 이미지 변환
    # ------------------------------------------------------------------ #

    def _to_pil(self, arr: np.ndarray, tag: str = "") -> Image.Image:
        """
        gRPC로 수신한 float32 [0,1] (H, W, 3) 텐서를 PIL RGB 이미지로 변환.
        generic_client.py의 _images_to_protos가 uint8→float32/255 변환해서 전송함.
        save_dir 설정 시 수신 이미지를 서버 로컬에 저장.
        """
        import time
        uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        pil = Image.fromarray(uint8, mode="RGB")
        if self._save_dir:
            import os
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{ts}_{tag}.png" if tag else f"{ts}.png"
            path = os.path.join(self._save_dir, fname)
            pil.save(path)
            logger.info("AmbResHandler: saved received image → %s", path)
        return pil

    # ------------------------------------------------------------------ #
    # 메인 핸들러
    # ------------------------------------------------------------------ #

    def handle(
        self,
        method:     str,
        payload:    Dict[str, Any],
        tensors:    List[np.ndarray],
        session_id: str,
    ) -> Dict[str, Any]:

        # ── reset ──────────────────────────────────────────────────────────
        if method == "reset":
            with self._lock:
                self._sessions.pop(session_id, None)
                self._model.reset_chat()
                self._model.images = []
            return self.ok({"message": "session reset"})

        # ── query ──────────────────────────────────────────────────────────
        elif method == "query":
            """
            payload:
                task_description (str): 로봇 작업 설명
            tensors[0]: 카메라 이미지 float32 [0,1] (H, W, 3)

            반환:
                task_objects     (list[str]): 작업에 필요한 객체 목록
                task_ambiguous   (bool):      모호성 여부
                clarifying_question (str):   모호할 경우 질문 (아니면 "")
            """
            if not tensors:
                return self.err("tensors[0] (이미지)가 필요합니다 — method='query'")

            task_description = payload.get("task_description", "")
            # AmbRes는 학습/평가 시 이미지를 4배 다운샘플하므로 동일하게 적용
            pil_img = self._to_pil(tensors[0], tag=f"query_{session_id}").reduce(4)

            with self._lock:
                self._load_session(session_id)
                result = self._model.handle_query(task_description, pil_img)
                self._save_session(session_id)

            # obj_detection_messages는 내부 Molmo 대화 기록이므로 제거
            result.pop("obj_detection_messages", None)
            return self.ok(result)

        # ── respond ────────────────────────────────────────────────────────
        elif method == "respond":
            """
            payload:
                response (str): 사용자의 명확화 답변

            반환:
                task_objects (list[str]): 모호성 해소 후 확정된 객체 목록
            """
            response_text = payload.get("response", "")

            with self._lock:
                self._load_session(session_id)
                result = self._model.handle_response(response_text)
                self._save_session(session_id)

            result.pop("obj_detection_messages", None)
            return self.ok(result)

        # ── detect ─────────────────────────────────────────────────────────
        elif method == "detect":
            """
            Molmo로 객체별 pixel 좌표를 검출합니다.
            query/respond와 독립적으로 호출 가능합니다.

            payload:
                objects (list[str]): 검출할 객체 이름 목록
            tensors[0]: 카메라 이미지 float32 [0,1] (H, W, 3)

            반환:
                detections (dict): {객체명: [[x, y], ...]} pixel 좌표
            """
            objects = payload.get("objects", [])
            if not objects:
                return self.err("payload에 'objects' 목록이 필요합니다 — method='detect'")
            if not tensors:
                return self.err("tensors[0] (이미지)가 필요합니다 — method='detect'")

            pil_img = self._to_pil(tensors[0], tag=f"detect_{session_id}")

            with self._lock:
                # detect_pretty는 self.images[0]만 사용하므로 직접 세팅
                self._model.images = [pil_img]
                result = self._model.detect_pretty(objects)

            return self.ok({"detections": result})

        # ── set_image_sam ──────────────────────────────────────────────────
        elif method == "set_image_sam":
            """
            SAM2의 이미지 인코딩을 미리 수행합니다.
            이후 query_mask 호출 시 이미지 재전송 없이 포인트만 보내면 됩니다.

            tensors[0]: 카메라 이미지 float32 [0,1] (H, W, 3)
            """
            if self._sam is None:
                return self.err("SAM2가 초기화되지 않았습니다. config에 use_sam: true를 설정하세요.")
            if not tensors:
                return self.err("tensors[0] (이미지)가 필요합니다 — method='set_image_sam'")

            rgb_uint8 = (np.clip(tensors[0], 0.0, 1.0) * 255).astype(np.uint8)
            with self._lock:
                self._sam.set_sam_image(rgb_uint8)
            return self.ok({"message": "SAM2 이미지 세팅 완료"})

        # ── query_mask ─────────────────────────────────────────────────────
        elif method == "query_mask":
            """
            set_image_sam으로 세팅된 이미지에서 포인트 기반 마스크를 추출합니다.

            payload:
                points (list[list[int]]): [[x, y], ...] pixel 좌표 (N개)
                labels (list[int], 선택): [1, 0, ...] 포지티브/네거티브 레이블
                                          생략 시 전부 포지티브(1)로 처리

            반환:
                mask (list[list[bool]]): (H, W) 마스크 (JSON 직렬화)
            """
            if self._sam is None:
                return self.err("SAM2가 초기화되지 않았습니다. config에 use_sam: true를 설정하세요.")

            points_raw = payload.get("points")
            if points_raw is None:
                return self.err("payload에 'points' 목록이 필요합니다 — method='query_mask'")

            pts = np.array(points_raw, dtype=np.float32)   # (N, 2)
            if pts.ndim != 2 or pts.shape[1] != 2:
                return self.err(f"'points'는 [[x, y], ...] 형식이어야 합니다. 받은 shape: {pts.shape}")

            labels_raw = payload.get("labels")
            labels = np.array(labels_raw, dtype=np.float32) if labels_raw else None

            with self._lock:
                mask = self._sam.query_sam_points(pts, labels)  # (1, H, W) bool

            return self.ok({"mask": mask[0].tolist()})

        # ── 미지원 메서드 ──────────────────────────────────────────────────
        else:
            supported = ["reset", "query", "respond", "detect", "set_image_sam", "query_mask"]
            return self.err(
                f"알 수 없는 method: {method!r}. 지원 목록: {supported}"
            )
