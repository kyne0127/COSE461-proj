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
    detect        : 이미지 + 객체 목록 → pixel 좌표 검출
                    (GroundingDINO 우선, 탐지 실패 시 Molmo fallback)
    set_image_sam : SAM2 이미지 사전 세팅
    query_mask    : pixel 좌표 points → 세그멘테이션 마스크

Config keys:
    model_type          (str):   "fs_prompt" | "finetune"  [default: "fs_prompt"]
    adapter_ckpt        (str):   finetune 체크포인트 이름 (model_type="finetune" 시 필수)
    use_detection       (bool):  query/respond 단계에서 Molmo detect 자동 수행 [default: False]
    use_sam             (bool):  SAM2 로드 여부 [default: False]
    use_gdino           (bool):  GroundingDINO 로드 여부 [default: False]
    gdino_model_id      (str):   HuggingFace 모델 ID [default: "IDEA-Research/grounding-dino-base"]
    gdino_box_threshold (float): bounding box confidence 최솟값 [default: 0.30]
    gdino_text_threshold(float): 텍스트 레이블 매칭 confidence 최솟값 [default: 0.25]
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from module.models.base_handler import BaseHandler, HandlerRegistry

logger = logging.getLogger(__name__)

# ── 추론 전용 파일 로거 ────────────────────────────────────────────────────────
_inference_log_path = os.environ.get(
    "AMBRES_INFERENCE_LOG",
    "/workspace/COSE461-proj/logs/ambres_inference.log",
)
os.makedirs(os.path.dirname(_inference_log_path), exist_ok=True)

_inf_logger = logging.getLogger("ambres.inference")
_inf_logger.setLevel(logging.DEBUG)
if not _inf_logger.handlers:
    _fh = logging.FileHandler(_inference_log_path, encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _inf_logger.addHandler(_fh)
    # ambres.ambres_model 로거도 같은 파일로 연결
    _model_logger = logging.getLogger("ambres.ambres_model")
    _model_logger.setLevel(logging.DEBUG)
    _model_logger.addHandler(_fh)


def _log_inference(session_id: str, method: str, **fields) -> None:
    """추론 이벤트를 구조화된 형식으로 기록."""
    lines = [f"{'─'*70}",
             f"[{method.upper()}]  session={session_id}"]
    for k, v in fields.items():
        v_str = str(v)
        # 긴 필드는 여러 줄로 분할
        if len(v_str) > 120:
            lines.append(f"  {k}:")
            for chunk in [v_str[i:i+110] for i in range(0, len(v_str), 110)]:
                lines.append(f"    {chunk}")
        else:
            lines.append(f"  {k}: {v_str}")
    _inf_logger.info("\n".join(lines))


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

        # GroundingDINO (선택) — detect() 메서드의 primary backend
        self._gdino = None
        use_gdino = config.get("use_gdino", False)
        if use_gdino:
            try:
                from module.models.ambres.gdino import GroundingDINO
                self._gdino = GroundingDINO(
                    model_id=config.get("gdino_model_id", "IDEA-Research/grounding-dino-base"),
                    device="cuda",
                    box_threshold=config.get("gdino_box_threshold", 0.30),
                    text_threshold=config.get("gdino_text_threshold", 0.25),
                )
                logger.info("AmbResHandler: GroundingDINO loaded (%s)",
                            config.get("gdino_model_id", "grounding-dino-base"))
            except Exception as e:
                logger.warning("GroundingDINO 로드 실패, detect()는 Molmo만 사용: %s", e)

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

    @staticmethod
    def _normalize_aspect(pil_img: Image.Image, target_ratio: float = 4 / 3) -> Image.Image:
        """
        이미지를 target_ratio(기본 4:3) 비율로 center-crop합니다.
        AmbRes 학습 데이터(640×480, 4:3)와 비율을 맞춰 도메인 갭을 줄입니다.
        ZED VGA(672×376, ~16:9) 같이 비율이 다른 카메라 이미지에 적용됩니다.
        이미 4:3에 가까운 이미지(오차 5% 이내)는 그대로 반환합니다.
        """
        w, h = pil_img.size
        current_ratio = w / h
        if abs(current_ratio - target_ratio) / target_ratio < 0.05:
            return pil_img
        if current_ratio > target_ratio:
            # 너무 넓음 → 좌우 crop
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            pil_img = pil_img.crop((left, 0, left + new_w, h))
        else:
            # 너무 좁음 → 상하 crop
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            pil_img = pil_img.crop((0, top, w, top + new_h))
        return pil_img

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

        # ── save_image ─────────────────────────────────────────────────────
        # 추론 없이 이미지만 저장 (데이터 수집 전용)
        if method == "save_image":
            if not tensors:
                return self.err("tensors[0] (이미지)가 필요합니다 — method='save_image'")
            tag = payload.get("tag", "capture")
            pil_img = self._to_pil(tensors[0], tag=tag)
            w, h = pil_img.size
            return self.ok({"saved": True, "width": w, "height": h})

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
            # 학습 데이터와 동일하게: ZED 672×376 원본 그대로 사용
            pil_img = self._to_pil(tensors[0], tag=f"query_{session_id}")

            t0 = time.time()
            with self._lock:
                self._load_session(session_id)
                result = self._model.handle_query(task_description, pil_img)

                # 로깅 전용: 객체별 pixel 좌표를 detect로 확인
                detections_log: dict = {}
                task_objects = result.get("task_objects") or []
                if task_objects:
                    try:
                        self._model.images = [pil_img]
                        detections_log = self._model.detect_pretty(task_objects)
                    except Exception as _e:
                        logger.warning("detect-for-logging failed: %s", _e)

                self._save_session(session_id)
            elapsed = time.time() - t0

            # obj_detection_messages는 내부 Molmo 대화 기록이므로 제거
            result.pop("obj_detection_messages", None)

            # ── 탐지 포인트 시각화 이미지 저장 ────────────────────────────
            if detections_log and self._save_dir:
                try:
                    from PIL import ImageDraw
                    palette = [
                        "#FF4444", "#44BB44", "#4488FF",
                        "#FFAA00", "#FF44FF", "#00CCCC",
                    ]
                    annotated = pil_img.copy()
                    draw = ImageDraw.Draw(annotated)
                    r = max(6, pil_img.width // 60)   # 점 반지름 (이미지 크기 비례)
                    for idx, (obj, pts) in enumerate(detections_log.items()):
                        color = palette[idx % len(palette)]
                        for (x, y) in pts:
                            x, y = int(x), int(y)
                            # 외곽 원 (객체 영역 표시)
                            draw.ellipse([x - r*2, y - r*2, x + r*2, y + r*2],
                                         outline=color, width=2)
                            # 중심점
                            draw.ellipse([x - r//2, y - r//2, x + r//2, y + r//2],
                                         fill=color)
                            # 레이블
                            draw.text((x + r*2 + 3, y - r), obj, fill=color)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    ann_path = os.path.join(
                        self._save_dir, f"{ts}_annotated_{session_id}.png"
                    )
                    annotated.save(ann_path)
                    logger.info("AmbResHandler: saved annotated image → %s", ann_path)
                except Exception as _e:
                    logger.warning("Annotation drawing failed: %s", _e)
            # ──────────────────────────────────────────────────────────────

            # ── 추론 과정 구조화 로그 ──────────────────────────────────────
            lines = [
                f"{'═'*70}",
                f"[QUERY]  session={session_id}",
                f"  task : {task_description}",
                f"  image: {pil_img.width}x{pil_img.height}",
                "",
                "  ┌─ STEP 1: Object Grounding (greedy)",
                f"  │  추출된 작업 객체: {task_objects}",
                "",
                "  ├─ STEP 2: Instance Detection (detect_pretty)",
            ]
            if detections_log:
                for idx, (obj, pts) in enumerate(detections_log.items()):
                    cnt = len(pts)
                    flag = "⚠ 복수 인스턴스!" if cnt > 1 else "✓ 단일"
                    coords = ", ".join(f"({int(x)},{int(y)})" for x, y in pts[:4])
                    if len(pts) > 4:
                        coords += f" ... +{len(pts)-4}개"
                    lines.append(f"  │    {obj!r:20s} → {cnt}개  {flag}")
                    lines.append(f"  │    {'':20s}   pts_raw: {pts!r}")
                    lines.append(f"  │    {'':20s}   coords: {coords}")
            else:
                lines.append("  │    (detect 실패 또는 객체 없음)")
            lines += [
                "",
                "  ├─ STEP 3: Ambiguity Reasoning (do_sample=True)",
                f"  │  ambiguous : {result.get('task_ambiguous')}",
                f"  │  explanation: {result.get('explanation', '')}",
                f"  │  question  : {result.get('clarifying_question', '') or '(없음)'}",
                "",
                f"  └─ 처리 시간: {elapsed:.2f}s",
                f"{'═'*70}",
            ]
            _inf_logger.info("\n".join(lines))
            # ────────────────────────────────────────────────────────────────

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

            t0 = time.time()
            with self._lock:
                self._load_session(session_id)
                result = self._model.handle_response(response_text)
                self._save_session(session_id)
            elapsed = time.time() - t0

            result.pop("obj_detection_messages", None)

            _log_inference(
                session_id, "respond",
                user_response=response_text,
                task_objects=result.get("task_objects"),
                task_ambiguous=result.get("task_ambiguous"),
                clarifying_question=result.get("clarifying_question", ""),
                explanation=result.get("explanation", ""),
                elapsed_s=f"{elapsed:.2f}",
            )
            return self.ok(result)

        # ── detect ─────────────────────────────────────────────────────────
        elif method == "detect":
            """
            객체별 pixel 좌표를 검출합니다.
            GroundingDINO(use_gdino=true)가 로드된 경우 primary backend로 사용하고,
            탐지 결과가 없거나 실패하면 Molmo로 fallback합니다.

            payload:
                objects (list[str]): 검출할 객체 이름 목록
            tensors[0]: 카메라 이미지 float32 [0,1] (H, W, 3)

            반환:
                detections (dict): {객체명: [[x, y], ...]} pixel 좌표
                backend    (str):  실제 사용된 backend ("gdino" | "molmo")
            """
            objects = payload.get("objects", [])
            if not objects:
                return self.err("payload에 'objects' 목록이 필요합니다 — method='detect'")
            if not tensors:
                return self.err("tensors[0] (이미지)가 필요합니다 — method='detect'")

            pil_img = self._to_pil(tensors[0], tag=f"detect_{session_id}")

            t0 = time.time()
            backend = "molmo"

            with self._lock:
                if self._gdino is not None:
                    try:
                        gdino_result = self._gdino.detect(pil_img, objects)
                        all_empty = all(len(v) == 0 for v in gdino_result.values())
                        if not all_empty:
                            result = gdino_result
                            backend = "gdino"
                        else:
                            logger.debug(
                                "GroundingDINO: 탐지 결과 없음 (objects=%s), Molmo fallback",
                                objects,
                            )
                            self._model.images = [pil_img]
                            result = self._model.detect_pretty(objects)
                    except Exception as _e:
                        logger.warning("GroundingDINO detect 실패 (%s), Molmo fallback", _e)
                        self._model.images = [pil_img]
                        result = self._model.detect_pretty(objects)
                else:
                    # detect_pretty는 self.images[0]만 사용하므로 직접 세팅
                    self._model.images = [pil_img]
                    result = self._model.detect_pretty(objects)

            elapsed = time.time() - t0

            _log_inference(
                session_id, "detect",
                backend=backend,
                objects=objects,
                image_size=f"{pil_img.width}x{pil_img.height}",
                detections=result,
                elapsed_s=f"{elapsed:.2f}",
            )
            return self.ok({"detections": result, "backend": backend})

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
