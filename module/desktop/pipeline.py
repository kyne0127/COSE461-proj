"""
module.desktop.pipeline
========================
범용 추론 파이프라인.

pre_handlers → action_loop 구조를 YAML 설정으로 정의합니다.
새 VLM/핸들러를 추가할 때 코드 변경 없이 YAML만 수정하면 됩니다.

핵심 개념:
    - HandlerStep : GenericInferenceService 핸들러 호출 한 단계
    - PipelineConfig: 전체 파이프라인 설정 (pre_handlers + action 모델)
    - InferencePipeline: 파이프라인 실행 엔진

YAML 설정 예시:
    action_model_id: pi0_main
    fps: 30.0
    pre_handlers:
      - handler_id: ambres
        method: query
        trigger: episode_start          # "episode_start" | "every_step"
        input_map:
          task_text: task_description   # context 키 → payload 키 매핑
        output_map:
          task_objects: task_text       # 결과 키 → context 키 매핑
        output_transform: join_list     # "identity" | "join_list" | "first"
        clarify_on: task_ambiguous      # 이 키가 True면 사용자 입력 요청
        clarify_prompt_key: clarifying_question
        clarify_method: respond

    checkpoint_monitor:
      enabled: true
      handler_id: ambres
      threshold: 50.0
      c1_step: 100      # 이 스텝에서 C1(pre-pick) 체크포인트 실행
      c2_step: 300      # 이 스텝에서 C2(pre-place) 체크포인트 실행

확장 방법:
    pre_handlers에 항목 추가만으로 새 VLM/핸들러 연결 가능.
    e.g. scene_classifier → ambres → action_model
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# src/ 패키지 경로 추가 (consistency_monitor, role_parser 사용)
_PIPELINE_DIR = Path(__file__).resolve().parent          # module/desktop
_REPO_ROOT    = _PIPELINE_DIR.parent.parent              # COSE461-proj
_SRC_ROOT     = _REPO_ROOT / "src"
for _p in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# 설정 dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class HandlerStepConfig:
    """
    파이프라인 한 단계의 설정.

    Attributes:
        handler_id      : GenericInferenceService에 등록된 핸들러 이름
        method          : 핸들러 내 메서드 이름 (e.g. "query", "classify")
        trigger         : 실행 시점 — "episode_start" | "every_step"
        input_map       : context 키 → payload 키 이름 변환
                          e.g. {"task_text": "task_description"}
        extra_payload   : 고정 payload 값 (YAML에서 하드코딩할 상수)
        output_map      : 결과 키 → context 키 이름 변환
                          e.g. {"task_objects": "task_text"}
        output_transform: 결과 값 변환 방식
                          "identity" : 그대로 사용
                          "join_list": list → 공백 구분 문자열
                          "first"    : list의 첫 번째 원소만 사용
        clarify_on      : 이 결과 키가 True이면 사용자 입력 단계 실행
        clarify_prompt_key : 사용자에게 보여줄 질문이 담긴 결과 키
        clarify_method  : 사용자 답변을 전달할 핸들러 메서드 이름
    """
    handler_id: str
    method: str
    trigger: str = "episode_start"
    input_map: Dict[str, str] = field(default_factory=dict)
    extra_payload: Dict[str, Any] = field(default_factory=dict)
    output_map: Dict[str, str] = field(default_factory=dict)
    output_transform: str = "identity"
    clarify_on: Optional[str] = None
    clarify_prompt_key: Optional[str] = None
    clarify_method: Optional[str] = None


@dataclass
class CheckpointMonitorConfig:
    """
    C1/C2 체크포인트 일관성 모니터 설정.

    Attributes:
        enabled    : 모니터 활성화 여부
        handler_id : gRPC AmbRes 핸들러 ID
        threshold  : G₀ 좌표 비교 픽셀 거리 임계값
        c1_step    : C1(pre-pick) 체크포인트를 실행할 action loop 스텝 번호 (0=비활성)
        c2_step    : C2(pre-place) 체크포인트를 실행할 action loop 스텝 번호 (0=비활성)
    """
    enabled: bool = False
    handler_id: str = "ambres"
    threshold: float = 50.0
    c1_step: int = 0
    c2_step: int = 0


@dataclass
class PipelineConfig:
    """
    전체 파이프라인 설정.

    Attributes:
        action_model_id       : 서버에 로드된 액션 모델 ID (서버 모드일 때 사용)
        local_model_type      : 로컬 모델 타입 키 (e.g. "smolvla"). 설정 시 서버 InferenceClient 미사용.
        local_model_checkpoint: 로컬 모델 체크포인트 경로 (빈 문자열이면 HF에서 다운로드)
        local_model_config    : 로컬 모델 설정 dict (ModelRegistry.build에 전달)
        fps                   : 제어 루프 주기
        max_episode_steps     : 에피소드 당 최대 스텝 수 (0이면 무제한)
        pre_handlers          : 액션 루프 전에 실행되는 핸들러 단계 목록
        checkpoint_monitor    : C1/C2 체크포인트 모니터 설정
    """
    action_model_id: str = ""
    local_model_type: str = ""
    local_model_checkpoint: str = ""
    local_model_config: Dict[str, Any] = field(default_factory=dict)
    fps: float = 30.0
    max_episode_steps: int = 500
    pre_handlers: List[HandlerStepConfig] = field(default_factory=list)
    checkpoint_monitor: CheckpointMonitorConfig = field(
        default_factory=CheckpointMonitorConfig
    )


def load_pipeline_config(path: str) -> PipelineConfig:
    """YAML 파일에서 PipelineConfig를 로드합니다."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    steps = []
    for h in raw.get("pre_handlers", []):
        steps.append(HandlerStepConfig(
            handler_id=h["handler_id"],
            method=h["method"],
            trigger=h.get("trigger", "episode_start"),
            input_map=h.get("input_map", {}),
            extra_payload=h.get("extra_payload", {}),
            output_map=h.get("output_map", {}),
            output_transform=h.get("output_transform", "identity"),
            clarify_on=h.get("clarify_on"),
            clarify_prompt_key=h.get("clarify_prompt_key"),
            clarify_method=h.get("clarify_method"),
        ))

    cm_raw = raw.get("checkpoint_monitor", {})
    cm_cfg = CheckpointMonitorConfig(
        enabled=cm_raw.get("enabled", False),
        handler_id=cm_raw.get("handler_id", "ambres"),
        threshold=float(cm_raw.get("threshold", 50.0)),
        c1_step=int(cm_raw.get("c1_step", 0)),
        c2_step=int(cm_raw.get("c2_step", 0)),
    )

    return PipelineConfig(
        action_model_id=raw.get("action_model_id", ""),
        local_model_type=raw.get("local_model_type", ""),
        local_model_checkpoint=raw.get("local_model_checkpoint", ""),
        local_model_config=raw.get("local_model_config", {}),
        fps=float(raw.get("fps", 30.0)),
        max_episode_steps=int(raw.get("max_episode_steps", 500)),
        pre_handlers=steps,
        checkpoint_monitor=cm_cfg,
    )


# ────────────────────────────────────────────────────────────────────────────
# 파이프라인 실행 엔진
# ────────────────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    범용 추론 파이프라인.

    실행 순서 (에피소드마다):
        1. episode_start 핸들러 순차 실행
           - 결과를 context dict에 누적
           - clarify_on 조건이 True이면 사용자 입력 요청 후 clarify_method 호출
        2. [checkpoint_monitor 활성 시] G₀ 추출 (t₀ 이미지 → AmbRes detect)
        3. 액션 루프 (fps 주기):
           a. 로봇에서 observation 수집
           b. every_step 핸들러 순차 실행
           c. [c1_step 도달 시] C1 체크포인트 — G₀ 일관성 확인 → CONTINUE/ASK/STOP
           d. [c2_step 도달 시] C2 체크포인트 — G₀ 일관성 확인 → CONTINUE/ASK/STOP
           e. context의 task_text로 액션 모델 추론
           f. 로봇에 액션 전송
    """

    def __init__(
        self,
        pipeline_cfg: PipelineConfig,
        robot_connector,
        grpc_host: str = "localhost",
        grpc_port: int = 50051,
        use_tls: bool = False,
        timeout: float = 60.0,
    ) -> None:
        self._cfg       = pipeline_cfg
        self._robot     = robot_connector
        self._host      = grpc_host
        self._port      = grpc_port
        self._use_tls   = use_tls
        self._timeout   = timeout
        self._cm_cfg    = pipeline_cfg.checkpoint_monitor

        self._generic      = None   # GenericClient
        self._infer        = None   # InferenceClient (grpc_client.py)
        self._local_model  = None   # BaseLeRobotModel (local, e.g. SmolVLA)

        self._episode_start_steps = [
            s for s in pipeline_cfg.pre_handlers if s.trigger == "episode_start"
        ]
        self._every_step_steps = [
            s for s in pipeline_cfg.pre_handlers if s.trigger == "every_step"
        ]

    # ------------------------------------------------------------------ #
    # 연결 관리
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        from module.desktop.generic_client import GenericClient

        self._generic = GenericClient(
            host=self._host, port=self._port,
            use_tls=self._use_tls, timeout=self._timeout,
        )
        self._generic.connect()

        if self._cfg.local_model_type:
            from module.utils.registry import ModelRegistry
            self._local_model = ModelRegistry.build(
                self._cfg.local_model_type,
                self._cfg.local_model_config,
            )
            self._local_model.load_checkpoint(self._cfg.local_model_checkpoint or "")
            logger.info(
                "Local model '%s' loaded on device=%s",
                self._cfg.local_model_type,
                self._cfg.local_model_config.get("device", "cuda"),
            )
        else:
            from module.desktop.grpc_client import InferenceClient
            self._infer = InferenceClient(
                host=self._host, port=self._port,
                use_tls=self._use_tls,
            )
            self._infer.connect()
            logger.info("Pipeline connected to %s:%d (server inference)", self._host, self._port)

    def disconnect(self) -> None:
        if self._generic:
            self._generic.disconnect()
        if self._infer:
            self._infer.disconnect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------ #
    # 메인 실행 루프
    # ------------------------------------------------------------------ #

    def run(
        self,
        task_text: str = "",
        n_episodes: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        """
        Args:
            task_text         : 기본 작업 설명 (episode_start 핸들러가 덮어쓸 수 있음)
            n_episodes        : 실행할 에피소드 수 (None이면 Ctrl+C까지 무한 반복)
            max_episode_steps : 에피소드 당 최대 스텝 수 (None이면 config 값 사용)
        """
        dt = 1.0 / self._cfg.fps
        max_steps = max_episode_steps if max_episode_steps is not None else self._cfg.max_episode_steps
        ep = 0

        with self._robot.session():
            try:
                while n_episodes is None or ep < n_episodes:
                    session_id = f"ep_{ep}_{uuid.uuid4().hex[:6]}"
                    context: Dict[str, Any] = {"task_text": task_text}

                    logger.info("=== Episode %d  session=%s ===", ep, session_id)

                    # ── episode_start: 모델 리셋 + 핸들러 실행 ───────────
                    if self._local_model is not None:
                        self._local_model.reset()

                    obs = self._robot.get_observation()
                    for step in self._episode_start_steps:
                        self._run_step(step, obs, context, session_id)

                    logger.info("Episode %d  task_text='%s'", ep, context.get("task_text", ""))

                    # ── G₀ 추출 (checkpoint_monitor 활성 시) ─────────────
                    if self._cm_cfg.enabled:
                        task_desc = context.get("task_text", task_text)
                        g0 = self._extract_g0_grpc(obs, task_desc, f"{session_id}_g0")
                        if g0 is not None:
                            context["_g0"] = g0
                            logger.info(
                                "G₀ extracted: target=%s@%s  dest=%s@%s",
                                g0["target"]["label"], g0["target"]["coord"],
                                g0["destination"]["label"], g0["destination"]["coord"],
                            )
                        else:
                            logger.warning(
                                "G₀ extraction failed — checkpoint monitor disabled this episode"
                            )

                    # ── 액션 루프 ─────────────────────────────────────────
                    episode_aborted = False
                    for step_idx in range(max_steps):
                        t0  = time.perf_counter()
                        obs = self._robot.get_observation()

                        # every_step 핸들러
                        for s in self._every_step_steps:
                            self._run_step(s, obs, context, session_id)

                        # C1 체크포인트
                        if (self._cm_cfg.enabled
                                and self._cm_cfg.c1_step > 0
                                and step_idx == self._cm_cfg.c1_step
                                and "_g0" in context):
                            aborted = self._run_checkpoint(
                                "C1", obs, context, task_text, session_id
                            )
                            if aborted:
                                episode_aborted = True
                                break

                        # C2 체크포인트
                        if (self._cm_cfg.enabled
                                and self._cm_cfg.c2_step > 0
                                and step_idx == self._cm_cfg.c2_step
                                and "_g0" in context):
                            aborted = self._run_checkpoint(
                                "C2", obs, context, task_text, session_id
                            )
                            if aborted:
                                episode_aborted = True
                                break

                        if episode_aborted:
                            break

                        # 액션 추론 (로컬 모델 or 서버 모델)
                        current_task = context.get("task_text", task_text)
                        if self._local_model is not None:
                            from module.models.base_model import Observation as ModelObs
                            obs_obj = ModelObs(
                                images=obs.images,
                                state=obs.state,
                                task_text=current_task,
                                episode_id=ep,
                                step=step_idx,
                            )
                            action = self._local_model.predict_action(obs_obj)
                        else:
                            action = self._infer.get_action(
                                model_id=self._cfg.action_model_id,
                                images=obs.images,
                                state=obs.state,
                                task_text=current_task,
                                episode_id=ep,
                                step=step_idx,
                            )
                        self._robot.send_action(action)

                        elapsed = time.perf_counter() - t0
                        if dt - elapsed > 0:
                            time.sleep(dt - elapsed)

                    if episode_aborted:
                        logger.info("Episode %d aborted by checkpoint monitor", ep)
                    ep += 1

            except KeyboardInterrupt:
                print(f"\n[pipeline] Stopped at episode {ep}")

    # ------------------------------------------------------------------ #
    # C1/C2 체크포인트 모니터
    # ------------------------------------------------------------------ #

    def _run_checkpoint(
        self,
        checkpoint: str,
        obs,
        context: Dict[str, Any],
        task_text: str,
        session_id: str,
    ) -> bool:
        """
        C1 또는 C2 체크포인트에서 G₀ 일관성을 확인합니다.

        Returns:
            True  — 에피소드를 중단해야 함 (STOP 결정)
            False — 계속 진행 (CONTINUE 또는 ASK 후 사용자 답변 반영)
        """
        from monitoring.consistency_monitor import Decision, check_grounding

        g0 = context["_g0"]
        role = "target" if checkpoint == "C1" else "destination"

        # 현재 체크포인트에서 detections 수집
        detections = self._detect_grpc(obs, g0, f"{session_id}_{checkpoint.lower()}")
        state, decision = check_grounding(g0, detections, checkpoint, self._cm_cfg.threshold)

        logger.info("[%s] state=%s  decision=%s", checkpoint, state.value, decision.value)
        print(f"\n[{checkpoint}] {state.value} → {decision.value}")

        if decision == Decision.STOP:
            print(f"[{checkpoint} STOP] {state.value} — 로봇을 정지합니다.")
            logger.warning("[%s STOP] %s", checkpoint, state.value)
            return True  # abort episode

        if decision == Decision.ASK:
            question = self._clarifying_question(g0, role, state.value)
            print(f"[{checkpoint} ASK] {question}")
            try:
                user_resp = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                user_resp = ""

            if user_resp:
                task_desc = context.get("task_text", task_text)
                updated_g0 = self._update_g0_grpc(
                    obs, task_desc, user_resp,
                    f"{session_id}_{checkpoint.lower()}_update",
                )
                if updated_g0 is not None:
                    context["_g0"] = updated_g0
                    logger.info(
                        "[%s] G₀ updated: target=%s@%s  dest=%s@%s",
                        checkpoint,
                        updated_g0["target"]["label"], updated_g0["target"]["coord"],
                        updated_g0["destination"]["label"], updated_g0["destination"]["coord"],
                    )
            else:
                logger.info("[%s] 사용자 답변 없음 — G₀ 유지", checkpoint)

        return False  # continue episode

    def _extract_g0_grpc(
        self,
        obs,
        task_description: str,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        t₀ observation에서 G₀를 추출합니다 (gRPC AmbRes 경유).

        순서: reset → query → respond("") → detect → G₀ dict 반환
        """
        hid = self._cm_cfg.handler_id

        self._generic.infer(hid, "reset", payload={}, session_id=session_id)

        step1 = self._generic.infer(
            hid, "query",
            payload={"task_description": task_description},
            images=obs.images,
            session_id=session_id,
        )
        if step1.get("task_ambiguous"):
            logger.warning("G₀ extraction: t₀ ambiguous — skipping")
            return None

        step2 = self._generic.infer(
            hid, "respond",
            payload={"response": ""},
            session_id=session_id,
        )

        object_list = step2.get("task_objects") or step1.get("task_objects") or []
        if isinstance(object_list, str):
            object_list = [o.strip() for o in object_list.split(",") if o.strip()]
        if not object_list:
            logger.warning("G₀ extraction: no task_objects returned")
            return None

        try:
            from extraction.role_parser import parse_roles
            roles = parse_roles(object_list, task_description)
        except Exception as exc:
            logger.warning("G₀ extraction: role_parser failed: %s", exc)
            return None

        labels = [roles["target"], roles["destination"]]
        detect_result = self._generic.infer(
            hid, "detect",
            payload={"objects": labels},
            images=obs.images,
            session_id=session_id,
        )
        detections = detect_result.get("detections", {})

        return {
            "target": {
                "label": roles["target"],
                "coord": self._first_coord(detections, roles["target"]),
            },
            "destination": {
                "label": roles["destination"],
                "coord": self._first_coord(detections, roles["destination"]),
            },
        }

    def _detect_grpc(
        self,
        obs,
        g0: Dict[str, Any],
        session_id: str,
    ) -> Dict[str, list]:
        """
        체크포인트 이미지에서 G₀ 라벨들을 detect합니다.

        Returns:
            detections_all: {"label": [[x,y], ...]} 형식 (all valid coords)
        """
        hid = self._cm_cfg.handler_id
        labels = [g0["target"]["label"], g0["destination"]["label"]]

        result = self._generic.infer(
            hid, "detect",
            payload={"objects": labels},
            images=obs.images,
            session_id=session_id,
        )
        raw = result.get("detections", {})

        detections_all: Dict[str, list] = {}
        for label, coords in raw.items():
            valid = []
            for c in (coords or []):
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    try:
                        valid.append([float(c[0]), float(c[1])])
                    except (TypeError, ValueError):
                        continue
            if valid:
                detections_all[label] = valid

        return detections_all

    def _update_g0_grpc(
        self,
        obs,
        task_description: str,
        user_response: str,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        사용자 답변으로 G₀를 rolling update합니다 (gRPC AmbRes 경유).

        순서: reset → query → respond(user_response) → detect → 새 G₀ 반환
        """
        hid = self._cm_cfg.handler_id

        self._generic.infer(hid, "reset", payload={}, session_id=session_id)

        step1 = self._generic.infer(
            hid, "query",
            payload={"task_description": task_description},
            images=obs.images,
            session_id=session_id,
        )
        step2 = self._generic.infer(
            hid, "respond",
            payload={"response": user_response},
            session_id=session_id,
        )

        object_list = step2.get("task_objects") or step1.get("task_objects") or []
        if isinstance(object_list, str):
            object_list = [o.strip() for o in object_list.split(",") if o.strip()]
        if not object_list:
            logger.warning("G₀ rolling update: no task_objects returned")
            return None

        try:
            from extraction.role_parser import parse_roles
            roles = parse_roles(object_list, task_description)
        except Exception as exc:
            logger.warning("G₀ rolling update: role_parser failed: %s", exc)
            return None

        labels = [roles["target"], roles["destination"]]
        detect_result = self._generic.infer(
            hid, "detect",
            payload={"objects": labels},
            images=obs.images,
            session_id=session_id,
        )
        detections = detect_result.get("detections", {})

        return {
            "target": {
                "label": roles["target"],
                "coord": self._first_coord(detections, roles["target"]),
            },
            "destination": {
                "label": roles["destination"],
                "coord": self._first_coord(detections, roles["destination"]),
            },
        }

    # ------------------------------------------------------------------ #
    # 핸들러 단계 실행
    # ------------------------------------------------------------------ #

    def _run_step(
        self,
        step: HandlerStepConfig,
        obs,
        context: Dict[str, Any],
        session_id: str,
    ) -> None:
        """
        핸들러 한 단계를 실행하고 결과를 context에 반영합니다.
        clarify_on 조건이 True이면 사용자 입력 후 clarify_method를 추가 호출합니다.
        """
        payload = self._build_payload(step, context)
        result  = self._generic.infer(
            handler_id=step.handler_id,
            method=step.method,
            payload=payload,
            images=obs.images,
            session_id=session_id,
        )
        self._apply_output(step, result, context)

        # ── 명확화 흐름 (AmbRes처럼 모호성 감지 시) ──────────────────────
        if step.clarify_on and result.get(step.clarify_on):
            prompt = result.get(step.clarify_prompt_key, "Please clarify:")
            print(f"\n[pipeline:{step.handler_id}] {prompt}")
            user_response = input("> ").strip()

            clarify_result = self._generic.infer(
                handler_id=step.handler_id,
                method=step.clarify_method,
                payload={"response": user_response},
                session_id=session_id,
            )
            self._apply_output(step, clarify_result, context)

    # ------------------------------------------------------------------ #
    # 내부 헬퍼
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_payload(
        step: HandlerStepConfig,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        context + input_map + extra_payload 를 합쳐 payload를 구성합니다.

        input_map: context 키를 다른 이름으로 payload에 추가
            e.g. input_map: {task_text: task_description}
                 → payload["task_description"] = context["task_text"]
        extra_payload: 고정 값 (config에서 하드코딩)
        """
        payload: Dict[str, Any] = {}
        for ctx_key, payload_key in step.input_map.items():
            if ctx_key in context:
                payload[payload_key] = context[ctx_key]
        payload.update(step.extra_payload)
        return payload

    @staticmethod
    def _apply_output(
        step: HandlerStepConfig,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """
        result 키를 output_map에 따라 context에 저장하고
        output_transform을 적용합니다.
        """
        for result_key, ctx_key in step.output_map.items():
            value = result.get(result_key)
            if value is None:
                continue

            if step.output_transform == "join_list" and isinstance(value, list):
                value = " ".join(str(v) for v in value)
            elif step.output_transform == "first" and isinstance(value, list):
                value = value[0] if value else ""

            context[ctx_key] = value
            logger.debug(
                "Pipeline context update: %s = %r  (from %s.%s)",
                ctx_key, value, step.handler_id, step.method,
            )

    @staticmethod
    def _first_coord(
        detections: Dict[str, list],
        label: str,
    ) -> Optional[list]:
        """detections에서 label의 첫 번째 유효 좌표를 int [x, y]로 반환합니다."""
        for c in detections.get(label, []):
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                try:
                    return [int(float(c[0])), int(float(c[1]))]
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _clarifying_question(g0: Dict[str, Any], role: str, state_value: str) -> str:
        """사용자에게 보여줄 명확화 질문을 생성합니다."""
        label = g0[role]["label"]
        templates = {
            "target":      f"어떤 '{label}'을(를) 집어야 하나요?",
            "destination": f"어떤 '{label}'에 놓아야 하나요?",
        }
        question = templates.get(role, f"'{label}'에 대해 명확히 해주세요.")
        return f"[{state_value}] {question}"
