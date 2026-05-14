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

확장 방법:
    pre_handlers에 항목 추가만으로 새 VLM/핸들러 연결 가능.
    e.g. scene_classifier → ambres → action_model
"""

from __future__ import annotations

import logging
import time
import uuid
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

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
class PipelineConfig:
    """
    전체 파이프라인 설정.

    Attributes:
        action_model_id    : 서버에 로드된 액션 모델 ID
        fps                : 제어 루프 주기
        max_episode_steps  : 에피소드 당 최대 스텝 수 (0이면 무제한)
        pre_handlers       : 액션 루프 전에 실행되는 핸들러 단계 목록
    """
    action_model_id: str
    fps: float = 30.0
    max_episode_steps: int = 500
    pre_handlers: List[HandlerStepConfig] = field(default_factory=list)


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

    return PipelineConfig(
        action_model_id=raw["action_model_id"],
        fps=float(raw.get("fps", 30.0)),
        max_episode_steps=int(raw.get("max_episode_steps", 500)),
        pre_handlers=steps,
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
        2. 액션 루프 (fps 주기):
           a. 로봇에서 observation 수집
           b. every_step 핸들러 순차 실행
           c. context의 task_text로 액션 모델 추론
           d. 로봇에 액션 전송
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

        self._generic   = None   # GenericClient
        self._infer     = None   # InferenceClient (grpc_client.py)

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
        from module.desktop.grpc_client import InferenceClient

        self._generic = GenericClient(
            host=self._host, port=self._port,
            use_tls=self._use_tls, timeout=self._timeout,
        )
        self._generic.connect()

        self._infer = InferenceClient(
            host=self._host, port=self._port,
            use_tls=self._use_tls,
        )
        self._infer.connect()
        logger.info("Pipeline connected to %s:%d", self._host, self._port)

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

                    # ── episode_start 핸들러 ──────────────────────────────
                    obs = self._robot.get_observation()
                    for step in self._episode_start_steps:
                        self._run_step(step, obs, context, session_id)

                    logger.info("Episode %d  task_text='%s'", ep, context.get("task_text", ""))

                    # ── 액션 루프 ─────────────────────────────────────────
                    for step_idx in range(max_steps):
                        t0  = time.perf_counter()
                        obs = self._robot.get_observation()

                        # every_step 핸들러
                        for s in self._every_step_steps:
                            self._run_step(s, obs, context, session_id)

                        # 액션 추론
                        action = self._infer.get_action(
                            model_id=self._cfg.action_model_id,
                            images=obs.images,
                            state=obs.state,
                            task_text=context.get("task_text", ""),
                            episode_id=ep,
                            step=step_idx,
                        )
                        self._robot.send_action(action)

                        elapsed = time.perf_counter() - t0
                        if dt - elapsed > 0:
                            time.sleep(dt - elapsed)

                    ep += 1

            except KeyboardInterrupt:
                print(f"\n[pipeline] Stopped at episode {ep}")

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
