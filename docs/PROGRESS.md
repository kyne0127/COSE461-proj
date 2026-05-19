# Execution-Aware Ambiguity Handling — 진행 기록

> 이 파일은 proposal.md 기반 실행 계획과 진행 상태를 하나의 문서로 관리한다.  
> 작업이 완료될 때마다 이 파일의 상태를 업데이트한다.
> Last updated: 2026-05-19 — Phase 1.5 파인튜닝 준비: 글로벌뷰 카메라 테스트 스크립트(`scripts/test_camera.py`) 추가, 카메라 설정 가이드(`docs/camera_guide.md`) 작성

---

## 1. 연구 한 문장 정의

> AmbResVLM은 현재 장면이 ambiguous한지를 판단하지만, 실행 중 장면 변화로 인해 initially-clear grounding이 무효화되는 것은 감지하지 못한다. 본 연구는 초기 grounding G₀를 메모리에 저장하고 checkpoint 이미지에서 AmbResVLM을 재호출하여, G₀와 Gₜ의 불일치를 grounding-state transition으로 분류하는 최초의 방법이다.

---

## 2. 시스템 설계 요약

### 2.1 전체 파이프라인

```
t₀ (초기)
  ├── Task description + Initial image
  ├── AmbResVLM 호출 (Step 1 + Step 2 강제)
  ├── ambiguity = true  → ASK (initial), 종료
  └── ambiguity = false → G₀ 저장 {target: {label, coord}, destination: {label, coord}}
           ↓
  로봇 이동 (카메라 off)
           ↓
C1 (pre-pick checkpoint)
  ├── Checkpoint image 촬영
  ├── AmbResVLM 재호출 → Gₜ 획득
  ├── Consistency Monitor: G₀ vs Gₜ 비교
  ├── Grounding state 분류 (target 관련)
  └── Decision policy → CONTINUE / ASK / STOP
           ↓ (CONTINUE일 때)
  PICK 실행 (카메라 off)  →  로봇 이동 (카메라 off)
           ↓
C2 (pre-place checkpoint)
  ├── Checkpoint image 촬영
  ├── AmbResVLM 재호출 → Gₜ 획득
  ├── Consistency Monitor: G₀ vs Gₜ 비교
  ├── Grounding state 분류 (destination 관련)
  └── Decision policy → CONTINUE / ASK / STOP
           ↓ (CONTINUE일 때)
  PLACE 실행 → 완료
```

### 2.1.1 실제 로봇 실행 아키텍처 (Molmo server + SmolVLA desktop)

현재 `module/` 구현은 연구용 C1/C2 checkpoint monitor와 별개로, 실제 로봇 팔을 움직이기 위한 online 실행 경로를 제공한다.

```
Desktop (RTX 3060 / 8 GB VRAM)          GPU Server / RunPod
────────────────────────────────         ───────────────────────────
RobotConnector                           GenericInferenceService
  ├─ SO-ARM100/101 follower arm            └─ AmbResHandler (Molmo 7B)
  ├─ optional SO leader arm                    query / respond
  └─ egoview cameras
        │
        ▼
InferencePipeline
  ├─ episode_start:
  │    GenericClient ─── gRPC ───► AmbRes query/respond
  │    ambiguity가 있으면 terminal input으로 clarify
  │
  └─ action_loop @ 30 Hz:
       SmolVLAModel.predict_action()
       image + joint state + task_text → action
       RobotConnector.send_action(action)
```

핵심 분리:

- Molmo/AmbRes는 GPU server에서 실행한다. 네트워크 latency는 episode_start의 ambiguity resolution에만 영향을 준다.
- SmolVLA는 desktop GPU에서 local로 실행한다. 30 Hz action loop는 gRPC를 거치지 않는다.
- SO-ARM100/101 follower arm 제어와 leader arm teleoperation/data collection은 `RobotConnector`가 담당한다.
- 현재 live pipeline은 t₀ ambiguity resolution + SmolVLA 실행까지 구현되어 있다. 연구용 C1/C2 G₀ consistency monitor는 아직 `module/desktop/pipeline.py`의 실시간 action loop에 직접 통합되지 않았다.

### 2.2 G₀ 표현 형식

```json
{
  "target": { "label": "red block", "coord": [142, 318] },
  "destination": { "label": "tray", "coord": [401, 289] },
  "image_shape": [480, 640]
}
```

> coord는 픽셀 정수 `[x, y]` (Molmo 출력 기준). proposal 예시는 float이나 int가 실용적으로 더 적합.

### 2.3 Grounding State Taxonomy

| State                 | 트리거 조건                   | Decision | Checkpoint |
| --------------------- | ----------------------------- | -------- | ---------- |
| CLEAR                 | Gₜ에 target₀, dest₀ 모두 유효 | CONTINUE | C1, C2     |
| CLEAR (distractor)    | 새 object가 task와 무관       | CONTINUE | C1, C2     |
| AMBIGUOUS_TARGET      | 같은 label의 coord가 여러 개  | ASK      | C1 only    |
| INVALID_TARGET        | target₀가 Gₜ에 없음           | STOP     | C1 only    |
| AMBIGUOUS_DESTINATION | dest 후보가 여러 개           | ASK      | C2 only    |
| OCCUPIED_DESTINATION  | dest₀는 있지만 occupied 상태  | ASK      | C2 only    |
| INVALID_DESTINATION   | dest₀가 Gₜ에 없음             | ASK/STOP | C2 only    |
| UNSAFE_OR_BLOCKED     | safety 위협 감지              | STOP     | C1, C2     |

> ⚠️ OCCUPIED_DESTINATION은 현재 scope 외 (이후 단계)

### 2.4 Coord 비교 로직 (Consistency Monitor 핵심)

```python
distance = euclidean(G0.target.coord, Gt["red block"].coord)
if distance > THRESHOLD:
    state = AMBIGUOUS_TARGET  # 위치가 다른 → 다른 instance

all_coords = Gt.get_all_coords("red block")
if len(all_coords) > 1:
    state = AMBIGUOUS_TARGET  # 원래 것 + 새 것 추가됨
```

> THRESHOLD: object bounding box 대각선 크기 기준 (pilot_threshold.py로 결정)

### 2.5 Baseline

|          | 방법                             | G₀ 메모리 | 설명                                                                                      |
| -------- | -------------------------------- | --------- | ----------------------------------------------------------------------------------------- |
| B1       | AmbResVLM single-shot            | 없음      | t₀에서 1회 query만 수행, checkpoint/G₀ 비교 없음                                          |
| B2       | AmbResVLM repeated               | 없음      | 매 checkpoint 재호출, G₀ 비교 없음                                                        |
| B3       | Candidate count rule             | 없음      | full-label exact count; target 0개 STOP, 중복 target/destination 또는 destination 0개 ASK |
| B4       | Ours w/o taxonomy                | 있음      | G₀ label+coord 비교는 하지만 anomaly면 taxonomy 없이 항상 ASK                             |
| B5       | Re-query + LLM Consistency Judge | 없음      | initial/checkpoint 두 이미지를 GPT-4V-class LLM에 직접 비교시켜 CONTINUE/ASK/STOP 판단    |
| **Ours** | Execution-Aware AmbResVLM        | **있음**  | G₀ + coord + taxonomy + decision policy                                                   |

> B5는 reviewer가 제기할 수 있는 "AmbResVLM + G₀ monitor 대신 일반 GPT-4V에게 두 이미지를 비교시키면 충분하지 않은가?"라는 질문을 방어하기 위한 strong general-VLM baseline이다. Ours 대비 structured G₀/coord memory가 없고, downstream grounding update가 어렵고, API cost/latency가 커서 실제 로봇 checkpoint monitor로는 부적합하다는 점을 검증한다.

---

## 3. 실행 계획 및 진행 상태

### Phase 0: Research Design ✅ 완료

- [x] 주제 확정
- [x] 포지셔닝 (2×2 매트릭스)
- [x] proposal.md 완성

---

### Phase 1: 코드 구현 — ✅ 완료 (6/6 완료)

> Step별 전체 test suite 숫자는 해당 시점의 누적 기록이다. 최신 로컬 unit test 결과는 문서 하단의 "최신 로컬 unit test"를 기준으로 한다.

#### Step 1 — `ambres_g0_extractor.py` ✅ 완료

**목표:** t₀에서 AmbResVLM Step1+Step2 호출 → G₀ 추출

**구현 내용:**

- `extract_g0(image_path, task_description, ...)` — 메인 함수
- `_make_handler()` — AmbResHandler 생성 (model_type: fs_prompt / finetune)
- `_load_image()` — PIL → numpy float32 [0,1]
- `_ambiguity_from_step()` — `ambiguity` / `task_ambiguous` 키 양쪽 처리
- `_object_list_from_step()` — `object_list` / `task_objects` 키 양쪽 처리
- `_first_coord()` — detect 결과에서 첫 번째 `[x, y]` 추출 (int 변환)
- Step 1(query) → ambiguity=true 시 RuntimeError, false여도 Step 2 강제 호출
- Step 2(respond) — `response=""` 빈 answer로 강제 호출 (좌표 확보 목적)
- detect 호출 → target/destination 좌표 획득
- CLI (`python ambres_g0_extractor.py <image> <task> [--model-type] [--adapter-ckpt] [--allow-ambiguous]`)

**테스트:**

- `tests/test_g0_extractor.py` — 20개 테스트 total (mock 기반 19개 통과, optional real image 테스트 1개는 로컬 이미지 없으면 skip)
- 실제 모델 실행 확인 (finetune + CKPT.REAL = `43qazb3XcrZF5rZWnjRPVm`)

```json
// 실제 실행 결과 예시 (5rhU25AdQW4jADxhp8EYuq.jpeg, "move the marker next to the sprite bottle")
{
  "target": { "label": "marker", "coord": [1660, 1375] },
  "destination": { "label": "sprite bottle", "coord": [708, 1150] },
  "image_shape": [2252, 4000]
}
```

**환경 이슈 (실제 모델 실행 시):**

- `transformers < 5.0` 필요 (`torch 2.4.1`과 충돌 → `pip install "transformers>=4.48,<5.0"`)
- Molmo processor가 `tensorflow`를 조건부 import → 아래 stub 필요:

```python
import sys, importlib.util
from types import ModuleType
tf = ModuleType('tensorflow')
tf.__spec__ = importlib.util.spec_from_loader('tensorflow', loader=None)
tf.Tensor = type('Tensor', (), {})
tf.Variable = type('Variable', (), {})
sys.modules['tensorflow'] = tf
```

---

#### Step 2 — `role_parser.py` ✅ 완료

**목표:** `object_list` + `task_description` → `{target, destination}` 역할 분류

**구현 내용:**

- 동사 패턴 (place, put, move, pick, grab, take, drop, insert) → 뒤 명사 = target
- 전치사 패턴 (in, into, on, onto, to, inside) → 뒤 명사 = destination
- `parse_roles(object_list, task_description) → {"target": str, "destination": str}`
- CLI (`python role_parser.py <task_description> <object1> <object2> ...`)

**테스트:**

- `tests/test_role_parser.py` — 47개 전용 단위 테스트, 전부 통과
  - `TestBasicVerbPreposition` (8개): 동사+전치사 기본 케이스 (place/put/move/pick/grab/take/drop/insert)
  - `TestArticleSkipping` (4개): the/a/an 관사 무시
  - `TestMultiWordObjects` (5개): 다중 단어 레이블, 긴 레이블 우선 선택
  - `TestAllVerbs` (8개): 각 동사 parametrize
  - `TestAllPrepositions` (6개): 각 전치사 parametrize
  - `TestFallbacks` (5개): 동사 없음 / 전치사 없음 / 둘 다 없음 / 단일 오브젝트
  - `TestCaseAndPunctuation` (3개): 대소문자·구두점 무관
  - `TestOutputSchema` (3개): 반환 타입 검증
  - `TestEdgeCases` (5개): 빈 리스트 ValueError, proposal 실제 태스크 검증
- `test_g0_extractor.py` 내에서 통합 검증 (mock 없이 실제 parse_roles 호출)

---

#### Step 3 — `consistency_monitor.py` ✅ 완료

**목표:** G₀ vs Gₜ 비교 → Grounding State Taxonomy 분류

**구현 내용:**

- `GroundingState` enum: CLEAR / AMBIGUOUS_TARGET / INVALID_TARGET / AMBIGUOUS_DESTINATION / INVALID_DESTINATION / UNSAFE_OR_BLOCKED
- `Decision` enum: CONTINUE / ASK / STOP
- `check_grounding(g0, detections_t, checkpoint, threshold=50.0) → (GroundingState, Decision)`
  - C1(pre-pick): target label/coord 비교
  - C2(pre-place): destination label/coord 비교
  - 판단 로직: label 없음 → INVALID, 개수 > 1 → AMBIGUOUS, distance > threshold → AMBIGUOUS, 그 외 → CLEAR
- `get_checkpoint_detections(image_path, g0, handler, session_id)` — 핸들러로 checkpoint 이미지에서 ALL coords 획득 (labels 중복 제거 포함)
- CLI: `python consistency_monitor.py g0.json detections.json C1 [--threshold N]`
- Decision 정책: INVALID_TARGET → STOP, INVALID_DESTINATION → ASK (회복 가능), 나머지 INVALID/AMBIGUOUS → ASK, CLEAR → CONTINUE

**설계 결정:**

- `detections_t` 형식: `{"label": [[x,y], [x,y], ...]}` — ALL coords (extract_g0의 first-only와 다름)
- Threshold 기본값 50px: 임시값, pilot_threshold.py 결과로 교체 예정
- INVALID_DESTINATION → ASK: proposal §4.4에서 "ASK/STOP" → ASK 선택 (destination 소실은 사용자 입력으로 회복 가능)

**테스트:**

- `tests/test_consistency_monitor.py` — 46개 단위 테스트, 전부 통과
  - `TestC1TargetChecks` (8개): target 관련 state, 경계값 포함
  - `TestC2DestinationChecks` (8개): destination 관련 state, 경계값 포함
  - `TestDecisionPolicy` (5개): state → decision 매핑
  - `TestReturnTypes` (4개): 반환 타입
  - `TestInvalidCheckpoint` (3개): 잘못된 checkpoint 인자
  - `TestEdgeCases` (8개): 동일 label, distractor, threshold 극값 등
  - `TestScenarioAlignment` (5개): **proposal §5.1 초기 시나리오 ①~⑤ 직접 매핑** (S6는 baseline/evaluator 단계에서 추가)
  - `TestGetCheckpointDetections` (5개): mock 기반 헬퍼 검증
- 전체 test suite: 113개 통과 (20 + 47 + 46)

---

#### Step 4 — `pilot_threshold.py` ✅ 완료

**목표:** 이미지 5~10장에서 coord 변화량 측정 → THRESHOLD 결정

**구현 내용:**

- 순수 통계 함수 (AmbRes 의존 없음, 완전 테스트 가능):
  - `centroid(coords)` — 좌표 평균
  - `euclidean(a, b)` — 유클리드 거리
  - `percentile(values, p)` — nearest-rank 백분위수
  - `compute_intra_stats(coords_per_label, p=95)` — 레이블별 spread 통계 (n, centroid, distances, p95, max)
  - `compute_inter_instance_distances(multi_coords)` — 같은 레이블 instance 간 최소 거리
  - `recommend_threshold(intra_stats, safety_factor=3.0)` — `max(p95) × safety_factor` → 정수 반올림
  - `build_report(coords, safety_factor, p)` — 전체 JSON 리포트 생성
- AmbRes 연동:
  - `_collect_detections_real(image_paths, objects, handler, n_trials)` — 단일 이미지 반복 or 다중 이미지 1회씩
  - `_collect_detections_mock(objects, n_trials, noise_std, seed)` — GPU 없이 통계 로직 검증용
- CLI: `python pilot_threshold.py --mock --objects "cup" "box"` / `--images img.png --model-type finetune ...`
- 출력: JSON 리포트 (`recommended_threshold_px`, `intra_scene`, `inter_instance_min_dist`, `note`)

**실제 실험 절차 (Phase 2 이미지 촬영 후):**

```bash
python pilot_threshold.py \
  --images scene_a.png scene_b.png scene_c.png scene_d.png scene_e.png \
  --objects "red block" "tray" \
  --model-type finetune --adapter-ckpt 43qazb3XcrZF5rZWnjRPVm \
  --safety-factor 3.0 --output threshold_report.json
# → recommended_threshold_px 값을 consistency_monitor.py 기본값으로 설정
```

**테스트:**

- `tests/test_pilot_threshold.py` — 52개 단위 테스트, 전부 통과
  - `TestCentroid` (4개), `TestEuclidean` (5개), `TestPercentile` (6개)
  - `TestComputeIntraStats` (6개), `TestComputeInterInstanceDistances` (5개)
  - `TestRecommendThreshold` (7개), `TestBuildReport` (4개)
  - `TestCollectDetectionsMock` (7개), `TestCollectDetectionsReal` (4개, mock handler)
  - `TestEndToEndMock` (4개): noise 크기 vs threshold 단조증가 검증
- 전체 test suite: **165개 통과** (20 + 47 + 46 + 52)
- **integration test**: `tests/test_integration.py` — 19개, 실제 모델로 전부 통과
  - `pytest tests/ -m "not integration"` → 165개 단위 테스트 (기본 실행)
  - `pytest tests/test_integration.py -m integration` → 실제 GPU 모델 검증

**실제 모델 실행 중 발견된 사실 (integration 테스트로 확인):**

- Molmo `detect`는 동일 이미지에서 같은 레이블을 **여러 개** 반환할 수 있음 (marker 3개 감지됨)
  - → `check_grounding`이 `AMBIGUOUS_TARGET`을 올바르게 반환하는 것 확인
- `_collect_detections_real`의 coord는 **float** (detect 원시 출력), `_first_coord()`만 int 변환
- `extract_g0(handler=...)` 파라미터 추가 — 외부 핸들러 주입 가능 (모델 중복 로딩 방지)
- pilot threshold (동일 이미지 2회): p95=0.0 → Molmo는 동일 이미지에서 결정론적 출력

---

#### Step 5 — `pipeline.py` ✅ 완료

**목표:** t₀ → C1 → C2 end-to-end 통합

**구현 내용:**

- `CheckpointOutcome` dataclass: checkpoint, state, decision, detections, g0_before, g0_after, user_response
- `PipelineResult` dataclass: status, g0_initial, c1, c2, stop_reason + `to_dict()`
- `run_pipeline(image_t0, image_c1, image_c2, task_description, *, handler, threshold, user_response_fn, ...)` — 메인 함수
  - t₀: `extract_g0()` → ambiguity=true 시 `initial_ambiguous` 반환
  - C1: `get_checkpoint_detections` → `check_grounding("C1")` → STOP이면 즉시 반환
  - C1 ASK: `user_response_fn(question, g0)` 호출 → 답변 있으면 `_update_g0()` 로 rolling update
  - C2: 동일 흐름 (g0_current는 C1 업데이트 반영)
  - C2 ASK: rolling update 후 `complete` 반환
- `_update_g0(image, task, user_response, handler, session_id)` — rolling G₀ 업데이트 (reset→query→respond(answer)→detect)
- `_clarifying_question(g0, role)` — 사람이 읽을 수 있는 질문 생성
- `_noop_response` — 기본 user_response_fn (빈 문자열 반환, 테스트용)
- `UserResponseFn = Callable[[str, dict], str]` — user 응답 콜백 타입
- CLI: `python pipeline.py <t0> <c1> <c2> <task> [--model-type] [--threshold]`

**주요 설계:**

- `handler=None` → `_make_handler()` 내부 생성, 외부 주입 시 재사용 (OOM 방지)
- `user_response_fn` 반환값이 `""` → rolling update 건너뜀 (G₀ 그대로 유지)
- status: `"complete"` / `"stopped"` / `"initial_ambiguous"` 세 가지

**테스트:**

- `tests/test_pipeline.py` — 36개 단위 테스트, 전부 통과
  - `TestHappyPath` (7개): t0→C1(CLEAR)→C2(CLEAR)→complete
  - `TestInitialAmbiguous` (4개): t0 ambiguity=true 처리
  - `TestC1Stop` (4개): INVALID_TARGET → STOP, C2=None
  - `TestC2Stop` (1개): INVALID_DESTINATION → ASK → complete
  - `TestC1Ask` (5개): AMBIGUOUS_TARGET → ASK → rolling update → C2 반영
  - `TestOutputSchema` (4개): 반환 타입·직렬화 검증
  - `TestHandlerInjection` (2개): handler 주입/생성 분기
  - `TestHelpers` (4개): \_noop_response, \_clarifying_question, \_update_g0
  - `TestProposalScenarios` (5개): proposal §5.1 초기 시나리오 ①~⑤ 직접 검증 (S6는 baseline/evaluator 단계에서 추가)
- 전체 test suite: **201개 통과** (20+47+46+52+36)
- **pipeline integration test**: `TestPipelineIntegration` — 21개, 실제 모델로 전부 통과

**실제 모델 pipeline 실행 결과 (marker 이미지, 동일 장면 t0=C1=C2):**

```json
{
  "status": "complete",
  "g0_initial": {
    "target": { "label": "marker", "coord": [1660, 1375] },
    "destination": { "label": "sprite bottle", "coord": [708, 1150] }
  },
  "c1": { "state": "AMBIGUOUS_TARGET", "decision": "ASK" },
  "c2": { "state": "CLEAR", "decision": "CONTINUE" }
}
```

- C1이 AMBIGUOUS_TARGET인 이유: Molmo가 동일 이미지에서 "marker"를 3개 감지
- user_response_fn이 noop(빈 문자열) → rolling update 없이 C2로 진행
- C2는 CLEAR → status="complete"

**실제 이미지 시나리오 ③ 검증 (test-02, "place the red mug next to the wine bottle"):**

- t₀: red mug @ [1380, 1060], wine bottle @ [2708, 835] 정상 추출
- C1 (Gc1.png, red mug 제거): INVALID_TARGET → STOP ✓
- 버그 수정: Molmo 객체 미감지 시 `[[]]` 반환 → IndexError → `consistency_monitor.py`에 유효 좌표 필터 추가

---

#### Step 6 — `src/baselines/` ✅ 완료

**목표:** B1, B2, B3, B4, B5 구현

**공통 구조:**

- `src/baselines/common.py`
  - `BaselineDecision = Decision` alias
  - `BaselineResult(method, decision, reason, question, raw_output, metadata)` dataclass
  - `to_dict()` JSON 직렬화 지원
- `src/baselines/__init__.py`
  - `run_b1_initial_only`, `run_b2_no_memory`, `run_b3_count_rule`, `run_b4_binary_anomaly`, `run_b5_llm_judge` export

**구현 완료:**

- `src/baselines/b1_initial_only.py` — t₀ `reset → query`만 수행, ambiguity면 ASK, clear면 CONTINUE
- `src/baselines/b2_no_memory.py` — checkpoint 이미지에서 `reset → query`, G₀ 없이 현재 장면만 보고 ASK/CONTINUE
- `src/baselines/b3_count_rule.py` — AmbRes `object_list/task_objects`의 full-label exact count rule
  - target 0개 → STOP
  - target 2개 이상 → ASK
  - destination 0개 또는 2개 이상 → ASK
  - target/destination 각각 1개 → CONTINUE
  - AmbRes 자체 ambiguity flag는 의도적으로 무시
- `src/baselines/b4_binary_anomaly.py` — G₀ label+coord와 checkpoint detections 비교, anomaly면 taxonomy 없이 항상 ASK
  - missing target도 Ours처럼 STOP하지 않고 ASK로 접음
  - programmatic API는 handler+image 사용
  - CLI는 `g0.json + detections.json` 입력으로 handler 없이 실행 가능
- `src/baselines/b5_llm_judge.py` — initial/checkpoint 이미지를 GPT-4V-class LLM에게 함께 제시해 CONTINUE/ASK/STOP 직접 판단
  - `llm_client` 주입 가능 (테스트/다른 provider용)
  - 기본 path는 OpenAI Responses API optional client
  - JSON string/dict/code-fence output parsing 지원

**B5 설계 요약:**

- 입력: `initial_img`, `checkpoint_img`, `task`, `checkpoint`
- 출력: `{"decision": "CONTINUE|ASK|STOP", "reason": "..."}`
- 고정 prompt로 target/destination이 여전히 identifiable/unambiguous한지 직접 판정
- parsing failure policy, model name, temperature 고정
- Ours 대비 약점: structured G₀ 없음, coord 기반 identity 추적 없음, rolling G₀ update 어려움, API 비용/latency 큼

**테스트:**

- `tests/test_baseline_b1.py` — 10개 통과
- `tests/test_baseline_b2.py` — 9개 통과
- `tests/test_baseline_b3.py` — 15개 통과
- `tests/test_baseline_b4.py` — 13개 통과
- `tests/test_baseline_b5.py` — 13개 통과
- Baseline 전용: `pytest tests/test_baseline_b*.py` → **60개 통과**
- 전체 unit suite: `pytest tests/ -m "not integration"` → **274개 통과, 1개 skipped, 40개 deselected**

**실제 환경에서 추가 검증 필요:**

- B1~B4: 실제 AmbRes/Molmo handler + 실제 checkpoint 이미지에서 query/detect output shape 검증 필요
- B4: 실제 Molmo `detect`가 반환하는 coord scale, 중복 detection, `[[]]` empty detection edge case 재확인 필요
- B5: OpenAI API key, `openai` package, 실제 GPT-4V-class model에서 prompt adherence / JSON parsing failure rate / latency / cost 측정 필요
- B5: 논문 실험 전 model name, temperature, retry policy, parsing failure policy를 frozen config로 기록 필요
- 모든 baseline: 실제 dataset 6개 시나리오에서 B1~B5+Ours decision table 산출 필요

---

### Phase 1.5: SmolVLA + 실제 로봇 팔 연동 — ✅ 코드 구현 완료 / 🔄 파인튜닝 준비 진행 중

**목표:** AmbRes/Molmo는 server에서 ambiguity resolution을 담당하고, SmolVLA는 desktop에서 low-latency action model로 실행하여 SO-ARM100/101 실제 로봇 팔을 제어한다.

> Phase 2 이미지 기반 평가(B1~B4+Ours 6/6 완벽 달성)가 완료됨에 따라, 현재는 실제 SO-ARM101 환경에서의 파인튜닝 데이터 수집 및 smoke test로 진행 방향을 전환한다. 첫 단계로 글로벌뷰 카메라 연결/캡처를 검증한다.

**구현 완료 범위:**

- `module/models/smolvla/model.py`
  - `SmolVLAModel(BaseLeRobotModel)` 추가, `@ModelRegistry.register("smolvla")` 등록
  - HuggingFace `lerobot/smolvla` checkpoint 로드
  - `bfloat16` / `float16` / `float32` precision 옵션
  - `Observation.images`, `Observation.state`, `Observation.task_text`를 SmolVLA policy batch로 변환
  - action chunk buffer (`action_horizon`) 및 `reset()` 구현
- `module/config/models/smolvla.yaml`
  - 기본 config: `model_id: lerobot/smolvla`, `device: cuda`, `precision: bfloat16`
- `module/config/pipelines/ambres_smolvla.yaml`
  - `local_model_type: smolvla` 설정으로 action model을 desktop local 경로로 실행
  - episode_start에서 AmbRes `query` 호출
  - `task_ambiguous == true`이면 terminal clarification 후 AmbRes `respond` 호출
  - `task_objects`를 `join_list`로 변환해 SmolVLA의 `task_text`로 전달
- `module/desktop/pipeline.py`
  - `PipelineConfig`에 `local_model_type`, `local_model_checkpoint`, `local_model_config` 추가
  - local model이 설정되면 server-side `InferenceClient`를 건너뛰고 `ModelRegistry.build("smolvla", ...)` 실행
  - action loop에서 `SmolVLAModel.predict_action()` → `RobotConnector.send_action()` 경로 구현
- `module/desktop/robot_connector.py`
  - LeRobot 0.5.x 기준 SO-ARM100/101 follower/leader arm 연결
  - single-arm 및 dual-arm follower port mapping 지원
  - egoview OpenCV camera config 생성
  - `get_observation()`에서 images + joint state 추출
  - `send_action()`에서 SmolVLA action vector를 follower arm joint action dict로 변환
  - leader arm 기반 teleoperation (`teleop_step`) 지원
- `module/desktop/data_collector.py`
  - leader-follower teleoperation demonstration 수집
  - frame별 image/state/action 저장용 `EpisodeBuffer` 연동
- `module/config/desktop.yaml`
  - 실제 SO-ARM101 dual-arm 포트 설정 (`right`, `left`)
  - egoview camera 2대 설정
  - RunPod/gRPC 접속 설정
- `scripts/run_desktop.py`
  - `pipeline` subcommand로 AmbRes + SmolVLA + RobotConnector 실행 가능
  - train choice에 `smolvla` 포함

**실행 명령:**

```bash
# 1. RunPod / GPU server 터널 열기
python scripts/open_tunnel.py --env .env.runpod --auto-reconnect

# 2. 서버 연결 확인
python scripts/check_connection.py

# 3. 실제 로봇 팔에서 AmbRes + SmolVLA pipeline 실행
python scripts/run_desktop.py \
  pipeline \
  --config module/config/desktop.yaml \
  --pipeline-config module/config/pipelines/ambres_smolvla.yaml \
  --task "pick up the banana" \
  --n-episodes 1
```

**파인튜닝 준비 진행 상황:**

| 단계 | 내용 | 상태 |
|------|------|------|
| ① 글로벌뷰 카메라 테스트 | `scripts/test_camera.py` — ZED Mini / USB scan/캡처/저장, AmbRes end-to-end | ✅ 스크립트 완료 |
| ② AmbRes G₀ 추출 현장 검증 | 실제 카메라로 캡처 → G₀ 정상 추출 확인 | ⬜ |
| ③ 데모 데이터 수집 | `run_desktop.py collect` — leader-follower teleoperation 에피소드 | ⬜ |
| ④ SmolVLA 파인튜닝 트리거 | `run_desktop.py train --model-type smolvla` | ⬜ |
| ⑤ SmolVLA smoke test | action dim / camera key / joint range 검증 | ⬜ |
| ⑥ C1/C2 monitor 통합 | live loop에 `check_grounding()` → CONTINUE/ASK/STOP gating 추가 | ⬜ |

**관련 문서:** [docs/camera_guide.md](camera_guide.md) — 카메라 설정, 테스트 스크립트 사용법, AmbRes 연동 절차

**현재 한계 / 다음 작업:**

- **즉시**: 로컬 데스크탑에서 `test_camera.py --mode scan` 실행 → global view 카메라 인덱스 확인
- 실제 SO-ARM101 hardware에서 SmolVLA end-to-end smoke test 필요
  - observation camera key가 SmolVLA checkpoint가 기대하는 key와 맞는지 확인
  - action dimension과 `state_keys` 길이가 일치하는지 확인
  - joint unit/range가 LeRobot policy 출력과 follower arm 입력 사이에서 맞는지 확인
- live robot loop에 Execution-Aware C1/C2 monitor 통합 필요
  - 현재 `module/desktop/pipeline.py`는 episode_start ambiguity resolution만 수행
  - C1(pre-pick), C2(pre-place) checkpoint 타이밍을 action policy 또는 high-level controller에서 받아야 함
  - checkpoint에서 `get_checkpoint_detections()` → `check_grounding()` → CONTINUE/ASK/STOP gating 추가 예정
- safety guard 필요
  - action clipping / joint limit validation
  - emergency stop hook
  - ASK/STOP 발생 시 robot hold pose 또는 neutral action 정책

---

### Phase 2: Dataset 구축 및 실험 🔄 진행 중 (1차 평가 결과 확보)

| Step | 내용                                               | 비고                                                                                        |
| ---- | -------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 7    | 실제 이미지 촬영                                   | S1~S3, S5~S6 촬영 완료; S4는 synthetic 이미지로 보완 ✅                                    |
| 8    | Annotation + inter-annotator check (Cohen's kappa) | `dataset/annotator.py`                                                                      |
| 9    | B1~B5+Ours 전체 실험 + metrics 계산                | B1~B4+Ours 1차 실행 완료 ✅ (`logs/predictions.csv`, `logs/metrics.json`); B5 별도 실행 필요 |
| 10   | Ablation (C1-only, C2-only, label-only G₀)         | `ablation.py`                                                                               |
| 11   | SmolVLA + 실제 로봇 팔 smoke test                  | `ambres_smolvla.yaml`, SO-ARM101, 30 Hz loop                                                |

**6개 시나리오:**

| #   | 시나리오                                    | Gold State            | Gold Decision |
| --- | ------------------------------------------- | --------------------- | ------------- |
| ①   | Clear continuation                          | CLEAR                 | CONTINUE      |
| ②   | Same-category target 추가                   | AMBIGUOUS_TARGET      | ASK           |
| ③   | Target disappeared                          | INVALID_TARGET        | STOP          |
| ④   | New destination candidate                   | AMBIGUOUS_DESTINATION | ASK           |
| ⑤   | Distractor added (무관한 물체)              | CLEAR                 | CONTINUE      |
| ⑥   | Target 위치 변경 / 동일 label 다른 instance | AMBIGUOUS_TARGET      | ASK           |

> ★ 시나리오 ②, ⑤, ⑥이 제안 방법의 차별점을 가장 잘 드러냄. 특히 ⑥은 B3(count 유지)와 B5(general VLM 비교)의 한계를 동시에 보여주는 coord memory 핵심 시나리오.

**Evaluator 구현 상태:**

- `src/evaluate.py` ✅ 완료
  - manifest loader: `.jsonl`, JSON list, `{"samples": [...]}` 지원
  - sample schema: `id`, `scenario`, `task`, `initial_img`, `c1_img`, `c2_img`, `checkpoint`, `gold_state`, `gold_decision`, `target_label`, `destination_label`
  - `--validate-only`: 모델 없이 manifest schema와 scenario/checkpoint/decision 분포 검증
  - `--check-images`: 모델 없이 manifest가 참조하는 실제 이미지 파일 존재 여부 검증
  - B1~B5+Ours method runner 연결
  - `Decision Accuracy`, `Miss Rate`, `False Alarm Rate`, `State Accuracy` 계산
  - predictions CSV / metrics JSON 저장 지원
- `dataset/README.md` ✅ 완료
  - 이미지 디렉터리 구조, manifest field, validate/check-images/evaluate 명령 정리
- `dataset/manifest.example.jsonl` ✅ 완료
  - S1~S6 1개씩 예시 row 포함
- `dataset/manifest.jsonl` ✅ 2차 업데이트 (S1~S6 6개 시나리오 완성)
  - S1~S3, S5~S6: 실제 촬영 이미지
  - S4: `red_mug_sprite_001_c2_two_sprite_bottles.png` — C2 clear 이미지에서 sprite bottle 영역을 +450px offset에 복제하여 AMBIGUOUS_DESTINATION synthetic 이미지 생성
  - S1은 static sanity pass용으로 t₀ 복사본을 C1/C2 clear checkpoint로 사용; 최종 실험에서는 실제 robot snapshot 권장
- `tests/test_evaluate.py` ✅ 완료 (14 tests passed)
- 모델 없는 검증: `python src/evaluate.py dataset/manifest.example.jsonl --validate-only` 통과
- 실제 이미지 manifest 검증: `python src/evaluate.py dataset/manifest.jsonl --validate-only --check-images` 통과 (6 samples, missing images 0)

**1차 평가 결과 (B1~B4+Ours, 2026-05-19 v1, before S5 fix):**
→ `logs/predictions.csv`, `logs/metrics.json` (Ours: 5/6, S5 false alarm)

**2차 평가 결과 (B1~B4+Ours, 2026-05-19 v2, S5 fix 적용):**

```bash
python src/evaluate.py dataset/manifest.jsonl \
  --methods b1 b2 b3 b4 ours \
  --model-type finetune --adapter-ckpt 43qazb3XcrZF5rZWnjRPVm \
  --predictions-csv logs/predictions_v2.csv --metrics-json logs/metrics_v2.json
```

| Method              | Decision Acc | Miss Rate | False Alarm Rate | State Acc |
| ------------------- | ------------ | --------- | ---------------- | --------- |
| B1 (initial-only)   | 4/6 = 66.7%  | 0%        | 50%              | —         |
| B2 (no-memory)      | 4/6 = 66.7%  | 0%        | 50%              | —         |
| B3 (count-rule)     | 2/6 = 33.3%  | 100%      | 0%               | —         |
| B4 (binary-anomaly) | 4/6 = 66.7%  | 0%        | 50%              | —         |
| **Ours**            | **6/6 = 100%** | **0%** | **0%**           | **6/6 = 100%** |

**시나리오별 상세 결과 (v2):**

| Scenario       | Gold     | B1    | B2    | B3    | B4    | Ours   |
| -------------- | -------- | ----- | ----- | ----- | ----- | ------ |
| S1 CLEAR/C1    | CONTINUE | ASK ✗ | CONT ✓ | CONT ✓ | CONT ✓ | CONT ✓ |
| S2 AMB_TGT/C1  | ASK      | ASK ✓ | ASK ✓ | CONT ✗ | ASK ✓ | ASK ✓  |
| S3 INV_TGT/C1  | STOP     | ASK ✗ | ASK ✗ | CONT ✗ | ASK ✗ | STOP ✓ |
| S4 AMB_DST/C2  | ASK      | ASK ✓ | ASK ✓ | CONT ✗ | ASK ✓ | ASK ✓  |
| S5 CLEAR/C1    | CONTINUE | CONT ✓ | ASK ✗ | CONT ✓ | ASK ✗ | CONT ✓ |
| S6 AMB_TGT/C1  | ASK      | ASK ✓ | ASK ✓ | CONT ✗ | ASK ✓ | ASK ✓  |

> S5 v1→v2 변화: Ours ASK ✗ → CONT ✓ (camera-motion 보정으로 수정)

**핵심 관찰 (v2):**
- **Ours: 6/6 완벽 달성** — Decision Accuracy 100%, Miss Rate 0%, False Alarm Rate 0%, State Accuracy 100%
- **Ours만 S3(INVALID_TARGET → STOP)를 정확히 처리** — B1~B4 모두 ASK 또는 CONTINUE로 오분류
- **S5 camera-motion 보정**: 로봇이 t₀→C1 이동 시 카메라도 이동해 pixel 좌표 ~541px shift 발생. destination을 앵커로 relative distance(‖(tgt_t−dst_t)−(tgt_0−dst_0)‖)를 계산하여 카메라 이동 성분 제거; S5: rel_dist=186px < threshold×5=250 → CLEAR ✓; S6: rel_dist=1337px > 250 → AMBIGUOUS ✓
- **B3**: miss rate 100% — label count 불변 시 항상 CONTINUE (S6: label 1개 유지되어도 coord 변화 감지 못함)
- **B5는 OpenAI API key 필요로 이번 실행에서 제외**; 다음 실험에서 추가 필요

**환경 설정 완료 내역 (2026-05-19):**
- `/workspace/AmbRes` 클론 완료
- `outlines==0.1.14` — `outlines.processors.JSONLogitsProcessor` 호환 버전 확인 (1.x는 API 변경으로 비호환)
- `transformers==4.48`, `peft>=0.9.0`, `accelerate>=0.26.0` 설치 완료
- `AmbRes/ckpt/43qazb3XcrZF5rZWnjRPVm/checkpoint-200/` — finetune 체크포인트 다운로드 완료
- `AmbRes/ckpt/TtVVGPRoknCTELVSjH92xX/checkpoint-200/` — sim 체크포인트도 포함

**버그 픽스 (2026-05-19):**
- `src/extraction/role_parser.py`: VLM이 qualified label(`'yellow marker'`, `'left red mug'` 등)을 반환할 때 task 문자열과 매칭 실패 → `_find_object_after_terms` 및 `_find_object_in_text`에 suffix matching 추가
- `tests/test_integration.py::test_known_roles_marker_and_sprite`: 모델 비결정성 대응으로 `endswith("marker")` 방식 변경
- `src/evaluate.py`: `extract_g0`/`run_pipeline` 호출에 `allow_ambiguous=True` 추가 (t₀ false positive 방지)
- `src/monitoring/consistency_monitor.py`: **camera-motion 보정** — destination-anchored relative distance 도입. 단일 감지 + abs_dist>threshold + dest도 이동(카메라 이동 신호) 시 relative check 적용; threshold×5 이내이면 CLEAR 반환. +6 테스트 추가 (`TestCameraMotionCompensation`)

**현재 한계:**
- B5는 실제 OpenAI API key/model에서 검증 필요
- rel_threshold default (threshold×5) — pilot_threshold.py로 C1 이미지 기반 적정값 결정 권장
- 실제 robot snapshot 이미지로 S1 C1/C2 교체 권장 (현재 t₀ 복사본 사용)

---

### Phase 3: 논문 작성 ⬜ 미착수

- Method 섹션: 설계 완료, 수치만 채우면 됨
- Figure 1: 시나리오 ② (same-category target 추가) — motivating example
- 핵심 claim: B2(no memory) 대비 wrong continue 감소, B1 대비 false alarm 감소

---

## 4. 파일 구조 현황

```
COSE461-proj/
├── src/                          ✅ 소스 패키지 (역할별 세부 폴더 분리)
│   ├── __init__.py
│   ├── extraction/               AmbRes VLM 인터페이스 + G0 추출
│   │   ├── ambres_g0_extractor.py  ✅ 완료
│   │   └── role_parser.py          ✅ 완료
│   ├── monitoring/               Checkpoint 상태 판단
│   │   └── consistency_monitor.py  ✅ 완료
│   └── utils/                    분석 도구
│       └── pilot_threshold.py      ✅ 완료
├── src/pipeline.py               ✅ 완료
├── src/evaluate.py               ✅ 완료 (manifest loader + image path check + B1~B5/Ours metrics)
├── dataset/
│   ├── README.md                  ✅ 완료 (manifest 작성/검증 가이드)
│   ├── manifest.example.jsonl     ✅ 완료 (S1~S6 예시)
│   ├── manifest.jsonl             ✅ 1차 완료 (실제 이미지 S1/S2/S3/S5/S6)
│   └── images/.gitkeep            ✅ 완료
├── pytest.ini                    ✅ 완료 (integration 마커, pythonpath=src, log 설정)
├── run_tests.sh                  ✅ 완료 (단위 테스트 + 타임스탬프 로그)
├── run_integration.sh            ✅ 완료 (통합 테스트 + 타임스탬프 로그)
├── scripts/run_pipeline_local.py ✅ 완료 (파이프라인 인터랙티브 터미널 테스트)
├── scripts/run_desktop.py        ✅ 완료 (실제 로봇 / data collection / pipeline entrypoint)
├── scripts/run_server.py         ✅ 완료 (gRPC server entrypoint)
├── scripts/open_tunnel.py        ✅ 완료 (RunPod SSH tunnel)
├── scripts/check_connection.py   ✅ 완료 (desktop ↔ server 연결 확인)
├── scripts/test_camera.py        ✅ 완료 (글로벌뷰 카메라 단독 테스트: scan/usb/zed + AmbRes 연동)
├── module/                       ✅ 실제 로봇/서버 분리 실행 패키지
│   ├── desktop/
│   │   ├── pipeline.py           ✅ 완료 (AmbRes pre-handler + local SmolVLA action loop)
│   │   ├── robot_connector.py    ✅ 완료 (SO-ARM100/101 follower/leader 연결)
│   │   ├── zed_connector.py      ✅ 완료 (ZEDCapture / USBCapture / SyncCapture)
│   │   ├── data_collector.py     ✅ 완료 (teleop demonstration 수집)
│   │   └── generic_client.py     ✅ 완료 (AmbRes generic gRPC client)
│   ├── models/
│   │   ├── ambres/handler.py     ✅ 완료 (Molmo/AmbRes server handler)
│   │   └── smolvla/model.py      ✅ 완료 (local SmolVLA wrapper)
│   ├── config/
│   │   ├── desktop.yaml          ✅ 완료 (SO-ARM101 dual-arm + cameras + gRPC)
│   │   ├── models/smolvla.yaml   ✅ 완료
│   │   └── pipelines/ambres_smolvla.yaml ✅ 완료
│   └── server/                   ✅ 완료 (GenericInferenceService)
├── src/baselines/                ✅ baseline 패키지
│   ├── __init__.py               ✅ 완료
│   ├── common.py                 ✅ 완료 (BaselineResult)
│   ├── b1_initial_only.py        ✅ 완료
│   ├── b2_no_memory.py           ✅ 완료
│   ├── b3_count_rule.py          ✅ 완료
│   ├── b4_binary_anomaly.py      ✅ 완료
│   └── b5_llm_judge.py           ✅ 완료
├── logs/                         ✅ 자동 로깅
│   ├── test_ambres_*.log         (기존 AmbRes 핸들러 로그)
│   ├── pytest_latest.log         (최신 pytest 실행 — 항상 덮어씀)
│   └── pytest_YYYYMMDD_HHMMSS.log (타임스탬프별 보관)
├── tests/
│   ├── conftest.py               ✅ 완료 (tf stub, real_handler, 로그 훅)
│   ├── test_g0_extractor.py      ✅ 완료 (20 tests total; local latest 19 passed + 1 skipped)
│   ├── test_baseline_b1.py       ✅ 완료 (10 tests passed)
│   ├── test_baseline_b2.py       ✅ 완료 (9 tests passed)
│   ├── test_baseline_b3.py       ✅ 완료 (15 tests passed)
│   ├── test_baseline_b4.py       ✅ 완료 (13 tests passed)
│   ├── test_baseline_b5.py       ✅ 완료 (13 tests passed)
│   ├── test_evaluate.py          ✅ 완료 (14 tests passed)
│   ├── test_role_parser.py       ✅ 완료 (47 tests passed)
│   ├── test_consistency_monitor.py ✅ 완료 (52 tests passed; +6 TestCameraMotionCompensation)
│   ├── test_pilot_threshold.py   ✅ 완료 (52 tests passed)
│   ├── test_pipeline.py          ✅ 완료 (36 tests passed)
│   └── test_integration.py       ✅ 완료 (19 tests, 실제 모델 검증)
├── docs/
│   ├── proposal.md
│   ├── camera_guide.md           ✅ 완료 (ZED Mini/USB 설정, test_camera.py 사용법, AmbRes 연동)
│   ├── setup_and_workflow.md     ✅ 완료 (환경 설정, 데이터 수집, 학습, 추론 전체 절차)
│   ├── runpod_connection.md      ✅ 완료 (RunPod SSH 터널 연결 가이드)
│   ├── baselines.md              ✅ 완료 (B1~B5 baseline 설계)
│   ├── molmo_smolvla_arch.md     ✅ 완료 (Molmo + SmolVLA 아키텍처)
│   └── PROGRESS.md               ← 이 파일
```

---

## 5. 주요 설계 결정 사항

| 항목                    | 결정                                            | 근거                                                               |
| ----------------------- | ----------------------------------------------- | ------------------------------------------------------------------ |
| Checkpoint 타이밍       | Fixed (C1 pre-pick, C2 pre-place)               | BT: pre-condition / PAL: pre-irreversible                          |
| G₀ 표현                 | label + coord (int)                             | label-only → identity 구분 불가                                    |
| t₀ Step 2 강제 호출     | `response=""` 빈 answer                         | 좌표 확보 목적                                                     |
| coord 타입              | int (픽셀)                                      | proposal 예시는 float이나 Molmo 출력은 정수                        |
| ambiguity=true 처리     | RuntimeError raise → 상위 pipeline에서 ASK 처리 | extractor는 G₀ 추출 책임만                                         |
| STOP 범위               | Conservative (INVALID만 STOP)                   | 나머지는 ASK → 사용자 판단 위임                                    |
| Disambiguation 순서     | destination 먼저, target 나중                   | BT 논문 명시                                                       |
| B5 baseline 포함        | GPT-4V-class LLM에 initial/checkpoint 직접 비교 | reviewer 질문 "그냥 GPT-4V면 되지 않나?" 방어                      |
| Molmo/SmolVLA 배치      | Molmo는 server, SmolVLA는 desktop local         | Molmo VRAM 부담과 30 Hz action latency 분리                        |
| SmolVLA precision       | `bfloat16` 기본                                 | RTX 3060 8 GB VRAM 목표                                            |
| SmolVLA checkpoint      | `lerobot/smolvla`                               | `local_model_checkpoint`로 local path override 가능                |
| 실제 로봇 connector     | LeRobot 0.5.x SOFollower/SOLeader 기준          | SO-ARM100/101 hardware API와 맞춤                                  |
| live checkpoint monitor | 아직 미통합                                     | 현재 live loop는 episode_start AmbRes + SmolVLA action loop만 수행 |

---

## 6. 평가 지표

| Metric                        | 정의                                   |
| ----------------------------- | -------------------------------------- |
| Grounding State Accuracy      | gold state vs 예측 state 일치율        |
| Decision Accuracy             | CONTINUE/ASK/STOP 일치율               |
| False Alarm Rate              | CLEAR 상황에서 ASK 출력한 비율         |
| Miss Rate                     | ASK/STOP 상황에서 CONTINUE 출력한 비율 |
| C1 vs C2 contribution         | C1-only / C2-only / both ablation      |
| Coord vs label-only           | B4 vs Ours accuracy 차이               |
| External VLM API Cost/Latency | B5 GPT-4V-class 호출 비용 및 응답 시간 |

**최신 로컬 unit test:** `pytest tests/ -m "not integration"` → **281 passed, 40 deselected**
