# Execution-Aware Ambiguity Handling — 진행 기록

> 이 파일은 proposal.md 기반 실행 계획과 진행 상태를 하나의 문서로 관리한다.  
> 작업이 완료될 때마다 이 파일의 상태를 업데이트한다.

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

### 2.2 G₀ 표현 형식

```json
{
  "target":      { "label": "red block", "coord": [142, 318] },
  "destination": { "label": "tray",      "coord": [401, 289] },
  "image_shape": [480, 640]
}
```

> coord는 픽셀 정수 `[x, y]` (Molmo 출력 기준). proposal 예시는 float이나 int가 실용적으로 더 적합.

### 2.3 Grounding State Taxonomy

| State | 트리거 조건 | Decision | Checkpoint |
|---|---|---|---|
| CLEAR | Gₜ에 target₀, dest₀ 모두 유효 | CONTINUE | C1, C2 |
| CLEAR (distractor) | 새 object가 task와 무관 | CONTINUE | C1, C2 |
| AMBIGUOUS_TARGET | 같은 label의 coord가 여러 개 | ASK | C1 only |
| INVALID_TARGET | target₀가 Gₜ에 없음 | STOP | C1 only |
| AMBIGUOUS_DESTINATION | dest 후보가 여러 개 | ASK | C2 only |
| OCCUPIED_DESTINATION | dest₀는 있지만 occupied 상태 | ASK | C2 only |
| INVALID_DESTINATION | dest₀가 Gₜ에 없음 | ASK/STOP | C2 only |
| UNSAFE_OR_BLOCKED | safety 위협 감지 | STOP | C1, C2 |

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

| | 방법 | G₀ 메모리 | 설명 |
|---|---|---|---|
| B1 | AmbResVLM single-shot | 없음 | checkpoint에서 현재 장면만 판단 |
| B2 | AmbResVLM repeated | 없음 | 매 checkpoint 재호출, G₀ 비교 없음 |
| B3 | Candidate count rule | 없음 | 후보 개수 > 1이면 ASK (BT 방식) |
| B4 | Ours w/o taxonomy | 있음 | G₀ 비교는 하지만 binary (valid/invalid)만 |
| **Ours** | Execution-Aware AmbResVLM | **있음** | G₀ + coord + taxonomy + decision policy |

---

## 3. 실행 계획 및 진행 상태

### Phase 0: Research Design ✅ 완료

- [x] 주제 확정
- [x] 포지셔닝 (2×2 매트릭스)
- [x] proposal.md 완성

---

### Phase 1: 코드 구현 — 🔄 진행 중 (5/6 완료)

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
- `tests/test_g0_extractor.py` — 20개 단위 테스트 (mock 기반), 전부 통과
- 실제 모델 실행 확인 (finetune + CKPT.REAL = `43qazb3XcrZF5rZWnjRPVm`)

```json
// 실제 실행 결과 예시 (5rhU25AdQW4jADxhp8EYuq.jpeg, "move the marker next to the sprite bottle")
{
  "target":      { "label": "marker",        "coord": [1660, 1375] },
  "destination": { "label": "sprite bottle", "coord": [708,  1150] },
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
  - `TestScenarioAlignment` (5개): **proposal §5.1 시나리오 ①~⑤ 직접 매핑**
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
  - `TestHelpers` (4개): _noop_response, _clarifying_question, _update_g0
  - `TestProposalScenarios` (5개): proposal §5.1 시나리오 ①~⑤ 직접 검증
- 전체 test suite: **201개 통과** (20+47+46+52+36)
- **pipeline integration test**: `TestPipelineIntegration` — 21개, 실제 모델로 전부 통과

**실제 모델 pipeline 실행 결과 (marker 이미지, 동일 장면 t0=C1=C2):**
```json
{
  "status": "complete",
  "g0_initial": {"target": {"label": "marker", "coord": [1660, 1375]},
                 "destination": {"label": "sprite bottle", "coord": [708, 1150]}},
  "c1": {"state": "AMBIGUOUS_TARGET", "decision": "ASK"},
  "c2": {"state": "CLEAR",           "decision": "CONTINUE"}
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

#### Step 6 — `baselines/` ⬜ 미착수

**목표:** B1, B2, B3, B4 구현

**구현 예정:**
- `baselines/b1_single_shot.py` — G₀ 메모리 없이 현재 장면만 판단
- `baselines/b2_repeated.py` — 매 checkpoint 재호출, 비교 없음
- `baselines/b3_count_rule.py` — 후보 개수 > 1이면 ASK
- `baselines/b4_binary.py` — G₀ 비교하되 binary valid/invalid만

---

### Phase 2: Dataset 구축 및 실험 ⬜ 미착수

| Step | 내용 | 비고 |
|---|---|---|
| 7 | 실제 이미지 촬영 | 5 시나리오 × 2 checkpoint × 5회 = 약 50장 |
| 8 | Annotation + inter-annotator check (Cohen's kappa) | `dataset/annotator.py` |
| 9 | B1~Ours 전체 실험 + metrics 계산 | `evaluate.py` |
| 10 | Ablation (C1-only, C2-only, label-only G₀) | `ablation.py` |

**5개 시나리오:**

| # | 시나리오 | Gold State | Gold Decision |
|---|---|---|---|
| ① | Clear continuation | CLEAR | CONTINUE |
| ② | Same-category target 추가 | AMBIGUOUS_TARGET | ASK |
| ③ | Target disappeared | INVALID_TARGET | STOP |
| ④ | New destination candidate | AMBIGUOUS_DESTINATION | ASK |
| ⑤ | Distractor added (무관한 물체) | CLEAR | CONTINUE |

> ★ 시나리오 ②와 ⑤가 제안 방법의 차별점을 가장 잘 드러냄

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
├── pytest.ini                    ✅ 완료 (integration 마커, pythonpath=src, log 설정)
├── run_tests.sh                  ✅ 완료 (단위 테스트 + 타임스탬프 로그)
├── run_integration.sh            ✅ 완료 (통합 테스트 + 타임스탬프 로그)
├── scripts/run_pipeline_local.py ✅ 완료 (파이프라인 인터랙티브 터미널 테스트)
├── baselines/
│   ├── b1_single_shot.py        ⬜ 예정
│   ├── b2_repeated.py           ⬜ 예정
│   ├── b3_count_rule.py         ⬜ 예정
│   └── b4_binary.py             ⬜ 예정
├── logs/                         ✅ 자동 로깅
│   ├── test_ambres_*.log         (기존 AmbRes 핸들러 로그)
│   ├── pytest_latest.log         (최신 pytest 실행 — 항상 덮어씀)
│   └── pytest_YYYYMMDD_HHMMSS.log (타임스탬프별 보관)
├── tests/
│   ├── conftest.py               ✅ 완료 (tf stub, real_handler, 로그 훅)
│   ├── test_g0_extractor.py      ✅ 완료 (20 tests passed)
│   ├── test_role_parser.py       ✅ 완료 (47 tests passed)
│   ├── test_consistency_monitor.py ✅ 완료 (46 tests passed)
│   ├── test_pilot_threshold.py   ✅ 완료 (52 tests passed)
│   ├── test_pipeline.py          ✅ 완료 (36 tests passed)
│   └── test_integration.py       ✅ 완료 (19 tests, 실제 모델 검증)
├── dataset/                      ⬜ 예정
├── docs/
│   ├── proposal.md
│   └── PROGRESS.md               ← 이 파일
```

---

## 5. 주요 설계 결정 사항

| 항목 | 결정 | 근거 |
|---|---|---|
| Checkpoint 타이밍 | Fixed (C1 pre-pick, C2 pre-place) | BT: pre-condition / PAL: pre-irreversible |
| G₀ 표현 | label + coord (int) | label-only → identity 구분 불가 |
| t₀ Step 2 강제 호출 | `response=""` 빈 answer | 좌표 확보 목적 |
| coord 타입 | int (픽셀) | proposal 예시는 float이나 Molmo 출력은 정수 |
| ambiguity=true 처리 | RuntimeError raise → 상위 pipeline에서 ASK 처리 | extractor는 G₀ 추출 책임만 |
| STOP 범위 | Conservative (INVALID만 STOP) | 나머지는 ASK → 사용자 판단 위임 |
| Disambiguation 순서 | destination 먼저, target 나중 | BT 논문 명시 |

---

## 6. 평가 지표

| Metric | 정의 |
|---|---|
| Grounding State Accuracy | gold state vs 예측 state 일치율 |
| Decision Accuracy | CONTINUE/ASK/STOP 일치율 |
| False Alarm Rate | CLEAR 상황에서 ASK 출력한 비율 |
| Miss Rate | ASK/STOP 상황에서 CONTINUE 출력한 비율 |
| C1 vs C2 contribution | C1-only / C2-only / both ablation |
| Coord vs label-only | B4 vs Ours accuracy 차이 |
