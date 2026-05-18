# Execution-Aware Ambiguity Handling for Language-Guided Robotic Manipulation

## Research Design Document — Baseline / Scenario / Metric 확정본 v2

---

## 1. 연구 한 문장 정의

> AmbResVLM은 현재 장면이 ambiguous한지를 판단하지만, 실행 중 장면 변화로 인해 initially-clear grounding이 무효화되는 것은 감지하지 못한다. 본 연구는 초기 grounding G₀를 label + coord로 저장하고 checkpoint 이미지에서 AmbResVLM을 재호출하여, G₀와 Gₜ의 불일치를 8개 grounding-state로 분류하는 최초의 방법이다.

---

## 2. 파이프라인 구조

```
t₀ (초기)
  ├── Task description + Initial image
  ├── AmbResVLM Step1 호출
  ├── ambiguity = true  → ASK (initial), 종료
  └── ambiguity = false → AmbResVLM Step2 강제 호출 (빈 answer)
                          → G₀ 저장 {target: {label, coord}, dest: {label, coord}, image_shape}

       ↓ 로봇 이동 (카메라 off)

C1 (pre-pick checkpoint)            ← target 관련 state 감지
  ├── 카메라 스냅샷
  ├── AmbResVLM Step2 재호출 → Gₜ
  ├── Consistency Monitor: G₀ vs Gₜ
  └── Decision: CONTINUE / ASK / STOP

       ↓ (CONTINUE일 때만)
  PICK 실행 (카메라 off)

       ↓ 로봇 이동 (카메라 off)

C2 (pre-place checkpoint)           ← destination 관련 state 감지
  ├── 카메라 스냅샷
  ├── AmbResVLM Step2 재호출 → Gₜ
  ├── Consistency Monitor: G₀ vs Gₜ
  └── Decision: CONTINUE / ASK / STOP

       ↓ (CONTINUE일 때만)
  PLACE 실행 → 완료
```

### Checkpoint 타이밍 근거

| Checkpoint     | 감지 대상                                  | 논문 근거                                                        |
| -------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| C1 (pre-pick)  | AMBIGUOUS_TARGET, INVALID_TARGET           | BT: Pick 전 Scene Clear? condition / PAL: Observe → Act          |
| C2 (pre-place) | AMBIGUOUS_DESTINATION, INVALID_DESTINATION | BT: "destination disambiguation first" / PAL: 비가역 action 직전 |
| 둘 다          | CLEAR, UNSAFE_OR_BLOCKED                   | PAL: 매 transition point safety check                            |

> BT 논문 명시: "manipulation 중에는 detection을 timed out" → 로봇 이동 및 manipulation 중 카메라 off

---

## 3. G₀ 표현 방식

### 저장 구조

```json
G₀ = {
  "target": {
    "label": "red block",
    "coord": [142.0, 318.5]
  },
  "destination": {
    "label": "gray tray",
    "coord": [401.0, 289.0]
  },
  "image_shape": [480, 640]
}
```

### label-only 방식의 문제점 (coord 저장 이유)

1. **Identity 구분 불가**: 동일 카테고리 내 instance 구분 불가. 원래 block 제거 + 새 block 추가 시 "있음"으로 판단 (S6).
2. **역할 구분 없음**: object_list에 target/destination 역할 정보 없음.
3. **위치 변화 감지 불가**: label이 동일하면 위치가 바뀌어도 CLEAR로 판단.

### t₀에서 Step2 강제 호출

- ambiguity = false여도 Step2를 빈 answer(`""`)로 호출해서 coord 획득
- Step2 output: `{"object_list": [...], "label": [x, y], ...}`

---

## 4. Grounding State Taxonomy

| State                 | 트리거 조건                              | Decision   | Checkpoint |
| --------------------- | ---------------------------------------- | ---------- | ---------- |
| CLEAR                 | G₀ target/dest 모두 coord 유효           | CONTINUE   | C1, C2     |
| CLEAR (distractor)    | target/dest 유효 + task 무관 object 추가 | CONTINUE   | C1, C2     |
| AMBIGUOUS_TARGET      | 동일 label coord 여러 개                 | ASK        | C1 only    |
| INVALID_TARGET        | target coord 없음 또는 dist >> threshold | STOP       | C1 only    |
| AMBIGUOUS_DESTINATION | 동일 label coord 여러 개                 | ASK        | C2 only    |
| INVALID_DESTINATION   | dest coord 없음                          | ASK / STOP | C2 only    |
| UNSAFE_OR_BLOCKED     | safety 위협 감지                         | STOP       | C1, C2     |

> ⚠️ OCCUPIED_DESTINATION은 현재 scope 외 (이후 단계에서 추가)

---

## 5. Baseline 설계

### 개요 비교표

| 방법                       | Checkpoint | G₀ memory | Coord | Taxonomy | 논문 근거                   |
| -------------------------- | ---------- | --------- | ----- | -------- | --------------------------- |
| B1 Initial-only            | ✗          | ✗         | ✗     | ✗        | AmbResVLM 실제 사용 방식    |
| B2 No memory               | ✓          | ✗         | ✗     | ✗        | INVIGORATE w/o history      |
| B3 Attribute-aware Count   | ✓          | ✗         | ✗     | ✗        | BT Scene Clear? (강화 버전) |
| B4 Binary Anomaly Detector | ✓          | ✓         | ✓     | ✗        | PAL No Watchdog ablation    |
| **Ours**                   | ✓          | ✓         | ✓     | ✓        | PAL + BT + AmbResVLM        |

### 비교 체인

```
B1 → B2 → B4 → Ours
               ↑
           +taxonomy

+ B3: attribute-aware rule cross-check
```

| 비교       | 추가 component                        | 검증 대상                                                       |
| ---------- | ------------------------------------- | --------------------------------------------------------------- |
| B1 → B2    | checkpoint 추가                       | 재호출 자체의 기여                                              |
| B2 → B4    | G₀ + coord 추가                       | G₀ memory의 기여                                                |
| B4 → Ours  | 8-state taxonomy                      | state 세분화의 기여 (anomaly detection vs state classification) |
| B3 vs Ours | attribute-aware rule vs G₀ comparison | G₀ coord memory의 추가 기여                                     |

---

### B1 — AmbResVLM Initial-only

**개념**: t₀에서 AmbResVLM을 1회 호출. 이후 checkpoint 없음. AmbResVLM 논문의 실제 사용 방식.

```python
def b1(initial_img, task):
    output = AmbResVLM_step1(initial_img, task)

    if output['ambiguity']:
        return 'ASK', output['clarifying_question']

    # clear → G₀ 저장 없이 바로 실행 (C1, C2 없음)
    return 'EXECUTE', None
```

**예상 실패**: S2, S3, S4, S6 → 모두 wrong CONTINUE (checkpoint 없으므로 감지 불가)

---

### B2 — Checkpoint, No Memory

**개념**: C1/C2에서 AmbResVLM 재호출하지만 G₀ 없음. 매 checkpoint 독립 판단.

```python
def b2(checkpoint_img, task):
    # G₀ 없음: 현재 장면만 보고 독립 판단
    output = AmbResVLM_step1(checkpoint_img, task)

    if output['ambiguity']:
        return 'ASK', output['clarifying_question']
    return 'CONTINUE', None

# C1, C2 각각 독립 호출 (이전 결과 무관)
```

**예상 실패**: S5(distractor) → false ASK / S6(위치 변경) → wrong CONTINUE

---

### B3 — Attribute-aware Candidate Count Rule

**개념**: AmbResVLM을 호출해 object_list를 얻되, VLM의 `ambiguity` 판단을 무시하고
attribute를 포함한 full label로 후보를 매칭해 개수를 센다. G₀ 없음.

단순 `object_list.count("block")`이 아니라 `"red block"` 전체 label로 매칭하므로
기존 count rule보다 훨씬 정교하다. 그럼에도 coord를 모르기 때문에 S6(위치 변경)는
여전히 감지하지 못한다 — 이 지점이 G₀ coord memory의 기여를 드러낸다.

```python
def b3(checkpoint_img, task, target_label, dest_label):
    """
    target_label: "red block" (attribute 포함 full label)
    dest_label:   "gray tray"
    """
    output = AmbResVLM_step1(checkpoint_img, task)
    obj_list = output['object_list']  # e.g. ["red block", "gray tray", "blue cup"]

    # attribute-aware 매칭 (exact match on full label)
    n_target = sum(1 for o in obj_list if o == target_label)
    n_dest   = sum(1 for o in obj_list if o == dest_label)

    # 개수 기반 decision (VLM 판단 무시)
    if n_target == 0:
        return 'STOP',     f'{target_label} not found in scene'
    elif n_target > 1:
        return 'ASK',      f'Multiple {target_label}s found. Which one?'
    elif n_dest == 0:
        return 'ASK',      f'{dest_label} not found. Where should I place it?'
    elif n_dest > 1:
        return 'ASK',      f'Multiple {dest_label}s found. Which one?'
    else:
        return 'CONTINUE', None
```

**기존 B3 대비 강화된 점**:

- `"block"` 단순 count → `"red block"` attribute-aware 매칭
- `n_target == 0 → STOP` 추가 (기존은 CONTINUE로 S3 오답)
- `n_dest == 0 → ASK` 추가

**그럼에도 못 잡는 케이스**:

- S6 (위치 변경): `n_target == 1` 유지 → wrong CONTINUE  
  → G₀ coord가 있어야 "같은 label이지만 다른 instance"를 감지할 수 있음  
  → **이 지점이 Ours의 G₀ coord memory 기여를 가장 선명하게 드러냄**

**예상 실패**: S5(same category distractor → false ASK) / S6(count=1 유지 → wrong CONTINUE)

---

### B4 — Binary Grounding Anomaly Detector

**개념**: G₀(label+coord)를 저장하고 checkpoint에서 coord 비교를 수행한다.
그러나 비교 결과를 **binary(anomaly detected / not detected)** 로만 처리하며,
state를 세분화하지 않아 anomaly이면 무조건 ASK를 반환한다.

이 baseline이 검증하는 질문:

> **"taxonomy 없이 anomaly만 감지하면 충분한가?"**

| 할 수 있는 것                   | 못하는 것                      |
| ------------------------------- | ------------------------------ |
| G₀와 달라졌는지 감지            | STOP과 ASK 구분                |
| coord 기반 identity 확인        | target/destination별 정책 구분 |
| distractor 무시 (G₀ coord 기준) | state 유형별 적합한 decision   |

```python
THRESHOLD = 50  # pixel, pilot experiment으로 결정

def b4(G0, checkpoint_img, task):
    """Binary Grounding Anomaly Detector: anomaly이면 무조건 ASK"""
    Gt = AmbResVLM_step2(checkpoint_img, task)

    # target coord 비교
    gt_target_coord = Gt.get_coord(G0.target.label)
    if gt_target_coord is None:
        target_valid = False
    else:
        target_valid = euclidean(G0.target.coord, gt_target_coord) < THRESHOLD

    # destination coord 비교
    gt_dest_coord = Gt.get_coord(G0.dest.label)
    if gt_dest_coord is None:
        dest_valid = False
    else:
        dest_valid = euclidean(G0.dest.coord, gt_dest_coord) < THRESHOLD

    # binary 판단: anomaly detected → ASK (state 세분화 없음)
    if not target_valid or not dest_valid:
        return 'ASK', 'Grounding anomaly detected'
    return 'CONTINUE', None
```

**B4 vs Ours — taxonomy의 기여**:

| 시나리오              | B4 (anomaly only)     | Ours (state classification)    | 차이              |
| --------------------- | --------------------- | ------------------------------ | ----------------- |
| S3 target disappeared | anomaly → **ASK**     | INVALID_TARGET → **STOP**      | wrong decision    |
| S5 distractor         | G₀ valid → CONTINUE ✓ | CLEAR(distractor) → CONTINUE ✓ | 동일              |
| S6 위치 변경          | anomaly → ASK ✓       | AMBIGUOUS_TARGET → ASK ✓       | 동일              |
| 일반 dest 없음        | anomaly → ASK         | INVALID_DESTINATION → ASK/STOP | state에 따라 다름 |

S3에서 B4는 ASK를 내고 Ours는 STOP을 낸다.
`INVALID_TARGET`(target 자체가 없어짐)은 더 이상 사용자에게 질문할 필요 없이 실행을 멈추는 게 맞다.
taxonomy가 있어야 이 구분이 가능하다.

---

### Ours — Execution-Aware AmbResVLM (Full System)

**개념**: G₀(label+coord) + checkpoint 재호출 + 8-state taxonomy + C1/C2 분리 감지 + state별 decision.

```python
THRESHOLD = 50

def consistency_monitor(G0, Gt, checkpoint):
    """checkpoint: 'C1' (pre-pick) or 'C2' (pre-place)"""

    if checkpoint == 'C1':  # target 관련
        candidates = Gt.get_all_coords(G0.target.label)

        if len(candidates) == 0:
            return 'INVALID_TARGET'

        nearest = min(candidates, key=lambda c: euclidean(c, G0.target.coord))
        dist = euclidean(nearest, G0.target.coord)

        if dist > THRESHOLD and len(candidates) == 1:
            return 'INVALID_TARGET'      # 위치 완전 불일치
        elif len(candidates) > 1:
            return 'AMBIGUOUS_TARGET'    # 원래 것 + 새 것 추가
        else:
            return 'CLEAR'

    if checkpoint == 'C2':  # destination 관련
        candidates = Gt.get_all_coords(G0.dest.label)

        if len(candidates) == 0:
            return 'INVALID_DESTINATION'

        nearest = min(candidates, key=lambda c: euclidean(c, G0.dest.coord))
        dist = euclidean(nearest, G0.dest.coord)

        if dist > THRESHOLD or len(candidates) > 1:
            return 'AMBIGUOUS_DESTINATION'
        else:
            return 'CLEAR'


DECISION_MAP = {
    'CLEAR':                 'CONTINUE',
    'INVALID_TARGET':        'STOP',        # B4와 다름: ASK 아닌 STOP
    'AMBIGUOUS_TARGET':      'ASK',
    'INVALID_DESTINATION':   'ASK',
    'AMBIGUOUS_DESTINATION': 'ASK',
    'UNSAFE_OR_BLOCKED':     'STOP',
}

def decision_policy(state):
    return DECISION_MAP.get(state, 'STOP')


def clarification_q(state, G0, Gt):
    base_q = Gt.clarifying_question
    if state == 'AMBIGUOUS_TARGET':
        return f'Multiple {G0.target.label}s found. {base_q}'
    elif state == 'AMBIGUOUS_DESTINATION':
        return f'Multiple {G0.dest.label}s found. {base_q}'
    elif state == 'INVALID_DESTINATION':
        return f'Original {G0.dest.label} is no longer visible. Where should I place it?'
    return base_q


def ours_pipeline(initial_img, task):
    # t₀
    out0 = AmbResVLM_step1(initial_img, task)
    if out0['ambiguity']:
        return 'ASK', out0['clarifying_question']
    G0 = AmbResVLM_step2(initial_img, task, answer='')
    G0 = parse_roles(G0, task)

    # C1
    Gt_c1 = AmbResVLM_step2(capture_snapshot(), task, answer='')
    state = consistency_monitor(G0, Gt_c1, 'C1')
    d, q  = decision_policy(state), clarification_q(state, G0, Gt_c1)
    if d != 'CONTINUE': return d, q

    robot.pick(G0.target)

    # C2
    Gt_c2 = AmbResVLM_step2(capture_snapshot(), task, answer='')
    state = consistency_monitor(G0, Gt_c2, 'C2')
    d, q  = decision_policy(state), clarification_q(state, G0, Gt_c2)
    if d != 'CONTINUE': return d, q

    robot.place(G0.dest)
    return 'SUCCESS', None
```

---

## 6. 시나리오 설계

### 시나리오 정의

| #   | 시나리오              | 장면 구성                                    | Gold State            | Gold Decision | 논문 근거                   |
| --- | --------------------- | -------------------------------------------- | --------------------- | ------------- | --------------------------- |
| S1  | Clear continuation    | 변화 없음                                    | CLEAR                 | CONTINUE      | AmbResVLM unambiguous case  |
| S2  | Target 추가           | 동일 label block 1→2개                       | AMBIGUOUS_TARGET      | ASK           | BT "banana 2개 삽입"        |
| S3  | Target disappeared ★  | block 제거 또는 완전 차폐                    | INVALID_TARGET        | STOP          | PAL "No Valid Target"       |
| S4  | Destination 후보 추가 | tray 1→2개                                   | AMBIGUOUS_DESTINATION | ASK           | BT "bowl 2개"               |
| S5  | Distractor 추가 ★     | task 무관 물체 추가                          | CLEAR (distractor)    | CONTINUE      | PAL "Distractor Robustness" |
| S6  | Target 위치 변경 ★    | 원래 block 제거 + 다른 위치에 동일 label 1개 | AMBIGUOUS_TARGET      | ASK           | coord memory 핵심 검증      |

> ★ = 강화된 B3도 틀리고 Ours만 정확한 핵심 차별점 시나리오

### Gold Label 기준 (논문 근거)

| 기준                      | 방법                                | 논문                                   |
| ------------------------- | ----------------------------------- | -------------------------------------- |
| scene construction = gold | 실험자가 장면을 직접 제어           | PAL: induced failure / BT: object 삽입 |
| 후보 수 = label           | 1개 = unambiguous, 2개 = ambiguous  | AmbResVLM                              |
| decision label            | 2인 독립 annotation + Cohen's kappa | —                                      |

### 시나리오 × Baseline 예상 결과

| 시나리오                | Gold Decision | B1  | B2  | B3          | B4          | Ours    |
| ----------------------- | ------------- | --- | --- | ----------- | ----------- | ------- |
| S1 Clear                | CONTINUE      | ✓   | ✓   | ✓           | ✓           | ✓       |
| S2 Target 추가          | ASK           | ✗   | △   | ✓           | ✓           | ✓       |
| S3 Target disappeared ★ | STOP          | ✗   | △   | ✓(STOP)     | ✗(ASK≠STOP) | ✓(STOP) |
| S4 Dest 후보 추가       | ASK           | ✗   | △   | ✓           | ✓           | ✓       |
| S5 Distractor ★         | CONTINUE      | △   | ✗   | ✗(same cat) | ✓           | ✓       |
| S6 위치 변경 ★          | ASK           | ✗   | ✗   | ✗           | ✓           | ✓       |

> ✓ = 정확 / ✗ = 오답 / △ = 불확실 (model-dependent)

**S3 주목**: B3(강화 버전)는 `n_target == 0 → STOP`으로 S3를 맞힌다.
그러나 S6에서 B3는 `n_target == 1` 유지로 wrong CONTINUE를 낸다.
이 차이가 G₀ coord memory의 기여를 선명하게 드러낸다.

**B4 vs Ours 주목 (S3)**: B4는 anomaly 감지 후 ASK, Ours는 INVALID_TARGET → STOP.
taxonomy 없이는 "사라진 것"과 "모호한 것"을 구분할 수 없다.

---

## 7. 평가 지표 (Metrics)

### Primary Metrics

| Metric                | 정의                                              | 산출 방법                           |
| --------------------- | ------------------------------------------------- | ----------------------------------- |
| **Decision Accuracy** | gold decision vs 예측 decision 일치율             | correct / total                     |
| **Miss Rate**         | ASK/STOP 해야 하는데 CONTINUE한 비율              | false_continue / (ASK+STOP samples) |
| **False Alarm Rate**  | CONTINUE해야 하는데 ASK/STOP한 비율               | false_ask_stop / (CONTINUE samples) |
| **State Accuracy**    | gold state vs 예측 state 일치율 (Ours, B4만 해당) | correct_state / total               |

**State Accuracy를 추가하는 이유**: Decision Accuracy만으로는 "올바른 이유로 올바른 결정을 내렸는가"를 알 수 없다. B4가 S3에서 ASK를 냈을 때, gold decision은 STOP이고 gold state는 INVALID_TARGET이다 — decision도 틀리고 state 분류도 없다. State Accuracy가 있어야 "올바른 결정을 내렸지만 이유가 다른 경우"를 분리할 수 있다.

**Miss Rate와 False Alarm Rate 분리 이유**: 두 오류 유형이 비대칭이다. Miss(unsafe continue)는 로봇이 잘못된 object를 집는 위험한 실패이고, False Alarm(불필요한 ASK)은 사용자 불편에 그친다. 단순 accuracy로 합치면 이 차이가 묻힌다. INVIGORATE도 성공률(↑)과 질문 수(↓)를 분리해서 보고했다.

### Secondary Metrics

| Metric                            | 정의                                                          | 적용 조건           |
| --------------------------------- | ------------------------------------------------------------- | ------------------- |
| **ASK Answerability**             | ASK 질문이 단일 응답으로 grounding 업데이트 가능한가 (binary) | Ours만 / human eval |
| **Grounding Update 성공률**       | ASK 후 G₀ 업데이트가 실제로 task를 해결했는가                 | 실제 로봇 연동 시   |
| **AmbResVLM 호출 횟수**           | latency proxy (t₀+C1+C2 = 최대 3회)                           | 이미지 기반         |
| **Decision-to-execution latency** | checkpoint 진입 ~ decision 출력 시간                          | 실제 로봇 연동 시   |

> ✕ 제외: Task Success Rate — 로봇 execution 노이즈(grasp 정밀도, 물리 충돌)가 개입해 파이프라인 비교와 무관

**ASK Quality → Answerability로 단순화한 이유**: 질문 품질의 전반적인 평가는 주관적이라 human evaluation 없이는 신뢰도 있는 metric이 안 된다. "Is the question answerable with a single yes/no or candidate selection?" (binary) 형태로 단순화하면 annotation 부담도 낮고 객관성도 확보된다.

**Execution Time → 이미지 기반에서는 호출 횟수로 대체**: AmbResVLM 호출 횟수가 latency의 주요 결정 요소이므로 이미지 기반 실험에서는 이 수치가 proxy로 충분하다. 실제 로봇 연동 시 PAL Table IV 방식으로 end-to-end latency를 추가 측정한다.

### 실험 반복 설계

- **N = 10회** per scenario per method (PAL 기준)
- 6개 시나리오 × 5개 방법(B1~B4+Ours) × 10회 = 300회
- 이미지 기반: 시나리오당 10장의 서로 다른 장면 세팅

### Inter-annotator Agreement

- decision label(CONTINUE/ASK/STOP)은 2인 독립 라벨링 후 Cohen's kappa 확인
- scene construction = gold이므로 state label은 별도 annotation 불필요

---

## 8. Ablation Study 설계

| Ablation      | 제거 component | 비교 대상               | 검증 질문                  |
| ------------- | -------------- | ----------------------- | -------------------------- |
| C1-only       | C2 제거        | Ours vs C1-only         | C2가 독립적으로 기여하는가 |
| C2-only       | C1 제거        | Ours vs C2-only         | C1이 독립적으로 기여하는가 |
| label-only G₀ | coord 제거     | B4(coord) vs label-only | coord가 기여하는가         |
| no taxonomy   | taxonomy 제거  | B4 vs Ours              | state 세분화가 기여하는가  |

---

## 9. 이후 구현 계획 (Claude Code)

| Step | 내용                                                           | 파일명                   |
| ---- | -------------------------------------------------------------- | ------------------------ |
| 1    | G₀ 추출기: t₀ → Step1+Step2 → label+coord                      | `ambres_g0_extractor.py` |
| 2    | target/destination 파서: task description → 역할 분류          | `role_parser.py`         |
| 3    | Consistency Monitor: G₀ vs Gₜ → 8-state                        | `consistency_monitor.py` |
| 4    | Threshold pilot: coord 변화량 측정                             | `pilot_threshold.py`     |
| 5    | 전체 파이프라인 통합 + 단위 테스트                             | `pipeline.py` + `tests/` |
| 6    | Baseline 구현: B1, B2, B3(attribute-aware), B4(binary anomaly) | `baselines/`             |
| 7    | 실험 실행 + metrics 계산                                       | `evaluate.py`            |
| 8    | Ablation 실험                                                  | `ablation.py`            |

---

## 10. 현재 진행 상태

| Phase                         | 상태      | 완료 항목                                                              |
| ----------------------------- | --------- | ---------------------------------------------------------------------- |
| Phase 0: Research Design      | ✅ 완료   | 주제 확정, 포지셔닝, proposal 개선본(docx)                             |
| Phase 1: System Design        | 🔄 ~85%   | 파이프라인/taxonomy/baseline/metric/pseudocode 확정 / 코드 구현 미착수 |
| Phase 2: Dataset & Experiment | ⬜ 미착수 | —                                                                      |
| Phase 3: Paper Writing        | ⬜ 미착수 | proposal 개선본(docx)만 완성                                           |
