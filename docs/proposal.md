# Execution-Aware Ambiguity Handling for Language-Guided Robotic Manipulation Using AmbResVLM

---

## 1. 연구 한 문장 정의

> AmbResVLM은 현재 장면이 ambiguous한지를 판단하지만, 실행 중 장면 변화로 인해 initially-clear grounding이 무효화되는 것은 감지하지 못한다. 본 연구는 초기 grounding G₀를 메모리에 저장하고 checkpoint 이미지에서 AmbResVLM을 재호출하여, G₀와 Gₜ의 불일치를 grounding-state transition으로 분류하는 최초의 방법이다.

---

## 2. 연구 동기 (Motivation)

VLM 기반 로봇 조작 시스템에서 기존 ambiguity resolution 방법들(KnowNo, CLARA, LAP, AmbResVLM)은 모두 **현재 장면 기준**으로 task가 ambiguous한지 판단한다. 그러나 다음과 같은 상황은 다루지 못한다.

**핵심 예시:**
- Instruction: *"Place the red block in the tray."*
- Initial scene: red block 1개, tray 1개 → AmbResVLM이 **clear** 판정, 실행 시작
- Mid-execution: 로봇이 block을 집고 이동하는 사이 → tray가 다른 물체로 occupied됨
- **문제:** G₀ = {target: red_block, destination: left_tray}가 더 이상 유효하지 않지만, 시스템은 이를 감지하지 못하고 계속 실행

이것은 initial ambiguity가 아니라 **mid-execution ambiguity emergence**다.

---

## 3. 연구 포지셔닝

### 3.1 기존 연구 landscape (2×2 매트릭스)

|  | Pre-execution intervention | Mid-execution intervention |
|---|---|---|
| **Static scene** | KnowNo, CLARA, LAP, AmbResVLM, SG-CoT (매우 많음) | DoReMi, Physical Agentic Loop (controller failure 기반, grounding 아님) |
| **Scene change during execution** | BT Disambiguation (same-category instances 한정) | **← 이 연구** |

### 3.2 핵심 차별점 (관련 연구별)

| 기존 연구 | 한계 | 이 연구의 차이 |
|---|---|---|
| AmbResVLM | 현재 장면 ambiguity만 판단, G₀ 없음 | G₀ 저장 후 Gₜ와 비교 |
| Physical Agentic Loop | gripper telemetry 기반 (post-action) | VLM grounding output 기반 (pre-action) |
| BT Disambiguation | same-category multiple instance만 감지 | occupied/disappeared/safety로 확장 |
| DoReMi | physical failure signal 기반 | semantic grounding validity 기반 |

**PAL과의 핵심 차이 한 문장:**
> PAL은 "방금 action이 물리적으로 성공했는가?"를 묻고, 이 연구는 "처음에 계획한 G₀가 지금도 semantic하게 유효한가?"를 묻는다.

---

## 4. 시스템 설계

### 4.1 전체 파이프라인

```
t₀ (초기)
  ├── Task description + Initial image
  ├── AmbResVLM 호출 (Step 1 + Step 2 강제)
  ├── ambiguity = true → ASK (initial), 종료
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
  PICK 실행 (카메라 off)
           ↓
  로봇 이동 (카메라 off)
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

### 4.2 Checkpoint 타이밍 근거

| Checkpoint | 감지 대상 | 논문 근거 |
|---|---|---|
| C1 (pre-pick) | target 관련 변화 (AMBIGUOUS_TARGET, INVALID_TARGET) | BT: pick 전 Scene Clear? condition / PAL: Observe → Act |
| C2 (pre-place) | destination 관련 변화 (OCCUPIED, AMBIGUOUS, INVALID) | BT: "destination disambiguation first" 원칙 / PAL: 비가역 action 직전 |
| 둘 다 | CLEAR, UNSAFE_OR_BLOCKED | PAL: 매 transition point safety check |

> BT Disambiguation 논문 명시: "manipulation 중에는 detection을 timed out" → 로봇이 움직이는 동안 카메라는 off

### 4.3 Grounding Consistency Monitor

**G₀ 표현 방식:** label + coord (AmbResVLM Step 2 output 활용)

```json
G₀ = {
  "target": {"label": "red block", "coord": [142.0, 318.5]},
  "destination": {"label": "tray", "coord": [401.0, 289.0]},
  "image_shape": [480, 640]
}
```

**label-only 방식의 문제점:**
1. 동일 카테고리 내 개별 instance 구분 불가 (원래 block이 사라지고 새 block이 추가돼도 "있음"으로 판단)
2. object_list에 target/destination 역할 구분 없음
3. occupied 상태를 object_list만으로 판단 불가

**coord 비교 로직:**
```python
# identity 확인
distance = euclidean(G0.target.coord, Gt["red block"].coord)
if distance > THRESHOLD:
    state = AMBIGUOUS_TARGET  # 위치가 다른 → 다른 instance

# multiple instance 처리
all_coords = Gt.get_all_coords("red block")
if len(all_coords) > 1:
    nearest = min(all_coords, key=lambda c: euclidean(c, G0.target.coord))
    state = AMBIGUOUS_TARGET  # 원래 것 + 새 것 추가됨
```

> THRESHOLD: object bounding box 대각선 크기 기준 (pilot experiment으로 결정)

### 4.4 Grounding State Taxonomy

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

> ⚠️ OCCUPIED_DESTINATION은 현재 scope 외 (이후 단계에서 추가)

### 4.5 G₀ 메모리 업데이트 방식

- **Rolling update:** user answer 후 AmbResVLM Step 2 → 새 G₀로 overwrite
- C1에서 user answer로 target 확정 → 새 G₀로 저장 → C2에서 그 G₀ 기준으로 destination 확인
- BT 논문 순서 원칙 적용: destination disambiguation → target disambiguation

---

## 5. 실험 설계

### 5.1 시나리오 (5개, OCCUPIED 제외)

| # | 시나리오 | Gold State | Gold Decision | B1 | B2 | B3 |
|---|---|---|---|---|---|---|
| ① | Clear continuation | CLEAR | CONTINUE | ✓ | ✓ | ✓ |
| ② | Same-category target 추가 | AMBIGUOUS_TARGET | ASK | △ | ✓ | ✓ |
| ③ | Target disappeared | INVALID_TARGET | STOP | ✗ | △ | ✗ |
| ④ | New destination candidate | AMBIGUOUS_DESTINATION | ASK | △ | ✓ | ✓ |
| ⑤ | Distractor added (무관한 물체) | CLEAR | CONTINUE | ✗ | ✗ | △ |

> ★ 시나리오 ②와 ⑤가 제안 방법의 차별점을 가장 잘 드러내는 케이스

### 5.2 Baseline

| | 방법 | G₀ 메모리 | 설명 |
|---|---|---|---|
| B1 | AmbResVLM single-shot | 없음 | checkpoint에서 현재 장면만 판단 |
| B2 | AmbResVLM repeated | 없음 | 매 checkpoint 재호출, G₀ 비교 없음 |
| B3 | Candidate count rule | 없음 | 후보 개수 > 1이면 ASK (BT 방식) |
| B4 | Ours w/o taxonomy | 있음 | G₀ 비교는 하지만 binary (valid/invalid)만 |
| **Ours** | Execution-Aware AmbResVLM | **있음** | G₀ + coord + taxonomy + decision policy |

### 5.3 Gold Label 기준 (논문 근거)

scene construction 자체가 gold label:
- **AmbResVLM 논문:** 후보 1개 = unambiguous, 2개 = ambiguous → scene setup = label
- **Physical Agentic Loop:** induced empty grasp = EMPTY label → 실험자가 장면 제어
- **BT Disambiguation:** 실행 중 banana 삽입 = ASK label → transition 자체 = gold

### 5.4 평가 지표

| Metric | 정의 |
|---|---|
| Grounding State Accuracy | gold state vs 예측 state 일치율 |
| Decision Accuracy | CONTINUE/ASK/STOP 일치율 |
| False Alarm Rate | CLEAR 상황에서 ASK 출력한 비율 |
| Miss Rate | ASK/STOP 상황에서 CONTINUE 출력한 비율 |
| C1 vs C2 contribution | C1-only / C2-only / both ablation |
| Coord vs label-only | B4 vs Ours accuracy 차이 |

---

## 6. 설계 결정 사항 정리

### 6.1 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| Checkpoint 타이밍 | Fixed checkpoint (C1, C2) | BT: pre-condition / PAL: pre-irreversible |
| G₀ 표현 | label + coord | label-only → identity 구분 불가 |
| t₀ Step 2 강제 호출 | 빈 answer로 호출하여 좌표 확보 | AmbResVLM Step 2 output 구조 |
| User answer 처리 | Single-shot update (rolling G₀) | AmbResVLM Step 2 → 새 G₀ overwrite |
| STOP 범위 | Conservative (INVALID만 STOP) | 나머지는 ASK → 사용자 판단 위임 |
| Disambiguation 순서 | destination 먼저, target 나중 | BT 논문 명시: 반대 순서는 noise |
| 질문 생성 | AmbResVLM output + G₀ context enrichment | Step 1의 clarifying_question 활용 |

### 6.2 논의된 대안 (미채택, future work)

| 항목 | 대안 | 미채택 이유 |
|---|---|---|
| 타이밍 | Event-driven (scene change 감지 시) | lightweight tracker 추가 구현 필요 |
| G₀ 표현 | Belief distribution (INVIGORATE 방식) | 구현 복잡도, 4주 범위 초과 |
| User answer | Elimination-based (INVIGORATE 방식) | user answer 형태 제한 필요 |
| STOP | Bounded retry (PAL 방식) | 현재 scope 내 필요성 낮음 |

---

## 7. 이후 실행 계획

### Phase 1 마무리 — 코드 구현 (Claude Code)

| Step | 내용 | 파일명 |
|---|---|---|
| 1 | G₀ 추출기: t₀ → Step1+Step2 → label+coord 저장 | `ambres_g0_extractor.py` |
| 2 | target/destination 파서: object_list + task → 역할 분류 | `role_parser.py` |
| 3 | Consistency Monitor: G₀ vs Gₜ → state taxonomy | `consistency_monitor.py` |
| 4 | Threshold 결정용 pilot: 이미지 5~10장 coord 변화량 측정 | `pilot_threshold.py` |
| 5 | 전체 파이프라인 통합: t₀ → C1 → C2 end-to-end | `pipeline.py` + `tests/` |
| 6 | Baseline 구현: B1, B2, B3, B4 | `baselines/b1_b2_b3_b4.py` |

**Claude Code 첫 번째 프롬프트 (Step 1~2):**
```
Context:
- AmbResVLM 실행 가능한 상태
- Step 1 output: {object_list, ambiguity, explanation, clarifying_question}
- Step 2 output: {object_list, label: [x, y], ...}

Task:
1. ambres_g0_extractor.py
   - 입력: image_path, task_description
   - Step 1 호출 → ambiguity=false 확인
   - Step 2 강제 호출 (빈 answer)로 좌표 획득
   - 출력: G0 = {target: {label, coord}, destination: {label, coord}, image_shape}

2. role_parser.py
   - 입력: object_list, task_description
   - heuristic: 동사 뒤 = target, in/on/to 뒤 = destination
   - 출력: {target: str, destination: str}

3. 이미지 1장으로 end-to-end 테스트
```

### Phase 2 — Dataset 구축 및 실험

| Step | 내용 | 비고 |
|---|---|---|
| 7 | 실제 이미지 촬영 | 5 시나리오 × 2 checkpoint × 5회 = 약 50장 |
| 8 | Annotation + inter-annotator check (Cohen's kappa) | `dataset/annotator.py` |
| 9 | B1~Ours 전체 실험 + metrics 계산 | `evaluate.py` |
| 10 | Ablation (C1-only, C2-only, label-only G₀) | `ablation.py` |

### Phase 3 — 논문 작성

- Method 섹션: 설계 완료, 수치만 채우면 됨
- Experiment 섹션: Phase 2 결과 직접 반영
- **Figure 1:** 시나리오 ② (same-category target 추가) — motivating example
- **핵심 claim:** B2(no memory) 대비 wrong continue 감소, B1 대비 false alarm 감소

---

## 8. 현재 진행 상태

| Phase | 상태 | 완료 항목 |
|---|---|---|
| Phase 0: Research Design | ✅ 완료 | 주제 확정, 포지셔닝, proposal 개선본 |
| Phase 1: System Design | 🔄 70% | 파이프라인, taxonomy, baseline, gold label 기준 확정 / 코드 구현 미착수 |
| Phase 2: Dataset & Experiment | ⬜ 미착수 | — |
| Phase 3: Paper Writing | ⬜ 미착수 | proposal 개선본(docx)만 완성 |