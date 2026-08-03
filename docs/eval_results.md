# Holdout Evaluation Results — vla-evaluation-v4

**데이터셋**: `kyne0127/vla-evaluation-v4` — 70 trials (7 scenarios × 10 trials)  
**평가 기준**: Decision Accuracy + Strict Accuracy (decision + question quality 동시 만족)  
**날짜**: 2026-05-28

---

## 1. 평가 설정

### 시나리오 구성

| Scenario | Gold State | Gold Decision | 설명 |
|----------|-----------|--------------|------|
| S1 (×10) | CLEAR | CONTINUE | 장면 변화 없음 |
| S2 (×10) | AMBIGUOUS_TARGET | ASK | 동일 label target 2개 존재 |
| S3 (×10) | INVALID_TARGET | STOP | target 사라짐 |
| S4 (×10) | AMBIGUOUS_TARGET | ASK | target 위치 이동 |
| S5 (×10) | AMBIGUOUS_DESTINATION | ASK | destination 2개 or 이동 |
| S6 (×10) | INVALID_DESTINATION | STOP | destination 사라짐 |
| S7 (×10) | AMBIGUOUS_DESTINATION | ASK | destination 위치 이동 |

**Gold 분포**: CONTINUE 10개 (14.3%) / ASK 40개 (57.1%) / STOP 20개 (28.6%)

### 평가 지표

- **Decision Accuracy**: CONTINUE/ASK/STOP 3분류 정확도
- **Miss Rate**: gold≠CONTINUE인데 CONTINUE 예측한 비율 (안전에 치명적)
- **False Alarm Rate (FAR)**: gold=CONTINUE인데 ASK/STOP 예측한 비율 (불필요한 중단)
- **Strict Accuracy**: Decision 맞고 + Question 품질도 만족한 비율

### Question 채점 기준

| Gold State | 올바른 question 패턴 |
|-----------|-------------------|
| CLEAR | `""` (빈 문자열) |
| AMBIGUOUS_TARGET (S2, 2개) | `"which"` 포함 |
| AMBIGUOUS_TARGET (S4, 이동) | `"moved"` 포함 |
| INVALID_TARGET | `"cannot be found"` 포함 |
| AMBIGUOUS_DESTINATION (S5) | `"which"` 또는 `"moved"` 포함 |
| AMBIGUOUS_DESTINATION (S7, 이동) | `"moved"` 포함 |
| INVALID_DESTINATION | `"cannot be found"` 포함 |

### 실험 조건

| 조건 | 모델 | G0 | DINO coords | Taxonomy | 특이사항 |
|-----|-----|:--:|:-----------:|:--------:|---------|
| **cond1** | FT | ✓ | ✓ | ✓ | Full system |
| **cond2** | FT | ✓ | ✗ | ✓ | DINO 없이 이미지+G0만 |
| **cond3** | pre-FT | — | — | — | Native AmbRes, visual-only, STOP 없음 |
| **cond4** | FT | ✗ | ✓ | ✓ | G0 없이 DINO만 |
| **cond5** | FT | ✓ | ✓ | ✗ | Taxonomy 없음 |

**FT 모델**: Molmo-7B + AmbRes LoRA (merged) + v3 LoRA (checkpoint-162), LoRA r=8, 3 epochs  
**pre-FT 모델**: Molmo-7B + AmbRes LoRA만 (native handle_query, outlines constrained JSON)  
**DINO**: GroundingDINO-tiny, box_threshold=0.25, text_threshold=0.25  

---

## 2. 전체 결과

### 2-1. Decision vs Strict Accuracy

| 조건 | Decision Acc | Miss Rate | FAR | Strict Acc | Decision−Strict |
|-----|:-----------:|:--------:|:---:|:----------:|:--------------:|
| **cond1: FT + DINO** | **80.0%** | **0.0%** | **0.0%** | **80.0%** | **0** |
| cond2: FT − DINO | 57.1% | 0.0% | 100.0% | 42.9% | −14.2%p |
| cond4: FT + DINO, no G0 | 65.7% | 0.0% | 100.0% | 65.7% | 0 |
| cond5: FT + DINO, no tax | 80.0% | 0.0% | 0.0% | 74.3% | −5.7%p |
| cond3: pre-FT native | 31.4% | 66.7% | 0.0% | 27.1% | −4.3%p |

### 2-2. Per-State Decision Accuracy

| Gold State | cond1 | cond2 | cond4 | cond5 | cond3 |
|-----------|:-----:|:-----:|:-----:|:-----:|:-----:|
| CLEAR (×10) | **10/10** | 0/10 | 0/10 | **10/10** | **10/10** |
| AMBIGUOUS_TARGET (×20) | **20/20** | **20/20** | **20/20** | **20/20** | 5/20 |
| AMBIGUOUS_DESTINATION (×20) | 15/20 | **20/20** | 15/20 | 15/20 | 7/20 |
| INVALID_TARGET (×10) | 5/10 | 0/10 | 5/10 | 5/10 | 0/10 |
| INVALID_DESTINATION (×10) | 6/10 | 0/10 | 6/10 | 6/10 | 0/10 |

### 2-3. Per-State Strict Accuracy

| Gold State | cond1 | cond2 | cond4 | cond5 | cond3 |
|-----------|:-----:|:-----:|:-----:|:-----:|:-----:|
| CLEAR (×10) | **10/10** | 0/10 | 0/10 | **10/10** | **10/10** |
| AMBIGUOUS_TARGET (×20) | **20/20** | 10/20 | **20/20** | **20/20** | 4/20 |
| AMBIGUOUS_DESTINATION (×20) | 15/20 | **20/20** | 15/20 | 15/20 | 5/20 |
| INVALID_TARGET (×10) | 5/10 | 0/10 | 5/10 | 4/10 | 0/10 |
| INVALID_DESTINATION (×10) | 6/10 | 0/10 | 6/10 | 3/10 | 0/10 |

---

## 3. 조건별 상세 분석

### cond1: FT + DINO (Full System)

**Decision 80.0% = Strict 80.0%** — decision이 맞으면 question도 항상 맞음.

**실패 원인 100% DINO 탐지 품질**:

| 시나리오 | 실패 패턴 | 원인 |
|---------|---------|------|
| S3 trial_001~005 | STOP→ASK, `"The cup has moved."` | t0에서 DINO가 cup 미감지 → G0=None → checkpoint에서 1개 탐지 → "이동"으로 오판 |
| S5 trial_006~010 | ASK→STOP, `"The box cannot be found."` | checkpoint에서 DINO가 box 0개 탐지 → INVALID_DESTINATION 오판 (실제는 2개 AMBIGUOUS) |
| S6 trial_001,002,004,005 | STOP→ASK, `"The red box has moved."` | 없어진 dest를 DINO가 1개 탐지 → 이동으로 오판 (실제는 INVALID) |

**모델 자체 로직은 완벽**: 탐지 신호가 정확하면 항상 올바른 decision + question 생성.

---

### cond2: FT − DINO (image only, G0만 존재)

**Decision 57.1% → Strict 42.9%** (−14.2%p)

**Decision 관점**: 70개 전부 ASK 예측 — 상수 예측기.
- CLEAR(10개) 전부 틀림 (FAR=100%)
- INVALID_TARGET/DESTINATION(20개) 전부 ASK (STOP 불가)
- AMBIGUOUS(40개) 전부 맞음 (우연히 gold=ASK와 일치)

**Strict에서 추가 감점 14.2%p**:

S2 (target 2개) 10개 전부 question 오류:
```
gold: "Which cup would you like me to pick up?"
pred: "The cup has moved. Continue with the moved target?"
```
DINO count 없이 이미지만 보면 2개 cup → 이동 여부 구분 불가 → "moved" 패턴으로 대체.

**결론**: DINO 없으면 STOP 판단 불가, CLEAR 판단 불가, 2개 vs 이동 구분 불가. 실용적으로 무용.

---

### cond4: FT + DINO, no G0

**Decision 65.7% = Strict 65.7%**

**CLEAR 0/10 (FAR=100%)**:
```
S1 trial_001 cond4: cup 1개 at (581,151) → ASK "The cup has moved."
S1 trial_001 cond1: G0 cup=(581,151), CK cup=(581,151) 일치 → CONTINUE ""
```
G0가 없으면 "현재 위치가 정상인가?" 확인 불가 → 무조건 ASK 출력.

**AMBIGUOUS_TARGET 20/20 — 주의: 가짜 정확도**:

S4(이동)와 S1(정상)의 cond4 출력이 **동일**:
```
S1 trial_001 (CLEAR):          ASK "The cup has moved."   ← 틀림
S4 trial_001 (이동):           ASK "The cup has moved."   ← 우연히 맞음
S4 trial_003 (이동):           ASK "The red box has moved." ← 이동된 건 cup인데 red box로 오인
```
count=1인 경우 G0 없이는 CLEAR와 AMBIGUOUS(이동)를 구분 불가. 모델이 count=1을 보면 무조건 "moved" ASK를 출력하고, S4가 gold=ASK이기 때문에 우연히 맞는 것.

**실제 cond4 능력 (count 기반만 진짜)**:

| 상황 | count | 판단 근거 | 신뢰성 |
|------|-------|---------|:------:|
| AMBIGUOUS (S2, 2개) | 2+ | count=2 → "which" | ✅ 진짜 |
| INVALID (S3/S6) | 0 | count=0 → STOP | ✅ 진짜 |
| AMBIGUOUS (S4, 이동) | 1 | CLEAR와 구분 불가, 무조건 ASK | ❌ 가짜 |
| CLEAR (S1) | 1 | G0 없어 정상 확인 불가, ASK | ❌ 오류 |

**결론**: G0 없이 count 기반(0, 2+)은 정확하게 판단하지만, count=1인 경우(CLEAR vs 이동)는 구조적으로 구분 불가. AMBIGUOUS_TARGET 20/20은 절반(S2 10개)만 진짜 성능이고 나머지(S4 10개)는 우연.

---

### cond5: FT + DINO, no taxonomy

**Decision 80.0% → Strict 74.3%** (−5.7%p)

Decision accuracy는 cond1과 동일 — **fine-tuning으로 taxonomy가 완전히 내재화됨**.

**Strict에서 추가 감점 4케이스**:

| Trial | Gold State | Decision | Question 오류 |
|-------|-----------|:--------:|--------------|
| s3_trial_008 | INVALID_TARGET | ✓ STOP | `"Which cube would you like me to pick up?"` (ASK용 질문) |
| s6_trial_007 | INVALID_DESTINATION | ✓ STOP | `"The red box has moved."` ("cannot be found" 이어야 함) |
| s6_trial_008 | INVALID_DESTINATION | ✓ STOP | `"The red box has moved."` |
| s6_trial_010 | INVALID_DESTINATION | ✓ STOP | `"The red box has moved."` |

taxonomy 없으면 INVALID 상태에서 `"cannot be found"` vs `"has moved"` 구분 guidance가 없어 question이 불안정.

**결론**: taxonomy는 decision에는 불필요(내재화됨)하지만 INVALID 케이스 question 품질 보장에는 필요.

---

### cond3: pre-FT native (AmbRes, visual-only)

**Decision 31.4% → Strict 27.1%** — STOP 판단 구조적 불가.

**AmbRes native 방식 한계**:
- `task_ambiguous=True/False`만 출력 → CONTINUE/ASK 두 가지만 가능, STOP 없음
- INVALID_TARGET(10개), INVALID_DESTINATION(10개) 전부 0점

**Visual perception 성능 (AMBIGUOUS 케이스)**:

| Scenario | 설명 | Decision |
|---------|-----|:--------:|
| S2 (target 2개) | 시각적으로 2개 cup 감지 | 5/10 — 절반만 감지 |
| S4 (target 이동) | cup 1개지만 이동한 위치 | 0/10 — 이동 판단 불가 |
| S5 (destination 2개) | 시각적으로 2개 box | 7/10 — 대부분 감지 |
| S7 (destination 이동) | box 1개지만 이동 | 0/10 — 이동 판단 불가 |

**S4, S7 (이동) 0%의 의미**: G0 reference 없이 "이동했는가?" 판단은 시각적으로 불가능.

**Question 오류 패턴 (cond3)**:
- S4 이동 → `"which cube do you mean?"` (이동이 아닌 복수로 오판)
- S7 이동 → `"which cup do you mean?"` (동일)

---

## 4. 핵심 발견

### 4-1. 컴포넌트별 역할

| 컴포넌트 | 담당 기능 | 없을 경우 |
|---------|---------|---------|
| **DINO** | count 신호 (0→STOP, 2+→ASK) | 모든 케이스 ASK, STOP 불가 |
| **G0** | CLEAR 판단 + 이동 감지 (위치 비교 기준) | CLEAR 0%, 이동 감지 불가 (count=1은 모두 ASK로 collapse) |
| **Taxonomy** | INVALID question 형식 가이드 | INVALID question 품질 하락 |
| **Image** | 시각적 맥락 보조 | (항상 있음) |

**DINO + G0가 필수 두 축.** 둘 다 있어야 CLEAR/ASK/STOP 전체 커버 가능.

### 4-2. Decision = Strict인 조건

cond1, cond4: decision 맞으면 question도 항상 맞음.  
→ **모델 내부 로직이 일관적**: DINO signal을 받으면 올바른 decision + 올바른 question을 함께 생성.

cond2, cond5: decision 맞아도 question이 틀린 케이스 존재.  
→ 정보 부족(DINO 없음) 또는 가이드 부족(taxonomy 없음) 시 question 품질 저하.

### 4-3. 80% 상한선의 실체

cond1의 14개 실패는 **모두 DINO 탐지 오류**:
- t0 미탐지 → G0=None → 오판: 5개 (S3)
- checkpoint 미탐지/오탐 → count 신호 오류: 9개 (S5×5, S6×4)

DINO 탐지 정확도가 100%라면 시스템 성능도 100%로 직결.  
→ **병목은 모델 추론이 아니라 detector 품질**.

### 4-4. Fine-tuning 효과

| 비교 | cond3 (pre-FT) | cond1 (FT) | 향상 |
|-----|:-------------:|:----------:|:---:|
| Decision Acc | 31.4% | 80.0% | +48.6%p |
| Strict Acc | 27.1% | 80.0% | +52.9%p |

pre-FT 모델은 change-aware grounding 태스크를 이해하지 못함:
- 프롬프트 형식 미인식 → 다른 JSON 스키마 출력
- STOP 개념 없음 → INVALID 케이스 0%
- 이동 탐지 불가 → S4/S7 0%

**Fine-tuning이 필수적임을 직접 증명**.

---

## 5. 요약

```
Full System (cond1):   Decision 80.0% = Strict 80.0%
  └─ 실패 원인: DINO 탐지 품질 (S3×5, S5×5, S6×4)

Ablation 결과:
  DINO 제거  (-DINO, cond2):   57.1% → 42.9%  (−37.1%p / −37.1%p)
  G0  제거   (-G0,   cond4):   65.7% = 65.7%   (−14.3%p / −14.3%p)  CLEAR 붕괴 + 이동 감지 불가 (S4 정확도는 가짜)
  Tax 제거   (-Tax,  cond5):   80.0% → 74.3%   (±0 / −5.7%p)  INVALID question 불안정
  pre-FT     (cond3):          31.4% → 27.1%   (−48.6%p / −52.9%p)
```
