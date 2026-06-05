# Fine-tuning 데이터 정리

**작성일**: 2026-05-29  
**프로젝트**: COSE461 Change-Aware Grounding Pipeline

---

## 1. 원본 데이터: vla-evaluation-v3 (학습용)

| 항목 | 내용 |
|------|------|
| 총 trial 수 | **80개** (S1×20 + S2~S7 각 10개) |
| 객체 조합 | cup+red box (35), cube+red box (35), cup+box (5), cube+box (5) |
| Primary checkpoint | C1: 50개, C2: 30개 |
| 수동 annotation | **80개 전체** — checkpoint 시점 target/dest 좌표를 직접 레이블링 |

### 시나리오 정의

| ID | Gold State | Decision | 설명 |
|----|-----------|----------|------|
| S1 | CLEAR | CONTINUE | 물체 정위치, 변화 없음 (×20) |
| S2 | AMBIGUOUS_TARGET | ASK | target 2개 동시 존재 |
| S3 | INVALID_TARGET | STOP | target 제거됨 |
| S4 | AMBIGUOUS_TARGET | ASK | target 이동 (count=1, 위치 변경) |
| S5 | AMBIGUOUS_DESTINATION | ASK | destination 2개 동시 존재 |
| S6 | INVALID_DESTINATION | STOP | destination 제거됨 |
| S7 | AMBIGUOUS_DESTINATION | ASK | destination 이동 (count=1, 위치 변경) |

---

## 2. Fine-tuning Dataset v2 구성

**Train/Val split: trial 단위 80/20 분리**  
동일 trial의 augmentation이 train/val에 동시에 들어가는 data leakage 방지.

| 구분 | Trial 수 | Example 수 |
|------|---------|-----------|
| Train | 64 | **512개** |
| Val | 16 | **16개** (base only, augmentation 없음) |

### Train 512개 Breakdown

| 타입 | 수 | 설명 |
|------|----|----|
| base | 64 | 원본 이미지 + 수동 annotation 좌표 |
| coord noise ×2 | 128 | G₀/ck 좌표 ±5px Gaussian perturbation |
| h-flip | 64 | 좌우 반전 이미지 + 좌표 반전 |
| brightness ×2 | 128 | 밝기 ×0.7 (dark), ×1.3 (bright) |
| **no-G₀** | **64** | G₀ 좌표 제거, count rule로 레이블 재지정 |
| **counterfactual** | **64** | 동일 이미지에서 G₀만 변경 (cf_same 32 + cf_shifted 32) |
| **합계** | **512** | |

### Gold State 분포 (train+val 528개)

| State | 수 |
|-------|----|
| CLEAR | 164 |
| AMBIGUOUS_TARGET | 120 |
| AMBIGUOUS_DESTINATION | 128 |
| INVALID_TARGET | 52 |
| INVALID_DESTINATION | 64 |
| **합계** | **528** |

---

## 3. 주요 설계 결정

### 수동 Annotation 사용 이유
DINO joint query ("cup"+"red box") 시 두 물체가 형태 유사할 경우 label competition 발생.  
S3 cube 시나리오에서 red box를 cube로 오탐지하는 문제가 반복 확인됨.  
→ checkpoint 좌표를 사람이 직접 레이블링해 학습 신호의 정확도 보장.

### no-G₀ Samples
v1 모델이 "G₀ 없어도 count=1이면 무조건 ASK" shortcut을 학습하는 문제 발견(S1 CLEAR 0/10).  
→ count=1이면 locally clear로 처리하도록 재레이블링.  
G₀ 없이는 이동 판정이 불가능함을 명시하는 것이 목적이며, G₀ 없는 상황에서도 이동이 실제로 없다는 의미는 아님.

**count rule:**
- count = 0 → STOP (object absent)
- count = 1 → CLEAR → CONTINUE (위치 비교 불가, locally clear 처리)
- count > 1 → ASK (multiple candidates)

### Counterfactual G₀ Pairs
동일 이미지 + 동일 ck 좌표에서 G₀ 위치만 변경한 대조 샘플 쌍.

| 타입 | G₀ 설정 | Gold Label |
|------|---------|-----------|
| cf_same | G₀ = ck 현재 좌표 | CLEAR → CONTINUE |
| cf_shifted | G₀ = ck + 200px | AMBIGUOUS → ASK |

count만 보는 shortcut이 아닌 G₀ 좌표 비교를 강제로 학습시키기 위한 설계.  
효과: MRR v1 73.2% → v2 100% (+26.8%).

---

## 4. Holdout Dataset: vla-evaluation-v4 (평가용)

| 항목 | 내용 |
|------|------|
| 총 trial 수 | **70개** (S1~S7 각 10개) |
| 학습 데이터와 분리 | ✓ (별도 recording session) |
| 객체/task 구성 | cup+red box (30), cube+red box (30), cup+box (5), cube+box (5) |
| gold_state 분포 | CLEAR 10, AMBIGUOUS_TARGET 20, AMBIGUOUS_DESTINATION 20, INVALID_TARGET 10, INVALID_DESTINATION 10 |

---

## 5. 파일 경로

| 파일 | 내용 |
|------|------|
| `dataset/manifest_train_v3.jsonl` | 학습용 manifest (80 trials) |
| `dataset/manual_annotations.json` | 수동 annotation 좌표 (80 trials) |
| `dataset/finetune_v2/train.jsonl` | v2 학습 데이터 (512개) |
| `dataset/finetune_v2/val.jsonl` | v2 검증 데이터 (16개) |
| `dataset/vla-evaluation-v3/` | 원본 학습 이미지 |
| `dataset/vla-evaluation-v4/` | holdout 평가 이미지 |
| `dataset/vla-evaluation-v4/manifest_eval-v4.jsonl` | 평가용 manifest (70 trials) |
