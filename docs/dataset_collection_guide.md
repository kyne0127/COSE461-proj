# Dataset Collection Guide — Evaluation Dataset (S1~S6)

> 이 문서는 `src/evaluate.py` pipeline 평가를 위한 실제 이미지 데이터셋 수집 절차를 정리한다.  
> SmolVLA 학습용 demo 수집이 아닌, **B1~B5+Ours 방법론 평가용 이미지** 수집이 목적이다.

---

## 1. 물리 오브젝트

| 역할 | 오브젝트 | 수량 |
|------|---------|------|
| Target | 종이컵 (동일) | 2개 |
| Target | 큐브 (동일) | 2개 |
| Destination | 빨간 박스 | 1개 |
| Destination | 노란 박스 | 1개 |

---

## 2. 촬영 도구

```bash
python scripts/record_scenario.py \
    --scenario <S1~S6> \
    --task "<task description>" \
    --target-label <cup|cube> \
    --destination-label <box|red box> \
    --out-dir <저장할_폴더>          # 생략 시 기본값: data-evaluation/
    --append-manifest
```

| 키 | 동작 |
|----|------|
| `SPACE` | t₀ 초기 장면 캡처 + 기록 시작 |
| `1` | C1 checkpoint 캡처 |
| `2` | C2 checkpoint 캡처 |
| `r` | 현재 trial 재시작 |
| `q` | 저장 후 종료 |

---

## 3. 시나리오별 세팅 절차

### 공통 초기 상태 (t₀)

모든 시나리오는 **명확한 초기 장면**으로 시작한다.  
target 1개 + destination 1개, 겹치지 않게 배치.

---

### S1 — 변화 없음

- **Gold**: `CLEAR` → `CONTINUE`
- **Checkpoint**: C1

```
t₀  컵 1개 + 빨간박스 1개
    ↓  (개입 없음)
C1  컵 1개 + 빨간박스 1개  (동일)
C2  컵 1개 + 빨간박스 1개  (동일)
```

> 장면을 그대로 유지한 채 SPACE → 1 → 2 → q 순서로 촬영.

**Cup (컵 5 trial)**
```bash
python scripts/record_scenario.py \
    --scenario S1 \
    --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --append-manifest
```

**Cube (큐브 5 trial)**
```bash
python scripts/record_scenario.py \
    --scenario S1 \
    --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --append-manifest
```

---

### S2 — 동일 target 추가

- **Gold**: `AMBIGUOUS_TARGET` → `ASK`
- **Checkpoint**: C1

```
t₀  컵 1개(위치 A) + 빨간박스 1개
    ↓  [C1 직전: 동일한 컵 1개를 위치 B에 추가]
C1  컵 2개 + 빨간박스 1개
C2  컵 2개 + 빨간박스 1개
```

> 추가하는 컵은 원래 컵에서 충분히 떨어진 위치에 놓는다 (너무 붙이면 1개처럼 보임).

**Cup (컵 5 trial)**
```bash
python scripts/record_scenario.py \
    --scenario S2 \
    --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --append-manifest
```

**Cube (큐브 5 trial)**
```bash
python scripts/record_scenario.py \
    --scenario S2 \
    --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --append-manifest
```

---

### S3 — target 제거

- **Gold**: `INVALID_TARGET` → `STOP`
- **Checkpoint**: C1

```
t₀  컵 1개 + 빨간박스 1개
    ↓  [C1 직전: 컵을 씬에서 제거]
C1  (컵 없음) + 빨간박스 1개
C2  (컵 없음) + 빨간박스 1개  ← C1과 동일 장면으로 촬영
```

> C1에서 STOP이므로 C2는 실질적 의미 없음. 같은 장면 그대로 `2` 눌러서 저장.

**Cup (컵 5 trial)**
```bash
python scripts/record_scenario.py \
    --scenario S3 \
    --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --append-manifest
```

**Cube (큐브 5 trial)**
```bash
python scripts/record_scenario.py \
    --scenario S3 \
    --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --append-manifest
```

---

### S4 — destination 추가

- **Gold**: `AMBIGUOUS_DESTINATION` → `ASK`
- **Checkpoint**: C2

```
t₀  컵 1개 + 빨간박스 1개
C1  컵 1개 + 빨간박스 1개  (C1은 변화 없음)
    ↓  [C2 직전: 노란박스 1개 추가]
C2  컵 1개 + 빨간박스 1개 + 노란박스 1개
```

> task에 "red box"라고 쓰면 노란박스를 추가해도 destination이 명확해져서 S4가 성립하지 않는다.  
> 반드시 "box"로만 표기.

**Cup (컵 5 trial)** ← destination-label도 반드시 `box`
```bash
python scripts/record_scenario.py \
    --scenario S4 \
    --task "pick the cup and put it in the box" \
    --target-label cup --destination-label box \
    --append-manifest
```

**Cube (큐브 5 trial)**
```bash
python scripts/record_scenario.py \
    --scenario S4 \
    --task "pick the cube and put it in the box" \
    --target-label cube --destination-label box \
    --append-manifest
```

---

### S5 — 무관 물체 추가 (distractor)

- **Gold**: `CLEAR` → `CONTINUE`
- **Checkpoint**: C1

```
t₀  컵 1개 + 빨간박스 1개
    ↓  [C1 직전: 큐브 1개 추가 (task와 무관)]
C1  컵 1개 + 빨간박스 1개 + 큐브 1개
C2  컵 1개 + 빨간박스 1개 + 큐브 1개
```

> 큐브는 task 오브젝트(컵, 빨간박스)와 다른 카테고리여야 distractor로 인식됨.  
> 큐브를 추가해도 target/destination은 여전히 각 1개 → CLEAR 유지.

**Cup (컵 5 trial)** — distractor: 큐브
```bash
python scripts/record_scenario.py \
    --scenario S5 \
    --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --append-manifest
```

**Cube (큐브 5 trial)** — distractor: 컵
```bash
python scripts/record_scenario.py \
    --scenario S5 \
    --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --append-manifest
```

---

### S6 — target 위치 이동

- **Gold**: `AMBIGUOUS_TARGET` → `ASK`
- **Checkpoint**: C1

```
t₀  컵 1개(위치 A) + 빨간박스 1개
    ↓  [C1 직전: 컵을 위치 A에서 위치 B로 이동]
C1  컵 1개(위치 B) + 빨간박스 1개
C2  컵 1개(위치 B) + 빨간박스 1개
```

> **이동 거리가 핵심**: 화면 기준으로 컵이 반대편으로 이동할 정도면 충분.  
> G₀ coord(위치 A)와 Gₜ coord(위치 B) 간 거리가 threshold(50px) × 5 = 250px 이상이어야 AMBIGUOUS로 분류됨.

**Cup (컵 5 trial)**
```bash
python scripts/record_scenario.py \
    --scenario S6 \
    --task "pick the cup and put it in the red box" \
    --target-label cup --destination-label "red box" \
    --append-manifest
```

**Cube (큐브 5 trial)**
```bash
python scripts/record_scenario.py \
    --scenario S6 \
    --task "pick the cube and put it in the red box" \
    --target-label cube --destination-label "red box" \
    --append-manifest
```

---

## 4. 큐브 변형 (동일 패턴 반복)

위 S1~S6를 큐브로 동일하게 진행. Task 문구만 교체:

| 시나리오 | Task (cube 버전) |
|---------|----------------|
| S1, S2, S3, S5, S6 | `"pick the cube and put it in the red box"` |
| S4 | `"pick the cube and put it in the box"` |

---

## 5. Trial 계획

**시나리오당 10 trial (컵 5 + 큐브 5) = 총 60개**

| 시나리오 | 컵 trial | 큐브 trial | 예상 시간 |
|---------|---------|----------|---------|
| S1 | 5 | 5 | 30분 |
| S2 | 5 | 5 | 50분 |
| S3 | 5 | 5 | 40분 |
| S4 | 5 | 5 | 50분 |
| S5 | 5 | 5 | 40분 |
| S6 | 5 | 5 | 40분 |
| **합계** | **30** | **30** | **~4.5시간** |

> 각 trial마다 오브젝트 위치를 다르게 세팅해야 의미 있는 visual variation이 생긴다.  
> 같은 위치에서 반복 촬영하면 10 trial이 1 trial과 다름없다.

---

## 6. 수집 후 평가 실행

```bash
# manifest 검증
python src/evaluate.py dataset/manifest.jsonl --validate-only --check-images

# 전체 평가 실행
python src/evaluate.py dataset/manifest.jsonl \
    --methods b1 b2 b3 b4 ours \
    --model-type finetune \
    --adapter-ckpt nFwD6qtf9T8dkJaQXU9vkW \
    --predictions-csv logs/predictions_real.csv \
    --metrics-json logs/metrics_real.json
```

---

## 7. 주의사항

- **S4 task 문구**: 반드시 "box" (색 없이). "red box" 쓰면 S4 gold label 무효.
- **S6 이동 거리**: 충분히 멀리 이동. 조금만 이동하면 CLEAR로 분류될 수 있음.
- **S5 distractor**: task에 등장하지 않는 오브젝트 사용 (컵 task라면 큐브, 큐브 task라면 컵).
- **위치 variation**: 매 trial마다 오브젝트 위치 변경 필수.
- **조명/배경**: 일정하게 유지하되, 약간의 변화는 허용 (real-world robustness).
