# Spec: Fixed Checkpoint → Trigger-based Ambiguity Monitor 전환
`Version 0.1 | 개인용 Draft`

---

## 0. 전환 배경

### 현재 구조의 학술적 한계

```
Fixed Checkpoint 구조:
  step 100 (C1) 도달 → AmbResVLM 호출
  step 300 (C2) 도달 → AmbResVLM 호출

문제:
  - 변화가 없어도 호출 (불필요한 중단)
  - 변화가 있어도 checkpoint 전이면 감지 불가
  - 인간의 주의 배분 방식과 불일치
```

### 목표 구조

```
Metric Monitor (항상 실행, 경량)
  → scene change score 계산 @ 매 프레임
  → score > θ 시 AmbResVLM trigger

AmbResVLM (on-demand, 무거움)
  → 변화의 의미 해석
  → grounding state 분류
  → decision 출력
```

### 이론적 근거

- **Event Segmentation Theory** (Zacks et al.): 인간은 prediction error spike 시점에만 event model을 업데이트
- **KnowNo** (CoRL 2023): conformal prediction set non-singleton 시에만 도움 요청 → help rate 10~24% 감소
- **LeMasurier et al.** (HRI 2024): event-triggered 방식이 checkpoint-triggered 대비 신뢰도(4.47 vs 4.08), 선호도(127 vs 49)에서 유의미하게 우수

---

## 1. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│  Desktop (RTX 3060)                                         │
│                                                             │
│  SmolVLA (30Hz) ─────────────────────────────────────────  │
│                                                             │
│  DINOv2 Monitor (5~10Hz) ──→ score > θ? ──→ [TRIGGER]     │
│       ↑                                          ↓          │
│  global_cam                            SmolVLA pause        │
│  (intent-relevant region crop)                   ↓          │
│                              ┌─── gRPC ───→ AmbResVLM      │
│                              │       (GPU 서버, on-demand)  │
│                              └─── decision ←───────────    │
│                                      ↓                      │
│                              CONTINUE / ASK / STOP          │
└─────────────────────────────────────────────────────────────┘
```

### 역할 분리

| 컴포넌트 | 역할 | 주기 | 위치 |
|---|---|---|---|
| **DINOv2 Monitor** | "뭔가 바뀌었는가?" 감지 | 5~10Hz | Desktop GPU (SmolVLA와 공유) |
| **AmbResVLM** | "무엇이 문제인가?" 해석 + decision | on-demand (trigger 시) | GPU 서버 (gRPC) |
| **SmolVLA** | action 생성 | 30Hz | Desktop GPU |

---

## 2. DINOv2 Monitor 설계

### 2.1 핵심 아이디어

```
G₀ 생성 시점에서 intent-relevant region의 DINOv2 feature를 저장
→ 매 프레임, 동일 region의 feature와 cosine distance 계산
→ score > θ 이면 AmbResVLM trigger
```

### 2.2 Intent-conditioned Region 추출

```python
# G₀ 생성 시 (t₀)
G₀ = {
    "target":      {"label": "red block", "coord": [142, 318]},
    "destination": {"label": "tray",      "coord": [401, 289]},
    "image_shape": [480, 640]
}

# coord → bbox 변환
# Molmo pointing coord를 중심으로 고정 크기 bbox 생성
BBOX_SIZE = 80  # px, pilot 실험으로 결정

def coord_to_bbox(coord, img_shape, size=BBOX_SIZE):
    x, y = coord
    h, w = img_shape[:2]
    x1 = max(0, x - size // 2)
    y1 = max(0, y - size // 2)
    x2 = min(w, x + size // 2)
    y2 = min(h, y + size // 2)
    return [x1, y1, x2, y2]

# G₀에서 bbox 계산 및 feature 저장
bbox_target = coord_to_bbox(G₀["target"]["coord"], img_shape)
bbox_dest   = coord_to_bbox(G₀["destination"]["coord"], img_shape)

f0_target = dinov2.extract(crop(frame_0, bbox_target))  # ∈ R^d
f0_dest   = dinov2.extract(crop(frame_0, bbox_dest))    # ∈ R^d
```

### 2.3 Score 계산

```python
# 매 프레임 (5~10Hz)
def compute_score(frame_t, checkpoint_type):
    """
    checkpoint_type: "C1" (target 감시) | "C2" (destination 감시)
    """
    if checkpoint_type == "C1":
        ft = dinov2.extract(crop(frame_t, bbox_target))
        score = 1 - cosine_sim(f0_target, ft)

    elif checkpoint_type == "C2":
        ft = dinov2.extract(crop(frame_t, bbox_dest))
        score = 1 - cosine_sim(f0_dest, ft)

    return score  # ∈ [0, 1]
```

### 2.4 Trigger 조건

```python
THRESHOLD_θ = 0.15  # pilot 실험으로 결정 (아래 §5 참고)

if score > THRESHOLD_θ:
    trigger_ambres()
```

### 2.5 DINOv2 모델 선택

| 모델 | 임베딩 차원 | GPU 메모리 | 추론 속도 | 권장 |
|---|---|---|---|---|
| ViT-S/14 | 384 | ~85MB | < 5ms | PoC 검증 |
| ViT-B/14 | 768 | ~330MB | ~10ms | 최종 평가 |

RTX 3060 (8GB VRAM)에서 SmolVLA와 공존 가능 여부 확인 필요.  
→ SmolVLA bfloat16 ~6GB + DINOv2 ViT-S ~85MB = 가능 추정.

---

## 3. 로봇 자체 움직임 노이즈 제거

### 문제

```
로봇 팔이 이동하면 → global_cam 기준으로도 장면이 변해 보임
→ score 상승 → false positive trigger
```

### 대응: Intent-conditioned Region만 비교

```
target bbox / destination bbox는 로봇 팔과 다른 위치
→ 팔이 bbox 밖에 있는 한 score 영향 없음

단, 팔이 target 근처로 접근하는 C1 phase에서는
팔 자체가 bbox 안에 들어올 수 있음
→ C1에서는 destination bbox만 감시 (또는 감시 일시 중단)
→ C2에서는 destination bbox 감시
```

### Phase별 감시 대상

```
t₀ → C1 구간: destination bbox만 감시
               (target 근처에 팔이 있어 target bbox는 노이즈)

C1 → C2 구간: target bbox 감시 (pick 완료 후 변화 감지)
               + destination bbox 감시 (occupied 여부)

C2 이후:      destination bbox 감시 (place 완료 확인)
```

---

## 4. 기존 코드 변경 범위

### 4.1 변경 대상 파일

```
module/desktop/pipeline.py          ← 주요 변경
module/models/dinov2/monitor.py     ← 신규 추가
module/config/pipelines/ambres_smolvla.yaml  ← monitor 설정 추가
```

### 4.2 `pipeline.py` 변경 내용

**현재 구조 (Fixed Checkpoint):**
```python
# action loop
for step in range(max_steps):
    action = smolvla.predict_action(obs)
    robot.send_action(action)

    # Fixed checkpoint
    if step == c1_step:
        _run_ambres_check("C1", obs)
    if step == c2_step:
        _run_ambres_check("C2", obs)
```

**변경 후 (Trigger-based):**
```python
# 초기화 (episode_start)
monitor = DINOv2Monitor(
    frame_0=obs["global_cam"],
    g0=g0_initial,
    model=dinov2_model,
    threshold=θ,
    sample_rate=6  # 30Hz 중 6Hz만 샘플링
)

# action loop
for step in range(max_steps):
    action = smolvla.predict_action(obs)
    robot.send_action(action)

    # Trigger-based monitor (매 N step)
    if step % monitor.sample_interval == 0:
        phase = _get_phase(step, max_steps)
        score = monitor.compute_score(obs["global_cam"], phase)

        if score > monitor.threshold:
            robot.hold_pose()           # SmolVLA 일시 정지
            decision = _run_ambres_check(phase, obs)
            if decision == "STOP":
                break
            elif decision == "ASK":
                _ask_user_and_update_g0(...)
            robot.resume()
```

### 4.3 신규: `module/models/dinov2/monitor.py`

```python
class DINOv2Monitor:
    def __init__(self, frame_0, g0, model, threshold, sample_rate):
        self.model = model          # DINOv2 ViT-S/14
        self.threshold = threshold
        self.sample_interval = 30 // sample_rate  # 30Hz 기준

        # G₀ feature 저장
        self.bbox_target = coord_to_bbox(g0["target"]["coord"], frame_0.shape)
        self.bbox_dest   = coord_to_bbox(g0["destination"]["coord"], frame_0.shape)
        self.f0_target   = self._extract(frame_0, self.bbox_target)
        self.f0_dest     = self._extract(frame_0, self.bbox_dest)

    def compute_score(self, frame_t, phase) -> float:
        if phase == "pre_pick":
            # 팔이 dest 근처 아직 없음 → dest 감시
            ft = self._extract(frame_t, self.bbox_dest)
            return 1 - cosine_sim(self.f0_dest, ft)
        elif phase == "pre_place":
            ft = self._extract(frame_t, self.bbox_dest)
            return 1 - cosine_sim(self.f0_dest, ft)

    def _extract(self, frame, bbox) -> np.ndarray:
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return self.model.encode(crop)  # → R^d, L2 normalized
```

### 4.4 `ambres_smolvla.yaml` 추가 설정

```yaml
checkpoint_monitor:
  enabled: true
  mode: trigger        # "fixed" | "trigger" (기존은 fixed)
  threshold: 0.15      # pilot_threshold.py로 결정
  sample_rate_hz: 6    # 30Hz 중 몇 Hz로 샘플링
  bbox_size_px: 80     # coord → bbox 변환 크기
  dinov2_model: vit_s  # "vit_s" | "vit_b"
  hold_pose_on_trigger: true
  max_triggers_per_episode: 3  # 무한 trigger 방지
```

---

## 5. Threshold θ 결정 방법

기존 `pilot_threshold.py` 활용:

```bash
# Step 1: CLEAR 케이스 (S1, S5)에서 DINOv2 score 측정
#         → θ의 상한 결정
python pilot_threshold.py \
  --images dataset/images/s1_t0.png dataset/images/s1_c1.png \
           dataset/images/s5_t0.png dataset/images/s5_c1.png \
  --mode dinov2 --dinov2-model vit_s \
  --output threshold_clear.json

# Step 2: AMBIGUOUS/INVALID 케이스 (S2, S3, S6)에서 score 측정
#         → θ의 하한 결정
python pilot_threshold.py \
  --images dataset/images/s2_t0.png dataset/images/s2_c1.png \
           dataset/images/s3_t0.png dataset/images/s3_c1.png \
  --mode dinov2 --output threshold_ambiguous.json

# Step 3: 두 분포가 분리되는 지점 → θ 설정
# 목표: clear_max < θ < ambiguous_min
```

### 기존 6개 시나리오 활용

| 시나리오 | Gold | 기대 DINOv2 score | θ 기준 |
|---|---|---|---|
| S1 (CLEAR) | CONTINUE | 낮음 | 상한 |
| S5 (CLEAR + distractor) | CONTINUE | 낮음 | 상한 |
| S2 (AMBIGUOUS_TARGET) | ASK | 높음 | 하한 |
| S3 (INVALID_TARGET) | STOP | 매우 높음 | 하한 |
| S6 (AMBIGUOUS_TARGET) | ASK | 높음 | 하한 |
| S4 (AMBIGUOUS_DEST) | ASK | 중간~높음 | 참고 |

---

## 6. Fixed vs Trigger-based Ablation

전환 후 두 방식을 직접 비교하는 ablation이 추가 contribution이 됩니다.

### 비교 지표

| 지표 | 측정 방법 |
|---|---|
| Decision Accuracy | gold label 대비 정확도 |
| False Positive Rate | CLEAR 상황에서 trigger 발생률 |
| False Negative Rate | 변화 있는 상황에서 미감지율 |
| AmbResVLM 호출 횟수 | episode당 평균 gRPC 호출 수 |
| Response Latency | 변화 발생 → decision 출력까지 시간 |
| Trigger Latency | 변화 발생 → trigger 시점까지 프레임 수 |

### 예상 결과

```
Fixed Checkpoint:
  AmbResVLM 호출: 항상 2회/episode (C1, C2)
  Trigger Latency: 변화 발생 후 최대 (c1_step or c2_step)까지 지연

Trigger-based:
  AmbResVLM 호출: 변화 없으면 0~1회, 있으면 즉시
  Trigger Latency: 변화 발생 후 1/sample_rate 이내
  → KnowNo 결과와 유사하게 호출 횟수 감소 예상
```

---

## 7. 구현 순서

### Step 1 — DINOv2 Monitor 추가 (Fixed Checkpoint 유지, ~2일)

```
[ ] module/models/dinov2/monitor.py 구현
[ ] RTX 3060에서 SmolVLA + DINOv2 동시 실행 VRAM 확인
[ ] 기존 6개 시나리오 이미지로 score 분포 측정
[ ] θ 결정
[ ] unit test 작성
```

### Step 2 — Trigger 로직 통합 (Fixed 병행 유지, ~2일)

```
[ ] pipeline.py에 trigger-based 분기 추가
    (mode: "fixed" | "trigger" 설정으로 전환)
[ ] ambres_smolvla.yaml에 monitor 섹션 추가
[ ] 실로봇에서 false positive 측정 (CLEAR 상황 10 trial)
[ ] θ 재조정
```

### Step 3 — Fixed Checkpoint 제거 + 비교 실험 (~3일)

```
[ ] mode: "trigger"로 전환
[ ] 6개 시나리오 전체 재실험 (Trigger-based Decision Accuracy)
[ ] Fixed vs Trigger ablation table 완성
[ ] 논문 Method 섹션 업데이트
```

---

## 8. Known Issues

| 이슈 | 상황 | 대응 |
|---|---|---|
| SmolVLA + DINOv2 VRAM 충돌 | RTX 3060 8GB, SmolVLA ~6GB | ViT-S 먼저 시도. 부족 시 DINOv2 CPU로 offload (latency 증가) |
| Bbox 추출 정확도 | Molmo coord가 object center가 아닐 수 있음 | BBOX_SIZE를 넉넉하게 (100~120px) 설정 후 조정 |
| 카메라 이동 노이즈 | S5에서 이미 경험 (541px shift) | 기존 camera-motion 보정 로직 (relative distance) DINOv2 Monitor에도 적용 |
| max_triggers_per_episode | 오탐이 연속으로 발생 시 무한 pause 가능 | 3회 초과 시 Fixed Checkpoint fallback |
| hold_pose 안전성 | pause 중 로봇이 불안정 위치에 있을 수 있음 | hold_pose = 현재 joint angle 유지 (SafetyGuard 연동) |

---

## 9. 기존 코드와의 관계

```
변경 없음:
  src/monitoring/consistency_monitor.py  (check_grounding 로직)
  src/extraction/ambres_g0_extractor.py  (G₀ 추출)
  src/baselines/                         (B1~B5 baseline)
  src/evaluate.py                        (평가 스크립트)
  dataset/manifest.jsonl                 (6개 시나리오)

추가:
  module/models/dinov2/monitor.py        (DINOv2 Monitor)
  module/models/dinov2/__init__.py

변경:
  module/desktop/pipeline.py             (trigger 분기 추가)
  module/config/pipelines/ambres_smolvla.yaml  (monitor 설정)
```

Fixed Checkpoint 로직은 `mode: "fixed"` 설정으로 그대로 유지.  
기존 6/6 실험 결과는 Fixed mode의 baseline으로 논문에 사용.

---

*Step 1 완료 후 score 분포 확인 결과에 따라 이 문서 v0.2 업데이트 예정.*
