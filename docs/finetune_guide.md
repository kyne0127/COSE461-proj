# AmbRes 재학습 End-to-End 가이드

ZED 카메라로 촬영한 실제 이미지로 AmbRes(Molmo-7B LoRA)를 재학습하는 전체 절차.

---

## 전체 흐름

```
[Step 1] 이미지 촬영 (ZED)
    ↓
[Step 2] 원시 데이터 레이블링 (obj_list / tasks / ambiguity_map)
    ↓
[Step 3] 학습 샘플 자동 생성 (make_samples.py)
    ↓
[Step 4] 파인튜닝 (train.py)
    ↓
[Step 5] 체크포인트 배포 및 서버 재시작
```

---

## Step 0: 사전 준비

### 디렉터리 구조 확인

AmbRes는 두 가지 경로를 사용합니다:

| 용도 | 경로 |
|------|------|
| 학습 이미지 | `~/datasets/ambres/real/{train,test}/` |
| 레이블 데이터 | `/workspace/AmbRes/assets/data/real/{train,test}/` |
| 체크포인트 | `/workspace/AmbRes/ckpt/<run_id>/checkpoint-<N>/` |

```bash
# 이미지 저장 디렉터리 생성
mkdir -p ~/datasets/ambres/real/train
mkdir -p ~/datasets/ambres/real/test

# 레이블 파일 초기화 (없으면)
touch /workspace/AmbRes/assets/data/real/train/data_raw.jsonl
touch /workspace/AmbRes/assets/data/real/test/data_raw.jsonl
```

---

## Step 1: 이미지 촬영

### 촬영 기준
- **한 씬당 1장** (여러 물체가 놓인 테이블 전경)
- 권장 해상도: **640×480 (4:3)** — ZED HD720 크롭 또는 USB 카메라
- 물체 간격을 충분히 두어 개별 식별 가능하게 배치
- **최소 50장 이상** (train 40장 + test 10장)

### 촬영 내용 권장
| 씬 유형 | 예시 | 비율 |
|---------|------|------|
| 동일 카테고리 물체 여러 개 | 빨간 머그 + 파란 머그 | ~40% |
| 카테고리별 1개씩 | 머그 1개 + 병 1개 + 마커 1개 | ~40% |
| 물체 1개만 | 머그 1개 | ~20% |

### 촬영 방법

```bash
# ZED로 이미지 촬영 후 저장
cd /workspace/COSE461-proj

# --n-frames 1 : 프레임 1장만 캡처
# --save-dir   : 저장 경로
python scripts/test_camera.py \
  --mode zed \
  --resolution HD720 \
  --n-frames 1 \
  --save-dir ~/datasets/ambres/real/train

# 또는 USB 카메라
python scripts/test_camera.py \
  --mode usb --camera-index 4 \
  --n-frames 1 \
  --save-dir ~/datasets/ambres/real/train
```

### 이미지 파일명 규칙

AmbRes는 `{id}.jpeg` 형식을 기대합니다. shortuuid 형태로 rename합니다:

```bash
cd ~/datasets/ambres/real/train
python3 - << 'EOF'
import os, shortuuid
from pathlib import Path

img_dir = Path(".")
for f in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
    new_name = shortuuid.uuid() + ".jpeg"
    # PIL로 JPEG 변환
    from PIL import Image
    img = Image.open(f).convert("RGB")
    img.save(new_name, "JPEG", quality=95)
    os.remove(f)
    print(f"{f.name} → {new_name}")
EOF
```

---

## Step 2: 원시 데이터 레이블링

각 이미지에 대해 아래 3가지를 레이블링합니다.

### 레이블 포맷 (`data_raw.jsonl`)

```jsonl
{"id": "3ZhaQJdD3YcXCJYL2xjrZb", "obj_list": ["red mug", "blue mug", "sprite bottle", "marker"], "tasks": ["Put the {mug} next to the {bottle}.", "Move the {marker} to the left of the {bottle}."], "ambiguity_map": {"mug": ["red mug", "blue mug"]}}
```

| 필드 | 설명 | 예시 |
|------|------|------|
| `id` | 이미지 파일명 (확장자 제외) | `"3ZhaQJdD3YcXCJYL2xjrZb"` |
| `obj_list` | 씬에 있는 모든 물체 이름 | `["red mug", "blue mug", "bottle"]` |
| `tasks` | 이 씬으로 수행 가능한 태스크 (`{object}` 플레이스홀더 사용) | `["Put the {mug} next to the {bottle}."]` |
| `ambiguity_map` | 모호한 물체명 → 구체적 인스턴스 목록 | `{"mug": ["red mug", "blue mug"]}` |

### 레이블링 규칙

**`obj_list`**: 씬에 보이는 모든 물체를 구체적으로 기술
```json
["red mug", "blue mug", "sprite bottle", "yellow marker", "tray"]
```

**`tasks`**: 이 씬에서 의미 있는 pick-and-place 태스크
- `{object}` 형태로 플레이스홀더 표시
- 2개 이상의 물체가 관련된 태스크만 포함 (single-object는 자동 생성)
```json
["Put the {mug} next to the {bottle}.", "Move the {marker} to the {tray}."]
```

**`ambiguity_map`**: task에서 사용된 물체 중 씬에 여러 인스턴스가 있는 것
```json
{"mug": ["red mug", "blue mug"]}
```
- 씬에 머그가 1개뿐이면 `ambiguity_map`에 포함하지 않음
- 씬에 머그가 2개면 `"mug": ["red mug", "blue mug"]`로 등록

### 레이블링 실행 (CLI 방식)

```bash
python3 /workspace/COSE461-proj/scripts/label_ambres_data.py \
  --env real --mode train \
  --image-dir ~/datasets/ambres/real/train
```

> 아래 Step 2.5에서 이 스크립트를 새로 만듭니다.

### Streamlit UI 방식 (선택)

AmbRes 원본 제공 레이블링 앱:

```bash
pip install streamlit
cd /workspace/AmbRes
streamlit run scripts/label_data.py
# → http://localhost:8501 에서 웹 UI로 레이블링
```

---

## Step 2.5: CLI 레이블링 스크립트 생성

웹 UI 없이 터미널에서 직접 레이블링하는 스크립트:

```bash
cat > /workspace/COSE461-proj/scripts/label_ambres_data.py << 'SCRIPT'
#!/usr/bin/env python3
"""
CLI 기반 AmbRes 원시 데이터 레이블링 스크립트.

사용법:
  python scripts/label_ambres_data.py --env real --mode train
"""

import json, os, sys, argparse
from pathlib import Path

sys.path.insert(0, "/workspace/AmbRes")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def label_image(img_id: str, img_path: Path, out_file: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"이미지: {img_path.name}")
    print(f"  → 이미지를 확인하세요: {img_path}")
    print(f"{'='*60}")

    while True:
        raw = input("obj_list (쉼표 구분, 예: red mug, blue mug, bottle): ").strip()
        if raw:
            obj_list = [o.strip() for o in raw.split(",") if o.strip()]
            break

    while True:
        raw = input('tasks (JSON 배열, 예: ["Put the {mug} next to the {bottle}."]): ').strip()
        try:
            tasks = json.loads(raw)
            break
        except:
            print("  JSON 형식 오류. 다시 입력하세요.")

    while True:
        raw = input('ambiguity_map (JSON dict, 없으면 {}): ').strip()
        try:
            ambiguity_map = json.loads(raw)
            break
        except:
            print("  JSON 형식 오류. 다시 입력하세요.")

    sample = {
        "id": img_id,
        "obj_list": obj_list,
        "tasks": tasks,
        "ambiguity_map": ambiguity_map,
    }
    with open(out_file, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  ✓ 저장됨")
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",  choices=["real","sim"], default="real")
    parser.add_argument("--mode", choices=["train","test"], default="train")
    parser.add_argument("--image-dir", required=True)
    args = parser.parse_args()

    from ambres import ASSETS_DIR
    raw_file = ASSETS_DIR.get_dir(args.env, args.mode) / "data_raw.jsonl"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.touch()

    # 이미 레이블링된 id 로드
    done_ids = set()
    if raw_file.exists():
        with open(raw_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])

    img_dir = Path(args.image_dir)
    imgs = sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    remaining = [p for p in imgs if p.stem not in done_ids]

    print(f"총 {len(imgs)}장  |  남은 {len(remaining)}장  |  출력: {raw_file}")

    for i, img_path in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}]", end="")
        try:
            label_image(img_path.stem, img_path, raw_file)
        except (KeyboardInterrupt, EOFError):
            print("\n중단. 재실행하면 이어서 진행합니다.")
            break

if __name__ == "__main__":
    main()
SCRIPT
chmod +x /workspace/COSE461-proj/scripts/label_ambres_data.py
```

---

## Step 3: 학습 샘플 자동 생성

레이블링이 끝나면 `make_samples.py`로 학습용 JSON을 생성합니다.

```bash
cd /workspace/AmbRes
python scripts/make_samples.py --env real
```

이 스크립트는:
- `data_raw.jsonl`의 각 이미지에서 **이미지당 20개** 태스크 샘플을 자동 생성
- ambiguity_bool=True/False 균형 맞춤
- 결과: `assets/data/real/{train,test}/data.json`

생성 결과 확인:
```bash
python3 - << 'EOF'
import json
with open("/workspace/AmbRes/assets/data/real/train/data.json") as f:
    data = json.load(f)
true_cnt  = sum(1 for d in data if d["ambiguity_bool"])
false_cnt = len(data) - true_cnt
print(f"총 샘플: {len(data)}")
print(f"ambiguous=True : {true_cnt} ({true_cnt/len(data)*100:.0f}%)")
print(f"ambiguous=False: {false_cnt} ({false_cnt/len(data)*100:.0f}%)")
EOF
```

---

## Step 4: 파인튜닝

### 의존성 설치

```bash
pip install wandb
wandb login  # API 키 입력
```

### 학습 실행

```bash
cd /workspace/AmbRes
HF_HOME=/workspace/hf_cache python scripts/train.py --env real
```

학습 설정 (기본값, `ambres/training/params.py`):

| 파라미터 | 값 |
|---------|-----|
| epochs | 1 |
| batch_size | 2 |
| learning_rate | 1e-4 |
| LoRA rank | 4 |
| LoRA alpha | 8 |
| optimizer | AdamW |
| precision | bfloat16 |
| DeepSpeed | ZeRO-3 offload |

> 데이터가 적을 경우(50장 이하) epochs를 늘리거나 eval_steps를 줄이세요:

```python
# ambres/training/params.py 수정
TRAINING_ARGS = TrainingArguments(
    num_train_epochs=3,      # 1 → 3
    eval_steps=20,           # 40 → 20
    ...
)
```

### 학습 모니터링

```bash
# 학습 중 로그 확인 (wandb 대신 로컬)
tail -f /workspace/AmbRes/ckpt/<run_id>/trainer_state.json

# loss 추이 확인
python3 - << 'EOF'
import json, glob
ckpt_dirs = glob.glob("/workspace/AmbRes/ckpt/*/trainer_state.json")
for p in ckpt_dirs:
    with open(p) as f: s = json.load(f)
    logs = [l for l in s.get("log_history", []) if "loss" in l and "eval" not in l]
    if logs:
        print(f"\n{p}")
        print(f"  step 1  loss: {logs[0]['loss']:.4f}")
        print(f"  step {logs[-1]['step']} loss: {logs[-1]['loss']:.4f}")
EOF
```

---

## Step 5: 체크포인트 배포

### 새 체크포인트 ID 확인

```bash
ls /workspace/AmbRes/ckpt/
# 새로 생성된 run_id 확인 (예: AbCdEfGhIjKl...)
```

### `server.yaml` 업데이트

```bash
# /workspace/COSE461-proj/module/config/server.yaml
handlers:
  - handler_id: ambres
    config:
      model_type: finetune
      adapter_ckpt: <새_run_id>   # ← 여기 교체
      use_detection: false
      use_sam: false
```

또는 런타임에 지정:
```bash
# test_camera.py에서
client.load_handler("ambres", config={
    "model_type": "finetune",
    "adapter_ckpt": "<새_run_id>",
    ...
})
```

### 서버 재시작

```bash
# 기존 프로세스 종료
ps aux | grep run_server | grep -v grep
kill -9 <PID들>
nvidia-smi | grep MiB  # 2MiB 확인 후

# 재시작
cd /workspace/COSE461-proj
HF_HOME=/workspace/hf_cache python scripts/run_server.py > /tmp/server.log 2>&1 &
tail -f /tmp/server.log
```

---

## 데이터 품질 체크리스트

레이블링 전에 확인:

- [ ] 이미지당 물체가 3개 이상 포함
- [ ] `obj_list`에 이미지에 보이는 **모든** 물체 기재
- [ ] `tasks`의 플레이스홀더 `{object}`가 `obj_list` 또는 `ambiguity_map`의 key와 일치
- [ ] 동일 종류 물체가 2개 이상 있을 때만 `ambiguity_map`에 등록
- [ ] `ambiguity_map`의 value 배열이 `obj_list`에 있는 구체적 이름과 일치
- [ ] train:test = 8:2 비율 유지

### 레이블 예시 (올바른 케이스)

```json
{
  "id": "AbCdEfGhIjKl",
  "obj_list": ["red mug", "blue mug", "sprite bottle", "yellow marker"],
  "tasks": [
    "Put the {mug} next to the {bottle}.",
    "Move the {marker} to the left of the {mug}."
  ],
  "ambiguity_map": {
    "mug": ["red mug", "blue mug"]
  }
}
```

`"bottle"`, `"marker"`은 1개뿐이므로 `ambiguity_map`에 없음.
`"mug"`는 2개(red, blue)이므로 등록.

---

## 예상 소요 시간

| 단계 | 50장 기준 |
|------|---------|
| 이미지 촬영 | 30분 |
| 레이블링 (장당 2분) | 1~2시간 |
| make_samples.py | 1분 |
| 파인튜닝 (1 epoch) | 15~30분 |
| 검증 | 15분 |
| **합계** | **3~4시간** |

---

## 트러블슈팅

### 학습 중 CUDA OOM
```python
# params.py에서 gradient_accumulation 늘리기
gradient_accumulation_steps=4,  # 1 → 4
per_device_train_batch_size=1,  # 2 → 1
```

### `data_raw.jsonl` 형식 오류
```bash
python3 -c "
import json
with open('/workspace/AmbRes/assets/data/real/train/data_raw.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except Exception as e:
            print(f'Line {i} error: {e}')
            print(f'  content: {line[:100]}')
"
```

### 이미지 경로 오류
```bash
python3 -c "
import json
from pathlib import Path
img_dir = Path('~/datasets/ambres/real/train').expanduser()
with open('/workspace/AmbRes/assets/data/real/train/data.json') as f:
    data = json.load(f)
missing = [d['id'] for d in data if not (img_dir / (d['id'] + '.jpeg')).exists()]
print(f'누락 이미지: {len(missing)}개')
for m in missing[:5]: print(f'  {m}.jpeg')
"
```
