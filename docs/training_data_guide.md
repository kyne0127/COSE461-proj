# Training Data Guide — AmbRes/Molmo Finetuning

> AmbRes(Molmo-7B-D + LoRA) finetuning을 위한 학습 데이터 수집 절차를 정리한다.
> 평가 데이터(`data-evaluation/`)와 **별도로** 수집하여 data leakage를 방지한다.

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
python scripts/capture_training.py                     # 기본 (조합당 10장)
python scripts/capture_training.py --target-per-combo 15  # 조합당 15장
python scripts/capture_training.py --out-dir <경로>    # 저장 경로 지정
```

| 키 | 동작 |
|----|------|
| `SPACE` | 현재 씬 촬영 |
| `n` | 다음 조합으로 이동 |
| `p` | 이전 조합으로 이동 |
| `r` | 마지막 촬영 취소 |
| `q` | 저장 후 종료 |

---

## 3. 씬 조합 (14개)

### 3-1. Clear (4개)

| combo_id | 세팅 | task |
|----------|------|------|
| `cup1_redbox1` | 컵 1개 + 빨간박스 1개 | pick the cup and put it in the red box |
| `cup1_yellowbox1` | 컵 1개 + 노란박스 1개 | pick the cup and put it in the yellow box |
| `cube1_redbox1` | 큐브 1개 + 빨간박스 1개 | pick the cube and put it in the red box |
| `cube1_yellowbox1` | 큐브 1개 + 노란박스 1개 | pick the cube and put it in the yellow box |

### 3-2. Ambiguous Target (4개)

| combo_id | 세팅 | task |
|----------|------|------|
| `cup2_redbox1` | 컵 2개 + 빨간박스 1개 | pick the cup and put it in the red box |
| `cup2_yellowbox1` | 컵 2개 + 노란박스 1개 | pick the cup and put it in the yellow box |
| `cube2_redbox1` | 큐브 2개 + 빨간박스 1개 | pick the cube and put it in the red box |
| `cube2_yellowbox1` | 큐브 2개 + 노란박스 1개 | pick the cube and put it in the yellow box |

> 두 target은 화면상 충분히 떨어진 위치에 배치 (너무 붙으면 1개로 인식됨).

### 3-3. Ambiguous Destination (2개)

| combo_id | 세팅 | task |
|----------|------|------|
| `cup1_twobox` | 컵 1개 + 빨간박스 + 노란박스 | pick the cup and put it in the box |
| `cube1_twobox` | 큐브 1개 + 빨간박스 + 노란박스 | pick the cube and put it in the box |

> task에 색을 명시하면 안 됨. 반드시 "box"로만 표기.

### 3-4. Distractor (2개)

| combo_id | 세팅 | task |
|----------|------|------|
| `cup1_redbox1_cube_distractor` | 컵 1개 + 빨간박스 + 큐브(무관) | pick the cup and put it in the red box |
| `cube1_redbox1_cup_distractor` | 큐브 1개 + 빨간박스 + 컵(무관) | pick the cube and put it in the red box |

### 3-5. Invalid Target (2개)

| combo_id | 세팅 | task |
|----------|------|------|
| `noobj_redbox1` | 빨간박스만 (컵/큐브 없음) | pick the cup and put it in the red box |
| `noobj_yellowbox1` | 노란박스만 (컵/큐브 없음) | pick the cube and put it in the yellow box |

---

## 4. 출력 구조

```
data-training/
  <combo_id>/
    img_001.png
    img_002.png
    ...
    meta.json
```

`meta.json` 포맷:
```json
{
  "combo_id": "cup1_redbox1",
  "desc": "컵 1개 + 빨간박스 1개",
  "task": "pick the cup and put it in the red box",
  "target_label": "cup",
  "destination_label": "red box",
  "gold_state": "CLEAR",
  "ambiguous": false,
  "image_count": 10
}
```

---

## 5. 수집 현황

| combo_id | gold_state | 수집 완료 |
|----------|-----------|---------|
| cup1_redbox1 | CLEAR | 11장 |
| cup1_yellowbox1 | CLEAR | 10장 |
| cup1_twobox | AMBIGUOUS_DESTINATION | 10장 |
| cup2_redbox1 | AMBIGUOUS_TARGET | 10장 |
| cup2_yellowbox1 | AMBIGUOUS_TARGET | 10장 |
| cube1_redbox1 | CLEAR | 10장 |
| cube1_yellowbox1 | CLEAR | 10장 |
| cube1_twobox | AMBIGUOUS_DESTINATION | 10장 |
| cube2_redbox1 | AMBIGUOUS_TARGET | 10장 |
| cube2_yellowbox1 | AMBIGUOUS_TARGET | 10장 |
| cup1_redbox1_cube_distractor | CLEAR | 미수집 |
| cube1_redbox1_cup_distractor | CLEAR | 미수집 |
| noobj_redbox1 | INVALID_TARGET | 미수집 |
| noobj_yellowbox1 | INVALID_TARGET | 미수집 |
| **합계** | | **101장** |

---

## 6. 수집 후 annotation

```bash
# 특정 조합 annotation
python scripts/annotate_coords.py \
    --dir data-training/cup1_redbox1 \
    --target-label cup --destination-label "red box"

# 전체 일괄 annotation (manifest 기반)
python scripts/annotate_coords.py --from-manifest dataset/manifest_train.jsonl
```

annotation 완료 후 → `/workspace/AmbRes/assets/data/real/train/` 에 업로드하여 LoRA 학습.

---

## 7. HuggingFace

- **Repository**: `kyne0127/vla-evaluation`
- **Path**: `data-training/`

```bash
# 추가 수집 후 재업로드
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path='data-training',
    repo_id='kyne0127/vla-evaluation',
    repo_type='dataset',
    path_in_repo='data-training',
)
"
```
