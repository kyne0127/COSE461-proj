# 카메라 설정 및 글로벌뷰 테스트 가이드

작성일: 2026-05-19

---

## 1. 카메라 구성 개요

본 프로젝트에서 카메라는 역할에 따라 두 종류로 구분된다.

| 종류 | 위치 | 목적 | 구현 |
|------|------|------|------|
| **Global view** | 작업 공간 위 고정 | AmbRes G₀ 추출, 장면 관측 | `ZEDCapture` (pyzed) 또는 USB OpenCV |
| **Egoview** | SO-ARM 엔드이펙터 | SmolVLA 정책 입력, 실시간 행동 제어 | LeRobot `OpenCVCameraConfig` |

```
[작업 공간]
                 ┌─────────────────┐
  ZED Mini ──── │  Global View    │  ← AmbRes G₀ 추출용
  (or USB)      │  640×360 / RGB+D│
                └─────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │         SO-ARM 101            │
        │  ┌──────────────────────┐     │
        │  │ Egoview Right (idx 0)│ ←──┘ SmolVLA 입력
        │  │ Egoview Left  (idx 2)│
        │  └──────────────────────┘
        └──────────────────────────────┘
```

---

## 2. 관련 파일

| 파일 | 역할 |
|------|------|
| `module/desktop/zed_connector.py` | ZED Mini / USB 캡처 클래스 (`ZEDCapture`, `USBCapture`, `SyncCapture`) |
| `module/desktop/robot_connector.py` | Egoview 카메라 포함 로봇 관측 통합 |
| `module/config/desktop.yaml` | 카메라 인덱스, 해상도, FPS 설정 |
| `scripts/test_camera.py` | 카메라 단독 테스트 (로봇 팔 불필요) |

---

## 3. 의존성 설치

### 3.1 공통 (OpenCV — USB 카메라 fallback)

```bash
pip install opencv-python
```

### 3.2 ZED Mini (pyzed SDK)

ZED SDK는 pip로 설치하지 않으며, [공식 ZED SDK](https://www.stereolabs.com/developers/release)를 먼저 설치한 뒤 pyzed를 설치한다.

```bash
# 1. ZED SDK 설치 (installer 실행, CUDA 버전에 맞게 선택)
#    https://www.stereolabs.com/developers/release → ZED SDK for Ubuntu

# 2. pyzed Python API 설치
python /usr/local/zed/get_python_api.py

# 3. 설치 확인
python -c "import pyzed.sl as sl; print('pyzed OK', sl.Camera().get_sdk_version())"
```

> **pyzed 없어도 동작함**: `ZEDCapture`는 pyzed를 찾지 못하면 자동으로 OpenCV fallback으로 전환한다. depth는 1.5m 고정값으로 시뮬레이션된다.

---

## 4. desktop.yaml 카메라 설정

```yaml
# module/config/desktop.yaml

robot:
  robot_config:
    cameras:
      egoview_right:
        camera_index: 0    # USB 카메라 인덱스
        fps: 60
        width: 640
        height: 480
      egoview_left:
        camera_index: 2
        fps: 60
        width: 640
        height: 480

sensors:
  zed_mini:
    enabled: true
    resolution: "VGA"        # VGA=640×360 | HD720=1280×720
    fps: 30
    depth_mode: "PERFORMANCE"
    depth_min_dist: 0.3      # metres
    depth_max_dist: 3.0      # metres
  usb_camera:
    enabled: true
    device_index: 0          # global view USB 카메라 인덱스
    fps: 30
    width: 640
    height: 480
```

---

## 5. 카메라 테스트 스크립트 (`scripts/test_camera.py`)

로봇 팔(USB 시리얼) 연결 없이 카메라만 단독으로 테스트한다.

### 5.1 연결된 카메라 전체 스캔

```bash
python scripts/test_camera.py --mode scan
```

출력 예시:
```
=== 카메라 스캔 (인덱스 0~7) ===

  [FOUND] index=0  640x480  30fps
  [  ---  ] index=1  연결 없음
  [FOUND] index=2  640x480  60fps
  [FOUND] index=4  640x360  30fps     ← global view USB

발견된 카메라: [0, 2, 4]
  → global view 테스트: python scripts/test_camera.py --mode usb --camera-index 4
```

### 5.2 USB 카메라 (Global View) 단독 테스트

```bash
# 기본 (3 프레임, 통계 출력)
python scripts/test_camera.py --mode usb --camera-index 4

# 5 프레임 캡처 후 PNG 저장
python scripts/test_camera.py --mode usb --camera-index 4 \
  --n-frames 5 --save-dir /tmp/cam_test

# 해상도 지정
python scripts/test_camera.py --mode usb --camera-index 4 \
  --width 1280 --height 720 --fps 30
```

출력 예시:
```
=== USB 카메라 테스트 (index=4) ===

  실제 해상도: 640x360  FPS: 30
  워밍업 5 프레임 스킵...
  3 프레임 캡처:
  [1/3  12.4ms] shape=(360,640,3)  dtype=uint8  min=0.000  max=255.000  mean=118.432
  [2/3  11.8ms] shape=(360,640,3)  dtype=uint8  min=0.000  max=255.000  mean=119.105
  [3/3  12.1ms] shape=(360,640,3)  dtype=uint8  min=0.000  max=255.000  mean=118.891
    → 저장: /tmp/cam_test/usb_cam4_20260519_130000_001.png
```

### 5.3 ZED Mini 테스트

```bash
# VGA 해상도, RGB+Depth 캡처
python scripts/test_camera.py --mode zed

# HD720, 저장 포함
python scripts/test_camera.py --mode zed \
  --resolution HD720 --n-frames 5 --save-dir /tmp/cam_test
```

Depth는 `zed_depth_YYYYMMDD_HHMMSS_NNN.png` (16-bit PNG, mm 단위)로 저장된다.

### 5.4 AmbRes G₀ 추출까지 End-to-End 테스트

카메라로 캡처한 이미지로 실제 AmbRes 파이프라인을 테스트한다. gRPC 서버가 실행 중이어야 한다.

```bash
# 1. 터널 열기 (RunPod)
python scripts/open_tunnel.py

# 2. 카메라 캡처 → AmbRes G₀ 추출
python scripts/test_camera.py --mode usb --camera-index 4 \
  --task "place the red mug next to the sprite bottle" \
  --run-ambres
```

출력 예시:
```
=== AmbRes G₀ 추출 테스트 ===
  task: 'place the red mug next to the sprite bottle'
  임시 이미지: /tmp/tmpXXXX.png

  G₀ 추출 결과:
  {
      "target": { "label": "red mug", "coord": [941, 723] },
      "destination": { "label": "sprite bottle", "coord": [466, 784] },
      "image_shape": [360, 640]
  }
```

---

## 6. 코드에서 직접 사용

### 6.1 ZEDCapture (글로벌뷰 RGB+Depth)

```python
from module.desktop.zed_connector import ZEDCapture

with ZEDCapture(resolution="VGA", fps=30, depth_mode="PERFORMANCE") as zed:
    rgb, depth = zed.capture_synchronized()
    # rgb:   (360, 640, 3) uint8
    # depth: (360, 640, 1) float32 [metres]
    print(f"RGB: {rgb.shape}, Depth range: {depth.min():.2f}–{depth.max():.2f} m")
```

### 6.2 USBCapture (글로벌뷰 RGB)

```python
from module.desktop.zed_connector import USBCapture

with USBCapture(device_index=4, width=640, height=360, fps=30) as cam:
    rgb = cam.capture()
    # rgb: (360, 640, 3) uint8
```

### 6.3 SyncCapture (ZED + USB 동기화)

```python
from module.desktop.zed_connector import SyncCapture

with SyncCapture(
    zed_resolution="VGA",
    usb_device_index=0,
) as sync:
    frames = sync.capture_normalized()
    # frames["zed_rgb"]:   (360, 640, 3) float32 [0,1]
    # frames["zed_depth"]: (360, 640, 1) float32 [metres]
    # frames["usb_rgb"]:   (480, 640, 3) float32 [0,1]
```

### 6.4 AmbRes 파이프라인에 글로벌뷰 이미지 주입

```python
import numpy as np
from PIL import Image
from module.desktop.zed_connector import USBCapture
from extraction.ambres_g0_extractor import extract_g0
from module.models.ambres.handler import AmbResHandler

# 1. 카메라 캡처
with USBCapture(device_index=4) as cam:
    rgb = cam.capture()   # (H, W, 3) uint8

# 2. PIL 저장 → 임시 파일 경로로 extract_g0 호출
import tempfile, cv2, os
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
    tmp = f.name
cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# 3. AmbRes handler + G₀ 추출
handler = AmbResHandler()
handler.setup({"model_type": "finetune", "adapter_ckpt": "43qazb3XcrZF5rZWnjRPVm"})
g0 = extract_g0(tmp, "place the red mug next to the bottle",
                handler=handler, allow_ambiguous=True)
os.unlink(tmp)
print(g0)
```

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `Failed to open camera` | 잘못된 인덱스 또는 장치 미연결 | `--mode scan` 으로 인덱스 확인 |
| 프레임이 검거나 통계 이상 | 카메라 노출 미안정 | `--warmup` 값 증가 (기본 5) |
| ZED 초기화 실패 → OpenCV fallback | pyzed 미설치 또는 ZED SDK 버전 불일치 | `python /usr/local/zed/get_python_api.py` 재실행 |
| AmbRes `gRPC connect failed` | 서버 미기동 또는 터널 끊김 | `scripts/open_tunnel.py` 재실행 후 `check_connection.py` 확인 |
| depth 값이 모두 1.5 | pyzed 없어서 시뮬레이션 depth 사용 중 | pyzed 설치 또는 RGB-only 파이프라인 사용 |
| `mean ≈ 0` (검은 화면) | 카메라가 렌즈 캡 씌워져 있거나 조도 부족 | 조명·캡 확인 |

---

## 8. 다음 단계 — 파인튜닝 데이터 수집 절차

글로벌뷰 카메라 테스트가 완료되면 아래 순서로 파인튜닝 데이터를 수집한다.

```
1. test_camera.py --mode scan       → 카메라 인덱스 확인
2. test_camera.py --mode usb        → 글로벌뷰 영상 품질 확인 + 저장
3. test_camera.py --run-ambres      → AmbRes G₀ 추출 품질 확인
4. run_desktop.py collect           → 리더-팔로워 텔레오퍼레이션 데모 수집
5. run_desktop.py train             → 서버에 학습 트리거
6. run_desktop.py pipeline          → AmbRes + SmolVLA 자율 실행
```

자세한 수집 절차는 [setup_and_workflow.md](setup_and_workflow.md)를 참고한다.
