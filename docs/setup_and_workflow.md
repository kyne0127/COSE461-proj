# 환경 설정 및 전체 동작 과정

작성일: 2026-04-13

---

## 1. 환경 구성 개요

| 항목 | 내용 |
|---|---|
| conda 환경명 | `vla-nlp` |
| Python 버전 | 3.12 |
| lerobot | 0.5.2 (GitHub 최신, feetech 포함) |
| PyTorch | 2.10.0+cu128 |
| gRPC | grpcio 1.80.0 / grpcio-tools 1.80.0 |

---

## 2. 환경 설정 절차

### 2.1 conda 환경 생성

```bash
conda create -n vla-nlp python=3.12 -y
conda activate vla-nlp
```

> Python 3.10은 lerobot 0.5.x 요구사항(`>=3.12`)을 충족하지 못하므로 3.12로 생성.

### 2.2 lerobot 설치 (feetech 모터 드라이버 포함)

```bash
PYTHONNOUSERSITE=1 /home/hands/miniforge3/envs/vla-nlp/bin/python -m pip install \
    "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git"
```

- `feetech` 옵션: SO-ARM100/101에 사용되는 Feetech STS3215 서보 드라이버 포함
- torch, torchvision, huggingface-hub 등 모델 의존성 자동 설치

### 2.3 gRPC 설치

```bash
PYTHONNOUSERSITE=1 /home/hands/miniforge3/envs/vla-nlp/bin/python -m pip install \
    grpcio grpcio-tools
```

### 2.4 프로젝트 모듈 설치 (editable)

```bash
cd /path/to/Project
PYTHONNOUSERSITE=1 /home/hands/miniforge3/envs/vla-nlp/bin/python -m pip install -e module/
```

> `module/pyproject.toml`의 `build-backend`를 `setuptools.build_meta`로 수정 필요
> (conda-forge setuptools는 `setuptools.backends` 서브패키지 미포함)

### 2.5 gRPC proto 스텁 생성

```bash
PYTHONNOUSERSITE=1 /home/hands/miniforge3/envs/vla-nlp/bin/python -m grpc_tools.protoc \
    -I module/proto \
    --python_out=module/proto \
    --grpc_python_out=module/proto \
    module/proto/lerobot.proto
```

생성 파일:
- `module/proto/lerobot_pb2.py`
- `module/proto/lerobot_pb2_grpc.py`

---

## 3. lerobot 0.5.x API 변경사항

lerobot 0.5.x에서 로봇 제어 API가 전면 재편됨.

| 구분 | 구버전 (0.4.x 이하) | 신버전 (0.5.x) |
|---|---|---|
| 로봇 클래스 | `lerobot.common.robot_devices.robots.so100.So100Robot` | `lerobot.robots.so_follower.SOFollower` |
| 리더 클래스 | `So100Robot` (내부 통합) | `lerobot.teleoperators.so_leader.SOLeader` (분리) |
| 관측 메서드 | `robot.capture_observation()` | `robot.get_observation()` |
| 텔레오퍼레이션 | `robot.teleop_step(record_data=True)` | `leader.get_action()` + `follower.send_action()` |
| 설정 방식 | dict kwargs | 데이터클래스 (`SOFollowerRobotConfig`, `SOLeaderTeleopConfig`) |
| 카메라 설정 | dict | `OpenCVCameraConfig(index_or_path, fps, width, height)` |

### robot_connector.py 대응

`module/desktop/robot_connector.py`를 lerobot 0.5.x 기준으로 업데이트:
- `SUPPORTED_ROBOTS`: `so100`, `so101` → 신 클래스 경로 매핑
- `RobotConnector`: 내부에 `_follower(SOFollower)` + `_leader(SOLeader)` 분리 보유
- `get_observation()`: `follower.get_observation()` 호출, dict → `RawObservation` 변환
- `teleop_step()`: `leader.get_action()` → `follower.send_action()` → `follower.get_observation()`
- `send_action(action_array)`: numpy 배열 → motor key dict 변환 후 전달

### desktop.yaml 설정

```yaml
robot:
  robot_type: "so101"   # so100 | so101
  fps: 30.0
  robot_config:
    follower_arms:
      main:
        port: "/dev/ttyUSB1"   # 팔로워 USB 포트
    leader_arms:
      main:
        port: "/dev/ttyUSB0"   # 리더 USB 포트
    cameras:
      top:
        camera_index: 0
        fps: 30
        width: 640
        height: 480
```

---

## 4. 시스템 구성

```
[데스크탑 (로컬)]                          [GPU 서버 (RunPod)]
  SO-ARM101 리더 ──USB──┐
                        ├── RobotConnector
  SO-ARM101 팔로워 ─USB─┘        │
  카메라 (top/wrist) ─USB─┘       │
                               gRPC (SSH 터널 / TCP)
                                   │
                          LeRobotGRPCServer :50051
                          ├── InferenceServicer  (모델 추론)
                          ├── TrainingServicer   (학습 잡 관리)
                          ├── GenericServicer    (범용 핸들러)
                          └── HealthServicer     (ping)
```

---

## 5. 전체 동작 과정

### 5.1 데이터 수집 모드 (`collect`)

서버 개입 없이 로컬에서 데모 데이터를 수집하고, 완료 후 서버에 업로드한다.

```
리더 암 조작 (사람)
     │
     ▼
leader.get_action()         ← 리더 관절 위치 읽기 (USB 시리얼)
     │
     ▼
follower.send_action()      ← 팔로워 암 이동 명령
     │
     ▼
follower.get_observation()  ← 팔로워 관절값 + 카메라 이미지 수집
     │
     ▼
EpisodeBuffer (로컬 메모리)  ← Frame(images, state, action) 적재
     │  (에피소드 반복)
     ▼
TrainingClient.upload_all_episodes()  ── gRPC ──► 서버 저장
                                                  /data/lerobot/datasets/
```

실행 명령:
```bash
python scripts/run_desktop.py collect \
    --config module/config/desktop.yaml \
    --n-episodes 10 \
    --task "pick up the red block"
```

---

### 5.2 학습 트리거 모드 (`train`)

서버에 학습 잡을 요청한다. 데이터는 이미 서버에 업로드된 상태여야 한다.

```
데스크탑
TrainingClient.start_training(model_type, dataset_id, config_yaml)
     │── gRPC ──►
                  서버 TrainingServicer
                  └── worker thread: 데이터 로딩 → 학습 루프
                        → checkpoint 저장 (/data/lerobot/checkpoints/)
                        → 로그 스트리밍 (StreamTrainingLogs)
```

실행 명령:
```bash
python scripts/run_desktop.py train \
    --config module/config/desktop.yaml \
    --model-type act \
    --dataset-id my_dataset \
    --follow   # 학습 로그 실시간 출력
```

---

### 5.3 자율 추론 모드 (`infer`)

학습된 모델을 서버에 로드하고, 로봇 관측을 실시간으로 서버에 전송해 행동을 수신한다.

```
SO-ARM101 팔로워
     │  (30 Hz 루프)
     ▼
follower.get_observation()     ← 카메라 이미지 + 관절값 읽기
     │
     ▼ gRPC GetAction
InferenceClient.get_action()   ──► 서버 InferenceServicer
     │                                  └── 모델 추론 (ACT / Diffusion / Pi0 등)
     ▼ gRPC 응답
connector.send_action(action)  ──► follower.send_action()
     │
     ▼
SO-ARM101 팔로워 동작 실행
```

실행 명령:
```bash
python scripts/run_desktop.py infer \
    --config module/config/desktop.yaml \
    --model-id run_001
```

---

### 5.4 전체 파이프라인 순서

```
1. [서버] run_server.py 실행
2. [데스크탑] collect 모드 → 데모 수집 + 서버 업로드
3. [데스크탑] train 모드 → 서버 학습 트리거
4. [데스크탑] infer 모드 → 자율 제어 루프
```

---

## 6. 서버 실행 (RunPod)

```bash
# RunPod pod에서 1회 실행 (환경 설정 + 서버 기동)
bash scripts/setup_runpod.sh

# 또는 직접 실행
python scripts/run_server.py \
    --config module/config/server.yaml \
    --regen-proto   # 최초 실행 시 proto 스텁 재생성
```

SSH 터널 연결 (데스크탑에서):
```bash
python scripts/open_tunnel.py \
    --pod-id $RUNPOD_POD_ID \
    --ssh-port $RUNPOD_SSH_PORT
```

`desktop.yaml`에서 서버 주소 설정:
```yaml
grpc:
  host: "localhost"   # SSH 터널 사용 시
  port: 50051
```

---

## 7. 환경 활성화 요약

```bash
# 매 세션 시작 시
conda activate vla-nlp

# 또는 PYTHONNOUSERSITE=1 으로 직접 실행 (user site-packages 충돌 방지)
PYTHONNOUSERSITE=1 /home/hands/miniforge3/envs/vla-nlp/bin/python scripts/run_desktop.py ...
```

> `PYTHONNOUSERSITE=1` 권장: `~/.local/lib/python3.x` 의 user-level 패키지가
> conda 환경 패키지와 충돌하는 것을 방지한다.
