# RunPod GPU 서버 연동 가이드

LeRobot 데스크탑(RTX 3060)과 RunPod GPU 인스턴스를 연결하는 방법을 단계별로 설명합니다.

---

## 전체 아키텍처

```
[데스크탑 — RTX 3060]                      [RunPod GPU Pod]
┌─────────────────────────────┐             ┌──────────────────────────────┐
│  RobotConnector             │             │  grpc_server.py              │
│    LeRobot 하드웨어 연결     │             │    ├─ InferenceServicer      │
│                             │             │    │    모델 로드 / 추론      │
│  DataCollector              │    gRPC     │    └─ TrainingServicer       │
│    텔레오퍼레이션 녹화       │◄──────────►│         학습 잡 관리         │
│                             │  (SSH 터널) │                              │
│  InferenceClient            │             │  /data/lerobot/              │
│    실시간 action 수신        │             │    ├─ datasets/              │
│                             │             │    └─ checkpoints/           │
│  TrainingClient             │             │                              │
│    에피소드 업로드 / 학습 트리거│           │  GPU: RTX 4090 / A100 등    │
└─────────────────────────────┘             └──────────────────────────────┘
```

---

## 방법 선택

| 방법 | 난이도 | 보안 | 추가 비용 | 추천 상황 |
|------|--------|------|-----------|-----------|
| **A. SSH 터널** | ★★☆ | ✅ 높음 | 없음 | **실사용 권장** |
| **B. TCP 포트 직접 노출** | ★☆☆ | ⚠ 낮음 | TCP 포트 비용 | 빠른 테스트 |
| **C. RunPod Serverless** | ★★★ | ✅ 높음 | API 요금 | 추론 전용 |

---

## 방법 A — SSH 터널 (권장)

포트를 외부에 노출하지 않고 SSH 터널로 gRPC 트래픽을 암호화하여 전달합니다.

### 1단계. RunPod Pod 생성

1. [RunPod 콘솔](https://www.runpod.io/console/pods) 접속
2. **Deploy** 클릭
3. GPU 선택 (RTX 4090 / A100 권장)
4. 템플릿: `RunPod PyTorch 2.x` 선택
5. **Storage**: Network Volume 연결 권장 (데이터 영속성)
6. **Expose Ports**: `22` (SSH) 만 노출 — gRPC 포트는 노출 불필요
7. **Deploy**

> **⚠ 중요:** SSH 포트만 노출하면 됩니다. gRPC 포트(50051)는 SSH 터널로 전달되므로 외부 노출 불필요.

### 2단계. Pod SSH 정보 확인

```
RunPod 대시보드 → Pod → Connect 버튼 클릭
→ "SSH over exposed TCP" 섹션 확인

예시 SSH 명령:
  ssh root@abc123def456.ssh.runpod.net -p 22042 -i ~/.ssh/id_rsa

여기서:
  POD_ID   = abc123def456
  SSH_PORT = 22042
```

### 3단계. Pod 초기 설정 (최초 1회)

```bash
# 데스크탑에서 Pod에 SSH 접속
ssh root@abc123def456.ssh.runpod.net -p 22042

# 코드 업로드 (Pod 안에서)
git clone https://github.com/yourname/your-lerobot-repo.git /workspace
cd /workspace

# 또는 scp로 직접 전송
# scp -P 22042 -r ./module root@abc123def456.ssh.runpod.net:/workspace/
```

```bash
# Pod 안에서 환경 설정 실행
cd /workspace
bash scripts/setup_runpod.sh
```

`setup_runpod.sh`가 자동으로 수행하는 작업:
- GPU 확인 (`nvidia-smi`)
- Python 패키지 설치 (lerobot, grpcio, torch 등)
- proto 스텁 생성
- gRPC 서버 백그라운드 시작

### 4단계. SSH 터널 열기 (데스크탑)

```bash
# 방법 1 — 환경변수
export RUNPOD_POD_ID="abc123def456"
export RUNPOD_SSH_PORT="22042"
python scripts/open_tunnel.py

# 방법 2 — .env 파일 사용
cat > .env.runpod << EOF
RUNPOD_POD_ID=abc123def456
RUNPOD_SSH_PORT=22042
EOF
python scripts/open_tunnel.py --env .env.runpod

# 방법 3 — 직접 인자
python scripts/open_tunnel.py \
    --pod-id abc123def456 \
    --ssh-port 22042

# TensorBoard(6006) + Jupyter(8888) 동시 포워딩
python scripts/open_tunnel.py \
    --pod-id abc123def456 \
    --ssh-port 22042 \
    --extra-tunnels 6006:6006 8888:8888

# 자동 재연결 (연결 끊겨도 자동 복구)
python scripts/open_tunnel.py \
    --pod-id abc123def456 \
    --ssh-port 22042 \
    --auto-reconnect
```

터널이 성공적으로 열리면:
```
========================================================
  LeRobot RunPod SSH Tunnel
========================================================
  Host       : root@abc123def456.ssh.runpod.net
  SSH Port   : 22042
  gRPC Tunnel: localhost:50051 → server:50051
  Reconnect  : ON
========================================================
[tunnel] Connecting ... (attempt 1)
[tunnel] ✓ Tunnel established
[tunnel]   gRPC: localhost:50051 → RunPod:50051
[tunnel]   Press Ctrl+C to close
```

### 5단계. desktop.yaml 설정

터널이 열린 상태에서는 `localhost`로 접속합니다.

```yaml
# module/config/desktop.yaml
grpc:
  host: "localhost"   # SSH 터널을 통해 RunPod으로 라우팅
  port: 50051
  use_tls: false
  timeout_secs: 5.0
```

### 6단계. 연결 확인

```bash
python scripts/check_connection.py
```

정상 출력 예시:
```
============================================================
  LeRobot Connection Health Check
  2024-01-15 14:23:45
  Target: localhost:50051
============================================================

  ✓ [proto 스텁] 스텁 생성됨 (last modified: 2024-01-15 13:00)
  ✓ [설정 파일] module/config/desktop.yaml
       host=localhost  port=50051  평문
  ✓ [SSH 터널] SSH 터널 활성 (PID: 12345)
  ✓ [TCP 연결] TCP localhost:50051 open
  ✓ [gRPC Ping] 응답 성공
       서버 ID : lerobot-gpu-server
       GPU     : NVIDIA RTX 4090
       VRAM    : 24.0 GB

============================================================
  ✓ 모든 체크 통과 — 서버와 정상 연결됨
============================================================
```

---

## 방법 B — TCP 포트 직접 노출

빠른 테스트 목적으로 gRPC 포트를 외부에 직접 노출합니다.

> ⚠ 인증이 없으므로 테스트 후 반드시 Pod를 종료하거나 TLS를 적용하세요.

### 1단계. Pod 생성 시 포트 노출 설정

```
Deploy → Edit Template
→ Expose TCP Ports: 50051
→ Deploy
```

RunPod이 발급하는 외부 엔드포인트 확인:
```
Pod → Connect → "TCP Port Mappings" 섹션

예시:
  50051 → abc123def456-50051.proxy.runpod.net:15432
                                               ↑ 외부 포트
```

### 2단계. desktop.yaml 설정

```yaml
grpc:
  host: "abc123def456-50051.proxy.runpod.net"
  port: 15432    # RunPod이 발급한 외부 포트
  use_tls: false
```

### 3단계. 서버 시작 후 바로 연결

SSH 터널 불필요 — 바로 `check_connection.py` 실행:

```bash
python scripts/check_connection.py \
    --host abc123def456-50051.proxy.runpod.net \
    --port 15432
```

---

## 방법 C — RunPod Serverless (추론 전용)

학습은 Secure Pod에서, 추론만 Serverless Endpoint로 분리하는 방식입니다.
사용량만큼만 과금되어 유휴 비용 없음.

### Serverless Handler 코드 (Pod에서 실행)

```python
# handler.py (RunPod Serverless용)
import runpod
import base64, numpy as np
from module.utils.registry import ModelRegistry

ModelRegistry.auto_discover()
_model = None

def load_model_once(model_type, checkpoint_path):
    global _model
    if _model is None:
        _model = ModelRegistry.build(model_type)
        _model.load_checkpoint(checkpoint_path)
    return _model

def handler(job):
    inp        = job["input"]
    model_type = inp.get("model_type", "act")
    ckpt_path  = inp.get("checkpoint_path", "/checkpoints/latest")
    model      = load_model_once(model_type, ckpt_path)

    # 이미지 디코딩
    from module.models.base_model import Observation
    images = {}
    for cam, b64 in inp.get("images", {}).items():
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        images[cam] = arr.reshape(inp["image_shapes"][cam])

    state = np.array(inp.get("state", []), dtype=np.float32)
    obs   = Observation(images=images, state=state,
                        task_text=inp.get("task_text", ""))
    action = model.predict_action(obs)
    return {"action": action.tolist()}

runpod.serverless.start({"handler": handler})
```

### 데스크탑 클라이언트

```python
# module/desktop/runpod_client.py 사용
from module.desktop.runpod_client import RunPodInferenceClient

client = RunPodInferenceClient(
    endpoint_id="your-endpoint-id",  # RunPod Serverless 엔드포인트 ID
    api_key="your-runpod-api-key",
)
action = client.get_action(images={"top": img}, state=state, model_id="act")
```

---

## 일반적인 문제 해결

### gRPC 연결 거부 (Connection Refused)

```
✗ [TCP 연결] TCP localhost:50051 refused
```

원인과 해결:
1. SSH 터널이 열려있지 않음 → `python scripts/open_tunnel.py` 실행
2. 서버가 Pod에서 실행 중이 아님 → Pod에 SSH 접속 후 `python scripts/run_server.py` 실행
3. 포트 번호 불일치 → `desktop.yaml`과 서버 포트 확인

### SSH 터널 연결 실패

```
[tunnel] Connection failed: Permission denied (publickey)
```

원인과 해결:
- RunPod에 SSH 공개키가 등록되어 있지 않음
- RunPod 콘솔 → Settings → SSH Keys에서 `~/.ssh/id_rsa.pub` 내용 등록
- 또는: `ssh-keygen -t rsa -b 4096`으로 새 키 생성 후 등록

### proto 스텁 없음

```
✗ [proto 스텁] 스텁 없음: lerobot_pb2.py, lerobot_pb2_grpc.py
```

해결:
```bash
# 데스크탑에서
bash scripts/gen_proto.sh

# Pod 안에서 (setup_runpod.sh가 자동 실행하지만 수동으로도 가능)
bash scripts/gen_proto.sh
```

### 터널은 연결되는데 gRPC 응답 없음

```
✓ [TCP 연결] TCP localhost:50051 open
✗ [gRPC Ping] 실패 — StatusCode.UNAVAILABLE
```

Pod 안에서 서버 상태 확인:
```bash
# Pod SSH 접속 후
tail -50 /data/lerobot/logs/server.log
ps aux | grep run_server

# 서버 재시작
pkill -f run_server.py
python scripts/run_server.py &
```

### 학습 중 연결 끊김

학습 중 터널이 끊겨도 서버는 계속 실행됩니다. 터널 재연결 후 상태를 확인하면 됩니다:

```bash
# 터널 재연결
python scripts/open_tunnel.py --pod-id abc123 --ssh-port 22042

# 학습 상태 확인
python scripts/run_desktop.py status
```

---

## .env 파일 관리

매번 Pod ID와 포트를 입력하지 않으려면 `.env.runpod` 파일을 사용합니다.

```bash
# .env.runpod 생성 (gitignore에 추가할 것)
cat > .env.runpod << 'EOF'
RUNPOD_POD_ID=abc123def456
RUNPOD_SSH_PORT=22042
RUNPOD_API_KEY=rp_xxxxxxxxxxxx
EOF

# .gitignore에 추가
echo ".env.runpod" >> .gitignore
echo ".env*"       >> .gitignore
```

사용:
```bash
python scripts/open_tunnel.py --env .env.runpod
```

---

## 전체 실행 순서 요약

```bash
# ── [RunPod Pod에서] 최초 1회 ──────────────────────────
ssh root@{POD_ID}.ssh.runpod.net -p {SSH_PORT}
cd /workspace && bash scripts/setup_runpod.sh
exit

# ── [데스크탑] 터미널 1 — 터널 유지 ────────────────────
python scripts/open_tunnel.py --env .env.runpod

# ── [데스크탑] 터미널 2 — 작업 실행 ───────────────────
# 연결 확인
python scripts/check_connection.py

# 데이터 수집 (로봇 연결)
python scripts/run_desktop.py collect \
    --n-episodes 20 \
    --task "pick up the red block"

# 학습 트리거
python scripts/run_desktop.py train \
    --model-type act \
    --model-config module/config/models/act.yaml \
    --follow

# 학습 완료 후 모델 로드 확인
python scripts/run_desktop.py status

# 자율 제어
python scripts/run_desktop.py infer --model-id run_001
```

---

## Pod 비용 최적화 팁

- **학습 중에만 Pod 시작:** Network Volume에 데이터/체크포인트를 저장하면 Pod를 종료해도 데이터가 유지됩니다.
- **스팟 인스턴스(Spot) 활용:** 학습 중단 시 자동 재개 로직을 `TrainingServicer`의 `resume_from` 옵션으로 처리합니다.
- **추론은 저렴한 GPU로:** 학습(A100)과 추론(RTX 4000 Ada)을 다른 Pod 타입으로 분리하면 비용을 절감할 수 있습니다.