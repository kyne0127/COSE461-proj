# SOArm-101 VLA 추론 파이프라인 노트

## 1. SOArm-101을 위한 VLA 추론 파이프라인 상세

### 1) 데이터 획득 (Local Edge)
- **Sensor Fusion**: ZED 카메라의 RGB+Depth 영상과 SOArm-101의 각 관절(Encoder) 상태를 동기화합니다.
- **Preprocessing**: 서버로 전송하기 전에 이미지를 리사이징하고, 관절값(7-DOF 등)을 Protobuf 메시지로 패킹합니다.

### 2) 통신 (Local ↔ Server)
- **Protocol**: gRPC 기반 양방향 스트리밍을 사용합니다.
- **Optimization**: 연구실 Wi-Fi 환경이 불안정하다면 HTTP/3(QUIC) 적용을 강력히 고려합니다. 패킷 지연으로 로봇 암이 멈칫하는 버벅임을 줄일 수 있습니다.

### 3) 추론 (GPU/TPU Server)
- **VLA Inference**: 서버에서 RT-2 또는 SmolVLA 같은 모델이 현재 상태를 받아 다음 동작(Action)을 계산합니다.
- **Output**: 텍스트가 아닌 이산화된 Action Tokens (`[q1, q2, ..., q7]`)을 반환합니다.

### 4) 실행 (Local Controller)
- **Action Chunking**: 서버 지연을 극복하기 위한 핵심입니다. 단일 관절값이 아니라 향후 `0.5s` 동작 시퀀스(Chunk)를 한 번에 전송합니다.
- **Interpolation**: 로컬에서 듬성듬성한 명령 사이를 보간해 SOArm-101을 부드럽게 구동합니다.

## 2. 실제 구현 시 마주할 성능 병목 포인트

SOArm-101에 이 아키텍처를 적용할 때 특히 주의해야 할 항목:

### 1) Serialization (직렬화) 속도
- 이미지 데이터를 `bytes`로 변환할 때 CPU 점유율이 급증하면 로컬 제어 주기(Control Loop)가 깨질 수 있습니다.
- **Tip**: 일반 JPEG 인코딩 대신 TurboJPEG 또는 NVJPEG 같은 하드웨어 가속 인코더를 고려합니다.

### 2) 동작 청킹(Action Chunking) 유무
- 로컬과 서버가 `1:1` 단일 스텝 통신만 하면 네트워크 흔들림 시 로봇 암 움직임이 끊길 수 있습니다.
- **해결**: 한 번의 추론으로 여러 단계의 미래 동작을 받아 로컬 오픈 루프로 실행하는 로직을 반드시 포함합니다.

### 3) 비상 정지(E-Stop) 로직
- 서버(원격 두뇌)와 통신이 끊기면 로봇이 마지막 명령을 계속 수행해 사고로 이어질 수 있습니다.
- **해결**: 로컬 코드에 Watchdog 타이머를 두고, 서버 명령이 일정 시간(예: `100ms`) 이상 없으면 즉시 정지하거나 안전 자세로 복귀하도록 합니다.

## 3. 추천 기술 스택 조합

성민님의 학업/프로젝트 배경을 고려한 효율적인 조합:

| 레이어 | 추천 기술 |
|---|---|
| 로봇 프레임워크 | LeRobot (Hugging Face) + ROS2 |
| 통신 인프라 | gRPC over HTTP/2 (또는 전용 UDP 스트리밍) |
| 데이터 포맷 | Protocol Buffers |
| 추론 서버 | NVIDIA Triton Inference Server |