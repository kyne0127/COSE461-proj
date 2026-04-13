# LeRobot Remote System Specification

작성일: 2026-04-12
버전: 0.1 (현재 구현 기준)

## 1. 문서 목적

본 문서는 현재 코드베이스에 구현된 LeRobot 원격 학습/추론 시스템의 기능 명세를 정의한다.
대상은 데스크탑(로봇 연결)과 GPU 서버(RunPod 포함) 간 gRPC 기반 분산 파이프라인이다.

## 2. 시스템 범위

### 2.1 포함 범위

- 데스크탑 측 로봇 연결/데이터 수집/추론 클라이언트
- GPU 서버 측 모델 로드/추론/학습 잡 관리
- gRPC 프로토콜(InferenceService, TrainingService, GenericInferenceService, HealthService)
- 모델 플러그인 구조(ACT, Diffusion, TD-MPC2, Pi0, Custom)
- 에피소드 데이터 업로드 및 서버 저장 포맷

### 2.2 비포함 범위

- 사용자 인증/권한 관리
- E2E 암호화(TLS 기본 적용)
- 분산 학습(멀티 노드) 스케줄링
- 실서비스 수준 모니터링 스택(Prometheus, Grafana 등)

## 3. 전체 아키텍처

```text
[Desktop (RTX 3060)]                                [GPU Server (RunPod/On-prem)]

RobotConnector  -> RawObservation            gRPC Server
DataCollector   -> EpisodeBuffer             - InferenceServicer
TrainingClient  -> StreamEpisode ----------> - TrainingServicer
InferenceClient -> GetAction   ----------->  - GenericServicer
                                            - HealthServicer

Server Storage:
- datasets: /data/lerobot/datasets/<dataset_id>/episode_xxxxxx
- checkpoints: /data/lerobot/checkpoints/<model_type>/<job_id>/
```

## 4. 주요 컴포넌트 명세

### 4.1 Desktop

- RobotConnector
  - LeRobot robot device 동적 로딩 및 연결 관리
  - 관측 수집: 카메라/상태(state)
  - 행동 실행: action 전송
  - 텔레오퍼레이션 step 지원
- DataCollector
  - 텔레오퍼레이션 기반 demonstration 수집
  - `fps`, `max_episode_steps`, `warmup_steps` 기반 루프 제어
  - 결과를 EpisodeBuffer에 적재
- InferenceClient
  - 서버 모델 로드/언로드/목록 조회
  - 단일 스텝 추론(`GetAction`) 및 스트리밍 추론(`GetActionStream`)
- TrainingClient
  - 에피소드 스트리밍 업로드(`StreamEpisode`)
  - 학습 잡 시작/상태조회/로그스트림/중지

### 4.2 Server

- LeRobotGRPCServer
  - 단일 포트에서 4개 gRPC service 호스팅
  - 기본 바인딩: `0.0.0.0:50051`
- InferenceServicer
  - 모델 메모리 상주 관리(ModelStore)
  - 모델별 action 추론 처리
  - chunk 기반 모델의 stream 출력 지원
- TrainingServicer
  - episode 수신 후 디스크 저장
  - worker thread 기반 학습 잡 큐 처리
  - 체크포인트 저장, 로그 스트리밍, 잡 상태 관리
- GenericServicer
  - handler_id + method + payload_json 기반 범용 라우팅
  - BaseHandler 기반 baseline 확장 구조

## 5. 실행 모드 명세 (Desktop CLI)

엔트리포인트: `scripts/run_desktop.py`

- `collect`
  - N개 에피소드 텔레오퍼레이션 수집
  - 수집 종료 후 서버로 에피소드 업로드
  - 주요 옵션: `--n-episodes`, `--task`, `--dataset-id`
- `infer`
  - 로봇 observation을 서버로 전달하고 action 수신
  - 수신 action을 로봇에 즉시 전송
  - 주요 옵션: `--model-id`
- `train`
  - 서버에 학습 잡 트리거
  - 주요 옵션: `--model-type`, `--model-config`, `--dataset-id`, `--follow`
- `status`
  - 단일 job 또는 전체 job 상태 조회
  - 주요 옵션: `--job-id`

## 6. 서버 실행 명세

엔트리포인트: `scripts/run_server.py`

- 설정 로딩: 기본 `module/config/server.yaml`
- 옵션:
  - `--config`
  - `--host`, `--port` (config override)
  - `--regen-proto` (proto 스텁 재생성)
- 서버 시작 후 `wait_for_termination()`로 블로킹

## 7. gRPC 인터페이스 명세

프로토 파일: `module/proto/lerobot.proto`

### 7.1 InferenceService

- `LoadModel(LoadModelRequest) -> LoadModelResponse`
  - 입력: model_type, model_id, checkpoint_path, config_yaml
  - 출력: success, message
- `UnloadModel(UnloadModelRequest) -> UnloadModelResponse`
- `ListModels(ListModelsRequest) -> ListModelsResponse`
- `GetAction(ObservationRequest) -> ActionResponse`
- `GetActionStream(ObservationRequest) -> stream ActionChunk`

ObservationRequest 주요 필드:
- model_id
- images: repeated Tensor
- state: Tensor
- task_text
- episode_id, step

### 7.2 TrainingService

- `StreamEpisode(stream EpisodeFrame) -> EpisodeUploadResponse`
- `StartTraining(TrainingRequest) -> TrainingJobResponse`
- `GetTrainingStatus(TrainingStatusRequest) -> TrainingStatusResponse`
- `StreamTrainingLogs(TrainingStatusRequest) -> stream TrainingLog`
- `StopTraining(TrainingStatusRequest) -> TrainingJobResponse`
- `ListJobs(ListJobsRequest) -> ListJobsResponse`

TrainingStatus.status 값:
- queued
- running
- done
- failed
- stopped

### 7.3 GenericInferenceService

- `Infer(GenericRequest) -> GenericResponse`
- `InferStream(GenericRequest) -> stream GenericResponse`

핵심 컨셉:
- `handler_id`로 핸들러 선택
- `method`로 핸들러 내부 동작 선택
- `payload_json`으로 범용 JSON 파라미터 전달

### 7.4 HealthService

- `Ping(PingRequest) -> PingResponse`
- 반환 정보: server_id, gpu_info, gpu_mem_gb

## 8. 데이터 명세

### 8.1 전송 단위

- Tensor
  - `shape: repeated int64`
  - `data: repeated float` (flattened)
  - `dtype: string`
  - `name: string`

### 8.2 서버 저장 구조

- 루트: `/data/lerobot/datasets/<dataset_id>/`
- 에피소드: `episode_000000/`
- 저장 파일:
  - `states.npy`
  - `actions.npy`
  - `rewards.npy`
  - `dones.npy`
  - `<camera_name>/<frame_idx>.npy`
- 메타:
  - `dataset_meta.json` (episodes 목록)

## 9. 모델 플러그인 명세

### 9.1 공통 인터페이스

`BaseLeRobotModel` 구현 필수 메서드:
- `load_checkpoint(path)`
- `predict_action(observation)`
- `train_step(batch)`

옵션 메서드:
- `reset()`
- `save_checkpoint(path)`

### 9.2 등록 및 디스커버리

- ModelRegistry 등록 키:
  - act
  - diffusion
  - tdmpc2
  - pi0
  - custom
- 서버는 미등록 시 `auto_discover()` 수행

### 9.3 현재 구현 상태

- ACT: chunk action buffer 기반 실행
- Diffusion: prediction/action horizon 기반 버퍼 실행
- TD-MPC2: MPC 기반 action 선택
- Pi0: task_text(언어 지시) 기반 VLA action 생성
- Custom: 예제 템플릿(MVP 수준)

## 10. 설정 파일 명세

### 10.1 desktop.yaml

- `grpc`: host, port, use_tls, timeout_secs
- `robot`: robot_type, fps, robot_config
- `collection`: max_episode_steps, warmup_steps, dataset_id
- `inference`: model_id, model_type
- `logging`: level, log_file

### 10.2 server.yaml

- `server`: host, port, max_workers, train_workers
- `data_root`, `checkpoint_root`, `log_dir`
- `logging`: level, log_file
- `gpu`: device, mixed_precision, cudnn_benchmark

## 11. 동작 시퀀스

### 11.1 데이터 수집 + 학습

1. Desktop collect 모드 시작
2. RobotConnector에서 teleop step 반복
3. EpisodeBuffer에 프레임 축적
4. TrainingClient로 `StreamEpisode` 업로드
5. Desktop train 모드로 `StartTraining` 요청
6. Server worker가 dataset 로딩 후 학습 루프 수행
7. 주기적 checkpoint 저장 및 로그 스트리밍

### 11.2 실시간 추론

1. InferenceClient가 서버 `LoadModel` 호출
2. 루프마다 observation 전송(`GetAction`)
3. 서버가 모델 추론 후 action 반환
4. Desktop이 로봇에 action 적용

## 12. 오류 처리 정책

- 모델 미로드 상태에서 추론 요청 시 서버 예외 반환
- 잘못된 YAML config는 `StartTraining` 단계에서 reject
- 학습 중 예외 발생 시 job status=`failed` 및 message 기록
- SIGINT 수신 시 data collection은 현재 episode 종료 후 중단

## 13. 비기능 요구사항(현재 구현 기준)

- 성능
  - 데스크탑 제어 루프 목표: robot fps(기본 30Hz)
  - gRPC 메시지 최대 송수신 크기: 256MB 설정
- 동시성
  - gRPC worker pool: 기본 10
  - 학습 worker thread: 기본 2
- 이식성
  - Python 3.10+
  - Windows/Linux 환경에서 클라이언트 스크립트 실행 가능

## 14. 알려진 제약사항

- 기본 gRPC 채널은 insecure(평문)이며 인증 미구현
- 학습 데이터셋은 로컬 파일 구조에 의존
- 모델별 입력 스키마 검증이 강하게 통합되어 있지 않음
- 프로덕션 수준 재시도/백오프/서킷브레이커 미구현
- 분산 학습/모델 버전 관리 체계는 별도 구성 필요

## 15. 향후 확장 권장

- TLS + 인증 토큰 기반 보안 강화
- 모델/핸들러 스키마 검증 레이어 추가
- 학습/추론 메트릭 수집 및 대시보드 통합
- 체크포인트/데이터셋 메타 버전 관리
- 통합 e2e 테스트(collect -> upload -> train -> infer) 자동화

## 16. 참고 구현 위치

- Desktop CLI: `scripts/run_desktop.py`
- Server CLI: `scripts/run_server.py`
- Proto: `module/proto/lerobot.proto`
- Server: `module/server/`
- Desktop: `module/desktop/`
- Models: `module/models/`
- Dataset utils: `module/utils/dataset.py`
