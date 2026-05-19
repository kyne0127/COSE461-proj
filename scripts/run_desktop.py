#!/usr/bin/env python3
"""
scripts/run_desktop.py
=======================
데스크탑(RTX 3060) 실행 스크립트.

4가지 모드 지원:
  collect   — 텔레오퍼레이션으로 데이터 수집 → 서버에 업로드
  infer     — 서버 모델로 실시간 추론 → 로봇 자율 제어
  pipeline  — VLM 전처리 핸들러 + 액션 모델 파이프라인 실행
  train     — 서버에 학습 잡 트리거

사용법:
    # 1. 데이터 수집 (10 에피소드)
    python scripts/run_desktop.py collect \
        --config module/config/desktop.yaml \
        --n-episodes 10 \
        --task "pick up the red block"

    # 2. 실시간 자율 제어 (핸들러 없이 액션 모델 직접 호출)
    python scripts/run_desktop.py infer \
        --config module/config/desktop.yaml \
        --model-id run_001

    # 3. VLM 핸들러 + 액션 모델 파이프라인 (AmbRes + Pi0 등)
    python scripts/run_desktop.py pipeline \
        --config module/config/desktop.yaml \
        --pipeline-config module/config/pipelines/ambres_pi0.yaml \
        --task "pick up the banana"

    # 4. 서버에 학습 트리거
    python scripts/run_desktop.py train \
        --config module/config/desktop.yaml \
        --model-type act \
        --model-config module/config/models/act.yaml
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path: str) -> dict:
    import yaml
    if not os.path.exists(path):
        print(f"[desktop] WARNING: config not found at {path}, using defaults")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ────────────────────────────────────────────────────────────────────────────
# collect mode
# ────────────────────────────────────────────────────────────────────────────

def run_collect(args, config: dict) -> None:
    from module.desktop.robot_connector import RobotConnector
    from module.desktop.data_collector import DataCollector
    from module.desktop.grpc_client import TrainingClient
    from module.utils.dataset import EpisodeBuffer
    from module.utils.logging import setup_logging

    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        component="lerobot.desktop",
    )

    robot_cfg = config.get("robot", {})
    grpc_cfg  = config.get("grpc", {})
    coll_cfg  = config.get("collection", {})

    dataset_id = args.dataset_id or coll_cfg.get("dataset_id", "default_dataset")
    n_episodes = args.n_episodes
    task_text  = args.task

    print(f"[collect] dataset_id={dataset_id}  n_episodes={n_episodes}")
    print(f"[collect] task: '{task_text}'")

    # Build components
    buffer    = EpisodeBuffer(dataset_id=dataset_id)
    connector = RobotConnector.from_config(robot_cfg)

    collector = DataCollector(
        robot=connector,
        buffer=buffer,
        fps=robot_cfg.get("fps", 30.0),
        max_episode_steps=coll_cfg.get("max_episode_steps", 500),
        warmup_steps=coll_cfg.get("warmup_steps", 10),
    )

    # Collect
    with connector.session(calibrate=args.calibrate):
        stats = collector.collect_episodes(
            n_episodes=n_episodes,
            task_text=task_text,
        )

    print(f"\n[collect] ✓ Collected {stats.n_episodes} eps / {stats.n_frames} frames "
          f"in {stats.elapsed_secs:.1f}s")

    # Upload to server
    if stats.n_episodes > 0:
        print(f"\n[upload] Connecting to server {grpc_cfg.get('host')}:{grpc_cfg.get('port', 50051)} ...")
        with TrainingClient(
            host=grpc_cfg.get("host", "localhost"),
            port=grpc_cfg.get("port", 50051),
        ) as client:
            n_ok = client.upload_all_episodes(buffer.completed_episodes)
            print(f"[upload] ✓ {n_ok}/{stats.n_episodes} episodes uploaded")


# ────────────────────────────────────────────────────────────────────────────
# infer mode
# ────────────────────────────────────────────────────────────────────────────

def run_infer(args, config: dict) -> None:
    from module.desktop.robot_connector import RobotConnector
    from module.desktop.grpc_client import InferenceClient
    from module.utils.logging import setup_logging

    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        component="lerobot.desktop",
    )

    robot_cfg = config.get("robot", {})
    grpc_cfg  = config.get("grpc", {})
    inf_cfg   = config.get("inference", {})

    model_id  = args.model_id or inf_cfg.get("model_id", "run_001")
    fps       = robot_cfg.get("fps", 30.0)
    dt        = 1.0 / fps

    connector = RobotConnector.from_config(robot_cfg)

    with InferenceClient(
        host=grpc_cfg.get("host", "localhost"),
        port=grpc_cfg.get("port", 50051),
    ) as client:
        # Ping
        pong = client.ping()
        print(f"[infer] Server: {pong['gpu_info']} ({pong['gpu_mem_gb']:.1f} GB)")

        with connector.session():
            print(f"[infer] Running inference with model '{model_id}' at {fps:.0f} fps")
            print("[infer] Press Ctrl+C to stop")
            ep_id = 0
            step  = 0

            try:
                while True:
                    t0  = time.perf_counter()
                    obs = connector.get_observation()

                    action = client.get_action(
                        model_id=model_id,
                        images=obs.images,
                        state=obs.state,
                        episode_id=ep_id,
                        step=step,
                    )
                    connector.send_action(action)
                    step += 1

                    elapsed = time.perf_counter() - t0
                    sleep_t = dt - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

            except KeyboardInterrupt:
                print(f"\n[infer] Stopped after {step} steps")


# ────────────────────────────────────────────────────────────────────────────
# pipeline mode
# ────────────────────────────────────────────────────────────────────────────

def run_pipeline(args, config: dict) -> None:
    """
    VLM 전처리 핸들러 + 액션 모델 파이프라인 실행.

    파이프라인 설정(--pipeline-config)에 pre_handlers 목록과 action_model_id를
    정의하면 코드 변경 없이 어떤 핸들러 조합이든 실행할 수 있습니다.

    e.g. AmbRes(모호성 해소) → Pi0(액션 생성) → 로봇 제어
    """
    from module.desktop.pipeline import InferencePipeline, load_pipeline_config
    from module.desktop.robot_connector import RobotConnector
    from module.utils.logging import setup_logging

    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        component="lerobot.pipeline",
    )

    robot_cfg    = config.get("robot", {})
    grpc_cfg     = config.get("grpc", {})

    pipeline_cfg = load_pipeline_config(args.pipeline_config)

    # CLI 옵션으로 action_model_id / fps 오버라이드 가능
    if args.model_id:
        pipeline_cfg.action_model_id = args.model_id
    if args.fps:
        pipeline_cfg.fps = args.fps

    if pipeline_cfg.local_model_type:
        print(f"[pipeline] action_model  : {pipeline_cfg.local_model_type} (local)")
        print(f"[pipeline] checkpoint    : {pipeline_cfg.local_model_checkpoint or '(HuggingFace)'}")
    else:
        print(f"[pipeline] action_model  : {pipeline_cfg.action_model_id} (server)")
    print(f"[pipeline] fps           : {pipeline_cfg.fps}")
    print(f"[pipeline] pre_handlers  : "
          f"{[s.handler_id for s in pipeline_cfg.pre_handlers]}")
    print(f"[pipeline] task          : '{args.task}'")
    if args.n_episodes:
        print(f"[pipeline] n_episodes    : {args.n_episodes}")
    else:
        print("[pipeline] n_episodes    : ∞ (Ctrl+C to stop)")

    connector = RobotConnector.from_config(robot_cfg)

    with InferencePipeline(
        pipeline_cfg=pipeline_cfg,
        robot_connector=connector,
        grpc_host=grpc_cfg.get("host", "localhost"),
        grpc_port=grpc_cfg.get("port", 50051),
        use_tls=grpc_cfg.get("use_tls", False),
        timeout=grpc_cfg.get("timeout_secs", 60.0),
    ) as pipe:
        pipe.run(
            task_text=args.task,
            n_episodes=args.n_episodes,
        )


# ────────────────────────────────────────────────────────────────────────────
# train trigger mode
# ────────────────────────────────────────────────────────────────────────────

def run_train(args, config: dict) -> None:
    from module.desktop.grpc_client import TrainingClient
    import yaml

    grpc_cfg   = config.get("grpc", {})
    model_type = args.model_type
    dataset_id = args.dataset_id or config.get("collection", {}).get("dataset_id", "default_dataset")

    model_config_path = args.model_config or f"module/config/models/{model_type}.yaml"
    if not os.path.exists(model_config_path):
        print(f"[train] ERROR: model config not found: {model_config_path}")
        sys.exit(1)

    with open(model_config_path) as f:
        config_yaml = f.read()

    print(f"[train] Triggering training on server")
    print(f"  model_type = {model_type}")
    print(f"  dataset_id = {dataset_id}")
    print(f"  config     = {model_config_path}")

    with TrainingClient(
        host=grpc_cfg.get("host", "localhost"),
        port=grpc_cfg.get("port", 50051),
    ) as client:
        job_id = client.start_training(
            model_type=model_type,
            dataset_id=dataset_id,
            config_yaml=config_yaml,
        )
        print(f"[train] ✓ Job started: {job_id}")

        if args.follow:
            print("[train] Following logs (Ctrl+C to detach) ...")
            try:
                for line in client.stream_logs(job_id):
                    print(line)
            except KeyboardInterrupt:
                print("[train] Detached from log stream")
        else:
            print(f"[train] Monitor with: python scripts/run_desktop.py status --job-id {job_id}")


# ────────────────────────────────────────────────────────────────────────────
# status mode
# ────────────────────────────────────────────────────────────────────────────

def run_status(args, config: dict) -> None:
    from module.desktop.grpc_client import TrainingClient

    grpc_cfg = config.get("grpc", {})
    with TrainingClient(
        host=grpc_cfg.get("host", "localhost"),
        port=grpc_cfg.get("port", 50051),
    ) as client:
        if args.job_id:
            status = client.get_status(args.job_id)
            for k, v in status.items():
                print(f"  {k:20s}: {v}")
        else:
            jobs = client.list_jobs()
            if not jobs:
                print("[status] No jobs found")
            for j in jobs:
                print(f"  {j['job_id']:12s} | {j['status']:10s} | loss={j.get('loss', 'N/A')}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeRobot Desktop Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="module/config/desktop.yaml")
    sub = parser.add_subparsers(dest="mode", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Record teleoperation episodes")
    p_collect.add_argument("--n-episodes",  type=int, default=10)
    p_collect.add_argument("--task",        default="")
    p_collect.add_argument("--dataset-id",  default=None)
    p_collect.add_argument("--calibrate",    action="store_true", default=False,
                           help="Run motor calibration on connect (interactive)")

    # infer
    p_infer = sub.add_parser("infer", help="Run autonomous inference")
    p_infer.add_argument("--model-id", default=None)

    # pipeline
    p_pipe = sub.add_parser(
        "pipeline",
        help="Run VLM pre-handler(s) + action model pipeline (e.g. AmbRes + Pi0)",
    )
    p_pipe.add_argument(
        "--pipeline-config", required=True,
        help="Path to pipeline YAML (e.g. module/config/pipelines/ambres_pi0.yaml)",
    )
    p_pipe.add_argument("--task",       default="", help="Default task description")
    p_pipe.add_argument("--model-id",   default=None, help="Override action_model_id")
    p_pipe.add_argument("--fps",        type=float, default=None, help="Override fps")
    p_pipe.add_argument("--n-episodes", type=int,   default=None,
                        help="Number of episodes to run (default: infinite)")

    # train
    p_train = sub.add_parser("train", help="Trigger server-side training")
    p_train.add_argument("--model-type",   required=True,
                         choices=["act", "diffusion", "tdmpc2", "pi0", "smolvla", "custom"])
    p_train.add_argument("--model-config", default=None)
    p_train.add_argument("--dataset-id",   default=None)
    p_train.add_argument("--follow",       action="store_true",
                         help="Follow training logs until completion")

    # status
    p_status = sub.add_parser("status", help="Check training job status")
    p_status.add_argument("--job-id", default=None)

    args = parser.parse_args()
    config = load_config(args.config)

    dispatch = {
        "collect":  run_collect,
        "infer":    run_infer,
        "pipeline": run_pipeline,
        "train":    run_train,
        "status":   run_status,
    }
    dispatch[args.mode](args, config)


if __name__ == "__main__":
    main()
