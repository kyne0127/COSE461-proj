#!/usr/bin/env python3
"""
scripts/run_server.py
======================
GPU 서버 실행 스크립트.

사용법:
    # 기본 설정 (config/server.yaml)
    python scripts/run_server.py

    # 설정 파일 지정
    python scripts/run_server.py --config /path/to/server.yaml

    # 포트 오버라이드
    python scripts/run_server.py --port 50052

    # proto 스텁 재생성 후 실행
    python scripts/run_server.py --regen-proto
"""

import argparse
import os
import subprocess
import sys

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def regen_proto() -> None:
    """proto/lerobot.proto → lerobot_pb2.py + lerobot_pb2_grpc.py 재생성."""
    root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proto_dir = os.path.join(root, "module", "proto")
    proto_file = os.path.join(proto_dir, "lerobot.proto")

    print(f"[proto] Generating stubs from {proto_file} ...")
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        proto_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[proto] ERROR:", result.stderr)
        sys.exit(1)
    print("[proto] ✓ Stubs generated in", proto_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot gRPC GPU Server")
    parser.add_argument("--config",      default="module/config/server.yaml")
    parser.add_argument("--host",        default=None)
    parser.add_argument("--port",        type=int, default=None)
    parser.add_argument("--log-level",   default="INFO")
    parser.add_argument("--regen-proto", action="store_true",
                        help="Regenerate gRPC stubs from .proto file before starting")
    args = parser.parse_args()

    if args.regen_proto:
        regen_proto()

    # Import after proto regen so stubs are fresh
    from module.utils.logging import setup_logging
    setup_logging(level=args.log_level, component="lerobot")

    import yaml
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
        print(f"[server] Loaded config from {args.config}")
    else:
        print(f"[server] Config not found at {args.config} — using defaults")

    if args.host:
        config.setdefault("server", {})["host"] = args.host
    if args.port:
        config.setdefault("server", {})["port"] = args.port

    from module.server.grpc_server import LeRobotGRPCServer
    server = LeRobotGRPCServer.from_config(config)
    server.start()

    print("[server] ✓ Running. Press Ctrl+C to stop.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[server] Shutting down ...")
        server.stop()


if __name__ == "__main__":
    main()
