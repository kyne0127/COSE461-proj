#!/usr/bin/env python3
"""
scripts/open_tunnel.py
=======================
RunPod SSH 터널 자동화 스크립트.

SSH 터널을 통해 데스크탑 로컬 포트(50051)를
RunPod GPU 서버의 gRPC 포트(50051)로 포워딩합니다.

사용법:
    # 환경변수로 설정
    export RUNPOD_POD_ID="abc123def456"
    export RUNPOD_SSH_PORT="22042"
    python scripts/open_tunnel.py

    # 인자로 직접 전달
    python scripts/open_tunnel.py --pod-id abc123def456 --ssh-port 22042

    # .env 파일 사용
    python scripts/open_tunnel.py --env .env.runpod

    # 백그라운드 실행 (자동 재연결)
    python scripts/open_tunnel.py --auto-reconnect --daemon

RunPod 대시보드에서 SSH 정보 확인:
    Pod → Connect → SSH over exposed TCP → 명령어에서 포트/ID 추출
    예: ssh root@abc123def456.ssh.runpod.net -p 22042
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────────
# SSH 호스트 포맷 (RunPod 제공 방식)
# ────────────────────────────────────────────────────────────────────────────

RUNPOD_SSH_HOST_FORMAT = "{pod_id}.ssh.runpod.net"


def build_ssh_command(
    pod_id:        str,
    ssh_port:      int,
    local_port:    int  = 50051,
    remote_port:   int  = 50051,
    key_path:      str  = "",
    extra_tunnels: list[tuple[int, int]] | None = None,
) -> list[str]:
    """
    SSH 포트 포워딩 커맨드 생성.
    -L local_port:localhost:remote_port 형태로 터널 설정.
    """
    host = RUNPOD_SSH_HOST_FORMAT.format(pod_id=pod_id)

    cmd = [
        "ssh",
        "-N",                               # 커맨드 실행 없이 터널만 유지
        "-T",                               # 가상 TTY 비활성화
        "-o", "StrictHostKeyChecking=no",   # 최초 접속 시 yes/no 건너뜀
        "-o", "ServerAliveInterval=20",     # 20초마다 keepalive 패킷 전송
        "-o", "ServerAliveCountMax=3",      # 3번 실패 시 연결 종료
        "-o", "ExitOnForwardFailure=yes",   # 포트 포워딩 실패 시 즉시 종료
        "-L", f"{local_port}:localhost:{remote_port}",  # gRPC 터널
    ]

    # 추가 터널 (예: TensorBoard 6006, Jupyter 8888)
    if extra_tunnels:
        for lp, rp in extra_tunnels:
            cmd += ["-L", f"{lp}:localhost:{rp}"]

    # SSH 키 경로
    if key_path and Path(key_path).exists():
        cmd += ["-i", key_path]

    cmd += [f"root@{host}", "-p", str(ssh_port)]
    return cmd


def load_env_file(env_path: str) -> dict[str, str]:
    """간단한 .env 파서 (python-dotenv 없이 동작)."""
    env: dict[str, str] = {}
    path = Path(env_path)
    if not path.exists():
        print(f"[tunnel] WARNING: .env file not found: {env_path}")
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# ────────────────────────────────────────────────────────────────────────────
# 터널 프로세스 관리
# ────────────────────────────────────────────────────────────────────────────

class TunnelManager:
    """SSH 터널 프로세스를 시작/모니터링/재연결하는 매니저."""

    def __init__(
        self,
        cmd:             list[str],
        auto_reconnect:  bool  = True,
        max_retries:     int   = 10,
        retry_delay:     float = 5.0,
    ) -> None:
        self._cmd            = cmd
        self._auto_reconnect = auto_reconnect
        self._max_retries    = max_retries
        self._retry_delay    = retry_delay
        self._proc:          subprocess.Popen | None = None
        self._running        = True

        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def start(self) -> None:
        retries = 0
        cmd_display = " ".join(self._cmd)
        print(f"[tunnel] Command: {cmd_display}\n")

        while self._running:
            print(f"[tunnel] Connecting ... (attempt {retries + 1})")
            self._proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # 1초 대기 후 즉시 종료 여부 확인
            time.sleep(1.0)
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read().decode().strip()
                print(f"[tunnel] Connection failed: {stderr}")
                retries += 1
                if not self._auto_reconnect or retries >= self._max_retries:
                    print("[tunnel] Max retries reached. Exiting.")
                    sys.exit(1)
                print(f"[tunnel] Retrying in {self._retry_delay:.0f}s ...")
                time.sleep(self._retry_delay)
                continue

            print("[tunnel] ✓ Tunnel established")
            print(f"[tunnel]   gRPC: localhost:50051 → RunPod:50051")
            print("[tunnel]   Press Ctrl+C to close\n")
            retries = 0

            # 프로세스가 종료될 때까지 대기
            self._proc.wait()
            stderr_out = self._proc.stderr.read().decode().strip() if self._proc.stderr else ""

            if not self._running:
                break

            print(f"[tunnel] Connection dropped. {stderr_out}")
            if self._auto_reconnect:
                retries += 1
                if retries >= self._max_retries:
                    print("[tunnel] Max retries reached. Exiting.")
                    sys.exit(1)
                print(f"[tunnel] Reconnecting in {self._retry_delay:.0f}s ...")
                time.sleep(self._retry_delay)
            else:
                break

    def stop(self) -> None:
        self._running = False
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        print("\n[tunnel] Tunnel closed.")

    def _handle_signal(self, signum, frame) -> None:
        self.stop()
        sys.exit(0)


# ────────────────────────────────────────────────────────────────────────────
# 데몬 모드 (백그라운드 실행)
# ────────────────────────────────────────────────────────────────────────────

def daemonize(pid_file: str = "/tmp/lerobot_tunnel.pid") -> None:
    """Unix double-fork 데몬화."""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    print(f"[tunnel] Running as daemon (PID: {os.getpid()}, pid file: {pid_file})")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RunPod SSH Tunnel Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 환경변수 사용
  RUNPOD_POD_ID=abc123 RUNPOD_SSH_PORT=22042 python scripts/open_tunnel.py

  # 직접 인자 전달
  python scripts/open_tunnel.py --pod-id abc123 --ssh-port 22042

  # TensorBoard + Jupyter 포트 추가
  python scripts/open_tunnel.py --pod-id abc123 --ssh-port 22042 \\
      --extra-tunnels 6006:6006 8888:8888

  # 자동 재연결 + 데몬
  python scripts/open_tunnel.py --pod-id abc123 --ssh-port 22042 \\
      --auto-reconnect --daemon
        """,
    )

    parser.add_argument("--pod-id",       default=os.environ.get("RUNPOD_POD_ID", ""))
    parser.add_argument("--ssh-port",     type=int,
                        default=int(os.environ.get("RUNPOD_SSH_PORT", "22")))
    parser.add_argument("--local-port",   type=int, default=50051,
                        help="로컬 gRPC 포트 (기본: 50051)")
    parser.add_argument("--remote-port",  type=int, default=50051,
                        help="서버 gRPC 포트 (기본: 50051)")
    parser.add_argument("--key",          default=os.path.expanduser("~/.ssh/id_rsa"),
                        help="SSH 개인키 경로")
    parser.add_argument("--env",          default=None,
                        help=".env 파일 경로 (RUNPOD_POD_ID, RUNPOD_SSH_PORT 로드)")
    parser.add_argument("--extra-tunnels", nargs="*", default=[],
                        metavar="LOCAL:REMOTE",
                        help="추가 포트 포워딩 (예: 6006:6006 8888:8888)")
    parser.add_argument("--auto-reconnect", action="store_true", default=True,
                        help="연결 끊김 시 자동 재연결 (기본: ON)")
    parser.add_argument("--no-reconnect",   action="store_true",
                        help="자동 재연결 비활성화")
    parser.add_argument("--max-retries",  type=int, default=10)
    parser.add_argument("--retry-delay",  type=float, default=5.0,
                        help="재연결 대기 시간(초)")
    parser.add_argument("--daemon",       action="store_true",
                        help="백그라운드 데몬으로 실행")

    args = parser.parse_args()

    # .env 파일 로드
    if args.env:
        env = load_env_file(args.env)
        if not args.pod_id and "RUNPOD_POD_ID" in env:
            args.pod_id = env["RUNPOD_POD_ID"]
        if args.ssh_port == 22 and "RUNPOD_SSH_PORT" in env:
            args.ssh_port = int(env["RUNPOD_SSH_PORT"])

    # 필수값 체크
    if not args.pod_id:
        print("[tunnel] ERROR: --pod-id 또는 RUNPOD_POD_ID 환경변수가 필요합니다.")
        print("[tunnel] RunPod 대시보드 → Pod → Connect → SSH over exposed TCP에서 확인")
        sys.exit(1)

    # 추가 터널 파싱 (LOCAL:REMOTE → (int, int))
    extra_tunnels: list[tuple[int, int]] = []
    for t in args.extra_tunnels:
        try:
            lp, rp = t.split(":")
            extra_tunnels.append((int(lp), int(rp)))
        except ValueError:
            print(f"[tunnel] WARNING: Invalid tunnel format '{t}', skipping")

    # 정보 출력
    host = RUNPOD_SSH_HOST_FORMAT.format(pod_id=args.pod_id)
    print("=" * 60)
    print("  LeRobot RunPod SSH Tunnel")
    print("=" * 60)
    print(f"  Host       : root@{host}")
    print(f"  SSH Port   : {args.ssh_port}")
    print(f"  gRPC Tunnel: localhost:{args.local_port} → server:{args.remote_port}")
    for lp, rp in extra_tunnels:
        print(f"  Extra      : localhost:{lp} → server:{rp}")
    print(f"  Reconnect  : {'ON' if args.auto_reconnect and not args.no_reconnect else 'OFF'}")
    print("=" * 60)

    cmd = build_ssh_command(
        pod_id=args.pod_id,
        ssh_port=args.ssh_port,
        local_port=args.local_port,
        remote_port=args.remote_port,
        key_path=args.key,
        extra_tunnels=extra_tunnels,
    )

    if args.daemon:
        daemonize()

    manager = TunnelManager(
        cmd=cmd,
        auto_reconnect=args.auto_reconnect and not args.no_reconnect,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    manager.start()


if __name__ == "__main__":
    main()
