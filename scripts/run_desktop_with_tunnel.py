#!/usr/bin/env python3
"""
Run desktop client with automatic RunPod SSH tunnel bootstrap.

Usage examples:
  python scripts/run_desktop_with_tunnel.py --pod-id <POD_ID> --ssh-port <PORT> infer --model-id run_001
  python scripts/run_desktop_with_tunnel.py --env .env.runpod collect --n-episodes 5 --task "pick"
"""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ".env.runpod"


def load_env_file(env_path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    path = Path(env_path)
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def resolve_env_file(explicit_env: str | None) -> str | None:
    if explicit_env:
        return explicit_env
    default_env = ROOT / DEFAULT_ENV_FILE
    if default_env.exists():
        return str(default_env)
    return None


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_default_key() -> str:
    for candidate in (Path.home() / ".ssh" / "id_ed25519", Path.home() / ".ssh" / "id_rsa"):
        if candidate.exists():
            return str(candidate)
    return str(Path.home() / ".ssh" / "id_ed25519")


def ensure_tunnel(pod_id: str, ssh_host: str, ssh_port: int, key_path: str, local_port: int, remote_port: int) -> None:
    if is_port_open("127.0.0.1", local_port):
        print(f"[tunnel] localhost:{local_port} already open, reusing existing tunnel")
        return

    host = ssh_host or f"{pod_id}.ssh.runpod.net"
    cmd = [
        "ssh",
        "-f",
        "-N",
        "-T",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        f"{local_port}:localhost:{remote_port}",
        "-i",
        key_path,
        f"root@{host}",
        "-p",
        str(ssh_port),
    ]
    print("[tunnel] starting:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)

    if not is_port_open("127.0.0.1", local_port, timeout=2.0):
        raise RuntimeError(f"tunnel failed: localhost:{local_port} is not reachable")


def run_health_check(local_port: int) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "check_connection.py"),
        "--host",
        "localhost",
        "--port",
        str(local_port),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def run_desktop(config: str, desktop_args: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_desktop.py"), "--config", config, *desktop_args]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tunnel wrapper for desktop client")
    parser.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID", ""))
    parser.add_argument("--ssh-host", default=os.environ.get("RUNPOD_SSH_HOST", ""))
    parser.add_argument("--ssh-port", type=int, default=int(os.environ.get("RUNPOD_SSH_PORT", "22")))
    parser.add_argument("--key", default=os.environ.get("RUNPOD_SSH_KEY", resolve_default_key()))
    parser.add_argument("--env", default=None, help="Optional .env file with RUNPOD_SSH_HOST/POD_ID/SSH_PORT/SSH_KEY")
    parser.add_argument("--local-port", type=int, default=50051)
    parser.add_argument("--remote-port", type=int, default=50051)
    parser.add_argument("--config", default="module/config/desktop.yaml")

    args, desktop_args = parser.parse_known_args()
    args.desktop_args = desktop_args
    return args


def main() -> None:
    args = parse_args()

    env_path = resolve_env_file(args.env)
    if env_path:
        env = load_env_file(env_path)
        if not args.pod_id and "RUNPOD_POD_ID" in env:
            args.pod_id = env["RUNPOD_POD_ID"]
        if not args.ssh_host and "RUNPOD_SSH_HOST" in env:
            args.ssh_host = env["RUNPOD_SSH_HOST"]
        if args.ssh_port == 22 and "RUNPOD_SSH_PORT" in env:
            args.ssh_port = int(env["RUNPOD_SSH_PORT"])
        if "RUNPOD_SSH_KEY" in env and args.key == os.environ.get("RUNPOD_SSH_KEY", resolve_default_key()):
            args.key = env["RUNPOD_SSH_KEY"]

    if not args.pod_id and not args.ssh_host:
        print("[run] ERROR: --pod-id/RUNPOD_POD_ID or --ssh-host/RUNPOD_SSH_HOST is required")
        sys.exit(1)

    ensure_tunnel(
        pod_id=args.pod_id,
        ssh_host=args.ssh_host,
        ssh_port=args.ssh_port,
        key_path=args.key,
        local_port=args.local_port,
        remote_port=args.remote_port,
    )

    run_health_check(local_port=args.local_port)
    rc = run_desktop(config=args.config, desktop_args=args.desktop_args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
