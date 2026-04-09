#!/usr/bin/env python3
"""
scripts/check_connection.py
============================
RunPod 서버 연결 상태 통합 헬스체크 스크립트.

터널을 열기 전/후, 또는 문제 발생 시 진단용으로 사용합니다.

사용법:
    # 기본 (config/desktop.yaml 읽음)
    python scripts/check_connection.py

    # 직접 지정
    python scripts/check_connection.py --host localhost --port 50051

    # 주기적 모니터링 (10초마다)
    python scripts/check_connection.py --watch --interval 10
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ────────────────────────────────────────────────────────────────────────────
# 체크 함수들
# ────────────────────────────────────────────────────────────────────────────

def check_tcp_port(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    """TCP 소켓 레벨 연결 가능 여부 확인."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"TCP {host}:{port} open"
    except socket.timeout:
        return False, f"TCP {host}:{port} timeout ({timeout}s)"
    except ConnectionRefusedError:
        return False, f"TCP {host}:{port} refused (서버가 실행 중이 아닙니다)"
    except socket.gaierror as e:
        return False, f"DNS 해석 실패: {host} — {e}"
    except Exception as e:
        return False, f"TCP 연결 오류: {e}"


def check_ssh_tunnel_active(local_port: int = 50051) -> tuple[bool, str]:
    """로컬 SSH 터널 프로세스가 살아있는지 확인."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", f"ssh.*-L.*{local_port}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            pid_line = result.stdout.strip().split("\n")[0]
            return True, f"SSH 터널 활성 (PID: {pid_line.split()[0]})"
        return False, "SSH 터널 프로세스 없음 (open_tunnel.py를 먼저 실행하세요)"
    except FileNotFoundError:
        # pgrep 없는 환경 (Windows)
        return None, "pgrep 없음 — 수동으로 터널 상태 확인"


def check_grpc_ping(host: str, port: int, timeout: float = 5.0) -> tuple[bool, dict]:
    """gRPC HealthService.Ping 호출."""
    try:
        from module.desktop.grpc_client import InferenceClient
        client = InferenceClient(host=host, port=port, timeout=timeout)
        client.connect()
        info = client.ping()
        client.disconnect()
        return True, info
    except ImportError:
        return None, {"error": "gRPC 스텁 미생성 (gen_proto.sh 실행 필요)"}
    except Exception as e:
        return False, {"error": str(e)}


def check_local_proto_stubs() -> tuple[bool, str]:
    """proto 스텁 파일이 생성되어 있는지 확인."""
    proto_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "module", "proto",
    )
    pb2      = os.path.join(proto_dir, "lerobot_pb2.py")
    pb2_grpc = os.path.join(proto_dir, "lerobot_pb2_grpc.py")

    if os.path.exists(pb2) and os.path.exists(pb2_grpc):
        mtime = datetime.fromtimestamp(os.path.getmtime(pb2)).strftime("%Y-%m-%d %H:%M")
        return True, f"스텁 생성됨 (last modified: {mtime})"
    missing = []
    if not os.path.exists(pb2):      missing.append("lerobot_pb2.py")
    if not os.path.exists(pb2_grpc): missing.append("lerobot_pb2_grpc.py")
    return False, f"스텁 없음: {', '.join(missing)} — bash scripts/gen_proto.sh 실행"


def check_config(config_path: str) -> tuple[bool, dict]:
    """desktop.yaml 읽기 및 gRPC 설정 파싱."""
    try:
        import yaml
        if not os.path.exists(config_path):
            return False, {"error": f"설정 파일 없음: {config_path}"}
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        grpc = cfg.get("grpc", {})
        return True, {
            "host": grpc.get("host", "localhost"),
            "port": grpc.get("port", 50051),
            "use_tls": grpc.get("use_tls", False),
        }
    except Exception as e:
        return False, {"error": str(e)}


# ────────────────────────────────────────────────────────────────────────────
# 결과 출력
# ────────────────────────────────────────────────────────────────────────────

STATUS_OK   = "✓"
STATUS_FAIL = "✗"
STATUS_WARN = "?"


def _status_symbol(ok) -> str:
    if ok is True:  return f"\033[92m{STATUS_OK}\033[0m"   # 초록
    if ok is False: return f"\033[91m{STATUS_FAIL}\033[0m" # 빨강
    return f"\033[93m{STATUS_WARN}\033[0m"                  # 노랑


def run_checks(host: str, port: int, config_path: str) -> bool:
    """전체 체크 실행 및 결과 출력. 모든 체크 통과 시 True 반환."""

    print(f"\n{'='*60}")
    print(f"  LeRobot Connection Health Check")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target: {host}:{port}")
    print(f"{'='*60}")

    all_ok = True

    # ── 1. proto 스텁 확인 ─────────────────────────────────────
    ok, msg = check_local_proto_stubs()
    sym = _status_symbol(ok)
    print(f"\n  {sym} [proto 스텁] {msg}")
    if ok is False:
        all_ok = False
        print(f"     → 해결: bash scripts/gen_proto.sh")

    # ── 2. 설정 파일 ─────────────────────────────────────────
    ok, info = check_config(config_path)
    sym = _status_symbol(ok)
    if ok:
        cfg_host = info["host"]
        cfg_port = info["port"]
        tls      = "TLS" if info["use_tls"] else "평문"
        print(f"  {sym} [설정 파일] {config_path}")
        print(f"       host={cfg_host}  port={cfg_port}  {tls}")
        # 설정과 실제 인자가 다르면 경고
        if cfg_host != host or cfg_port != port:
            print(f"     ⚠ 실행 인자({host}:{port})와 설정({cfg_host}:{cfg_port})이 다릅니다")
    else:
        print(f"  {sym} [설정 파일] {info.get('error', '오류')}")

    # ── 3. SSH 터널 상태 ──────────────────────────────────────
    ok, msg = check_ssh_tunnel_active(port)
    sym = _status_symbol(ok)
    print(f"  {sym} [SSH 터널] {msg}")
    if ok is False and host in ("localhost", "127.0.0.1"):
        all_ok = False
        print(f"     → 해결: python scripts/open_tunnel.py --pod-id <POD_ID> --ssh-port <PORT>")

    # ── 4. TCP 포트 열림 확인 ─────────────────────────────────
    ok, msg = check_tcp_port(host, port)
    sym = _status_symbol(ok)
    print(f"  {sym} [TCP 연결] {msg}")
    if ok is False:
        all_ok = False
        if host in ("localhost", "127.0.0.1"):
            print(f"     → SSH 터널이 닫혔거나 서버가 실행 중이 아닙니다")
        else:
            print(f"     → RunPod에서 포트 {port}가 TCP로 노출되어 있는지 확인하세요")

    # ── 5. gRPC Ping ──────────────────────────────────────────
    print(f"  ", end="", flush=True)
    ok, info = check_grpc_ping(host, port)
    sym = _status_symbol(ok)
    if ok is True:
        gpu  = info.get("gpu_info", "unknown")
        mem  = info.get("gpu_mem_gb", 0)
        sid  = info.get("server_id", "")
        print(f"{sym} [gRPC Ping] 응답 성공")
        print(f"       서버 ID : {sid}")
        print(f"       GPU     : {gpu}")
        print(f"       VRAM    : {mem:.1f} GB")
    elif ok is False:
        all_ok = False
        err = info.get("error", "알 수 없는 오류")
        print(f"{sym} [gRPC Ping] 실패 — {err}")
    else:
        print(f"{sym} [gRPC Ping] {info.get('error', '')}")

    # ── 최종 결과 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_ok:
        print(f"  \033[92m✓ 모든 체크 통과 — 서버와 정상 연결됨\033[0m")
    else:
        print(f"  \033[91m✗ 일부 체크 실패 — 위의 해결 방법을 참고하세요\033[0m")
    print(f"{'='*60}\n")

    return all_ok


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot 연결 헬스체크")
    parser.add_argument("--host",     default=None, help="gRPC 서버 호스트")
    parser.add_argument("--port",     type=int, default=None, help="gRPC 포트")
    parser.add_argument("--config",   default="module/config/desktop.yaml")
    parser.add_argument("--watch",    action="store_true", help="주기적 모니터링")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="모니터링 주기(초), --watch 시 사용")
    args = parser.parse_args()

    # config에서 host/port 읽기 (인자 우선)
    host = args.host
    port = args.port
    if host is None or port is None:
        try:
            import yaml
            if os.path.exists(args.config):
                with open(args.config) as f:
                    cfg = yaml.safe_load(f) or {}
                grpc = cfg.get("grpc", {})
                host = host or grpc.get("host", "localhost")
                port = port or grpc.get("port", 50051)
        except Exception:
            pass
    host = host or "localhost"
    port = port or 50051

    if args.watch:
        print(f"[check] 주기적 모니터링 시작 (간격: {args.interval}s)  Ctrl+C로 종료")
        try:
            while True:
                run_checks(host, port, args.config)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[check] 모니터링 종료")
    else:
        ok = run_checks(host, port, args.config)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
