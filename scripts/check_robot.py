#!/usr/bin/env python3
"""
scripts/check_robot.py
======================
로봇 파이프라인 사전 점검 스크립트.

검사 항목:
  1. USB 포트  — config에 명시된 ttyACM 장치 존재 여부
  2. 카메라    — config에 명시된 video 인덱스 개방 가능 여부 + 해상도 확인
  3. gRPC      — 서버 연결 가능 여부 (선택; 터널 없으면 SKIP)
  4. 로봇 연결 — SOFollower connect() + get_observation() 1회 실행
  5. 관절 상태 — 관절 각도 출력 (비정상 범위 경고)

Usage:
    python scripts/check_robot.py
    python scripts/check_robot.py --config module/config/desktop.yaml
    python scripts/check_robot.py --skip-robot   # 포트/카메라/gRPC만 확인
    python scripts/check_robot.py --skip-grpc    # 서버 확인 생략
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "✓"
FAIL = "✗"
WARN = "!"
SKIP = "-"

def clr(text, code):
    """ANSI 색상 (코드: 32=green, 31=red, 33=yellow, 36=cyan)."""
    return f"\033[{code}m{text}\033[0m"

def ok(msg):   print(f"  {clr(PASS, 32)} {msg}")
def fail(msg): print(f"  {clr(FAIL, 31)} {msg}")
def warn(msg): print(f"  {clr(WARN, 33)} {msg}")
def skip(msg): print(f"  {clr(SKIP, 36)} {msg}")
def header(msg): print(f"\n{clr('═' * 52, 36)}\n{clr(msg, 1)}\n{clr('═' * 52, 36)}")


# ────────────────────────────────────────────────────────────────────────────
# 1. USB 포트 확인
# ────────────────────────────────────────────────────────────────────────────

def check_ports(robot_cfg: dict) -> bool:
    header("1. USB 포트 확인")
    rc = robot_cfg.get("robot_config", {})
    all_ok = True

    port_map: dict[str, str] = {}
    for arm_name, arm_cfg in rc.get("follower_arms", {}).items():
        port_map[f"follower/{arm_name}"] = arm_cfg.get("port", "")
    for arm_name, arm_cfg in rc.get("leader_arms", {}).items():
        port_map[f"leader/{arm_name}"]   = arm_cfg.get("port", "")
    if "port" in rc:
        port_map["follower/main"] = rc["port"]
    if "leader_port" in rc:
        port_map["leader/main"]   = rc["leader_port"]

    if not port_map:
        warn("robot_config에 포트 설정 없음")
        return False

    for role, port in port_map.items():
        if os.path.exists(port):
            ok(f"{role:25s}  {port}")
        else:
            fail(f"{role:25s}  {port}  ← 장치 없음")
            all_ok = False

    # 실제 연결된 ACM 장치 목록
    acm_list = sorted(p for p in os.listdir("/dev") if p.startswith("ttyACM"))
    print(f"\n  /dev 에서 발견된 ttyACM 장치: {['/dev/' + p for p in acm_list] or '없음'}")
    return all_ok


# ────────────────────────────────────────────────────────────────────────────
# 2. 카메라 확인
# ────────────────────────────────────────────────────────────────────────────

def check_cameras(robot_cfg: dict) -> bool:
    header("2. 카메라 확인")
    try:
        import cv2
    except ImportError:
        warn("opencv-python 미설치 — 카메라 확인 생략")
        return True

    cameras = robot_cfg.get("robot_config", {}).get("cameras", {})
    if not cameras:
        skip("카메라 설정 없음")
        return True

    all_ok = True
    for cam_name, cam_cfg in cameras.items():
        idx   = cam_cfg.get("camera_index", 0)
        w_cfg = cam_cfg.get("width",  640)
        h_cfg = cam_cfg.get("height", 480)
        fps_cfg = cam_cfg.get("fps", 30)

        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            fail(f"{cam_name:20s}  index={idx}  ← 개방 실패")
            all_ok = False
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w_cfg)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_cfg)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            fail(f"{cam_name:20s}  index={idx}  ← 프레임 읽기 실패")
            all_ok = False
        else:
            h, w, c = frame.shape
            match = "✓" if (w == w_cfg and h == h_cfg) else "≠"
            msg = (f"{cam_name:20s}  index={idx}"
                   f"  실제={w}×{h}  설정={w_cfg}×{h_cfg} {match}"
                   f"  dtype={frame.dtype}")
            if w == w_cfg and h == h_cfg:
                ok(msg)
            else:
                warn(msg)

    video_list = sorted(p for p in os.listdir("/dev") if p.startswith("video"))
    print(f"\n  /dev 에서 발견된 video 장치: {['/dev/' + p for p in video_list] or '없음'}")
    return all_ok


# ────────────────────────────────────────────────────────────────────────────
# 3. gRPC 서버 연결
# ────────────────────────────────────────────────────────────────────────────

def check_grpc(grpc_cfg: dict) -> bool:
    header("3. gRPC 서버 연결")
    host = grpc_cfg.get("host", "localhost")
    port = grpc_cfg.get("port", 50051)
    print(f"  연결 대상: {host}:{port}")

    try:
        from module.desktop.grpc_client import InferenceClient
        with InferenceClient(host=host, port=port) as client:
            pong = client.ping()
        ok(f"ping 성공  GPU={pong.get('gpu_info','?')}  "
           f"메모리={pong.get('gpu_mem_gb', 0):.1f}GB")
        return True
    except Exception as e:
        warn(f"서버 응답 없음: {e}")
        warn("SSH 터널이 열려 있지 않으면 정상입니다 (inference에서만 필요)")
        return False


# ────────────────────────────────────────────────────────────────────────────
# 4 & 5. 로봇 연결 + 관절 상태
# ────────────────────────────────────────────────────────────────────────────

def check_robot(robot_cfg: dict) -> bool:
    header("4. 로봇 연결 (SOFollower connect)")
    from module.desktop.robot_connector import RobotConnector

    connector = RobotConnector.from_config(robot_cfg)
    try:
        connector.connect(calibrate=False)
    except Exception as e:
        fail(f"connect() 실패: {e}")
        return False

    ok(f"연결 성공  followers={connector._follower_order}  "
       f"leaders={connector._leader_order}")

    # ── 관절 상태 ──────────────────────────────────────────────────────────
    header("5. Observation 읽기")
    try:
        obs = connector.get_observation()
    except Exception as e:
        fail(f"get_observation() 실패: {e}")
        connector.disconnect()
        return False

    print(f"\n  cameras  : {list(obs.images.keys())}")
    for cam_name, img in obs.images.items():
        ok(f"  {cam_name:20s}  shape={img.shape}  dtype={img.dtype}")

    print(f"\n  state    : shape={obs.state.shape}  dtype={obs.state.dtype}")
    print(f"  state_keys: {obs.state_keys}")

    # 관절 각도 출력
    print()
    for key, val in zip(obs.state_keys, obs.state):
        deg = float(val)
        line = f"  {key:35s}  {deg:8.2f}°"
        if abs(deg) > 170:
            warn(line + "  ← 한계 근접")
        else:
            print(line)

    connector.disconnect()
    ok("\n  disconnect 완료")
    return True


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    import yaml
    if not os.path.exists(path):
        print(f"[warn] config not found at {path}, using defaults")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="로봇 파이프라인 사전 점검")
    parser.add_argument("--config",      default="module/config/desktop.yaml")
    parser.add_argument("--skip-robot",  action="store_true", help="로봇 연결 단계 생략")
    parser.add_argument("--skip-grpc",   action="store_true", help="gRPC 확인 생략")
    args = parser.parse_args()

    config    = load_config(args.config)
    robot_cfg = config.get("robot", {})
    grpc_cfg  = config.get("grpc",  {})

    results: dict[str, bool | None] = {}

    results["ports"]   = check_ports(robot_cfg)
    results["cameras"] = check_cameras(robot_cfg)

    if args.skip_grpc:
        skip("gRPC 확인 생략 (--skip-grpc)")
        results["grpc"] = None
    else:
        results["grpc"] = check_grpc(grpc_cfg)

    if args.skip_robot:
        skip("로봇 연결 생략 (--skip-robot)")
        results["robot"] = None
    else:
        results["robot"] = check_robot(robot_cfg)

    # ── 최종 요약 ──────────────────────────────────────────────────────────
    header("요약")
    labels = {
        "ports":   "USB 포트",
        "cameras": "카메라",
        "grpc":    "gRPC 서버",
        "robot":   "로봇 연결",
    }
    any_fail = False
    for key, label in labels.items():
        r = results.get(key)
        if r is None:
            skip(f"{label:12s}  (생략)")
        elif r:
            ok(f"{label:12s}")
        else:
            fail(f"{label:12s}")
            if key != "grpc":   # gRPC 실패는 치명적이지 않음
                any_fail = True

    print()
    if any_fail:
        print(clr("  파이프라인 실행 전 위 항목을 확인하세요.", 31))
        sys.exit(1)
    else:
        print(clr("  모든 필수 항목 통과 — 파이프라인 실행 가능합니다.", 32))


if __name__ == "__main__":
    main()
