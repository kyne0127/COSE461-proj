"""
scripts/test_camera.py
=======================
Global-view 카메라 단독 테스트 스크립트 (로봇 팔 연결 불필요).

지원 모드:
  1. ZED Mini (pyzed SDK)  — 글로벌뷰 RGB + 뎁스
  2. USB 카메라 (OpenCV)   — 글로벌뷰 RGB
  3. 여러 인덱스 스캔       — 연결된 카메라 전체 탐색

사용 예시:
  # ZED Mini 테스트 (pyzed 설치 필요)
  python scripts/test_camera.py --mode zed

  # USB 카메라 인덱스 4번 테스트
  python scripts/test_camera.py --mode usb --camera-index 4

  # 사용 가능한 카메라 전체 스캔 (인덱스 0~7)
  python scripts/test_camera.py --mode scan

  # 설정 파일에서 global view 카메라 인덱스 읽어서 테스트
  python scripts/test_camera.py --mode usb --config module/config/desktop.yaml

  # N 프레임 캡처 후 저장
  python scripts/test_camera.py --mode usb --camera-index 4 --n-frames 5 --save-dir /tmp/cam_test

  # AmbRes 파이프라인까지 실제 테스트 (gRPC 서버 연결 필요)
  python scripts/test_camera.py --mode usb --camera-index 4 --task "place the red mug next to the bottle" --run-ambres
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Path setup
# ────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _print_frame_stats(name: str, frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    ch = frame.shape[2] if frame.ndim == 3 else 1
    mn, mx = frame.min(), frame.max()
    mean = frame.mean()
    print(f"  [{name}] shape=({h},{w},{ch})  dtype={frame.dtype}  "
          f"min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}")


def _save_frame(frame_uint8: np.ndarray, save_dir: Path, name: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / name
    bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out), bgr)
    return out


def _load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("[warn] pyyaml not installed — skipping config file")
        return {}
    except FileNotFoundError:
        print(f"[warn] Config not found: {config_path}")
        return {}


# ────────────────────────────────────────────────────────────────────────────
# Mode: SCAN — find all connected cameras
# ────────────────────────────────────────────────────────────────────────────

def run_scan(max_index: int = 8) -> None:
    print(f"\n=== 카메라 스캔 (인덱스 0~{max_index - 1}) ===\n")
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  [FOUND] index={idx}  {w}x{h}  {fps:.0f}fps")
                found.append(idx)
            else:
                print(f"  [NO READ] index={idx}  (열리지만 프레임 없음)")
        else:
            print(f"  [  ---  ] index={idx}  연결 없음")
        cap.release()

    print(f"\n발견된 카메라: {found}")
    if found:
        print(f"  → global view 테스트: python scripts/test_camera.py --mode usb --camera-index {found[0]}")


# ────────────────────────────────────────────────────────────────────────────
# Mode: USB — single USB camera capture test
# ────────────────────────────────────────────────────────────────────────────

def run_usb(
    camera_index: int,
    n_frames: int,
    save_dir: Path | None,
    width: int,
    height: int,
    fps_target: int,
    warmup: int,
) -> np.ndarray | None:
    print(f"\n=== USB 카메라 테스트 (index={camera_index}) ===\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] 카메라 index={camera_index} 열기 실패")
        print("  → 사용 가능한 인덱스 확인: python scripts/test_camera.py --mode scan")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_target)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  실제 해상도: {actual_w}x{actual_h}  FPS: {actual_fps:.0f}")

    # Warmup: 카메라 노출 안정화
    print(f"  워밍업 {warmup} 프레임 스킵...")
    for _ in range(warmup):
        cap.read()

    last_frame_rgb = None
    print(f"  {n_frames} 프레임 캡처:")
    for i in range(n_frames):
        t0 = time.perf_counter()
        ret, frame_bgr = cap.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not ret or frame_bgr is None:
            print(f"    [{i+1}/{n_frames}] FAIL  (프레임 읽기 실패)")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _print_frame_stats(f"{i+1}/{n_frames}  {elapsed_ms:.1f}ms", frame_rgb)

        if save_dir is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = _save_frame(frame_rgb, save_dir, f"usb_cam{camera_index}_{ts}_{i+1:03d}.png")
            print(f"    → 저장: {out}")

        last_frame_rgb = frame_rgb
        time.sleep(max(0, 1.0 / fps_target - elapsed_ms / 1000))

    cap.release()
    print("\n  USB 카메라 테스트 완료")
    return last_frame_rgb


# ────────────────────────────────────────────────────────────────────────────
# Mode: ZED Mini
# ────────────────────────────────────────────────────────────────────────────

def run_zed(
    resolution: str,
    fps_target: int,
    depth_mode: str,
    n_frames: int,
    save_dir: Path | None,
    warmup: int,
    fallback_device_index: int = 0,
) -> np.ndarray | None:
    print(f"\n=== ZED Mini 테스트 (resolution={resolution}) ===\n")

    try:
        from module.desktop.zed_connector import ZEDCapture
    except ImportError as e:
        print(f"[ERROR] zed_connector import 실패: {e}")
        return None

    try:
        zed = ZEDCapture(
            resolution=resolution,
            fps=fps_target,
            depth_mode=depth_mode,
            fallback_device_index=fallback_device_index,
        )
    except Exception as e:
        print(f"[ERROR] ZEDCapture 초기화 실패: {e}")
        return None

    # Warmup
    print(f"  워밍업 {warmup} 프레임 스킵...")
    for _ in range(warmup):
        try:
            zed.capture_synchronized()
        except Exception:
            pass

    last_rgb = None
    print(f"  {n_frames} 프레임 캡처:")
    for i in range(n_frames):
        t0 = time.perf_counter()
        try:
            rgb, depth = zed.capture_synchronized()
        except Exception as e:
            print(f"    [{i+1}/{n_frames}] FAIL: {e}")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        _print_frame_stats(f"RGB  {i+1}/{n_frames}  {elapsed_ms:.1f}ms", rgb)
        _print_frame_stats(f"DEPT {i+1}/{n_frames}", depth)

        if save_dir is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_rgb = _save_frame(rgb, save_dir, f"zed_rgb_{ts}_{i+1:03d}.png")
            # Save depth as 16-bit PNG (millimetres)
            depth_mm = (depth[..., 0] * 1000).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(save_dir / f"zed_depth_{ts}_{i+1:03d}.png"), depth_mm)
            print(f"    → RGB 저장: {out_rgb}")
            print(f"    → Depth 저장 (mm 16-bit)")

        last_rgb = rgb

    zed.release()
    print("\n  ZED Mini 테스트 완료")
    return last_rgb


# ────────────────────────────────────────────────────────────────────────────
# Optional: run AmbRes on captured frame
# ────────────────────────────────────────────────────────────────────────────

def run_ambres_on_frame(rgb_frame: np.ndarray, task: str, config: dict) -> None:
    """Send captured frame to RunPod server via gRPC and run AmbRes query."""
    print(f"\n=== AmbRes gRPC 전송 테스트 ===")
    print(f"  task : {task!r}")
    print(f"  frame: {rgb_frame.shape}  dtype={rgb_frame.dtype}")

    grpc_cfg = config.get("grpc", {})
    host = grpc_cfg.get("host", "localhost")
    port = grpc_cfg.get("port", 50051)
    print(f"  gRPC : {host}:{port}")

    try:
        from module.desktop.generic_client import GenericClient

        with GenericClient(host=host, port=port, timeout=120.0) as client:
            # 1. Load AmbRes handler on server
            print("\n  [1/3] 서버에 AmbRes 핸들러 로드 중...")
            ok = client.load_handler("ambres", config={
                "model_type": "finetune",
                "adapter_ckpt": "nFwD6qtf9T8dkJaQXU9vkW",
                "use_detection": True,
                "save_dir": "/tmp/train_captures",
            })
            if not ok:
                print("  [ERROR] 핸들러 로드 실패")
                return
            print("  ✓ 핸들러 로드 완료")

            # 2. Reset session
            print("\n  [2/3] 세션 리셋...")
            client.infer("ambres", "reset", session_id="cam_test")
            print("  ✓ 리셋 완료")

            # 3. Send frame + query
            print("\n  [3/3] 이미지 전송 및 query 실행...")
            t0 = time.perf_counter()
            result = client.infer(
                handler_id="ambres",
                method="query",
                payload={"task_description": task},
                images={"image": rgb_frame},
                session_id="cam_test",
            )
            elapsed = time.perf_counter() - t0
            print(f"  ✓ 응답 수신 ({elapsed:.2f}s)")

            print("\n  AmbRes query 결과:")
            print(json.dumps(result, indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"  [ERROR] gRPC 전송 실패: {e}")
        import traceback
        traceback.print_exc()


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global-view 카메라 단독 테스트 (로봇 팔 불필요)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["usb", "zed", "scan"], default="scan",
        help="테스트 모드: usb | zed | scan (기본: scan)",
    )
    # USB 옵션
    parser.add_argument("--camera-index", type=int, default=0,
                        help="USB 카메라 인덱스 (기본: 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    # ZED 옵션
    parser.add_argument("--resolution", choices=["VGA", "HD720"], default="VGA",
                        help="ZED 해상도 (기본: VGA = 640x360)")
    parser.add_argument("--depth-mode", choices=["PERFORMANCE", "QUALITY", "ULTRA"],
                        default="PERFORMANCE")
    # 공통 옵션
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n-frames", type=int, default=3,
                        help="캡처할 프레임 수 (기본: 3)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="캡처 전 스킵할 워밍업 프레임 수 (기본: 5)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="캡처 이미지 저장 디렉터리 (미지정 시 저장 안 함)")
    parser.add_argument("--scan-max", type=int, default=8,
                        help="scan 모드에서 탐색할 최대 인덱스 (기본: 8)")
    # AmbRes 연동
    parser.add_argument("--run-ambres", action="store_true",
                        help="캡처 이미지로 AmbRes G₀ 추출 테스트 (gRPC 서버 필요)")
    parser.add_argument("--task", type=str, default="pick up the object",
                        help="AmbRes에 전달할 task description")
    # Config
    parser.add_argument("--config", type=str,
                        default="module/config/desktop.yaml",
                        help="desktop.yaml 경로")

    args = parser.parse_args()

    # Load config (optional)
    config = _load_config(args.config)

    # desktop.yaml에서 USB camera index 자동 검출
    usb_cfg = config.get("sensors", {}).get("usb_camera", {})
    if args.mode == "usb" and args.camera_index == 0 and usb_cfg.get("device_index") is not None:
        auto_idx = usb_cfg["device_index"]
        print(f"[config] desktop.yaml에서 camera_index={auto_idx} 자동 설정")
        args.camera_index = auto_idx

    # desktop.yaml에서 ZED 설정 자동 적용 (CLI에서 명시하지 않은 경우)
    zed_cfg = config.get("sensors", {}).get("zed_mini", {})
    if args.mode == "zed" and zed_cfg:
        # argparse default와 비교해 명시적으로 지정되지 않은 항목만 덮어씀
        if args.resolution == "VGA" and zed_cfg.get("resolution"):
            args.resolution = zed_cfg["resolution"]
            print(f"[config] desktop.yaml에서 resolution={args.resolution} 자동 설정")
        if args.fps == 30 and zed_cfg.get("fps"):
            args.fps = zed_cfg["fps"]
            print(f"[config] desktop.yaml에서 fps={args.fps} 자동 설정")
        if args.depth_mode == "PERFORMANCE" and zed_cfg.get("depth_mode"):
            args.depth_mode = zed_cfg["depth_mode"]
            print(f"[config] desktop.yaml에서 depth_mode={args.depth_mode} 자동 설정")

    save_dir = Path(args.save_dir) if args.save_dir else None

    # ── 실행 ──────────────────────────────────────────────────────────
    last_frame = None

    if args.mode == "scan":
        run_scan(max_index=args.scan_max)

    elif args.mode == "usb":
        last_frame = run_usb(
            camera_index=args.camera_index,
            n_frames=args.n_frames,
            save_dir=save_dir,
            width=args.width,
            height=args.height,
            fps_target=args.fps,
            warmup=args.warmup,
        )

    elif args.mode == "zed":
        last_frame = run_zed(
            resolution=args.resolution,
            fps_target=args.fps,
            depth_mode=args.depth_mode,
            n_frames=args.n_frames,
            save_dir=save_dir,
            warmup=args.warmup,
            fallback_device_index=args.camera_index,
        )

    # ── AmbRes 연동 테스트 ────────────────────────────────────────────
    if args.run_ambres:
        if last_frame is None:
            print("\n[ERROR] --run-ambres는 프레임 캡처 성공 후에만 실행됩니다.")
        else:
            run_ambres_on_frame(last_frame, args.task, config)


if __name__ == "__main__":
    main()
