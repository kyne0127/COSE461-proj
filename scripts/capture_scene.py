#!/usr/bin/env python3
"""
단계별 씬 촬영 스크립트 (데스크탑에서 실행)

- ZED 또는 USB 카메라로 씬을 1장씩 수동 촬영
- AmbRes 추론 없이 이미지만 RunPod에 저장 (빠름)
- 데이터 수집 전용 (레이블링은 RunPod에서 별도 진행)

사용법:
  python scripts/capture_scene.py --mode zed
  python scripts/capture_scene.py --mode usb --camera-index 4
  python scripts/capture_scene.py --mode zed --host <runpod-ip> --port 50051
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def capture_frame(source: str, camera_index: int, resolution: str, warmup: int = 5):
    """카메라에서 프레임 1장 캡처 → uint8 RGB numpy 배열 반환."""
    import cv2
    import numpy as np

    if source == "zed":
        from module.desktop.zed_connector import ZEDCapture
        cam = ZEDCapture(resolution=resolution, fps=30)
        for _ in range(warmup):
            cam.capture_synchronized()
        rgb, _ = cam.capture_synchronized()
        cam.release()
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"USB 카메라 index={camera_index} 열기 실패")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(warmup):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("프레임 캡처 실패")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return rgb


def send_to_server(rgb, host: str, port: int, tag: str) -> bool:
    """이미지를 gRPC로 서버에 전송 (save_image 메서드 — 추론 없음)."""
    import numpy as np
    from module.desktop.generic_client import GenericClient

    with GenericClient(host=host, port=port, timeout=30.0) as client:
        result = client.infer(
            handler_id="ambres",
            method="save_image",
            payload={"tag": tag},
            images={"image": rgb},
            session_id="capture",
        )
    return result.get("saved", False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="단계별 씬 촬영 — RunPod에 이미지 저장 (추론 없음)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["zed", "usb"], default="zed")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--resolution", choices=["VGA", "HD720"], default="VGA",
                        help="ZED 해상도")
    parser.add_argument("--host", default="localhost", help="RunPod gRPC 호스트")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--warmup", type=int, default=5, help="워밍업 프레임 수")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  단계별 씬 촬영")
    print(f"  카메라 : {args.mode.upper()}")
    print(f"  서버   : {args.host}:{args.port}")
    print(f"  저장   : RunPod /tmp/train_captures/")
    print(f"{'='*55}")
    print("  Enter = 촬영  |  r = 재촬영  |  q = 종료\n")

    scene_num = 0
    try:
        while True:
            input(f"  [씬 #{scene_num + 1}] 씬을 구성한 뒤 Enter 입력 ... ")

            while True:
                print("  캡처 중...", end=" ", flush=True)
                try:
                    rgb = capture_frame(args.mode, args.camera_index,
                                        args.resolution, args.warmup)
                    h, w = rgb.shape[:2]
                    print(f"완료 ({w}×{h})")
                except Exception as e:
                    print(f"실패: {e}")
                    retry = input("  재시도? (Enter = 재시도, q = 종료): ").strip().lower()
                    if retry == "q":
                        return
                    continue

                # 서버 전송
                print("  서버 전송 중...", end=" ", flush=True)
                tag = f"scene{scene_num + 1:03d}"
                try:
                    ok = send_to_server(rgb, args.host, args.port, tag)
                    if ok:
                        print(f"저장 완료 → /tmp/train_captures/{tag}.png")
                    else:
                        print("저장 실패 (서버 응답 없음)")
                except Exception as e:
                    print(f"전송 실패: {e}")
                    retry = input("  재촬영? (r = 재촬영, Enter = 건너뜀): ").strip().lower()
                    if retry == "r":
                        continue
                    break

                ans = input("  이 이미지로 확정? (Enter = 확정, r = 재촬영): ").strip().lower()
                if ans != "r":
                    scene_num += 1
                    print(f"  ✓ 씬 #{scene_num} 저장 완료  (누적: {scene_num}장)\n")
                    break
                print("  재촬영합니다...")

    except (KeyboardInterrupt, EOFError):
        pass

    print(f"\n  총 {scene_num}장 촬영 완료.")
    print(f"  다음 단계: RunPod에서 레이블링")
    print(f"    python scripts/collect_and_label.py --mode train --source skip")


if __name__ == "__main__":
    main()
