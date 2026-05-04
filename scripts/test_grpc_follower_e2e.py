#!/usr/bin/env python3
"""
End-to-end gRPC -> model inference -> follower arm motion smoke test.

What this validates:
1) Desktop can call server gRPC inference endpoint.
2) Server can load and run a model on GPU.
3) Returned action is consumed locally to command follower arms.

Safety notes:
- Uses small action deltas by default.
- Runs at a low control rate and short duration.
- Sends followers back to initial pose at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path: str) -> Dict:
    import yaml

    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _arm_indices(state_keys: list[str], arm_name: str) -> list[int]:
    prefix = f"{arm_name}:"
    return [i for i, k in enumerate(state_keys) if k.startswith(prefix)]


def _pick_act_images(images: Dict[str, np.ndarray], arm_name: str) -> Dict[str, np.ndarray]:
    # Prefer arm-local cameras like right/<cam>, then fall back to all images.
    arm_prefix = f"{arm_name}/"
    arm_imgs = [(k, v) for k, v in images.items() if k.startswith(arm_prefix)]
    src = arm_imgs if arm_imgs else list(images.items())
    if not src:
        return {}
    if len(src) == 1:
        return {"context": src[0][1], "wrist": src[0][1]}
    return {"context": src[0][1], "wrist": src[1][1]}


def _downsample_images(images: Dict[str, np.ndarray], stride: int) -> Dict[str, np.ndarray]:
    if stride <= 1:
        return images
    out: Dict[str, np.ndarray] = {}
    for k, v in images.items():
        out[k] = v[::stride, ::stride].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E test: gRPC model output drives follower arms")
    parser.add_argument("--config", default="module/config/desktop.yaml")
    parser.add_argument("--model-type", default="custom", choices=["custom", "act", "camera_probe"])
    parser.add_argument("--model-id", default="grpc_smoke_custom")
    parser.add_argument("--checkpoint-path", default="/tmp/grpc_smoke_custom")
    parser.add_argument("--act-arm", default="right", choices=["right", "left"],
                        help="Which arm to drive when model output dim < full state dim")
    parser.add_argument("--config-yaml", default="",
                        help="Optional config_yaml string passed to LoadModel")
    parser.add_argument("--steps", type=int, default=20, help="Number of control steps")
    parser.add_argument("--hz", type=float, default=5.0, help="Control frequency")
    parser.add_argument("--delta-scale", type=float, default=0.4,
                        help="Scale for model output delta added to current state")
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print loop stats every N steps")
    parser.add_argument("--task", default="grpc smoke test")
    parser.add_argument("--camera-ablation-check", action="store_true",
                        help="Compare action(real images) vs action(zero images) before control loop")
    parser.add_argument("--rpc-timeout", type=float, default=None,
                        help="Override gRPC timeout seconds for inference calls")
    parser.add_argument("--image-downsample-stride", type=int, default=4,
                        help="Downsample stride for camera_probe image RPC payload")
    parser.add_argument("--act-image-downsample-stride", type=int, default=1,
                        help="Downsample stride for ACT image RPC payload")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip calibration prompt and use existing calibration files")
    args = parser.parse_args()

    from module.desktop.grpc_client import InferenceClient
    from module.desktop.robot_connector import RobotConnector

    cfg = load_config(args.config)
    robot_cfg = cfg.get("robot", {})
    grpc_cfg = cfg.get("grpc", {})

    host = grpc_cfg.get("host", "localhost")
    port = int(grpc_cfg.get("port", 50051))
    timeout = float(args.rpc_timeout if args.rpc_timeout is not None else grpc_cfg.get("timeout_secs", 5.0))

    print("=" * 70)
    print("E2E TEST: gRPC model serving -> follower motion")
    print(f"target gRPC: {host}:{port}")
    print(f"steps={args.steps}, hz={args.hz}, delta_scale={args.delta_scale}")
    print("=" * 70)

    connector = RobotConnector.from_config(robot_cfg)

    with InferenceClient(host=host, port=port, timeout=timeout) as client:
        pong = client.ping()
        print(
            f"[1/5] Ping OK: server_id={pong['server_id']} gpu={pong['gpu_info']} "
            f"vram={pong['gpu_mem_gb']:.1f}GB"
        )

        with connector.session(calibrate=not args.no_calibrate):
            obs0 = connector.get_observation()
            action_dim = int(obs0.state.shape[0])
            print(f"[2/5] Robot connected: action_dim/state_dim={action_dim}")

            config_yaml = args.config_yaml
            if args.model_type == "custom" and not config_yaml:
                config_yaml = (
                    "device: cuda\n"
                    f"action_dim: {action_dim}\n"
                    f"state_dim: {action_dim}\n"
                    "hidden_dim: 128\n"
                )
            elif args.model_type == "camera_probe" and not config_yaml:
                config_yaml = (
                    "device: cpu\n"
                    f"action_dim: {action_dim}\n"
                    "gain: 0.4\n"
                )

            ok = client.load_model(
                model_type=args.model_type,
                model_id=args.model_id,
                checkpoint_path=args.checkpoint_path,
                config_yaml=config_yaml,
            )
            if not ok:
                raise RuntimeError(f"LoadModel failed for model_type={args.model_type}")
            print(f"[3/5] Model loaded: model_id={args.model_id} (type={args.model_type})")

            t_step = 1.0 / max(args.hz, 1e-6)
            initial_state = obs0.state.copy()
            state_keys = obs0.state_keys
            act_idxs = _arm_indices(state_keys, args.act_arm)
            if args.model_type == "act":
                print(f"      ACT arm mapping: {args.act_arm} indices={act_idxs}")

            mean_pred_norm = 0.0
            mean_cmd_norm = 0.0
            mean_loop_dt = 0.0

            if args.camera_ablation_check:
                obs_chk = connector.get_observation()
                infer_images_chk = obs_chk.images
                infer_state_chk = obs_chk.state
                if args.model_type == "act":
                    infer_images_chk = _pick_act_images(obs_chk.images, args.act_arm)
                    infer_images_chk = _downsample_images(infer_images_chk, args.act_image_downsample_stride)
                    infer_state_chk = obs_chk.state[act_idxs] if act_idxs else obs_chk.state[:6]
                elif args.model_type == "camera_probe":
                    infer_images_chk = _downsample_images(obs_chk.images, args.image_downsample_stride)

                pred_real = client.get_action(
                    model_id=args.model_id,
                    images=infer_images_chk,
                    state=infer_state_chk,
                    task_text=args.task,
                    episode_id=0,
                    step=0,
                ).reshape(-1)

                zero_images = {k: np.zeros_like(v) for k, v in infer_images_chk.items()}
                pred_zero = client.get_action(
                    model_id=args.model_id,
                    images=zero_images,
                    state=infer_state_chk,
                    task_text=args.task,
                    episode_id=0,
                    step=1,
                ).reshape(-1)

                cam_effect = float(np.linalg.norm(pred_real - pred_zero))
                print(f"[3.5/5] Camera ablation delta ||a(real)-a(zero)|| = {cam_effect:.6f}")

            print("[4/5] Running control loop...")
            for step in range(args.steps):
                t0 = time.perf_counter()
                obs = connector.get_observation()

                infer_images = obs.images
                infer_state = obs.state
                if args.model_type == "act":
                    infer_images = _pick_act_images(obs.images, args.act_arm)
                    infer_images = _downsample_images(infer_images, args.act_image_downsample_stride)
                    if act_idxs:
                        infer_state = obs.state[act_idxs]
                    else:
                        infer_state = obs.state[:6]
                elif args.model_type == "camera_probe":
                    infer_images = _downsample_images(obs.images, args.image_downsample_stride)

                pred = client.get_action(
                    model_id=args.model_id,
                    images=infer_images,
                    state=infer_state,
                    task_text=args.task,
                    episode_id=0,
                    step=step,
                ).reshape(-1)

                # Use server output as small delta around current state for safe motion.
                cmd = obs.state.copy()
                if pred.shape[0] == obs.state.shape[0]:
                    cmd = obs.state + (args.delta_scale * pred)
                elif act_idxs and pred.shape[0] <= len(act_idxs):
                    local = cmd[act_idxs[: pred.shape[0]]]
                    cmd[act_idxs[: pred.shape[0]]] = local + (args.delta_scale * pred)
                else:
                    n = min(pred.shape[0], cmd.shape[0])
                    cmd[:n] = cmd[:n] + (args.delta_scale * pred[:n])

                connector.send_action(cmd.astype(np.float32), state_keys=state_keys)

                pred_norm = float(np.linalg.norm(pred))
                cmd_delta_norm = float(np.linalg.norm(cmd - obs.state))
                mean_pred_norm += pred_norm
                mean_cmd_norm += cmd_delta_norm

                if args.log_every <= 1 or (step % args.log_every == 0) or (step == args.steps - 1):
                    print(
                        f"  step={step:03d} pred_norm={pred_norm:.3f} "
                        f"cmd_delta_norm={cmd_delta_norm:.3f}"
                    )

                dt = time.perf_counter() - t0
                mean_loop_dt += dt
                if t_step - dt > 0:
                    time.sleep(t_step - dt)

            # Return to initial state to leave robot in a predictable pose.
            connector.send_action(initial_state.astype(np.float32), state_keys=state_keys)
            print("[5/5] Returned follower to initial state")

            client.unload_model(args.model_id)

            mean_pred_norm /= max(args.steps, 1)
            mean_cmd_norm /= max(args.steps, 1)
            mean_loop_dt /= max(args.steps, 1)
            achieved_hz = (1.0 / mean_loop_dt) if mean_loop_dt > 0 else float("inf")
            print("-" * 70)
            print("E2E RESULT: PASS")
            print(f"avg_pred_norm={mean_pred_norm:.3f}")
            print(f"avg_cmd_delta_norm={mean_cmd_norm:.3f}")
            print(f"avg_loop_dt={mean_loop_dt:.4f}s achieved_hz={achieved_hz:.2f}")
            print("Model action from gRPC was applied to follower arm command path.")
            print("-" * 70)


if __name__ == "__main__":
    main()
