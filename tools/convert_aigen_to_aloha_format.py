#!/usr/bin/env python3
"""
Convert Aigen GPS driving HDF5 episodes to ALOHA-compatible format for Cosmos Policy.

Takes our format:
    images:  (T, 224, 224, 3) uint8
    actions: (T, 2) float32  [linear_vel, angular_vel]
    proprio: (T, 5) float32  [x, y, theta, v, omega]
    @command: str

Produces ALOHA-style preprocessed format:
    observations/qpos:  (T, 5) float32  -- native proprio (no zero-padding)
    observations/qvel:  (T, 5) float32  -- zeros (we don't have this)
    observations/effort: (T, 5) float32 -- zeros (we don't have this)
    observations/video_paths/{cam_high, cam_left_wrist, cam_right_wrist}: str  -- MP4 filename
    action:          (T, 2) float32  -- native actions (no zero-padding)
    relative_action: (T, 2) float32  -- delta actions
    @sim: False
    @command: str

The Cosmos Policy model is dimension-agnostic — it flattens actions/proprio
into latent tensors and tile-repeats to fill the space. No need to pad to
ALOHA's 14-DOF.

We duplicate cam_high for wrist cameras since we only have one camera.
The ALOHA loader expects 3 cameras; using the same image ensures structural
compatibility while being honest about our single-camera setup.

Output structure:
    out_dir/
        train/
            episode_0.hdf5
            episode_0_cam_high.mp4
            episode_0_cam_left_wrist.mp4
            episode_0_cam_right_wrist.mp4
            ...
        val/
            ...

Usage:
    python tools/convert_aigen_to_aloha_format.py \
        --input_dir /data/cosmos_policy_data/train \
        --output_dir /data/cosmos_policy_data_aloha \
        --percent_val 0.05
"""

import argparse
import glob
import os
import random

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def create_video_from_images(images: np.ndarray, video_path: str, fps: int = 25):
    """Create MP4 video from (T, H, W, 3) uint8 RGB array."""
    if len(images) == 0:
        raise ValueError("No images provided")

    h, w = images.shape[1], images.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for img in images:
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    out.release()


def compute_relative_actions(actions: np.ndarray) -> np.ndarray:
    """Compute delta actions (same as ALOHA preprocessing)."""
    relative = np.zeros_like(actions)
    relative[:-1] = actions[1:] - actions[:-1]
    if len(relative) > 1:
        relative[-1] = relative[-2]
    return relative


def convert_episode(
    input_path: str,
    output_dir: str,
    episode_idx: int,
    video_fps: int = 25,
):
    """Convert a single Aigen episode to ALOHA-compatible format."""
    with h5py.File(input_path, "r") as f:
        images = f["images"][:]        # (T, H, W, 3) uint8
        actions = f["actions"][:]      # (T, 2) float32
        proprio = f["proprio"][:]      # (T, 5) float32

        command = "follow the crop rows"  # default
        if "command" in f.attrs:
            command = f.attrs["command"]
            if isinstance(command, bytes):
                command = command.decode("utf-8")
        elif "command" in f:
            val = f["command"][()]
            command = val.decode("utf-8") if isinstance(val, bytes) else str(val)

    T = len(images)
    action_dim = actions.shape[1]
    proprio_dim = proprio.shape[1]

    # Compute relative actions (native dims, no padding)
    relative_actions = compute_relative_actions(actions)

    # qvel and effort are zeros in our case (same shape as proprio)
    qvel = np.zeros_like(proprio)
    effort = np.zeros_like(proprio)

    # Create video files (same image for all 3 "cameras")
    cam_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    video_filenames = {}
    for cam in cam_names:
        fname = f"episode_{episode_idx}_{cam}.mp4"
        vpath = os.path.join(output_dir, fname)
        create_video_from_images(images, vpath, fps=video_fps)
        video_filenames[cam] = fname

    # Write ALOHA-format HDF5
    out_path = os.path.join(output_dir, f"episode_{episode_idx}.hdf5")
    with h5py.File(out_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["command"] = command
        root.attrs["action_dim"] = action_dim
        root.attrs["proprio_dim"] = proprio_dim

        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=proprio)
        obs.create_dataset("qvel", data=qvel)
        obs.create_dataset("effort", data=effort)

        video_paths_group = obs.create_group("video_paths")
        for cam, fname in video_filenames.items():
            video_paths_group.create_dataset(cam, data=fname.encode("utf-8"))

        root.create_dataset("action", data=actions)
        root.create_dataset("relative_action", data=relative_actions)

    return {
        "input_path": input_path,
        "output_path": out_path,
        "episode_idx": episode_idx,
        "num_steps": T,
        "command": command,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Aigen GPS HDF5 to ALOHA format")
    parser.add_argument("--input_dir", required=True, help="Dir with Aigen .hdf5/.h5 files")
    parser.add_argument("--output_dir", required=True, help="Output dir for ALOHA-format data")
    parser.add_argument("--percent_val", type=float, default=0.05, help="Fraction for val split")
    parser.add_argument("--video_fps", type=int, default=25, help="FPS for output MP4s")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    # Find all input episodes
    files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.hdf5"))
        + glob.glob(os.path.join(args.input_dir, "*.h5"))
    )
    if not files:
        print(f"No HDF5 files found in {args.input_dir}")
        return

    print(f"Found {len(files)} episodes")

    # Train/val split
    random.seed(args.seed)
    indices = list(range(len(files)))
    random.shuffle(indices)
    n_val = max(1, int(len(files) * args.percent_val))
    val_indices = set(indices[:n_val])

    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_count = 0
    val_count = 0
    total_steps = 0

    for i, fpath in enumerate(tqdm(files, desc="Converting")):
        if i in val_indices:
            out_dir = val_dir
            ep_idx = val_count
            val_count += 1
        else:
            out_dir = train_dir
            ep_idx = train_count
            train_count += 1

        meta = convert_episode(fpath, out_dir, ep_idx, video_fps=args.video_fps)
        total_steps += meta["num_steps"]

    print(f"\nDone! {train_count} train, {val_count} val episodes ({total_steps} total steps)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
