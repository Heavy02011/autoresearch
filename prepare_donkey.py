"""
Fixed preparation and evaluation utilities for DonkeyCar AutoResearch experiments.

Usage:
    python prepare_donkey.py --generate                # generate tub with 20k steps on port 9091
    python prepare_donkey.py --generate --num-steps 40000 --port 9092

Data is stored in ~/donkeycar/data/sim_tub/.

This file is READ-ONLY for the agent — do not modify.
"""

import os
import sys
import json
import glob
import time
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 120, 160           # camera resolution: height=120, width=160 (DonkeyCar default)
IMG_CHANNELS = 3                  # RGB
TIME_BUDGET = 300                 # 5-minute wall-clock training budget (seconds)
VAL_FRACTION = 0.15               # fraction of tubs reserved for validation
STEERING_SCALE = 1.0              # steering already in [-1, 1]
THROTTLE_SCALE = 1.0              # throttle already in [0, 1]

# Paths
CACHE_DIR = os.path.join(os.path.expanduser("~"), "donkeycar", "data")
TUB_DIR = os.path.join(CACHE_DIR, "sim_tub")
BEST_MODEL_PATH = "best_model.pth"

# Simulator evaluation settings
SIM_PORT = 9091                   # port where sdsandbox is listening
SIM_ENV_ID = "donkey-generated-track-v0"
SIM_EVAL_STEPS = 500              # number of sim steps per evaluation episode
SIM_THROTTLE = 0.5                # fixed throttle during evaluation

# ---------------------------------------------------------------------------
# Tub data utilities
# ---------------------------------------------------------------------------

def _parse_tub(tub_path):
    """
    Parse a DonkeyCar tub directory and return a list of
    (image_path, steering, throttle) tuples.

    Supports both catalogue-based tubs (manifest.json + catalogue files)
    and simple numbered JSON record tubs (record_*.json).
    """
    records = []
    tub_path = str(tub_path)

    # Try catalogue-based tub format first
    manifest_path = os.path.join(tub_path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        cat_paths = manifest.get("paths", [])
        for cat_rel in cat_paths:
            cat_path = os.path.join(tub_path, cat_rel)
            if not os.path.exists(cat_path):
                continue
            with open(cat_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    img_name = entry.get("cam/image_array", "")
                    steering = entry.get("user/angle", 0.0)
                    throttle = entry.get("user/throttle", 0.0)
                    img_path = os.path.join(tub_path, "images", img_name) if img_name else ""
                    if img_path and os.path.exists(img_path):
                        records.append((img_path, float(steering), float(throttle)))
        if records:
            return records

    # Fallback: simple record_*.json format
    record_files = sorted(glob.glob(os.path.join(tub_path, "record_*.json")))
    for rec_file in record_files:
        with open(rec_file, "r") as f:
            try:
                entry = json.load(f)
            except json.JSONDecodeError:
                continue
        img_name = entry.get("cam/image_array", "")
        steering = entry.get("user/angle", 0.0)
        throttle = entry.get("user/throttle", 0.0)
        img_path = os.path.join(tub_path, img_name) if img_name else ""
        if img_path and os.path.exists(img_path):
            records.append((img_path, float(steering), float(throttle)))

    return records


class TubDataset(Dataset):
    """PyTorch Dataset over DonkeyCar tub directories."""

    def __init__(self, records, augment=False):
        """
        Args:
            records: list of (image_path, steering, throttle) tuples
            augment: if True, apply horizontal flip + brightness jitter (train only)
        """
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, steering, throttle = self.records[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0  # [0, 1]

        if self.augment:
            # Horizontal flip with 50% probability (mirror steering)
            if random.random() < 0.5:
                img = np.flip(img, axis=1).copy()
                steering = -steering
            # Brightness jitter
            brightness_factor = 0.8 + random.random() * 0.4  # [0.8, 1.2]
            img = np.clip(img * brightness_factor, 0.0, 1.0)

        # HWC -> CHW
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        label_tensor = torch.tensor([steering, throttle], dtype=torch.float32)
        return img_tensor, label_tensor


def make_dataloader(tub_paths, split, batch_size, augment=None):
    """
    Returns a PyTorch DataLoader over DonkeyCar tub directories.

    Each sample: (image_tensor [3,H,W] float32, label_tensor [2] float32)
    label_tensor = [steering, throttle]

    'split' is "train" or "val" — split is done deterministically by tub index.
    Images are normalised to [0, 1] and optionally augmented (train only).
    """
    assert split in ("train", "val")

    # Collect all records from all tub paths
    all_records = []
    for tub_path in tub_paths:
        records = _parse_tub(tub_path)
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(f"No records found in tub paths: {tub_paths}")

    # Deterministic split by record index (sorted by image path for stability)
    all_records.sort(key=lambda r: r[0])
    n = len(all_records)
    n_val = max(1, int(n * VAL_FRACTION))

    if split == "val":
        records = all_records[:n_val]
    else:
        records = all_records[n_val:]

    if augment is None:
        augment = (split == "train")

    dataset = TubDataset(records, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Simulator data generator
# ---------------------------------------------------------------------------

def generate_sim_tub(num_steps=20_000, port=SIM_PORT, out_dir=None):
    """
    Drives the simulated car using a PD lane-centering controller for `num_steps`
    steps, saving each (image, steering, throttle) frame to a DonkeyCar tub.
    Returns the path to the created tub directory.

    This is called ONCE by the human before experiments begin, not by the agent.
    The tub is then used as training data for all subsequent experiments.
    """
    import gymnasium as gym
    import gym_donkeycar  # noqa: F401 — registers envs

    if out_dir is None:
        out_dir = TUB_DIR
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    conf = {"exe_path": "already_running", "port": port}
    env = gym.make(SIM_ENV_ID, conf=conf)

    # PD controller gains
    Kp = 0.8
    Kd = 0.3
    prev_cte = 0.0

    catalogue_path = os.path.join(out_dir, "catalogue_0.jsonl")
    catalogue_entries = []

    throttle = SIM_THROTTLE

    try:
        obs, info = env.reset()
        print(f"Generating tub with {num_steps} steps...")

        for step in range(num_steps):
            cte = info.get("cte", 0.0)
            # PD controller on cross-track error
            steering = -(Kp * cte + Kd * (cte - prev_cte))
            steering = float(np.clip(steering, -1.0, 1.0))
            prev_cte = cte

            # Save image
            img_name = f"{step:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)
            img = Image.fromarray(obs.astype(np.uint8))
            img.save(img_path, quality=95)

            # Save record
            entry = {
                "cam/image_array": img_name,
                "user/angle": steering,
                "user/throttle": throttle,
            }
            catalogue_entries.append(json.dumps(entry))

            # Step the environment
            action = np.array([steering, throttle])
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()
                prev_cte = 0.0

            if (step + 1) % 1000 == 0:
                print(f"  Step {step + 1}/{num_steps}")

    finally:
        env.close()

    # Write catalogue
    with open(catalogue_path, "w") as f:
        f.write("\n".join(catalogue_entries) + "\n")

    # Write manifest
    manifest = {"paths": ["catalogue_0.jsonl"]}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    print(f"Tub saved to {out_dir} ({len(catalogue_entries)} frames)")
    return out_dir


# ---------------------------------------------------------------------------
# Simulator evaluation harness (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sim(model, device, port=SIM_PORT, num_steps=SIM_EVAL_STEPS):
    """
    Runs the model closed-loop in sdsandbox for `num_steps` steps.
    Returns mean absolute cross-track error (val_cte) — lower is better.
    This is the ground-truth metric — do not modify.

    At each step:
      1. Receive obs (120x160x3 image) from the simulator.
      2. Run model forward pass: [steer, throttle] = model(obs).
      3. Send [steer, SIM_THROTTLE] back to the simulator.
      4. Accumulate |info["cte"]| over all steps.

    If the car hits a wall (info["hit"] != "none"), the episode is terminated
    early and the remaining steps are penalised with cte = 1.0.

    If the simulator connection fails (socket error, timeout), raises
    RuntimeError so the caller can treat it as a crash.
    """
    import gymnasium as gym
    import gym_donkeycar  # noqa: F401 — registers envs

    conf = {"exe_path": "already_running", "port": port}
    try:
        env = gym.make(SIM_ENV_ID, conf=conf)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to simulator on port {port}: {e}")

    model.eval()
    total_cte = 0.0
    steps_done = 0

    try:
        obs, info = env.reset()

        for step in range(num_steps):
            # Preprocess observation: uint8 HWC -> float32 CHW [0, 1]
            img = torch.from_numpy(obs.astype(np.float32)).permute(2, 0, 1) / 255.0
            img = img.unsqueeze(0).to(device)  # (1, 3, H, W)

            pred = model(img)  # -> tensor (1, 2): [steering, throttle]
            steer = pred[0, 0].clamp(-1, 1).item()

            action = np.array([steer, SIM_THROTTLE])
            obs, reward, terminated, truncated, info = env.step(action)

            # Check for collision
            hit = info.get("hit", "none")
            if hit != "none":
                # Penalise remaining steps
                remaining = num_steps - step - 1
                total_cte += abs(info.get("cte", 1.0)) + remaining * 1.0
                steps_done = num_steps
                break

            total_cte += abs(info.get("cte", 0.0))
            steps_done += 1

            if terminated or truncated:
                obs, info = env.reset()

    except Exception as e:
        raise RuntimeError(f"Simulator evaluation failed: {e}")
    finally:
        env.close()

    if steps_done == 0:
        raise RuntimeError("No evaluation steps completed")

    return total_cte / num_steps


# ---------------------------------------------------------------------------
# Tub discovery utility
# ---------------------------------------------------------------------------

def find_tub_paths(base_dir=None):
    """Find all tub directories under base_dir (defaults to TUB_DIR)."""
    if base_dir is None:
        base_dir = TUB_DIR
    base_dir = os.path.expanduser(base_dir)

    # If the base_dir itself is a tub (has manifest.json or record_*.json), return it
    if os.path.exists(os.path.join(base_dir, "manifest.json")):
        return [base_dir]
    record_files = glob.glob(os.path.join(base_dir, "record_*.json"))
    if record_files:
        return [base_dir]

    # Otherwise, search subdirectories
    tub_paths = []
    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            if os.path.exists(os.path.join(full_path, "manifest.json")):
                tub_paths.append(full_path)
            elif glob.glob(os.path.join(full_path, "record_*.json")):
                tub_paths.append(full_path)

    return tub_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for DonkeyCar AutoResearch")
    parser.add_argument("--generate", action="store_true",
                        help="Generate training tub from sdsandbox scripted driver")
    parser.add_argument("--num-steps", type=int, default=20_000,
                        help="Number of sim steps for tub generation")
    parser.add_argument("--port", type=int, default=SIM_PORT,
                        help="sdsandbox TCP port")
    args = parser.parse_args()

    if args.generate:
        generate_sim_tub(num_steps=args.num_steps, port=args.port)
    else:
        print("Nothing to do. Use --generate to create a training tub.")
