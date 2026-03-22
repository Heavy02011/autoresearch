"""
DonkeyCar AutoResearch training script. Single-GPU, single-file.
Trains a CNN autopilot model on simulator tub data and evaluates
closed-loop in sdsandbox.

Usage: python train_donkey.py
       python train_donkey.py --tub /path/to/tub
       DONKEY_TUB=/path/to/tub python train_donkey.py
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_donkey import (
    TIME_BUDGET,
    IMG_H, IMG_W, IMG_CHANNELS,
    BEST_MODEL_PATH,
    find_tub_paths,
    make_dataloader,
    evaluate_sim,
)

# ---------------------------------------------------------------------------
# Editable hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adam"           # "adam" | "sgd" | "adamw"
AUGMENT = True
STEERING_WEIGHT = 1.0       # relative loss weight for steering
THROTTLE_WEIGHT = 0.5       # relative loss weight for throttle

# ---------------------------------------------------------------------------
# DonkeyCar Baseline CNN Model
# ---------------------------------------------------------------------------

class DonkeyNet(nn.Module):
    """Baseline CNN autopilot. Agent may replace entirely."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened feature size from encoder output shape
        with torch.no_grad():
            dummy = torch.zeros(1, IMG_CHANNELS, IMG_H, IMG_W)
            flat_size = self.encoder(dummy).shape[1]  # 1152 for default 120x160 input
        self.head = nn.Sequential(
            nn.Linear(flat_size, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 2),    # [steering, throttle]
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Train DonkeyCar autopilot")
    parser.add_argument("--tub", type=str, default=None,
                        help="Path to tub directory (overrides DONKEY_TUB env var and default "
                             "~/donkeycar/data/sim_tub). Accepts sim-generated or real-world tubs.")
    args = parser.parse_args()

    # Resolve tub base directory: CLI flag > env var > default
    tub_base = args.tub or os.environ.get("DONKEY_TUB")
    if tub_base:
        tub_base = os.path.expanduser(tub_base)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Find tub data
    tub_paths = find_tub_paths(tub_base)
    if not tub_paths:
        if tub_base:
            print(f"ERROR: No tub data found at '{tub_base}'.")
        else:
            print("ERROR: No tub data found. Run: python prepare_donkey.py --generate")
        return
    print(f"Found {len(tub_paths)} tub(s)")

    # Data loaders
    train_loader = make_dataloader(tub_paths, "train", BATCH_SIZE, augment=AUGMENT)
    val_loader = make_dataloader(tub_paths, "val", BATCH_SIZE, augment=False)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Model
    model = DonkeyNet().to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params / 1000:.1f}k)")

    # Optimizer
    if OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                    weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    # Loss weights
    loss_weights = torch.tensor([STEERING_WEIGHT, THROTTLE_WEIGHT], device=device)

    # Training loop
    print(f"\nTraining for {TIME_BUDGET}s...")
    model.train()
    best_val_loss = float("inf")
    epoch = 0
    total_samples = 0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        epoch += 1
        epoch_loss = 0.0
        epoch_samples = 0

        for images, labels in train_loader:
            if time.time() - start_time >= TIME_BUDGET:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            pred = model(images)
            # Weighted MSE loss
            diff = (pred - labels) ** 2
            loss = (diff * loss_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            total_samples += batch_size

        if epoch_samples > 0:
            avg_loss = epoch_loss / epoch_samples
            # Quick validation loss on static dataset
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    pred = model(images)
                    diff = (pred - labels) ** 2
                    loss = (diff * loss_weights).mean()
                    val_loss += loss.item() * images.size(0)
                    val_count += images.size(0)
            val_loss = val_loss / val_count if val_count > 0 else float("inf")
            model.train()

            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} | train_loss: {avg_loss:.6f} | val_loss: {val_loss:.6f} | "
                  f"samples: {total_samples} | elapsed: {elapsed:.1f}s")

            # Save best model (by static val loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)

    training_seconds = time.time() - start_time

    # Load best model for simulator evaluation
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Closed-loop evaluation in sdsandbox
    print("\nEvaluating in simulator...")
    eval_start = time.time()
    try:
        val_cte = evaluate_sim(model, device)
    except RuntimeError as e:
        print(f"Simulator evaluation failed: {e}")
        val_cte = float("inf")
    eval_seconds = time.time() - eval_start
    total_seconds = time.time() - start_time

    # Peak VRAM
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    # Summary
    print("---")
    print(f"val_cte:          {val_cte:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_epochs:       {epoch}")
    print(f"num_samples:      {total_samples}")
    print(f"num_params_k:     {num_params / 1000:.0f}")


if __name__ == "__main__":
    main()
