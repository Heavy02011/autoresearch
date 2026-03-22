"""Training integration for autoresearch iteration loop."""

import gc
import time
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .logging_config import get_logger

logger = get_logger(__name__)


class TrainingIterator:
    """
    Wraps the training loop with time budget constraints.
    
    Initialized once, then called repeatedly for each iteration
    to train for a bounded amount of time and return a checkpoint.
    """

    def __init__(self, time_budget_minutes: float = 5.0):
        """
        Initialize training setup.
        
        Args:
            time_budget_minutes: Time budget per iteration (minutes)
        """
        self.time_budget_seconds = time_budget_minutes * 60.0
        self.logger = get_logger("training")
        
        # Lazy-initialized globals (loaded from train.py on first use)
        self._initialized = False
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.device = None
        self.autocast_ctx = None
        self.x = None
        self.y = None
        self.step = 0
        self.total_training_time = 0.0
        self.smooth_train_loss = 0.0

    def _lazy_init(self):
        """Initialize training infrastructure from train.py on first use."""
        if self._initialized:
            return
        
        self.logger.info("Initializing training loop", time_budget_seconds=self.time_budget_seconds)
        
        try:
            # Import train.py variables (requires that train.py has been run at module level)
            from train import (
                model, optimizer, train_loader, device, autocast_ctx,
                TOTAL_BATCH_SIZE, grad_accum_steps
            )
            
            self.model = model
            self.optimizer = optimizer
            self.train_loader = train_loader
            self.device = device
            self.autocast_ctx = autocast_ctx
            self.TOTAL_BATCH_SIZE = TOTAL_BATCH_SIZE
            self.grad_accum_steps = grad_accum_steps
            
            # Prefetch first batch
            self.x, self.y, self.epoch = next(self.train_loader)
            
            self._initialized = True
            self.logger.info("Training initialization complete")
            
        except ImportError as e:
            self.logger.error("Failed to import train.py", error=str(e))
            raise RuntimeError(
                "Training initialization failed. Make sure train.py has been run "
                "and the training environment (tokenizer, model, dataloader) is available."
            ) from e

    def run_iteration(self, iteration: int, checkpoint_path: Path) -> Path:
        """
        Train model for one iteration with time budget.
        
        Runs the training loop with a bounded time budget, then saves
        the current model state to checkpoint.
        
        Args:
            iteration: Iteration number (for logging)
            checkpoint_path: Path where to save the model checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        if not self._initialized:
            self._lazy_init()
        
        self.logger.info(
            "Starting training iteration",
            iteration=iteration,
            time_budget_seconds=self.time_budget_seconds
        )
        
        iteration_start_time = time.time()
        iteration_training_time = 0.0
        
        # Run training loop with time budget constraint
        while iteration_training_time < self.time_budget_seconds:
            step_start_time = time.time()
            
            # Gradient accumulation
            for micro_step in range(self.grad_accum_steps):
                with self.autocast_ctx:
                    loss = self.model(self.x, self.y)
                
                train_loss = loss.detach()
                loss = loss / self.grad_accum_steps
                loss.backward(set_to_none=True)
                
                # Prefetch next batch
                try:
                    self.x, self.y, self.epoch = next(self.train_loader)
                except StopIteration:
                    # Restart dataset if exhausted
                    self.train_loader = iter(self.train_loader)
                    self.x, self.y, self.epoch = next(self.train_loader)
            
            # Optimizer step
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)
            
            # Timing
            torch.cuda.synchronize()
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            
            if self.step > 10:
                iteration_training_time += step_time
                self.total_training_time += step_time
            
            # Logging
            train_loss_f = train_loss.item()
            ema_beta = 0.9
            self.smooth_train_loss = ema_beta * self.smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased_loss = self.smooth_train_loss / (1 - ema_beta ** (self.step + 1))
            
            self.logger.info(
                "Training step",
                iteration=iteration,
                step=self.step,
                loss=debiased_loss,
                step_time_ms=step_time * 1000,
            )
            
            # Fail fast on exploding loss
            if train_loss_f > 100:
                self.logger.error("Loss exploded", iteration=iteration, loss=train_loss_f)
                raise RuntimeError(f"Training loss exploded: {train_loss_f}")
            
            # GC management
            if self.step == 0:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif (self.step + 1) % 1000 == 0:
                gc.collect()
            
            self.step += 1
        
        # Save checkpoint
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Saving checkpoint",
            iteration=iteration,
            checkpoint_path=str(checkpoint_path),
            model_device=str(self.model.device) if hasattr(self.model, 'device') else 'unknown',
        )
        
        # Convert model to CPU for checkpoint to reduce memory usage
        model_device = next(self.model.parameters()).device
        self.model = self.model.cpu()
        torch.save(self.model.state_dict(), checkpoint_path)
        self.model = self.model.to(model_device)
        
        elapsed = time.time() - iteration_start_time
        self.logger.info(
            "Training iteration complete",
            iteration=iteration,
            elapsed_seconds=elapsed,
            training_time_seconds=iteration_training_time,
        )
        
        return checkpoint_path


# Global training iterator (initialized on first use)
_training_iterator = None


def get_training_iterator(time_budget_minutes: float = 5.0) -> TrainingIterator:
    """Get or create the global training iterator."""
    global _training_iterator
    if _training_iterator is None:
        _training_iterator = TrainingIterator(time_budget_minutes=time_budget_minutes)
    return _training_iterator
