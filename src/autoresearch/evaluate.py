"""Simulator evaluation for DonkeyCar models."""

import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
import time
import numpy as np

import torch

from .config import EvaluationConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class SimulatorEvaluator:
    """
    Runs trained CNN models in sdsandbox simulator.
    
    Returns lap times and validates simulator operability.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = get_logger("evaluator")

    def evaluate_model(
        self,
        model_path: Path,
        iteration: int,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run model in simulator and return metrics.
        
        Args:
            model_path: Path to .pt model checkpoint
            iteration: Training iteration number
            dry_run: If True, return dummy metrics
        
        Returns:
            Dict with lap_time, cte, success, timestamp
        """
        self.logger.info("Starting evaluation", iteration=iteration, model_path=str(model_path))

        if dry_run:
            return self._dummy_metrics(iteration)

        # Check simulator is available
        if not self._check_simulator_ready():
            raise RuntimeError("Simulator not ready or not found")

        # Run evaluation (integration point with gym-donkeycar)
        metrics = self._run_simulator_loop(model_path, iteration)
        
        self.logger.info("Evaluation complete", iteration=iteration, metrics=metrics)
        return metrics

    def check_operability(self, dry_run: bool = False) -> bool:
        """Validate simulator can run (health check)."""
        if dry_run:
            return True
        
        # Try to instantiate simulator environment
        try:
            import gym
            from gym_donkeycar.envs.donkey_sim_env import DonkeySimEnv
            
            # Quick instantiation check
            env = DonkeySimEnv(
                exe_path=self._find_simulator_exe(),
                port=9091,
                headless=True,
            )
            env.close()
            return True
        except Exception as e:
            self.logger.error("Operability check failed", error=str(e))
            return False

    def _run_simulator_loop(self, model_path: Path, iteration: int) -> Dict[str, Any]:
        """
        Execute laps in gym-donkeycar simulator with trained model.
        
        Loads checkpoint, drives N laps with deterministic seed,
        records lap times and cross-track errors.
        
        Args:
            model_path: Path to PyTorch checkpoint (.pt file)
            iteration: Iteration number for logging
            
        Returns:
            Dict with aggregated metrics: lap_time, cte_mean, cte_max, success, timestamp
        """
        import gym
        from gym_donkeycar.envs.donkey_sim_env import DonkeySimEnv
        
        start_time = time.time()
        
        try:
            # Load trained model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(model_path, map_location=device)
            model.eval()
            
            self.logger.info(
                "Model loaded",
                iteration=iteration,
                device=str(device),
                model_path=str(model_path),
            )
            
            # Create environment with deterministic seed
            env = DonkeySimEnv(
                exe_path=self._find_simulator_exe(),
                port=self.config.port if hasattr(self.config, 'port') else 9091,
                headless=True,
                seed=self.config.seed,
                conf={
                    "map_name": self.config.map_name,
                    "frame_skip": 1,
                    "body_style": "donkey",
                    "steer_limit_left": 1.0,
                    "steer_limit_right": 1.0,
                }
            )
            
            # Track metrics across laps
            lap_times = []
            cte_errors = []
            all_success = True
            
            # Run N evaluation laps
            for lap_num in range(self.config.num_laps):
                self.logger.info("Starting lap", iteration=iteration, lap=lap_num + 1)
                
                obs = env.reset()
                lap_start_time = time.time()
                step_count = 0
                max_steps = int(self.config.timeout_seconds * 30)  # Assume ~30 FPS
                lap_cte_errors = []
                lap_success = True
                
                while step_count < max_steps:
                    # Prepare observation for model
                    # Assuming obs is image (H, W, C), convert to tensor
                    if isinstance(obs, np.ndarray):
                        obs_tensor = torch.from_numpy(obs).float().to(device)
                        # Normalize to [0, 1] if needed
                        if obs_tensor.max() > 1.0:
                            obs_tensor = obs_tensor / 255.0
                        # Add batch dimension if needed
                        if obs_tensor.dim() == 3:
                            obs_tensor = obs_tensor.unsqueeze(0)
                    else:
                        obs_tensor = obs.to(device)
                    
                    # Get model prediction (steering angle, throttle)
                    with torch.no_grad():
                        action = model(obs_tensor)
                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()
                            if action.ndim > 1:
                                action = action[0]  # Remove batch
                    
                    # Step environment
                    obs, reward, done, info = env.step(action)
                    
                    # Track cross-track error if available
                    if "cte" in info:
                        lap_cte_errors.append(abs(info["cte"]))
                    
                    step_count += 1
                    
                    if done:
                        lap_success = True
                        break
                
                # Record lap time
                lap_time = time.time() - lap_start_time
                lap_times.append(lap_time)
                
                # Record lap CTE stats
                if lap_cte_errors:
                    cte_errors.extend(lap_cte_errors)
                
                if not lap_success or step_count >= max_steps:
                    all_success = False
                
                self.logger.info(
                    "Lap complete",
                    iteration=iteration,
                    lap=lap_num + 1,
                    lap_time=lap_time,
                    cte_mean=np.mean(lap_cte_errors) if lap_cte_errors else None,
                )
            
            # Clean up environment
            env.close()
            
            # Aggregate metrics
            metrics = {
                "lap_time": float(np.mean(lap_times)),
                "cte_mean": float(np.mean(cte_errors)) if cte_errors else 0.0,
                "cte_max": float(np.max(cte_errors)) if cte_errors else 0.0,
                "success": bool(all_success),
                "timestamp": time.time(),
            }
            
            self.logger.info(
                "Evaluation complete",
                iteration=iteration,
                aggregated_metrics=metrics,
                elapsed_seconds=time.time() - start_time,
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(
                "Evaluation failed",
                iteration=iteration,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return failure metrics
            return {
                "lap_time": float(self.config.timeout_seconds),
                "cte_mean": float('inf'),
                "cte_max": float('inf'),
                "success": False,
                "timestamp": time.time(),
            }

    def _check_simulator_ready(self) -> bool:
        """Check if simulator executable exists and is runnable."""
        try:
            exe = self._find_simulator_exe()
            return exe.exists()
        except Exception:
            return False

    @staticmethod
    def _find_simulator_exe() -> Path:
        """Find sdsandbox simulator executable."""
        # Search common locations
        candidates = [
            Path.home() / ".donkey" / "donkey_sim.exe",
            Path.home() / ".donkey" / "donkey_sim",
            Path("/opt/donkeycar/donkey_sim"),
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        # Fallback - user must set DONKEY_SIM_PATH env var
        import os
        sim_path = os.environ.get("DONKEY_SIM_PATH")
        if sim_path:
            return Path(sim_path)
        
        raise FileNotFoundError("Simulator executable not found")

    def _dummy_metrics(self, iteration: int) -> Dict[str, Any]:
        """Generate dummy metrics for testing (dry-run mode)."""
        return {
            "lap_time": 30.0 - (2.0 * iteration),  # Improve with iterations
            "cte_mean": 0.3,
            "cte_max": 0.8,
            "success": True,
            "timestamp": time.time(),
        }
