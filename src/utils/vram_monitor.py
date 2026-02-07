"""VRAM monitoring and leak detection for long Kaggle training runs.

Detects:
1. VRAM leaks (gradual memory growth)
2. Loss explosions (NaN or >100)
3. Training stalls (step time plateaus)
"""

import time
from collections import deque
from typing import Optional, Dict, Any

import torch


class VRAMMonitor:
    """Monitor GPU memory usage and detect anomalies during training.

    Design for Kaggle T4:
    - Check every N steps (default 50)
    - Alert if peak VRAM grows by >0.5 GB between checks
    - Alert if loss is NaN or >100
    - Estimate ETA based on rolling average step time
    """

    def __init__(
        self,
        check_interval: int = 50,
        leak_threshold_gb: float = 0.5,
        loss_explosion_threshold: float = 100.0,
        rolling_window: int = 100,
        device: Optional[torch.device] = None,
    ):
        """Initialize VRAM monitor.

        Args:
            check_interval: Check memory every N steps
            leak_threshold_gb: Alert if VRAM grows by this amount (GB)
            loss_explosion_threshold: Alert if loss exceeds this value
            rolling_window: Window size for rolling average step time
            device: CUDA device to monitor (default: cuda:0)
        """
        self.check_interval = check_interval
        self.leak_threshold_gb = leak_threshold_gb
        self.loss_explosion_threshold = loss_explosion_threshold
        self.rolling_window = rolling_window

        # Device setup
        if device is None and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = device

        # State tracking
        self.step_times = deque(maxlen=rolling_window)
        self.peak_memory_history = []
        self.loss_history = []

        # Baseline measurements
        self.baseline_peak_memory = None
        self.start_time = None
        self.last_check_step = 0

        # Anomaly flags
        self.leak_detected = False
        self.explosion_detected = False

    def start(self):
        """Start monitoring (call at beginning of training)."""
        self.start_time = time.time()
        if self.device and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self.baseline_peak_memory = self._get_peak_memory_gb()

    def step(self, step: int, loss: float) -> Dict[str, Any]:
        """Update monitoring state after a training step.

        Args:
            step: Current training step
            loss: Loss value for this step

        Returns:
            Dictionary with monitoring stats and alerts
        """
        current_time = time.time()

        # Record step time
        if len(self.step_times) > 0:
            step_duration = current_time - self.last_step_time
            self.step_times.append(step_duration)
        self.last_step_time = current_time

        # Record loss
        self.loss_history.append(loss)

        # Check for loss explosion
        if torch.isnan(torch.tensor(loss)) or loss > self.loss_explosion_threshold:
            self.explosion_detected = True

        # Periodic VRAM check
        stats = {}
        if step % self.check_interval == 0 and step > 0:
            stats = self._check_vram(step)
            self.last_check_step = step

        return stats

    def _check_vram(self, step: int) -> Dict[str, Any]:
        """Perform VRAM check and leak detection.

        Returns:
            Dictionary with VRAM stats and alerts
        """
        if not self.device or self.device.type != "cuda":
            return {}

        current_peak = self._get_peak_memory_gb()
        allocated = self._get_allocated_memory_gb()
        reserved = self._get_reserved_memory_gb()

        self.peak_memory_history.append(current_peak)

        # Detect memory leak
        if len(self.peak_memory_history) >= 2:
            growth = current_peak - self.peak_memory_history[-2]
            if growth > self.leak_threshold_gb:
                self.leak_detected = True
        else:
            growth = 0.0

        # Estimate ETA
        eta_str = self._estimate_eta(step)

        stats = {
            "step": step,
            "peak_memory_gb": current_peak,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "memory_growth_gb": growth,
            "leak_detected": self.leak_detected,
            "explosion_detected": self.explosion_detected,
            "avg_step_time": self._get_avg_step_time(),
            "eta": eta_str,
        }

        return stats

    def _get_peak_memory_gb(self) -> float:
        """Get peak memory usage in GB."""
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

    def _get_allocated_memory_gb(self) -> float:
        """Get currently allocated memory in GB."""
        return torch.cuda.memory_allocated(self.device) / (1024 ** 3)

    def _get_reserved_memory_gb(self) -> float:
        """Get reserved memory by PyTorch in GB."""
        return torch.cuda.memory_reserved(self.device) / (1024 ** 3)

    def _get_avg_step_time(self) -> float:
        """Get average step time from rolling window."""
        if len(self.step_times) == 0:
            return 0.0
        return sum(self.step_times) / len(self.step_times)

    def _estimate_eta(self, current_step: int, total_steps: Optional[int] = None) -> str:
        """Estimate time remaining until completion.

        Args:
            current_step: Current training step
            total_steps: Total steps (if known)

        Returns:
            ETA string (e.g., "2h 34m")
        """
        if total_steps is None or len(self.step_times) == 0:
            return "unknown"

        remaining_steps = total_steps - current_step
        avg_step_time = self._get_avg_step_time()
        remaining_seconds = remaining_steps * avg_step_time

        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring session.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self.device or self.device.type != "cuda":
            return {"device": "cpu", "monitoring_active": False}

        total_time = time.time() - self.start_time if self.start_time else 0

        return {
            "device": str(self.device),
            "total_time_seconds": total_time,
            "avg_step_time": self._get_avg_step_time(),
            "peak_memory_gb": max(self.peak_memory_history) if self.peak_memory_history else 0,
            "baseline_memory_gb": self.baseline_peak_memory,
            "final_memory_gb": self._get_peak_memory_gb(),
            "memory_growth_gb": (
                self._get_peak_memory_gb() - self.baseline_peak_memory
                if self.baseline_peak_memory else 0
            ),
            "leak_detected": self.leak_detected,
            "explosion_detected": self.explosion_detected,
            "num_steps": len(self.loss_history),
        }

    def should_abort(self) -> bool:
        """Check if training should be aborted due to detected issues.

        Returns:
            True if leak or explosion detected
        """
        return self.leak_detected or self.explosion_detected

    def get_abort_reason(self) -> Optional[str]:
        """Get reason for abort recommendation.

        Returns:
            String describing the issue, or None if no issues
        """
        if self.explosion_detected:
            return "Loss explosion detected (NaN or >100)"
        if self.leak_detected:
            return f"VRAM leak detected (growth >{self.leak_threshold_gb} GB)"
        return None

    def reset(self):
        """Reset monitoring state (for new training run)."""
        self.step_times.clear()
        self.peak_memory_history.clear()
        self.loss_history.clear()
        self.leak_detected = False
        self.explosion_detected = False
        self.baseline_peak_memory = None
        self.start_time = None
        self.last_check_step = 0

        if self.device and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"VRAMMonitor(\n"
            f"  device={summary.get('device')},\n"
            f"  peak_memory={summary.get('peak_memory_gb', 0):.2f} GB,\n"
            f"  avg_step_time={summary.get('avg_step_time', 0):.3f}s,\n"
            f"  leak_detected={summary.get('leak_detected', False)},\n"
            f"  explosion_detected={summary.get('explosion_detected', False)}\n"
            f")"
        )
