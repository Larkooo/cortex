"""
Core tracker that training loops use to report metrics.

Usage in a training loop:
    from cortex import tracker

    tracker.config(lr=3e-4, hidden=256, batch_size=64)

    for step in range(total_steps):
        loss = train_step()
        tracker.log(step=step, loss=loss, lr=current_lr)

        # Pick up agent adjustments
        new_lr = tracker.get_override("lr")
        if new_lr is not None:
            optimizer.lr = new_lr

        # Handle rollback requests
        rollback_tag = tracker.get_override("__rollback__")
        if rollback_tag:
            load_checkpoint(rollback_tag)

The MCP server reads from the tracker's shared state.
"""

import threading
import time
from collections import defaultdict
from typing import Any, Callable


class Tracker:
    """Thread-safe metric tracker that the MCP server reads from."""

    def __init__(self):
        self._lock = threading.Lock()
        self._config: dict[str, Any] = {}
        self._metrics: dict[str, list[tuple[int, float, float]]] = defaultdict(list)  # name -> [(step, value, timestamp)]
        self._latest: dict[str, float] = {}
        self._step = 0
        self._start_time: float = 0.0
        self._total_steps: int = 0
        self._phase: str = "idle"
        self._checkpoints: dict[str, dict] = {}
        self._param_overrides: dict[str, Any] = {}  # set by MCP, read by training loop
        self._rollback_handler: Callable = None  # user-provided rollback function
        self._checkpoint_handler: Callable = None  # user-provided save function
        self._paused = False

    def config(self, total_steps: int = 0, **kwargs):
        """Set training configuration. Call once at start."""
        with self._lock:
            self._config.update(kwargs)
            self._total_steps = total_steps
            self._start_time = time.time()

    def log(self, step: int = None, **metrics):
        """Log one or more metrics at the current step.

        Args:
            step: Training step (auto-increments if not provided)
            **metrics: Named metric values (e.g., loss=0.5, entropy=1.2)
        """
        now = time.time()
        with self._lock:
            if step is not None:
                self._step = step
            for name, value in metrics.items():
                self._metrics[name].append((self._step, float(value), now))
                self._latest[name] = float(value)

    def phase(self, name: str):
        """Set current training phase (e.g., 'rollout', 'update', 'eval')."""
        with self._lock:
            self._phase = name

    def checkpoint(self, tag: str, metadata: dict = None):
        """Register a checkpoint the agent can reference."""
        with self._lock:
            self._checkpoints[tag] = {
                "step": self._step,
                "time": time.time(),
                "metrics": dict(self._latest),
                "metadata": metadata or {},
            }

    def on_checkpoint(self, handler: Callable):
        """Register a callback for when the agent requests a checkpoint save.

        The handler receives (tag: str) and should save model state.

            @tracker.on_checkpoint
            def save(tag):
                torch.save(model.state_dict(), f"checkpoints/{tag}.pt")
        """
        self._checkpoint_handler = handler
        return handler

    def on_rollback(self, handler: Callable):
        """Register a callback for when the agent requests a rollback.

        The handler receives (tag: str) and should restore model state.
        Return True if rollback succeeded, False otherwise.

            @tracker.on_rollback
            def rollback(tag):
                state = torch.load(f"checkpoints/{tag}.pt")
                model.load_state_dict(state)
                return True
        """
        self._rollback_handler = handler
        return handler

    def get_override(self, name: str, default=None):
        """Check if the MCP agent has overridden a parameter.

        Call this in your training loop to pick up agent adjustments:
            lr = tracker.get_override("lr", default=current_lr)
        """
        with self._lock:
            return self._param_overrides.pop(name, default)

    def get_overrides(self) -> dict:
        """Get and clear all pending overrides."""
        with self._lock:
            overrides = dict(self._param_overrides)
            self._param_overrides.clear()
            return overrides

    def poll(self):
        """Process pending agent commands (checkpoints, rollbacks, pause/resume).

        Call this once per training step to handle async agent requests.
        Returns a dict of events that occurred, e.g.:
            {"checkpoint_saved": "step_50000", "paused": True}
        """
        events = {}

        # Handle checkpoint requests
        tag = self.get_override("__save_checkpoint__")
        if tag and self._checkpoint_handler:
            self._checkpoint_handler(tag)
            self.checkpoint(tag)
            events["checkpoint_saved"] = tag

        # Handle rollback requests
        tag = self.get_override("__rollback__")
        if tag:
            if self._rollback_handler:
                success = self._rollback_handler(tag)
                if success:
                    with self._lock:
                        cp = self._checkpoints.get(tag)
                        if cp:
                            self._step = cp["step"]
                    events["rollback"] = tag
                else:
                    events["rollback_failed"] = tag
            else:
                events["rollback_no_handler"] = tag

        # Handle pause/resume
        if self.get_override("__pause__"):
            self._paused = True
            events["paused"] = True
            self.phase("paused")

        if self.get_override("__resume__"):
            self._paused = False
            events["resumed"] = True

        # Block while paused
        while self._paused:
            time.sleep(0.1)
            # Check for resume
            if self.get_override("__resume__"):
                self._paused = False
                events["resumed"] = True
                break
            # Also check for param adjustments while paused
            if self.get_override("__rollback__"):
                # Handle rollback while paused
                tag = self._param_overrides.get("__rollback__")
                if tag and self._rollback_handler:
                    self._rollback_handler(tag)
                    events["rollback"] = tag

        return events

    # ── Read methods (used by MCP server) ──

    def get_status(self) -> dict:
        with self._lock:
            elapsed = time.time() - self._start_time if self._start_time else 0
            sps = self._step / elapsed if elapsed > 0 else 0
            eta = (self._total_steps - self._step) / sps if sps > 0 else 0
            return {
                "step": self._step,
                "total_steps": self._total_steps,
                "phase": self._phase,
                "elapsed_seconds": round(elapsed, 1),
                "steps_per_second": round(sps, 1),
                "eta_seconds": round(eta, 1),
                "progress": round(self._step / self._total_steps, 4) if self._total_steps > 0 else 0,
                "has_rollback_handler": self._rollback_handler is not None,
                "has_checkpoint_handler": self._checkpoint_handler is not None,
            }

    def get_config(self) -> dict:
        with self._lock:
            return dict(self._config)

    def get_latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def get_history(self, metric: str, last_n: int = 0) -> list[dict]:
        with self._lock:
            entries = self._metrics.get(metric, [])
            if last_n > 0:
                entries = entries[-last_n:]
            return [{"step": s, "value": v, "time": t} for s, v, t in entries]

    def get_metric_names(self) -> list[str]:
        with self._lock:
            return list(self._metrics.keys())

    def get_checkpoints(self) -> dict:
        with self._lock:
            return dict(self._checkpoints)

    def set_override(self, name: str, value: Any):
        """Called by MCP server to queue a parameter change."""
        with self._lock:
            self._param_overrides[name] = value

    def reset(self):
        """Reset all state."""
        with self._lock:
            self._metrics.clear()
            self._latest.clear()
            self._config.clear()
            self._step = 0
            self._start_time = 0.0
            self._total_steps = 0
            self._phase = "idle"
            self._checkpoints.clear()
            self._param_overrides.clear()


# Global singleton — training code imports this
tracker = Tracker()
