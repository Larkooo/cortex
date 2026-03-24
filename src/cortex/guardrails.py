"""
Safety guardrails for agent interventions.

Prevents the agent from making changes that are too large, too frequent,
or without a safety checkpoint. Every intervention goes through guardrails
before being applied.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from cortex.tracker import Tracker


@dataclass
class GuardrailConfig:
    """Configuration for intervention safety limits."""
    max_change_pct: float = 50.0          # max % change per adjustment (e.g., lr can't jump more than 50%)
    cooldown_seconds: float = 30.0        # min seconds between adjustments to the same param
    require_checkpoint_before_action: bool = True  # must have at least one checkpoint before any adjustment
    max_interventions_per_hour: int = 20  # rate limit


@dataclass
class InterventionRecord:
    """Logged record of every intervention the agent makes."""
    timestamp: float
    step: int
    action: str             # e.g., "adjust_param", "rollback", "save_checkpoint"
    param: Optional[str]    # which param was changed
    old_value: Any          # value before change
    new_value: Any          # value after change
    reason: str             # why the agent made this change
    metrics_before: dict    # snapshot of metrics at time of intervention

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "action": self.action,
            "param": self.param,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "metrics_before": self.metrics_before,
        }


class Guardrails:
    """Validates and logs all agent interventions."""

    def __init__(self, tracker: Tracker, config: GuardrailConfig = None):
        self.tracker = tracker
        self.config = config or GuardrailConfig()
        self.log: list[InterventionRecord] = []
        self._last_adjustment: dict[str, float] = {}  # param -> timestamp

    def validate_adjustment(self, param: str, new_value: float, reason: str = "") -> dict:
        """Validate a parameter adjustment. Returns {"allowed": True/False, ...}."""
        now = time.time()
        result = {"allowed": True, "warnings": [], "param": param, "new_value": new_value}
        tunable_params = {entry["name"]: entry for entry in self.tracker.get_tunable_params()}
        param_spec = tunable_params.get(param)

        if not param_spec:
            result["allowed"] = False
            result["error"] = (
                f"Parameter '{param}' is not registered as runtime-tunable. "
                "The training loop must declare allowed knobs with tracker.define_param(...)."
            )
            result["allowed_params"] = list(tunable_params.keys())
            return result

        # Check checkpoint requirement
        if self.config.require_checkpoint_before_action:
            checkpoints = self.tracker.get_checkpoints()
            if not checkpoints:
                result["allowed"] = False
                result["error"] = "No checkpoint exists. Save a checkpoint before making adjustments (save_checkpoint tool)."
                return result

        # Check cooldown
        last = self._last_adjustment.get(param, 0)
        elapsed = now - last
        if elapsed < self.config.cooldown_seconds:
            remaining = self.config.cooldown_seconds - elapsed
            result["allowed"] = False
            result["error"] = f"Cooldown active for '{param}'. Wait {remaining:.0f}s before adjusting again."
            return result

        # Check rate limit
        recent = [r for r in self.log if now - r.timestamp < 3600]
        if len(recent) >= self.config.max_interventions_per_hour:
            result["allowed"] = False
            result["error"] = f"Rate limit reached ({self.config.max_interventions_per_hour} interventions/hour). Wait before making more changes."
            return result

        # Check max change percentage
        latest = self.tracker.get_latest()
        config = self.tracker.get_config()
        if param in latest:
            current = latest[param]
        else:
            current = config.get(param)

        min_value = param_spec.get("min_value")
        if min_value is not None and new_value < min_value:
            result["allowed"] = False
            result["error"] = f"Value too small for '{param}': {new_value} < {min_value}."
            return result

        max_value = param_spec.get("max_value")
        if max_value is not None and new_value > max_value:
            result["allowed"] = False
            result["error"] = f"Value too large for '{param}': {new_value} > {max_value}."
            return result

        max_change_pct = param_spec.get("max_change_pct", self.config.max_change_pct)
        if max_change_pct is None:
            max_change_pct = self.config.max_change_pct
        if current is not None and current != 0:
            change_pct = abs(new_value - current) / abs(current) * 100
            if change_pct > max_change_pct:
                result["allowed"] = False
                result["error"] = (
                    f"Change too large: {param} {current} → {new_value} is a {change_pct:.0f}% change. "
                    f"Max allowed is {max_change_pct:.0f}%. Make smaller incremental adjustments."
                )
                return result
            if change_pct > max_change_pct * 0.7:
                result["warnings"].append(f"Large change: {change_pct:.0f}% (limit is {max_change_pct:.0f}%)")

        result["old_value"] = current
        return result

    def record_intervention(self, action: str, param: str = None, old_value: Any = None,
                           new_value: Any = None, reason: str = ""):
        """Record an intervention in the log."""
        record = InterventionRecord(
            timestamp=time.time(),
            step=self.tracker.get_status()["step"],
            action=action,
            param=param,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            metrics_before=self.tracker.get_latest(),
        )
        self.log.append(record)
        if param:
            self._last_adjustment[param] = time.time()
        return record

    def get_log(self, last_n: int = 0) -> list[dict]:
        """Get intervention log."""
        entries = self.log
        if last_n > 0:
            entries = entries[-last_n:]
        return [r.to_dict() for r in entries]

    def get_summary(self) -> dict:
        """Get a summary of all interventions."""
        now = time.time()
        recent = [r for r in self.log if now - r.timestamp < 3600]
        return {
            "total_interventions": len(self.log),
            "interventions_last_hour": len(recent),
            "rate_limit": self.config.max_interventions_per_hour,
            "max_change_pct": self.config.max_change_pct,
            "cooldown_seconds": self.config.cooldown_seconds,
            "require_checkpoint": self.config.require_checkpoint_before_action,
            "params_on_cooldown": {
                p: round(self.config.cooldown_seconds - (now - t), 1)
                for p, t in self._last_adjustment.items()
                if now - t < self.config.cooldown_seconds
            },
        }
