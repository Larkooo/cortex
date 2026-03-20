"""
Rule-based anomaly detectors that produce typed findings.

These run on metric history and produce structured signals the agent
can reason about — instead of the agent interpreting raw numbers.

Each detector returns a list of Finding objects, or an empty list if healthy.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from cortex.tracker import Tracker


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Finding:
    """A structured detection result."""
    detector: str           # which detector produced this
    severity: Severity
    message: str            # human-readable explanation
    metric: str             # which metric triggered it
    current_value: float
    reference_value: float  # what it's compared against
    recommendation: str     # suggested action
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "detector": self.detector,
            "severity": self.severity.value,
            "message": self.message,
            "metric": self.metric,
            "current_value": round(self.current_value, 6),
            "reference_value": round(self.reference_value, 6),
            "recommendation": self.recommendation,
            "details": self.details,
        }


def run_all(tracker: Tracker) -> list[Finding]:
    """Run all detectors and return findings."""
    findings = []
    for detector in ALL_DETECTORS:
        findings.extend(detector(tracker))
    return findings


def detect_entropy_collapse(tracker: Tracker, window: int = 20, drop_threshold: float = 0.4) -> list[Finding]:
    """Detect when entropy drops too fast — policy committing before exploring enough."""
    history = tracker.get_history("entropy", last_n=window)
    if len(history) < 5:
        return []

    values = [h["value"] for h in history]
    first_half = values[:len(values)//2]
    second_half = values[len(values)//2:]
    avg_early = sum(first_half) / len(first_half)
    avg_late = sum(second_half) / len(second_half)

    if avg_early <= 0:
        return []

    drop_pct = (avg_early - avg_late) / avg_early

    if drop_pct > drop_threshold:
        severity = Severity.CRITICAL if drop_pct > 0.6 else Severity.WARNING
        return [Finding(
            detector="entropy_collapse",
            severity=severity,
            message=f"Entropy dropped {drop_pct:.0%} over the last {window} readings ({avg_early:.3f} → {avg_late:.3f}). Policy may be committing to a suboptimal strategy before exploring enough.",
            metric="entropy",
            current_value=avg_late,
            reference_value=avg_early,
            recommendation=f"Increase ent_coef to slow entropy decay. Current drop is {drop_pct:.0%}.",
            details={"drop_percent": round(drop_pct, 3), "window": window},
        )]
    return []


def detect_loss_divergence(tracker: Tracker, window: int = 10) -> list[Finding]:
    """Detect NaN/inf loss or rapidly increasing loss."""
    history = tracker.get_history("loss", last_n=window)
    if len(history) < 3:
        return []

    values = [h["value"] for h in history]
    findings = []

    # NaN/inf check
    if any(math.isnan(v) or math.isinf(v) for v in values):
        findings.append(Finding(
            detector="loss_divergence",
            severity=Severity.CRITICAL,
            message="Loss contains NaN or Inf values. Model has diverged.",
            metric="loss",
            current_value=values[-1],
            reference_value=values[0],
            recommendation="Immediately rollback to last good checkpoint and reduce learning rate.",
        ))
        return findings

    # Rapid increase check
    if len(values) >= 5:
        early = sum(values[:3]) / 3
        late = sum(values[-3:]) / 3
        if early > 0 and late / early > 3.0:
            findings.append(Finding(
                detector="loss_divergence",
                severity=Severity.WARNING,
                message=f"Loss increased {late/early:.1f}x over the last {window} readings ({early:.4f} → {late:.4f}). Training may be unstable.",
                metric="loss",
                current_value=late,
                reference_value=early,
                recommendation="Consider reducing learning rate or rolling back to a checkpoint.",
                details={"increase_ratio": round(late / early, 2)},
            ))

    return findings


def detect_loss_imbalance(tracker: Tracker) -> list[Finding]:
    """Detect when one loss component dominates (e.g., value loss >> policy loss)."""
    latest = tracker.get_latest()
    pg = latest.get("pg_loss")
    vf = latest.get("vf_loss")

    if pg is None or vf is None:
        return []
    if abs(pg) < 1e-10:
        return []

    ratio = abs(vf) / max(abs(pg), 1e-10)

    if ratio > 100:
        return [Finding(
            detector="loss_imbalance",
            severity=Severity.WARNING,
            message=f"Value loss ({vf:.4f}) is {ratio:.0f}x larger than policy loss ({pg:.6f}). The value head is dominating gradient updates, starving the policy of learning signal.",
            metric="vf_loss",
            current_value=vf,
            reference_value=pg,
            recommendation=f"Reduce vf_coef to rebalance. Current ratio is {ratio:.0f}x.",
            details={"ratio": round(ratio, 1), "pg_loss": pg, "vf_loss": vf},
        )]
    return []


def detect_grad_instability(tracker: Tracker, window: int = 20, spike_threshold: float = 5.0) -> list[Finding]:
    """Detect gradient norm spikes that often precede divergence."""
    history = tracker.get_history("grad_norm", last_n=window)
    if len(history) < 5:
        return []

    values = [h["value"] for h in history]
    mean = sum(values) / len(values)
    if mean <= 0:
        return []

    latest = values[-1]
    ratio = latest / mean

    if ratio > spike_threshold:
        return [Finding(
            detector="grad_instability",
            severity=Severity.WARNING if ratio < 10 else Severity.CRITICAL,
            message=f"Gradient norm spiked to {latest:.4f} ({ratio:.1f}x the recent average of {mean:.4f}). This often precedes loss divergence.",
            metric="grad_norm",
            current_value=latest,
            reference_value=mean,
            recommendation="Save checkpoint immediately. Consider reducing learning rate.",
            details={"spike_ratio": round(ratio, 2), "mean": round(mean, 4)},
        )]
    return []


def detect_eval_plateau(tracker: Tracker, window: int = 10, tolerance: float = 0.01) -> list[Finding]:
    """Detect when eval score stops improving."""
    history = tracker.get_history("eval_score", last_n=window)
    if len(history) < 5:
        return []

    values = [h["value"] for h in history]
    first_half = values[:len(values)//2]
    second_half = values[len(values)//2:]

    avg_early = sum(first_half) / len(first_half)
    avg_late = sum(second_half) / len(second_half)

    if avg_early <= 0:
        return []

    improvement = (avg_late - avg_early) / abs(avg_early)

    if abs(improvement) < tolerance and len(history) >= 8:
        return [Finding(
            detector="eval_plateau",
            severity=Severity.INFO,
            message=f"Eval score has plateaued around {avg_late:.3f} (only {improvement:+.1%} change over last {window} evals). Training may have converged.",
            metric="eval_score",
            current_value=avg_late,
            reference_value=avg_early,
            recommendation="Consider: (1) increasing entropy to explore more, (2) adjusting learning rate, or (3) stopping training if this is good enough.",
            details={"improvement_pct": round(improvement * 100, 2)},
        )]
    return []


def detect_learning_stall(tracker: Tracker, window: int = 30, tolerance: float = 0.02) -> list[Finding]:
    """Detect when loss stops decreasing — different from eval plateau."""
    history = tracker.get_history("loss", last_n=window)
    if len(history) < 10:
        return []

    values = [h["value"] for h in history]
    first_third = values[:len(values)//3]
    last_third = values[-len(values)//3:]

    avg_early = sum(first_third) / len(first_third)
    avg_late = sum(last_third) / len(last_third)

    if avg_early <= 0:
        return []

    change = (avg_early - avg_late) / avg_early  # positive = loss decreasing = good

    if change < tolerance and change > -0.1:  # not decreasing, but not diverging either
        return [Finding(
            detector="learning_stall",
            severity=Severity.INFO,
            message=f"Loss has stalled around {avg_late:.4f} (only {change:+.1%} change over {window} readings). Learning rate may need adjustment.",
            metric="loss",
            current_value=avg_late,
            reference_value=avg_early,
            recommendation="Try increasing learning rate slightly, or check if entropy is too low for further exploration.",
            details={"change_pct": round(change * 100, 2)},
        )]
    return []


ALL_DETECTORS = [
    detect_entropy_collapse,
    detect_loss_divergence,
    detect_loss_imbalance,
    detect_grad_instability,
    detect_eval_plateau,
    detect_learning_stall,
]
