# Cortex

MCP server that lets AI agents monitor and tune neural network training in real time.

Instead of staring at loss curves and manually adjusting hyperparameters, connect an AI agent to your training loop. It watches metrics, detects anomalies, and makes guarded interventions — with safety rails to prevent it from doing more harm than good.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Training Loop   │────▶│    Telemetry      │◀────│   AI Agent     │
│                  │     │  (tracker.log)    │     │  (via MCP)     │
│  PyTorch / MLX   │◀────│                  │────▶│                │
│  JAX / any       │     │  ┌────────────┐  │     │  diagnose()    │
│                  │     │  │ Detectors  │  │     │  adjust_param()│
│  tracker.poll()  │     │  │ Guardrails │  │     │  rollback()    │
│  get_override()  │     │  │ Log        │  │     │  save/restore  │
└─────────────────┘     └──────────────────┘     └───────────────┘
```

Three layers:

1. **Telemetry** — metrics, config, checkpoints, history
2. **Detectors** — rule-based anomaly detection producing typed findings (entropy collapse, loss divergence, gradient spikes, eval plateau, loss imbalance)
3. **Guardrails** — safety constraints on interventions (max % change, cooldowns, rate limits, checkpoint-before-action requirement, full intervention log)

## Quick Start

### In your training code

```python
from cortex import tracker

tracker.config(total_steps=100000, lr=3e-4, batch_size=64)

@tracker.on_checkpoint
def save(tag):
    torch.save(model.state_dict(), f"checkpoints/{tag}.pt")

@tracker.on_rollback
def rollback(tag):
    model.load_state_dict(torch.load(f"checkpoints/{tag}.pt"))
    return True

for step in range(100000):
    loss = train_step()
    tracker.log(step=step, loss=loss, entropy=ent, grad_norm=gnorm)

    # Process agent commands (checkpoint, rollback, pause/resume)
    tracker.poll()

    # Pick up live param adjustments
    new_lr = tracker.get_override("lr")
    if new_lr is not None:
        optimizer.lr = new_lr
```

### Connect an agent

```json
{
  "mcpServers": {
    "cortex": {
      "command": "cortex"
    }
  }
}
```

## MCP Tools

### Observe
| Tool | Description |
|------|-------------|
| `get_status` | Step, progress %, ETA, steps/sec, phase |
| `get_metrics` | Latest values of all tracked metrics |
| `get_metric_history` | Time series for trend analysis |
| `get_config` | Hyperparameter configuration |
| `list_metrics` | All metric names being tracked |

### Detect
| Tool | Description |
|------|-------------|
| `diagnose` | Run all anomaly detectors, get structured findings with severity and recommendations |

### Intervene
| Tool | Description |
|------|-------------|
| `adjust_param` | Change a hyperparameter (guarded: max % change, cooldown, requires checkpoint) |
| `save_checkpoint` | Save model state for later rollback |
| `rollback` | Restore model to a checkpoint |
| `pause_training` | Pause the loop to analyze and decide |
| `resume_training` | Continue after pause |

### Review
| Tool | Description |
|------|-------------|
| `get_intervention_log` | Full history of every change with before/after metrics |
| `get_guardrail_status` | Rate limits, cooldowns, intervention count |

## Detectors

Built-in anomaly detectors that produce structured findings:

| Detector | What it catches |
|----------|----------------|
| `entropy_collapse` | Policy committing too early (entropy dropping fast while not improving) |
| `loss_divergence` | NaN/inf loss, or loss increasing rapidly |
| `loss_imbalance` | Value loss drowning out policy loss signal |
| `grad_instability` | Gradient norm spikes that precede divergence |
| `eval_plateau` | Eval score stopped improving |
| `learning_stall` | Loss stopped decreasing |

Each finding includes severity, explanation, the specific metric values, and a recommended action.

## Guardrails

Every intervention is validated before execution:

- **Max % change**: Can't change a param by more than 50% at once (configurable)
- **Cooldown**: 30s minimum between adjustments to the same param
- **Rate limit**: Max 20 interventions per hour
- **Checkpoint required**: Must save a checkpoint before making any adjustment
- **Full logging**: Every intervention recorded with before/after metrics and reason

## Install

```bash
pip install cortex-mcp
```

Or from source:

```bash
pip install -e .
```
