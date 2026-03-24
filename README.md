# Cortex

MCP server that lets AI agents monitor and tune neural network training in real time.

Instead of staring at loss curves and manually adjusting hyperparameters, connect an AI agent to your training loop. It watches metrics, adjusts the runtime knobs your trainer exposes, runs evals, and rolls back bad changes.

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
3. **Guardrails** — safety constraints on interventions (trainer-declared knob registry, bounds, max % change, cooldowns, rate limits, checkpoint-before-action requirement)

## Quick Start

### In your training code

```python
from cortex import tracker

tracker.config(total_steps=100000, lr=3e-4, batch_size=64)
tracker.define_params(
    {"name": "lr", "description": "Optimizer learning rate", "min_value": 1e-6, "max_value": 1e-2, "max_change_pct": 50},
    {"name": "weight_decay", "description": "Optimizer weight decay", "min_value": 0.0, "max_value": 0.1, "max_change_pct": 50},
    {"name": "dropout", "description": "Model dropout", "min_value": 0.0, "max_value": 0.8, "max_change_pct": 25},
)

@tracker.on_checkpoint
def save(tag):
    torch.save(model.state_dict(), f"checkpoints/{tag}.pt")

@tracker.on_rollback
def rollback(tag):
    model.load_state_dict(torch.load(f"checkpoints/{tag}.pt"))
    return True

@tracker.on_eval
def run_eval(request):
    episodes = int(request.get("episodes", 100))
    score = evaluate_policy(model, eval_env, episodes=episodes)
    tracker.log(step=step, eval_score=score)
    return {"episodes": episodes, "eval_score": score}

for step in range(100000):
    loss = train_step()
    tracker.log(step=step, loss=loss, entropy=ent, grad_norm=gnorm)

    # Process agent commands (checkpoint, rollback, pause/resume, eval)
    tracker.poll()

    # Pick up any runtime knobs you registered
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
| `get_tunable_params` | List the runtime knobs declared by the training loop |
| `list_metrics` | All metric names being tracked |

### Detect
| Tool | Description |
|------|-------------|
| `diagnose` | Run all anomaly detectors, get structured findings with severity and recommendations |

### Intervene
| Tool | Description |
|------|-------------|
| `adjust_param` | Change any registered runtime knob |
| `save_checkpoint` | Save model state for later rollback |
| `run_eval` | Trigger an on-demand evaluation run |
| `rollback` | Restore model to a checkpoint |
| `pause_training` | Pause the loop to analyze and decide |
| `resume_training` | Continue after pause |

### Review
| Tool | Description |
|------|-------------|
| `get_intervention_log` | Full history of changes with reasons and metric snapshots |
| `get_guardrail_status` | Rate limits, cooldowns, intervention count |
| `get_eval_runs` | Recent on-demand eval requests and results |

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

Every adjustment is validated before execution:

- **Trainer-defined registry**: Only knobs declared with `tracker.define_param(...)` are mutable
- **Optional bounds**: Each knob can declare `min_value` and `max_value`
- **Max % change**: Can't change a param by more than 50% at once (configurable)
- **Cooldown**: 30s minimum between adjustments to the same param
- **Rate limit**: Max 20 interventions per hour
- **Checkpoint required**: Must save a checkpoint before making any adjustment

## Simple Autonomous Loop

The intended v1 workflow is:

1. Observe metrics and run `diagnose`
2. `save_checkpoint`
3. Change one registered parameter slightly
4. Let training run for a bit
5. `run_eval`
6. Keep the change if eval improves, otherwise `rollback`

This keeps the LLM focused on managing training rather than editing model internals, and it works for RL, transformers, LLM finetuning, or any other training loop that exposes runtime knobs.

## Install

```bash
pip install cortex-mcp
```

Or from source:

```bash
pip install -e .
```
