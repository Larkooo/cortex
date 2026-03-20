# Cortex

MCP server that lets AI agents monitor and tune neural network training in real time.

Instead of staring at TensorBoard and manually adjusting hyperparameters, connect an AI agent to your training loop. It watches metrics, spots problems (entropy collapse, gradient explosion, loss plateaus), and can adjust parameters mid-run.

## Quick Start

### In your training code

```python
from cortex import tracker

tracker.config(total_steps=100000, lr=3e-4, batch_size=64)

for step in range(100000):
    loss = train_step()

    tracker.log(step=step, loss=loss, entropy=ent, grad_norm=gnorm)

    # Pick up agent adjustments
    new_lr = tracker.get_override("lr")
    if new_lr is not None:
        optimizer.lr = new_lr
```

### Connect an agent

Add to your MCP client config (e.g. Claude Desktop):

```json
{
  "mcpServers": {
    "cortex": {
      "command": "cortex"
    }
  }
}
```

The agent can then call tools like `get_metrics`, `get_metric_history`, `adjust_param`, and `save_checkpoint`.

## Install

```bash
pip install cortex-mcp
```

Or from source:

```bash
pip install -e .
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `get_status` | Step, progress %, ETA, steps/sec, phase |
| `get_metrics` | Latest values of all tracked metrics |
| `get_metric_history` | Time series for a specific metric (for trend analysis) |
| `get_config` | Hyperparameter configuration |
| `list_metrics` | All metric names being tracked |
| `list_checkpoints` | Saved checkpoints with metadata |
| `adjust_param` | Override a hyperparameter mid-training |
| `save_checkpoint` | Request the training loop to save state |

## MCP Resources

| URI | Description |
|-----|-------------|
| `training://status` | Live status + latest metrics |
| `training://config` | Full hyperparameter config |

## How It Works

1. Your training code imports `tracker` and logs metrics each step
2. The `cortex` MCP server runs alongside, reading from the same tracker
3. An AI agent connects via MCP and can observe + intervene

The tracker is thread-safe. The MCP server and training loop can run in the same process (the server runs on a background thread) or separately (via shared state).

## What an agent can catch

- Entropy collapse (policy committing too early)
- Value loss dominating policy loss
- Gradient norm spikes before NaN divergence
- Learning rate too high/low based on loss trajectory
- Eval score plateaus suggesting convergence
- Cyclic loss patterns suggesting batch size issues

## Framework Support

Cortex is framework-agnostic. It works with:
- PyTorch
- MLX
- JAX
- TensorFlow
- Any Python training loop

See `examples/` for integration patterns.
