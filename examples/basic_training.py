"""
Example: integrating Cortex with a basic training loop.

This shows the minimal integration — just import the tracker and log metrics.
An AI agent connected via MCP can then monitor and adjust parameters in real time.

Run the training:
    python examples/basic_training.py

In another terminal, connect an agent to the MCP server:
    cortex
"""

import math
import time
import random

from cortex import tracker


def fake_loss(step, lr):
    """Simulate a loss curve that responds to learning rate."""
    base = 2.0 * math.exp(-step * lr * 0.01)
    noise = random.gauss(0, 0.05)
    return max(0.01, base + noise)


def main():
    # Configure
    lr = 3e-4
    total_steps = 10000

    tracker.config(
        total_steps=total_steps,
        lr=lr,
        batch_size=64,
        model="simple_mlp",
        optimizer="adam",
    )

    print("Training started. Connect an agent to 'cortex' MCP server to monitor.")
    print(f"Steps: {total_steps}, LR: {lr}")

    for step in range(total_steps):
        tracker.phase("train")

        # Check for agent overrides
        new_lr = tracker.get_override("lr")
        if new_lr is not None:
            print(f"[step {step}] Agent adjusted LR: {lr} → {new_lr}")
            lr = new_lr

        # Check for checkpoint requests
        save_tag = tracker.get_override("__save_checkpoint__")
        if save_tag:
            print(f"[step {step}] Agent requested checkpoint: {save_tag}")
            tracker.checkpoint(save_tag)

        # Simulate training
        loss = fake_loss(step, lr)
        entropy = max(0.1, 2.0 - step * 0.0002 + random.gauss(0, 0.02))

        tracker.log(
            step=step,
            loss=loss,
            entropy=entropy,
            lr=lr,
            grad_norm=random.uniform(0.1, 1.5),
        )

        # Periodic eval
        if step % 500 == 0 and step > 0:
            tracker.phase("eval")
            eval_score = 50 + step * 0.005 + random.gauss(0, 2)
            tracker.log(step=step, eval_score=eval_score)
            print(f"  step {step:>5d} | loss {loss:.4f} | entropy {entropy:.3f} | eval {eval_score:.1f}")

        time.sleep(0.001)  # simulate compute time

    print("Training complete.")


if __name__ == "__main__":
    main()
