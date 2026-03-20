"""
Example: integrating Cortex with a basic training loop.

Shows: logging metrics, handling param overrides, checkpoints, and rollback.
An AI agent connected via MCP can monitor and adjust parameters in real time.

Run the training:
    python examples/basic_training.py

Connect an agent to the MCP server (in another terminal or via Claude Desktop).
"""

import math
import time
import random

from cortex import tracker


# Simulated model state
model_weights = {"w": 1.0}
saved_weights = {}


@tracker.on_checkpoint
def save(tag):
    """Called when the agent requests a checkpoint save."""
    saved_weights[tag] = dict(model_weights)
    print(f"  [checkpoint] saved '{tag}' at step {tracker._step}")


@tracker.on_rollback
def rollback(tag):
    """Called when the agent requests a rollback."""
    if tag in saved_weights:
        model_weights.update(saved_weights[tag])
        print(f"  [rollback] restored '{tag}'")
        return True
    print(f"  [rollback] '{tag}' not found!")
    return False


def fake_loss(step, lr):
    """Simulate a loss curve that responds to learning rate."""
    base = 2.0 * math.exp(-step * lr * 0.01)
    noise = random.gauss(0, 0.05)
    return max(0.01, base + noise)


def main():
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
    print()
    print("The agent can:")
    print("  - get_metrics / get_metric_history  → observe")
    print("  - adjust_param('lr', 1e-3)          → tune live")
    print("  - save_checkpoint('good_state')      → save")
    print("  - rollback('good_state')             → restore")
    print("  - pause_training / resume_training   → pause to think")
    print()

    for step in range(total_steps):
        tracker.phase("train")

        # Process agent commands (checkpoints, rollbacks, pause/resume)
        events = tracker.poll()
        for event, value in events.items():
            print(f"  [{event}] {value}")

        # Check for live param adjustments
        new_lr = tracker.get_override("lr")
        if new_lr is not None:
            print(f"  [adjust] lr: {lr} → {new_lr}")
            lr = new_lr

        # Simulate training
        loss = fake_loss(step, lr)
        entropy = max(0.1, 2.0 - step * 0.0002 + random.gauss(0, 0.02))
        model_weights["w"] += lr * random.gauss(0, 1)

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
            print(f"  step {step:>5d} | loss {loss:.4f} | ent {entropy:.3f} | eval {eval_score:.1f}")

        time.sleep(0.001)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
