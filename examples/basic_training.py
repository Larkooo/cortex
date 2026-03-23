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


@tracker.on_eval
def run_eval(request):
    """Called when the agent requests an on-demand evaluation run."""
    episodes = int(request.get("episodes", 100))
    eval_score = 50 + tracker.get_status()["step"] * 0.005 + random.gauss(0, 1)
    result = {"episodes": episodes, "eval_score": round(eval_score, 3)}
    print(f"  [eval] {request.get('tag') or request['id']} -> {result}")
    tracker.log(step=tracker.get_status()["step"], eval_score=eval_score)
    return result


def fake_loss(step, lr):
    """Simulate a loss curve that responds to learning rate."""
    base = 2.0 * math.exp(-step * lr * 0.01)
    noise = random.gauss(0, 0.05)
    return max(0.01, base + noise)


def main():
    lr = 3e-4
    ent_coef = 0.05
    vf_coef = 0.25
    reward_shaping_weight = 0.1
    total_steps = 10000

    tracker.config(
        total_steps=total_steps,
        lr=lr,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        reward_shaping_weight=reward_shaping_weight,
        batch_size=64,
        model="simple_mlp",
        optimizer="adam",
    )
    tracker.define_params(
        {"name": "lr", "description": "Optimizer learning rate", "min_value": 1e-6, "max_value": 1e-2, "max_change_pct": 50},
        {"name": "ent_coef", "description": "Exploration coefficient", "min_value": 0.0, "max_value": 1.0, "max_change_pct": 50},
        {"name": "vf_coef", "description": "Value-loss coefficient", "min_value": 0.0, "max_value": 2.0, "max_change_pct": 50},
        {"name": "reward_shaping_weight", "description": "Reward shaping weight", "min_value": 0.0, "max_value": 1.0, "max_change_pct": 50},
    )

    print("Training started. Connect an agent to 'cortex' MCP server to monitor.")
    print(f"Steps: {total_steps}, LR: {lr}")
    print()
    print("The agent can:")
    print("  - get_metrics / get_metric_history   → observe")
    print("  - get_tunable_params()               → discover allowed knobs")
    print("  - adjust_param(name, value, reason)  → tune one knob live")
    print("  - save_checkpoint('good_state')      → save")
    print("  - run_eval(episodes=100)             → measure change")
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

        new_ent_coef = tracker.get_override("ent_coef")
        if new_ent_coef is not None:
            print(f"  [adjust] ent_coef: {ent_coef} → {new_ent_coef}")
            ent_coef = new_ent_coef

        new_vf_coef = tracker.get_override("vf_coef")
        if new_vf_coef is not None:
            print(f"  [adjust] vf_coef: {vf_coef} → {new_vf_coef}")
            vf_coef = new_vf_coef

        new_reward_shaping_weight = tracker.get_override("reward_shaping_weight")
        if new_reward_shaping_weight is not None:
            print(
                "  [adjust] reward_shaping_weight: "
                f"{reward_shaping_weight} → {new_reward_shaping_weight}"
            )
            reward_shaping_weight = new_reward_shaping_weight

        # Simulate training
        loss = fake_loss(step, lr)
        entropy = max(0.1, 2.0 - step * 0.0002 - ent_coef * 0.2 + random.gauss(0, 0.02))
        model_weights["w"] += lr * random.gauss(0, 1)

        tracker.log(
            step=step,
            loss=loss,
            entropy=entropy,
            lr=lr,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            reward_shaping_weight=reward_shaping_weight,
            reward=1.0 - reward_shaping_weight * 0.1 + random.gauss(0, 0.05),
            pg_loss=max(0.001, 0.02 + ent_coef * 0.1 + random.gauss(0, 0.005)),
            vf_loss=max(0.001, 0.3 * vf_coef + random.gauss(0, 0.02)),
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
