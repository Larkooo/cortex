"""
Example: integrating Cortex with a real PyTorch training loop.

Shows how to:
1. Log metrics from your training loop
2. Let the agent adjust learning rate mid-training
3. Save checkpoints on agent request

    pip install torch cortex-mcp
    python examples/pytorch_integration.py
"""

# Uncomment below for a real PyTorch integration:
#
# import torch
# import torch.nn as nn
# from cortex import tracker
#
# model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# criterion = nn.CrossEntropyLoss()
#
# ent_coef = 0.05
# vf_coef = 0.25
# reward_shaping_weight = 0.1
#
# tracker.config(
#     total_steps=10000,
#     lr=3e-4,
#     ent_coef=ent_coef,
#     vf_coef=vf_coef,
#     reward_shaping_weight=reward_shaping_weight,
#     model="mlp_784_256_10",
#     optimizer="adam",
#     framework="pytorch",
# )
#
# tracker.define_params(
#     {"name": "lr", "description": "Optimizer learning rate", "min_value": 1e-6, "max_value": 1e-2, "max_change_pct": 50},
#     {"name": "weight_decay", "description": "Optimizer weight decay", "min_value": 0.0, "max_value": 0.1, "max_change_pct": 50},
#     {"name": "dropout", "description": "Model dropout", "min_value": 0.0, "max_value": 0.8, "max_change_pct": 25},
#     {"name": "label_smoothing", "description": "Cross-entropy label smoothing", "min_value": 0.0, "max_value": 0.3, "max_change_pct": 50},
# )
#
# @tracker.on_eval
# def run_eval(request):
#     episodes = int(request.get("episodes", 100))
#     score = evaluate_policy(model, eval_env, episodes=episodes)
#     tracker.log(step=step, eval_score=score)
#     return {"episodes": episodes, "eval_score": score}
#
# for step, (x, y) in enumerate(dataloader):
#     # Check for registered runtime knobs
#     new_lr = tracker.get_override("lr")
#     if new_lr is not None:
#         for pg in optimizer.param_groups:
#             pg["lr"] = new_lr
#
#     new_weight_decay = tracker.get_override("weight_decay")
#     if new_weight_decay is not None:
#         for pg in optimizer.param_groups:
#             pg["weight_decay"] = new_weight_decay
#
#     new_dropout = tracker.get_override("dropout")
#     if new_dropout is not None:
#         model.dropout.p = new_dropout
# 
#     new_label_smoothing = tracker.get_override("label_smoothing")
#     if new_label_smoothing is not None:
#         criterion = nn.CrossEntropyLoss(label_smoothing=new_label_smoothing)
#
#     # Forward + backward
#     pred = model(x)
#     loss = criterion(pred, y)
#     optimizer.zero_grad()
#     loss.backward()
#
#     # Log metrics
#     grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
#     tracker.log(
#         step=step,
#         loss=loss.item(),
#         grad_norm=grad_norm,
#         lr=optimizer.param_groups[0]["lr"],
#         weight_decay=optimizer.param_groups[0]["weight_decay"],
#         dropout=model.dropout.p,
#     )
#
#     events = tracker.poll()
#     if "eval_completed" in events:
#         print(f"Eval finished: {events['eval_completed']}")
#
#     optimizer.step()

print("See comments in this file for PyTorch integration example.")
print("The pattern is the same for any framework:")
print("  1. tracker.config(...) at start")
print("  2. tracker.define_param(...) for any runtime knob you want the agent to control")
print("  2. tracker.log(step=..., loss=...) each step")
print("  3. tracker.get_override(...) for the knobs your trainer registered")
print("  4. tracker.poll() to process checkpoint / rollback / eval requests")
