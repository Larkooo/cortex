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
# tracker.config(
#     total_steps=10000,
#     lr=3e-4,
#     model="mlp_784_256_10",
#     optimizer="adam",
#     framework="pytorch",
# )
#
# for step, (x, y) in enumerate(dataloader):
#     # Check for agent overrides
#     new_lr = tracker.get_override("lr")
#     if new_lr is not None:
#         for pg in optimizer.param_groups:
#             pg["lr"] = new_lr
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
#     )
#
#     optimizer.step()
#
#     # Checkpoint on request
#     tag = tracker.get_override("__save_checkpoint__")
#     if tag:
#         torch.save(model.state_dict(), f"checkpoints/{tag}.pt")
#         tracker.checkpoint(tag, {"path": f"checkpoints/{tag}.pt"})

print("See comments in this file for PyTorch integration example.")
print("The pattern is the same for any framework:")
print("  1. tracker.config(...) at start")
print("  2. tracker.log(step=..., loss=...) each step")
print("  3. tracker.get_override('lr') to pick up agent adjustments")
