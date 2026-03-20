"""
Cortex MCP Server — exposes training metrics and controls to AI agents.

Tools:
    get_status          - current step, progress, ETA, phase
    get_metrics         - latest values of all tracked metrics
    get_metric_history  - time series for a specific metric
    get_config          - training configuration/hyperparameters
    list_checkpoints    - saved checkpoints
    adjust_param        - override a hyperparameter mid-training
    save_checkpoint     - tell the training loop to save a checkpoint

Resources:
    training://status   - live training status
    training://config   - hyperparameter configuration
"""

import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
    Resource,
)

from cortex.tracker import tracker

logger = logging.getLogger("cortex")

server = Server("cortex")


# ──────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_status",
            description="Get current training status: step, progress percentage, ETA, steps/sec, phase",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_metrics",
            description="Get the latest value of all tracked metrics (loss, entropy, eval scores, etc.)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_metric_history",
            description="Get the time series history for a specific metric. Use this to analyze trends.",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "Metric name (e.g., 'loss', 'entropy', 'eval_score')"},
                    "last_n": {"type": "integer", "description": "Only return the last N data points. 0 = all.", "default": 50},
                },
                "required": ["metric"],
            },
        ),
        Tool(
            name="get_config",
            description="Get training configuration and hyperparameters (lr, batch_size, etc.)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_metrics",
            description="List all metric names being tracked",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_checkpoints",
            description="List all saved checkpoints with their step, time, and metrics at save time",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="adjust_param",
            description="Override a hyperparameter mid-training. The training loop will pick this up on the next step. Common params: lr, ent_coef, vf_coef, clip_eps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Parameter name (e.g., 'lr', 'ent_coef')"},
                    "value": {"type": "number", "description": "New value"},
                },
                "required": ["name", "value"],
            },
        ),
        Tool(
            name="save_checkpoint",
            description="Request the training loop to save a checkpoint with the given tag",
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "Checkpoint tag/name"},
                },
                "required": ["tag"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_status":
        result = tracker.get_status()
    elif name == "get_metrics":
        result = tracker.get_latest()
    elif name == "get_metric_history":
        metric = arguments["metric"]
        last_n = arguments.get("last_n", 50)
        history = tracker.get_history(metric, last_n)
        if not history:
            available = tracker.get_metric_names()
            result = {"error": f"Metric '{metric}' not found", "available_metrics": available}
        else:
            result = {"metric": metric, "count": len(history), "data": history}
    elif name == "get_config":
        result = tracker.get_config()
    elif name == "list_metrics":
        result = {"metrics": tracker.get_metric_names()}
    elif name == "list_checkpoints":
        result = tracker.get_checkpoints()
    elif name == "adjust_param":
        param_name = arguments["name"]
        value = arguments["value"]
        tracker.set_override(param_name, value)
        result = {"status": "queued", "param": param_name, "value": value, "note": "Will take effect on next training step"}
    elif name == "save_checkpoint":
        tag = arguments["tag"]
        tracker.set_override("__save_checkpoint__", tag)
        result = {"status": "queued", "tag": tag}
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ──────────────────────────────────────────────────────────────
# Resources
# ──────────────────────────────────────────────────────────────

@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="training://status",
            name="Training Status",
            description="Live training status including step, progress, ETA",
            mimeType="application/json",
        ),
        Resource(
            uri="training://config",
            name="Training Config",
            description="Hyperparameter configuration",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str):
    if uri == "training://status":
        status = tracker.get_status()
        status["latest_metrics"] = tracker.get_latest()
        return json.dumps(status, indent=2, default=str)
    elif uri == "training://config":
        return json.dumps(tracker.get_config(), indent=2, default=str)
    else:
        return json.dumps({"error": f"Unknown resource: {uri}"})


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    main()
