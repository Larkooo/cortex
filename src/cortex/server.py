"""
Cortex MCP Server — exposes training metrics, detectors, and guarded controls to AI agents.
"""

import json
import logging
import time

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool, Resource

from cortex.tracker import tracker
from cortex.detectors import run_all as run_detectors
from cortex.guardrails import Guardrails, GuardrailConfig

logger = logging.getLogger("cortex")
server = Server("cortex")
guardrails = Guardrails(tracker)


# ──────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        # ── Observe ──
        Tool(
            name="get_status",
            description="Get current training status: step, progress %, ETA, steps/sec, phase.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_metrics",
            description="Get the latest value of all tracked metrics (loss, entropy, eval scores, etc.).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_metric_history",
            description="Get time series history for a specific metric. Use this to analyze trends, detect patterns, and compare before/after an intervention.",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "Metric name (e.g., 'loss', 'entropy', 'eval_score')"},
                    "last_n": {"type": "integer", "description": "Return the last N data points (default: 50)", "default": 50},
                },
                "required": ["metric"],
            },
        ),
        Tool(
            name="get_config",
            description="Get training configuration and hyperparameters.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_tunable_params",
            description="List the runtime-tunable parameters declared by the training loop.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_metrics",
            description="List all metric names being tracked.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Detect ──
        Tool(
            name="diagnose",
            description="Run all anomaly detectors on the current training state. Returns structured findings with severity, explanation, and recommended action. Use this regularly to check training health.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Intervene ──
        Tool(
            name="adjust_param",
            description="Adjust a runtime-tunable parameter declared by the training loop. Subject to guardrails: registry, bounds, max % change, cooldown between adjustments, and checkpoint requirement. Always provide a reason.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Registered parameter name (for example 'lr', 'dropout', 'weight_decay', or 'label_smoothing')"},
                    "value": {"type": "number", "description": "New value"},
                    "reason": {"type": "string", "description": "Why you're making this change (logged for review)"},
                },
                "required": ["name", "value", "reason"],
            },
        ),
        Tool(
            name="save_checkpoint",
            description="Save a checkpoint of the current model state. Always do this before making adjustments, so you can rollback if things go wrong.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "Checkpoint name (e.g., 'before_lr_change', 'best_eval', 'healthy_entropy')"},
                },
                "required": ["tag"],
            },
        ),
        Tool(
            name="rollback",
            description="Restore model weights to a previously saved checkpoint. Use when training has diverged, entropy collapsed, or eval crashed. Weights are restored but step count continues.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "Checkpoint tag to restore (from list_checkpoints)"},
                    "reason": {"type": "string", "description": "Why you're rolling back"},
                },
                "required": ["tag", "reason"],
            },
        ),
        Tool(
            name="list_checkpoints",
            description="List all saved checkpoints with their step, metrics at save time, and metadata.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="pause_training",
            description="Pause the training loop. Use this when you need time to analyze metrics and plan interventions. Training blocks until you call resume_training.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="run_eval",
            description="Request an evaluation run from the training loop. Use this after a checkpoint or parameter change to decide whether to keep or rollback.",
            inputSchema={
                "type": "object",
                "properties": {
                    "episodes": {"type": "integer", "description": "Number of eval episodes to run", "default": 100},
                    "tag": {"type": "string", "description": "Short label for the eval run"},
                    "reason": {"type": "string", "description": "Why you're requesting this eval"},
                },
            },
        ),
        Tool(
            name="resume_training",
            description="Resume training after a pause.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Review ──
        Tool(
            name="get_intervention_log",
            description="Get the log of all interventions (param changes, rollbacks, checkpoints) with before/after metrics and reasons.",
            inputSchema={
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Return last N entries (default: all)", "default": 0},
                },
            },
        ),
        Tool(
            name="get_guardrail_status",
            description="Check current guardrail state: rate limits, cooldowns, intervention count.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_eval_runs",
            description="Get the recent on-demand evaluation runs requested through MCP.",
            inputSchema={
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Return last N eval runs (default: all)", "default": 0},
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    result = _handle_tool(name, arguments)
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


def _handle_tool(name: str, arguments: dict) -> dict:
    # ── Observe ──
    if name == "get_status":
        return tracker.get_status()

    elif name == "get_metrics":
        return tracker.get_latest()

    elif name == "get_metric_history":
        metric = arguments["metric"]
        last_n = arguments.get("last_n", 50)
        history = tracker.get_history(metric, last_n)
        if not history:
            return {"error": f"Metric '{metric}' not found", "available_metrics": tracker.get_metric_names()}
        return {"metric": metric, "count": len(history), "data": history}

    elif name == "get_config":
        return tracker.get_config()

    elif name == "get_tunable_params":
        return {"params": tracker.get_tunable_params()}

    elif name == "list_metrics":
        return {"metrics": tracker.get_metric_names()}

    # ── Detect ──
    elif name == "diagnose":
        findings = run_detectors(tracker)
        if not findings:
            return {"status": "healthy", "findings": [], "message": "No anomalies detected."}
        return {
            "status": "issues_detected",
            "count": len(findings),
            "findings": [f.to_dict() for f in findings],
        }

    # ── Intervene ──
    elif name == "adjust_param":
        return _adjust_param(arguments["name"], arguments["value"], arguments.get("reason", ""))

    elif name == "save_checkpoint":
        tag = arguments["tag"]
        tracker.set_override("__save_checkpoint__", tag)
        guardrails.record_intervention(action="save_checkpoint", param=tag, reason="agent requested")
        return {"status": "queued", "tag": tag}

    elif name == "rollback":
        tag = arguments["tag"]
        reason = arguments.get("reason", "")
        checkpoints = tracker.get_checkpoints()
        if tag not in checkpoints:
            return {"error": f"Checkpoint '{tag}' not found", "available": list(checkpoints.keys())}

        tracker.set_override("__rollback__", tag)
        cp = checkpoints[tag]
        guardrails.record_intervention(
            action="rollback", param=tag,
            old_value=tracker.get_status()["step"], new_value=cp["step"], reason=reason,
        )
        return {
            "status": "queued", "tag": tag,
            "restoring_to_step": cp["step"],
            "metrics_at_checkpoint": cp["metrics"],
            "reason": reason,
        }

    elif name == "list_checkpoints":
        return tracker.get_checkpoints()

    elif name == "pause_training":
        tracker.set_override("__pause__", True)
        guardrails.record_intervention(action="pause", reason="agent requested")
        return {"status": "paused"}

    elif name == "run_eval":
        eval_request = {
            "id": f"eval-{int(time.time() * 1000)}",
            "episodes": arguments.get("episodes", 100),
            "tag": arguments.get("tag", ""),
            "reason": arguments.get("reason", ""),
        }
        tracker.set_override("__run_eval__", eval_request)
        guardrails.record_intervention(
            action="run_eval",
            param=eval_request["tag"] or eval_request["id"],
            new_value=eval_request["episodes"],
            reason=eval_request["reason"] or "agent requested",
        )
        return {"status": "queued", "request": eval_request}

    elif name == "resume_training":
        tracker.set_override("__resume__", True)
        guardrails.record_intervention(action="resume", reason="agent requested")
        return {"status": "resumed"}

    # ── Review ──
    elif name == "get_intervention_log":
        last_n = arguments.get("last_n", 0)
        return {"interventions": guardrails.get_log(last_n)}

    elif name == "get_guardrail_status":
        return guardrails.get_summary()

    elif name == "get_eval_runs":
        last_n = arguments.get("last_n", 0)
        return {"eval_runs": tracker.get_eval_runs(last_n)}

    else:
        return {"error": f"Unknown tool: {name}"}


def _adjust_param(param_name: str, value: float, reason: str) -> dict:
    check = guardrails.validate_adjustment(param_name, value, reason)
    if not check["allowed"]:
        result = {"status": "blocked", "error": check["error"]}
        if "allowed_params" in check:
            result["allowed_params"] = check["allowed_params"]
        return result

    guardrails.record_intervention(
        action="adjust_param", param=param_name,
        old_value=check.get("old_value"), new_value=value, reason=reason,
    )
    tracker.set_override(param_name, value)

    result = {
        "status": "applied",
        "param": param_name,
        "old_value": check.get("old_value"),
        "new_value": value,
        "reason": reason,
    }
    if check.get("warnings"):
        result["warnings"] = check["warnings"]
    return result


# ──────────────────────────────────────────────────────────────
# Resources
# ──────────────────────────────────────────────────────────────

@server.list_resources()
async def list_resources():
    return [
        Resource(uri="training://status", name="Training Status",
                 description="Live training status + latest metrics", mimeType="application/json"),
        Resource(uri="training://config", name="Training Config",
                 description="Hyperparameter configuration", mimeType="application/json"),
        Resource(uri="training://diagnosis", name="Training Diagnosis",
                 description="Current anomaly detection findings", mimeType="application/json"),
    ]


@server.read_resource()
async def read_resource(uri: str):
    if uri == "training://status":
        status = tracker.get_status()
        status["latest_metrics"] = tracker.get_latest()
        return json.dumps(status, indent=2, default=str)
    elif uri == "training://config":
        return json.dumps(tracker.get_config(), indent=2, default=str)
    elif uri == "training://diagnosis":
        findings = run_detectors(tracker)
        return json.dumps({"findings": [f.to_dict() for f in findings]}, indent=2, default=str)
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
