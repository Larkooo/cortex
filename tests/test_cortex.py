import sys
import threading
import time
import types
import unittest

sys.path.insert(0, "src")

mcp_module = types.ModuleType("mcp")
server_module = types.ModuleType("mcp.server")
stdio_module = types.ModuleType("mcp.server.stdio")
types_module = types.ModuleType("mcp.types")


class _DecoratorServer:
    def __init__(self, *_args, **_kwargs):
        pass

    def list_tools(self):
        return lambda fn: fn

    def call_tool(self):
        return lambda fn: fn

    def list_resources(self):
        return lambda fn: fn

    def read_resource(self):
        return lambda fn: fn

    def create_initialization_options(self):
        return {}

    async def run(self, *_args, **_kwargs):
        return None


class _SimpleType:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


server_module.Server = _DecoratorServer
stdio_module.stdio_server = None
types_module.TextContent = _SimpleType
types_module.Tool = _SimpleType
types_module.Resource = _SimpleType
sys.modules.setdefault("mcp", mcp_module)
sys.modules.setdefault("mcp.server", server_module)
sys.modules.setdefault("mcp.server.stdio", stdio_module)
sys.modules.setdefault("mcp.types", types_module)

from cortex.guardrails import Guardrails
from cortex.server import _handle_tool
from cortex.tracker import Tracker


class TrackerTests(unittest.TestCase):
    def test_log_auto_increments_when_step_is_omitted(self):
        tracker = Tracker()
        tracker.log(loss=1.0)
        tracker.log(loss=0.5)

        history = tracker.get_history("loss")
        self.assertEqual([entry["step"] for entry in history], [1, 2])

    def test_rollback_runs_while_paused(self):
        tracker = Tracker()
        called = []

        @tracker.on_rollback
        def rollback(tag):
            called.append(tag)
            return True

        tracker.checkpoint("safe")
        tracker.set_override("__pause__", True)

        def run_poll():
            tracker.poll()

        thread = threading.Thread(target=run_poll)
        thread.start()
        time.sleep(0.15)
        tracker.set_override("__rollback__", "safe")
        time.sleep(0.15)
        tracker.set_override("__resume__", True)
        thread.join(timeout=2)

        self.assertEqual(called, ["safe"])

    def test_eval_request_is_recorded(self):
        tracker = Tracker()

        @tracker.on_eval
        def run_eval(request):
            return {"episodes": request["episodes"], "eval_score": 12.5}

        tracker.set_override("__run_eval__", {"id": "eval-1", "episodes": 25})
        events = tracker.poll()

        self.assertEqual(events["eval_completed"], "eval-1")
        self.assertEqual(tracker.get_eval_runs()[-1]["result"]["eval_score"], 12.5)


class GuardrailTests(unittest.TestCase):
    def test_zero_value_param_is_tracked_in_guardrails(self):
        tracker = Tracker()
        tracker.checkpoint("safe")
        tracker.define_param(name="ent_coef", description="Entropy coefficient", min_value=0.0, max_value=1.0)
        tracker.log(step=1, ent_coef=0.0)
        guardrails = Guardrails(tracker)

        result = guardrails.validate_adjustment("ent_coef", 0.5)

        self.assertIn("old_value", result)
        self.assertEqual(result["old_value"], 0.0)

    def test_unregistered_param_is_blocked(self):
        tracker = Tracker()
        tracker.checkpoint("safe")
        guardrails = Guardrails(tracker)

        result = guardrails.validate_adjustment("dropout", 0.2)

        self.assertFalse(result["allowed"])
        self.assertIn("allowed_params", result)


class ServerTests(unittest.TestCase):
    def setUp(self):
        from cortex import server as server_module

        server_module.tracker.reset()
        server_module.guardrails.log.clear()
        server_module.guardrails._last_adjustment.clear()

    def test_adjust_param_blocks_unknown_name(self):
        from cortex import server as server_module

        server_module.tracker.define_param(name="lr", description="Learning rate")
        result = _handle_tool("adjust_param", {"name": "clip_eps", "value": 0.1, "reason": "test"})

        self.assertEqual(result["status"], "blocked")
        self.assertIn("allowed_params", result)

    def test_get_tunable_params_lists_safe_knobs(self):
        from cortex import server as server_module

        server_module.tracker.define_param(name="lr", description="Learning rate")
        server_module.tracker.define_param(name="dropout", description="Dropout", min_value=0.0, max_value=0.8)
        result = _handle_tool("get_tunable_params", {})

        names = [entry["name"] for entry in result["params"]]
        self.assertEqual(names, ["lr", "dropout"])


if __name__ == "__main__":
    unittest.main()
