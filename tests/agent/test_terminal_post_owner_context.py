"""Terminal observer ownership survives a copied inner-dispatch ContextVar.

No model/API is used. Lifecycle is an observation sink; the emitters and their
suppression state are real. Gate/dispatcher integration is checked separately.
"""
import contextvars
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from agent.inline_tool_executors import emit_terminal_post_tool_call
from model_tools import _emit_post_tool_call_hook, _post_tool_call_hook_suppressed, suppress_post_tool_call_hook


class TerminalPostOwnerContextTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.agent = SimpleNamespace(session_id="fixture-session", _current_turn_id="fixture-turn",
                                     _current_api_request_id="fixture-api")
        self.kwargs: dict[str, Any] = dict(function_name="fixture_tool", function_args={"query": "fixture"},
                           result={"result": "explicit fixture"}, effective_task_id="fixture-task",
                           tool_call_id="fixture-call", status="ok")
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("hermes_cli.lifecycle.has_hook", return_value=True))
        stack.enter_context(patch("hermes_cli.lifecycle.invoke_hook", side_effect=lambda event, **kw: self.events.append((event, kw))))

    def assert_post(self, status="ok"):
        self.assertEqual(len(self.events), 1, "terminal Post vanished under inherited suppression")
        event, payload = self.events[0]
        self.assertEqual(event, "post_tool_call")
        for key, value in dict(session_id="fixture-session", turn_id="fixture-turn", tool_call_id="fixture-call",
                               task_id="fixture-task", api_request_id="fixture-api", status=status).items():
            self.assertEqual(payload[key], value)

    def test_terminal_owner_emits_once_in_copied_suppressed_context(self):
        with suppress_post_tool_call_hook():
            inherited = contextvars.copy_context()
        inherited.run(emit_terminal_post_tool_call, self.agent, **self.kwargs)
        self.assert_post()
        self.assertTrue(inherited.run(_post_tool_call_hook_suppressed.get))
        self.assertFalse(_post_tool_call_hook_suppressed.get())

    def test_normal_terminal_owner_emits_once(self):
        emit_terminal_post_tool_call(self.agent, **self.kwargs)
        self.assert_post()

    def test_inner_registry_emitter_stays_suppressed(self):
        with suppress_post_tool_call_hook():
            _emit_post_tool_call_hook(function_name="fixture", function_args={}, result={}, status="ok")
        self.assertEqual(self.events, [])

    def test_terminal_failure_is_not_promoted_to_success(self):
        kwargs = dict(self.kwargs, result={"error": "fixture failure"}, status="error",
                      error_type="tool_error", error_message="fixture failure")
        with suppress_post_tool_call_hook():
            emit_terminal_post_tool_call(self.agent, **kwargs)
        self.assert_post("error")
        self.assertEqual(self.events[0][1]["error_type"], "tool_error")
