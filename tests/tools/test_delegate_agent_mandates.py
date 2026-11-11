"""Canonical TOML mandates on the existing registry-dispatched delegate_task path.

No inference: only the AIAgent transport boundary is stubbed; routing, validation,
child prompt/toolset construction and host-owned lifecycle are real.
"""

import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools import delegate_tool as dt
from tools.registry import registry


@pytest.fixture
def dispatch_env(tmp_path, monkeypatch):
    from run_agent import AIAgent

    native_dispatch = AIAgent._dispatch_delegate_task
    root = tmp_path / "agents"
    root.mkdir()
    config = {"agents_dir": str(root), "max_spawn_depth": 2}
    monkeypatch.setattr(dt, "_load_config", lambda: config)
    monkeypatch.setattr(dt, "_resolve_workspace_hint", lambda parent: str(tmp_path))
    parent = SimpleNamespace(
        _delegate_depth=1, _active_children=[], _active_children_lock=threading.Lock(),
        _session_db=None, enabled_toolsets=["terminal", "file", "delegation"],
        disabled_toolsets=[], request_overrides={}, provider="custom",
        base_url="https://fixture.invalid/v1", api_key="fixture-only",
        api_mode="chat_completions", model="parent-model", platform="cli",
        session_id="fixture-parent", _print_fn=None, tool_progress_callback=None,
        thinking_callback=None, _delegate_spinner=None,
    )
    constructed = []

    def construct(**kwargs):
        constructed.append(kwargs)
        child = MagicMock()
        child._credential_pool = None
        child._delegate_saved_tool_names = []
        child.session_prompt_tokens = child.session_completion_tokens = 0
        child.model = kwargs["model"]
        child.run_conversation.return_value = {
            "final_response": "done", "completed": True, "api_calls": 0, "messages": [],
        }
        return child

    monkeypatch.setattr("run_agent.AIAgent", construct)

    def dispatch(tasks, **kwargs):
        entry = registry.get_entry("delegate_task")
        assert entry is not None
        return json.loads(entry.handler(
            {"tasks": tasks, **kwargs}, parent_agent=parent,
        ))

    return SimpleNamespace(root=root, config=config, parent=parent,
                           constructed=constructed, dispatch=dispatch, native_dispatch=native_dispatch)


def _role(root, name="coder", instructions="Write only the requested implementation.", extra=""):
    path = root / f"{name}.toml"
    path.write_text(
        f'name = {json.dumps(name)}\ndescription = "Implementation mandate"\n'
        f'developer_instructions = {json.dumps(instructions)}\n{extra}', encoding="utf-8",
    )
    return path


def test_selected_mandate_reaches_child_without_changing_route_or_capabilities(dispatch_env):
    env = dispatch_env
    mandate = "Follow this exact canonical mandate.\nDo not commit or push."
    _role(env.root, instructions=mandate, extra='model = "not-the-parent"\nsandbox_mode = "danger-full-access"\n')
    tasks = [{"goal": "Implement the agreed fixture", "context": "Keep caller context.", "agent": "coder"}]
    before = copy.deepcopy(tasks)
    result = env.dispatch(tasks)
    assert "error" not in result
    assert result["results"][0]["status"] == "completed"
    built, = env.constructed
    assert mandate in built["ephemeral_system_prompt"]
    assert "Keep caller context." in built["ephemeral_system_prompt"]
    assert built["model"] == "parent-model" and built["provider"] == "custom"
    assert built["base_url"] == env.parent.base_url and built["api_key"] == "fixture-only"
    assert "delegation" not in built["enabled_toolsets"]
    assert tasks == before


def test_agent_schema_is_opt_in_and_does_not_mutate_static_schema(dispatch_env):
    env = dispatch_env
    static = copy.deepcopy(dt.DELEGATE_TASK_SCHEMA)
    enabled = dt._build_dynamic_schema_overrides()["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert enabled["agent"]["type"] == "string"
    assert "routing" in enabled["agent"]["description"]
    env.config.pop("agents_dir")
    disabled = dt._build_dynamic_schema_overrides()["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "agent" not in disabled
    assert dt.DELEGATE_TASK_SCHEMA == static


@pytest.mark.parametrize("selector", ["", " coder", "CODER", "../coder", "/coder", "a/b", "a\\b", 4, {}, []])
def test_invalid_selector_refuses_before_routes_or_children(dispatch_env, monkeypatch, selector):
    env = dispatch_env
    route = MagicMock(side_effect=AssertionError("route resolution must not run"))
    monkeypatch.setattr(dt, "_routes_for", route)
    result = env.dispatch([{"goal": "Inspect this fixture", "agent": selector}])
    assert "Task 0" in result["error"]
    assert env.constructed == []
    route.assert_not_called()


@pytest.mark.parametrize("contents", [
    'name = "coder"\ndeveloper_instructions = [',
    'name = "someone-else"\ndeveloper_instructions = "Do the job"',
    'name = "coder"',
    'name = "coder"\ndeveloper_instructions = 4',
    'name = "coder"\ndeveloper_instructions = "  "',
    'name = "coder"\ndeveloper_instructions = "Do the job"\ndescription = 12',
])
def test_invalid_mandate_refuses_entire_batch(dispatch_env, monkeypatch, contents):
    env = dispatch_env
    _role(env.root, "verifier")
    (env.root / "coder.toml").write_text(contents, encoding="utf-8")
    transcripts = MagicMock(side_effect=AssertionError("transcripts must not be created"))
    monkeypatch.setattr("tools.delegation_live_log.create_live_transcripts", transcripts)
    result = env.dispatch([
        {"goal": "Verify this fixture", "agent": "verifier"},
        {"goal": "Implement this fixture", "agent": "coder"},
    ])
    assert "Task 1" in result["error"]
    assert env.constructed == []
    transcripts.assert_not_called()


@pytest.mark.parametrize("directory", [None, "", "  ", 123])
def test_unconfigured_directory_refuses_only_selected_tasks(dispatch_env, directory):
    env = dispatch_env
    env.config["agents_dir"] = directory
    refused = env.dispatch([{"goal": "Inspect this fixture", "agent": "coder"}])
    assert "delegation.agents_dir" in refused["error"] and env.constructed == []
    plain = env.dispatch([{"goal": "Inspect this fixture", "context": "Only caller context."}])
    assert plain["results"][0]["status"] == "completed"
    assert "Canonical agent mandate" not in env.constructed[0]["ephemeral_system_prompt"]


def test_missing_agent_is_not_silently_replaced(dispatch_env):
    env = dispatch_env
    result = env.dispatch([{"goal": "Inspect this fixture", "agent": "missing"}])
    assert "missing" in result["error"] and env.constructed == []


@pytest.mark.platforms("posix")
def test_agent_symlink_cannot_escape_configured_directory(dispatch_env, tmp_path):
    env = dispatch_env
    outside = tmp_path / "outside"
    outside.mkdir()
    source = _role(outside)
    (env.root / "coder.toml").symlink_to(source)
    result = env.dispatch([{"goal": "Inspect this fixture", "agent": "coder"}])
    assert "Cannot load canonical agent" in result["error"] and env.constructed == []


def test_relative_directory_uses_parent_home_not_ambient_profile(dispatch_env, tmp_path, monkeypatch):
    env = dispatch_env
    ambient = tmp_path / "ambient"
    owner = tmp_path / "owner"
    for home in (ambient, owner):
        (home / "agents").mkdir(parents=True)
    _role(ambient / "agents", instructions="Wrong ambient profile.")
    _role(owner / "agents", instructions="Correct owning profile.")
    env.config["agents_dir"] = "agents"
    env.parent._session_db = SimpleNamespace(db_path=owner / "state.db")
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    monkeypatch.setattr(dt, "_open_child_session_db", lambda parent: None)
    result = env.dispatch([{"goal": "Inspect this fixture", "agent": "coder"}])
    assert "error" not in result
    prompt = env.constructed[0]["ephemeral_system_prompt"]
    assert "Correct owning profile." in prompt and "Wrong ambient profile." not in prompt


def test_routing_gets_original_context_and_cannot_redefine_agent(dispatch_env, monkeypatch):
    from tools import delegate_tool_routes as routes

    env = dispatch_env
    _role(env.root)
    payloads = []
    backend = SimpleNamespace(
        name="probe", description=lambda cfg: "",
        task_parameters=lambda cfg: {"agent": {"type": "string", "enum": ["override"]}},
        route_task=lambda task, cfg: payloads.append(task) or {},
    )
    monkeypatch.setattr(routes, "_active_provider", lambda cfg: backend)
    result = env.dispatch([{"goal": "Inspect this fixture", "context": "Route on this only.", "agent": "coder"}])
    assert "error" not in result
    assert payloads == [{"goal": "Inspect this fixture", "context": "Route on this only.", "index": 0}]
    props = dt._build_dynamic_schema_overrides()["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "enum" not in props["agent"]


def test_mandate_snapshot_does_not_reread_changed_source_during_build(dispatch_env, monkeypatch):
    env = dispatch_env
    _role(env.root, instructions="Original frozen mandate.")
    original = dt._routes_for

    def routes(*args):
        _role(env.root, instructions="Changed after admission.")
        return original(*args)

    monkeypatch.setattr(dt, "_routes_for", routes)
    result = env.dispatch([{"goal": "Inspect this fixture", "agent": "coder"}])
    assert "error" not in result
    prompt = env.constructed[0]["ephemeral_system_prompt"]
    assert "Original frozen mandate." in prompt and "Changed after admission." not in prompt


def test_mandates_coexist_with_output_schema_images_and_plain_sibling(dispatch_env):
    env = dispatch_env
    _role(env.root)
    result = env.dispatch([
        {"goal": "Inspect this fixture", "agent": "coder", "images": ["https://fixture.invalid/image.png"],
         "output_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}},
        {"goal": "Inspect the plain sibling", "context": "Unchanged plain context."},
    ])
    assert "error" not in result
    selected, plain = env.constructed
    assert "Canonical agent mandate: coder" in selected["ephemeral_system_prompt"]
    assert '"ok"' in selected["ephemeral_system_prompt"]
    assert result["results"][0]["schema_valid"] is False
    assert "Unchanged plain context." in plain["ephemeral_system_prompt"]
    assert "Canonical agent mandate" not in plain["ephemeral_system_prompt"]


def test_managed_worker_refuses_unfrozen_mandate_reads(dispatch_env, monkeypatch):
    env = dispatch_env
    _role(env.root)
    monkeypatch.setattr("agent.safe_worker_policy.worker_config_snapshot", dict)
    result = env.dispatch([{"goal": "Inspect this fixture", "agent": "coder"}])
    assert "managed worker" in result["error"] and env.constructed == []


def test_semantic_role_cannot_change_depth_granted_orchestration(dispatch_env):
    env = dispatch_env
    _role(env.root)
    env.config["max_spawn_depth"] = 3
    result = env.dispatch([{"goal": "Inspect this fixture", "agent": "coder"}])
    assert "error" not in result
    assert "delegation" in env.constructed[0]["enabled_toolsets"]
    assert "CAN spawn your own subagents" in env.constructed[0]["ephemeral_system_prompt"]


def test_native_agent_dispatch_forwards_canonical_selector(dispatch_env):
    env = dispatch_env
    _role(env.root, instructions="Native dispatch mandate.")
    result = json.loads(env.native_dispatch(env.parent, {
        "tasks": [{"goal": "Inspect the native fixture", "agent": "coder"}],
    }))
    assert result["results"][0]["status"] == "completed"
    assert "Native dispatch mandate." in env.constructed[0]["ephemeral_system_prompt"]


def test_directory_can_be_set_through_existing_config_cli(dispatch_env, tmp_path, monkeypatch):
    from hermes_cli.config import load_config_readonly, set_config_value

    home = tmp_path / "cli-home"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    set_config_value("delegation.agents_dir", str(dispatch_env.root))
    assert load_config_readonly()["delegation"]["agents_dir"] == str(dispatch_env.root)
