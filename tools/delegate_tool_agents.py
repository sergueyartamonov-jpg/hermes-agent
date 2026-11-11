"""Opt-in semantic mandates for delegate_task; routing and capabilities stay separate."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_AGENT_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")


@dataclass(frozen=True)
class AgentMandate:
    name: str
    instructions: str


def _read_mandate(root: Path, name: Any) -> AgentMandate:
    if not isinstance(name, str) or _AGENT_NAME.fullmatch(name) is None:
        raise ValueError("'agent' must be a canonical agent name, not a path.")
    try:
        root = root.resolve(strict=True)
        path = (root / f"{name}.toml").resolve(strict=True)
        if path.parent != root or not path.is_file():
            raise ValueError("agent file must stay inside delegation.agents_dir.")
        with path.open("rb") as stream:
            spec = tomllib.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load canonical agent '{name}' from delegation.agents_dir.") from exc
    instructions = spec.get("developer_instructions")
    if spec.get("name") != name or not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"Canonical agent '{name}' needs its matching name and non-empty developer_instructions.")
    if "description" in spec and not isinstance(spec["description"], str):
        raise ValueError(f"Canonical agent '{name}' description must be a string.")
    # Other TOML keys (model, sandbox, tools, ...) never become runtime settings.
    return AgentMandate(name, instructions)


def resolve_task_agents(tasks: list[dict], config: Mapping[str, Any], home: Path | None = None
                        ) -> tuple[list[AgentMandate | None], str | None]:
    """Freeze selected files before routes, transcripts or children; omitted selectors perform no I/O."""
    if not any(task.get("agent") is not None for task in tasks):
        return [None] * len(tasks), None
    from agent.safe_worker_policy import worker_config_snapshot
    if worker_config_snapshot() is not None:
        return [], "Canonical agent files are unavailable in a managed worker (its session config is frozen)."
    directory = config.get("agents_dir")
    if not isinstance(directory, str) or not directory.strip():
        return [], "Canonical agent selection requires delegation.agents_dir in config.yaml."
    root = Path(directory).expanduser()
    if not root.is_absolute():
        from hermes_constants import get_hermes_home
        root = (home if home is not None else get_hermes_home()) / root
    resolved = []
    by_name = {}
    for index, task in enumerate(tasks):
        name = task.get("agent")
        if name is None:
            resolved.append(None)
            continue
        try:
            # Validate before indexing: model input may be a list/dict, not a hashable name.
            if not isinstance(name, str):
                raise ValueError("'agent' must be a canonical agent name, not a path.")
            if name not in by_name:
                by_name[name] = _read_mandate(root, name)
            resolved.append(by_name[name])
        except ValueError as exc:
            return [], f"Task {index}: {exc}"
    return resolved, None


def append_agent_mandate(context: str | None, mandate: AgentMandate | None) -> str | None:
    """Augment only the new child's existing prompt-context channel, not its provider's routing input."""
    if mandate is None:
        return context
    block = f"## Canonical agent mandate: {mandate.name}\n{mandate.instructions}"
    return f"{block}\n\n{context}" if context else block
