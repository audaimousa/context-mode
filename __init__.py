"""Hermes Agent adapter for context-mode's existing hooks and MCP server.

This modified integration is intentionally thin: Hermes keeps tool execution,
approvals, and profile scoping; context-mode supplies routing, indexing, search,
and diagnostics through its public CLI and MCP tools.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import uuid
from typing import Any


__version__ = "1.0.169"

_TIMEOUT = 2.0
_INDEX_TIMEOUT = 3.0
_MAX_CAPTURE = 2 * 1024 * 1024
_INDEX_THRESHOLD = 16 * 1024
_CTX_PREFIX = "mcp__context_mode__ctx_"
_TOOL_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "terminal": ("Bash", {"command": "command"}),
    "read_file": ("Read", {"path": "file_path"}),
    "search_files": ("Grep", {"pattern": "pattern"}),
    "web_extract": ("WebFetch", {"urls": "url"}),
}
_RESULT_TOOLS = frozenset({
    "read_file",
    "search_files",
    "web_extract",
    "web_search",
    "browser_snapshot",
    "browser_console",
    "browser_extract",
})
_index_lock = threading.Lock()
_ctx: Any = None


def _sid(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("session_id") or kwargs.get("task_id") or "hermes")


def _project(kwargs: dict[str, Any]) -> str:
    return os.path.realpath(str(kwargs.get("project_dir") or kwargs.get("cwd") or os.getcwd()))


def _profile_home() -> str:
    plugin_dir = Path(__file__).resolve().parent
    if plugin_dir.parent.name == "plugins":
        return str(plugin_dir.parent.parent)
    configured = os.environ.get("HERMES_HOME")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(Path.home() / ".hermes")


def _map_to_context_mode(tool_name: str, args: Any) -> tuple[str, dict[str, Any]]:
    source = args if isinstance(args, dict) else {}
    canonical, names = _TOOL_MAP.get(tool_name, (tool_name, {}))
    mapped = {names.get(key, key): value for key, value in source.items()}
    if tool_name == "web_extract" and isinstance(mapped.get("url"), list):
        mapped["url"] = next((url for url in mapped["url"] if isinstance(url, str) and url), "")
    return canonical, mapped


def _map_to_hermes(tool_name: str, args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    _, names = _TOOL_MAP.get(tool_name, (tool_name, {}))
    reverse = {canonical: hermes for hermes, canonical in names.items()}
    mapped = {reverse.get(key, key): value for key, value in args.items()}
    if tool_name == "web_extract" and isinstance(mapped.get("urls"), str):
        mapped["urls"] = [mapped["urls"]]
    return mapped


def _run_hook(event: str, payload: dict[str, Any], timeout: float = _TIMEOUT) -> dict[str, Any] | None:
    executable = os.environ.get("CONTEXT_MODE_EXECUTABLE") or shutil.which("context-mode")
    if not executable:
        return None
    env = os.environ.copy()
    env["CONTEXT_MODE_PLATFORM"] = "hermes"
    env["HERMES_HOME"] = _profile_home()
    env["CONTEXT_MODE_PROJECT_DIR"] = str(payload.get("cwd") or os.getcwd())
    try:
        proc = subprocess.run(
            [executable, "hook", "hermes", event],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        if proc.returncode != 0 or len(proc.stdout) > _MAX_CAPTURE:
            return None
        output = proc.stdout.strip()
        parsed = json.loads(output) if output else {}
        return parsed if isinstance(parsed, dict) else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _pre_tool_call(tool_name: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any] | None:
    if tool_name in {_CTX_PREFIX + "index", _CTX_PREFIX + "search"}:
        project_id = sha256(_project(kwargs).encode()).hexdigest()[:12]
        source = str(args.get("source") or "manual")
        if f":{project_id}:" not in source:
            scoped = dict(args)
            scoped["source"] = (
                project_id if tool_name.endswith("ctx_search")
                else f"hermes:manual:{project_id}:{source}"
            )
            return {"action": "modify", "args": scoped}
    canonical, mapped_args = _map_to_context_mode(tool_name, args)
    response = _run_hook("pretooluse", {
        "tool_name": canonical,
        "tool_input": mapped_args,
        "session_id": _sid(kwargs),
        "cwd": _project(kwargs),
    })
    if not response:
        return None
    action = response.get("action")
    if action == "modify":
        modified = _map_to_hermes(tool_name, response.get("args"))
        return {"action": "modify", "args": modified} if modified else None
    if action in {"block", "approve"}:
        return {"action": action, "message": str(response.get("message") or "context-mode routing")}
    return None


def _post_tool_call(tool_name: str, args: dict[str, Any], result: Any, **kwargs: Any) -> None:
    canonical, mapped_args = _map_to_context_mode(tool_name, args)
    _run_hook("posttooluse", {
        "tool_name": canonical,
        "tool_input": mapped_args,
        "tool_response": result,
        "session_id": _sid(kwargs),
        "cwd": _project(kwargs),
    })


def _session_start(**kwargs: Any) -> None:
    _run_hook("sessionstart", {
        "source": "startup",
        "session_id": _sid(kwargs),
        "cwd": _project(kwargs),
    })


def _session_boundary(**kwargs: Any) -> None:
    _run_hook("stop", {"session_id": _sid(kwargs), "cwd": _project(kwargs)})


def _dispatch_index(args: dict[str, Any]) -> Any:
    if _ctx is None or not _index_lock.acquire(blocking=False):
        return None
    done = threading.Event()
    outcome: dict[str, Any] = {}
    ctx = _ctx

    def run() -> None:
        try:
            outcome["value"] = ctx.dispatch_tool(_CTX_PREFIX + "index", args)
        except Exception:
            outcome["error"] = True
        finally:
            _index_lock.release()
            done.set()

    threading.Thread(target=run, name="context-mode-index", daemon=True).start()
    if not done.wait(_INDEX_TIMEOUT) or outcome.get("error"):
        return None
    return outcome.get("value")


def _index_succeeded(value: Any) -> bool:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return False
    if not isinstance(parsed, dict) or parsed.get("error") or parsed.get("isError") is True:
        return False
    if parsed.get("success") is True:
        return True
    result = parsed.get("result", parsed)
    if isinstance(result, str):
        return result.startswith("Indexed ")
    if not isinstance(result, dict) or result.get("error") or result.get("isError") is True:
        return False
    if result.get("success") is True:
        return True
    content = result.get("content")
    return isinstance(content, list) and any(
        isinstance(item, dict) and "Indexed " in str(item.get("text", "")) for item in content
    )


def _transform(tool_name: str, result: Any, **kwargs: Any) -> str | None:
    if _ctx is None or tool_name.startswith(_CTX_PREFIX) or tool_name not in _RESULT_TOOLS:
        return None
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    size = len(text.encode("utf-8"))
    if size < _INDEX_THRESHOLD:
        return None
    project_id = sha256(_project(kwargs).encode()).hexdigest()[:12]
    session_id = sha256(_sid(kwargs).encode()).hexdigest()[:12]
    call_id = str(kwargs.get("tool_call_id") or uuid.uuid4().hex)
    source = f"hermes:{tool_name}:{project_id}:{session_id}:{call_id}"
    if not _index_succeeded(_dispatch_index({"content": text, "source": source})):
        return None
    return (
        f'[context-mode: indexed {size} bytes from {tool_name} as source "{source}". '
        f"Use {_CTX_PREFIX}search to retrieve details.]"
    )


def _command(tool: str, raw_args: str = "") -> str:
    if _ctx is None:
        return "context-mode is unavailable"
    args = {"queries": [raw_args]} if tool == "search" and raw_args else {}
    return _ctx.dispatch_tool(_CTX_PREFIX + tool, args)


def _clear_context() -> None:
    global _ctx
    _ctx = None


def register(ctx: Any) -> None:
    """Register only documented Hermes hooks and commands."""
    global _ctx
    _ctx = ctx
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("on_session_start", _session_start)
    ctx.register_hook("on_session_end", _session_boundary)
    ctx.register_hook("on_session_finalize", _session_boundary)
    ctx.register_hook("transform_tool_result", _transform)
    ctx.register_command("ctx-stats", lambda raw_args="": _command("stats", raw_args), "Show context-mode statistics")
    ctx.register_command("ctx-doctor", lambda raw_args="": _command("doctor", raw_args), "Run context-mode diagnostics")
    ctx.register_command("ctx-search", lambda raw_args="": _command("search", raw_args), "Search indexed context", "<query>")
    if hasattr(ctx, "on_unload"):
        ctx.on_unload(_clear_context)
