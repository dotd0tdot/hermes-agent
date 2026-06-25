---
sidebar_position: 4.5
title: "Context Pruner"
description: "Context-aware tool pruning — detect runtime environment, prune irrelevant toolsets, and lazily import tool modules"
---

# Context Pruner

The context pruner (`tools/context_pruner.py`) reduces input‑token overhead by detecting *where* the agent is running and skipping tools that can never be useful in that environment. It works in three layers:

1. **`detect_runtime_context()`** — probes env vars to identify the runtime environment (CLI, gateway platform, TUI, subagent, etc.)
2. **Static pruning** — filters `enabled_toolsets` by a per‑context relevance map before they are resolved into tool schemas
3. **Lazy discovery** — imports tool modules only when their toolset is actually needed, skipping modules whose toolsets were pruned

Together these save hundreds of tokens per API request — no point sending `kanban` or `computer_use` schemas when you're in a Telegram chat — and avoid importing modules that will never be called.

Source: `tools/context_pruner.py` · Integration: `model_tools.py` → `get_tool_definitions()` · Tests: `tests/tools/test_context_pruner.py`

---

## `detect_runtime_context()`

The detection function probes the process environment (env vars set by the agent runner) and returns a `RuntimeContext` enum member. The check order is deliberate — a kanban worker may also be running in CLI mode, so more specific environments are checked first.

```python
from tools.context_pruner import detect_runtime_context

ctx = detect_runtime_context()
print(ctx.value)  # e.g. "cli", "gateway:telegram", "kanban_worker"
```

### Detection logic (priority order)

| Priority | Env var | RuntimeContext | Set by |
|----------|---------|----------------|--------|
| 1 | `HERMES_KANBAN_TASK` | `KANBAN_WORKER` | Kanban dispatcher when spawning a task worker |
| 2 | `HERMES_PARENT_SESSION_ID` | `SUBAGENT` | `delegate_task` when spawning a child agent |
| 3 | `HERMES_BATCH_MODE` | `BATCH` | `batch_runner.py` / data generation |
| 4a | `HERMES_GATEWAY_PLATFORM=telegram` | `GATEWAY_TELEGRAM` | Gateway runner per‑platform agent init |
| 4b | `HERMES_GATEWAY_PLATFORM=discord` | `GATEWAY_DISCORD` | Same |
| 4c | `HERMES_GATEWAY_PLATFORM=slack` | `GATEWAY_SLACK` | Same |
| 4d–k | `HERMES_GATEWAY_PLATFORM=whatsapp, signal, feishu, matrix, wecom, yuanbao` | Corresponding `GATEWAY_*` | Same |
| 4l | `HERMES_GATEWAY_PLATFORM=<other>` | `GATEWAY_GENERIC` | Unlisted platform |
| 5 | `HERMES_TUI` or `HERMES_DESKTOP` | `TUI` | Ink‑based TUI or Electron desktop |
| 6 | *(none of the above)* | `CLI` | Fallback — interactive CLI or headless |

The platform string is lower‑cased before matching, so `Telegram` → `gateway:telegram`. An empty `HERMES_GATEWAY_PLATFORM` does not match; it falls through to CLI detection.

### Enum values

```python
class RuntimeContext(enum.Enum):
    CLI = "cli"
    GATEWAY_TELEGRAM = "gateway:telegram"
    GATEWAY_DISCORD = "gateway:discord"
    GATEWAY_SLACK = "gateway:slack"
    GATEWAY_WHATSAPP = "gateway:whatsapp"
    GATEWAY_SIGNAL = "gateway:signal"
    GATEWAY_FEISHU = "gateway:feishu"
    GATEWAY_MATRIX = "gateway:matrix"
    GATEWAY_WECOM = "gateway:wecom"
    GATEWAY_YUANBAO = "gateway:yuanbao"
    GATEWAY_GENERIC = "gateway:generic"
    TUI = "tui"
    KANBAN_WORKER = "kanban_worker"
    SUBAGENT = "subagent"
    BATCH = "batch"
    UNKNOWN = "unknown"
```

---

## Runtime Context Matrix

The `_CONTEXT_IRRELEVANT_TOOLSETS` dictionary maps each `RuntimeContext` to a set of toolset names that are **never useful** in that context. Toolsets not listed are treated as potentially relevant and pass through to normal `check_fn` filtering.

| RuntimeContext | Pruned toolsets | Rationale |
|---------------|------------------|-----------|
| `CLI` | `kanban`, `computer_use` | Kanban tools are lifecycle‑only; computer use requires macOS CUA driver |
| `TUI` | `kanban`, `computer_use` | Same reasoning as CLI |
| `GATEWAY_*` | *(none — empty set)* | All toolsets may fire in gateway contexts; gatekeeping is left to `check_fn` |
| `KANBAN_WORKER` | `computer_use` | Kanban workers need their own toolsets but not macOS automation |
| `SUBAGENT` | `kanban` | Subagents should not spawn further subagents or manage kanban lifecycle; browser/media tools are intentionally left available |
| `BATCH` | *(none)* | Batch processing may use any toolset |
| `UNKNOWN` | *(none)* | Conservative fallback — assume everything is relevant |

The pruning is **static** — it runs before `check_fn` evaluations and before any tool modules are imported. This means an irrelevant toolset is never even registered for the session, saving both import time and prompt tokens.

### `pruned_toolsets()` helper

```python
from tools.context_pruner import pruned_toolsets

pruned, removed = pruned_toolsets(
    enabled_toolsets=["web", "kanban", "terminal", "computer_use", "file"],
    context=RuntimeContext.CLI,
)
# pruned  = ["web", "terminal", "file"]
# removed = ["kanban", "computer_use"]
```

When `enabled_toolsets` is `None` (auto‑mode), pruning is a no‑op and `(None, [])` is returned.

### `toolset_is_relevant()` predicate

```python
from tools.context_pruner import toolset_is_relevant, RuntimeContext

if toolset_is_relevant("kanban", RuntimeContext.CLI) is False:
    # Skip importing kanban modules
```

Returns `True` for toolsets not present in the context's exclude set. Unknown toolset names always return `True`.

---

## `lazy_discover_tools()`

The standard `discover_builtin_tools()` eagerly imports every `tools/*.py` module that has a top‑level `registry.register()` call. `lazy_discover_tools()` is an alternative that **only imports modules whose toolsets are relevant and enabled**:

```python
from tools.context_pruner import lazy_discover_tools

# Instead of discover_builtin_tools(), call:
lazy_discover_tools(
    enabled_toolsets=["terminal", "web", "file"],
    context=detect_runtime_context(),
)
```

### How it works

1. For each enabled toolset, check `toolset_is_relevant()` — skip if irrelevant in the current context.
2. Resolve the toolset to its individual tool names via `toolsets.resolve_toolset()`.
3. Scan `tools/*.py` files, import each module, and track it in the `_imported_tool_modules` set (avoids double‑importing across calls).
4. Return the list of imported module names.

### Lazy‑import toolset list

Some toolsets are intentionally excluded from eager startup and only imported when explicitly enabled. These are listed in `_LAZY_TOOLSETS`:

| Toolset | When it loads |
|---------|---------------|
| `kanban` | Only when `HERMES_KANBAN_TASK` is set |
| `computer_use` | Only on macOS with `cua-driver` |
| `homeassistant` | Only when `HASS_TOKEN` is configured |
| `x_search` | Only with xAI credentials |
| `spotify` | Only when Spotify credentials configured |
| `discord` | Only when Discord gateway active |
| `discord_admin` | Only when Discord admin configured |

When `enabled_toolsets` is `None` (no explicit set), `lazy_discover_tools()` falls back to `discover_builtin_tools()` which eagerly imports everything — a conservative fallback that preserves backward compatibility.

---

## Integration with `model_tools.py`

The context pruner hooks into the tool resolution pipeline through `get_tool_definitions()` in `model_tools.py`. Every call follows this flow:

```text
get_tool_definitions(enabled_toolsets, disabled_toolsets)
    │
    ├─ Build cache key (includes RuntimeContext — so pruning is stable per context)
    │
    ├─ _compute_tool_definitions()
    │    │
    │    ├─ pruned_toolsets(enabled_toolsets, detect_runtime_context())
    │    │   → removes kanban from CLI, computer_use from kanban worker, etc.
    │    │
    │    ├─ resolve_toolset() for each remaining toolset
    │    │
    │    └─ registry.get_definitions(tool_names) → check_fn per tool
    │
    └─ Cache result per (enabled_toolsets, disabled_toolsets, context, generation, config_fp)
```

The cache key includes `detect_runtime_context()` so that a cache entry computed for `gateway:telegram` is not reused for `gateway:discord` (which may have a different pruning profile).

When the agent runs without explicit `enabled_toolsets` (the default), pruning still applies: `_compute_tool_definitions()` iterates all known toolsets and skips those classified as irrelevant for the current context.

---

## Estimated Token Savings

The `format_toolset_report()` function renders a human‑readable summary including estimated token savings:

```python
from tools.context_pruner import format_toolset_report, detect_runtime_context

print(format_toolset_report(
    enabled_toolsets=["web", "terminal", "file", "kanban", "computer_use"],
    context=detect_runtime_context(),
))
# Runtime context: cli
# Active toolsets:  3
# Pruned toolsets:  2
# Removed: kanban, computer_use
# Estimated savings: ~500 tokens/req (~4 tools × ~500B)
```

The estimate is calculated as:
- `tool_count × 500` bytes per tool (average schema size)
- `bytes ÷ 4` tokens (rough JSON‑to‑token conversion)

The constant `_AVERAGE_TOOL_SCHEMA_BYTES = 500` is intentionally conservative; actual schema sizes range from ~300 to ~800 bytes depending on parameter complexity.

---

## Config Knobs

The context pruner does not have its own `config.yaml` section — it operates on **environment variables** set by the agent runner and on **static code maps**. There are two ways to influence its behavior:

### 1. Environment variables (runtime detection)

| Env var | Effect |
|---------|--------|
| `HERMES_KANBAN_TASK` | Forces `KANBAN_WORKER` context; prunes non‑kanban lifecycle tools |
| `HERMES_PARENT_SESSION_ID` | Forces `SUBAGENT` context; disables kanban toolset |
| `HERMES_BATCH_MODE` | Forces `BATCH` context; no pruning applied |
| `HERMES_GATEWAY_PLATFORM` | Forces gateway context; platform‑specific mapping |
| `HERMES_TUI` | Forces `TUI` context; prunes `computer_use`, `kanban` |
| `HERMES_DESKTOP` | Same as `HERMES_TUI` |

### 2. Toolset configuration in `config.yaml`

Indirect control via the agent's toolset configuration:

```yaml
# Only enable specific toolsets — pruning then acts on this subset
agent:
  enabled_toolsets:
    - web
    - terminal
    - file
  disabled_toolsets: []
```

When `enabled_toolsets` is explicitly set, the context pruner only skips those that are irrelevant in the current context. When left empty (the default), the pruner iterates all known toolsets and applies the same filtering.

### 3. Extending the relevance map (code change)

To add a new context or modify pruning behavior, edit `_CONTEXT_IRRELEVANT_TOOLSETS` in `tools/context_pruner.py`:

```python
_CONTEXT_IRRELEVANT_TOOLSETS: Dict[RuntimeContext, Set[str]] = {
    RuntimeContext.CLI: {"kanban", "computer_use"},
    RuntimeContext.TUI: {"kanban", "computer_use"},
    RuntimeContext.KANBAN_WORKER: {"computer_use"},
    RuntimeContext.SUBAGENT: {"kanban"},
    # ... gateway contexts all have empty sets
}
```

The `_LAZY_TOOLSETS` set controls which tool modules skip eager startup:

```python
_LAZY_TOOLSETS = {
    "kanban", "computer_use", "homeassistant",
    "x_search", "spotify", "discord", "discord_admin",
}
```

---

## Related docs

- [Tools Runtime](./tools-runtime.md) — registration, dispatch, `check_fn`, terminal environments
- [Toolsets Reference](../reference/toolsets-reference.md) — available toolsets and their composition
- [Agent Loop Internals](./agent-loop.md) — how tool definitions flow into the agent loop
- [Context Compression and Caching](./context-compression-and-caching.md) — runtime context window management
