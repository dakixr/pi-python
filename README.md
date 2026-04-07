# pi-python

`pi-python` is a modern Python rewrite of the local `pi-mono` workspace with two layers:

- a native Python agent/runtime/SDK under `pi.agent`, `pi.ai`, and `pi.cli`
- compatibility entrypoints for upstream packages that still make more sense to delegate

The compatibility layer targets `/tmp/pi-mono-aBS53I` by default, or `PI_MONO_REPO` when set.

## Port Status

This repo is intentionally explicit about what is native, hybrid, and still delegated.

| Package | Status | Notes |
| --- | --- | --- |
| `pi.agent` | Native | Python loop, tool execution, hooks, parallel tool mode, context compaction |
| `pi.ai` / `pi-ai` | Hybrid | Native Python SDK surface plus upstream OAuth/provider CLI flows |
| `pi.cli` / `pi-core` | Native | Python one-shot and interactive CLI with JSONL-backed sessions |
| `pi.coding_agent` / `pi` | Wrapper | Delegates to upstream TypeScript coding-agent |
| `pi.pods` / `pi-pods` | Hybrid | Native config/store, upstream operational CLI |
| `pi.mom` / `mom` | Hybrid | Native sandbox parsing, upstream bot/runtime |
| `pi.tui` | Native subset | Python text helpers only |
| `pi.web_ui` | Wrapper | Python helpers around upstream web UI assets |

For a machine-readable view, import `pi.porting.port_status()`.

## Native Python Surface

The native core now covers the pieces that are actually useful to embed from Python:

- `pi.agent.Agent` with sequential or parallel tool execution
- `before_tool_call` and `after_tool_call` hooks
- `ToolRegistry` plus native `read`, `bash`, `edit`, `write`, `grep`, `find`, and `ls`
- shared truncation semantics closer to upstream: `2000` lines or `50KB`
- JSONL-backed session persistence with compatibility snapshots
- `pi.ai.complete(...)`, `pi.ai.stream(...)`, `pi.ai.create_agent(...)`, and `pi.ai.run_task(...)`
- provider support for native `ZAIProvider` and a generic `OpenAICompatibleProvider`

## Example

```python
from pathlib import Path

from pi.ai import ZAIConfig, ZAIProvider, run_task

provider = ZAIProvider(ZAIConfig(api_key="...", model="glm-5.1"))
result = run_task(
    "Create a hello.py file that prints hello",
    provider=provider,
    root=Path.cwd(),
    system_prompt="You are a coding assistant.",
)

print(result.output)
```

For direct completion-style usage without the agent loop:

```python
from pi.ai import Context, complete

response = complete(
    provider=provider,
    context=Context.from_prompt("Summarize this repository", system_prompt="Be concise."),
)

print(response.output)
```

## Entry Points

```bash
uv run pi --help
uv run pi-core --help
uv run pi-ai --help
uv run pi-pods --help
uv run mom --help
```

`pi`, `pi-ai`, `pi-pods`, and `mom` resolve the upstream repository, install npm dependencies on demand, and invoke the TypeScript entrypoint through `tsx`.

## Configuration

- `PI_MONO_REPO`: override the upstream repo path
- `PI_NODE_PACKAGE_MANAGER`: override the package manager, defaults to `npm`
- `PI_AUTO_INSTALL_UPSTREAM=0`: disable automatic dependency installation for upstream wrappers

## Development

```bash
uv sync --dev
uv run pytest
```
