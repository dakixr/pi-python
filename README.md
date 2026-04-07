# pi-python

`pi-python` is a modern Python rewrite of the `pi` tooling stack with two layers:

- a native Python agent/runtime/SDK under `pi.agent`, `pi.ai`, and `pi.cli`
- compatibility entrypoints for package names that now stay within this repo

## Port Status

This repo is intentionally explicit about what is native and what remains a smaller local subset.

| Package | Status | Notes |
| --- | --- | --- |
| `pi.agent` | Native | Python loop, tool execution, hooks, parallel tool mode, context compaction |
| `pi.ai` / `pi-ai` | Hybrid | Native Python SDK surface plus a local provider metadata CLI |
| `pi.cli` / `pi-core` | Native | Python one-shot and interactive CLI with JSONL-backed sessions |
| `pi.coding_agent` / `pi` | Native subset | Local alias over the native core agent CLI |
| `pi.pods` / `pi-pods` | Native subset | Native config/store and a local metadata CLI |
| `pi.mom` / `mom` | Native subset | Native sandbox parsing and a local helper CLI |
| `pi.tui` | Native subset | Python text helpers only |
| `pi.web_ui` | Native subset | Local placeholder web UI asset helpers |

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

## Configuration

- `PI_CONFIG_DIR`: override the config directory used by local pod/session helpers

## Development

```bash
uv sync --dev
uv run pytest
```
