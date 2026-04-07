# pi-python

`pi-python` is a focused native Python agent/runtime/SDK.

The codebase is intentionally narrow:

- `pi.agent`: agent loop, context handling, tool runtime, and providers
- `pi.ai`: embeddable Python SDK helpers
- `pi.cli`: local one-shot and interactive CLI, exported as `pi`

## Native Surface

The base includes:

- `pi.agent.Agent` with sequential or parallel tool execution
- `before_tool_call` and `after_tool_call` hooks
- `ToolRegistry` plus native `read`, `bash`, `edit`, `write`, `grep`, `find`, and `ls`
- shared truncation semantics: `2000` lines or `50KB`
- JSONL-backed session persistence
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

## Entry Points

```bash
uv run pi --help
```

## Development

```bash
uv sync --dev
uvx pyright
uv run pytest
```
