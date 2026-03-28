# pi-python

`pi-python` is the Python package and distribution name. Internally, the runtime, module path, and CLI are all `pi`.

The current implementation stays intentionally non-streaming. Each turn runs to completion, returns one assistant response, and optionally persists the full conversation locally for later reuse.

The codebase stays deliberately small:

- `pi.agent`: message models, provider abstraction, context management, tool definitions, and the agent loop
- `pi.cli`: argument parsing, session persistence, and an interactive or one-shot CLI

## Why this shape

The MVP is meant to be easy to extend without carrying framework weight too early:

- provider logic is isolated behind a `Provider` protocol
- tool schemas are generated from dedicated Pydantic input models attached to concrete tool classes
- the loop only knows about context preparation, providers, and tool execution
- the CLI depends on the loop, not the other way around

## Project layout

```text
pi-python/
├── pyproject.toml
├── README.md
├── src/pi/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── context.py
│   │   ├── loop.py
│   │   ├── models.py
│   │   ├── tools.py
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       └── zai.py
│   └── cli/
│       ├── __init__.py
│       ├── main.py
│       └── session.py
└── tests/
    ├── test_cli.py
    ├── test_loop.py
    ├── test_provider_zai.py
    └── test_tools.py
```

## ZAI provider

The first provider is `ZAIProvider`, targeting the OpenAI-style chat completions interface documented by Z.AI:

- endpoint: `https://api.z.ai/api/coding/paas/v4/chat/completions`
- model default: `glm-5.1`
- tool calling: OpenAI-style `tools` + `tool_choice="auto"`
- retries: up to 3 total attempts for `429` and transient `5xx` responses
- rate limits: honors `Retry-After` when present, otherwise uses a short exponential backoff

The implementation is ready for real API keys, but the tests keep it fully mocked. Provider failures are translated into concise runtime errors so the CLI exits cleanly instead of dumping a traceback for expected API problems.

## Tools

The runtime is now definition-first: each tool is a child `BaseModel` with:

- stable metadata (`name`, `description`)
- a dedicated child Pydantic schema for arguments
- local execution logic

That makes registration simple and extensible without hard-coding schemas in the registry.

The built-in tool factories mirror the TypeScript reference structure:

- coding tools: `read`, `bash`, `edit`, `write`
- read-only tools: `read`, `grep`, `find`, `ls`
- all tools: both sets combined

File operations are restricted to a configured workspace root and must use relative paths. The runtime also rejects oversized text payloads, caps captured shell output, and supports richer schemas such as:

- `read(path, offset?, limit?)`
- `edit(path, oldText/newText)` or `edit(path, edits=[...])`
- `bash(command, timeout?)`

## Context Engineering

The loop now uses a small `ContextManager` before the provider boundary:

- initialize a run from prior messages plus the new prompt
- inject the system prompt only when needed
- optionally transform messages before the provider sees them
- append structured tool-result messages in one place

This keeps the non-streaming loop small while leaving a clean seam for future pruning, summarization, or external context injection.

## Sessions

Use `--session <name>` to load and persist conversation state across separate CLI invocations. Sessions are stored as JSON in:

```text
<workspace-root>/.pi/sessions/<name>.json
```

That state includes the full message history used by the agent loop, so a later run can continue the prior conversation without streaming or an external database.

## Usage

Set an API key:

```bash
export ZAI_API_KEY=your-api-key
```

Run one prompt:

```bash
uv run pi --prompt "Create a hello world file in this workspace"
```

If the provider returns a retryable API error and the retries are exhausted, the CLI prints a short error to stderr and exits with code `1`.

Run one prompt and persist the conversation under a named session:

```bash
uv run pi --session demo --prompt "Create a hello world file in this workspace"
```

Run interactively:

```bash
uv run pi
```

Resume the same conversation later:

```bash
uv run pi --session demo
```

Point tools at a specific workspace root:

```bash
uv run pi --root /path/to/workspace --prompt "Read README.md"
```

## Development

Install dependencies:

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```

## Testing scope

The tests cover:

- ZAI response parsing and request payload shape
- tool execution, workspace boundary checks, and Pydantic-backed tool registration
- the agent loop from tool call to final assistant output, including context transforms
- one-shot CLI execution plus named session persistence

## Assumptions

- `glm-5.1` is used as the default model name for the coding-plan-oriented MVP
- the Z.AI API remains compatible with the documented chat completions + function calling shape
- streaming remains intentionally out of scope

## Next steps

- add streaming without changing the context/tool abstractions
- layer context pruning or summarization into `ContextManager`
- support additional providers behind the same protocol
