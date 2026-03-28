# pi-python

`pi-python` is a minimal Python implementation inspired by `pi`: a small agent runtime plus a CLI wrapper around it.

The codebase stays deliberately small:

- `pi_python.agent`: message models, provider abstraction, core tools, and the agent loop
- `pi_python.cli`: argument parsing and an interactive or one-shot CLI

## Why this shape

The MVP is meant to be easy to extend without carrying framework weight too early:

- provider logic is isolated behind a `Provider` protocol
- tool schemas are generated from typed argument models
- the loop only knows about messages, providers, and tool execution
- the CLI depends on the loop, not the other way around

## Project layout

```text
pi-python/
├── pyproject.toml
├── README.md
├── src/pi_python/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── loop.py
│   │   ├── models.py
│   │   ├── tools.py
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       └── zai.py
│   └── cli/
│       ├── __init__.py
│       └── main.py
└── tests/
    ├── test_cli.py
    ├── test_loop.py
    ├── test_provider_zai.py
    └── test_tools.py
```

## ZAI provider

The first provider is `ZAIProvider`, targeting the OpenAI-style chat completions interface documented by Z.AI:

- endpoint: `https://api.z.ai/api/paas/v4/chat/completions`
- model default: `glm-5.1`
- tool calling: OpenAI-style `tools` + `tool_choice="auto"`

The implementation is ready for real API keys, but the tests keep it fully mocked.

## Tools

The built-in tool registry currently exposes four minimal tools:

- `read_file`
- `write_file`
- `edit_file`
- `bash`

All file operations are restricted to a configured workspace root.

## Usage

Set an API key:

```bash
export ZAI_API_KEY=your-api-key
```

Run one prompt:

```bash
uv run pi-python --prompt "Create a hello world file in this workspace"
```

Run interactively:

```bash
uv run pi-python
```

Point tools at a specific workspace root:

```bash
uv run pi-python --root /path/to/workspace --prompt "Read README.md"
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
- core tool execution and workspace boundary checks
- the agent loop from tool call to final assistant output
- one-shot CLI execution

## Assumptions

- `glm-5.1` is used as the default model name for the coding-plan-oriented MVP
- the Z.AI API remains compatible with the documented chat completions + function calling shape
- streaming is intentionally out of scope for the first cut

## Next steps

- add streaming responses
- add richer edit primitives and safer shell controls
- persist conversation sessions across CLI turns
- support additional providers behind the same protocol
