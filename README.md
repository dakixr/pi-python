# pi-python

`pi-python` is a minimal Python implementation inspired by `pi`: a small agent runtime plus a CLI wrapper around it.

The current implementation stays intentionally non-streaming. Each turn runs to completion, returns one assistant response, and optionally persists the full conversation locally for later reuse.

The codebase stays deliberately small:

- `pi_python.agent`: message models, provider abstraction, core tools, and the agent loop
- `pi_python.cli`: argument parsing, session persistence, and an interactive or one-shot CLI

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

The built-in tool registry currently exposes four minimal tools:

- `read`
- `write`
- `edit`
- `bash`

File operations are restricted to a configured workspace root and must use relative paths. The runtime also rejects oversized text payloads, caps captured shell output, and returns clearer timeout/error payloads from tools.

## Sessions

Use `--session <name>` to load and persist conversation state across separate CLI invocations. Sessions are stored as JSON in:

```text
<workspace-root>/.pi-python/sessions/<name>.json
```

That state includes the full message history used by the agent loop, so a later run can continue the prior conversation without streaming or an external database.

## Usage

Set an API key:

```bash
export ZAI_API_KEY=your-api-key
```

Run one prompt:

```bash
uv run pi-python --prompt "Create a hello world file in this workspace"
```

If the provider returns a retryable API error and the retries are exhausted, the CLI prints a short error to stderr and exits with code `1`.

Run one prompt and persist the conversation under a named session:

```bash
uv run pi-python --session demo --prompt "Create a hello world file in this workspace"
```

Run interactively:

```bash
uv run pi-python
```

Resume the same conversation later:

```bash
uv run pi-python --session demo
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
- core tool execution, renamed tool definitions, and workspace boundary checks
- the agent loop from tool call to final assistant output, including continued history
- one-shot CLI execution plus named session persistence

## Assumptions

- `glm-5.1` is used as the default model name for the coding-plan-oriented MVP
- the Z.AI API remains compatible with the documented chat completions + function calling shape
- streaming remains intentionally out of scope

## Next steps

- add richer edit primitives
- make bash controls more policy-driven if stricter environments are needed
- support additional providers behind the same protocol
