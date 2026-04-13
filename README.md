# FrostClaw — Snowflake Cortex Plugin for OpenClaw

An [OpenClaw](https://github.com/openclaw/openclaw) provider plugin that routes LLM requests through [Snowflake Cortex AI](https://docs.snowflake.com/en/user-guide/snowflake-cortex). Claude models go to the Anthropic Messages API, everything else goes to the OpenAI-compatible Chat Completions API — both behind a single Snowflake PAT.

## Supported Models

### Claude (Anthropic Messages API)

| Model ID | Context | Notes |
|----------|---------|-------|
| `claude-opus-4-5` | 200k | |
| `claude-sonnet-4-5` | 200k | |
| `claude-opus-4-6` | 200k | |
| `claude-sonnet-4-6` | 200k | |
| `claude-opus-4-6-1m` | 1M | Opt-in via `context-1m` beta header |
| `claude-sonnet-4-6-1m` | 1M | Opt-in via `context-1m` beta header |

The `-1m` variants are virtual — the plugin strips the suffix before sending to Snowflake and attaches the `context-1m-2025-08-07` beta header automatically.

### OpenAI (Chat Completions API)

| Model ID | Context |
|----------|---------|
| `openai-gpt-5` | 128k |
| `openai-gpt-5-mini` | 128k |
| `openai-gpt-5-nano` | 128k |
| `openai-gpt-4-1` | 1M |

### Open-Source (Chat Completions API)

| Model ID | Context |
|----------|---------|
| `llama4-maverick` | 1M |
| `llama3.1-405b` | 128k |
| `llama3.1-70b` | 128k |
| `llama3.1-8b` | 128k |
| `mistral-large` | 32k |
| `mistral-large2` | 128k |
| `deepseek-r1` | 64k |
| `snowflake-arctic` | 4k |

Tool calling is supported for Claude and OpenAI models. Open-source models have tools stripped automatically.

## Setup

### Prerequisites

- A self-hosted [OpenClaw](https://github.com/openclaw/openclaw) instance
- A Snowflake account with Cortex AI enabled
- A [Programmatic Access Token (PAT)](https://docs.snowflake.com/en/user-guide/programmatic-access-tokens) for your Snowflake account

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SNOWFLAKE_PAT` | Yes | Snowflake Programmatic Access Token |
| `SNOWFLAKE_BASE_URL` | Yes | Account URL, e.g. `https://my-account.snowflakecomputing.com` |

`SNOWFLAKE_CORTEX_API_KEY` is accepted as a fallback for `SNOWFLAKE_PAT`.

### Install

Copy the plugin into your OpenClaw extensions directory:

```sh
cp -r plugins/snowflake-cortex/ ~/.openclaw/plugins/snowflake-cortex/
```

Set the environment variables (in your shell profile, `.env`, or `openclaw.json`), then restart OpenClaw.

### Build from Source

```sh
cd plugins/snowflake-cortex
bun install
bun run build
```

This produces `dist/index.js`.

## How It Works

- **Claude models** → Anthropic Messages API at `/api/v2/cortex/v1/messages`
- **Everything else** → OpenAI Chat Completions at `/api/v2/cortex/v1/chat/completions`
- Auth is handled via `X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN` header
- Snowflake Cortex beta headers (interleaved thinking, 128k output, effort control, token-efficient tools) are attached to all Claude requests automatically
- Models that don't support tool calling have tools stripped via the `normalizeToolSchemas` hook
- Orphaned tool use/result pairs are repaired for Claude via `buildReplayPolicy`

## License

MIT — see [LICENSE](LICENSE).
