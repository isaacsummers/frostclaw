# FrostClaw — Snowflake Cortex Plugin for OpenClaw

An [OpenClaw](https://github.com/openclaw/openclaw) provider plugin that routes LLM requests through [Snowflake Cortex AI](https://docs.snowflake.com/en/user-guide/snowflake-cortex). Claude models go to the Anthropic Messages API, everything else goes to the OpenAI-compatible Chat Completions API — both behind a single Snowflake PAT.

## Features

- **LLM Provider** — 16 models across Claude, GPT, Llama, Mistral, DeepSeek, and Arctic
- **Embedding Provider** — Snowflake Arctic Embed models for OpenClaw memory search
- **Zero cost routing** — uses Snowflake credit-based billing (no per-token API fees)

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

### Embedding Models

| Model ID | Dimensions | Notes |
|----------|-----------|-------|
| `snowflake-arctic-embed-m-v1.5` | 768 | Default, widely available |
| `snowflake-arctic-embed-m` | 768 | |
| `snowflake-arctic-embed-l-v2.0` | 1024 | Higher quality, larger |
| `e5-base-v2` | 768 | |

See [Snowflake embed API docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api/embed-api) for model availability by region.

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
| `SNOWFLAKE_CORTEX_API_KEY` | No | Fallback for `SNOWFLAKE_PAT` |
| `SNOWFLAKE_EMBED_PROXY_PORT` | No | Embed proxy port (default: `18790`) |

### Install

Copy the plugin into your OpenClaw extensions directory:

```sh
cp -r . ~/.openclaw/extensions/snowflake-cortex/
```

Or install from a local path:

```jsonc
// openclaw.json
{
  "plugins": {
    "load": { "paths": ["/path/to/frostclaw"] },
    "allow": ["snowflake-cortex"]
  }
}
```

Set the environment variables, then restart OpenClaw.

### Build from Source

```sh
bun install
bun run build
```

This produces `dist/index.js`.

## Embeddings

FrostClaw provides Snowflake Arctic Embed models for OpenClaw's memory search system. This replaces external embedding providers (Gemini, OpenAI, etc.) with Snowflake's native embed API.

### Architecture

FrostClaw registers a native embedding adapter via `registerMemoryEmbeddingProvider` in the plugin SDK. The gateway process uses this adapter directly for real-time memory sync.

For CLI tools like `openclaw memory index`, a lightweight embed proxy (`embed-proxy.mjs`) translates between OpenAI's embedding format and Snowflake's native embed API:

```
OpenClaw CLI (openai provider) → embed-proxy.mjs:18790 → Snowflake /api/v2/cortex/inference:embed
```

The proxy is needed because OpenClaw's CLI loads plugins in a limited registration mode that doesn't support `registerMemoryEmbeddingProvider`. The proxy handles the format differences:

- **Request**: OpenAI uses `input` (string|string[]) → Snowflake uses `text` (string[])
- **Response**: Snowflake wraps vectors in `[[vec]]` → OpenAI expects `[vec]`

### Configuration

```jsonc
// openclaw.json
{
  "agents": {
    "defaults": {
      "memorySearch": {
        "provider": "openai",
        "model": "snowflake-arctic-embed-m-v1.5",
        "remote": {
          "baseUrl": "http://127.0.0.1:18790/v1/",
          "apiKey": "unused"
        }
      }
    }
  }
}
```

### Running the Embed Proxy

As a systemd user service (recommended):

```sh
cat > ~/.config/systemd/user/snowflake-embed-proxy.service << 'EOF'
[Unit]
Description=Frostclaw Snowflake Embed Proxy
After=network.target
PartOf=openclaw-gateway.service

[Service]
Type=simple
ExecStart=node /path/to/frostclaw/embed-proxy.mjs
EnvironmentFile=~/.openclaw/.env
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now snowflake-embed-proxy
```

Or run directly:

```sh
node embed-proxy.mjs
```

Health check: `GET http://127.0.0.1:18790/health`

## How It Works

### LLM Routing

- **Claude models** → Anthropic Messages API at `/api/v2/cortex/v1/messages`
- **Everything else** → OpenAI Chat Completions at `/api/v2/cortex/v1/chat/completions`
- Auth via `X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN` header
- Snowflake Cortex beta headers (interleaved thinking, 128k output, effort control, token-efficient tools) attached to all Claude requests
- Models without tool support have tools stripped via `normalizeToolSchemas`
- Orphaned tool use/result pairs repaired for Claude via `buildReplayPolicy`

### Embedding

- Gateway: native adapter via `registerMemoryEmbeddingProvider` (direct Snowflake API calls)
- CLI: embed proxy with OpenAI-compatible format translation
- Both paths hit Snowflake's `/api/v2/cortex/inference:embed` endpoint

## License

MIT — see [LICENSE](LICENSE).
