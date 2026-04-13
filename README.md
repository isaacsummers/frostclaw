# OpenClaw — Personal Instance

This repo holds the config and setup notes for my self-hosted OpenClaw instance, running on a VPS.

## What is OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is an open-source (MIT) self-hosted AI agent platform. It connects LLMs to tools and messaging apps through a gateway daemon. It supports persistent memory via `SOUL.md`, `MEMORY.md`, and `USER.md` files, is model-agnostic, and has 20+ chat platform integrations.

## Connecting to the VPS

SSH in using the alias:

```sh
ssh oca
```

This requires the `oca` host entry in your `~/.ssh/config`. If you don't have it, get the host/port/user details from me and add your public key to the VPS's `~/.ssh/authorized_keys`.

## Directory Layout

Once connected, OpenClaw lives at:

| Path | What |
|------|------|
| `~/.openclaw/` | Main OpenClaw directory |
| `~/.openclaw/openclaw.json` | Primary config file |
| `~/.openclaw/SOUL.md` | Agent personality / system prompt |
| `~/.openclaw/MEMORY.md` | Persistent agent memory |
| `~/.openclaw/USER.md` | User context the agent references |
| `~/.openclaw/logs/` | Log output |

## Start / Stop / Restart

```sh
# Start the daemon
openclaw start

# Stop it
openclaw stop

# Restart (after config changes, etc.)
openclaw restart

# Check status
openclaw status
```

Logs stream to `~/.openclaw/logs/`. Tail them with:

```sh
tail -f ~/.openclaw/logs/openclaw.log
```

## For Agents / Developers Working on This Setup

### Key files

- **`openclaw.json`** — all runtime config: model provider keys, enabled integrations, tool permissions, gateway settings. Edit this to change behavior.
- **`SOUL.md`** — defines the agent's personality and system-level instructions. Changes take effect on next restart.
- **`MEMORY.md`** — the agent's long-term memory. It writes to this itself. You can edit it manually, but be careful not to corrupt its structure.
- **`USER.md`** — context about the user. The agent reads this to personalize responses.

### Making changes

1. SSH in via `ssh oca`
2. Edit the relevant file(s) in `~/.openclaw/`
3. Run `openclaw restart` to pick up config changes
4. Tail the logs to verify nothing broke

### Watch out for

- **Don't nuke `MEMORY.md`** — it contains accumulated context the agent has built up. Back it up before making structural edits.
- **API keys live in `openclaw.json`** — treat it as sensitive. Don't copy it off the VPS or commit it anywhere.
- **The daemon binds to a port** — if it won't start, check if something else grabbed the port or if a previous instance is still running (`openclaw status`).
- **Config is JSON** — a trailing comma or syntax error will prevent startup. Validate before restarting.

## Snowflake Cortex Plugin

The `plugins/snowflake-cortex/` directory contains a custom provider plugin that routes requests through Snowflake's Cortex AI gateway. Claude models go to the Anthropic Messages API (`/messages`), everything else goes to the OpenAI-compatible Chat Completions API (`/chat/completions`).

### Required env vars

| Variable | Purpose |
|----------|---------|
| `SNOWFLAKE_PAT` | Snowflake Programmatic Access Token (primary) |
| `SNOWFLAKE_BASE_URL` | Account URL, e.g. `https://my-account.snowflakecomputing.com` |

`SNOWFLAKE_CORTEX_API_KEY` is accepted as a fallback for `SNOWFLAKE_PAT` so older setups don't break, but `SNOWFLAKE_PAT` is preferred.

### Deploying to the VPS

1. Copy the plugin directory to the VPS:
   ```sh
   scp -r plugins/snowflake-cortex/ oca:~/.openclaw/plugins/snowflake-cortex/
   ```
2. Set the env vars on the VPS (add to `~/.openclaw/openclaw.json` or your shell profile).
3. Restart OpenClaw:
   ```sh
   ssh oca 'openclaw restart'
   ```

### Supported models

Claude Opus 4.5/4.6, Claude Sonnet 4.5/4.6, GPT-5/5-Mini/5-Nano, GPT-4.1, Llama 4 Maverick, Llama 3.1 (8B/70B/405B), Mistral Large/Large 2, DeepSeek R1, Snowflake Arctic.
