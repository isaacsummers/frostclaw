/**
 * Live model catalog probe — verifies every catalogued model is actually
 * callable on Snowflake Cortex with a minimal valid request.
 *
 * Claude models   → Messages API  (/api/v2/cortex/v1/messages)
 * OpenAI/OSS      → Chat Completions (/api/v2/cortex/v1/chat/completions)
 * Embedding       → Embed API (/api/v2/cortex/inference:embed)
 *
 * Reasoning models (openai-gpt-5.1/5.2/5.4) require `reasoning_effort`
 * instead of `max_completion_tokens` alone — tested with both fields.
 *
 * Run: bun test tests/live/model-catalog.live.test.ts
 * Skips automatically when credentials are not set.
 * Do NOT run as part of the standard `bun test` suite (CI).
 */

import { describe, test, expect } from "bun:test";

const PAT      = process.env.SNOWFLAKE_PAT ?? process.env.SNOWFLAKE_CORTEX_API_KEY ?? "";
const BASE_URL = (process.env.SNOWFLAKE_BASE_URL ?? "").replace(/\/$/, "");
const HAVE_CREDS = !!PAT && !!BASE_URL;

const HEADERS = {
  "Authorization": `Bearer ${PAT}`,
  "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
  "Content-Type": "application/json",
  "Accept": "application/json",
};

async function callMessages(model: string): Promise<{ ok: boolean; detail: string }> {
  const res = await fetch(`${BASE_URL}/api/v2/cortex/v1/messages`, {
    method: "POST",
    headers: { ...HEADERS, "anthropic-version": "2023-06-01" },
    body: JSON.stringify({ model, max_tokens: 1, messages: [{ role: "user", content: "hi" }] }),
    signal: AbortSignal.timeout(25_000),
  });
  const body = await res.json() as Record<string, unknown>;
  if (Array.isArray(body.content)) return { ok: true, detail: String(body.model ?? model) };
  return { ok: false, detail: String(body.message ?? JSON.stringify(body)).slice(0, 120) };
}

async function callChat(model: string, extra: Record<string, unknown> = {}): Promise<{ ok: boolean; detail: string }> {
  const res = await fetch(`${BASE_URL}/api/v2/cortex/v1/chat/completions`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ model, max_completion_tokens: 1, messages: [{ role: "user", content: "hi" }], ...extra }),
    signal: AbortSignal.timeout(25_000),
  });
  const body = await res.json() as Record<string, unknown>;
  if (Array.isArray(body.choices)) return { ok: true, detail: String(body.model ?? model) };
  return { ok: false, detail: String(body.message ?? JSON.stringify(body)).slice(0, 120) };
}

async function callEmbed(model: string): Promise<{ ok: boolean; detail: string }> {
  const res = await fetch(`${BASE_URL}/api/v2/cortex/inference:embed`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ model, text: ["probe"] }),
    signal: AbortSignal.timeout(25_000),
  });
  const body = await res.json() as Record<string, unknown>;
  if (Array.isArray(body.data) && (body.data as unknown[]).length > 0) {
    const row = (body.data as unknown[])[0];
    const dims = Array.isArray(row) ? row.length : (row as Record<string, unknown[]>).embedding?.length ?? 0;
    return { ok: true, detail: `dims=${dims}` };
  }
  return { ok: false, detail: String(body.message ?? JSON.stringify(body)).slice(0, 120) };
}

// ---------------------------------------------------------------------------
// Claude — Messages API
// ---------------------------------------------------------------------------

describe("live: Claude models (Messages API)", () => {
  if (!HAVE_CREDS) { test.skip("no credentials", () => {}); }

  const CLAUDE_MODELS = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
  ];

  // claude-fable-5: access suspended as of June 2026 — expected failure
  test("claude-fable-5 is suspended (expected unavailable)", async () => {
    const { ok, detail } = await callMessages("claude-fable-5");
    expect(ok, `Expected fable-5 to be unavailable but got ok; detail: ${detail}`).toBe(false);
    expect(detail.toLowerCase()).toMatch(/unavailable|suspended|unknown/);
  });

  for (const model of CLAUDE_MODELS) {
    test(`${model} responds ok`, async () => {
      const { ok, detail } = await callMessages(model);
      expect(ok, `${model} failed: ${detail}`).toBe(true);
    });
  }
});

// ---------------------------------------------------------------------------
// OpenAI — Chat Completions (standard)
// ---------------------------------------------------------------------------

describe("live: OpenAI models — standard (Chat Completions)", () => {
  if (!HAVE_CREDS) { test.skip("no credentials", () => {}); }

  const STANDARD_MODELS = [
    "openai-gpt-5",
    "openai-gpt-5-mini",
    "openai-gpt-5-nano",
    "openai-gpt-4.1",
  ];

  for (const model of STANDARD_MODELS) {
    test(`${model} responds ok`, async () => {
      const { ok, detail } = await callChat(model);
      expect(ok, `${model} failed: ${detail}`).toBe(true);
    });
  }
});

// ---------------------------------------------------------------------------
// OpenAI — Chat Completions (reasoning models — require reasoning_effort)
// ---------------------------------------------------------------------------

describe("live: OpenAI reasoning models — require reasoning_effort", () => {
  if (!HAVE_CREDS) { test.skip("no credentials", () => {}); }

  // These models return "invalid request" without reasoning_effort.
  // frostclaw does NOT currently inject reasoning_effort for OpenAI models —
  // callers must pass it explicitly via the thinkingLevel / OpenClaw reasoning
  // config. Tests confirm the correct call shape works.
  const REASONING_MODELS = [
    "openai-gpt-5.1",
    "openai-gpt-5.2",
    "openai-gpt-5.4",
  ];

  for (const model of REASONING_MODELS) {
    test(`${model} with reasoning_effort=low responds ok`, async () => {
      const { ok, detail } = await callChat(model, { reasoning_effort: "low" });
      expect(ok, `${model} failed: ${detail}`).toBe(true);
    });

    test(`${model} without reasoning_effort returns invalid request`, async () => {
      const { ok, detail } = await callChat(model);
      expect(ok, `Expected ${model} to fail without reasoning_effort`).toBe(false);
      expect(detail.toLowerCase()).toMatch(/invalid/);
    });
  }
});

// ---------------------------------------------------------------------------
// Open-source — Chat Completions
// ---------------------------------------------------------------------------

describe("live: Open-source models (Chat Completions)", () => {
  if (!HAVE_CREDS) { test.skip("no credentials", () => {}); }

  const OSS_MODELS = [
    "deepseek-r1",
    "llama3.1-405b",
    "llama3.1-70b",
    "llama3.1-8b",
    "llama4-maverick",
    "mistral-7b",
    "mistral-large",
    "mistral-large2",
    "snowflake-llama-3.3-70b",
  ];

  // llama3.3-70b: region-locked, not dead — skip rather than fail
  test.skip("llama3.3-70b — region-locked (cross-region not enabled)", () => {});

  for (const model of OSS_MODELS) {
    test(`${model} responds ok`, async () => {
      const { ok, detail } = await callChat(model);
      expect(ok, `${model} failed: ${detail}`).toBe(true);
    });
  }
});

// ---------------------------------------------------------------------------
// Embedding models
// ---------------------------------------------------------------------------

describe("live: Embedding models", () => {
  if (!HAVE_CREDS) { test.skip("no credentials", () => {}); }

  const EMBED_MODELS = [
    "snowflake-arctic-embed-m-v1.5",
    "snowflake-arctic-embed-m",
    "snowflake-arctic-embed-l-v2.0",
    "e5-base-v2",
  ];

  for (const model of EMBED_MODELS) {
    test(`${model} responds ok`, async () => {
      const { ok, detail } = await callEmbed(model);
      expect(ok, `${model} failed: ${detail}`).toBe(true);
    });
  }
});
