/**
 * Live integration tests — `eager_input_streaming` field acceptance on
 * Snowflake Cortex.
 *
 * Background: pi-ai's Anthropic provider injects `eager_input_streaming: true`
 * onto every tool definition by default (compat flag
 * `supportsEagerToolInputStreaming`, default true). Snowflake Cortex's request
 * schema validator rejects this field with:
 *
 *   400 invalid request parameters:
 *     "tools.0.custom.eager_input_streaming: Extra inputs are not permitted"
 *
 * This file probes the matrix of (model × beta header × field present/stripped)
 * to confirm:
 *   - Whether stripping `eager_input_streaming` from tool schemas fixes the 400
 *   - Whether any beta header makes Snowflake accept the field
 *   - Whether the rejection is uniform across Sonnet 4.6 and Haiku 4.5
 *
 * Endpoint: ${SNOWFLAKE_BASE_URL}/api/v2/cortex/v1/messages
 *
 * Run: bun test tests/eager-input-streaming.live.test.ts
 *
 * Skips automatically if credentials are not set.
 */

import { describe, test, expect } from "bun:test";

const PAT =
  process.env.SNOWFLAKE_PAT ?? process.env.SNOWFLAKE_CORTEX_API_KEY ?? "";
const BASE_URL = process.env.SNOWFLAKE_BASE_URL ?? "";
const HAVE_CREDS = !!PAT && !!BASE_URL;
const ENDPOINT = HAVE_CREDS
  ? `${BASE_URL.replace(/\/$/, "")}/api/v2/cortex/v1/messages`
  : "";

const SONNET = "claude-sonnet-4-6";
const HAIKU = "claude-haiku-4-5";

const FLAG_TOKEN_EFFICIENT = "token-efficient-tools-2025-02-19";
const FLAG_FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14";

interface CallResult {
  status: number;
  ok: boolean;
  elapsedMs: number;
  body: any;
  rawText: string;
  errorMessage?: string;
}

async function call(
  model: string,
  body: Record<string, unknown>,
  betaFlags: string[] | null,
  timeoutMs = 60000,
): Promise<CallResult> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${PAT}`,
    "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
  };
  if (betaFlags && betaFlags.length > 0) {
    headers["anthropic-beta"] = betaFlags.join(",");
  }
  const start = Date.now();
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers,
    body: JSON.stringify({ model, ...body }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const elapsed = Date.now() - start;
  const text = await res.text();
  let parsed: any = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    /* non-JSON body */
  }
  return {
    status: res.status,
    ok: res.ok,
    elapsedMs: elapsed,
    body: parsed,
    rawText: text,
    errorMessage:
      parsed?.error?.message ??
      parsed?.message ??
      (res.ok ? undefined : text.slice(0, 500)),
  };
}

function logResult(label: string, r: CallResult): void {
  const usage = r.body?.usage ?? r.body?.message?.usage;
  console.log(
    `  [${label}] status=${r.status} elapsed=${r.elapsedMs}ms ` +
      (usage
        ? `in=${usage.input_tokens ?? "?"} out=${usage.output_tokens ?? "?"} `
        : "") +
      (r.errorMessage ? `err="${r.errorMessage.slice(0, 250)}"` : "ok"),
  );
}

// Tool definition WITH `eager_input_streaming: true` — mirrors what pi-ai
// emits when compat.supportsEagerToolInputStreaming is left at default (true).
const TOOL_WITH_EAGER = {
  name: "get_weather",
  description: "Get the current weather in a given location.",
  eager_input_streaming: true,
  input_schema: {
    type: "object" as const,
    properties: {
      location: {
        type: "string",
        description: "City and state, e.g. 'San Francisco, CA'",
      },
      unit: { type: "string", enum: ["celsius", "fahrenheit"] },
    },
    required: ["location"],
  },
};

// Same tool, eager field stripped. Mirrors the proposed normalizeToolSchemas
// fix: remove `eager_input_streaming` before sending to Snowflake.
const TOOL_WITHOUT_EAGER = {
  name: "get_weather",
  description: "Get the current weather in a given location.",
  input_schema: TOOL_WITH_EAGER.input_schema,
};

const USER_MESSAGE = {
  role: "user" as const,
  content: "What's the weather in Boston, MA in fahrenheit? Use the tool.",
};

const MAX_TOKENS = 1024;

describe("live integration — eager_input_streaming on Snowflake Cortex", () => {
  if (!HAVE_CREDS) {
    test.skip("credentials missing (SNOWFLAKE_PAT / SNOWFLAKE_CORTEX_API_KEY + SNOWFLAKE_BASE_URL)", () => {});
    return;
  }

  // ───────────────────────────────────────────────────────────────────────
  // Sonnet 4.6
  // ───────────────────────────────────────────────────────────────────────

  describe("claude-sonnet-4-6", () => {
    test("1. tool with eager_input_streaming, NO beta headers", async () => {
      const r = await call(
        SONNET,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITH_EAGER],
          messages: [USER_MESSAGE],
        },
        null,
      );
      logResult("sonnet eager + no beta", r);
      console.log(
        `    → ${r.ok ? "ACCEPTED" : "REJECTED"} eager_input_streaming on Sonnet w/ no beta`,
      );
    }, 60000);

    test("2. tool with eager_input_streaming + token-efficient-tools beta", async () => {
      const r = await call(
        SONNET,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITH_EAGER],
          messages: [USER_MESSAGE],
        },
        [FLAG_TOKEN_EFFICIENT],
      );
      logResult("sonnet eager + token-efficient", r);
      console.log(
        `    → ${r.ok ? "ACCEPTED" : "REJECTED"} eager_input_streaming on Sonnet w/ token-efficient`,
      );
    }, 60000);

    test("3. tool with eager_input_streaming STRIPPED, no beta headers", async () => {
      const r = await call(
        SONNET,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITHOUT_EAGER],
          messages: [USER_MESSAGE],
        },
        null,
      );
      logResult("sonnet stripped + no beta", r);
      expect(r.status).toBe(200);
      console.log(
        "    → confirms stripping eager_input_streaming makes the request succeed on Sonnet",
      );
    }, 60000);

    // Bonus: try the legacy fine-grained beta as an alternate path.
    // Pi-ai falls back to this header when supportsEagerToolInputStreaming
    // is set to false.
    test("4. tool stripped + fine-grained-tool-streaming beta (legacy path)", async () => {
      const r = await call(
        SONNET,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITHOUT_EAGER],
          messages: [USER_MESSAGE],
        },
        [FLAG_FINE_GRAINED],
      );
      logResult("sonnet stripped + fine-grained", r);
      console.log(
        `    → fine-grained-tool-streaming beta ${r.ok ? "ACCEPTED" : "REJECTED"} by Snowflake`,
      );
    }, 60000);

    test("5. eager field present + fine-grained beta (does Cortex accept the field with FGTS beta?)", async () => {
      const r = await call(
        SONNET,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITH_EAGER],
          messages: [USER_MESSAGE],
        },
        [FLAG_FINE_GRAINED],
      );
      logResult("sonnet eager + fine-grained", r);
      console.log(
        `    → eager_input_streaming + fine-grained beta on Sonnet: ${r.ok ? "ACCEPTED" : "REJECTED"}`,
      );
    }, 60000);
  });

  // ───────────────────────────────────────────────────────────────────────
  // Haiku 4.5
  // ───────────────────────────────────────────────────────────────────────

  describe("claude-haiku-4-5", () => {
    test("1. tool with eager_input_streaming, NO beta headers", async () => {
      const r = await call(
        HAIKU,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITH_EAGER],
          messages: [USER_MESSAGE],
        },
        null,
      );
      logResult("haiku eager + no beta", r);
      console.log(
        `    → ${r.ok ? "ACCEPTED" : "REJECTED"} eager_input_streaming on Haiku w/ no beta`,
      );
      // This is the case that broke the gateway-health-check cron.
      // We expect a 400 here.
    }, 60000);

    test("2. tool with eager_input_streaming + token-efficient-tools beta", async () => {
      const r = await call(
        HAIKU,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITH_EAGER],
          messages: [USER_MESSAGE],
        },
        [FLAG_TOKEN_EFFICIENT],
      );
      logResult("haiku eager + token-efficient", r);
      console.log(
        `    → ${r.ok ? "ACCEPTED" : "REJECTED"} eager_input_streaming on Haiku w/ token-efficient (token-efficient is also rejected on Haiku per beta-headers.live.test.ts)`,
      );
    }, 60000);

    test("3. tool with eager_input_streaming STRIPPED, no beta headers", async () => {
      const r = await call(
        HAIKU,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITHOUT_EAGER],
          messages: [USER_MESSAGE],
        },
        null,
      );
      logResult("haiku stripped + no beta", r);
      expect(r.status).toBe(200);
      console.log(
        "    → confirms stripping eager_input_streaming fixes the cron crash on Haiku",
      );
    }, 60000);

    test("4. tool stripped + fine-grained-tool-streaming beta", async () => {
      const r = await call(
        HAIKU,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITHOUT_EAGER],
          messages: [USER_MESSAGE],
        },
        [FLAG_FINE_GRAINED],
      );
      logResult("haiku stripped + fine-grained", r);
      console.log(
        `    → fine-grained-tool-streaming beta ${r.ok ? "ACCEPTED" : "REJECTED"} on Haiku`,
      );
    }, 60000);

    test("5. eager field present + fine-grained beta", async () => {
      const r = await call(
        HAIKU,
        {
          max_tokens: MAX_TOKENS,
          tools: [TOOL_WITH_EAGER],
          messages: [USER_MESSAGE],
        },
        [FLAG_FINE_GRAINED],
      );
      logResult("haiku eager + fine-grained", r);
      console.log(
        `    → eager_input_streaming + fine-grained beta on Haiku: ${r.ok ? "ACCEPTED" : "REJECTED"}`,
      );
    }, 60000);
  });
});
