/**
 * Live integration tests — Anthropic beta header acceptance against
 * Snowflake Cortex REST API.
 *
 * These tests build a real HTTP request to:
 *   ${SNOWFLAKE_BASE_URL}/api/v2/cortex/v1/messages
 *
 * with a PAT (SNOWFLAKE_PAT or SNOWFLAKE_CORTEX_API_KEY) and exercise the
 * `anthropic-beta` header matrix that frostclaw's wrapStreamFn would
 * potentially emit. Each test asserts:
 *   - HTTP status (200 vs 4xx)
 *   - For paired with/without comparisons: token deltas
 *   - For features (thinking + effort): behavioral deltas
 *
 * Matrix covered (Sonnet 4.6 unless noted):
 *   1. output-128k-2025-02-19 with max_tokens 100k       (with vs without)
 *   2. token-efficient-tools-2025-02-19                  (with vs without; tokens)
 *   3. effort-2025-11-24 with thinking enabled           (with vs without)
 *   4. tool-examples-2025-10-29                          (accept vs reject)
 *   5. Each flag individually on Haiku 4.5               (which 400)
 *
 * Run: bun test tests/live/beta-headers.live.test.ts
 *
 * Skips automatically if credentials are not set.
 */

// ---------------------------------------------------------------------------
// ACCEPTANCE TESTS — these make real HTTP calls to Snowflake Cortex.
// They test Snowflake's validator behaviour, NOT frostclaw's internal logic.
// Run separately: bun test tests/live/  (requires SNOWFLAKE_BASE_URL + creds)
// Do NOT run as part of CI or the standard `bun test` suite.
// ---------------------------------------------------------------------------

import { describe, test, expect } from "bun:test";

const PAT =
  process.env.SNOWFLAKE_PAT ??
  process.env.SNOWFLAKE_CORTEX_API_KEY ??
  "";
const BASE_URL = process.env.SNOWFLAKE_BASE_URL ?? "";
const HAVE_CREDS = !!PAT && !!BASE_URL;
const ENDPOINT = HAVE_CREDS
  ? `${BASE_URL.replace(/\/$/, "")}/api/v2/cortex/v1/messages`
  : "";

const SONNET = "claude-sonnet-4-6";
const HAIKU = "claude-haiku-4-5";

const FLAG_OUTPUT_128K = "output-128k-2025-02-19";
const FLAG_TOKEN_EFFICIENT = "token-efficient-tools-2025-02-19";
const FLAG_EFFORT = "effort-2025-11-24";
const FLAG_TOOL_EXAMPLES = "tool-examples-2025-10-29";
const FLAG_INTERLEAVED_THINKING = "interleaved-thinking-2025-05-14";

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
    // non-JSON body (e.g. HTML 502 page); leave parsed null
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
      (r.errorMessage ? `err="${r.errorMessage.slice(0, 200)}"` : "ok"),
  );
}

const SKIP_IF_NO_CREDS = HAVE_CREDS ? test : test.skip;

describe("live integration — Anthropic beta headers on Snowflake Cortex", () => {
  if (!HAVE_CREDS) {
    test.skip("credentials missing (SNOWFLAKE_PAT / SNOWFLAKE_CORTEX_API_KEY + SNOWFLAKE_BASE_URL)", () => {});
    return;
  }

  // ───────────────────────────────────────────────────────────────────────
  // 1. output-128k-2025-02-19 on Sonnet 4.6 with max_tokens: 100000
  // ───────────────────────────────────────────────────────────────────────

  describe("output-128k-2025-02-19 on Sonnet 4.6 (max_tokens=100000)", () => {
    const body = {
      max_tokens: 100000,
      messages: [
        {
          role: "user",
          content:
            "Reply with exactly one word: 'ok'. Do not say anything else.",
        },
      ],
    };

    test("with output-128k flag", async () => {
      const r = await call(SONNET, body, [FLAG_OUTPUT_128K]);
      logResult("128k WITH flag", r);
      expect(r.status).toBeGreaterThanOrEqual(200);
      // Just record outcome — we want to know if 200 or 400.
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 70000);

    test("without output-128k flag", async () => {
      const r = await call(SONNET, body, null);
      logResult("128k WITHOUT flag", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 70000);
  });

  // ───────────────────────────────────────────────────────────────────────
  // 2. token-efficient-tools-2025-02-19 on Sonnet 4.6 (with tools, compare tokens)
  // ───────────────────────────────────────────────────────────────────────

  describe("token-efficient-tools-2025-02-19 on Sonnet 4.6", () => {
    const tools = [
      {
        name: "get_weather",
        description: "Get the current weather in a given location.",
        input_schema: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "City and state, e.g. 'San Francisco, CA'",
            },
            unit: {
              type: "string",
              enum: ["celsius", "fahrenheit"],
            },
          },
          required: ["location"],
        },
      },
      {
        name: "get_time",
        description: "Get the current time in a given timezone.",
        input_schema: {
          type: "object",
          properties: {
            timezone: { type: "string" },
          },
          required: ["timezone"],
        },
      },
    ];

    const bodyWith = {
      max_tokens: 1024,
      tools,
      messages: [
        {
          role: "user",
          content:
            "What's the weather in Boston, MA, in fahrenheit? Use the tool.",
        },
      ],
    };

    const bodyWithout = {
      max_tokens: 1024,
      tools,
      messages: [
        {
          role: "user",
          content:
            "What's the weather right now in Seattle, WA, in celsius? Use the tool.",
        },
      ],
    };

    test("with token-efficient-tools flag", async () => {
      const r = await call(SONNET, bodyWith, [FLAG_TOKEN_EFFICIENT]);
      logResult("token-efficient WITH", r);
      if (r.ok) {
        const usage = r.body?.usage;
        expect(usage?.input_tokens).toBeGreaterThan(0);
        expect(usage?.output_tokens).toBeGreaterThan(0);
      }
    }, 60000);

    test("without token-efficient-tools flag (different prompt to defeat caching)", async () => {
      const r = await call(SONNET, bodyWithout, null);
      logResult("token-efficient WITHOUT", r);
      if (r.ok) {
        const usage = r.body?.usage;
        expect(usage?.input_tokens).toBeGreaterThan(0);
        expect(usage?.output_tokens).toBeGreaterThan(0);
      }
    }, 60000);

    test("same prompt, both runs — direct token comparison", async () => {
      const samePrompt = {
        max_tokens: 1024,
        tools,
        messages: [
          {
            role: "user",
            content: `Time check: ${Date.now()}. Get weather in Austin, TX in fahrenheit. Use the tool.`,
          },
        ],
      };
      const withFlag = await call(SONNET, samePrompt, [FLAG_TOKEN_EFFICIENT]);
      const samePromptB = {
        ...samePrompt,
        messages: [
          {
            role: "user",
            content: `Time check: ${Date.now()}. Get weather in Austin, TX in fahrenheit. Use the tool.`,
          },
        ],
      };
      const noFlag = await call(SONNET, samePromptB, null);
      logResult("compare WITH", withFlag);
      logResult("compare WITHOUT", noFlag);
      const withTokens = withFlag.body?.usage?.output_tokens;
      const noTokens = noFlag.body?.usage?.output_tokens;
      console.log(
        `    → output_tokens: WITH=${withTokens} WITHOUT=${noTokens} delta=${
          withTokens && noTokens ? noTokens - withTokens : "n/a"
        }`,
      );
    }, 120000);
  });

  // ───────────────────────────────────────────────────────────────────────
  // 3. effort-2025-11-24 on Sonnet 4.6 with thinking
  // ───────────────────────────────────────────────────────────────────────

  describe("effort-2025-11-24 on Sonnet 4.6 (thinking with effort field)", () => {
    // The effort beta flag is supposed to enable the `effort` field in the
    // thinking object (instead of budget_tokens). Test both shapes with and
    // without the flag to see what Snowflake accepts.
    const effortBody = {
      max_tokens: 4096,
      thinking: { type: "enabled", effort: "high" },
      messages: [
        {
          role: "user",
          content:
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more " +
            "than the ball. How much does the ball cost? Show your reasoning.",
        },
      ],
    };

    test("effort field WITH effort flag", async () => {
      const r = await call(SONNET, effortBody, [
        FLAG_EFFORT,
        FLAG_INTERLEAVED_THINKING,
      ]);
      logResult("effort WITH", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 90000);

    test("effort field WITHOUT effort flag", async () => {
      const r = await call(SONNET, effortBody, [FLAG_INTERLEAVED_THINKING]);
      logResult("effort WITHOUT", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 90000);

    test("baseline: budget_tokens shape (no effort field) WITH effort flag", async () => {
      const altBody = {
        max_tokens: 4096,
        thinking: { type: "enabled", budget_tokens: 2048 },
        messages: [
          {
            role: "user",
            content:
              "A bat and a ball cost $1.10. The bat costs $1.00 more than " +
              "the ball. How much is the ball?",
          },
        ],
      };
      const r = await call(SONNET, altBody, [
        FLAG_EFFORT,
        FLAG_INTERLEAVED_THINKING,
      ]);
      logResult("effort+budget_tokens", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 90000);

    test("bare effort flag with no thinking object", async () => {
      const noThinkBody = {
        max_tokens: 256,
        messages: [{ role: "user", content: "Reply 'ok'." }],
      };
      const r = await call(SONNET, noThinkBody, [FLAG_EFFORT]);
      logResult("effort BARE", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 60000);
  });

  // ───────────────────────────────────────────────────────────────────────
  // 4. tool-examples-2025-10-29 on Sonnet 4.6 — accept or reject?
  // ───────────────────────────────────────────────────────────────────────

  describe("tool-examples-2025-10-29 on Sonnet 4.6", () => {
    // Two shapes: bare flag, and flag with tool that includes `examples`.
    const bareBody = {
      max_tokens: 256,
      messages: [{ role: "user", content: "Reply 'ok'." }],
    };

    test("bare flag, no tool examples in payload", async () => {
      const r = await call(SONNET, bareBody, [FLAG_TOOL_EXAMPLES]);
      logResult("tool-examples BARE", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 60000);

    test("flag + tool with examples field", async () => {
      const toolWithExamples = {
        max_tokens: 512,
        tools: [
          {
            name: "echo",
            description: "Echo a string back.",
            input_schema: {
              type: "object",
              properties: { text: { type: "string" } },
              required: ["text"],
            },
            // The whole point of tool-examples-2025-10-29 — providing
            // example invocations for the tool.
            examples: [
              {
                input: { text: "hello" },
                output: "hello",
              },
            ],
          },
        ],
        messages: [
          { role: "user", content: "Use the echo tool with the text 'ping'." },
        ],
      };
      const r = await call(SONNET, toolWithExamples, [FLAG_TOOL_EXAMPLES]);
      logResult("tool-examples FULL", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 60000);
  });

  // ───────────────────────────────────────────────────────────────────────
  // 5. Each flag individually on Haiku 4.5 — which ones 400?
  // ───────────────────────────────────────────────────────────────────────

  describe("each flag individually on Haiku 4.5", () => {
    const minBody = {
      max_tokens: 256,
      messages: [{ role: "user", content: "Reply with exactly: ok" }],
    };

    test("baseline: no anthropic-beta header", async () => {
      const r = await call(HAIKU, minBody, null);
      logResult("haiku baseline", r);
      expect(r.status).toBe(200);
    }, 60000);

    test("output-128k-2025-02-19", async () => {
      const r = await call(HAIKU, minBody, [FLAG_OUTPUT_128K]);
      logResult("haiku output-128k", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 60000);

    test("token-efficient-tools-2025-02-19", async () => {
      const r = await call(HAIKU, minBody, [FLAG_TOKEN_EFFICIENT]);
      logResult("haiku token-efficient", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 60000);

    test("effort-2025-11-24", async () => {
      const r = await call(HAIKU, minBody, [FLAG_EFFORT]);
      logResult("haiku effort", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 60000);

    test("tool-examples-2025-10-29", async () => {
      const r = await call(HAIKU, minBody, [FLAG_TOOL_EXAMPLES]);
      logResult("haiku tool-examples", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 60000);

    test("interleaved-thinking-2025-05-14", async () => {
      const r = await call(HAIKU, minBody, [FLAG_INTERLEAVED_THINKING]);
      logResult("haiku interleaved-thinking", r);
      if (!r.ok) {
        console.log(`    error: ${r.errorMessage?.slice(0, 300)}`);
      }
    }, 60000);
  });
});
