/**
 * Catalog compat declaration + onPayload defensive-strip tests.
 *
 * Imports directly from src/catalog.ts and src/onpayload.ts — no openclaw
 * dependency, no mocking, no source sentinels. These files have zero openclaw
 * imports so they resolve cleanly at test time.
 */

import { describe, test, expect } from "bun:test";
import {
  buildModelCatalog,
  CLAUDE_MODELS,
  OPENAI_MODELS,
  OPEN_SOURCE_MODELS,
} from "../src/catalog.js";
import { applyEagerInputStreamingStrip } from "../src/onpayload.js";

// ---------------------------------------------------------------------------
// Suite 1 — Catalog compat declaration
// ---------------------------------------------------------------------------

describe("catalog compat — supportsEagerToolInputStreaming", () => {
  const catalog = buildModelCatalog();

  test("every Claude model has compat.supportsEagerToolInputStreaming === false", () => {
    const claudeEntries = catalog.filter((m) => m.id.startsWith("claude-"));
    expect(claudeEntries.length).toBeGreaterThan(0);
    for (const m of claudeEntries) {
      expect(
        (m.compat as Record<string, unknown> | undefined)
          ?.supportsEagerToolInputStreaming,
        `expected claude model "${m.id}" to have supportsEagerToolInputStreaming: false`,
      ).toBe(false);
    }
  });

  test("non-Claude models do NOT set supportsEagerToolInputStreaming", () => {
    const nonClaude = catalog.filter((m) => !m.id.startsWith("claude-"));
    expect(nonClaude.length).toBeGreaterThan(0);
    for (const m of nonClaude) {
      const flag = (m.compat as Record<string, unknown> | undefined)
        ?.supportsEagerToolInputStreaming;
      expect(
        flag,
        `non-Claude model "${m.id}" should not set supportsEagerToolInputStreaming (got ${flag})`,
      ).toBeUndefined();
    }
  });

  test("CLAUDE_MODELS spec list covers all catalog claude- entries", () => {
    const catalogClaudeIds = new Set(
      catalog.filter((m) => m.id.startsWith("claude-")).map((m) => m.id),
    );
    for (const spec of CLAUDE_MODELS) {
      expect(catalogClaudeIds.has(spec.id)).toBe(true);
    }
  });

  test("catalog contains all OPENAI_MODELS and OPEN_SOURCE_MODELS", () => {
    const allIds = new Set(catalog.map((m) => m.id));
    for (const spec of OPENAI_MODELS) {
      expect(allIds.has(spec.id)).toBe(true);
    }
    for (const spec of OPEN_SOURCE_MODELS) {
      expect(allIds.has(spec.id)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// Suite 2 — applyEagerInputStreamingStrip
// ---------------------------------------------------------------------------

function makePayload(withFlag: boolean) {
  return {
    tools: [
      {
        name: "test_tool",
        description: "A test tool",
        input_schema: { type: "object", properties: {} },
        custom: {
          ...(withFlag ? { eager_input_streaming: true } : {}),
          other_field: "keep_me",
        },
      },
    ],
  };
}

describe("applyEagerInputStreamingStrip — Claude strips, non-Claude unchanged", () => {
  test("Claude Sonnet: strips eager_input_streaming from tool custom", () => {
    const payload = makePayload(true);
    const result = applyEagerInputStreamingStrip("claude-sonnet-4-6", payload) as typeof payload;
    expect(
      (result.tools[0].custom as Record<string, unknown>).eager_input_streaming,
    ).toBeUndefined();
    expect(
      (result.tools[0].custom as Record<string, unknown>).other_field,
    ).toBe("keep_me");
  });

  test("Claude Haiku: strips eager_input_streaming from tool custom", () => {
    const payload = makePayload(true);
    const result = applyEagerInputStreamingStrip("claude-haiku-4-5", payload) as typeof payload;
    expect(
      (result.tools[0].custom as Record<string, unknown>).eager_input_streaming,
    ).toBeUndefined();
  });

  test("Claude Opus: strips eager_input_streaming from tool custom", () => {
    const payload = makePayload(true);
    const result = applyEagerInputStreamingStrip("claude-opus-4-7", payload) as typeof payload;
    expect(
      (result.tools[0].custom as Record<string, unknown>).eager_input_streaming,
    ).toBeUndefined();
  });

  test("openai-gpt-5: payload returned unchanged (same reference)", () => {
    const payload = makePayload(true);
    const result = applyEagerInputStreamingStrip("openai-gpt-5", payload);
    expect(result).toBe(payload);
  });

  test("llama3.3-70b: payload returned unchanged (same reference)", () => {
    const payload = makePayload(true);
    const result = applyEagerInputStreamingStrip("llama3.3-70b", payload);
    expect(result).toBe(payload);
  });

  test("Claude Sonnet with no tools: payload untouched", () => {
    const payload = { messages: [{ role: "user", content: "hello" }] };
    const result = applyEagerInputStreamingStrip("claude-sonnet-4-6", payload);
    expect((result as Record<string, unknown>).tools).toBeUndefined();
  });

  test("null payload: returned as-is for Claude model", () => {
    expect(applyEagerInputStreamingStrip("claude-haiku-4-5", null)).toBeNull();
  });

  test("undefined payload: returned as-is for Claude model", () => {
    expect(applyEagerInputStreamingStrip("claude-haiku-4-5", undefined)).toBeUndefined();
  });

  test("multiple tools: only the one with the field is touched", () => {
    const payload = {
      tools: [
        { custom: { eager_input_streaming: true, keep: 1 } },
        { custom: { keep: 2 } },
        { name: "no_custom" },
      ],
    };
    const result = applyEagerInputStreamingStrip("claude-sonnet-4-6", payload) as typeof payload;
    expect(
      (result.tools[0].custom as Record<string, unknown>).eager_input_streaming,
    ).toBeUndefined();
    expect((result.tools[0].custom as Record<string, unknown>).keep).toBe(1);
    expect((result.tools[1].custom as Record<string, unknown>).keep).toBe(2);
  });
});
