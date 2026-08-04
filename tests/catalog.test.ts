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
  findCatalogEntry,
  isAdaptiveOnly,
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
// Suite 1b — adaptiveOnly flag (Opus 4.7/4.8 reject manual extended thinking)
// ---------------------------------------------------------------------------

describe("catalog adaptiveOnly flag", () => {
  test("claude-opus-4-8 spec has adaptiveOnly: true", () => {
    const spec = CLAUDE_MODELS.find((m) => m.id === "claude-opus-4-8");
    expect(spec?.adaptiveOnly).toBe(true);
  });

  test("claude-opus-4-7 spec has adaptiveOnly: true", () => {
    const spec = CLAUDE_MODELS.find((m) => m.id === "claude-opus-4-7");
    expect(spec?.adaptiveOnly).toBe(true);
  });

  test("claude-sonnet-4-6 spec does NOT set adaptiveOnly", () => {
    const spec = CLAUDE_MODELS.find((m) => m.id === "claude-sonnet-4-6");
    expect(spec?.adaptiveOnly).toBeUndefined();
  });

  test("claude-opus-4-6 spec does NOT set adaptiveOnly (enabled still works)", () => {
    const spec = CLAUDE_MODELS.find((m) => m.id === "claude-opus-4-6");
    expect(spec?.adaptiveOnly).toBeUndefined();
  });

  test("catalog entries propagate adaptiveOnly for Opus 4.7/4.8", () => {
    expect(findCatalogEntry("claude-opus-4-8")?.adaptiveOnly).toBe(true);
    expect(findCatalogEntry("claude-opus-4-7")?.adaptiveOnly).toBe(true);
    expect(findCatalogEntry("claude-sonnet-4-6")?.adaptiveOnly).toBeUndefined();
  });

  test("isAdaptiveOnly true for Opus 4.7/4.8 and claude-4-opus, false otherwise", () => {
    expect(isAdaptiveOnly("claude-opus-4-8")).toBe(true);
    expect(isAdaptiveOnly("claude-opus-4-7")).toBe(true);
    expect(isAdaptiveOnly("claude-sonnet-4-6")).toBe(false);
    expect(isAdaptiveOnly("claude-opus-4-6")).toBe(false);
  });

  test("isAdaptiveOnly strips the snowflake-cortex/ prefix", () => {
    expect(isAdaptiveOnly("snowflake-cortex/claude-opus-4-8")).toBe(true);
    expect(isAdaptiveOnly("snowflake-cortex/claude-sonnet-4-6")).toBe(false);
  });

  test("isAdaptiveOnly false for unknown model id", () => {
    expect(isAdaptiveOnly("does-not-exist")).toBe(false);
  });

  // --- Full coverage: every named Claude model's adaptiveOnly status ---
  // This table is the source of truth for what frostclaw sends to Snowflake.
  // adaptiveOnly: true  → Cortex rejects { type: "enabled", budget_tokens: N } with HTTP 400;
  //                        frostclaw silently redirects to { type: "adaptive" } + output_config.effort.
  // adaptiveOnly: false → Both enabled (budget_tokens) and adaptive (effort) paths work on Cortex.
  // reasoning: false    → No thinking support at all; thinking fields must not be sent.
  describe("per-model adaptiveOnly ground truth", () => {
    const cases: Array<{ id: string; adaptiveOnly: boolean; reasoning: boolean }> = [
      // Opus — newer models (4.7+) dropped budget_tokens support on Cortex
      { id: "claude-opus-4-8",               adaptiveOnly: true,  reasoning: true  },
      { id: "claude-opus-4-7",               adaptiveOnly: true,  reasoning: true  },
      { id: "claude-opus-4-6",               adaptiveOnly: false, reasoning: true  },
      { id: "claude-opus-4-5",               adaptiveOnly: false, reasoning: true  },
      // Sonnet — 5 is adaptive-only; 4.x support full reasoning levels
      { id: "claude-sonnet-5",               adaptiveOnly: true,  reasoning: true  },
      { id: "claude-sonnet-4-6",             adaptiveOnly: false, reasoning: true  },
      { id: "claude-sonnet-4-5",             adaptiveOnly: false, reasoning: true  },
      // Haiku — no reasoning
      { id: "claude-haiku-4-5",              adaptiveOnly: false, reasoning: false },
    ];

    for (const { id, adaptiveOnly, reasoning } of cases) {
      test(`${id}: adaptiveOnly=${adaptiveOnly}, reasoning=${reasoning}`, () => {
        const spec = CLAUDE_MODELS.find((m) => m.id === id);
        expect(spec, `model "${id}" not found in CLAUDE_MODELS`).toBeDefined();
        if (adaptiveOnly) {
          expect(spec?.adaptiveOnly).toBe(true);
        } else {
          expect(spec?.adaptiveOnly).toBeUndefined();
        }
        expect(spec?.reasoning).toBe(reasoning);
      });
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
        ...(withFlag ? { eager_input_streaming: true } : {}),
        other_field: "keep_me",
      },
    ],
  };
}

describe("applyEagerInputStreamingStrip — Claude strips, non-Claude unchanged", () => {
  test("Claude Sonnet: strips eager_input_streaming from tool top-level", () => {
    const payload = makePayload(true);
    const result = applyEagerInputStreamingStrip("claude-sonnet-4-6", payload) as typeof payload;
    expect(
      (result.tools[0] as Record<string, unknown>).eager_input_streaming,
    ).toBeUndefined();
    expect(
      (result.tools[0] as Record<string, unknown>).other_field,
    ).toBe("keep_me");
  });

  test("Claude Haiku: strips eager_input_streaming from tool top-level", () => {
    const payload = makePayload(true);
    const result = applyEagerInputStreamingStrip("claude-haiku-4-5", payload) as typeof payload;
    expect(
      (result.tools[0] as Record<string, unknown>).eager_input_streaming,
    ).toBeUndefined();
  });

  test("Claude Opus: strips eager_input_streaming from tool top-level", () => {
    const payload = makePayload(true);
    const result = applyEagerInputStreamingStrip("claude-opus-4-7", payload) as typeof payload;
    expect(
      (result.tools[0] as Record<string, unknown>).eager_input_streaming,
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
        { name: "a", eager_input_streaming: true, keep: 1 },
        { name: "b", keep: 2 },
        { name: "c" },
      ],
    };
    const result = applyEagerInputStreamingStrip("claude-sonnet-4-6", payload) as typeof payload;
    expect(
      (result.tools[0] as Record<string, unknown>).eager_input_streaming,
    ).toBeUndefined();
    expect((result.tools[0] as Record<string, unknown>).keep).toBe(1);
    expect((result.tools[1] as Record<string, unknown>).keep).toBe(2);
  });
});
