/**
 * Cache control passthrough + cache pricing tests.
 *
 * Covers:
 *   1. cache_control passthrough — none of the payload transforms strip
 *      `cache_control` markers from message content blocks.
 *   2. Catalog cache pricing — every Claude model has non-zero cacheRead and
 *      cacheWrite rates.
 *   3. Debug logging source sentinel — the message_start SSE log and the
 *      result log include cache token fields.
 */

import { describe, test, expect } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  fixEmptyTextBlocks,
  fixTrailingAssistant,
  normalizeThinkingBudget,
  clampMaxTokens,
} from "../src/transforms.js";
import {
  buildModelCatalog,
  CLAUDE_MODELS,
} from "../src/catalog.js";

// ---------------------------------------------------------------------------
// Suite 1 — cache_control passthrough through every payload transform
//
// Snowflake Cortex supports the ephemeral 5-minute TTL for prompt caching.
// `cache_control: { type: "ephemeral" }` is set by OpenClaw on content blocks
// to mark cache breakpoints. None of frostclaw's transforms should strip it.
// ---------------------------------------------------------------------------

const CACHE_CONTROL = { type: "ephemeral" } as const;

describe("cache_control passthrough — fixEmptyTextBlocks", () => {
  test("cache_control preserved on non-empty text block (fast path, same ref)", () => {
    const msg = {
      role: "user",
      content: [{ type: "text", text: "hello", cache_control: CACHE_CONTROL }],
    };
    const arr = [msg];
    const result = fixEmptyTextBlocks(arr);
    expect(result).toBe(arr); // fast path — no allocation
    const block = (result[0] as typeof msg).content[0];
    expect(block.cache_control).toEqual(CACHE_CONTROL);
  });

  test("cache_control preserved on empty/whitespace text block (slow path fix)", () => {
    const msg = {
      role: "user",
      content: [{ type: "text", text: "  ", cache_control: CACHE_CONTROL }],
    };
    const result = fixEmptyTextBlocks([msg]) as Array<{
      content: Array<{ type: string; text: string; cache_control?: typeof CACHE_CONTROL }>;
    }>;
    expect(result[0].content[0].text).toBe("\u200b");
    expect(result[0].content[0].cache_control).toEqual(CACHE_CONTROL);
  });

  test("cache_control preserved on image block (not touched by text fixer)", () => {
    const msg = {
      role: "user",
      content: [
        { type: "image", source: { type: "base64", data: "abc" }, cache_control: CACHE_CONTROL },
      ],
    };
    const result = fixEmptyTextBlocks([msg]) as typeof msg[];
    expect((result[0] as typeof msg).content[0].cache_control).toEqual(CACHE_CONTROL);
  });

  test("cache_control preserved on tool_result block", () => {
    const msg = {
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "toolu_123",
          content: "ok",
          cache_control: CACHE_CONTROL,
        },
      ],
    };
    const result = fixEmptyTextBlocks([msg]);
    const block = (result[0] as typeof msg).content[0];
    expect(block.cache_control).toEqual(CACHE_CONTROL);
  });

  test("empty content array: rebuilt with ZWS block; other message-level fields unaffected", () => {
    // fixEmptyTextBlocks only rebuilds the content array — other keys are spread.
    const msg = { role: "user", content: [], extra: "keep_me" };
    const result = fixEmptyTextBlocks([msg]) as Array<{
      extra: string;
      content: Array<{ type: string; text: string }>;
    }>;
    expect(result[0].extra).toBe("keep_me");
    expect(result[0].content[0].type).toBe("text");
    expect(result[0].content[0].text).toBe("\u200b");
  });
});

describe("cache_control passthrough — fixTrailingAssistant", () => {
  test("cache_control on user message is unaffected when no trailing assistant", () => {
    const arr = [
      {
        role: "user",
        content: [{ type: "text", text: "hi", cache_control: CACHE_CONTROL }],
      },
    ];
    expect(fixTrailingAssistant(arr)).toBe(arr);
  });

  test("cache_control on remaining messages preserved after trailing assistant stripped", () => {
    const userMsg = {
      role: "user",
      content: [{ type: "text", text: "hi", cache_control: CACHE_CONTROL }],
    };
    const assistantMsg = { role: "assistant", content: "bye" };
    const result = fixTrailingAssistant([userMsg, assistantMsg]) as typeof userMsg[];
    expect(result).toHaveLength(1);
    expect(result[0].content[0].cache_control).toEqual(CACHE_CONTROL);
  });
});

describe("cache_control passthrough — normalizeThinkingBudget", () => {
  test("cache_control on message content blocks is untouched", () => {
    // normalizeThinkingBudget only mutates 'thinking' and 'output_config'.
    const payload: Record<string, unknown> = {
      thinking: { type: "adaptive" },
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "hi", cache_control: CACHE_CONTROL }],
        },
      ],
    };
    normalizeThinkingBudget(payload, "high");
    const block = (payload.messages as Array<{
      content: Array<{ cache_control: typeof CACHE_CONTROL }>;
    }>)[0].content[0];
    expect(block.cache_control).toEqual(CACHE_CONTROL);
  });
});

describe("cache_control passthrough — clampMaxTokens", () => {
  test("cache_control on message content blocks is untouched", () => {
    const payload: Record<string, unknown> = {
      max_tokens: 0,
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "hi", cache_control: CACHE_CONTROL }],
        },
      ],
    };
    clampMaxTokens(payload);
    expect(payload.max_tokens).toBe(1024); // clamped
    const block = (payload.messages as Array<{
      content: Array<{ cache_control: typeof CACHE_CONTROL }>;
    }>)[0].content[0];
    expect(block.cache_control).toEqual(CACHE_CONTROL);
  });
});

// ---------------------------------------------------------------------------
// Suite 2 — Catalog cache pricing
//
// Every Claude model must have non-zero cacheRead and cacheWrite rates so
// OpenClaw can calculate cache savings in the cost breakdown.
// ---------------------------------------------------------------------------

describe("catalog cache pricing — Claude models", () => {
  const catalog = buildModelCatalog();
  const claudeEntries = catalog.filter((m) => m.id.startsWith("claude-"));

  test("catalog has at least one Claude entry", () => {
    expect(claudeEntries.length).toBeGreaterThan(0);
  });

  test("every Claude model has cacheRead > 0", () => {
    for (const m of claudeEntries) {
      expect(
        m.cost.cacheRead,
        `claude model "${m.id}" missing cacheRead pricing`,
      ).toBeGreaterThan(0);
    }
  });

  test("every Claude model has cacheWrite > 0", () => {
    for (const m of claudeEntries) {
      expect(
        m.cost.cacheWrite,
        `claude model "${m.id}" missing cacheWrite pricing`,
      ).toBeGreaterThan(0);
    }
  });

  test("cacheWrite > cacheRead for all Claude models (writes are more expensive)", () => {
    for (const m of claudeEntries) {
      expect(
        m.cost.cacheWrite,
        `claude model "${m.id}": cacheWrite should exceed cacheRead`,
      ).toBeGreaterThan(m.cost.cacheRead);
    }
  });

  test("CLAUDE_MODELS spec count matches claude catalog entries", () => {
    expect(claudeEntries.length).toBe(CLAUDE_MODELS.length);
  });
});

// ---------------------------------------------------------------------------
// Suite 3 — Debug logging source sentinel
//
// FROSTCLAW_DEBUG_REQUESTS=1 logs SSE events. The `message_start` event
// should log cache token fields so operators can see prompt-caching activity.
// The result line (← Snowflake) should also report them for audit scripts.
// ---------------------------------------------------------------------------

describe("debug logging — cache token fields in source", () => {
  const indexSrc = readFileSync(
    join(import.meta.dir, "..", "index.ts"),
    "utf8",
  );

  test("message_start SSE debug log includes cache_read_input_tokens", () => {
    // Check the message_start handler logs cache read token count.
    const match = indexSrc.match(
      /evType === "message_start"[\s\S]{0,1000}cache_read_input_tokens/,
    );
    expect(match).not.toBeNull();
  });

  test("message_start SSE debug log includes cache_creation_input_tokens", () => {
    const match = indexSrc.match(
      /evType === "message_start"[\s\S]{0,1000}cache_creation_input_tokens/,
    );
    expect(match).not.toBeNull();
  });

  test("stream result debug log (\u2190 Snowflake) includes cache_read_input_tokens", () => {
    // The ← Snowflake result line is the single-line audit summary.
    const match = indexSrc.match(
      /← Snowflake[\s\S]{0,600}cache_read_input_tokens/,
    );
    expect(match).not.toBeNull();
  });

  test("stream result debug log (\u2190 Snowflake) includes cache_creation_input_tokens", () => {
    const match = indexSrc.match(
      /← Snowflake[\s\S]{0,600}cache_creation_input_tokens/,
    );
    expect(match).not.toBeNull();
  });
});
