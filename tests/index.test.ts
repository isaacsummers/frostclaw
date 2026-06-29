import { describe, test, expect } from "bun:test";
import {
  fixTrailingAssistant,
  fixEmptyTextBlocks,
  normalizeThinkingBudget,
  clampMaxTokens,
  stripEagerInputStreaming,
  isClaudeModel,
  levelBudget,
  levelEffort,
} from "../src/transforms.js";

// ---------------------------------------------------------------------------
// fixTrailingAssistant
// ---------------------------------------------------------------------------

describe("fixTrailingAssistant", () => {
  test("empty array returns same reference", () => {
    const arr: unknown[] = [];
    expect(fixTrailingAssistant(arr)).toBe(arr);
  });

  test("no trailing assistant returns same reference", () => {
    const arr = [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" },
      { role: "user", content: "goodbye" },
    ];
    expect(fixTrailingAssistant(arr)).toBe(arr);
  });

  test("trailing assistant role strips last message", () => {
    const arr = [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" },
    ];
    const result = fixTrailingAssistant(arr);
    expect(result).not.toBe(arr);
    expect(result).toHaveLength(1);
    expect((result[0] as { role: string }).role).toBe("user");
  });

  test("only one assistant message returns empty array", () => {
    const arr = [{ role: "assistant", content: "hi" }];
    const result = fixTrailingAssistant(arr);
    expect(result).toHaveLength(0);
  });

  test("trailing user role is untouched (same reference)", () => {
    const arr = [
      { role: "assistant", content: "hi" },
      { role: "user", content: "bye" },
    ];
    expect(fixTrailingAssistant(arr)).toBe(arr);
  });
});

// ---------------------------------------------------------------------------
// fixEmptyTextBlocks
// ---------------------------------------------------------------------------

describe("fixEmptyTextBlocks", () => {
  test("no empty blocks returns same reference (fast path)", () => {
    const arr = [
      { role: "user", content: [{ type: "text", text: "hello" }] },
    ];
    expect(fixEmptyTextBlocks(arr)).toBe(arr);
  });

  test("message with empty content array is replaced with ZWS text block", () => {
    const arr = [{ role: "assistant", content: [] }];
    const result = fixEmptyTextBlocks(arr) as Array<{ content: Array<{ type: string; text: string }> }>;
    expect(result).not.toBe(arr);
    expect(result[0].content).toHaveLength(1);
    expect(result[0].content[0].type).toBe("text");
    expect(result[0].content[0].text).toBe("\u200b");
  });

  test("whitespace-only text block is replaced with ZWS", () => {
    const arr = [{ role: "user", content: [{ type: "text", text: "   " }] }];
    const result = fixEmptyTextBlocks(arr) as Array<{ content: Array<{ type: string; text: string }> }>;
    expect(result).not.toBe(arr);
    expect(result[0].content[0].text).toBe("\u200b");
  });

  test("message with normal text is untouched", () => {
    const arr = [{ role: "user", content: [{ type: "text", text: "hello world" }] }];
    expect(fixEmptyTextBlocks(arr)).toBe(arr);
  });

  test("mixed: only bad blocks fixed, good ones unchanged", () => {
    const arr = [
      { role: "user", content: [{ type: "text", text: "good" }] },
      { role: "assistant", content: [{ type: "text", text: "  " }] },
    ] as Array<{ role: string; content: Array<{ type: string; text: string }> }>;
    const result = fixEmptyTextBlocks(arr) as typeof arr;
    expect(result).not.toBe(arr);
    expect(result[0].content[0].text).toBe("good");
    expect(result[1].content[0].text).toBe("\u200b");
  });
});

// ---------------------------------------------------------------------------
// normalizeThinkingBudget
// ---------------------------------------------------------------------------

describe("normalizeThinkingBudget", () => {
  test('{ type: "disabled" } leaves payload unchanged', () => {
    const payload = { thinking: { type: "disabled" }, other: "val" };
    const before = JSON.stringify(payload);
    normalizeThinkingBudget(payload, "high");
    expect(JSON.stringify(payload)).toBe(before);
  });

  // --- bare adaptive (no thinkingLevel / undefined) ---
  // Native Anthropic API accepts { type: "adaptive" } with no extra fields and
  // lets the model self-regulate. Snowflake Cortex requires output_config.effort
  // alongside it, so frostclaw always injects effort — defaulting to "high" when
  // no level is specified.
  test('{ type: "adaptive" } + undefined thinkingLevel defaults effort to "high" (Snowflake requires effort; Anthropic native does not)', () => {
    const payload: Record<string, unknown> = { thinking: { type: "adaptive" } };
    normalizeThinkingBudget(payload, undefined);
    expect((payload.thinking as { type: string }).type).toBe("adaptive");
    expect((payload.output_config as { effort: string }).effort).toBe("high");
  });

  test('{ type: "adaptive" } + undefined thinkingLevel does NOT set budget_tokens (adaptive path never uses budget_tokens)', () => {
    const payload: Record<string, unknown> = { thinking: { type: "adaptive" } };
    normalizeThinkingBudget(payload, undefined);
    expect((payload.thinking as Record<string, unknown>).budget_tokens).toBeUndefined();
  });

  test('{ type: "adaptive" } preserves existing output_config fields when injecting effort', () => {
    const payload: Record<string, unknown> = {
      thinking: { type: "adaptive" },
      output_config: { format: "text", other_field: 42 },
    };
    normalizeThinkingBudget(payload, "medium");
    const oc = payload.output_config as Record<string, unknown>;
    expect(oc.effort).toBe("medium");
    expect(oc.format).toBe("text");
    expect(oc.other_field).toBe(42);
  });

  // --- enabled with undefined level ---
  test('{ type: "enabled" } + undefined thinkingLevel defaults budget_tokens to 16000', () => {
    const payload: Record<string, unknown> = { thinking: { type: "enabled", budget_tokens: 0 } };
    normalizeThinkingBudget(payload, undefined);
    expect((payload.thinking as { budget_tokens: number }).budget_tokens).toBe(16000);
    // enabled path never touches output_config
    expect(payload.output_config).toBeUndefined();
  });

  test('{ type: "adaptive" } + level "high" sets output_config.effort = "high"', () => {
    const payload: Record<string, unknown> = { thinking: { type: "adaptive" } };
    normalizeThinkingBudget(payload, "high");
    expect((payload.output_config as { effort: string }).effort).toBe("high");
  });

  test('{ type: "adaptive" } + level "low" sets effort = "low"', () => {
    const payload: Record<string, unknown> = { thinking: { type: "adaptive" } };
    normalizeThinkingBudget(payload, "low");
    expect((payload.output_config as { effort: string }).effort).toBe("low");
  });

  test('{ type: "adaptive" } + level "minimal" maps to effort = "low"', () => {
    const payload: Record<string, unknown> = { thinking: { type: "adaptive" } };
    normalizeThinkingBudget(payload, "minimal");
    expect((payload.output_config as { effort: string }).effort).toBe("low");
  });

  test('{ type: "enabled", budget_tokens: 1234 } + level "medium" overwrites budget to 8000', () => {
    const payload: Record<string, unknown> = { thinking: { type: "enabled", budget_tokens: 1234 } };
    normalizeThinkingBudget(payload, "medium");
    expect((payload.thinking as { type: string; budget_tokens: number }).budget_tokens).toBe(8000);
  });

  test('{ type: "enabled", budget_tokens: 1234 } + level "minimal" overwrites budget to 1024', () => {
    const payload: Record<string, unknown> = { thinking: { type: "enabled", budget_tokens: 1234 } };
    normalizeThinkingBudget(payload, "minimal");
    expect((payload.thinking as { type: string; budget_tokens: number }).budget_tokens).toBe(1024);
  });

  test("no thinking field leaves payload unchanged", () => {
    const payload: Record<string, unknown> = { model: "claude-test", max_tokens: 1000 };
    const before = JSON.stringify(payload);
    normalizeThinkingBudget(payload, "high");
    expect(JSON.stringify(payload)).toBe(before);
  });

  // adaptiveOnly = true (Opus 4.7/4.8): "enabled" is redirected to "adaptive"
  test('adaptiveOnly: { type: "enabled" } redirected to adaptive + effort "high"', () => {
    const payload: Record<string, unknown> = {
      thinking: { type: "enabled", budget_tokens: 1234 },
    };
    normalizeThinkingBudget(payload, "high", true);
    expect((payload.thinking as { type: string }).type).toBe("adaptive");
    expect((payload.thinking as Record<string, unknown>).budget_tokens).toBeUndefined();
    expect((payload.output_config as { effort: string }).effort).toBe("high");
  });

  test('adaptiveOnly: { type: "enabled" } + level "low" → adaptive + effort "low"', () => {
    const payload: Record<string, unknown> = {
      thinking: { type: "enabled", budget_tokens: 9999 },
    };
    normalizeThinkingBudget(payload, "low", true);
    expect((payload.thinking as { type: string }).type).toBe("adaptive");
    expect((payload.output_config as { effort: string }).effort).toBe("low");
  });

  test('adaptiveOnly: { type: "adaptive" } unaffected — stays adaptive + sets effort', () => {
    const payload: Record<string, unknown> = { thinking: { type: "adaptive" } };
    normalizeThinkingBudget(payload, "medium", true);
    expect((payload.thinking as { type: string }).type).toBe("adaptive");
    expect((payload.output_config as { effort: string }).effort).toBe("medium");
  });

  test('adaptiveOnly: { type: "disabled" } unaffected', () => {
    const payload = { thinking: { type: "disabled" }, other: "val" };
    const before = JSON.stringify(payload);
    normalizeThinkingBudget(payload, "high", true);
    expect(JSON.stringify(payload)).toBe(before);
  });

  test('adaptiveOnly defaults to false: { type: "enabled" } keeps budget_tokens path', () => {
    const payload: Record<string, unknown> = {
      thinking: { type: "enabled", budget_tokens: 1234 },
    };
    normalizeThinkingBudget(payload, "medium");
    expect((payload.thinking as { type: string }).type).toBe("enabled");
    expect(
      (payload.thinking as { budget_tokens: number }).budget_tokens,
    ).toBe(8000);
    expect(payload.output_config).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// clampMaxTokens
// ---------------------------------------------------------------------------

describe("clampMaxTokens", () => {
  test("max_tokens already above floor: unchanged", () => {
    const payload: Record<string, unknown> = { max_tokens: 5000 };
    clampMaxTokens(payload);
    expect(payload.max_tokens).toBe(5000);
  });

  test("max_tokens: 0 with no thinking: clamped to 1024", () => {
    const payload: Record<string, unknown> = { max_tokens: 0 };
    clampMaxTokens(payload);
    expect(payload.max_tokens).toBe(1024);
  });

  test("max_tokens: -1 with thinking enabled budget 4000: clamped to 5024", () => {
    const payload: Record<string, unknown> = {
      max_tokens: -1,
      thinking: { type: "enabled", budget_tokens: 4000 },
    };
    clampMaxTokens(payload);
    expect(payload.max_tokens).toBe(5024);
  });

  test("max_tokens: 100 with adaptive thinking: clamped to 4096", () => {
    const payload: Record<string, unknown> = {
      max_tokens: 100,
      thinking: { type: "adaptive" },
    };
    clampMaxTokens(payload);
    expect(payload.max_tokens).toBe(4096);
  });

  test("no max_tokens field: unchanged", () => {
    const payload: Record<string, unknown> = { model: "claude-test" };
    clampMaxTokens(payload);
    expect(payload.max_tokens).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// isClaudeModel
// ---------------------------------------------------------------------------

describe("isClaudeModel", () => {
  test('"claude-sonnet-4-6" is true', () => {
    expect(isClaudeModel("claude-sonnet-4-6")).toBe(true);
  });

  test('"claude-haiku-4-5" is true', () => {
    expect(isClaudeModel("claude-haiku-4-5")).toBe(true);
  });

  test('"CLAUDE-opus-4-7" is true (case insensitive)', () => {
    expect(isClaudeModel("CLAUDE-opus-4-7")).toBe(true);
  });

  test('"gpt-5-mini" is false', () => {
    expect(isClaudeModel("gpt-5-mini")).toBe(false);
  });

  test('"openai-gpt-5" is false', () => {
    expect(isClaudeModel("openai-gpt-5")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// levelBudget
// ---------------------------------------------------------------------------

describe("levelBudget", () => {
  test('"minimal" returns 1024', () => expect(levelBudget("minimal")).toBe(1024));
  test('"low" returns 4000', () => expect(levelBudget("low")).toBe(4000));
  test('"medium" returns 8000', () => expect(levelBudget("medium")).toBe(8000));
  test('"high" returns 16000', () => expect(levelBudget("high")).toBe(16000));
  test("undefined returns 16000 (default)", () => expect(levelBudget(undefined)).toBe(16000));
});

// ---------------------------------------------------------------------------
// levelEffort
// ---------------------------------------------------------------------------

describe("levelEffort", () => {
  test('"minimal" returns "low"', () => expect(levelEffort("minimal")).toBe("low"));
  test('"low" returns "low"', () => expect(levelEffort("low")).toBe("low"));
  test('"medium" returns "medium"', () => expect(levelEffort("medium")).toBe("medium"));
  test('"high" returns "high"', () => expect(levelEffort("high")).toBe("high"));
  test('"adaptive" returns "high"', () => expect(levelEffort("adaptive")).toBe("high"));
  test('undefined returns "high" (default)', () => expect(levelEffort(undefined)).toBe("high"));
});

// ---------------------------------------------------------------------------
// stripEagerInputStreaming
// ---------------------------------------------------------------------------

describe("stripEagerInputStreaming", () => {
  test("no tools field: no-op", () => {
    const payload: Record<string, unknown> = { messages: [] };
    stripEagerInputStreaming(payload);
    expect(payload).toEqual({ messages: [] });
  });

  test("tools without eager_input_streaming: no-op (same reference)", () => {
    const tool = { name: "bash", description: "run bash", input_schema: { type: "object" } };
    const payload: Record<string, unknown> = { tools: [tool] };
    stripEagerInputStreaming(payload);
    expect((payload.tools as unknown[])[0]).toBe(tool);
    expect(tool).toEqual({ name: "bash", description: "run bash", input_schema: { type: "object" } });
  });

  test("tool with eager_input_streaming alongside other fields: field removed, others kept", () => {
    const payload: Record<string, unknown> = {
      tools: [
        {
          name: "lookup",
          description: "Look something up",
          input_schema: { type: "object" },
          eager_input_streaming: true,
        },
      ],
    };
    stripEagerInputStreaming(payload);
    expect(payload.tools).toEqual([
      {
        name: "lookup",
        description: "Look something up",
        input_schema: { type: "object" },
      },
    ]);
  });

  test("tool with only eager_input_streaming: field removed, tool otherwise unchanged", () => {
    const payload: Record<string, unknown> = {
      tools: [{ name: "x", eager_input_streaming: true }],
    };
    stripEagerInputStreaming(payload);
    expect(payload.tools).toEqual([{ name: "x" }]);
  });

  test("multiple tools: only ones carrying the field are touched", () => {
    const payload: Record<string, unknown> = {
      tools: [
        { name: "a", eager_input_streaming: true },
        { name: "b" },
        { name: "c", eager_input_streaming: false },
      ],
    };
    stripEagerInputStreaming(payload);
    expect(payload.tools).toEqual([
      { name: "a" },
      { name: "b" },
      { name: "c" },
    ]);
  });

  test("non-array tools field: no-op", () => {
    const payload: Record<string, unknown> = { tools: "not-an-array" };
    stripEagerInputStreaming(payload);
    expect(payload.tools).toBe("not-an-array");
  });
});
