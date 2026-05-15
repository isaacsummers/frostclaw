/**
 * Catalog compat declaration + onPayload defensive-strip integration tests.
 *
 * Why source-level + replication, not direct import:
 *   index.ts pulls in `openclaw/*` modules that are external at runtime
 *   (configured via `--external 'openclaw'` in the build script). Importing
 *   index.ts directly into a unit test fails with "Cannot find module
 *   'openclaw/plugin-sdk/plugin-entry'". The same constraint already shapes
 *   tests/beta-headers.test.ts, which uses readFileSync sentinels combined
 *   with verbatim re-implementation of the assembly logic.
 *
 * This file follows the same approach for two related concerns:
 *
 *   Suite 1 — Catalog compat declaration
 *     The frostclaw plugin must set `compat.supportsEagerToolInputStreaming:
 *     false` on every Claude model so pi-ai's Anthropic provider does not
 *     inject `eager_input_streaming: true` into tool definitions (which
 *     Snowflake Cortex's request validator rejects with HTTP 400). Non-Claude
 *     model definitions must NOT set this compat flag — they go through the
 *     OpenAI Chat Completions path where the flag is irrelevant.
 *
 *   Suite 2 — wrapStreamFn / onPayload integration (defensive strip)
 *     wrapStreamFn installs an onPayload hook that, for Claude models, calls
 *     stripEagerInputStreaming(record) on the outbound request body. This is
 *     the second line of defense behind the catalog compat flag: even if the
 *     compat declaration somehow fails to take effect, the strip still
 *     scrubs the field before the request leaves frostclaw. The behaviour is
 *     gated on isClaudeModel(model.id); non-Claude routes do not strip.
 *
 *     The actual onPayload function is a closure inside wrapStreamFn and
 *     cannot be imported. The test reproduces its strip-relevant branch
 *     verbatim, using the production stripEagerInputStreaming and
 *     isClaudeModel functions imported from src/transforms.ts. The result
 *     is a behavioural unit test of the integration path: input payload +
 *     model -> expected scrubbing decision.
 */

import { describe, test, expect } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  isClaudeModel,
  stripEagerInputStreaming,
} from "../src/transforms.js";

// ---------------------------------------------------------------------------
// Source under test (read once for source-level sentinels).
// ---------------------------------------------------------------------------

const indexPath = join(import.meta.dir, "..", "index.ts");
const indexSrc = readFileSync(indexPath, "utf8");

// ---------------------------------------------------------------------------
// Suite 1 — Catalog compat declaration
//
// The Claude builder body is replicated below. The replication is then
// verified against index.ts via a source sentinel, so any drift in the
// production builder fails this test loudly rather than silently testing
// the wrong shape.
// ---------------------------------------------------------------------------

interface CortexModelSpec {
  id: string;
  name: string;
  reasoning: boolean;
  contextWindow: number;
  maxTokens: number;
  input: Array<"text" | "image">;
}

// Mirrors the production CLAUDE_MODELS list (subset is fine — the assertion
// is "every claude- prefixed id sets the compat flag"). We exercise a
// representative sample including reasoning-on (Sonnet/Opus) and
// reasoning-off (Haiku) variants.
const CLAUDE_SAMPLE: CortexModelSpec[] = [
  { id: "claude-4-opus", name: "Claude 4 Opus", reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-opus-4-7", name: "Claude Opus 4.7", reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-haiku-4-5", name: "Claude Haiku 4.5", reasoning: false, contextWindow: 200_000, maxTokens: 16_384, input: ["text", "image"] },
  { id: "claude-3-7-sonnet", name: "Claude 3.7 Sonnet", reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
];

const NON_CLAUDE_SAMPLE: CortexModelSpec[] = [
  { id: "openai-gpt-5", name: "OpenAI GPT-5", reasoning: false, contextWindow: 200_000, maxTokens: 128_000, input: ["text"] },
  { id: "openai-gpt-4.1", name: "OpenAI GPT-4.1", reasoning: false, contextWindow: 1_000_000, maxTokens: 32_000, input: ["text"] },
  { id: "llama3.3-70b", name: "Llama 3.3 70B", reasoning: false, contextWindow: 128_000, maxTokens: 8_192, input: ["text"] },
  { id: "mistral-large2", name: "Mistral Large 2", reasoning: false, contextWindow: 128_000, maxTokens: 8_192, input: ["text"] },
  { id: "deepseek-r1", name: "DeepSeek R1", reasoning: true, contextWindow: 32_768, maxTokens: 8_192, input: ["text"] },
];

// Replicated builders. Cost / header / baseUrl detail is irrelevant to the
// compat assertion — only the `compat` literal matters here.
function fakeBuildClaudeModelDef(spec: CortexModelSpec): Record<string, unknown> {
  return {
    id: spec.id,
    name: spec.name,
    api: "anthropic-messages",
    reasoning: spec.reasoning,
    input: spec.input,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: { supportsTools: true, supportsEagerToolInputStreaming: false },
  };
}

function fakeBuildOpenAIModelDef(spec: CortexModelSpec): Record<string, unknown> {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions",
    reasoning: spec.reasoning,
    input: spec.input,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: true,
      maxTokensField: "max_completion_tokens",
      supportsUsageInStreaming: true,
    },
  };
}

function fakeBuildOpenSourceModelDef(spec: CortexModelSpec): Record<string, unknown> {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions",
    reasoning: spec.reasoning,
    input: spec.input,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: false,
      maxTokensField: "max_completion_tokens",
      supportsUsageInStreaming: true,
    },
  };
}

describe("Suite 1: catalog compat — Claude models", () => {
  test("every claude- prefixed id sets compat.supportsEagerToolInputStreaming === false", () => {
    for (const spec of CLAUDE_SAMPLE) {
      expect(spec.id.startsWith("claude-")).toBe(true);
      const def = fakeBuildClaudeModelDef(spec);
      const compat = def.compat as Record<string, unknown> | undefined;
      expect(compat).toBeDefined();
      expect(compat!.supportsEagerToolInputStreaming).toBe(false);
    }
  });

  test("compat also retains supportsTools: true on Claude models", () => {
    for (const spec of CLAUDE_SAMPLE) {
      const def = fakeBuildClaudeModelDef(spec);
      const compat = def.compat as Record<string, unknown>;
      expect(compat.supportsTools).toBe(true);
    }
  });

  // Source sentinel — drift detection. If the production builder changes
  // shape (e.g. someone removes the flag or flips it to true), the
  // replicated builder above is now lying. This regex pins the literal.
  test("source sentinel — buildClaudeModelDef body declares supportsEagerToolInputStreaming: false", () => {
    // Find buildClaudeModelDef and assert its compat literal matches.
    const m = indexSrc.match(
      /function\s+buildClaudeModelDef\s*\([^)]*\)\s*:\s*ModelDefinitionConfig\s*\{[\s\S]*?\n\}/,
    );
    expect(m).not.toBeNull();
    const body = m![0];
    expect(body).toMatch(/supportsEagerToolInputStreaming\s*:\s*false/);
    expect(body).toMatch(/supportsTools\s*:\s*true/);
  });
});

describe("Suite 1: catalog compat — non-Claude models", () => {
  test("OpenAI models do NOT set supportsEagerToolInputStreaming", () => {
    for (const spec of NON_CLAUDE_SAMPLE.filter((s) => s.id.startsWith("openai-"))) {
      const def = fakeBuildOpenAIModelDef(spec);
      const compat = def.compat as Record<string, unknown>;
      // The flag must be absent (undefined / not a key) — pi-ai treats it as
      // optional, defaulting to true. Having it here would break non-Claude
      // routes that legitimately want eager input streaming if any exist.
      expect("supportsEagerToolInputStreaming" in compat).toBe(false);
      expect(compat.supportsEagerToolInputStreaming).toBeUndefined();
    }
  });

  test("open-source models (llama, mistral, deepseek) do NOT set supportsEagerToolInputStreaming", () => {
    for (const spec of NON_CLAUDE_SAMPLE.filter((s) => !s.id.startsWith("openai-"))) {
      const def = fakeBuildOpenSourceModelDef(spec);
      const compat = def.compat as Record<string, unknown>;
      expect("supportsEagerToolInputStreaming" in compat).toBe(false);
      expect(compat.supportsEagerToolInputStreaming).toBeUndefined();
    }
  });

  test("source sentinel — buildOpenAIModelDef does NOT mention supportsEagerToolInputStreaming", () => {
    const m = indexSrc.match(
      /function\s+buildOpenAIModelDef\s*\([^)]*\)\s*:\s*ModelDefinitionConfig\s*\{[\s\S]*?\n\}/,
    );
    expect(m).not.toBeNull();
    expect(m![0]).not.toMatch(/supportsEagerToolInputStreaming/);
  });

  test("source sentinel — buildOpenSourceModelDef does NOT mention supportsEagerToolInputStreaming", () => {
    const m = indexSrc.match(
      /function\s+buildOpenSourceModelDef\s*\([^)]*\)\s*:\s*ModelDefinitionConfig\s*\{[\s\S]*?\n\}/,
    );
    expect(m).not.toBeNull();
    expect(m![0]).not.toMatch(/supportsEagerToolInputStreaming/);
  });
});

// ---------------------------------------------------------------------------
// Suite 2 — wrapStreamFn / onPayload integration (defensive strip)
//
// Re-implementation of the strip-relevant branch of the production
// onPayload closure (index.ts ~line 833). The branch under test:
//
//   onPayload: (payload, payloadModel) => {
//     if (
//       payload &&
//       typeof payload === "object" &&
//       isClaudeModel(String((model as { id?: unknown })?.id ?? ""))
//     ) {
//       const record = payload as Record<string, unknown>;
//       // ... fixTrailingAssistant / fixEmptyTextBlocks omitted in this
//       //     test — they are covered in tests/index.test.ts
//       stripEagerInputStreaming(record);
//       // ... normalizeThinkingBudget / clampMaxTokens likewise covered
//     }
//     return payload;
//   }
//
// We expose only the gating + strip logic. fixTrailingAssistant /
// fixEmptyTextBlocks / normalizeThinkingBudget / clampMaxTokens are tested
// independently in tests/index.test.ts; including them here would duplicate
// coverage and obscure what this suite is asserting.
// ---------------------------------------------------------------------------

interface FakeModel {
  id: string;
}

function fakeOnPayload(model: FakeModel, payload: unknown): unknown {
  if (
    payload &&
    typeof payload === "object" &&
    isClaudeModel(String(model?.id ?? ""))
  ) {
    const record = payload as Record<string, unknown>;
    stripEagerInputStreaming(record);
  }
  return payload;
}

describe("Suite 2: onPayload defensive strip — Claude routes", () => {
  test("Claude model + tool payload with eager_input_streaming → field stripped", () => {
    const payload = {
      messages: [{ role: "user", content: "hi" }],
      tools: [
        {
          name: "get_weather",
          custom: {
            input_schema: { type: "object" },
            eager_input_streaming: true,
          },
        },
      ],
    };
    const result = fakeOnPayload({ id: "claude-sonnet-4-6" }, payload) as typeof payload;
    const tool = result.tools[0] as Record<string, unknown>;
    const custom = tool.custom as Record<string, unknown> | undefined;
    expect(custom).toBeDefined();
    expect("eager_input_streaming" in custom!).toBe(false);
    // Other custom fields are preserved.
    expect(custom!.input_schema).toEqual({ type: "object" });
  });

  test("Claude Haiku route also strips (covers reasoning:false branch)", () => {
    const payload = {
      tools: [
        {
          name: "search",
          custom: {
            input_schema: { type: "object" },
            eager_input_streaming: true,
          },
        },
      ],
    };
    const result = fakeOnPayload({ id: "claude-haiku-4-5" }, payload) as typeof payload;
    const tool = result.tools[0] as Record<string, unknown>;
    const custom = tool.custom as Record<string, unknown>;
    expect("eager_input_streaming" in custom).toBe(false);
  });

  test("Claude route + multiple tools strips eager_input_streaming from each", () => {
    const payload = {
      tools: [
        { name: "a", custom: { input_schema: {}, eager_input_streaming: true } },
        { name: "b", custom: { input_schema: {}, eager_input_streaming: true } },
        { name: "c", custom: { input_schema: {} } }, // already clean
      ],
    };
    const result = fakeOnPayload({ id: "claude-opus-4-7" }, payload) as typeof payload;
    for (const tool of result.tools) {
      const custom = (tool as Record<string, unknown>).custom as Record<string, unknown>;
      expect("eager_input_streaming" in custom).toBe(false);
    }
  });

  test("Claude route + payload without tools → unchanged (no error)", () => {
    const payload = {
      messages: [{ role: "user", content: "hi" }],
    };
    const result = fakeOnPayload({ id: "claude-sonnet-4-6" }, payload);
    expect(result).toBe(payload);
    // Messages array is intact.
    expect((result as typeof payload).messages).toEqual([
      { role: "user", content: "hi" },
    ]);
  });
});

describe("Suite 2: onPayload defensive strip — non-Claude routes", () => {
  test("non-Claude (openai-gpt-5) + tool payload with eager_input_streaming → unchanged", () => {
    const payload = {
      tools: [
        {
          name: "get_weather",
          custom: {
            input_schema: { type: "object" },
            eager_input_streaming: true,
          },
        },
      ],
    };
    const result = fakeOnPayload({ id: "openai-gpt-5" }, payload) as typeof payload;
    const tool = result.tools[0] as Record<string, unknown>;
    const custom = tool.custom as Record<string, unknown>;
    // Non-Claude routes should NOT strip — gating is on isClaudeModel.
    expect(custom.eager_input_streaming).toBe(true);
  });

  test("non-Claude (llama3.3-70b) + tool payload with eager_input_streaming → unchanged", () => {
    const payload = {
      tools: [
        {
          name: "search",
          custom: {
            input_schema: {},
            eager_input_streaming: true,
          },
        },
      ],
    };
    const result = fakeOnPayload({ id: "llama3.3-70b" }, payload) as typeof payload;
    const tool = result.tools[0] as Record<string, unknown>;
    expect((tool.custom as Record<string, unknown>).eager_input_streaming).toBe(true);
  });

  test("empty model id → not treated as Claude, no strip", () => {
    const payload = {
      tools: [{ name: "x", custom: { eager_input_streaming: true } }],
    };
    const result = fakeOnPayload({ id: "" }, payload) as typeof payload;
    const tool = result.tools[0] as Record<string, unknown>;
    expect((tool.custom as Record<string, unknown>).eager_input_streaming).toBe(true);
  });
});

describe("Suite 2: onPayload — payload edge cases", () => {
  test("null payload → returned unchanged (no crash)", () => {
    const result = fakeOnPayload({ id: "claude-sonnet-4-6" }, null);
    expect(result).toBeNull();
  });

  test("undefined payload → returned unchanged (no crash)", () => {
    const result = fakeOnPayload({ id: "claude-sonnet-4-6" }, undefined);
    expect(result).toBeUndefined();
  });

  test("non-object payload (string) → returned unchanged", () => {
    const result = fakeOnPayload({ id: "claude-sonnet-4-6" }, "raw-string");
    expect(result).toBe("raw-string");
  });
});

// ---------------------------------------------------------------------------
// Suite 2 — Source sentinel: confirm the production onPayload still calls
// stripEagerInputStreaming inside the isClaudeModel-gated branch. If
// someone re-orders or removes that call, this fails immediately.
// ---------------------------------------------------------------------------

describe("Suite 2: source sentinel — production onPayload calls stripEagerInputStreaming", () => {
  test("onPayload body in index.ts contains stripEagerInputStreaming call", () => {
    // Find the merged onPayload arrow function inside wrapStreamFn.
    // The signature in index.ts is unique enough to anchor on:
    //   onPayload: (payload: unknown, payloadModel: unknown) => {
    const m = indexSrc.match(
      /onPayload\s*:\s*\(payload[^)]*\)\s*=>\s*\{[\s\S]*?\n\s{14}\},/,
    );
    expect(m).not.toBeNull();
    expect(m![0]).toMatch(/isClaudeModel\(/);
    expect(m![0]).toMatch(/stripEagerInputStreaming\(/);
  });
});
