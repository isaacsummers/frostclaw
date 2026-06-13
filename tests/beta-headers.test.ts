/**
 * Beta header assembly tests + Claude model catalog regression tests.
 *
 * The frostclaw plugin assembles the `anthropic-beta` request header inside
 * `wrapStreamFn` (index.ts). The exact source-of-truth code cannot be
 * imported directly here because index.ts pulls in `openclaw/*` modules that
 * are external at runtime.
 *
 * Instead, this file mirrors the *exact* assembly logic byte-for-byte against
 * the corresponding constants imported from index.ts where possible
 * (BETA_ALWAYS / BETA_THINKING are not exported, so they are pasted verbatim
 * with a sentinel test that re-reads the source to detect drift).
 *
 * Current state of the buckets (after Snowflake/Bedrock matrix audit):
 *
 *   BETA_ALWAYS    \u2014 empty. Every legacy "always-safe" flag turned out to be
 *                    either Sonnet-3.7-only (output-128k-2025-02-19,
 *                    token-efficient-tools-2025-02-19), Opus-4.5-only
 *                    (effort-2025-11-24, tool-examples-2025-10-29), or
 *                    rendered redundant on Claude 4+ where 128K output is
 *                    native and driven solely by max_tokens.
 *
 *   BETA_THINKING  \u2014 interleaved-thinking-2025-05-14. Sent only when
 *                    extended thinking is active AND the model is
 *                    reasoning-capable. Bedrock-supported on Sonnet 4.5 /
 *                    Haiku 4.5 / Opus 4 family; GA (header no-op) on 4.6+.
 *
 * Assertions under test:
 *   1. Catalog headers (`anthropicBetaHeaders()`) are empty when BETA_ALWAYS
 *      is empty (no blank `anthropic-beta` header attached).
 *   2. `BETA_THINKING` is appended only when both thinking is active AND the
 *      model supports reasoning. Haiku (reasoning:false) never receives it.
 *   3. Haiku 4.5 catalog entry pins maxTokens to 64_000.
 *   4. The legacy flags removed from this build never reappear in
 *      production source (regression sentinels).
 */

import { describe, test, expect } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// Verbatim copies of the index.ts constants. The drift sentinel below
// re-reads index.ts and asserts these match \u2014 if upstream changes, the
// test file will fail loudly rather than silently testing the wrong thing.
// ---------------------------------------------------------------------------

const BETA_ALWAYS: string[] = [];

const BETA_THINKING = [
  "interleaved-thinking-2025-05-14",
];

// Flags that previously lived in BETA_ALWAYS / BETA_REASONING_ONLY /
// BETA_THINKING but were dropped after the Bedrock-matrix audit. Listed here
// purely so the regression sentinels can keep them out of production source.
const REMOVED_FLAGS = [
  "output-128k-2025-02-19",
  "token-efficient-tools-2025-02-19",
  "effort-2025-11-24",
  "tool-examples-2025-10-29",
];

// ---------------------------------------------------------------------------
// Drift sentinel \u2014 keep the test's copy of the BETA_* arrays in lockstep
// with the production source.
// ---------------------------------------------------------------------------

describe("BETA constants drift sentinel", () => {
  // BETA_ALWAYS and catalog functions moved to src/catalog.ts; BETA_THINKING stays in index.ts
  const indexSrc = readFileSync(
    join(import.meta.dir, "..", "index.ts"),
    "utf8",
  );
  const catalogSrc = readFileSync(
    join(import.meta.dir, "..", "src", "catalog.ts"),
    "utf8",
  );

  test("BETA_ALWAYS matches catalog.ts source (empty array)", () => {
    // BETA_ALWAYS moved to src/catalog.ts; scan that file now.
    const m =
      catalogSrc.match(/const BETA_ALWAYS\s*:\s*string\[\]\s*=\s*\[([\s\S]*?)\];/) ??
      catalogSrc.match(/const BETA_ALWAYS\s*=\s*\[([\s\S]*?)\];/);
    expect(m).not.toBeNull();
    const literals = Array.from(m![1].matchAll(/"([^"]+)"/g)).map((x) => x[1]);
    expect(literals).toEqual(BETA_ALWAYS);
  });

  test("BETA_THINKING matches index.ts source", () => {
    const m = indexSrc.match(/const BETA_THINKING = \[([\s\S]*?)\];/);
    expect(m).not.toBeNull();
    const literals = Array.from(m![1].matchAll(/"([^"]+)"/g)).map((x) => x[1]);
    expect(literals).toEqual(BETA_THINKING);
  });

  test("BETA_REASONING_ONLY no longer exists in production source", () => {
    // Earlier build had a reasoning-only bucket carrying token-efficient-tools.
    // After the audit that flag is dropped entirely; the bucket goes with it.
    expect(indexSrc).not.toMatch(/const\s+BETA_REASONING_ONLY/);
  });

  test("removed flags are not present in non-comment production source", () => {
    // The migration block at the top of index.ts intentionally names every
    // dropped flag so future readers understand the audit. This test scans
    // only non-comment lines to make sure no live code re-introduces them.
    const stripped = indexSrc
      // strip /* ... */ blocks (including JSDoc / banner blocks)
      .replace(/\/\*[\s\S]*?\*\//g, "")
      // strip // line comments (whole-line and trailing)
      .replace(/^\s*\/\/.*$/gm, "")
      .replace(/\s\/\/.*$/gm, "");
    for (const flag of REMOVED_FLAGS) {
      expect(stripped).not.toContain(flag);
    }
  });
});

// ---------------------------------------------------------------------------
// Re-implementation of the assembly logic from index.ts wrapStreamFn.
//
// Reproduced verbatim:
//
//   const catalogBeta =
//     (model as { headers?: Record<string,string> })?.headers?.["anthropic-beta"] ?? "";
//   const betaFlags = catalogBeta ? [catalogBeta] : [];
//   const modelSupportsReasoning =
//     (modelObj as { reasoning?: boolean } | undefined)?.reasoning === true;
//   if (thinkingActive && modelSupportsReasoning) {
//     betaFlags.push(BETA_THINKING.join(","));
//   }
//   ...
//   "anthropic-beta": betaFlags.join(",")
//
// Catalog headers come from `anthropicBetaHeaders()`:
//   BETA_ALWAYS.length > 0
//     ? { "anthropic-beta": BETA_ALWAYS.join(",") }
//     : {}
// They are attached to every Claude model via buildClaudeModelDef.
// ---------------------------------------------------------------------------

interface FakeModel {
  id: string;
  reasoning: boolean;
  headers?: Record<string, string>;
}

function assembleAnthropicBetaHeader(
  model: FakeModel,
  thinkingLevel: string | undefined,
): string {
  const thinkingActive =
    thinkingLevel !== undefined && thinkingLevel !== "off";
  const catalogBeta = model.headers?.["anthropic-beta"] ?? "";
  const betaFlags = catalogBeta ? [catalogBeta] : [];
  const modelSupportsReasoning = model.reasoning === true;
  if (thinkingActive && modelSupportsReasoning) {
    betaFlags.push(BETA_THINKING.join(","));
  }
  return betaFlags.join(",");
}

// Standard catalog headers as produced by anthropicBetaHeaders(). With
// BETA_ALWAYS empty the function returns an empty object \u2014 no
// `anthropic-beta` key is present, so the FakeModel below has no `headers`.
const CLAUDE_CATALOG_HEADERS: Record<string, string> | undefined =
  BETA_ALWAYS.length > 0
    ? { "anthropic-beta": BETA_ALWAYS.join(",") }
    : undefined;

// ---------------------------------------------------------------------------
// Sonnet (reasoning: true) \u2014 receives BETA_THINKING when thinking is active.
// ---------------------------------------------------------------------------

describe("anthropic-beta header \u2014 claude-sonnet-4-6 (reasoning: true)", () => {
  const sonnet: FakeModel = {
    id: "claude-sonnet-4-6",
    reasoning: true,
    headers: CLAUDE_CATALOG_HEADERS,
  };

  test("thinking active (high): includes BETA_THINKING (interleaved-thinking)", () => {
    const header = assembleAnthropicBetaHeader(sonnet, "high");
    const flags = header.split(",").filter(Boolean);

    expect(flags).toContain("interleaved-thinking-2025-05-14");
    // None of the removed legacy flags should ever appear here.
    for (const flag of REMOVED_FLAGS) {
      expect(flags).not.toContain(flag);
    }
  });

  test("thinking off: header is empty (no flags to send)", () => {
    const header = assembleAnthropicBetaHeader(sonnet, "off");
    expect(header).toBe("");
  });

  test("thinking undefined: header is empty", () => {
    const header = assembleAnthropicBetaHeader(sonnet, undefined);
    expect(header).toBe("");
  });
});

// ---------------------------------------------------------------------------
// Opus (reasoning: true) \u2014 same shape as Sonnet.
// ---------------------------------------------------------------------------

describe("anthropic-beta header \u2014 claude-opus-4-7 (reasoning: true)", () => {
  const opus: FakeModel = {
    id: "claude-opus-4-7",
    reasoning: true,
    headers: CLAUDE_CATALOG_HEADERS,
  };

  test("thinking off: header is empty (no always-on flags, thinking gated off)", () => {
    expect(assembleAnthropicBetaHeader(opus, "off")).toBe("");
  });

  test("thinking high: header carries only interleaved-thinking", () => {
    const header = assembleAnthropicBetaHeader(opus, "high");
    expect(header.split(",").filter(Boolean)).toEqual([
      "interleaved-thinking-2025-05-14",
    ]);
  });
});

// ---------------------------------------------------------------------------
// Haiku (reasoning: false) \u2014 must receive an empty header regardless of
// thinking level, because BETA_THINKING is gated on reasoning:true and
// BETA_ALWAYS is empty.
// ---------------------------------------------------------------------------

describe("anthropic-beta header \u2014 claude-haiku-4-5 (reasoning: false)", () => {
  const haiku: FakeModel = {
    id: "claude-haiku-4-5",
    reasoning: false,
    headers: CLAUDE_CATALOG_HEADERS,
  };

  test("thinking off: BETA_THINKING flags excluded", () => {
    const header = assembleAnthropicBetaHeader(haiku, "off");
    expect(header.split(",").filter(Boolean)).not.toContain(
      "interleaved-thinking-2025-05-14",
    );
  });

  test("thinking active (high) on non-reasoning model: BETA_THINKING still excluded (gated by model.reasoning)", () => {
    const header = assembleAnthropicBetaHeader(haiku, "high");
    expect(header.split(",").filter(Boolean)).not.toContain(
      "interleaved-thinking-2025-05-14",
    );
  });

  test("FIX: Haiku does NOT receive token-efficient-tools-2025-02-19 (legacy flag fully removed)", () => {
    // Bedrock matrix lists token-efficient-tools-2025-02-19 as Sonnet-3.7-only.
    // Snowflake's strict validator rejects it for Haiku 4.5. After the audit
    // it is removed everywhere; this test guards against any future
    // re-introduction.
    const headerOff = assembleAnthropicBetaHeader(haiku, "off");
    const headerHigh = assembleAnthropicBetaHeader(haiku, "high");
    expect(headerOff).not.toContain("token-efficient-tools-2025-02-19");
    expect(headerHigh).not.toContain("token-efficient-tools-2025-02-19");
  });

  test("Haiku does NOT receive output-128k-2025-02-19 (legacy 3.7-era flag, dropped)", () => {
    // output-128k-2025-02-19 is Bedrock-listed for Sonnet 3.7 only and
    // meaningless on Haiku (16K output cap); never send it.
    const headerOff = assembleAnthropicBetaHeader(haiku, "off");
    const headerHigh = assembleAnthropicBetaHeader(haiku, "high");
    expect(headerOff).not.toContain("output-128k-2025-02-19");
    expect(headerHigh).not.toContain("output-128k-2025-02-19");
  });

  test("Haiku final header is empty regardless of thinking level", () => {
    expect(assembleAnthropicBetaHeader(haiku, "off")).toBe("");
    expect(assembleAnthropicBetaHeader(haiku, "high")).toBe("");
    expect(assembleAnthropicBetaHeader(haiku, undefined)).toBe("");
  });
});

// ---------------------------------------------------------------------------
// Verify the catalog header attachment: when BETA_ALWAYS is empty,
// anthropicBetaHeaders() must return an empty object (no key) so we don't
// emit a blank `anthropic-beta` header.
// ---------------------------------------------------------------------------

describe("catalog headers (anthropicBetaHeaders)", () => {
  test("returns an empty object when BETA_ALWAYS is empty", () => {
    // Reproduce anthropicBetaHeaders():
    //   BETA_ALWAYS.length > 0
    //     ? { "anthropic-beta": BETA_ALWAYS.join(",") }
    //     : {};
    const headers =
      BETA_ALWAYS.length > 0
        ? { "anthropic-beta": BETA_ALWAYS.join(",") }
        : {};

    expect(headers).toEqual({});
    expect((headers as Record<string, string>)["anthropic-beta"]).toBeUndefined();
  });

  test("buildClaudeModelDef applies the catalog headers function uniformly", () => {
    const src = readFileSync(
      join(import.meta.dir, "..", "src", "catalog.ts"),
      "utf8",
    );

    const buildClaude = src.match(
      /function buildClaudeModelDef\(spec: CortexModelSpec[^)]*\)[^{]*\{([\s\S]*?)^\}/m,
    );
    expect(buildClaude).not.toBeNull();
    const body = buildClaude![1];

    expect(body).toContain("headers: anthropicBetaHeaders()");
  });

  test("anthropicBetaHeaders() implementation guards against empty header attachment", () => {
    // buildClaudeModelDef and anthropicBetaHeaders moved to src/catalog.ts
    const src = readFileSync(
      join(import.meta.dir, "..", "src", "catalog.ts"),
      "utf8",
    );

    expect(src).toMatch(
      /BETA_ALWAYS\.length\s*>\s*0\s*\?\s*\{\s*"anthropic-beta":\s*BETA_ALWAYS\.join\(","\)\s*\}\s*:\s*\{\s*\}/,
    );
  });

  test("wrapStreamFn appends BETA_THINKING when thinkingActive && modelSupportsReasoning", () => {
    const indexSrc = readFileSync(
      join(import.meta.dir, "..", "index.ts"),
      "utf8",
    );
    expect(indexSrc).toMatch(
      /if\s*\(\s*thinkingActive\s*&&\s*modelSupportsReasoning\s*\)\s*\{\s*betaFlags\.push\(BETA_THINKING\.join\(","\)\)/,
    );
  });
});

// ---------------------------------------------------------------------------
// normalizeToolSchemas behavior \u2014 unchanged from before. Claude (incl. Haiku)
// keeps tools, non-Claude / non-OpenAI strip tools, openai-* keep tools.
// ---------------------------------------------------------------------------

import { isClaudeModel } from "../src/transforms.js";

function modelSupportsTools(modelId: string): boolean {
  return modelId.toLowerCase().startsWith("openai-");
}

function normalizeToolSchemasReplica(
  modelId: string | undefined,
  tools: unknown[],
): unknown[] {
  if (!modelId) return tools;
  if (isClaudeModel(modelId)) return tools;
  if (!modelSupportsTools(modelId)) return [];
  return tools;
}

describe("normalizeToolSchemas", () => {
  const fakeTools = [
    { name: "search", description: "search the web" },
    { name: "calc", description: "do math" },
  ];

  test("Claude Sonnet keeps tools (handled by Anthropic Messages API)", () => {
    expect(normalizeToolSchemasReplica("claude-sonnet-4-6", fakeTools)).toEqual(
      fakeTools,
    );
  });

  test("Claude Haiku keeps tools (Claude path, NOT stripped)", () => {
    expect(normalizeToolSchemasReplica("claude-haiku-4-5", fakeTools)).toEqual(
      fakeTools,
    );
  });

  test("Claude Opus keeps tools", () => {
    expect(normalizeToolSchemasReplica("claude-opus-4-7", fakeTools)).toEqual(
      fakeTools,
    );
  });

  test("OpenAI GPT-5 keeps tools (modelSupportsTools = true)", () => {
    expect(normalizeToolSchemasReplica("openai-gpt-5", fakeTools)).toEqual(
      fakeTools,
    );
  });

  test("Llama 3.3-70b: tools stripped (not Claude, not openai-)", () => {
    expect(normalizeToolSchemasReplica("llama3.3-70b", fakeTools)).toEqual([]);
  });

  test("DeepSeek R1: tools stripped", () => {
    expect(normalizeToolSchemasReplica("deepseek-r1", fakeTools)).toEqual([]);
  });

  test("Mistral Large 2: tools stripped", () => {
    expect(normalizeToolSchemasReplica("mistral-large2", fakeTools)).toEqual(
      [],
    );
  });

  test("undefined modelId: passes tools through", () => {
    expect(normalizeToolSchemasReplica(undefined, fakeTools)).toEqual(
      fakeTools,
    );
  });
});

// ---------------------------------------------------------------------------
// Direct check on the model catalog: confirm claude-haiku-4-5 has
// reasoning: false, maxTokens: 64_000 (Anthropic's documented Haiku 4.5
// output cap), and is in the CLAUDE_MODELS list.
// ---------------------------------------------------------------------------

describe("model catalog \u2014 claude-haiku-4-5 routing", () => {
  // CLAUDE_MODELS and buildClaudeModelDef moved to src/catalog.ts
  const indexSrc = readFileSync(
    join(import.meta.dir, "..", "src", "catalog.ts"),
    "utf8",
  );

  test("claude-haiku-4-5 is registered in CLAUDE_MODELS with reasoning:false", () => {
    const m = indexSrc.match(
      /id:\s*"claude-haiku-4-5"[^}]*reasoning:\s*(true|false)/,
    );
    expect(m).not.toBeNull();
    expect(m![1]).toBe("false");
  });

  test("claude-haiku-4-5 maxTokens is 64_000 (Anthropic's documented Haiku 4.5 output cap)", () => {
    // Tolerate the underscore-separated literal (64_000) or the bare number
    // (64000). Updated from legacy 16_384 to match Anthropic pricing page (2026-06-13).
    const m = indexSrc.match(
      /id:\s*"claude-haiku-4-5"[^}]*maxTokens:\s*([\d_]+)/,
    );
    expect(m).not.toBeNull();
    const literal = m![1].replace(/_/g, "");
    expect(literal).toBe("64000");
  });

  test("buildClaudeModelDef sets api: 'anthropic-messages' for all Claude models", () => {
    const m = indexSrc.match(
      /function buildClaudeModelDef[\s\S]*?api:\s*"([^"]+)"/,
    );
    expect(m).not.toBeNull();
    expect(m![1]).toBe("anthropic-messages");
  });
});
