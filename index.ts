/**
 * Snowflake Cortex — OpenClaw Plugin
 *
 * Routes Claude models to the Anthropic Messages API (`/messages`) and all
 * other models to the OpenAI-compatible Chat Completions API
 * (`/chat/completions`). Both endpoints live under the same Snowflake Cortex
 * gateway and share PAT authentication.
 *
 * Base URL: https://<account>.snowflakecomputing.com
 *
 * All payload transforms (tool stripping, max_tokens rewrite, tool-use repair)
 * are handled via SDK hooks and model compat flags — no raw JSON body patching.
 */

import {
  definePluginEntry,
  type ProviderWrapStreamFnContext,
  type ProviderNormalizeToolSchemasContext,
  type ProviderReplayPolicyContext,
} from "openclaw/plugin-sdk/plugin-entry";
import { resolveClaudeThinkingProfile } from "openclaw/plugin-sdk/provider-model-shared";
import type {
  MemoryEmbeddingProviderAdapter,
  MemoryEmbeddingProviderCreateOptions,
} from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import { createProviderApiKeyAuthMethod } from "openclaw/plugin-sdk/provider-auth-api-key";
import type {
  ModelDefinitionConfig,
  ModelApi,
} from "openclaw/plugin-sdk/provider-model-types";

// ---------------------------------------------------------------------------
// Environment — lazy getters so env vars are read at call time, not import time
// ---------------------------------------------------------------------------

function getApiKey(): string {
  return process.env.SNOWFLAKE_CORTEX_API_KEY ?? process.env.SNOWFLAKE_PAT ?? "";
}
function getBaseURL(): string {
  return process.env.SNOWFLAKE_BASE_URL ?? "";
}

// ---------------------------------------------------------------------------
// Structured debug logger — writes to stderr, which OpenClaw captures into
// its plugin log. Kept dependency-free and side-effect-only.
// ---------------------------------------------------------------------------

// Debug logging is gated behind FROSTCLAW_DEBUG because every call here
// goes through synchronous console.error -> stderr -> journald, which blocks
// the event loop on hot paths (wrapStreamFn, onPayload, resolveDynamicModel
// fire many times per request). Error-path logging (log.error) stays
// unconditional so we never silently lose failure signals.
const DEBUG_ENABLED: boolean = ((): boolean => {
  const v = process.env.FROSTCLAW_DEBUG;
  if (!v) return false;
  const s = v.toLowerCase();
  return s !== "0" && s !== "false" && s !== "off" && s !== "";
})();

function log(event: string, data?: Record<string, unknown>): void {
  if (!DEBUG_ENABLED) return;
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  console.error(line);
}

function logError(event: string, data?: Record<string, unknown>): void {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  console.error(line);
}

function assertConfig(): void {
  if (!getApiKey()) {
    throw new Error(
      "[snowflake-cortex] Missing auth token. " +
        "Set SNOWFLAKE_PAT (preferred) or SNOWFLAKE_CORTEX_API_KEY.",
    );
  }
  if (!getBaseURL()) {
    throw new Error(
      "[snowflake-cortex] Missing SNOWFLAKE_BASE_URL. " +
        "Expected: https://<account>.snowflakecomputing.com",
    );
  }
}

// ---------------------------------------------------------------------------
// Anthropic beta headers — split into always-safe and conditional sets.
//
// Snowflake Cortex rejects beta flags it hasn't activated with a 400 error
// ("invalid beta flag"). We only send flags that are actually needed:
//
//   BETA_ALWAYS  — broadly needed for all Claude calls (catalog headers)
//   BETA_THINKING — only when the request uses extended thinking
//   BETA_1M_FLAG — only for 1M context model variants
//
// Note: Snowflake Cortex keeps these as active beta headers even where
// Anthropic has GA'd them. Use them as-is for Snowflake compatibility.
// ---------------------------------------------------------------------------

/** Flags safe to send on every Claude request */
const BETA_ALWAYS = [
  "output-128k-2025-02-19",
  "token-efficient-tools-2025-02-19",
];

/** Flags that should only be sent when thinking is active */
const BETA_THINKING = [
  "interleaved-thinking-2025-05-14",
  "effort-2025-11-24",
  "tool-examples-2025-10-29",
];


/**
 * Claude 4.6+ (Opus, Sonnet) dropped assistant message prefill support.
 * Any payload ending with role "assistant" returns HTTP 400:
 *   "This model does not support assistant message prefill."
 *
 * The trailing assistant message is always a complete, already-delivered
 * prior turn — typically produced by context compaction or session resume.
 * Trimming it is safe: no content the user hasn't seen is lost, earlier
 * cache breakpoints are unaffected (Anthropic cache keys are prefix-based),
 * and the model regenerates correctly from the preceding user turn.
 *
 * We trim rather than inject a synthetic user turn because injection
 * changes conversation semantics — the model responds to the injected
 * content rather than continuing naturally.
 *
 * Returns the original array reference when no trim is needed (no allocation).
 */
function fixTrailingAssistant(messages: unknown[]): unknown[] {
  const last = messages[messages.length - 1];
  if (!last || typeof last !== "object") return messages;
  if ((last as Record<string, unknown>).role !== "assistant") return messages;
  return messages.slice(0, -1);
}

/**
 * Snowflake Cortex rejects:
 *   - text content blocks with empty or whitespace-only text
 *   - messages with an empty content array (content: [])
 *
 * For empty text blocks: replace with a zero-width space to preserve structure.
 * For empty content arrays: replace with a single zero-width-space text block.
 *
 * The empty-content-array case happens when an assistant turn fails mid-stream
 * (the error placeholder turn has content: []) and gets included in the next
 * request after session repair.
 */
function fixEmptyTextBlocks(messages: unknown[]): unknown[] {
  // Fast path: scan for any message that needs fixing before allocating.
  // In well-formed sessions (the common case) this returns the original
  // array reference with zero allocations.
  let needsFix = false;
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const m = msg as Record<string, unknown>;
    if (!Array.isArray(m.content)) continue;
    if (m.content.length === 0) { needsFix = true; break; }
    for (const block of m.content) {
      if (!block || typeof block !== "object") continue;
      const b = block as Record<string, unknown>;
      if (b.type === "text" && typeof b.text === "string" && b.text.trim() === "") {
        needsFix = true;
        break;
      }
    }
    if (needsFix) break;
  }
  if (!needsFix) return messages;

  // Slow path: only reached when a bad block is found (error recovery cases).
  return messages.map((msg) => {
    if (!msg || typeof msg !== "object") return msg;
    const m = msg as Record<string, unknown>;
    if (!Array.isArray(m.content)) return msg;
    if (m.content.length === 0) {
      return { ...m, content: [{ type: "text", text: "\u200b" }] };
    }
    const fixed = m.content.map((block: unknown) => {
      if (!block || typeof block !== "object") return block;
      const b = block as Record<string, unknown>;
      if (b.type !== "text" || typeof b.text !== "string") return block;
      if (b.text.trim() === "") return { ...b, text: "\u200b" };
      return block;
    });
    return { ...m, content: fixed };
  });
}

/**
 * Map a thinking level to a budget_tokens value for non-adaptive (enabled) thinking.
 * Used only when the thinking type is "enabled" (explicit budget path).
 */
function levelBudget(thinkingLevel: string | undefined): number {
  switch (thinkingLevel) {
    case "minimal": return 1024;
    case "low":     return 4000;
    case "medium":  return 8000;
    case "high":
    default:        return 16000;
  }
}

/**
 * Map a thinking level to Snowflake Cortex's output_config.effort value.
 * Snowflake uses effort (max/high/medium/low) instead of budget_tokens for
 * adaptive thinking depth control.
 */
function levelEffort(thinkingLevel: string | undefined): string {
  switch (thinkingLevel) {
    case "minimal":
    case "low":      return "low";
    case "medium":   return "medium";
    case "high":
    case "adaptive":
    default:         return "high";
  }
}

/**
 * Normalize the `thinking` field in an Anthropic payload for Snowflake Cortex:
 *
 * - `{ type: "adaptive" }` → left as-is; set output_config.effort from thinkingLevel.
 *   Snowflake natively supports adaptive thinking with { type: "adaptive" }.
 * - `{ type: "enabled", budget_tokens: N }` → overwrite budget with levelBudget(level).
 *   Corrects OpenClaw's 1024 fallback for non-adaptive levels.
 * - `{ type: "disabled" }` → left untouched.
 */
function normalizeThinkingBudget(
  payload: Record<string, unknown>,
  thinkingLevel: string | undefined,
): void {
  const thinking = payload.thinking;
  if (!thinking || typeof thinking !== "object") return;
  const t = thinking as Record<string, unknown>;
  if (t.type === "disabled") return;

  if (t.type === "adaptive") {
    // Snowflake supports { type: "adaptive" } natively — leave it intact.
    // Map the thinking level to output_config.effort for depth control.
    const effort = levelEffort(thinkingLevel);
    const existing = payload.output_config as Record<string, unknown> | undefined;
    payload.output_config = { ...existing, effort };
    return;
  }

  if (t.type === "enabled") {
    // Explicit budget path — overwrite budget_tokens with level-appropriate value.
    payload.thinking = { type: "enabled", budget_tokens: levelBudget(thinkingLevel) };
  }
}

/**
 * Clamp `max_tokens` to a safe positive value.
 *
 * When conversation history is long, OpenClaw computes
 * `max_tokens = contextWindow - usedTokens`, which can underflow to ≤ 0 and
 * causes Snowflake Cortex to 400 the request. Anthropic also requires
 * `max_tokens > thinking.budget_tokens`, so when thinking is enabled the
 * floor must sit above the thinking budget by a comfortable margin for
 * output tokens.
 *
 * Floors:
 *   - no thinking        → 1024
 *   - thinking "enabled" → budget_tokens + 1024
 *   - thinking "adaptive"→ 4096 (Snowflake uses effort, no explicit budget)
 */
const MAX_TOKENS_FLOOR_NO_THINKING = 1024;
const MAX_TOKENS_FLOOR_ADAPTIVE = 4096;
const MAX_TOKENS_OUTPUT_HEADROOM = 1024;

function clampMaxTokens(payload: Record<string, unknown>): void {
  const current = payload.max_tokens;
  if (typeof current !== "number") return;

  const thinking = payload.thinking as Record<string, unknown> | undefined;
  const thinkingType =
    thinking && typeof thinking === "object" ? thinking.type : undefined;

  let floor = MAX_TOKENS_FLOOR_NO_THINKING;
  if (thinkingType === "enabled") {
    const budget = thinking?.budget_tokens;
    const budgetNum = typeof budget === "number" ? budget : 0;
    floor = budgetNum + MAX_TOKENS_OUTPUT_HEADROOM;
  } else if (thinkingType === "adaptive") {
    floor = MAX_TOKENS_FLOOR_ADAPTIVE;
  }

  if (current >= floor) return;

  log("clampMaxTokens", {
    received: current,
    clampedTo: floor,
    thinkingType: thinkingType ?? "none",
  });
  payload.max_tokens = floor;
}

// ---------------------------------------------------------------------------
// Model classification — pure functions
// ---------------------------------------------------------------------------

function isClaudeModel(modelId: string): boolean {
  return modelId.toLowerCase().startsWith("claude");
}

/**
 * Returns true for models that support tool calling on the OpenAI Chat
 * Completions path. Claude models are excluded here because they use the
 * Anthropic Messages API, which handles tools natively and independently.
 * Returning true for openai-* here ensures tools aren't stripped for
 * those models. Everything else (Llama, DeepSeek, Mistral, etc.) gets
 * tools stripped via normalizeToolSchemas.
 */
function modelSupportsTools(modelId: string): boolean {
  return modelId.toLowerCase().startsWith("openai-");
}

// Note: Snowflake Cortex only supports the 5-minute ephemeral TTL for prompt
// caching. Injecting ttl:"1h" is not supported and may cause unexpected
// behavior. Cache breakpoints are left as-is.

// ---------------------------------------------------------------------------
// Model catalog — all Cortex models with per-model API routing + compat
// ---------------------------------------------------------------------------

interface CortexModelSpec {
  id: string;
  name: string;
  reasoning: boolean;
  contextWindow: number;
  maxTokens: number;
  input: Array<"text" | "image">;
}

const CLAUDE_MODELS: CortexModelSpec[] = [
  // Claude 4 family
  { id: "claude-4-opus",                   name: "Claude 4 Opus",                    reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] }, // $15/$75 tier
  { id: "claude-opus-4-7",                  name: "Claude Opus 4.7",                  reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-opus-4-6",                  name: "Claude Opus 4.6",                  reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-opus-4-5",                  name: "Claude Opus 4.5",                  reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-4-sonnet",                  name: "Claude 4 Sonnet",                  reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6",                name: "Claude Sonnet 4.6",                reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5",                name: "Claude Sonnet 4.5",                reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5-long-context",   name: "Claude Sonnet 4.5 (Long Context)", reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-haiku-4-5",                 name: "Claude Haiku 4.5",                 reasoning: false, contextWindow: 200_000, maxTokens: 8_192,   input: ["text", "image"] },
  // Claude 3 family
  { id: "claude-3-7-sonnet",                name: "Claude 3.7 Sonnet",                reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
];

const OPENAI_MODELS: CortexModelSpec[] = [
  { id: "openai-gpt-5.5",              name: "GPT-5.5",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] }, // preview
  { id: "openai-gpt-5.5-long-context", name: "GPT-5.5 (Long Context)", reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] }, // preview
  { id: "openai-gpt-5.4",              name: "GPT-5.4",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5.4-long-context", name: "GPT-5.4 (Long Context)", reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5.2",              name: "GPT-5.2",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5.1",              name: "GPT-5.1",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5",               name: "GPT-5",                  reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] }, // preview
  { id: "openai-gpt-5-mini",           name: "GPT-5 Mini",             reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] }, // preview
  { id: "openai-gpt-5-nano",           name: "GPT-5 Nano",             reasoning: false, contextWindow: 128_000,   maxTokens: 16_384, input: ["text", "image"] }, // preview
  { id: "openai-gpt-4.1",              name: "GPT-4.1",                reasoning: false, contextWindow: 1_047_576, maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-o4-mini",              name: "o4-mini",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
];

// No open-source models in the current Snowflake pricing table (2026-05).
// Dynamic model resolution handles any future additions via resolveDynamicModel.
const OPEN_SOURCE_MODELS: CortexModelSpec[] = [];

// ---------------------------------------------------------------------------
// Cost per token (USD) — sourced from Snowflake Service Consumption Table
// (effective 2026-04-20), Tables 6(b) and 6(c): Cortex REST API pricing.
// All values are USD per token (divide per-1M rate by 1,000,000).
//
// Table 6(b) — REST API with Prompt Caching (Claude + OpenAI models)
// Table 6(c) — REST API without Prompt Caching (Llama, Mistral, DeepSeek)
//
// Cache read is billed at 10% of the input rate (90% discount).
// Cache write (cacheWrite) is not separately billed on Snowflake Cortex.
//
// Note: claude-opus-4-7 is preview and not yet listed in the table;
// using the same rates as claude-opus-4-6 until official pricing is published.
// ---------------------------------------------------------------------------

// Costs sourced from Snowflake Service Consumption Table (2026-05), AWS/Azure Regional pricing.
// All values are USD per token (divide per-1M rate by 1,000,000).
// Global pricing is ~10% cheaper; we track Regional as the conservative default.

// Claude models (AWS Regional)
const COST_CLAUDE_4_OPUS  = { input: 0.000015,   output: 0.000075,   cacheRead: 0.0000015,  cacheWrite: 0.00001875 }; // $15.00/$75.00 | claude-4-opus
const COST_OPUS           = { input: 0.0000055,  output: 0.0000275,  cacheRead: 0.00000055, cacheWrite: 0.000006875 }; // $5.50/$27.50  | claude-opus-4-5/4-6/4-7
const COST_SONNET         = { input: 0.0000033,  output: 0.0000165,  cacheRead: 0.00000033, cacheWrite: 0.000004125 }; // $3.30/$16.50  | claude-sonnet-4-5/4-6
const COST_SONNET_LONG    = { input: 0.0000066,  output: 0.00002475, cacheRead: 0.00000066, cacheWrite: 0.00000825  }; // $6.60/$24.75  | long-context variant
const COST_HAIKU          = { input: 0.0000011,  output: 0.0000055,  cacheRead: 0.00000011, cacheWrite: 0.000001375 }; // $1.10/$5.50   | claude-haiku-4-5
const COST_CLAUDE_37      = { input: 0.000003,   output: 0.000015,   cacheRead: 0.0000003,  cacheWrite: 0.000003750 }; // $3.00/$15.00  | claude-3-7-sonnet
const COST_CLAUDE_4S      = { input: 0.000003,   output: 0.000015,   cacheRead: 0.0000003,  cacheWrite: 0.000003750 }; // $3.00/$15.00  | claude-4-sonnet

// OpenAI models (Azure Regional)
const COST_GPT55          = { input: 0.0000055,  output: 0.000033,   cacheRead: 0.00000055, cacheWrite: 0 }; // $5.50/$33.00  | gpt-5.5
const COST_GPT55_LONG     = { input: 0.000011,   output: 0.0000495,  cacheRead: 0.0000011,  cacheWrite: 0 }; // $11.00/$49.50 | gpt-5.5-long-context
const COST_GPT54          = { input: 0.00000275, output: 0.0000165,  cacheRead: 0.00000028, cacheWrite: 0 }; // $2.75/$16.50  | gpt-5.4
const COST_GPT54_LONG     = { input: 0.0000055,  output: 0.00002475, cacheRead: 0.00000055, cacheWrite: 0 }; // $5.50/$24.75  | gpt-5.4-long-context
const COST_GPT52          = { input: 0.00000193, output: 0.0000154,  cacheRead: 0.00000019, cacheWrite: 0 }; // $1.93/$15.40  | gpt-5.2
const COST_GPT51          = { input: 0.00000138, output: 0.000011,   cacheRead: 0.00000014, cacheWrite: 0 }; // $1.38/$11.00  | gpt-5 / gpt-5.1
const COST_GPT5_MINI      = { input: 0.00000028, output: 0.0000022,  cacheRead: 0.000000028,cacheWrite: 0 }; // $0.28/$2.20   | gpt-5-mini
const COST_GPT5_NANO      = { input: 0.00000006, output: 0.00000044, cacheRead: 0.000000006,cacheWrite: 0 }; // $0.06/$0.44   | gpt-5-nano
const COST_GPT41          = { input: 0.0000022,  output: 0.0000088,  cacheRead: 0.00000055, cacheWrite: 0 }; // $2.20/$8.80   | gpt-4.1
const COST_O4_MINI        = { input: 0.0000011,  output: 0.0000044,  cacheRead: 0.00000028, cacheWrite: 0 }; // $1.10/$4.40   | o4-mini

/** Catalog-level beta headers — always-safe flags only. Thinking flags are
 *  added per-request in wrapStreamFn based on ctx.thinkingLevel. */
function anthropicBetaHeaders(): Record<string, string> {
  return { "anthropic-beta": BETA_ALWAYS.join(",") };
}

/** Map a Claude model ID to its cost tier */
function claudeCost(id: string): typeof COST_OPUS {
  if (id === "claude-4-opus")              return COST_CLAUDE_4_OPUS;
  if (id.startsWith("claude-opus"))        return COST_OPUS;
  if (id.endsWith("-long-context"))        return COST_SONNET_LONG;
  if (id === "claude-4-sonnet")            return COST_CLAUDE_4S;
  if (id === "claude-3-7-sonnet")          return COST_CLAUDE_37;
  if (id.startsWith("claude-sonnet"))      return COST_SONNET;
  if (id.startsWith("claude-haiku"))       return COST_HAIKU;
  return COST_OPUS; // fallback to most expensive
}

function buildClaudeModelDef(spec: CortexModelSpec): ModelDefinitionConfig {
  return {
    id: spec.id,
    name: spec.name,
    api: "anthropic-messages" as ModelApi,
    // baseUrl must be set per-model: pi-ai's getOpenRouterAttributionHeaders
    // (pi-coding-agent/dist/core/sdk.js:31) does `model.baseUrl.includes(...)`
    // without a null-check after the `provider !== "openrouter"` short-circuit,
    // so an undefined baseUrl crashes every stream call with
    // "Cannot read properties of undefined (reading 'includes')".
    baseUrl: `${getBaseURL()}/api/v2/cortex`,
    reasoning: spec.reasoning,
    input: spec.input,
    cost: claudeCost(spec.id),
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    headers: anthropicBetaHeaders(),
    compat: { supportsTools: true },
  };
}

/** Map an OpenAI model ID to its cost tier */
function openaiCost(id: string): typeof COST_GPT51 {
  if (id === "openai-gpt-5.5-long-context") return COST_GPT55_LONG;
  if (id === "openai-gpt-5.5")              return COST_GPT55;
  if (id === "openai-gpt-5.4-long-context") return COST_GPT54_LONG;
  if (id === "openai-gpt-5.4")              return COST_GPT54;
  if (id === "openai-gpt-5.2")              return COST_GPT52;
  if (id === "openai-gpt-5.1")              return COST_GPT51;
  if (id === "openai-gpt-5")               return COST_GPT51; // same tier as 5.1
  if (id === "openai-gpt-5-mini")           return COST_GPT5_MINI;
  if (id === "openai-gpt-5-nano")           return COST_GPT5_NANO;
  if (id.startsWith("openai-gpt-4"))        return COST_GPT41;
  if (id === "openai-o4-mini")              return COST_O4_MINI;
  return COST_GPT51; // fallback
}

function buildOpenAIModelDef(spec: CortexModelSpec): ModelDefinitionConfig {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions" as ModelApi,
    baseUrl: `${getBaseURL()}/api/v2/cortex`,
    reasoning: spec.reasoning,
    input: spec.input,
    cost: openaiCost(spec.id),
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: true,
      maxTokensField: "max_completion_tokens",
      supportsUsageInStreaming: true,
    },
  };
}

/** Map an open-source model ID to its cost tier */
function openSourceCost(_id: string): typeof COST_GPT51 {
  // No open-source models in current catalog; fallback to a mid-tier rate
  // so dynamic model resolution doesn't silently bill at zero.
  return COST_GPT51;
}

function buildOpenSourceModelDef(spec: CortexModelSpec): ModelDefinitionConfig {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions" as ModelApi,
    baseUrl: `${getBaseURL()}/api/v2/cortex`,
    reasoning: spec.reasoning,
    input: spec.input,
    cost: openSourceCost(spec.id),
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: false,
      maxTokensField: "max_completion_tokens",
      supportsUsageInStreaming: true,
    },
  };
}

// Memoize the catalog so we don't rebuild the full model list (22 specs ->
// full definitions with fresh cost/header allocations) on every request.
// The catalog depends on SNOWFLAKE_BASE_URL via buildClaudeModelDef ->
// getBaseURL(), so we key the cache on the current base URL and rebuild
// only when it changes (effectively never during a gateway run).
let CATALOG_CACHE: { baseURL: string; catalog: ModelDefinitionConfig[] } | undefined;
let CATALOG_INDEX: Map<string, ModelDefinitionConfig> | undefined;

function buildModelCatalog(): ModelDefinitionConfig[] {
  const baseURL = getBaseURL();
  if (CATALOG_CACHE && CATALOG_CACHE.baseURL === baseURL) {
    return CATALOG_CACHE.catalog;
  }
  const catalog: ModelDefinitionConfig[] = [
    ...CLAUDE_MODELS.map(buildClaudeModelDef),
    ...OPENAI_MODELS.map(buildOpenAIModelDef),
    ...OPEN_SOURCE_MODELS.map(buildOpenSourceModelDef),
  ];
  CATALOG_CACHE = { baseURL, catalog };
  CATALOG_INDEX = new Map(catalog.map((m) => [m.id, m]));
  return catalog;
}

/**
 * Locate the full catalog entry for a model id (with or without the
 * `snowflake-cortex/` provider prefix). Returns a complete ModelDefinitionConfig
 * including cost, maxTokens, contextWindow, reasoning and compat — everything
 * pi-ai's calculateCost and openclaw's max_tokens resolver need to see.
 *
 * Used by resolveDynamicModel so the dynamic-resolution path returns a
 * fully-populated model instead of a minimal stub. OpenClaw's
 * applyConfiguredProviderOverrides spreads the discoveredModel verbatim
 * (see openclaw/dist/model-iYcWZTML.js:381), so fields we omit here are
 * missing in the downstream pipeline — triggering `max_tokens: 0` payloads
 * and `calculateCost: model.cost missing` warnings.
 */
function findCatalogEntry(modelId: string): ModelDefinitionConfig | undefined {
  const bareId = modelId.replace(/^snowflake-cortex\//, "");
  // Ensure catalog (and its index) are built; index lookup is O(1) vs O(n).
  if (!CATALOG_INDEX) buildModelCatalog();
  return CATALOG_INDEX?.get(bareId);
}

// ---------------------------------------------------------------------------
// Plugin entry
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Snowflake Cortex Embedding Provider
//
// Snowflake's embed API is NOT OpenAI-compatible:
//   - Request uses `text` (array) instead of `input`
//   - Response wraps vectors in an extra array: [[...]] instead of [...]
// So we use a custom adapter rather than the OpenAI embedding provider.
// ---------------------------------------------------------------------------

// Default model — cheapest, widely available, 768 dims
const DEFAULT_SNOWFLAKE_EMBED_MODEL = "snowflake-arctic-embed-m-v1.5";

async function snowflakeEmbed(
  texts: string[],
  model: string,
): Promise<number[][]> {
  const apiKey = getApiKey();
  const baseUrl = getBaseURL();
  if (!apiKey || !baseUrl) {
    throw new Error(
      "[snowflake-cortex] Missing SNOWFLAKE_BASE_URL or SNOWFLAKE_CORTEX_API_KEY",
    );
  }

  const url = `${baseUrl}/api/v2/cortex/inference:embed`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ text: texts, model }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `[snowflake-cortex] Embed request failed (${res.status}): ${body}`,
    );
  }

  const json = await res.json() as {
    data: Array<{ embedding: number[][] | number[]; index: number }>;
  };

  // Snowflake wraps each vector in an extra array: [[vec]] — flatten if needed
  return json.data
    .sort((a, b) => a.index - b.index)
    .map(({ embedding }) =>
      Array.isArray(embedding[0]) ? (embedding as number[][])[0] : (embedding as number[]),
    );
}

const snowflakeCortexEmbeddingAdapter: MemoryEmbeddingProviderAdapter = {
  id: "snowflake-cortex",
  defaultModel: DEFAULT_SNOWFLAKE_EMBED_MODEL,
  transport: "remote",
  // Low priority — only selected when explicitly configured
  autoSelectPriority: -1,

  async create(options: MemoryEmbeddingProviderCreateOptions) {
    const model = options.model || DEFAULT_SNOWFLAKE_EMBED_MODEL;
    const hasKey = !!getApiKey();
    const hasBaseUrl = !!getBaseURL();
    log("embedding.create", { model, hasKey, hasBaseUrl });

    if (!hasKey || !hasBaseUrl) {
      log("embedding.create returning null — missing config");
      return { provider: null };
    }

    return {
      provider: {
        id: "snowflake-cortex",
        model,
        maxInputTokens: 4096,
        embedQuery: (text: string) => snowflakeEmbed([text], model).then((v) => v[0]),
        embedBatch: (texts: string[]) => snowflakeEmbed(texts, model),
      },
    };
  },
};

export default definePluginEntry({
  id: "snowflake-cortex",
  name: "Snowflake Cortex",
  description:
    "Snowflake Cortex AI — routes Claude models to Anthropic Messages API " +
    "and all other models to OpenAI-compatible Chat Completions, both " +
    "behind PAT authentication.",

  register(api) {
    try {
      // Silent on success to avoid per-load log spam (openclaw re-imports
      // this plugin frequently; debug output is gated to FROSTCLAW_DEBUG).
      log("plugin registered");
      api.registerMemoryEmbeddingProvider(snowflakeCortexEmbeddingAdapter);
      api.registerProvider({
      id: "snowflake-cortex",
      label: "Snowflake Cortex",

      auth: [
        createProviderApiKeyAuthMethod({
          providerId: "snowflake-cortex",
          methodId: "snowflake-pat",
          label: "Snowflake PAT",
          hint: "Programmatic Access Token for Snowflake Cortex",
          optionKey: "snowflakePat",
          flagName: "--snowflake-pat",
          envVar: "SNOWFLAKE_CORTEX_API_KEY",
          promptMessage:
            "Enter your Snowflake Programmatic Access Token (PAT):",
        }),
      ],

      catalog: {
        run: async (ctx) => {
          try {
            const resolved = ctx.resolveProviderApiKey("snowflake-cortex");
            const resolvedKey = resolved.apiKey ?? getApiKey();
            const envKey = getApiKey();
            const baseURL = getBaseURL();

            log("catalog.run", {
              resolvedKeyPresent: !!resolved.apiKey,
              resolvedKeyLength: resolved.apiKey?.length ?? 0,
              envKeyPresent: !!envKey,
              envKeyLength: envKey.length,
              baseURL: baseURL || "(not set)",
            });

            if (!resolvedKey || !baseURL) {
              log("catalog.run returning null — missing config", {
                resolvedKey: !!resolvedKey,
                baseURL: !!baseURL,
              });
              return null;
            }

            const models = buildModelCatalog();
            log("catalog.run returning catalog", { modelCount: models.length });
            return {
              provider: {
                baseUrl: `${baseURL}/api/v2/cortex`,
                apiKey: resolvedKey,
                api: "openai-completions" as ModelApi,
                authHeader: true,
                models,
              },
            };
          } catch (err) {
            logError("catalog.run ERROR", {
              error: String(err),
              stack: err instanceof Error ? err.stack : undefined,
            });
            throw err;
          }
        },
      },

      // -----------------------------------------------------------------------
      // Hook: Resolve model API family when catalog hasn't loaded yet.
      //
      // Returns the full catalog ModelDefinitionConfig when we recognize the
      // model id. This is important: openclaw's applyConfiguredProviderOverrides
      // (model-iYcWZTML.js:380-394) spreads the discoveredModel and pulls
      // `cost`, `maxTokens`, `contextWindow`, `reasoning`, `compat` directly
      // from it — fields we omit here simply never reach the downstream
      // pipeline, causing pi-ai's calculateCost to warn `model.cost missing`
      // and openclaw's resolveAnthropicMessagesMaxTokens to emit `max_tokens: 0`.
      //
      // For unknown ids we fall back to a minimal stub so the request doesn't
      // crash: openclaw's provider-stream reads `model.input.includes(...)`
      // and pi-ai's sdk reads `model.baseUrl.includes(...)` unconditionally.
      // Claude models support text + image; other unknown ids default to
      // text-only since dropping images is safer than crashing.
      // -----------------------------------------------------------------------
      resolveDynamicModel(ctx) {
        const modelId = ctx.modelId;
        if (!modelId) {
          log("resolveDynamicModel: no modelId");
          return null;
        }

        const catalogEntry = findCatalogEntry(modelId);
        if (catalogEntry) {
          log("resolveDynamicModel (catalog hit)", {
            modelId,
            api: catalogEntry.api,
            contextWindow: catalogEntry.contextWindow,
            maxTokens: catalogEntry.maxTokens,
            hasCost: !!catalogEntry.cost,
          });
          return catalogEntry;
        }

        const claude = isClaudeModel(modelId);
        const api: ModelApi = claude ? "anthropic-messages" : "openai-completions";
        const input: Array<"text" | "image"> = claude ? ["text", "image"] : ["text"];
        const baseUrl = `${getBaseURL()}/api/v2/cortex`;
        log("resolveDynamicModel (unknown id, minimal stub)", { modelId, api, input, baseUrl });
        return { id: modelId, name: modelId, api, input, baseUrl };
      },

      // -----------------------------------------------------------------------
      // Hook: Strip tools for models that don't support them
      // -----------------------------------------------------------------------
      normalizeToolSchemas(ctx: ProviderNormalizeToolSchemasContext) {
        if (!ctx.modelId) return ctx.tools;
        if (isClaudeModel(ctx.modelId)) return ctx.tools;  // handled by Anthropic API
        if (!modelSupportsTools(ctx.modelId)) return [];
        return ctx.tools;
      },

      // -----------------------------------------------------------------------
      // Hook: Inject Snowflake PAT header type and conditionally add thinking
      // beta flags based on ctx.thinkingLevel for Claude models.
      //
      // Beta flags are split: always-safe flags come from the catalog headers,
      // while thinking-specific flags (interleaved-thinking, effort,
      // tool-examples) are only added when the request uses thinking. This
      // prevents Snowflake Cortex from rejecting unknown beta flags with 400.
      // -----------------------------------------------------------------------
      wrapStreamFn(ctx: ProviderWrapStreamFnContext) {
        log("wrapStreamFn", {
          modelId: (ctx as unknown as Record<string, unknown>).modelId as string | undefined,
          thinkingLevel: ctx.thinkingLevel,
          thinkingActive: ctx.thinkingLevel !== undefined && ctx.thinkingLevel !== "off",
          hasStreamFn: !!ctx.streamFn,
        });
        if (!ctx.streamFn) return undefined;

        const inner = ctx.streamFn;

        // OpenClaw re-invokes wrapStreamFn(ctx) fresh per request, so
        // ctx.thinkingLevel is current here — not stale from registration time.
        // thinkingLevel must be read from the outer ctx; the inner StreamFn's
        // context param (messages, tools, systemPrompt) does not carry it.
        const thinkingActive =
          ctx.thinkingLevel !== undefined && ctx.thinkingLevel !== "off";
        const thinkingLevel = ctx.thinkingLevel;

        return (model, context, options) => {
          try {
            // Guard: two known openclaw/pi-ai sites dereference model fields
            // unconditionally and crash on undefined:
            //   1. provider-stream-CQzDRxyR.js:154 reads model.input.includes(...)
            //   2. pi-coding-agent/dist/core/sdk.js:31 reads model.baseUrl.includes(...)
            //      after `provider !== "openrouter"` (no short-circuit on baseUrl).
            // Patch both if missing so the downstream pipeline never sees an
            // illegal state.
            const modelObj = model as {
              id?: unknown;
              api?: unknown;
              input?: unknown;
              baseUrl?: unknown;
            } | undefined;
            if (modelObj && !Array.isArray(modelObj.input)) {
              const id = String(modelObj.id ?? "");
              const inferred: Array<"text" | "image"> = isClaudeModel(id)
                ? ["text", "image"]
                : ["text"];
              log("wrapStreamFn.inner: patching missing input", {
                modelId: id,
                inferred,
                priorInput: modelObj.input,
                keys: Object.keys(modelObj as Record<string, unknown>),
              });
              (modelObj as Record<string, unknown>).input = inferred;
            }
            if (modelObj && typeof modelObj.baseUrl !== "string") {
              const fallbackBaseUrl = `${getBaseURL()}/api/v2/cortex`;
              log("wrapStreamFn.inner: patching missing baseUrl", {
                modelId: String(modelObj.id ?? ""),
                fallbackBaseUrl,
                priorBaseUrl: modelObj.baseUrl,
              });
              (modelObj as Record<string, unknown>).baseUrl = fallbackBaseUrl;
            }
            log("wrapStreamFn.inner", {
              modelId: modelObj?.id,
              modelApi: modelObj?.api,
              modelInput: modelObj?.input,
              modelBaseUrl: modelObj?.baseUrl,
              modelKeys: modelObj
                ? Object.keys(modelObj as Record<string, unknown>)
                : undefined,
              hasContext: !!context,
              hasOptions: !!options,
              messageCount: Array.isArray((context as Record<string, unknown> | undefined)?.messages)
                ? ((context as Record<string, unknown>).messages as unknown[]).length
                : undefined,
            });
            const originalOnPayload = options?.onPayload;

            // Build per-request anthropic-beta header: start with any
            // catalog-level flags from the model definition, then append
            // thinking flags when the request actually uses thinking.
            const catalogBeta =
              (model as { headers?: Record<string, string> })?.headers?.[
                "anthropic-beta"
              ] ?? "";
            const betaFlags = catalogBeta ? [catalogBeta] : [];
            // Only send thinking beta flags for models that support extended
            // thinking (reasoning: true). Non-reasoning models like Haiku accept
            // the BETA_ALWAYS headers fine but reject BETA_THINKING flags with a
            // 400 even when the payload itself has no thinking field — OpenClaw
            // core already skips the thinking field for reasoning:false models,
            // but the headers are added here independently of the payload.
            const modelSupportsReasoning =
              (modelObj as { reasoning?: boolean } | undefined)?.reasoning === true;
            if (thinkingActive && modelSupportsReasoning) {
              betaFlags.push(BETA_THINKING.join(","));
            }

            // Snowflake Cortex requires `Authorization: Bearer <PAT>` for
            // every request, including its Anthropic-Messages endpoint. The
            // SDK's Anthropic transport (provider-stream:335) instead sends
            // `x-api-key: <key>`, which Snowflake rejects with
            //   400 Bearer token is missing
            // So for Claude models we must attach the Bearer header ourselves.
            // Non-Claude (openai-completions) routes already get Bearer from
            // the SDK's default flow and need no patching here.
            const modelId = String((model as { id?: unknown })?.id ?? "");
            const isClaudeRoute = isClaudeModel(modelId);
            const optionsApiKey =
              (options as { apiKey?: unknown } | undefined)?.apiKey;
            const bearerKey =
              typeof optionsApiKey === "string" && optionsApiKey.length > 0
                ? optionsApiKey
                : getApiKey();
            const authHeader =
              isClaudeRoute && bearerKey
                ? { Authorization: `Bearer ${bearerKey}` }
                : {};

            const merged = {
              ...options,
              headers: {
                ...options?.headers,
                "X-Snowflake-Authorization-Token-Type":
                  "PROGRAMMATIC_ACCESS_TOKEN",
                ...authHeader,
                ...(betaFlags.length > 0
                  ? { "anthropic-beta": betaFlags.join(",") }
                  : {}),
              },
              onPayload: (payload: unknown, payloadModel: unknown) => {
                const payloadModelObj = payloadModel as { id?: unknown } | undefined;
                log("onPayload", {
                  payloadType: typeof payload,
                  isObject: payload !== null && typeof payload === "object",
                  payloadModelId: payloadModelObj?.id,
                  isClaudeModelResult: payload && typeof payload === "object"
                    ? isClaudeModel(String((model as { id?: unknown })?.id ?? ""))
                    : false,
                });
                if (
                  payload &&
                  typeof payload === "object" &&
                  isClaudeModel(String((model as { id?: unknown })?.id ?? ""))
                ) {
                  const record = payload as Record<string, unknown>;
                  if (Array.isArray(record.messages)) {
                    record.messages = fixTrailingAssistant(record.messages);
                    record.messages = fixEmptyTextBlocks(record.messages);
                  }
                  normalizeThinkingBudget(record, thinkingLevel);
                  clampMaxTokens(record);
                }
                return (originalOnPayload as
                  | ((p: unknown, m: unknown) => unknown)
                  | undefined)?.(payload, payloadModel);
              },
            };
            const streamResult = inner(model, context, merged as typeof options);
            // The openclaw runtime may invoke streamFn as streamSimple, which
            // returns a Promise. Wrap to observe rejection so we can capture
            // the stack trace for the "Cannot read properties of undefined"
            // crash that otherwise surfaces only as opaque text.
            if (
              streamResult &&
              typeof (streamResult as { then?: unknown }).then === "function"
            ) {
              return (streamResult as Promise<unknown>).then(
                (value) => {
                  const valueObj = value as
                    | { errorMessage?: unknown; stopReason?: unknown; content?: unknown }
                    | undefined;
                  if (
                    valueObj &&
                    typeof valueObj === "object" &&
                    (valueObj.errorMessage || valueObj.stopReason === "error")
                  ) {
                    log("wrapStreamFn.promise resolved with error", {
                      errorMessage: valueObj.errorMessage,
                      stopReason: valueObj.stopReason,
                    });
                  }
                  return value;
                },
                (err) => {
                  logError("wrapStreamFn.promise REJECTED", {
                    error: String(err),
                    stack: err instanceof Error ? err.stack : undefined,
                  });
                  throw err;
                },
              ) as typeof streamResult;
            }
            return streamResult;
          } catch (err) {
            logError("wrapStreamFn.inner ERROR", {
              error: String(err),
              stack: err instanceof Error ? err.stack : undefined,
            });
            throw err;
          }
        };
      },

      // -----------------------------------------------------------------------
      // Thinking profile: expose adaptive and all supported levels for Claude
      // models so the Control UI thinking dropdown shows them correctly.
      // -----------------------------------------------------------------------
      resolveThinkingProfile(ctx) {
        if (!ctx.modelId) return null;
        if (!isClaudeModel(ctx.modelId)) return null;
        // Strip the "snowflake-cortex/" prefix before calling the SDK function
        const bareId = ctx.modelId.replace(/^snowflake-cortex\//, "");
        return resolveClaudeThinkingProfile(bareId) ?? null;
      },

      // -----------------------------------------------------------------------
      // Replay policy: repair orphaned tool_use/result pairs for Claude,
      // validate Anthropic turn structure
      // -----------------------------------------------------------------------
      buildReplayPolicy(ctx: ProviderReplayPolicyContext) {
        if (!ctx.modelId) return null;

        if (isClaudeModel(ctx.modelId)) {
          return {
            repairToolUseResultPairing: true,
            allowSyntheticToolResults: true,
            validateAnthropicTurns: true,
          };
        }

        // Chat completions models: default policy is fine
        return null;
      },
    });
    } catch (err) {
      logError("register ERROR", {
        error: String(err),
        stack: err instanceof Error ? err.stack : undefined,
      });
      throw err;
    }
  },
});
