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
// Anthropic beta headers (anthropic-beta) — Snowflake Cortex constraints.
//
// Snowflake Cortex docs: "Only Bedrock-compatible anthropic-beta header
// values are supported." Cortex rejects unrecognized flags with a 400
// ("invalid beta flag" / "provider rejected the request schema"), so the
// matrix below tracks AWS Bedrock's published Anthropic beta flag list,
// not the broader anthropic.com flag set.
//
// Per Anthropic's Claude 4+ migration guide, several flags we used to send
// are now legacy or model-specific:
//
//   output-128k-2025-02-19          DROPPED.
//                                   Bedrock-listed for Claude 3.7 Sonnet only,
//                                   where it gated 64K → 128K output. On all
//                                   Claude 4+ models 128K output is native and
//                                   driven purely by max_tokens in the body
//                                   (catalog already sets maxTokens: 128_000).
//                                   Migration guide explicitly says:
//                                   "Remove ... output-128k-2025-02-19."
//                                   Sending it to Haiku 4.5 (16K output cap)
//                                   is meaningless. Drop universally.
//
//   token-efficient-tools-2025-02-19  DROPPED.
//                                   Bedrock-listed for Claude 3.7 Sonnet only.
//                                   On Claude 4+ token-efficient tool use is
//                                   built into the model. Migration guide:
//                                   "Remove ... token-efficient-tools-2025-02-19.
//                                   All Claude 4+ models have built-in
//                                   token-efficient tool use." Snowflake
//                                   already rejects it for Haiku.
//
//   effort-2025-11-24                DROPPED.
//                                   Bedrock-listed for Claude Opus 4.5 only.
//                                   GA on Opus 4.6/4.7 (header now no-op there).
//                                   We can't safely send Opus-only flags to
//                                   the Sonnet/Haiku/3.7 catalog entries that
//                                   share this code path; per-model gating
//                                   would require a model-id list and the
//                                   benefit (effort param on a single model
//                                   tier) doesn't justify the surface area.
//
//   tool-examples-2025-10-29         DROPPED.
//                                   Bedrock-listed for Claude Opus 4.5 only.
//                                   Same reasoning as effort-2025-11-24:
//                                   sending an Opus-only flag to Sonnet/
//                                   Haiku/3.7 risks rejection on a strict
//                                   provider, and the feature isn't wired
//                                   into our request shape anyway.
//
//   interleaved-thinking-2025-05-14  KEPT, thinking-only.
//                                   Bedrock-supported on Claude Sonnet 4.5
//                                   and Claude Haiku 4.5 (and Opus 4 family).
//                                   GA on Claude 4.6+ (header still accepted
//                                   but no-op there). Adaptive thinking on
//                                   Opus 4.7 turns it on automatically.
//                                   Only meaningful when thinking is active,
//                                   and only on reasoning-capable models —
//                                   gated on both conditions below.
//
// Buckets:
//   BETA_ALWAYS         — sent on every Claude request. Currently empty:
//                         every legacy "always-safe" flag turned out to be
//                         either model-specific or rendered redundant by the
//                         body's max_tokens.
//   BETA_THINKING       — sent only when extended thinking is active AND
//                         model.reasoning === true. Haiku (reasoning:false)
//                         and any other non-reasoning model never receive
//                         these.
// ---------------------------------------------------------------------------

/** Flags safe to send on every Claude request. Empty by design — see comment above. */
const BETA_ALWAYS: string[] = [];

/** Flags sent only when extended thinking is active on a reasoning-capable model. */
const BETA_THINKING = [
  "interleaved-thinking-2025-05-14",
];


// Pure payload transforms — extracted to src/transforms.ts for testability.
import {
  fixTrailingAssistant,
  fixEmptyTextBlocks,
  levelBudget,
  levelEffort,
  normalizeThinkingBudget,
  clampMaxTokens,
  stripEagerInputStreaming,
  isClaudeModel,
} from "./src/transforms.js";

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
  { id: "claude-haiku-4-5",                 name: "Claude Haiku 4.5",                 reasoning: false, contextWindow: 200_000, maxTokens: 16_384,  input: ["text", "image"] },
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

const OPEN_SOURCE_MODELS: CortexModelSpec[] = [
  { id: "deepseek-r1",              name: "DeepSeek R1",                reasoning: false, contextWindow: 64_000,     maxTokens: 16_384, input: ["text"] },
  { id: "llama3.1-405b",            name: "Llama 3.1 405B",            reasoning: false, contextWindow: 128_000,    maxTokens: 32_768, input: ["text"] },
  { id: "llama3.1-70b",             name: "Llama 3.1 70B",             reasoning: false, contextWindow: 128_000,    maxTokens: 32_768, input: ["text"] },
  { id: "llama3.1-8b",              name: "Llama 3.1 8B",              reasoning: false, contextWindow: 128_000,    maxTokens: 32_768, input: ["text"] },
  { id: "llama3.2-1b",              name: "Llama 3.2 1B",              reasoning: false, contextWindow: 128_000,    maxTokens: 16_384, input: ["text"] },
  { id: "llama3.2-3b",              name: "Llama 3.2 3B",              reasoning: false, contextWindow: 128_000,    maxTokens: 16_384, input: ["text"] },
  { id: "llama3.3-70b",             name: "Llama 3.3 70B",             reasoning: false, contextWindow: 128_000,    maxTokens: 32_768, input: ["text"] },
  { id: "llama4-maverick",          name: "Llama 4 Maverick",          reasoning: false, contextWindow: 1_047_576,  maxTokens: 32_768, input: ["text"] },
  { id: "mistral-large",            name: "Mistral Large",             reasoning: false, contextWindow: 32_000,     maxTokens: 8_192,  input: ["text"] },
  { id: "mistral-large2",           name: "Mistral Large 2",           reasoning: false, contextWindow: 128_000,    maxTokens: 32_768, input: ["text"] },
  { id: "mistral-7b",               name: "Mistral 7B",                reasoning: false, contextWindow: 32_000,     maxTokens: 8_192,  input: ["text"] },
  { id: "snowflake-llama-3.3-70b",  name: "Snowflake Llama 3.3 70B",  reasoning: false, contextWindow: 128_000,    maxTokens: 32_768, input: ["text"] },
];

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
 *  added per-request in wrapStreamFn based on ctx.thinkingLevel.
 *  Returns an empty object when BETA_ALWAYS is empty so we don't attach a
 *  blank `anthropic-beta` header (Snowflake's request schema validator
 *  rejects empty header values on some Cortex builds). */
function anthropicBetaHeaders(): Record<string, string> {
  return BETA_ALWAYS.length > 0
    ? { "anthropic-beta": BETA_ALWAYS.join(",") }
    : {};
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
    compat: { supportsTools: true, supportsEagerToolInputStreaming: false },
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

// Open-source model costs (Table 6c — no cache columns)
const COST_DEEPSEEK_R1  = { input: 0.00000135,  output: 0.0000054,   cacheRead: 0, cacheWrite: 0 }; // $1.35/$5.40
const COST_LLAMA_405B   = { input: 0.0000024,   output: 0.0000024,   cacheRead: 0, cacheWrite: 0 }; // $2.40/$2.40
const COST_LLAMA_70B    = { input: 0.00000072,  output: 0.00000072,  cacheRead: 0, cacheWrite: 0 }; // $0.72/$0.72
const COST_LLAMA_8B     = { input: 0.00000022,  output: 0.00000022,  cacheRead: 0, cacheWrite: 0 }; // $0.22/$0.22
const COST_LLAMA_1B     = { input: 0.0000001,   output: 0.0000001,   cacheRead: 0, cacheWrite: 0 }; // $0.10/$0.10
const COST_LLAMA_3B     = { input: 0.00000015,  output: 0.00000015,  cacheRead: 0, cacheWrite: 0 }; // $0.15/$0.15
const COST_LLAMA4_MAV   = { input: 0.00000024,  output: 0.00000097,  cacheRead: 0, cacheWrite: 0 }; // $0.24/$0.97
const COST_MISTRAL_LG   = { input: 0.000004,    output: 0.000012,    cacheRead: 0, cacheWrite: 0 }; // $4.00/$12.00
const COST_MISTRAL_LG2  = { input: 0.000002,    output: 0.000006,    cacheRead: 0, cacheWrite: 0 }; // $2.00/$6.00
const COST_MISTRAL_7B   = { input: 0.00000015,  output: 0.0000002,   cacheRead: 0, cacheWrite: 0 }; // $0.15/$0.20

/** Map an open-source model ID to its cost tier */
function openSourceCost(id: string): typeof COST_LLAMA_70B {
  if (id === "deepseek-r1")             return COST_DEEPSEEK_R1;
  if (id === "llama3.1-405b")           return COST_LLAMA_405B;
  if (id === "llama3.1-70b")            return COST_LLAMA_70B;
  if (id === "llama3.1-8b")             return COST_LLAMA_8B;
  if (id === "llama3.2-1b")             return COST_LLAMA_1B;
  if (id === "llama3.2-3b")             return COST_LLAMA_3B;
  if (id === "llama3.3-70b")            return COST_LLAMA_70B;
  if (id === "llama4-maverick")         return COST_LLAMA4_MAV;
  if (id === "mistral-large")           return COST_MISTRAL_LG;
  if (id === "mistral-large2")          return COST_MISTRAL_LG2;
  if (id === "mistral-7b")              return COST_MISTRAL_7B;
  if (id === "snowflake-llama-3.3-70b") return COST_LLAMA_70B;
  return COST_LLAMA_70B; // fallback
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
            // Per-request, append BETA_THINKING only when extended thinking
            // is active and the model is reasoning-capable. Catalog headers
            // (BETA_ALWAYS) are seeded above; everything else lives here.
            // Haiku 4.5 (reasoning:false) never receives thinking flags.
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
                  // Defensive: strip `eager_input_streaming` from tool schemas.
                  // The catalog sets supportsEagerToolInputStreaming: false to
                  // prevent pi-ai's Anthropic provider from adding it, but we
                  // also scrub it here so any future SDK regression or alternate
                  // code path can't re-introduce a Cortex-fatal field.
                  stripEagerInputStreaming(record);
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
