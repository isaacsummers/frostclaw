/**
 * Model catalog — all Snowflake Cortex models with per-model API routing,
 * cost, and compat flags.
 *
 * This module has ZERO openclaw imports so it can be imported directly in
 * unit tests without any module stubbing.
 */

import { isClaudeModel } from "./transforms.js";

// ---------------------------------------------------------------------------
// Local type aliases — structurally identical to the openclaw SDK types but
// defined here so this file has no openclaw dependency.
// ---------------------------------------------------------------------------

export type ModelApi = string;

export type CostConfig = {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
};

export type CompatConfig = Record<string, unknown>;

export type ModelDefinitionConfig = {
  id: string;
  name: string;
  provider?: string;
  api?: ModelApi;
  baseUrl?: string;
  reasoning: boolean;
  input: Array<"text" | "image">;
  cost: CostConfig;
  contextWindow: number;
  maxTokens: number;
  headers?: Record<string, string>;
  compat?: CompatConfig;
};

// ---------------------------------------------------------------------------
// Base URL helper — reads env at call time (same logic as index.ts)
// ---------------------------------------------------------------------------

export function getCatalogBaseURL(): string {
  const raw = process.env.SNOWFLAKE_PROXY_BASE_URL ?? process.env.SNOWFLAKE_BASE_URL ?? "";
  return raw ? `${raw.replace(/\/$/, "")}/api/v2/cortex` : "";
}

// ---------------------------------------------------------------------------
// Beta headers
// ---------------------------------------------------------------------------

/** Flags safe to send on every Claude request. Empty by design — see index.ts. */
export const BETA_ALWAYS: string[] = [];

/** Catalog-level beta headers. Returns {} when BETA_ALWAYS is empty so we
 *  don't attach a blank anthropic-beta header. */
export function anthropicBetaHeaders(): Record<string, string> {
  return BETA_ALWAYS.length > 0
    ? { "anthropic-beta": BETA_ALWAYS.join(",") }
    : {};
}

// ---------------------------------------------------------------------------
// Model specs
// ---------------------------------------------------------------------------

export interface CortexModelSpec {
  id: string;
  name: string;
  reasoning: boolean;
  contextWindow: number;
  maxTokens: number;
  input: Array<"text" | "image">;
}

export const CLAUDE_MODELS: CortexModelSpec[] = [
  // Claude 4 family
  { id: "claude-4-opus",                   name: "Claude 4 Opus",                    reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-opus-4-8",                  name: "Claude Opus 4.8",                  reasoning: true,  contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
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

export const OPENAI_MODELS: CortexModelSpec[] = [
  { id: "openai-gpt-5.5",              name: "GPT-5.5",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5.5-long-context", name: "GPT-5.5 (Long Context)", reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5.4",              name: "GPT-5.4",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5.4-long-context", name: "GPT-5.4 (Long Context)", reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5.2",              name: "GPT-5.2",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5.1",              name: "GPT-5.1",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5",               name: "GPT-5",                  reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5-mini",           name: "GPT-5 Mini",             reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5-nano",           name: "GPT-5 Nano",             reasoning: false, contextWindow: 128_000,   maxTokens: 16_384, input: ["text", "image"] },
  { id: "openai-gpt-4.1",              name: "GPT-4.1",                reasoning: false, contextWindow: 1_047_576, maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-o4-mini",              name: "o4-mini",                reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
];

export const OPEN_SOURCE_MODELS: CortexModelSpec[] = [
  { id: "deepseek-r1",              name: "DeepSeek R1",               reasoning: false, contextWindow: 64_000,    maxTokens: 16_384, input: ["text"] },
  { id: "llama3.1-405b",            name: "Llama 3.1 405B",            reasoning: false, contextWindow: 128_000,   maxTokens: 32_768, input: ["text"] },
  { id: "llama3.1-70b",             name: "Llama 3.1 70B",             reasoning: false, contextWindow: 128_000,   maxTokens: 32_768, input: ["text"] },
  { id: "llama3.1-8b",              name: "Llama 3.1 8B",              reasoning: false, contextWindow: 128_000,   maxTokens: 32_768, input: ["text"] },
  { id: "llama3.2-1b",              name: "Llama 3.2 1B",              reasoning: false, contextWindow: 128_000,   maxTokens: 16_384, input: ["text"] },
  { id: "llama3.2-3b",              name: "Llama 3.2 3B",              reasoning: false, contextWindow: 128_000,   maxTokens: 16_384, input: ["text"] },
  { id: "llama3.3-70b",             name: "Llama 3.3 70B",             reasoning: false, contextWindow: 128_000,   maxTokens: 32_768, input: ["text"] },
  { id: "llama4-maverick",          name: "Llama 4 Maverick",          reasoning: false, contextWindow: 1_047_576, maxTokens: 32_768, input: ["text"] },
  { id: "mistral-large",            name: "Mistral Large",             reasoning: false, contextWindow: 32_000,    maxTokens: 8_192,  input: ["text"] },
  { id: "mistral-large2",           name: "Mistral Large 2",           reasoning: false, contextWindow: 128_000,   maxTokens: 32_768, input: ["text"] },
  { id: "mistral-7b",               name: "Mistral 7B",                reasoning: false, contextWindow: 32_000,    maxTokens: 8_192,  input: ["text"] },
  { id: "snowflake-llama-3.3-70b",  name: "Snowflake Llama 3.3 70B",  reasoning: false, contextWindow: 128_000,   maxTokens: 32_768, input: ["text"] },
];

// ---------------------------------------------------------------------------
// Cost constants
// ---------------------------------------------------------------------------

// Claude models (AWS Regional)
const COST_CLAUDE_4_OPUS  = { input: 0.000015,   output: 0.000075,   cacheRead: 0.0000015,  cacheWrite: 0.00001875 };
const COST_OPUS           = { input: 0.0000055,  output: 0.0000275,  cacheRead: 0.00000055, cacheWrite: 0.000006875 };
const COST_SONNET         = { input: 0.0000033,  output: 0.0000165,  cacheRead: 0.00000033, cacheWrite: 0.000004125 };
const COST_SONNET_LONG    = { input: 0.0000066,  output: 0.00002475, cacheRead: 0.00000066, cacheWrite: 0.00000825 };
const COST_HAIKU          = { input: 0.0000011,  output: 0.0000055,  cacheRead: 0.00000011, cacheWrite: 0.000001375 };
const COST_CLAUDE_37      = { input: 0.000003,   output: 0.000015,   cacheRead: 0.0000003,  cacheWrite: 0.000003750 };
const COST_CLAUDE_4S      = { input: 0.000003,   output: 0.000015,   cacheRead: 0.0000003,  cacheWrite: 0.000003750 };

// OpenAI models (Azure Regional)
const COST_GPT55          = { input: 0.0000055,  output: 0.000033,   cacheRead: 0.00000055, cacheWrite: 0 };
const COST_GPT55_LONG     = { input: 0.000011,   output: 0.0000495,  cacheRead: 0.0000011,  cacheWrite: 0 };
const COST_GPT54          = { input: 0.00000275, output: 0.0000165,  cacheRead: 0.00000028, cacheWrite: 0 };
const COST_GPT54_LONG     = { input: 0.0000055,  output: 0.00002475, cacheRead: 0.00000055, cacheWrite: 0 };
const COST_GPT52          = { input: 0.00000193, output: 0.0000154,  cacheRead: 0.00000019, cacheWrite: 0 };
const COST_GPT51          = { input: 0.00000138, output: 0.000011,   cacheRead: 0.00000014, cacheWrite: 0 };
const COST_GPT5_MINI      = { input: 0.00000028, output: 0.0000022,  cacheRead: 0.000000028, cacheWrite: 0 };
const COST_GPT5_NANO      = { input: 0.00000006, output: 0.00000044, cacheRead: 0.000000006, cacheWrite: 0 };
const COST_GPT41          = { input: 0.0000022,  output: 0.0000088,  cacheRead: 0.00000055, cacheWrite: 0 };
const COST_O4_MINI        = { input: 0.0000011,  output: 0.0000044,  cacheRead: 0.00000028, cacheWrite: 0 };

// Open-source models
const COST_DEEPSEEK_R1    = { input: 0.00000135, output: 0.0000054,  cacheRead: 0, cacheWrite: 0 };
const COST_LLAMA_405B     = { input: 0.0000024,  output: 0.0000024,  cacheRead: 0, cacheWrite: 0 };
const COST_LLAMA_70B      = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 };
const COST_LLAMA_8B       = { input: 0.00000022, output: 0.00000022, cacheRead: 0, cacheWrite: 0 };
const COST_LLAMA_1B       = { input: 0.0000001,  output: 0.0000001,  cacheRead: 0, cacheWrite: 0 };
const COST_LLAMA_3B       = { input: 0.00000015, output: 0.00000015, cacheRead: 0, cacheWrite: 0 };
const COST_LLAMA4_MAV     = { input: 0.00000024, output: 0.00000097, cacheRead: 0, cacheWrite: 0 };
const COST_MISTRAL_LG     = { input: 0.000004,   output: 0.000012,   cacheRead: 0, cacheWrite: 0 };
const COST_MISTRAL_LG2    = { input: 0.000002,   output: 0.000006,   cacheRead: 0, cacheWrite: 0 };
const COST_MISTRAL_7B     = { input: 0.00000015, output: 0.0000002,  cacheRead: 0, cacheWrite: 0 };

// ---------------------------------------------------------------------------
// Builder functions
// ---------------------------------------------------------------------------

function claudeCost(id: string): CostConfig {
  if (id === "claude-4-opus")         return COST_CLAUDE_4_OPUS;
  if (id.startsWith("claude-opus"))   return COST_OPUS;
  if (id.endsWith("-long-context"))   return COST_SONNET_LONG;
  if (id === "claude-4-sonnet")       return COST_CLAUDE_4S;
  if (id === "claude-3-7-sonnet")     return COST_CLAUDE_37;
  if (id.startsWith("claude-sonnet")) return COST_SONNET;
  if (id.startsWith("claude-haiku"))  return COST_HAIKU;
  return COST_OPUS;
}

export function buildClaudeModelDef(spec: CortexModelSpec, baseURL?: string): ModelDefinitionConfig {
  const url = baseURL ?? getCatalogBaseURL();
  return {
    id: spec.id,
    name: spec.name,
    provider: "snowflake-cortex",
    api: "anthropic-messages",
    baseUrl: url,
    reasoning: spec.reasoning,
    input: spec.input,
    cost: claudeCost(spec.id),
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    headers: anthropicBetaHeaders(),
    compat: { supportsTools: true, supportsEagerToolInputStreaming: false },
  };
}

function openaiCost(id: string): CostConfig {
  if (id === "openai-gpt-5.5-long-context") return COST_GPT55_LONG;
  if (id === "openai-gpt-5.5")              return COST_GPT55;
  if (id === "openai-gpt-5.4-long-context") return COST_GPT54_LONG;
  if (id === "openai-gpt-5.4")              return COST_GPT54;
  if (id === "openai-gpt-5.2")              return COST_GPT52;
  if (id === "openai-gpt-5.1")              return COST_GPT51;
  if (id === "openai-gpt-5")               return COST_GPT51;
  if (id === "openai-gpt-5-mini")           return COST_GPT5_MINI;
  if (id === "openai-gpt-5-nano")           return COST_GPT5_NANO;
  if (id.startsWith("openai-gpt-4"))        return COST_GPT41;
  if (id === "openai-o4-mini")              return COST_O4_MINI;
  return COST_GPT51;
}

export function buildOpenAIModelDef(spec: CortexModelSpec, baseURL?: string): ModelDefinitionConfig {
  const url = baseURL ?? getCatalogBaseURL();
  return {
    id: spec.id,
    name: spec.name,
    provider: "snowflake-cortex",
    api: "openai-completions",
    baseUrl: url,
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

function openSourceCost(id: string): CostConfig {
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
  return COST_LLAMA_70B;
}

export function buildOpenSourceModelDef(spec: CortexModelSpec, baseURL?: string): ModelDefinitionConfig {
  const url = baseURL ?? getCatalogBaseURL();
  return {
    id: spec.id,
    name: spec.name,
    provider: "snowflake-cortex",
    api: "openai-completions",
    baseUrl: url,
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

// ---------------------------------------------------------------------------
// Catalog + index
// ---------------------------------------------------------------------------

let CATALOG_CACHE: { baseURL: string; catalog: ModelDefinitionConfig[] } | undefined;
let CATALOG_INDEX: Map<string, ModelDefinitionConfig> | undefined;

export function buildModelCatalog(): ModelDefinitionConfig[] {
  const baseURL = getCatalogBaseURL();
  if (CATALOG_CACHE && CATALOG_CACHE.baseURL === baseURL) {
    return CATALOG_CACHE.catalog;
  }
  const catalog: ModelDefinitionConfig[] = [
    ...CLAUDE_MODELS.map((s) => buildClaudeModelDef(s, baseURL)),
    ...OPENAI_MODELS.map((s) => buildOpenAIModelDef(s, baseURL)),
    ...OPEN_SOURCE_MODELS.map((s) => buildOpenSourceModelDef(s, baseURL)),
  ];
  CATALOG_CACHE = { baseURL, catalog };
  CATALOG_INDEX = new Map(catalog.map((m) => [m.id, m]));
  return catalog;
}

export function findCatalogEntry(modelId: string): ModelDefinitionConfig | undefined {
  const bareId = modelId.replace(/^snowflake-cortex\//, "");
  if (!CATALOG_INDEX) buildModelCatalog();
  return CATALOG_INDEX?.get(bareId);
}

// Re-export isClaudeModel for convenience (used in index.ts alongside catalog)
export { isClaudeModel };
