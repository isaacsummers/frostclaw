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
  type ProviderNormalizeModelIdContext,
} from "openclaw/plugin-sdk/plugin-entry";
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
  return process.env.SNOWFLAKE_PAT ?? process.env.SNOWFLAKE_CORTEX_API_KEY ?? "";
}
function getBaseURL(): string {
  return process.env.SNOWFLAKE_BASE_URL ?? "";
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
// Anthropic beta headers — attached per-model via catalog headers field
//
// Note: Snowflake Cortex keeps these as active beta headers even where
// Anthropic has GA'd them. Use them as-is for Snowflake compatibility.
// ---------------------------------------------------------------------------

const ANTHROPIC_BETA_DEFAULT = [
  "interleaved-thinking-2025-05-14",
  "output-128k-2025-02-19",
  "effort-2025-11-24",
  "token-efficient-tools-2025-02-19",
  "tool-examples-2025-10-29",
].join(",");

// 1M context requires an explicit opt-in header on Snowflake Cortex.
// Applied only to the *-1m model variants — never to the base models.
const ANTHROPIC_BETA_1M = [
  "context-1m-2025-08-07",
  "interleaved-thinking-2025-05-14",
  "output-128k-2025-02-19",
  "effort-2025-11-24",
  "token-efficient-tools-2025-02-19",
  "tool-examples-2025-10-29",
].join(",");

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

// ---------------------------------------------------------------------------
// Model catalog — all 16 Cortex models with per-model API routing + compat
// ---------------------------------------------------------------------------

interface CortexModelSpec {
  id: string;
  name: string;
  reasoning: boolean;
  contextWindow: number;
  maxTokens: number;
  input: Array<"text" | "image">;
  extendedContext?: boolean; // true → attach context-1m beta header
}

const CLAUDE_MODELS: CortexModelSpec[] = [
  { id: "claude-opus-4-7",       name: "Claude Opus 4.7",           reasoning: true, contextWindow:   200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-opus-4-6",       name: "Claude Opus 4.6",           reasoning: true, contextWindow:   200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-opus-4-5",       name: "Claude Opus 4.5",           reasoning: true, contextWindow:   200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6",     name: "Claude Sonnet 4.6",         reasoning: true, contextWindow:   200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5",     name: "Claude Sonnet 4.5",         reasoning: true, contextWindow:   200_000, maxTokens: 128_000, input: ["text", "image"] },
  // 1M context variants — requires context-1m-2025-08-07 beta header (Snowflake Cortex)
  { id: "claude-opus-4-7-1m",   name: "Claude Opus 4.7 (1M)",   reasoning: true, contextWindow: 1_000_000, maxTokens: 128_000, input: ["text", "image"], extendedContext: true },
  { id: "claude-opus-4-6-1m",   name: "Claude Opus 4.6 (1M)",   reasoning: true, contextWindow: 1_000_000, maxTokens: 128_000, input: ["text", "image"], extendedContext: true },
  { id: "claude-sonnet-4-6-1m", name: "Claude Sonnet 4.6 (1M)", reasoning: true, contextWindow: 1_000_000, maxTokens: 128_000, input: ["text", "image"], extendedContext: true },
];

const OPENAI_MODELS: CortexModelSpec[] = [
  { id: "openai-gpt-5", name: "GPT-5", reasoning: true, contextWindow: 128_000, maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5-mini", name: "GPT-5 Mini", reasoning: true, contextWindow: 128_000, maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5-nano", name: "GPT-5 Nano", reasoning: false, contextWindow: 128_000, maxTokens: 16_384, input: ["text", "image"] },
  { id: "openai-gpt-4-1", name: "GPT-4.1", reasoning: false, contextWindow: 1_047_576, maxTokens: 32_768, input: ["text", "image"] },
];

const OPEN_SOURCE_MODELS: CortexModelSpec[] = [
  { id: "llama4-maverick", name: "Llama 4 Maverick", reasoning: false, contextWindow: 1_048_576, maxTokens: 16_384, input: ["text"] },
  { id: "llama3.1-70b", name: "Llama 3.1 70B", reasoning: false, contextWindow: 128_000, maxTokens: 4_096, input: ["text"] },
  { id: "llama3.1-8b", name: "Llama 3.1 8B", reasoning: false, contextWindow: 128_000, maxTokens: 4_096, input: ["text"] },
  { id: "llama3.1-405b", name: "Llama 3.1 405B", reasoning: false, contextWindow: 128_000, maxTokens: 4_096, input: ["text"] },
  { id: "mistral-large", name: "Mistral Large", reasoning: false, contextWindow: 32_000, maxTokens: 8_192, input: ["text"] },
  { id: "mistral-large2", name: "Mistral Large 2", reasoning: false, contextWindow: 128_000, maxTokens: 8_192, input: ["text"] },
  { id: "deepseek-r1", name: "DeepSeek R1", reasoning: true, contextWindow: 64_000, maxTokens: 8_192, input: ["text"] },
  { id: "snowflake-arctic", name: "Snowflake Arctic", reasoning: false, contextWindow: 4_096, maxTokens: 4_096, input: ["text"] },
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

// Table 6(b) — Claude models
const COST_OPUS   = { input: 0.000005,    output: 0.000025,    cacheRead: 0.0000005,  cacheWrite: 0 }; // $5/$25/$0.50 per 1M
const COST_SONNET = { input: 0.000003,    output: 0.000015,    cacheRead: 0.0000003,  cacheWrite: 0 }; // $3/$15/$0.30 per 1M
const COST_HAIKU  = { input: 0.000001,    output: 0.000005,    cacheRead: 0.0000001,  cacheWrite: 0 }; // $1/$5/$0.10 per 1M

// Table 6(b) — OpenAI models (no separate cache-read pricing listed; using 10% heuristic)
const COST_GPT5   = { input: 0.0000025,   output: 0.00001,     cacheRead: 0.00000025, cacheWrite: 0 }; // $2.50/$10 per 1M (gpt-5.4 rate)
const COST_GPT41  = { input: 0.000002,    output: 0.000008,    cacheRead: 0.0000002,  cacheWrite: 0 }; // $2/$8 per 1M

// Table 6(c) — Open-source models (no prompt caching)
const COST_LLAMA_405B = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 }; // approx, not in 6(c); using 70b rate
const COST_LLAMA_70B  = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 }; // $0.72/$0.72 per 1M
const COST_LLAMA_8B   = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 }; // not in 6(c); using 70b rate as fallback
const COST_LLAMA4_MAV = { input: 0.00000024, output: 0.00000097, cacheRead: 0, cacheWrite: 0 }; // $0.24/$0.97 per 1M
const COST_MISTRAL_L  = { input: 0.000002,   output: 0.000006,   cacheRead: 0, cacheWrite: 0 }; // legacy; using L2 rate
const COST_MISTRAL_L2 = { input: 0.000002,   output: 0.000006,   cacheRead: 0, cacheWrite: 0 }; // $2/$6 per 1M
const COST_DEEPSEEK   = { input: 0.00000135, output: 0.0000054,  cacheRead: 0, cacheWrite: 0 }; // $1.35/$5.40 per 1M
const COST_ARCTIC     = { input: 0,          output: 0,           cacheRead: 0, cacheWrite: 0 }; // Snowflake native — free tier

function anthropicBetaHeaders(extendedContext = false): Record<string, string> {
  return { "anthropic-beta": extendedContext ? ANTHROPIC_BETA_1M : ANTHROPIC_BETA_DEFAULT };
}

/** Map a Claude model ID to its cost tier */
function claudeCost(id: string): typeof COST_OPUS {
  if (id.startsWith("claude-opus")) return COST_OPUS;
  if (id.startsWith("claude-sonnet")) return COST_SONNET;
  if (id.startsWith("claude-haiku")) return COST_HAIKU;
  return COST_OPUS; // fallback to most expensive
}

function buildClaudeModelDef(spec: CortexModelSpec): ModelDefinitionConfig {
  return {
    id: spec.id,
    name: spec.name,
    api: "anthropic-messages" as ModelApi,
    reasoning: spec.reasoning,
    input: spec.input,
    cost: claudeCost(spec.id),
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    headers: anthropicBetaHeaders(spec.extendedContext),
    compat: { supportsTools: true },
  };
}

/** Map an OpenAI model ID to its cost tier */
function openaiCost(id: string): typeof COST_GPT5 {
  if (id.startsWith("openai-gpt-5")) return COST_GPT5;
  if (id.startsWith("openai-gpt-4")) return COST_GPT41;
  return COST_GPT5; // fallback
}

function buildOpenAIModelDef(spec: CortexModelSpec): ModelDefinitionConfig {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions" as ModelApi,
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
function openSourceCost(id: string): typeof COST_LLAMA_70B {
  if (id === "llama4-maverick") return COST_LLAMA4_MAV;
  if (id === "llama3.1-405b") return COST_LLAMA_405B;
  if (id === "llama3.1-70b" || id === "llama3.3-70b") return COST_LLAMA_70B;
  if (id === "llama3.1-8b") return COST_LLAMA_8B;
  if (id === "mistral-large") return COST_MISTRAL_L;
  if (id === "mistral-large2") return COST_MISTRAL_L2;
  if (id === "deepseek-r1") return COST_DEEPSEEK;
  if (id === "snowflake-arctic") return COST_ARCTIC;
  return COST_LLAMA_70B; // fallback
}

function buildOpenSourceModelDef(spec: CortexModelSpec): ModelDefinitionConfig {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions" as ModelApi,
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

function buildModelCatalog(): ModelDefinitionConfig[] {
  return [
    ...CLAUDE_MODELS.map(buildClaudeModelDef),
    ...OPENAI_MODELS.map(buildOpenAIModelDef),
    ...OPEN_SOURCE_MODELS.map(buildOpenSourceModelDef),
  ];
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

    if (!getApiKey() || !getBaseURL()) {
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
          envVar: "SNOWFLAKE_PAT",
          promptMessage:
            "Enter your Snowflake Programmatic Access Token (PAT):",
        }),
      ],

      catalog: {
        run: async (ctx) => {
          const resolvedKey =
            ctx.resolveProviderApiKey("snowflake-cortex").apiKey ?? getApiKey();
          if (!resolvedKey || !getBaseURL()) return null;

          return {
            provider: {
              baseUrl: `${getBaseURL()}/api/v2/cortex/v1`,
              apiKey: resolvedKey,
              api: "openai-completions" as ModelApi,
              authHeader: true,
              models: buildModelCatalog(),
            },
          };
        },
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
      // Hook: Strip -1m suffix before sending model ID to Snowflake.
      // The -1m variants are virtual catalog entries — the real wire model ID
      // is the base name (e.g. "claude-sonnet-4-6-1m" → "claude-sonnet-4-6").
      // The context-1m beta header is already set per-model in the catalog.
      // -----------------------------------------------------------------------
      normalizeModelId(ctx: ProviderNormalizeModelIdContext) {
        return ctx.modelId.replace(/-1m$/, "") || null;
      },

      // -----------------------------------------------------------------------
      // Hook 20: Inject Snowflake PAT header type — no body patching
      // -----------------------------------------------------------------------
      wrapStreamFn(ctx: ProviderWrapStreamFnContext) {
        if (!ctx.streamFn) return undefined;

        const inner = ctx.streamFn;
        return (model, context, options) => {
          const merged = {
            ...options,
            headers: {
              ...options?.headers,
              "X-Snowflake-Authorization-Token-Type":
                "PROGRAMMATIC_ACCESS_TOKEN",
            },
          };
          return inner(model, context, merged);
        };
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
  },
});
