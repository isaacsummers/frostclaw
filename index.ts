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
 * Snowflake Cortex rejects text content blocks with empty or whitespace-only
 * text. Replace empty/whitespace strings with a zero-width space so the block
 * structure is preserved while satisfying the API constraint.
 * This handles the /new command case where OpenClaw sends a blank user turn.
 */
function fixEmptyTextBlocks(messages: unknown[]): unknown[] {
  return messages.map((msg) => {
    if (!msg || typeof msg !== "object") return msg;
    const m = msg as Record<string, unknown>;
    if (!Array.isArray(m.content)) return msg;
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
  { id: "claude-opus-4-7",   name: "Claude Opus 4.7",   reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-opus-4-6",   name: "Claude Opus 4.6",   reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-opus-4-5",   name: "Claude Opus 4.5",   reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", reasoning: true, contextWindow: 200_000, maxTokens: 128_000, input: ["text", "image"] },
];

const OPENAI_MODELS: CortexModelSpec[] = [
  { id: "openai-gpt-5.4",      name: "GPT-5.4",       reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] }, // preview
  { id: "openai-gpt-5.2",      name: "GPT-5.2",       reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5.1",      name: "GPT-5.1",       reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] },
  { id: "openai-gpt-5",        name: "GPT-5",         reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] }, // preview
  { id: "openai-gpt-5-mini",   name: "GPT-5 Mini",    reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text", "image"] }, // preview
  { id: "openai-gpt-5-nano",   name: "GPT-5 Nano",    reasoning: false, contextWindow: 128_000,   maxTokens: 16_384, input: ["text", "image"] }, // preview
  { id: "openai-gpt-oss-120b", name: "GPT OSS 120B",  reasoning: true,  contextWindow: 128_000,   maxTokens: 32_768, input: ["text"] },           // preview
  { id: "openai-gpt-4.1",      name: "GPT-4.1",       reasoning: false, contextWindow: 1_047_576, maxTokens: 32_768, input: ["text", "image"] },
];

const OPEN_SOURCE_MODELS: CortexModelSpec[] = [
  { id: "llama4-maverick", name: "Llama 4 Maverick", reasoning: false, contextWindow: 1_048_576, maxTokens: 16_384, input: ["text"] },
  { id: "llama3.1-70b", name: "Llama 3.1 70B", reasoning: false, contextWindow: 128_000, maxTokens: 4_096, input: ["text"] },
  { id: "llama3.1-8b", name: "Llama 3.1 8B", reasoning: false, contextWindow: 128_000, maxTokens: 4_096, input: ["text"] },
  { id: "llama3.1-405b", name: "Llama 3.1 405B", reasoning: false, contextWindow: 128_000, maxTokens: 4_096, input: ["text"] },
  { id: "mistral-large", name: "Mistral Large", reasoning: false, contextWindow: 32_000, maxTokens: 8_192, input: ["text"] },
  { id: "mistral-large2", name: "Mistral Large 2", reasoning: false, contextWindow: 128_000, maxTokens: 8_192, input: ["text"] },
  { id: "deepseek-r1", name: "DeepSeek R1", reasoning: true, contextWindow: 64_000, maxTokens: 8_192, input: ["text"] },
  { id: "snowflake-arctic",        name: "Snowflake Arctic",         reasoning: false, contextWindow:   4_096, maxTokens: 4_096, input: ["text"] },
  { id: "snowflake-llama-3.3-70b", name: "Snowflake Llama 3.3 70B", reasoning: false, contextWindow: 128_000, maxTokens: 4_096, input: ["text"] },
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
const COST_OPUS   = { input: 0.0000055,   output: 0.0000275,   cacheRead: 0.00000055, cacheWrite: 0 }; // $5.50/$27.50/$0.55 per 1M (AWS Regional)
const COST_SONNET = { input: 0.0000033,   output: 0.0000165,   cacheRead: 0.00000033, cacheWrite: 0 }; // $3.30/$16.50/$0.33 per 1M (AWS Regional)
const COST_HAIKU  = { input: 0.0000011,   output: 0.0000055,   cacheRead: 0.00000011, cacheWrite: 0 }; // $1.10/$5.50/$0.11 per 1M (AWS Regional)

// Table 6(b) — OpenAI models (no separate cache-read pricing listed; using 10% heuristic)
const COST_GPT54  = { input: 0.0000025,   output: 0.000015,    cacheRead: 0.00000025,  cacheWrite: 0 }; // $2.50/$15/$0.25 per 1M (gpt-5.4)
const COST_GPT52  = { input: 0.00000175,  output: 0.000014,    cacheRead: 0.000000175, cacheWrite: 0 }; // $1.75/$14/$0.18 per 1M (gpt-5.2)
const COST_GPT5   = { input: 0.00000125,  output: 0.00001,     cacheRead: 0.000000125, cacheWrite: 0 }; // $1.25/$10/$0.13 per 1M (gpt-5, gpt-5.1)
const COST_GPT41  = { input: 0.000002,    output: 0.000008,    cacheRead: 0.0000002,   cacheWrite: 0 }; // $2/$8/$0.20 per 1M (gpt-4.1)

// Table 6(c) — Open-source models (no prompt caching)
const COST_LLAMA_405B = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 }; // approx, not in 6(c); using 70b rate
const COST_LLAMA_70B  = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 }; // $0.72/$0.72 per 1M
const COST_LLAMA_8B   = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 }; // not in 6(c); using 70b rate as fallback
const COST_LLAMA4_MAV = { input: 0.00000024, output: 0.00000097, cacheRead: 0, cacheWrite: 0 }; // $0.24/$0.97 per 1M
const COST_MISTRAL_L  = { input: 0.000002,   output: 0.000006,   cacheRead: 0, cacheWrite: 0 }; // legacy; using L2 rate
const COST_MISTRAL_L2 = { input: 0.000002,   output: 0.000006,   cacheRead: 0, cacheWrite: 0 }; // $2/$6 per 1M
const COST_DEEPSEEK   = { input: 0.00000135, output: 0.0000054,  cacheRead: 0, cacheWrite: 0 }; // $1.35/$5.40 per 1M
const COST_ARCTIC     = { input: 0,          output: 0,           cacheRead: 0, cacheWrite: 0 }; // Snowflake native — free tier

/** Catalog-level beta headers — always-safe flags only. Thinking flags are
 *  added per-request in wrapStreamFn based on ctx.thinkingLevel. */
function anthropicBetaHeaders(): Record<string, string> {
  return { "anthropic-beta": BETA_ALWAYS.join(",") };
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
    headers: anthropicBetaHeaders(),
    compat: { supportsTools: true },
  };
}

/** Map an OpenAI model ID to its cost tier */
function openaiCost(id: string): typeof COST_GPT5 {
  if (id === "openai-gpt-5.4") return COST_GPT54;
  if (id === "openai-gpt-5.2") return COST_GPT52;
  if (id.startsWith("openai-gpt-4")) return COST_GPT41;
  return COST_GPT5; // gpt-5, gpt-5.1, gpt-5-mini, gpt-5-nano, gpt-oss-120b
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
  if (id === "snowflake-llama-3.3-70b") return COST_LLAMA_70B;
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
      // Hook: Inject Snowflake PAT header type and conditionally add thinking
      // beta flags based on ctx.thinkingLevel for Claude models.
      //
      // Beta flags are split: always-safe flags come from the catalog headers,
      // while thinking-specific flags (interleaved-thinking, effort,
      // tool-examples) are only added when the request uses thinking. This
      // prevents Snowflake Cortex from rejecting unknown beta flags with 400.
      // -----------------------------------------------------------------------
      wrapStreamFn(ctx: ProviderWrapStreamFnContext) {
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
          const originalOnPayload = options?.onPayload;

          // Build per-request anthropic-beta header: start with any
          // catalog-level flags from the model definition, then append
          // thinking flags when the request actually uses thinking.
          const catalogBeta =
            (model as { headers?: Record<string, string> })?.headers?.[
              "anthropic-beta"
            ] ?? "";
          const betaFlags = catalogBeta ? [catalogBeta] : [];
          if (thinkingActive) {
            betaFlags.push(BETA_THINKING.join(","));
          }

          const merged = {
            ...options,
            headers: {
              ...options?.headers,
              "X-Snowflake-Authorization-Token-Type":
                "PROGRAMMATIC_ACCESS_TOKEN",
              ...(betaFlags.length > 0
                ? { "anthropic-beta": betaFlags.join(",") }
                : {}),
            },
            onPayload: (payload: unknown, payloadModel: unknown) => {
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
              }
              return (originalOnPayload as
                | ((p: unknown, m: unknown) => unknown)
                | undefined)?.(payload, payloadModel);
            },
          };
          return inner(model, context, merged as typeof options);
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
  },
});
