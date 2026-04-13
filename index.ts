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
  { id: "claude-opus-4-5",       name: "Claude Opus 4.5",           reasoning: true, contextWindow:   200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5",     name: "Claude Sonnet 4.5",         reasoning: true, contextWindow:   200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-opus-4-6",       name: "Claude Opus 4.6",           reasoning: true, contextWindow:   200_000, maxTokens: 128_000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6",     name: "Claude Sonnet 4.6",         reasoning: true, contextWindow:   200_000, maxTokens: 128_000, input: ["text", "image"] },
  // 1M context variants — requires context-1m-2025-08-07 beta header (Snowflake Cortex)
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

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

function anthropicBetaHeaders(extendedContext = false): Record<string, string> {
  return { "anthropic-beta": extendedContext ? ANTHROPIC_BETA_1M : ANTHROPIC_BETA_DEFAULT };
}

function buildClaudeModelDef(spec: CortexModelSpec): ModelDefinitionConfig {
  return {
    id: spec.id,
    name: spec.name,
    api: "anthropic-messages" as ModelApi,
    reasoning: spec.reasoning,
    input: spec.input,
    cost: ZERO_COST,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    headers: anthropicBetaHeaders(spec.extendedContext),
    compat: { supportsTools: true },
  };
}

function buildOpenAIModelDef(spec: CortexModelSpec): ModelDefinitionConfig {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions" as ModelApi,
    reasoning: spec.reasoning,
    input: spec.input,
    cost: ZERO_COST,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: true,
      maxTokensField: "max_completion_tokens",
    },
  };
}

function buildOpenSourceModelDef(spec: CortexModelSpec): ModelDefinitionConfig {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions" as ModelApi,
    reasoning: spec.reasoning,
    input: spec.input,
    cost: ZERO_COST,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: false,
      maxTokensField: "max_completion_tokens",
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

export default definePluginEntry({
  id: "snowflake-cortex",
  name: "Snowflake Cortex",
  description:
    "Snowflake Cortex AI — routes Claude models to Anthropic Messages API " +
    "and all other models to OpenAI-compatible Chat Completions, both " +
    "behind PAT authentication.",

  register(api) {
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
