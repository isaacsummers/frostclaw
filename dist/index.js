// index.ts
import {
  definePluginEntry
} from "openclaw/plugin-sdk/plugin-entry";
import { createProviderApiKeyAuthMethod } from "openclaw/plugin-sdk/provider-auth-api-key";
function getApiKey() {
  return process.env.SNOWFLAKE_PAT ?? process.env.SNOWFLAKE_CORTEX_API_KEY ?? "";
}
function getBaseURL() {
  return process.env.SNOWFLAKE_BASE_URL ?? "";
}
var ANTHROPIC_BETA_DEFAULT = [
  "interleaved-thinking-2025-05-14",
  "output-128k-2025-02-19",
  "effort-2025-11-24",
  "token-efficient-tools-2025-02-19",
  "tool-examples-2025-10-29"
].join(",");
var ANTHROPIC_BETA_1M = [
  "context-1m-2025-08-07",
  "interleaved-thinking-2025-05-14",
  "output-128k-2025-02-19",
  "effort-2025-11-24",
  "token-efficient-tools-2025-02-19",
  "tool-examples-2025-10-29"
].join(",");
function isClaudeModel(modelId) {
  return modelId.toLowerCase().startsWith("claude");
}
function modelSupportsTools(modelId) {
  return modelId.toLowerCase().startsWith("openai-");
}
var CLAUDE_MODELS = [
  { id: "claude-opus-4-5", name: "Claude Opus 4.5", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-6", name: "Claude Opus 4.6", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-6-1m", name: "Claude Opus 4.6 (1M)", reasoning: true, contextWindow: 1e6, maxTokens: 128000, input: ["text", "image"], extendedContext: true },
  { id: "claude-sonnet-4-6-1m", name: "Claude Sonnet 4.6 (1M)", reasoning: true, contextWindow: 1e6, maxTokens: 128000, input: ["text", "image"], extendedContext: true }
];
var OPENAI_MODELS = [
  { id: "openai-gpt-5", name: "GPT-5", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5-mini", name: "GPT-5 Mini", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5-nano", name: "GPT-5 Nano", reasoning: false, contextWindow: 128000, maxTokens: 16384, input: ["text", "image"] },
  { id: "openai-gpt-4-1", name: "GPT-4.1", reasoning: false, contextWindow: 1047576, maxTokens: 32768, input: ["text", "image"] }
];
var OPEN_SOURCE_MODELS = [
  { id: "llama4-maverick", name: "Llama 4 Maverick", reasoning: false, contextWindow: 1048576, maxTokens: 16384, input: ["text"] },
  { id: "llama3.1-70b", name: "Llama 3.1 70B", reasoning: false, contextWindow: 128000, maxTokens: 4096, input: ["text"] },
  { id: "llama3.1-8b", name: "Llama 3.1 8B", reasoning: false, contextWindow: 128000, maxTokens: 4096, input: ["text"] },
  { id: "llama3.1-405b", name: "Llama 3.1 405B", reasoning: false, contextWindow: 128000, maxTokens: 4096, input: ["text"] },
  { id: "mistral-large", name: "Mistral Large", reasoning: false, contextWindow: 32000, maxTokens: 8192, input: ["text"] },
  { id: "mistral-large2", name: "Mistral Large 2", reasoning: false, contextWindow: 128000, maxTokens: 8192, input: ["text"] },
  { id: "deepseek-r1", name: "DeepSeek R1", reasoning: true, contextWindow: 64000, maxTokens: 8192, input: ["text"] },
  { id: "snowflake-arctic", name: "Snowflake Arctic", reasoning: false, contextWindow: 4096, maxTokens: 4096, input: ["text"] }
];
var ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
function anthropicBetaHeaders(extendedContext = false) {
  return { "anthropic-beta": extendedContext ? ANTHROPIC_BETA_1M : ANTHROPIC_BETA_DEFAULT };
}
function buildClaudeModelDef(spec) {
  return {
    id: spec.id,
    name: spec.name,
    api: "anthropic-messages",
    reasoning: spec.reasoning,
    input: spec.input,
    cost: ZERO_COST,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    headers: anthropicBetaHeaders(spec.extendedContext),
    compat: { supportsTools: true }
  };
}
function buildOpenAIModelDef(spec) {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions",
    reasoning: spec.reasoning,
    input: spec.input,
    cost: ZERO_COST,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: true,
      maxTokensField: "max_completion_tokens"
    }
  };
}
function buildOpenSourceModelDef(spec) {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions",
    reasoning: spec.reasoning,
    input: spec.input,
    cost: ZERO_COST,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: false,
      maxTokensField: "max_completion_tokens"
    }
  };
}
function buildModelCatalog() {
  return [
    ...CLAUDE_MODELS.map(buildClaudeModelDef),
    ...OPENAI_MODELS.map(buildOpenAIModelDef),
    ...OPEN_SOURCE_MODELS.map(buildOpenSourceModelDef)
  ];
}
var openclaw_default = definePluginEntry({
  id: "snowflake-cortex",
  name: "Snowflake Cortex",
  description: "Snowflake Cortex AI — routes Claude models to Anthropic Messages API " + "and all other models to OpenAI-compatible Chat Completions, both " + "behind PAT authentication.",
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
          promptMessage: "Enter your Snowflake Programmatic Access Token (PAT):"
        })
      ],
      catalog: {
        run: async (ctx) => {
          const resolvedKey = ctx.resolveProviderApiKey("snowflake-cortex").apiKey ?? getApiKey();
          if (!resolvedKey || !getBaseURL())
            return null;
          return {
            provider: {
              baseUrl: `${getBaseURL()}/api/v2/cortex/v1`,
              apiKey: resolvedKey,
              api: "openai-completions",
              authHeader: true,
              models: buildModelCatalog()
            }
          };
        }
      },
      normalizeToolSchemas(ctx) {
        if (!ctx.modelId)
          return ctx.tools;
        if (isClaudeModel(ctx.modelId))
          return ctx.tools;
        if (!modelSupportsTools(ctx.modelId))
          return [];
        return ctx.tools;
      },
      normalizeModelId(ctx) {
        return ctx.modelId.replace(/-1m$/, "") || null;
      },
      wrapStreamFn(ctx) {
        if (!ctx.streamFn)
          return;
        const inner = ctx.streamFn;
        return (model, context, options) => {
          const merged = {
            ...options,
            headers: {
              ...options?.headers,
              "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN"
            }
          };
          return inner(model, context, merged);
        };
      },
      buildReplayPolicy(ctx) {
        if (!ctx.modelId)
          return null;
        if (isClaudeModel(ctx.modelId)) {
          return {
            repairToolUseResultPairing: true,
            allowSyntheticToolResults: true,
            validateAnthropicTurns: true
          };
        }
        return null;
      }
    });
  }
});
export {
  openclaw_default as default
};
