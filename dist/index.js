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
  "tool-examples-2025-10-29",
  "extended-cache-ttl-2025-02-19"
].join(",");
var ANTHROPIC_BETA_1M = [
  "context-1m-2025-08-07",
  "interleaved-thinking-2025-05-14",
  "output-128k-2025-02-19",
  "effort-2025-11-24",
  "token-efficient-tools-2025-02-19",
  "tool-examples-2025-10-29",
  "extended-cache-ttl-2025-02-19"
].join(",");
function isClaudeModel(modelId) {
  return modelId.toLowerCase().startsWith("claude");
}
function modelSupportsTools(modelId) {
  return modelId.toLowerCase().startsWith("openai-");
}
function promoteEphemeralCacheToLongTtl(payload) {
  const promote = (block) => {
    if (!block || typeof block !== "object")
      return;
    const record = block;
    const cc = record.cache_control;
    if (cc && typeof cc === "object" && cc.type === "ephemeral" && cc.ttl === undefined) {
      cc.ttl = "1h";
    }
  };
  const system = payload.system;
  if (Array.isArray(system)) {
    for (const block of system)
      promote(block);
  }
  const messages = payload.messages;
  if (Array.isArray(messages)) {
    for (const msg of messages) {
      if (!msg || typeof msg !== "object")
        continue;
      const content = msg.content;
      if (Array.isArray(content)) {
        for (const block of content)
          promote(block);
      }
    }
  }
  const tools = payload.tools;
  if (Array.isArray(tools)) {
    for (const tool of tools)
      promote(tool);
  }
}
var CLAUDE_MODELS = [
  { id: "claude-opus-4-7", name: "Claude Opus 4.7", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-6", name: "Claude Opus 4.6", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-5", name: "Claude Opus 4.5", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-7-1m", name: "Claude Opus 4.7 (1M)", reasoning: true, contextWindow: 1e6, maxTokens: 128000, input: ["text", "image"], extendedContext: true },
  { id: "claude-opus-4-6-1m", name: "Claude Opus 4.6 (1M)", reasoning: true, contextWindow: 1e6, maxTokens: 128000, input: ["text", "image"], extendedContext: true },
  { id: "claude-sonnet-4-6-1m", name: "Claude Sonnet 4.6 (1M)", reasoning: true, contextWindow: 1e6, maxTokens: 128000, input: ["text", "image"], extendedContext: true }
];
var OPENAI_MODELS = [
  { id: "openai-gpt-5.4", name: "GPT-5.4", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5.2", name: "GPT-5.2", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5.1", name: "GPT-5.1", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5", name: "GPT-5", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5-mini", name: "GPT-5 Mini", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5-nano", name: "GPT-5 Nano", reasoning: false, contextWindow: 128000, maxTokens: 16384, input: ["text", "image"] },
  { id: "openai-gpt-oss-120b", name: "GPT OSS 120B", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "openai-gpt-4.1", name: "GPT-4.1", reasoning: false, contextWindow: 1047576, maxTokens: 32768, input: ["text", "image"] }
];
var OPEN_SOURCE_MODELS = [
  { id: "llama4-maverick", name: "Llama 4 Maverick", reasoning: false, contextWindow: 1048576, maxTokens: 16384, input: ["text"] },
  { id: "llama3.1-70b", name: "Llama 3.1 70B", reasoning: false, contextWindow: 128000, maxTokens: 4096, input: ["text"] },
  { id: "llama3.1-8b", name: "Llama 3.1 8B", reasoning: false, contextWindow: 128000, maxTokens: 4096, input: ["text"] },
  { id: "llama3.1-405b", name: "Llama 3.1 405B", reasoning: false, contextWindow: 128000, maxTokens: 4096, input: ["text"] },
  { id: "mistral-large", name: "Mistral Large", reasoning: false, contextWindow: 32000, maxTokens: 8192, input: ["text"] },
  { id: "mistral-large2", name: "Mistral Large 2", reasoning: false, contextWindow: 128000, maxTokens: 8192, input: ["text"] },
  { id: "deepseek-r1", name: "DeepSeek R1", reasoning: true, contextWindow: 64000, maxTokens: 8192, input: ["text"] },
  { id: "snowflake-arctic", name: "Snowflake Arctic", reasoning: false, contextWindow: 4096, maxTokens: 4096, input: ["text"] },
  { id: "snowflake-llama-3.3-70b", name: "Snowflake Llama 3.3 70B", reasoning: false, contextWindow: 128000, maxTokens: 4096, input: ["text"] }
];
var COST_OPUS = { input: 0.0000055, output: 0.0000275, cacheRead: 0.00000055, cacheWrite: 0 };
var COST_SONNET = { input: 0.0000033, output: 0.0000165, cacheRead: 0.00000033, cacheWrite: 0 };
var COST_HAIKU = { input: 0.0000011, output: 0.0000055, cacheRead: 0.00000011, cacheWrite: 0 };
var COST_GPT54 = { input: 0.0000025, output: 0.000015, cacheRead: 0.00000025, cacheWrite: 0 };
var COST_GPT52 = { input: 0.00000175, output: 0.000014, cacheRead: 0.000000175, cacheWrite: 0 };
var COST_GPT5 = { input: 0.00000125, output: 0.00001, cacheRead: 0.000000125, cacheWrite: 0 };
var COST_GPT41 = { input: 0.000002, output: 0.000008, cacheRead: 0.0000002, cacheWrite: 0 };
var COST_LLAMA_405B = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_70B = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_8B = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA4_MAV = { input: 0.00000024, output: 0.00000097, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_L = { input: 0.000002, output: 0.000006, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_L2 = { input: 0.000002, output: 0.000006, cacheRead: 0, cacheWrite: 0 };
var COST_DEEPSEEK = { input: 0.00000135, output: 0.0000054, cacheRead: 0, cacheWrite: 0 };
var COST_ARCTIC = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
function anthropicBetaHeaders(extendedContext = false) {
  return { "anthropic-beta": extendedContext ? ANTHROPIC_BETA_1M : ANTHROPIC_BETA_DEFAULT };
}
function claudeCost(id) {
  if (id.startsWith("claude-opus"))
    return COST_OPUS;
  if (id.startsWith("claude-sonnet"))
    return COST_SONNET;
  if (id.startsWith("claude-haiku"))
    return COST_HAIKU;
  return COST_OPUS;
}
function buildClaudeModelDef(spec) {
  return {
    id: spec.id,
    name: spec.name,
    api: "anthropic-messages",
    reasoning: spec.reasoning,
    input: spec.input,
    cost: claudeCost(spec.id),
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    headers: anthropicBetaHeaders(spec.extendedContext),
    compat: { supportsTools: true }
  };
}
function openaiCost(id) {
  if (id === "openai-gpt-5.4")
    return COST_GPT54;
  if (id === "openai-gpt-5.2")
    return COST_GPT52;
  if (id.startsWith("openai-gpt-4"))
    return COST_GPT41;
  return COST_GPT5;
}
function buildOpenAIModelDef(spec) {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions",
    reasoning: spec.reasoning,
    input: spec.input,
    cost: openaiCost(spec.id),
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: true,
      maxTokensField: "max_completion_tokens",
      supportsUsageInStreaming: true
    }
  };
}
function openSourceCost(id) {
  if (id === "llama4-maverick")
    return COST_LLAMA4_MAV;
  if (id === "llama3.1-405b")
    return COST_LLAMA_405B;
  if (id === "llama3.1-70b" || id === "llama3.3-70b")
    return COST_LLAMA_70B;
  if (id === "llama3.1-8b")
    return COST_LLAMA_8B;
  if (id === "mistral-large")
    return COST_MISTRAL_L;
  if (id === "mistral-large2")
    return COST_MISTRAL_L2;
  if (id === "deepseek-r1")
    return COST_DEEPSEEK;
  if (id === "snowflake-arctic")
    return COST_ARCTIC;
  if (id === "snowflake-llama-3.3-70b")
    return COST_LLAMA_70B;
  return COST_LLAMA_70B;
}
function buildOpenSourceModelDef(spec) {
  return {
    id: spec.id,
    name: spec.name,
    api: "openai-completions",
    reasoning: spec.reasoning,
    input: spec.input,
    cost: openSourceCost(spec.id),
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    compat: {
      supportsTools: false,
      maxTokensField: "max_completion_tokens",
      supportsUsageInStreaming: true
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
var DEFAULT_SNOWFLAKE_EMBED_MODEL = "snowflake-arctic-embed-m-v1.5";
async function snowflakeEmbed(texts, model) {
  const apiKey = getApiKey();
  const baseUrl = getBaseURL();
  if (!apiKey || !baseUrl) {
    throw new Error("[snowflake-cortex] Missing SNOWFLAKE_BASE_URL or SNOWFLAKE_CORTEX_API_KEY");
  }
  const url = `${baseUrl}/api/v2/cortex/inference:embed`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify({ text: texts, model })
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`[snowflake-cortex] Embed request failed (${res.status}): ${body}`);
  }
  const json = await res.json();
  return json.data.sort((a, b) => a.index - b.index).map(({ embedding }) => Array.isArray(embedding[0]) ? embedding[0] : embedding);
}
var snowflakeCortexEmbeddingAdapter = {
  id: "snowflake-cortex",
  defaultModel: DEFAULT_SNOWFLAKE_EMBED_MODEL,
  transport: "remote",
  autoSelectPriority: -1,
  async create(options) {
    const model = options.model || DEFAULT_SNOWFLAKE_EMBED_MODEL;
    if (!getApiKey() || !getBaseURL()) {
      return { provider: null };
    }
    return {
      provider: {
        id: "snowflake-cortex",
        model,
        maxInputTokens: 4096,
        embedQuery: (text) => snowflakeEmbed([text], model).then((v) => v[0]),
        embedBatch: (texts) => snowflakeEmbed(texts, model)
      }
    };
  }
};
var frostclaw_default = definePluginEntry({
  id: "snowflake-cortex",
  name: "Snowflake Cortex",
  description: "Snowflake Cortex AI — routes Claude models to Anthropic Messages API " + "and all other models to OpenAI-compatible Chat Completions, both " + "behind PAT authentication.",
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
          const originalOnPayload = options?.onPayload;
          const merged = {
            ...options,
            headers: {
              ...options?.headers,
              "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN"
            },
            onPayload: (payload, payloadModel) => {
              if (payload && typeof payload === "object" && isClaudeModel(String(model?.id ?? ""))) {
                promoteEphemeralCacheToLongTtl(payload);
              }
              return originalOnPayload?.(payload, payloadModel);
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
  frostclaw_default as default
};
