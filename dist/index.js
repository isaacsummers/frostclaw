// index.ts
import {
  definePluginEntry
} from "openclaw/plugin-sdk/plugin-entry";
import { resolveClaudeThinkingProfile } from "openclaw/plugin-sdk/provider-model-shared";
import { createProviderApiKeyAuthMethod } from "openclaw/plugin-sdk/provider-auth-api-key";

// src/transforms.ts
function fixTrailingAssistant(messages) {
  const last = messages[messages.length - 1];
  if (!last || typeof last !== "object")
    return messages;
  if (last.role !== "assistant")
    return messages;
  return messages.slice(0, -1);
}
function fixEmptyTextBlocks(messages) {
  let needsFix = false;
  for (const msg of messages) {
    if (!msg || typeof msg !== "object")
      continue;
    const m = msg;
    if (!Array.isArray(m.content))
      continue;
    if (m.content.length === 0) {
      needsFix = true;
      break;
    }
    for (const block of m.content) {
      if (!block || typeof block !== "object")
        continue;
      const b = block;
      if (b.type === "text" && typeof b.text === "string" && b.text.trim() === "") {
        needsFix = true;
        break;
      }
    }
    if (needsFix)
      break;
  }
  if (!needsFix)
    return messages;
  return messages.map((msg) => {
    if (!msg || typeof msg !== "object")
      return msg;
    const m = msg;
    if (!Array.isArray(m.content))
      return msg;
    if (m.content.length === 0) {
      return { ...m, content: [{ type: "text", text: "​" }] };
    }
    const fixed = m.content.map((block) => {
      if (!block || typeof block !== "object")
        return block;
      const b = block;
      if (b.type !== "text" || typeof b.text !== "string")
        return block;
      if (b.text.trim() === "")
        return { ...b, text: "​" };
      return block;
    });
    return { ...m, content: fixed };
  });
}
function stripEagerInputStreaming(payload) {
  const tools = payload.tools;
  if (!Array.isArray(tools))
    return;
  for (const tool of tools) {
    if (!tool || typeof tool !== "object")
      continue;
    const custom = tool.custom;
    if (!custom || typeof custom !== "object")
      continue;
    const customRec = custom;
    if (!("eager_input_streaming" in customRec))
      continue;
    delete customRec.eager_input_streaming;
    if (Object.keys(customRec).length === 0) {
      delete tool.custom;
    }
  }
}
function levelBudget(thinkingLevel) {
  switch (thinkingLevel) {
    case "minimal":
      return 1024;
    case "low":
      return 4000;
    case "medium":
      return 8000;
    case "high":
    default:
      return 16000;
  }
}
function levelEffort(thinkingLevel) {
  switch (thinkingLevel) {
    case "minimal":
    case "low":
      return "low";
    case "medium":
      return "medium";
    case "high":
    case "adaptive":
    default:
      return "high";
  }
}
function normalizeThinkingBudget(payload, thinkingLevel) {
  const thinking = payload.thinking;
  if (!thinking || typeof thinking !== "object")
    return;
  const t = thinking;
  if (t.type === "disabled")
    return;
  if (t.type === "adaptive") {
    const effort = levelEffort(thinkingLevel);
    const existing = payload.output_config;
    payload.output_config = { ...existing, effort };
    return;
  }
  if (t.type === "enabled") {
    payload.thinking = { type: "enabled", budget_tokens: levelBudget(thinkingLevel) };
  }
}
var MAX_TOKENS_FLOOR_NO_THINKING = 1024;
var MAX_TOKENS_FLOOR_ADAPTIVE = 4096;
var MAX_TOKENS_OUTPUT_HEADROOM = 1024;
function clampMaxTokens(payload) {
  const current = payload.max_tokens;
  if (typeof current !== "number")
    return;
  const thinking = payload.thinking;
  const thinkingType = thinking && typeof thinking === "object" ? thinking.type : undefined;
  let floor = MAX_TOKENS_FLOOR_NO_THINKING;
  if (thinkingType === "enabled") {
    const budget = thinking?.budget_tokens;
    const budgetNum = typeof budget === "number" ? budget : 0;
    floor = budgetNum + MAX_TOKENS_OUTPUT_HEADROOM;
  } else if (thinkingType === "adaptive") {
    floor = MAX_TOKENS_FLOOR_ADAPTIVE;
  }
  if (current >= floor)
    return;
  payload.max_tokens = floor;
}
function isClaudeModel(modelId) {
  return modelId.toLowerCase().startsWith("claude");
}

// src/catalog.ts
function getCatalogBaseURL() {
  return process.env.SNOWFLAKE_PROXY_BASE_URL ?? process.env.SNOWFLAKE_BASE_URL ?? "";
}
var BETA_ALWAYS = [];
function anthropicBetaHeaders() {
  return BETA_ALWAYS.length > 0 ? { "anthropic-beta": BETA_ALWAYS.join(",") } : {};
}
var CLAUDE_MODELS = [
  { id: "claude-4-opus", name: "Claude 4 Opus", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-7", name: "Claude Opus 4.7", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-6", name: "Claude Opus 4.6", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-5", name: "Claude Opus 4.5", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-4-sonnet", name: "Claude 4 Sonnet", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5-long-context", name: "Claude Sonnet 4.5 (Long Context)", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-haiku-4-5", name: "Claude Haiku 4.5", reasoning: false, contextWindow: 200000, maxTokens: 16384, input: ["text", "image"] },
  { id: "claude-3-7-sonnet", name: "Claude 3.7 Sonnet", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] }
];
var OPENAI_MODELS = [
  { id: "openai-gpt-5.5", name: "GPT-5.5", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5.5-long-context", name: "GPT-5.5 (Long Context)", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5.4", name: "GPT-5.4", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5.4-long-context", name: "GPT-5.4 (Long Context)", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5.2", name: "GPT-5.2", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5.1", name: "GPT-5.1", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5", name: "GPT-5", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5-mini", name: "GPT-5 Mini", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5-nano", name: "GPT-5 Nano", reasoning: false, contextWindow: 128000, maxTokens: 16384, input: ["text", "image"] },
  { id: "openai-gpt-4.1", name: "GPT-4.1", reasoning: false, contextWindow: 1047576, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-o4-mini", name: "o4-mini", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] }
];
var OPEN_SOURCE_MODELS = [
  { id: "deepseek-r1", name: "DeepSeek R1", reasoning: false, contextWindow: 64000, maxTokens: 16384, input: ["text"] },
  { id: "llama3.1-405b", name: "Llama 3.1 405B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama3.1-70b", name: "Llama 3.1 70B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama3.1-8b", name: "Llama 3.1 8B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama3.2-1b", name: "Llama 3.2 1B", reasoning: false, contextWindow: 128000, maxTokens: 16384, input: ["text"] },
  { id: "llama3.2-3b", name: "Llama 3.2 3B", reasoning: false, contextWindow: 128000, maxTokens: 16384, input: ["text"] },
  { id: "llama3.3-70b", name: "Llama 3.3 70B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama4-maverick", name: "Llama 4 Maverick", reasoning: false, contextWindow: 1047576, maxTokens: 32768, input: ["text"] },
  { id: "mistral-large", name: "Mistral Large", reasoning: false, contextWindow: 32000, maxTokens: 8192, input: ["text"] },
  { id: "mistral-large2", name: "Mistral Large 2", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "mistral-7b", name: "Mistral 7B", reasoning: false, contextWindow: 32000, maxTokens: 8192, input: ["text"] },
  { id: "snowflake-llama-3.3-70b", name: "Snowflake Llama 3.3 70B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] }
];
var COST_CLAUDE_4_OPUS = { input: 0.000015, output: 0.000075, cacheRead: 0.0000015, cacheWrite: 0.00001875 };
var COST_OPUS = { input: 0.0000055, output: 0.0000275, cacheRead: 0.00000055, cacheWrite: 0.000006875 };
var COST_SONNET = { input: 0.0000033, output: 0.0000165, cacheRead: 0.00000033, cacheWrite: 0.000004125 };
var COST_SONNET_LONG = { input: 0.0000066, output: 0.00002475, cacheRead: 0.00000066, cacheWrite: 0.00000825 };
var COST_HAIKU = { input: 0.0000011, output: 0.0000055, cacheRead: 0.00000011, cacheWrite: 0.000001375 };
var COST_CLAUDE_37 = { input: 0.000003, output: 0.000015, cacheRead: 0.0000003, cacheWrite: 0.00000375 };
var COST_CLAUDE_4S = { input: 0.000003, output: 0.000015, cacheRead: 0.0000003, cacheWrite: 0.00000375 };
var COST_GPT55 = { input: 0.0000055, output: 0.000033, cacheRead: 0.00000055, cacheWrite: 0 };
var COST_GPT55_LONG = { input: 0.000011, output: 0.0000495, cacheRead: 0.0000011, cacheWrite: 0 };
var COST_GPT54 = { input: 0.00000275, output: 0.0000165, cacheRead: 0.00000028, cacheWrite: 0 };
var COST_GPT54_LONG = { input: 0.0000055, output: 0.00002475, cacheRead: 0.00000055, cacheWrite: 0 };
var COST_GPT52 = { input: 0.00000193, output: 0.0000154, cacheRead: 0.00000019, cacheWrite: 0 };
var COST_GPT51 = { input: 0.00000138, output: 0.000011, cacheRead: 0.00000014, cacheWrite: 0 };
var COST_GPT5_MINI = { input: 0.00000028, output: 0.0000022, cacheRead: 0.000000028, cacheWrite: 0 };
var COST_GPT5_NANO = { input: 0.00000006, output: 0.00000044, cacheRead: 0.000000006, cacheWrite: 0 };
var COST_GPT41 = { input: 0.0000022, output: 0.0000088, cacheRead: 0.00000055, cacheWrite: 0 };
var COST_O4_MINI = { input: 0.0000011, output: 0.0000044, cacheRead: 0.00000028, cacheWrite: 0 };
var COST_DEEPSEEK_R1 = { input: 0.00000135, output: 0.0000054, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_405B = { input: 0.0000024, output: 0.0000024, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_70B = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_8B = { input: 0.00000022, output: 0.00000022, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_1B = { input: 0.0000001, output: 0.0000001, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_3B = { input: 0.00000015, output: 0.00000015, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA4_MAV = { input: 0.00000024, output: 0.00000097, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_LG = { input: 0.000004, output: 0.000012, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_LG2 = { input: 0.000002, output: 0.000006, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_7B = { input: 0.00000015, output: 0.0000002, cacheRead: 0, cacheWrite: 0 };
function claudeCost(id) {
  if (id === "claude-4-opus")
    return COST_CLAUDE_4_OPUS;
  if (id.startsWith("claude-opus"))
    return COST_OPUS;
  if (id.endsWith("-long-context"))
    return COST_SONNET_LONG;
  if (id === "claude-4-sonnet")
    return COST_CLAUDE_4S;
  if (id === "claude-3-7-sonnet")
    return COST_CLAUDE_37;
  if (id.startsWith("claude-sonnet"))
    return COST_SONNET;
  if (id.startsWith("claude-haiku"))
    return COST_HAIKU;
  return COST_OPUS;
}
function buildClaudeModelDef(spec, baseURL) {
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
    compat: { supportsTools: true, supportsEagerToolInputStreaming: false }
  };
}
function openaiCost(id) {
  if (id === "openai-gpt-5.5-long-context")
    return COST_GPT55_LONG;
  if (id === "openai-gpt-5.5")
    return COST_GPT55;
  if (id === "openai-gpt-5.4-long-context")
    return COST_GPT54_LONG;
  if (id === "openai-gpt-5.4")
    return COST_GPT54;
  if (id === "openai-gpt-5.2")
    return COST_GPT52;
  if (id === "openai-gpt-5.1")
    return COST_GPT51;
  if (id === "openai-gpt-5")
    return COST_GPT51;
  if (id === "openai-gpt-5-mini")
    return COST_GPT5_MINI;
  if (id === "openai-gpt-5-nano")
    return COST_GPT5_NANO;
  if (id.startsWith("openai-gpt-4"))
    return COST_GPT41;
  if (id === "openai-o4-mini")
    return COST_O4_MINI;
  return COST_GPT51;
}
function buildOpenAIModelDef(spec, baseURL) {
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
      supportsUsageInStreaming: true
    }
  };
}
function openSourceCost(id) {
  if (id === "deepseek-r1")
    return COST_DEEPSEEK_R1;
  if (id === "llama3.1-405b")
    return COST_LLAMA_405B;
  if (id === "llama3.1-70b")
    return COST_LLAMA_70B;
  if (id === "llama3.1-8b")
    return COST_LLAMA_8B;
  if (id === "llama3.2-1b")
    return COST_LLAMA_1B;
  if (id === "llama3.2-3b")
    return COST_LLAMA_3B;
  if (id === "llama3.3-70b")
    return COST_LLAMA_70B;
  if (id === "llama4-maverick")
    return COST_LLAMA4_MAV;
  if (id === "mistral-large")
    return COST_MISTRAL_LG;
  if (id === "mistral-large2")
    return COST_MISTRAL_LG2;
  if (id === "mistral-7b")
    return COST_MISTRAL_7B;
  if (id === "snowflake-llama-3.3-70b")
    return COST_LLAMA_70B;
  return COST_LLAMA_70B;
}
function buildOpenSourceModelDef(spec, baseURL) {
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
      supportsUsageInStreaming: true
    }
  };
}
var CATALOG_CACHE;
var CATALOG_INDEX;
function buildModelCatalog() {
  const baseURL = getCatalogBaseURL();
  if (CATALOG_CACHE && CATALOG_CACHE.baseURL === baseURL) {
    return CATALOG_CACHE.catalog;
  }
  const catalog = [
    ...CLAUDE_MODELS.map((s) => buildClaudeModelDef(s, baseURL)),
    ...OPENAI_MODELS.map((s) => buildOpenAIModelDef(s, baseURL)),
    ...OPEN_SOURCE_MODELS.map((s) => buildOpenSourceModelDef(s, baseURL))
  ];
  CATALOG_CACHE = { baseURL, catalog };
  CATALOG_INDEX = new Map(catalog.map((m) => [m.id, m]));
  return catalog;
}
function findCatalogEntry(modelId) {
  const bareId = modelId.replace(/^snowflake-cortex\//, "");
  if (!CATALOG_INDEX)
    buildModelCatalog();
  return CATALOG_INDEX?.get(bareId);
}

// index.ts
function getApiKey() {
  return process.env.SNOWFLAKE_CORTEX_API_KEY ?? process.env.SNOWFLAKE_PAT ?? "";
}
function getBaseURL() {
  return process.env.SNOWFLAKE_PROXY_BASE_URL ?? process.env.SNOWFLAKE_BASE_URL ?? "";
}
var DEBUG_ENABLED = (() => {
  const v = process.env.FROSTCLAW_DEBUG;
  if (!v)
    return false;
  const s = v.toLowerCase();
  return s !== "0" && s !== "false" && s !== "off" && s !== "";
})();
function log(event, data) {
  if (!DEBUG_ENABLED)
    return;
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  console.error(line);
}
function logError(event, data) {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  console.error(line);
}
var BETA_THINKING = [
  "interleaved-thinking-2025-05-14"
];
function modelSupportsTools(modelId) {
  return modelId.toLowerCase().startsWith("openai-");
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
    try {
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
            promptMessage: "Enter your Snowflake Programmatic Access Token (PAT):"
          })
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
                baseURL: baseURL || "(not set)"
              });
              if (!resolvedKey || !baseURL) {
                log("catalog.run returning null — missing config", {
                  resolvedKey: !!resolvedKey,
                  baseURL: !!baseURL
                });
                return null;
              }
              const models = buildModelCatalog();
              log("catalog.run returning catalog", { modelCount: models.length });
              return {
                provider: {
                  baseUrl: baseURL,
                  apiKey: resolvedKey,
                  api: "openai-completions",
                  authHeader: true,
                  models
                }
              };
            } catch (err) {
              logError("catalog.run ERROR", {
                error: String(err),
                stack: err instanceof Error ? err.stack : undefined
              });
              throw err;
            }
          }
        },
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
              hasCost: !!catalogEntry.cost
            });
            return catalogEntry;
          }
          const claude = isClaudeModel(modelId);
          const api2 = claude ? "anthropic-messages" : "openai-completions";
          const input = claude ? ["text", "image"] : ["text"];
          const baseUrl = getBaseURL();
          log("resolveDynamicModel (unknown id, minimal stub)", { modelId, api: api2, input, baseUrl });
          return { id: modelId, name: modelId, api: api2, input, baseUrl };
        },
        normalizeToolSchemas(ctx) {
          if (!ctx.modelId)
            return ctx.tools;
          if (isClaudeModel(ctx.modelId)) {
            return ctx.tools.map((tool) => {
              const custom = tool.custom;
              if (!custom || !("eager_input_streaming" in custom))
                return tool;
              const { eager_input_streaming: _dropped, ...rest } = custom;
              if (Object.keys(rest).length > 0)
                return { ...tool, custom: rest };
              const { custom: _c, ...toolWithoutCustom } = tool;
              return toolWithoutCustom;
            });
          }
          if (!modelSupportsTools(ctx.modelId))
            return [];
          return ctx.tools;
        },
        wrapStreamFn(ctx) {
          log("wrapStreamFn", {
            modelId: ctx.modelId,
            thinkingLevel: ctx.thinkingLevel,
            thinkingActive: ctx.thinkingLevel !== undefined && ctx.thinkingLevel !== "off",
            hasStreamFn: !!ctx.streamFn
          });
          if (!ctx.streamFn)
            return;
          const inner = ctx.streamFn;
          const thinkingActive = ctx.thinkingLevel !== undefined && ctx.thinkingLevel !== "off";
          const thinkingLevel = ctx.thinkingLevel;
          return (model, context, options) => {
            try {
              const modelObj = model;
              if (modelObj && !Array.isArray(modelObj.input)) {
                const id = String(modelObj.id ?? "");
                const inferred = isClaudeModel(id) ? ["text", "image"] : ["text"];
                log("wrapStreamFn.inner: patching missing input", {
                  modelId: id,
                  inferred,
                  priorInput: modelObj.input,
                  keys: Object.keys(modelObj)
                });
                modelObj.input = inferred;
              }
              if (modelObj && typeof modelObj.baseUrl !== "string") {
                const fallbackBaseUrl = getBaseURL();
                log("wrapStreamFn.inner: patching missing baseUrl", {
                  modelId: String(modelObj.id ?? ""),
                  fallbackBaseUrl,
                  priorBaseUrl: modelObj.baseUrl
                });
                modelObj.baseUrl = fallbackBaseUrl;
              }
              log("wrapStreamFn.inner", {
                modelId: modelObj?.id,
                modelApi: modelObj?.api,
                modelInput: modelObj?.input,
                modelBaseUrl: modelObj?.baseUrl,
                modelKeys: modelObj ? Object.keys(modelObj) : undefined,
                hasContext: !!context,
                hasOptions: !!options,
                messageCount: Array.isArray(context?.messages) ? context.messages.length : undefined
              });
              const originalOnPayload = options?.onPayload;
              const catalogBeta = model?.headers?.["anthropic-beta"] ?? "";
              const betaFlags = catalogBeta ? [catalogBeta] : [];
              const modelSupportsReasoning = modelObj?.reasoning === true;
              if (thinkingActive && modelSupportsReasoning) {
                betaFlags.push(BETA_THINKING.join(","));
              }
              const modelId = String(model?.id ?? "");
              const isClaudeRoute = isClaudeModel(modelId);
              const optionsApiKey = options?.apiKey;
              const bearerKey = typeof optionsApiKey === "string" && optionsApiKey.length > 0 ? optionsApiKey : getApiKey();
              const authHeader = isClaudeRoute && bearerKey ? { Authorization: `Bearer ${bearerKey}` } : {};
              const merged = {
                ...options,
                headers: {
                  ...options?.headers,
                  "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
                  ...authHeader,
                  ...betaFlags.length > 0 ? { "anthropic-beta": betaFlags.join(",") } : {}
                },
                onPayload: (payload, payloadModel) => {
                  const payloadModelObj = payloadModel;
                  log("onPayload", {
                    payloadType: typeof payload,
                    isObject: payload !== null && typeof payload === "object",
                    payloadModelId: payloadModelObj?.id,
                    isClaudeModelResult: payload && typeof payload === "object" ? isClaudeModel(String(model?.id ?? "")) : false
                  });
                  if (payload && typeof payload === "object" && isClaudeModel(String(model?.id ?? ""))) {
                    const record = payload;
                    if (Array.isArray(record.messages)) {
                      record.messages = fixTrailingAssistant(record.messages);
                      record.messages = fixEmptyTextBlocks(record.messages);
                    }
                    stripEagerInputStreaming(record);
                    normalizeThinkingBudget(record, thinkingLevel);
                    clampMaxTokens(record);
                  }
                  const chained = originalOnPayload?.(payload, payloadModel);
                  return chained !== undefined ? chained : payload;
                }
              };
              const streamResult = inner(model, context, merged);
              if (streamResult && typeof streamResult.then === "function") {
                const EMPTY_STOP_MAX_RETRIES = 2;
                return (async () => {
                  let currentResult = streamResult;
                  for (let attempt = 0;attempt <= EMPTY_STOP_MAX_RETRIES; attempt++) {
                    let value;
                    try {
                      value = await currentResult;
                    } catch (err) {
                      logError("wrapStreamFn.promise REJECTED", {
                        error: String(err),
                        stack: err instanceof Error ? err.stack : undefined
                      });
                      throw err;
                    }
                    const valueObj = value;
                    if (valueObj && typeof valueObj === "object" && (valueObj.errorMessage || valueObj.stopReason === "error")) {
                      log("wrapStreamFn.promise resolved with error", {
                        errorMessage: valueObj.errorMessage,
                        stopReason: valueObj.stopReason
                      });
                    }
                    const isThinkingOnlyContent = Array.isArray(valueObj?.content) && valueObj.content.length > 0 && valueObj.content.every((blk) => blk?.type === "thinking");
                    const isEmptyOrThinkingOnlyStop = attempt < EMPTY_STOP_MAX_RETRIES && valueObj && typeof valueObj === "object" && valueObj.stopReason === "stop" && (Array.isArray(valueObj.content) && valueObj.content.length === 0 || isThinkingOnlyContent);
                    if (isEmptyOrThinkingOnlyStop) {
                      log("wrapStreamFn.promise: empty/thinking-only stop from Snowflake, retrying", {
                        modelId,
                        attempt,
                        isThinkingOnly: isThinkingOnlyContent,
                        contentLength: Array.isArray(valueObj?.content) ? valueObj.content.length : 0
                      });
                      const retryResult = inner(model, context, merged);
                      if (retryResult && typeof retryResult.then === "function") {
                        currentResult = retryResult;
                        continue;
                      }
                      return retryResult;
                    }
                    return value;
                  }
                })();
              }
              return streamResult;
            } catch (err) {
              logError("wrapStreamFn.inner ERROR", {
                error: String(err),
                stack: err instanceof Error ? err.stack : undefined
              });
              throw err;
            }
          };
        },
        resolveThinkingProfile(ctx) {
          if (!ctx.modelId)
            return null;
          if (!isClaudeModel(ctx.modelId))
            return null;
          const bareId = ctx.modelId.replace(/^snowflake-cortex\//, "");
          return resolveClaudeThinkingProfile(bareId) ?? null;
        },
        buildReplayPolicy(ctx) {
          if (!ctx.modelId)
            return null;
          if (isClaudeModel(ctx.modelId)) {
            return {
              sanitizeToolCallIds: true,
              toolCallIdMode: "strict",
              preserveNativeAnthropicToolUseIds: true,
              repairToolUseResultPairing: true,
              allowSyntheticToolResults: true,
              validateAnthropicTurns: true,
              preserveSignatures: true
            };
          }
          return null;
        }
      });
    } catch (err) {
      logError("register ERROR", {
        error: String(err),
        stack: err instanceof Error ? err.stack : undefined
      });
      throw err;
    }
  }
});
export {
  frostclaw_default as default
};
