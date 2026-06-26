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
function normalizeThinkingBudget(payload, thinkingLevel, adaptiveOnly = false) {
  const thinking = payload.thinking;
  if (!thinking || typeof thinking !== "object")
    return;
  const t = thinking;
  if (t.type === "disabled")
    return;
  if (t.type === "adaptive" || adaptiveOnly && t.type === "enabled") {
    const effort = levelEffort(thinkingLevel);
    const existing = payload.output_config;
    payload.thinking = { type: "adaptive" };
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
function stripResponseFormat(payload) {
  if ("response_format" in payload) {
    delete payload.response_format;
  }
}
function stripDocumentBlocks(messages) {
  let needsFix = false;
  outer:
    for (const msg of messages) {
      if (!msg || typeof msg !== "object")
        continue;
      const m = msg;
      if (!Array.isArray(m.content))
        continue;
      for (const block of m.content) {
        if (!block || typeof block !== "object")
          continue;
        if (block.type === "document") {
          needsFix = true;
          break outer;
        }
      }
    }
  if (!needsFix)
    return messages;
  return messages.map((msg) => {
    if (!msg || typeof msg !== "object")
      return msg;
    const m = msg;
    if (!Array.isArray(m.content))
      return msg;
    const hasDoc = m.content.some((b) => b && typeof b === "object" && b.type === "document");
    if (!hasDoc)
      return msg;
    const fixed = m.content.map((block) => {
      if (!block || typeof block !== "object")
        return block;
      const b = block;
      if (b.type !== "document")
        return block;
      const source = b.source;
      const mediaType = typeof source?.media_type === "string" ? source.media_type : "unknown";
      const title = typeof b.title === "string" ? ` ("${b.title}")` : "";
      return {
        type: "text",
        text: `[PDF/document block stripped${title} — Snowflake Cortex does not support native document blocks; media_type=${mediaType}]`
      };
    });
    return { ...m, content: fixed };
  });
}

// src/catalog.ts
function getCatalogBaseURL() {
  const raw = process.env.SNOWFLAKE_PROXY_BASE_URL ?? process.env.SNOWFLAKE_BASE_URL ?? "";
  return raw ? `${raw.replace(/\/$/, "")}/api/v2/cortex` : "";
}
var BETA_ALWAYS = [];
function anthropicBetaHeaders() {
  return BETA_ALWAYS.length > 0 ? { "anthropic-beta": BETA_ALWAYS.join(",") } : {};
}
var CLAUDE_MODELS = [
  { id: "claude-fable-5", name: "Claude Fable 5", reasoning: true, adaptiveOnly: true, contextWindow: 1e6, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-8", name: "Claude Opus 4.8", reasoning: true, adaptiveOnly: true, contextWindow: 1e6, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-7", name: "Claude Opus 4.7", reasoning: true, adaptiveOnly: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-6", name: "Claude Opus 4.6", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-opus-4-5", name: "Claude Opus 4.5", reasoning: true, contextWindow: 200000, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", reasoning: true, contextWindow: 1e6, maxTokens: 64000, input: ["text", "image"] },
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", reasoning: true, contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  { id: "claude-haiku-4-5", name: "Claude Haiku 4.5", reasoning: false, contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] }
];
var OPENAI_MODELS = [
  { id: "openai-gpt-5.4", name: "GPT-5.4", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5.2", name: "GPT-5.2", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5.1", name: "GPT-5.1", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5", name: "GPT-5", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5-mini", name: "GPT-5 Mini", reasoning: true, contextWindow: 128000, maxTokens: 32768, input: ["text", "image"] },
  { id: "openai-gpt-5-nano", name: "GPT-5 Nano", reasoning: false, contextWindow: 128000, maxTokens: 16384, input: ["text", "image"] },
  { id: "openai-gpt-4.1", name: "GPT-4.1", reasoning: false, contextWindow: 1047576, maxTokens: 32768, input: ["text", "image"] }
];
var OPEN_SOURCE_MODELS = [
  { id: "deepseek-r1", name: "DeepSeek R1", reasoning: false, contextWindow: 64000, maxTokens: 16384, input: ["text"] },
  { id: "llama3.1-405b", name: "Llama 3.1 405B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama3.1-70b", name: "Llama 3.1 70B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama3.1-8b", name: "Llama 3.1 8B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama3.3-70b", name: "Llama 3.3 70B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama4-maverick", name: "Llama 4 Maverick", reasoning: false, contextWindow: 1047576, maxTokens: 32768, input: ["text"] },
  { id: "mistral-large", name: "Mistral Large", reasoning: false, contextWindow: 32000, maxTokens: 8192, input: ["text"] },
  { id: "mistral-large2", name: "Mistral Large 2", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "mistral-7b", name: "Mistral 7B", reasoning: false, contextWindow: 32000, maxTokens: 8192, input: ["text"] },
  { id: "snowflake-llama-3.3-70b", name: "Snowflake Llama 3.3 70B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] }
];
var COST_FABLE = { input: 0.00001, output: 0.00005, cacheRead: 0.000001, cacheWrite: 0.0000125 };
var COST_OPUS = { input: 0.000005, output: 0.000025, cacheRead: 0.0000005, cacheWrite: 0.00000625 };
var COST_SONNET = { input: 0.000003, output: 0.000015, cacheRead: 0.0000003, cacheWrite: 0.00000375 };
var COST_SONNET_LONG = { input: 0.000006, output: 0.00003, cacheRead: 0.0000006, cacheWrite: 0.0000075 };
var COST_HAIKU = { input: 0.000001, output: 0.000005, cacheRead: 0.0000001, cacheWrite: 0.00000125 };
var COST_GPT54 = { input: 0.00000275, output: 0.0000165, cacheRead: 0.00000028, cacheWrite: 0 };
var COST_GPT52 = { input: 0.00000193, output: 0.0000154, cacheRead: 0.00000019, cacheWrite: 0 };
var COST_GPT51 = { input: 0.00000138, output: 0.000011, cacheRead: 0.00000014, cacheWrite: 0 };
var COST_GPT5_MINI = { input: 0.00000028, output: 0.0000022, cacheRead: 0.000000028, cacheWrite: 0 };
var COST_GPT5_NANO = { input: 0.00000006, output: 0.00000044, cacheRead: 0.000000006, cacheWrite: 0 };
var COST_GPT41 = { input: 0.0000022, output: 0.0000088, cacheRead: 0.00000055, cacheWrite: 0 };
var COST_DEEPSEEK_R1 = { input: 0.00000135, output: 0.0000054, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_405B = { input: 0.0000024, output: 0.0000024, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_70B = { input: 0.00000072, output: 0.00000072, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_8B = { input: 0.00000022, output: 0.00000022, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA4_MAV = { input: 0.00000024, output: 0.00000097, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_LG = { input: 0.000004, output: 0.000012, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_LG2 = { input: 0.000002, output: 0.000006, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_7B = { input: 0.00000015, output: 0.0000002, cacheRead: 0, cacheWrite: 0 };
function claudeCost(id) {
  if (id === "claude-fable-5")
    return COST_FABLE;
  if (id.startsWith("claude-opus"))
    return COST_OPUS;
  if (id.endsWith("-long-context"))
    return COST_SONNET_LONG;
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
    ...spec.adaptiveOnly ? { adaptiveOnly: true } : {},
    input: spec.input,
    cost: claudeCost(spec.id),
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
    headers: anthropicBetaHeaders(),
    compat: { supportsTools: true, supportsEagerToolInputStreaming: false }
  };
}
function openaiCost(id) {
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
function isAdaptiveOnly(modelId) {
  return findCatalogEntry(modelId)?.adaptiveOnly === true;
}

// src/event-stream.ts
class EventStream {
  isComplete;
  extractResult;
  queue = [];
  waiting = [];
  done = false;
  finalResultPromise;
  resolveFinalResult;
  constructor(isComplete, extractResult) {
    this.isComplete = isComplete;
    this.extractResult = extractResult;
    this.finalResultPromise = new Promise((resolve) => {
      this.resolveFinalResult = resolve;
    });
  }
  push(event) {
    if (this.done)
      return;
    if (this.isComplete(event)) {
      this.done = true;
      this.resolveFinalResult(this.extractResult(event));
    }
    const waiter = this.waiting.shift();
    if (waiter) {
      waiter({ value: event, done: false });
    } else {
      this.queue.push(event);
    }
  }
  end(result) {
    this.done = true;
    if (result !== undefined) {
      this.resolveFinalResult(result);
    }
    while (this.waiting.length > 0) {
      const waiter = this.waiting.shift();
      waiter({ value: undefined, done: true });
    }
  }
  async* [Symbol.asyncIterator]() {
    while (true) {
      if (this.queue.length > 0) {
        yield this.queue.shift();
      } else if (this.done) {
        return;
      } else {
        const result = await new Promise((resolve) => this.waiting.push(resolve));
        if (result.done)
          return;
        yield result.value;
      }
    }
  }
  result() {
    return this.finalResultPromise;
  }
}

class AssistantMessageEventStream extends EventStream {
  constructor() {
    super((event) => {
      const e = event;
      return e.type === "done" || e.type === "error";
    }, (event) => {
      const e = event;
      if (e.type === "done")
        return e.message;
      else if (e.type === "error")
        return e.error;
      throw new Error("Unexpected event type for final result");
    });
  }
}
function createAssistantMessageEventStream() {
  return new AssistantMessageEventStream;
}

// index.ts
function getApiKey() {
  return process.env.SNOWFLAKE_CORTEX_API_KEY ?? process.env.SNOWFLAKE_PAT ?? "";
}
function getBaseURL() {
  return process.env.SNOWFLAKE_PROXY_BASE_URL ?? process.env.SNOWFLAKE_BASE_URL ?? "";
}
var _pluginLogger = null;
var LOG_LEVEL_ORDER = {
  trace: 0,
  debug: 1,
  info: 2,
  warn: 3,
  error: 4,
  fatal: 5
};
var _debugForceLevel = "info";
function setPluginLogger(logger, cfg) {
  _pluginLogger = logger;
  const fileLevel = cfg?.logging?.level ?? "info";
  const consoleLevel = cfg?.logging?.consoleLevel ?? "info";
  const fileOrder = LOG_LEVEL_ORDER[fileLevel] ?? LOG_LEVEL_ORDER.info;
  const consoleOrder = LOG_LEVEL_ORDER[consoleLevel] ?? LOG_LEVEL_ORDER.info;
  const effectiveOrder = Math.min(fileOrder, consoleOrder);
  if (effectiveOrder <= LOG_LEVEL_ORDER.debug) {
    _debugForceLevel = "debug";
  } else if (effectiveOrder <= LOG_LEVEL_ORDER.info) {
    _debugForceLevel = "info";
  } else if (effectiveOrder <= LOG_LEVEL_ORDER.warn) {
    _debugForceLevel = "warn";
  } else {
    _debugForceLevel = "error";
  }
}
var DEBUG_ENABLED = (() => {
  const v = process.env.FROSTCLAW_DEBUG;
  if (!v)
    return false;
  const s = v.toLowerCase();
  return s !== "0" && s !== "false" && s !== "off" && s !== "";
})();
function log(event, data) {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  if (_pluginLogger) {
    if (DEBUG_ENABLED) {
      const method = _pluginLogger[_debugForceLevel];
      if (typeof method === "function")
        method.call(_pluginLogger, line);
      else
        _pluginLogger.info(line);
    } else {
      _pluginLogger.debug?.(line);
    }
  } else if (DEBUG_ENABLED) {
    console.log(line);
  }
}
function logWarn(event, data) {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  if (_pluginLogger) {
    _pluginLogger.warn(line);
  } else {
    console.warn(line);
  }
}
function logError(event, data) {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  if (_pluginLogger) {
    _pluginLogger.error(line);
  } else {
    console.error(line);
  }
}
function isRequestDebugEnabled() {
  const v = process.env.FROSTCLAW_DEBUG_REQUESTS;
  if (!v)
    return false;
  const s = v.toLowerCase();
  return s === "1" || s === "true";
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
  const rawBaseUrl = process.env.SNOWFLAKE_PROXY_BASE_URL ?? process.env.SNOWFLAKE_BASE_URL ?? "";
  if (!apiKey || !rawBaseUrl) {
    throw new Error("[snowflake-cortex] Missing SNOWFLAKE_BASE_URL or SNOWFLAKE_CORTEX_API_KEY");
  }
  const url = `${rawBaseUrl.replace(/\/$/, "")}/api/v2/cortex/inference:embed`;
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
        embed: (input, _options) => {
          const text = typeof input === "string" ? input : input.text;
          return snowflakeEmbed([text], model).then((v) => v[0]);
        },
        embedBatch: (inputs, _options) => {
          const texts = inputs.map((i) => typeof i === "string" ? i : i.text);
          return snowflakeEmbed(texts, model);
        }
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
      setPluginLogger(api.logger, api.config);
      log("plugin registered");
      const FETCH_INTERCEPT_MARKER = Symbol.for("frostclaw.fetchIntercepted.v2");
      if (!globalThis[FETCH_INTERCEPT_MARKER]) {
        globalThis[FETCH_INTERCEPT_MARKER] = true;
        const originalFetch = globalThis.fetch;
        let _snowflakeDispatcher = undefined;
        try {
          const _nodeModule = process.getBuiltinModule("module");
          const _runtimeRequire = _nodeModule.createRequire(import.meta.url);
          const undici = _runtimeRequire("/home/ubuntu/.npm-global/lib/node_modules/openclaw/node_modules/undici");
          _snowflakeDispatcher = new undici.Agent({
            headersTimeout: 30000,
            bodyTimeout: 0,
            keepAliveTimeout: 90000,
            keepAliveMaxTimeout: 300000
          });
          _pluginLogger?.info("[frostclaw:fetch] undici Agent dispatcher configured (bodyTimeout=0, keepAlive=90s)");
        } catch (e) {
          _pluginLogger?.warn(`[frostclaw:fetch] undici not available, using default dispatcher: ${e}`);
        }
        globalThis.fetch = async function frostclawFetch(input, init) {
          const url = typeof input === "string" ? input : input.url;
          const method = (init?.method ?? "GET").toUpperCase();
          const isSnowflakeInference = method === "POST" && (url.includes("/api/v2/cortex/v1/messages") || url.includes("/api/v2/cortex/v1/chat/completions"));
          if (!isSnowflakeInference) {
            return originalFetch(input, init);
          }
          if (_snowflakeDispatcher) {
            init = { ...init, dispatcher: _snowflakeDispatcher };
          }
          const isAnthropicFormat = url.includes("/api/v2/cortex/v1/messages");
          {
            const rawHeaders = init?.headers;
            const getHeader = (name) => {
              if (!rawHeaders)
                return null;
              if (rawHeaders instanceof Headers)
                return rawHeaders.get(name);
              const rec = rawHeaders;
              const lower = name.toLowerCase();
              for (const key of Object.keys(rec)) {
                if (key.toLowerCase() === lower)
                  return rec[key];
              }
              return null;
            };
            const xApiKey = getHeader("x-api-key");
            const authorization = getHeader("Authorization");
            if (xApiKey && !authorization) {
              const newHeaders = {};
              if (rawHeaders instanceof Headers) {
                rawHeaders.forEach((value, key) => {
                  if (key.toLowerCase() !== "x-api-key")
                    newHeaders[key] = value;
                });
              } else {
                for (const [key, value] of Object.entries(rawHeaders)) {
                  if (key.toLowerCase() !== "x-api-key")
                    newHeaders[key] = value;
                }
              }
              newHeaders["Authorization"] = `Bearer ${xApiKey}`;
              newHeaders["X-Snowflake-Authorization-Token-Type"] = "PROGRAMMATIC_ACCESS_TOKEN";
              init = { ...init, headers: newHeaders };
              _pluginLogger?.info("[frostclaw:fetch] patched x-api-key → Bearer auth for Snowflake Cortex direct call");
            }
          }
          let bodyForRetry = init?.body;
          if (bodyForRetry instanceof ReadableStream) {
            const chunks = [];
            const reader = bodyForRetry.getReader();
            while (true) {
              const { value, done } = await reader.read();
              if (done)
                break;
              if (value)
                chunks.push(value);
            }
            const total = chunks.reduce((a, c) => a + c.length, 0);
            const merged = new Uint8Array(total);
            let off = 0;
            for (const c of chunks) {
              merged.set(c, off);
              off += c.length;
            }
            bodyForRetry = merged;
            _pluginLogger?.info("[frostclaw:fetch] body was ReadableStream — buffered to Uint8Array for retry safety");
          }
          const FETCH_MAX_RETRIES = 2;
          for (let attempt = 0;attempt <= FETCH_MAX_RETRIES; attempt++) {
            const retryInit = bodyForRetry !== init?.body ? { ...init, body: bodyForRetry } : init;
            const response = await originalFetch(input, retryInit);
            if (!response.ok) {
              const errorBody = await response.text().catch(() => "");
              const isThrottled400 = response.status === 400 && errorBody.toLowerCase().includes("throttled");
              const isBudget402 = response.status === 402;
              const isRateLimit429 = response.status === 429;
              const isTimeout503 = response.status === 503;
              const isGoaway = errorBody.includes("GOAWAY") || errorBody.includes("392606");
              const retryable = isThrottled400 || isBudget402 || isRateLimit429 || isTimeout503 || isGoaway;
              if (retryable && attempt < FETCH_MAX_RETRIES) {
                _pluginLogger?.warn(`[frostclaw:fetch] retryable HTTP ${response.status} (attempt ${attempt + 1}/${FETCH_MAX_RETRIES + 1}), retrying... body=${errorBody.slice(0, 300)}`);
                await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
                continue;
              }
              _pluginLogger?.warn(`[frostclaw:fetch] non-2xx HTTP ${response.status}${retryable ? " (retries exhausted)" : " (non-retryable)"} body=${errorBody.slice(0, 300)}`);
              return new Response(errorBody, {
                status: response.status,
                statusText: response.statusText,
                headers: response.headers
              });
            }
            const ct = response.headers.get("content-type") ?? "";
            if (!ct.includes("text/event-stream") || !response.body) {
              return response;
            }
            const chunks = [];
            const reader = response.body.getReader();
            const decoder = new TextDecoder;
            let accumulated = "";
            let hasContentBlock = false;
            let hasMessageStop = false;
            let done = false;
            while (!done && !hasContentBlock && !hasMessageStop) {
              const { value, done: readerDone } = await reader.read();
              done = readerDone;
              if (value) {
                chunks.push(value);
                accumulated += decoder.decode(value, { stream: true });
                if (isAnthropicFormat) {
                  if (accumulated.includes("content_block_start"))
                    hasContentBlock = true;
                  if (accumulated.includes("message_stop"))
                    hasMessageStop = true;
                } else {
                  if (/"delta"\s*:\s*\{[^}]*"content"\s*:\s*"[^"]+"/.test(accumulated))
                    hasContentBlock = true;
                  if (accumulated.includes("[DONE]"))
                    hasMessageStop = true;
                }
              }
            }
            if (hasContentBlock || !hasMessageStop) {
              const remaining = reader;
              const combined = new ReadableStream({
                async start(controller) {
                  for (const chunk of chunks)
                    controller.enqueue(chunk);
                  while (true) {
                    const { value, done: done2 } = await remaining.read();
                    if (done2)
                      break;
                    if (value)
                      controller.enqueue(value);
                  }
                  controller.close();
                }
              });
              return new Response(combined, {
                status: response.status,
                statusText: response.statusText,
                headers: response.headers
              });
            }
            reader.cancel();
            const rawSse = accumulated.replace(/\n/g, "\\n").slice(0, 800);
            if (attempt < FETCH_MAX_RETRIES) {
              _pluginLogger?.warn(`[frostclaw:fetch] empty-stop detected (attempt ${attempt + 1}/${FETCH_MAX_RETRIES + 1}), retrying... raw=${rawSse}`);
              await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
              continue;
            }
            _pluginLogger?.warn(`[frostclaw:fetch] empty-stop persists after ${FETCH_MAX_RETRIES} retries, passing through raw=${rawSse}`);
            const emptyStream = new ReadableStream({
              start(controller) {
                for (const chunk of chunks)
                  controller.enqueue(chunk);
                controller.close();
              }
            });
            return new Response(emptyStream, {
              status: response.status,
              statusText: response.statusText,
              headers: response.headers
            });
          }
          return originalFetch(input, init);
        };
        _pluginLogger?.info("[frostclaw:fetch] empty-200 retry interceptor installed");
      } else {
        _pluginLogger?.info("[frostclaw:fetch] interceptor already installed, skipping");
      }
      api.registerEmbeddingProvider(snowflakeCortexEmbeddingAdapter);
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
          runtimeAugment: true,
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
          const baseUrl = getCatalogBaseURL();
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
              let isRetryableError = function(err) {
                if (!err)
                  return false;
                const msg = String(err);
                if (msg.includes("APIConnectionTimeoutError") || msg.includes("Connection timeout"))
                  return true;
                if (/timed?\s?out/i.test(msg))
                  return true;
                if (msg.includes("ECONNRESET") || msg.includes("ECONNREFUSED") || msg.includes("ENOTFOUND"))
                  return true;
                if (msg.includes("UND_ERR_SOCKET") || msg.includes("UND_ERR_CONNECT_TIMEOUT") || msg.includes("UND_ERR_HEADERS_TIMEOUT") || msg.includes("UND_ERR_BODY_TIMEOUT"))
                  return true;
                if (msg.includes("AbortError") || msg.includes("The operation was aborted"))
                  return true;
                if (msg.includes("network error") || msg.includes("fetch failed"))
                  return true;
                if (/\bterminated\b/i.test(msg))
                  return true;
                if (msg.includes("GOAWAY") || msg.includes("392606"))
                  return true;
                return false;
              }, isEmptyStop = function(msg, attempt) {
                if (!msg || typeof msg !== "object")
                  return false;
                const sr = msg.stopReason;
                if (sr !== "stop" && sr !== undefined && sr !== null)
                  return false;
                if (attempt >= EMPTY_STOP_MAX_RETRIES)
                  return false;
                if (!Array.isArray(msg.content))
                  return false;
                if (msg.content.length === 0)
                  return true;
                return msg.content.every((blk) => blk?.type === "thinking");
              };
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
                const fallbackBaseUrl = getCatalogBaseURL();
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
              if (isRequestDebugEnabled()) {
                const ts = new Date().toISOString();
                const thinkingInfo = thinkingActive ? `level=${thinkingLevel}` : "off";
                const ctxRecord = context;
                const msgCount = Array.isArray(ctxRecord?.messages) ? ctxRecord.messages.length : 0;
                const sysPrompt = typeof ctxRecord?.systemPrompt === "string" ? ctxRecord.systemPrompt : "";
                const maxTok = options?.maxTokens;
                _pluginLogger?.info(`[frostclaw:debug] ${ts} → Snowflake | model=${modelId} | attempt=1 | messages=${msgCount} | maxTokens=${maxTok} | thinking=${thinkingInfo} | systemPromptChars=${sysPrompt.length}`);
              }
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
                      record.messages = stripDocumentBlocks(record.messages);
                    }
                    stripEagerInputStreaming(record);
                    stripResponseFormat(record);
                    normalizeThinkingBudget(record, thinkingLevel, isAdaptiveOnly(String(model?.id ?? "")));
                    clampMaxTokens(record);
                  }
                  const chained = originalOnPayload?.(payload, payloadModel);
                  return chained !== undefined ? chained : payload;
                }
              };
              const streamResult = inner(model, context, merged);
              const EMPTY_STOP_MAX_RETRIES = 2;
              const RETRY_BACKOFF_MS = (attempt) => 5000 * (attempt + 1);
              if (streamResult && typeof streamResult.then === "function" && typeof streamResult.result !== "function") {
                return (async () => {
                  let currentResult = streamResult;
                  for (let attempt = 0;attempt <= EMPTY_STOP_MAX_RETRIES; attempt++) {
                    if (attempt > 0 && isRequestDebugEnabled()) {
                      const ts = new Date().toISOString();
                      const ctxRecord2 = context;
                      const msgCount2 = Array.isArray(ctxRecord2?.messages) ? ctxRecord2.messages.length : 0;
                      const maxTok2 = options?.maxTokens;
                      const thinkingInfo2 = thinkingActive ? `level=${thinkingLevel}` : "off";
                      const sysPrompt2 = typeof ctxRecord2?.systemPrompt === "string" ? ctxRecord2.systemPrompt : "";
                      _pluginLogger?.info(`[frostclaw:debug] ${ts} → Snowflake | model=${modelId} | attempt=${attempt + 1} | messages=${msgCount2} | maxTokens=${maxTok2} | thinking=${thinkingInfo2} | systemPromptChars=${sysPrompt2.length}`);
                    }
                    let value;
                    try {
                      value = await currentResult;
                    } catch (err) {
                      if (isRetryableError(err) && attempt < EMPTY_STOP_MAX_RETRIES) {
                        logWarn("wrapStreamFn.promise: retryable error from Snowflake, retrying", {
                          modelId,
                          attempt,
                          error: String(err)
                        });
                        await new Promise((r) => setTimeout(r, RETRY_BACKOFF_MS(attempt)));
                        const retryResult = inner(model, context, merged);
                        if (retryResult && typeof retryResult.then === "function") {
                          currentResult = retryResult;
                          continue;
                        }
                        return retryResult;
                      }
                      logError("wrapStreamFn.promise REJECTED", {
                        error: String(err),
                        stack: err instanceof Error ? err.stack : undefined
                      });
                      throw err;
                    }
                    const valueObj = value;
                    if (valueObj && typeof valueObj === "object" && (valueObj.errorMessage || valueObj.stopReason === "error")) {
                      logWarn("wrapStreamFn.promise resolved with error", {
                        errorMessage: valueObj.errorMessage,
                        stopReason: valueObj.stopReason
                      });
                    }
                    if (isEmptyStop(valueObj, attempt)) {
                      const isThinkingOnly = Array.isArray(valueObj?.content) && valueObj.content.length > 0;
                      logWarn("wrapStreamFn.promise: empty/thinking-only stop from Snowflake, retrying", {
                        modelId,
                        attempt,
                        isThinkingOnly,
                        contentLength: Array.isArray(valueObj?.content) ? valueObj.content.length : 0
                      });
                      const retryResult = inner(model, context, merged);
                      if (retryResult && typeof retryResult.then === "function") {
                        currentResult = retryResult;
                        continue;
                      }
                      return retryResult;
                    }
                    if (isRequestDebugEnabled()) {
                      const ts = new Date().toISOString();
                      const finalObj = valueObj;
                      const contentBlocks = Array.isArray(finalObj?.content) ? finalObj.content.length : 0;
                      _pluginLogger?.info(`[frostclaw:debug] ${ts} ← Snowflake | model=${modelId} | attempt=${attempt + 1} | stop_reason=${finalObj?.stopReason} | content_blocks=${contentBlocks} | retry=false`);
                    }
                    return value;
                  }
                })();
              }
              if (streamResult && typeof streamResult[Symbol.asyncIterator] === "function" && typeof streamResult.result === "function") {
                const outerStream = createAssistantMessageEventStream();
                (async () => {
                  let currentStream = streamResult;
                  for (let attempt = 0;attempt <= EMPTY_STOP_MAX_RETRIES; attempt++) {
                    if (attempt > 0 && isRequestDebugEnabled()) {
                      const ts = new Date().toISOString();
                      const ctxRecord3 = context;
                      const msgCount3 = Array.isArray(ctxRecord3?.messages) ? ctxRecord3.messages.length : 0;
                      const maxTok3 = options?.maxTokens;
                      const thinkingInfo3 = thinkingActive ? `level=${thinkingLevel}` : "off";
                      const sysPrompt3 = typeof ctxRecord3?.systemPrompt === "string" ? ctxRecord3.systemPrompt : "";
                      _pluginLogger?.info(`[frostclaw:debug] ${ts} → Snowflake | model=${modelId} | attempt=${attempt + 1} | messages=${msgCount3} | maxTokens=${maxTok3} | thinking=${thinkingInfo3} | systemPromptChars=${sysPrompt3.length}`);
                    }
                    const buffer = [];
                    let hasContent = false;
                    let sseSeq = 0;
                    let _dbgContentLen = 0;
                    try {
                      for await (const event of currentStream) {
                        const evType = event?.type;
                        if (isRequestDebugEnabled()) {
                          const evObj = event;
                          if (evType === "message_start") {
                            const usage = evObj.usage;
                            _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType} | input_tokens=${usage?.input_tokens} | cache_read_input_tokens=${usage?.cache_read_input_tokens ?? 0} | cache_creation_input_tokens=${usage?.cache_creation_input_tokens ?? 0}`);
                          } else if (evType === "content_block_delta") {
                            const delta = evObj.delta;
                            const deltaType = delta?.type;
                            if (typeof delta?.text === "string")
                              _dbgContentLen += delta.text.length;
                            if (typeof delta?.partial_json === "string")
                              _dbgContentLen += delta.partial_json.length;
                            if (typeof delta?.thinking === "string")
                              _dbgContentLen += delta.thinking.length;
                            _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType} | delta_type=${deltaType} | content_len=${_dbgContentLen}`);
                          } else if (evType === "message_delta") {
                            const delta2 = evObj.delta;
                            const usage2 = evObj.usage;
                            _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType} | stop_reason=${delta2?.stop_reason} | output_tokens=${usage2?.output_tokens}`);
                          } else if (evType === "message_stop") {
                            _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType}`);
                          } else if (evType === "content_block_start") {
                            const cb = evObj.content_block;
                            _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType} | block_type=${cb?.type}`);
                          } else if (evType === "content_block_stop") {
                            _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType}`);
                          } else {
                            _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType}`);
                          }
                        }
                        const isContentEvent = evType === "text_start" || evType === "text_delta" || evType === "text_end" || evType === "toolcall_start" || evType === "toolcall_delta" || evType === "toolcall_end" || evType === "thinking_start" || evType === "thinking_delta" || evType === "thinking_end";
                        if (!hasContent && isContentEvent) {
                          hasContent = true;
                          for (const buffered of buffer) {
                            outerStream.push(buffered);
                          }
                          buffer.length = 0;
                        }
                        if (hasContent) {
                          outerStream.push(event);
                        } else {
                          buffer.push(event);
                        }
                      }
                    } catch (err) {
                      if (!hasContent && isRetryableError(err) && attempt < EMPTY_STOP_MAX_RETRIES) {
                        logWarn("wrapStreamFn.stream: retryable error from Snowflake (no content emitted), retrying", {
                          modelId,
                          attempt,
                          error: String(err)
                        });
                        buffer.length = 0;
                        await new Promise((r) => setTimeout(r, RETRY_BACKOFF_MS(attempt)));
                        const retryResult = inner(model, context, merged);
                        if (retryResult && typeof retryResult[Symbol.asyncIterator] === "function" && typeof retryResult.result === "function") {
                          currentStream = retryResult;
                          continue;
                        }
                      }
                      logError("wrapStreamFn.stream ERROR", {
                        attempt,
                        error: String(err),
                        stack: err instanceof Error ? err.stack : undefined
                      });
                      for (const buffered of buffer) {
                        outerStream.push(buffered);
                      }
                      outerStream.end();
                      return;
                    }
                    let finalMsg;
                    try {
                      finalMsg = await currentStream.result();
                    } catch {
                      finalMsg = undefined;
                    }
                    const msg = finalMsg;
                    if (isRequestDebugEnabled()) {
                      const ts = new Date().toISOString();
                      const contentBlocks = Array.isArray(msg?.content) ? msg.content.length : 0;
                      const isEmpty = !hasContent && isEmptyStop(msg, attempt);
                      const usage = msg?.usage;
                      _pluginLogger?.info(`[frostclaw:debug] ${ts} ← Snowflake | model=${modelId} | attempt=${attempt + 1} | stop_reason=${msg?.stopReason} | content_blocks=${contentBlocks} | empty_stop=${isEmpty} | input_tokens=${usage?.inputTokens ?? usage?.input_tokens} | output_tokens=${usage?.outputTokens ?? usage?.output_tokens} | cache_read_input_tokens=${usage?.cacheReadInputTokens ?? usage?.cache_read_input_tokens ?? 0} | cache_creation_input_tokens=${usage?.cacheCreationInputTokens ?? usage?.cache_creation_input_tokens ?? 0}`);
                    }
                    if (!hasContent && isEmptyStop(msg, attempt)) {
                      const isThinkingOnly = Array.isArray(msg?.content) && msg.content.length > 0;
                      logWarn("wrapStreamFn.stream: empty/thinking-only stop from Snowflake, retrying", {
                        modelId,
                        attempt,
                        isThinkingOnly
                      });
                      buffer.length = 0;
                      const retryResult = inner(model, context, merged);
                      if (retryResult && typeof retryResult[Symbol.asyncIterator] === "function" && typeof retryResult.result === "function") {
                        currentStream = retryResult;
                        continue;
                      }
                      outerStream.end(finalMsg);
                      return;
                    }
                    for (const buffered of buffer) {
                      outerStream.push(buffered);
                    }
                    outerStream.end(finalMsg);
                    return;
                  }
                })();
                return outerStream;
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
        normalizeResolvedModel: (_ctx) => {
          return { ..._ctx.model };
        },
        applyConfigDefaults: (_ctx) => {
          return null;
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
