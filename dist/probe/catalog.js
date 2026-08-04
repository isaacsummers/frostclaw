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
    const t = tool;
    if (!("eager_input_streaming" in t))
      continue;
    delete t.eager_input_streaming;
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
function peekResponseFormat(payload) {
  if (!("response_format" in payload))
    return null;
  const rf = payload.response_format;
  const type = typeof rf?.type === "string" ? rf.type : "json_object";
  if (type !== "json_object" && type !== "json_schema")
    return null;
  const schema = type === "json_schema" ? rf?.json_schema?.schema : undefined;
  return { type, schema };
}
function stripResponseFormat(payload) {
  if (!("response_format" in payload))
    return null;
  const rf = payload.response_format;
  const type = typeof rf?.type === "string" ? rf.type : "json_object";
  const schema = type === "json_schema" ? rf?.json_schema?.schema : undefined;
  delete payload.response_format;
  return { type, schema };
}
var JSON_OBJECT_SYSTEM_PROMPT = "You must respond with ONLY valid JSON. " + "Do not include any text before or after the JSON object. " + "Do not wrap the response in markdown code fences or backticks. " + "Do not add explanations, comments, or any prose. " + "Your entire response must be a single, complete, parseable JSON object.";
function buildJsonPrompt(type, schema) {
  if (type === "json_schema" && schema !== undefined) {
    return `You must respond with ONLY valid JSON conforming exactly to this schema:
` + JSON.stringify(schema, null, 2) + `

` + "Do not include any text before or after the JSON. " + "Do not wrap in markdown code fences or backticks. " + "No explanations, comments, or prose. " + "Your entire response must be a single, complete, parseable JSON object matching the schema above.";
  }
  return JSON_OBJECT_SYSTEM_PROMPT;
}
function injectJsonSystemPrompt(messages, stripped) {
  if (stripped?.type !== "json_object" && stripped?.type !== "json_schema") {
    return messages;
  }
  const prefix = buildJsonPrompt(stripped.type, stripped.schema);
  const sysIdx = messages.findIndex((m) => m && typeof m === "object" && m.role === "system");
  if (sysIdx === -1) {
    return [{ role: "system", content: prefix }, ...messages];
  }
  const result = [...messages];
  const sys = { ...result[sysIdx] };
  if (typeof sys.content === "string") {
    sys.content = `${prefix}

${sys.content}`;
  } else if (Array.isArray(sys.content)) {
    sys.content = [
      { type: "text", text: prefix },
      ...sys.content
    ];
  } else {
    sys.content = prefix;
  }
  result[sysIdx] = sys;
  return result;
}
function historyRequiresThinkingBlocks(messages) {
  for (const msg of messages) {
    if (!msg || typeof msg !== "object")
      continue;
    const m = msg;
    if (m.role !== "assistant")
      continue;
    if (!Array.isArray(m.content) || m.content.length === 0)
      continue;
    const firstBlock = m.content[0];
    if (!firstBlock)
      continue;
    const firstType = firstBlock.type;
    if (firstType === "thinking" || firstType === "redacted_thinking")
      continue;
    const hasToolUse = m.content.some((b) => b && typeof b === "object" && b.type === "tool_use");
    if (hasToolUse)
      return true;
  }
  return false;
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
  { id: "claude-sonnet-5", name: "Claude Sonnet 5", reasoning: true, adaptiveOnly: true, contextWindow: 1e6, maxTokens: 128000, input: ["text", "image"] },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", reasoning: true, contextWindow: 1e6, maxTokens: 64000, input: ["text", "image"] },
  { id: "claude-4-sonnet", name: "Claude 4 Sonnet", reasoning: true, contextWindow: 1e6, maxTokens: 64000, input: ["text", "image"] },
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
  { id: "llama3.1-70b", name: "Llama 3.1 70B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama3.1-8b", name: "Llama 3.1 8B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama3.3-70b", name: "Llama 3.3 70B", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "llama4-maverick", name: "Llama 4 Maverick", reasoning: false, contextWindow: 1047576, maxTokens: 32768, input: ["text"] },
  { id: "mistral-large2", name: "Mistral Large 2", reasoning: false, contextWindow: 128000, maxTokens: 32768, input: ["text"] },
  { id: "mistral-7b", name: "Mistral 7B", reasoning: false, contextWindow: 32000, maxTokens: 8192, input: ["text"] }
];
var COST_FABLE = { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 };
var COST_OPUS = { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 };
var COST_SONNET = { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 };
var COST_SONNET_LONG = { input: 6, output: 30, cacheRead: 0.6, cacheWrite: 7.5 };
var COST_HAIKU = { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 };
var COST_GPT54 = { input: 2.75, output: 16.5, cacheRead: 0.275, cacheWrite: 0 };
var COST_GPT52 = { input: 1.93, output: 15.4, cacheRead: 0.193, cacheWrite: 0 };
var COST_GPT51 = { input: 1.38, output: 11, cacheRead: 0.138, cacheWrite: 0 };
var COST_GPT5_MINI = { input: 0.275, output: 2.2, cacheRead: 0.028, cacheWrite: 0 };
var COST_GPT5_NANO = { input: 0.06, output: 0.44, cacheRead: 0.006, cacheWrite: 0 };
var COST_GPT41 = { input: 2.2, output: 8.8, cacheRead: 0.55, cacheWrite: 0 };
var COST_LLAMA_70B = { input: 0.72, output: 0.72, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA_8B = { input: 0.22, output: 0.22, cacheRead: 0, cacheWrite: 0 };
var COST_LLAMA4_MAV = { input: 0.24, output: 0.97, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_LG2 = { input: 2, output: 6, cacheRead: 0, cacheWrite: 0 };
var COST_MISTRAL_7B = { input: 0.15, output: 0.2, cacheRead: 0, cacheWrite: 0 };
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
  if (id === "llama3.1-70b")
    return COST_LLAMA_70B;
  if (id === "llama3.1-8b")
    return COST_LLAMA_8B;
  if (id === "llama3.3-70b")
    return COST_LLAMA_70B;
  if (id === "llama4-maverick")
    return COST_LLAMA4_MAV;
  if (id === "mistral-large2")
    return COST_MISTRAL_LG2;
  if (id === "mistral-7b")
    return COST_MISTRAL_7B;
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
export {
  isClaudeModel,
  isAdaptiveOnly,
  getCatalogBaseURL,
  findCatalogEntry,
  buildOpenSourceModelDef,
  buildOpenAIModelDef,
  buildModelCatalog,
  buildClaudeModelDef,
  anthropicBetaHeaders,
  OPEN_SOURCE_MODELS,
  OPENAI_MODELS,
  CLAUDE_MODELS,
  BETA_ALWAYS
};
