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
export {
  stripResponseFormat,
  stripEagerInputStreaming,
  stripDocumentBlocks,
  peekResponseFormat,
  normalizeThinkingBudget,
  levelEffort,
  levelBudget,
  isClaudeModel,
  injectJsonSystemPrompt,
  fixTrailingAssistant,
  fixEmptyTextBlocks,
  clampMaxTokens
};
