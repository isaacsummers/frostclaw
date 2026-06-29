/**
 * Pure payload transform functions for the Snowflake Cortex plugin.
 *
 * Extracted here so they can be tested in isolation without importing the
 * plugin entry point (which drags in openclaw SDK deps unavailable in tests).
 */

// ---------------------------------------------------------------------------
// Message repair — trailing assistant & empty text blocks
// ---------------------------------------------------------------------------

/**
 * Claude 4.6+ dropped assistant message prefill support.  Any payload ending
 * with role "assistant" returns HTTP 400.  Trim the trailing message — it is
 * always a completed prior turn, so nothing the user hasn't seen is lost.
 *
 * Returns the original array reference when no trim is needed (zero allocation).
 */
export function fixTrailingAssistant(messages: unknown[]): unknown[] {
  const last = messages[messages.length - 1];
  if (!last || typeof last !== "object") return messages;
  if ((last as Record<string, unknown>).role !== "assistant") return messages;
  return messages.slice(0, -1);
}

/**
 * Snowflake Cortex rejects:
 *   - text content blocks with empty or whitespace-only text
 *   - messages with an empty content array (content: [])
 *
 * For empty text blocks: replace with a zero-width space to preserve structure.
 * For empty content arrays: replace with a single zero-width-space text block.
 *
 * Returns the original array reference when no fix is needed (fast path,
 * zero allocations — the common case in well-formed sessions).
 */
export function fixEmptyTextBlocks(messages: unknown[]): unknown[] {
  // Fast path: scan for any message that needs fixing before allocating.
  let needsFix = false;
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const m = msg as Record<string, unknown>;
    if (!Array.isArray(m.content)) continue;
    if (m.content.length === 0) { needsFix = true; break; }
    for (const block of m.content) {
      if (!block || typeof block !== "object") continue;
      const b = block as Record<string, unknown>;
      if (b.type === "text" && typeof b.text === "string" && b.text.trim() === "") {
        needsFix = true;
        break;
      }
    }
    if (needsFix) break;
  }
  if (!needsFix) return messages;

  // Slow path: only reached when a bad block is found (error recovery cases).
  return messages.map((msg) => {
    if (!msg || typeof msg !== "object") return msg;
    const m = msg as Record<string, unknown>;
    if (!Array.isArray(m.content)) return msg;
    if (m.content.length === 0) {
      return { ...m, content: [{ type: "text", text: "\u200b" }] };
    }
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

// ---------------------------------------------------------------------------
// Tool schema scrubbing — eager_input_streaming
// ---------------------------------------------------------------------------

/**
 * Strip `eager_input_streaming` from every tool on an outbound Anthropic
 * Messages payload.
 *
 * `eager_input_streaming` is a top-level tool field (not inside `custom`)
 * supported on the direct Anthropic API, Bedrock, Vertex, and Foundry —
 * but NOT on Snowflake Cortex. Cortex\'s strict request validator rejects
 * any field it does not recognise with:
 *   400 invalid request parameters: tools.0.eager_input_streaming:
 *       Extra inputs are not permitted
 *
 * The catalog already sets `supportsEagerToolInputStreaming: false`, which
 * tells OpenClaw\'s Anthropic provider not to add the field. This function
 * is a defensive belt-and-suspenders strip on the final outbound payload —
 * it survives SDK upgrades that might add the field back, plugin
 * misconfiguration, or any future code path that re-introduces it.
 *
 * Mutates payload in place. Fast path: returns immediately when no tool
 * carries the field (the steady-state common case). No allocations on the
 * hot path.
 */
export function stripEagerInputStreaming(
  payload: Record<string, unknown>,
): void {
  const tools = payload.tools;
  if (!Array.isArray(tools)) return;

  for (const tool of tools) {
    if (!tool || typeof tool !== "object") continue;
    const t = tool as Record<string, unknown>;
    if (!("eager_input_streaming" in t)) continue;
    delete t.eager_input_streaming;
  }
}

// ---------------------------------------------------------------------------
// Thinking budget normalisation
// ---------------------------------------------------------------------------

/**
 * Map a thinking level to a budget_tokens value for non-adaptive (enabled) thinking.
 * Used only when the thinking type is "enabled" (explicit budget path).
 */
export function levelBudget(thinkingLevel: string | undefined): number {
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
export function levelEffort(thinkingLevel: string | undefined): string {
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
 * - `{ type: "enabled", budget_tokens: N }` → overwrite budget with levelBudget(level).
 * - `{ type: "disabled" }` → left untouched.
 *
 * When `adaptiveOnly` is true (e.g. Claude Opus 4.7/4.8, which reject manual
 * extended thinking with HTTP 400), an `{ type: "enabled" }` request is
 * silently redirected to the adaptive path: `thinking` becomes
 * `{ type: "adaptive" }` and `output_config.effort` is set from thinkingLevel,
 * with no `budget_tokens`.
 *
 * Mutates payload in place.
 */
export function normalizeThinkingBudget(
  payload: Record<string, unknown>,
  thinkingLevel: string | undefined,
  adaptiveOnly: boolean = false,
): void {
  const thinking = payload.thinking;
  if (!thinking || typeof thinking !== "object") return;
  const t = thinking as Record<string, unknown>;
  if (t.type === "disabled") return;

  if (t.type === "adaptive" || (adaptiveOnly && t.type === "enabled")) {
    const effort = levelEffort(thinkingLevel);
    const existing = payload.output_config as Record<string, unknown> | undefined;
    payload.thinking = { type: "adaptive" };
    payload.output_config = { ...existing, effort };
    return;
  }

  if (t.type === "enabled") {
    payload.thinking = { type: "enabled", budget_tokens: levelBudget(thinkingLevel) };
  }
}

// ---------------------------------------------------------------------------
// max_tokens clamping
// ---------------------------------------------------------------------------

const MAX_TOKENS_FLOOR_NO_THINKING = 1024;
const MAX_TOKENS_FLOOR_ADAPTIVE = 4096;
const MAX_TOKENS_OUTPUT_HEADROOM = 1024;

/**
 * Clamp `max_tokens` to a safe positive value.
 *
 * Floors:
 *   - no thinking         → 1024
 *   - thinking "enabled"  → budget_tokens + 1024
 *   - thinking "adaptive" → 4096
 *
 * Mutates payload in place; no-op when max_tokens is already above the floor.
 */
export function clampMaxTokens(payload: Record<string, unknown>): void {
  const current = payload.max_tokens;
  if (typeof current !== "number") return;

  const thinking = payload.thinking as Record<string, unknown> | undefined;
  const thinkingType =
    thinking && typeof thinking === "object" ? thinking.type : undefined;

  let floor = MAX_TOKENS_FLOOR_NO_THINKING;
  if (thinkingType === "enabled") {
    const budget = thinking?.budget_tokens;
    const budgetNum = typeof budget === "number" ? budget : 0;
    floor = budgetNum + MAX_TOKENS_OUTPUT_HEADROOM;
  } else if (thinkingType === "adaptive") {
    floor = MAX_TOKENS_FLOOR_ADAPTIVE;
  }

  if (current >= floor) return;
  payload.max_tokens = floor;
}

// ---------------------------------------------------------------------------
// Model classification
// ---------------------------------------------------------------------------

/**
 * Returns true when the model ID identifies a Claude model (case-insensitive
 * prefix match on "claude").
 */
export function isClaudeModel(modelId: string): boolean {
  return modelId.toLowerCase().startsWith("claude");
}

// ---------------------------------------------------------------------------
// response_format stripping — Snowflake Cortex does not support OpenAI-style
// response_format (neither json_object nor json_schema). Strip the field
// entirely so graphiti-core's prompt-injected schema guidance still works.
// ---------------------------------------------------------------------------

/**
 * Extract JSON schema info from response_format without deleting the field.
 * Returns null when the field is absent or type is not json_object/json_schema.
 */
export function peekResponseFormat(
  payload: Record<string, unknown>,
): { type: string; schema?: unknown } | null {
  if (!("response_format" in payload)) return null;
  const rf = payload.response_format as Record<string, unknown> | undefined;
  const type = typeof rf?.type === "string" ? rf.type : "json_object";
  if (type !== "json_object" && type !== "json_schema") return null;
  const schema =
    type === "json_schema"
      ? (rf?.json_schema as Record<string, unknown> | undefined)?.schema
      : undefined;
  return { type, schema };
}

/**
 * Strip the `response_format` field from an outbound OpenAI chat completions
 * payload before forwarding to Snowflake Cortex.
 *
 * Snowflake Cortex rejects both `{ type: "json_object" }` and
 * `{ type: "json_schema", ... }` with HTTP 400 for non-OpenAI models.
 * OpenAI models support response_format natively on Cortex's completions
 * endpoint — use peekResponseFormat + injectJsonSystemPrompt for those
 * instead (keep the field, only inject the prompt).
 *
 * Returns `{ type, schema }` or `null` when the field was absent.
 * Mutates payload in place.
 */
export function stripResponseFormat(
  payload: Record<string, unknown>,
): { type: string; schema?: unknown } | null {
  if (!("response_format" in payload)) return null;
  const rf = payload.response_format as Record<string, unknown> | undefined;
  const type = typeof rf?.type === "string" ? rf.type : "json_object";
  const schema =
    type === "json_schema"
      ? (rf?.json_schema as Record<string, unknown> | undefined)?.schema
      : undefined;
  delete payload.response_format;
  return { type, schema };
}

const JSON_OBJECT_SYSTEM_PROMPT =
  "You must respond with ONLY valid JSON. " +
  "Do not include any text before or after the JSON object. " +
  "Do not wrap the response in markdown code fences or backticks. " +
  "Do not add explanations, comments, or any prose. " +
  "Your entire response must be a single, complete, parseable JSON object.";

/**
 * Build the appropriate JSON system prompt.
 * For json_schema with a schema definition, embeds the schema so the model
 * knows the exact structure expected.
 * For json_object (or json_schema without a schema), uses the generic prompt.
 */
function buildJsonPrompt(type: string, schema?: unknown): string {
  if (type === "json_schema" && schema !== undefined) {
    return (
      "You must respond with ONLY valid JSON conforming exactly to this schema:\n" +
      JSON.stringify(schema, null, 2) + "\n\n" +
      "Do not include any text before or after the JSON. " +
      "Do not wrap in markdown code fences or backticks. " +
      "No explanations, comments, or prose. " +
      "Your entire response must be a single, complete, parseable JSON object matching the schema above."
    );
  }
  return JSON_OBJECT_SYSTEM_PROMPT;
}

/**
 * After stripping `response_format`, inject a strong system-prompt instruction
 * so the model still returns valid JSON without the API-level constraint.
 *
 * For json_schema, embeds the actual schema in the prompt so the model knows
 * the exact structure expected. For json_object, uses a generic JSON-only
 * instruction.
 *
 * Handles both string-content and array-content system messages:
 * - If a system message already exists, prepends the JSON instruction to its
 *   content so the original prompt is preserved.
 * - If no system message exists, inserts one at position 0.
 *
 * No-op when stripped is null or type is not json_object/json_schema.
 */
export function injectJsonSystemPrompt(
  messages: unknown[],
  stripped: { type: string; schema?: unknown } | null,
): unknown[] {
  // Only inject when the caller actually wanted JSON output.
  // response_format can also be { type: "text" } — injecting a JSON
  // constraint in that case would be actively wrong.
  if (
    stripped?.type !== "json_object" &&
    stripped?.type !== "json_schema"
  ) {
    return messages;
  }

  const prefix = buildJsonPrompt(stripped.type, stripped.schema);

  // Find an existing system message.
  const sysIdx = messages.findIndex(
    (m) =>
      m && typeof m === "object" &&
      (m as Record<string, unknown>).role === "system",
  );

  if (sysIdx === -1) {
    // No system message — prepend one.
    return [{ role: "system", content: prefix }, ...messages];
  }

  // Mutate a copy of the messages array, not the original.
  const result = [...messages];
  const sys = { ...(result[sysIdx] as Record<string, unknown>) };

  if (typeof sys.content === "string") {
    sys.content = `${prefix}\n\n${sys.content}`;
  } else if (Array.isArray(sys.content)) {
    // Array-form content blocks (Anthropic style that leaked in).
    sys.content = [
      { type: "text", text: prefix },
      ...sys.content,
    ];
  } else {
    // Unknown shape — replace with string.
    sys.content = prefix;
  }

  result[sysIdx] = sys;
  return result;
}

// ---------------------------------------------------------------------------
// Document block stripping — Snowflake Cortex does not support native PDF
// document blocks (Anthropic API feature). Strip before forwarding.
// ---------------------------------------------------------------------------

/**
 * Strip native document/PDF content blocks from messages before sending to
 * Snowflake Cortex. Snowflake rejects document blocks with HTTP 401 because
 * its Anthropic Messages endpoint is a strict subset of the full Anthropic
 * API — native PDF document blocks are not supported.
 *
 * Each stripped document block is replaced with a plain-text placeholder
 * noting the content was removed, so the model still sees a description
 * rather than a silent gap in the conversation.
 *
 * Returns the original array reference when no document blocks are found
 * (zero-allocation fast path — the common case).
 */
export function stripDocumentBlocks(messages: unknown[]): unknown[] {
  // Fast path: scan for any document block before allocating.
  let needsFix = false;
  outer: for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const m = msg as Record<string, unknown>;
    if (!Array.isArray(m.content)) continue;
    for (const block of m.content) {
      if (!block || typeof block !== "object") continue;
      if ((block as Record<string, unknown>).type === "document") {
        needsFix = true;
        break outer;
      }
    }
  }
  if (!needsFix) return messages;

  // Slow path: replace document blocks with text placeholders.
  return messages.map((msg) => {
    if (!msg || typeof msg !== "object") return msg;
    const m = msg as Record<string, unknown>;
    if (!Array.isArray(m.content)) return msg;
    const hasDoc = (m.content as unknown[]).some(
      (b) => b && typeof b === "object" && (b as Record<string, unknown>).type === "document",
    );
    if (!hasDoc) return msg;
    const fixed = (m.content as unknown[]).map((block: unknown) => {
      if (!block || typeof block !== "object") return block;
      const b = block as Record<string, unknown>;
      if (b.type !== "document") return block;
      const source = b.source as Record<string, unknown> | undefined;
      const mediaType = typeof source?.media_type === "string" ? source.media_type : "unknown";
      const title = typeof b.title === "string" ? ` ("${b.title}")` : "";
      return {
        type: "text",
        text: `[PDF/document block stripped${title} — Snowflake Cortex does not support native document blocks; media_type=${mediaType}]`,
      };
    });
    return { ...m, content: fixed };
  });
}
