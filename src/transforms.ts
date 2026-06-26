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
 * Strip `eager_input_streaming` from every tool's `custom` object on an
 * outbound Anthropic Messages payload.
 *
 * Snowflake Cortex's strict request validator (Haiku 4.5 and other tiers)
 * rejects the field with:
 *   400 invalid request parameters: tools.0.custom.eager_input_streaming:
 *       Extra inputs are not permitted
 *
 * The catalog already sets `supportsEagerToolInputStreaming: false`, which
 * tells pi-ai's Anthropic provider not to add the field. This function is a
 * defensive belt-and-suspenders strip on the final outbound payload — it
 * survives SDK upgrades that might add the field back, plugin
 * misconfiguration, or any future code path in pi-ai that re-introduces it.
 *
 * Scope: only `eager_input_streaming` is removed from each tool's `custom`
 * object. Other `custom` fields (name, description, input_schema, etc.) are
 * preserved. If `custom` is left empty after stripping, the key itself is
 * removed so we never send `"custom": {}`.
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
    const custom = (tool as Record<string, unknown>).custom;
    if (!custom || typeof custom !== "object") continue;
    const customRec = custom as Record<string, unknown>;
    if (!("eager_input_streaming" in customRec)) continue;

    delete customRec.eager_input_streaming;
    // Drop the now-empty container rather than emit `"custom": {}`,
    // which Cortex's strict validator would also reject as extra input.
    if (Object.keys(customRec).length === 0) {
      delete (tool as Record<string, unknown>).custom;
    }
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
 * Strip the `response_format` field from an outbound OpenAI chat completions
 * payload before forwarding to Snowflake Cortex.
 *
 * Snowflake Cortex rejects both `{ type: "json_object" }` and
 * `{ type: "json_schema", ... }` with HTTP 400. Removing the field lets the
 * caller rely on prompt-level JSON schema injection (graphiti-core's
 * json_object fallback path) instead of constrained decoding, which Cortex
 * does not expose.
 *
 * Mutates payload in place; no-op when the field is absent.
 */
export function stripResponseFormat(payload: Record<string, unknown>): void {
  if ("response_format" in payload) {
    delete payload.response_format;
  }
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
