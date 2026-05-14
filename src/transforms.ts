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
 * Mutates payload in place.
 */
export function normalizeThinkingBudget(
  payload: Record<string, unknown>,
  thinkingLevel: string | undefined,
): void {
  const thinking = payload.thinking;
  if (!thinking || typeof thinking !== "object") return;
  const t = thinking as Record<string, unknown>;
  if (t.type === "disabled") return;

  if (t.type === "adaptive") {
    const effort = levelEffort(thinkingLevel);
    const existing = payload.output_config as Record<string, unknown> | undefined;
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
