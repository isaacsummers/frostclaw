/**
 * Extracted onPayload Claude-strip logic.
 *
 * Pure function with no openclaw imports — directly testable.
 */

import { isClaudeModel, stripEagerInputStreaming } from "./transforms.js";

/**
 * Defensive strip of eager_input_streaming from tool payloads for Claude
 * models. Called from the onPayload hook in index.ts.
 *
 * For Claude models: strips the field (defensive, in case compat flag ever
 * regresses). For non-Claude models: returns payload unchanged.
 */
export function applyEagerInputStreamingStrip(
  modelId: string,
  payload: unknown,
): unknown {
  if (!isClaudeModel(modelId)) return payload;
  if (!payload || typeof payload !== "object") return payload;
  stripEagerInputStreaming(payload as Record<string, unknown>);
  return payload;
}
