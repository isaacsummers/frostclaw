#!/usr/bin/env bun
/**
 * thinking-matrix.mjs — Matrix test of frostclaw thinking levels × token variants.
 *
 * Faithfully replicates frostclaw's PRODUCTION request path:
 *   1. OpenClaw core buildParams() decides thinking.type from the level
 *      (both sonnet-4-6 & opus-4-8 are adaptive-capable → adaptive/disabled).
 *   2. frostclaw normalizeThinkingBudget() sets output_config.effort (adaptive)
 *      or budget_tokens (enabled) from the level.
 *   3. frostclaw clampMaxTokens() applies floors.
 *   4. Sends through the live snowflake-proxy (which injects PAT auth), exactly
 *      as the gateway does in production.
 *
 * READ-ONLY: imports frostclaw's real transforms; does not modify source.
 */

import {
  levelEffort,
  levelBudget,
  normalizeThinkingBudget,
  clampMaxTokens,
} from "../src/transforms.ts";

const PROXY = process.env.SNOWFLAKE_PROXY_BASE_URL ?? "http://127.0.0.1:18790";
const ENDPOINT = `${PROXY}/api/v2/cortex/v1/messages`;

const MODELS = ["claude-sonnet-4-6", "claude-opus-4-8"];
const CATALOG_MAX_TOKENS = 128_000; // both are reasoning Claude, catalog maxTokens
const LEVELS = ["adaptive", "high", "medium", "low", "minimal", "off", "enabled"];
const TOKEN_VARIANTS = [
  { label: "default", max: CATALOG_MAX_TOKENS },
  { label: "1000", max: 1000 },
  { label: "16000", max: 16000 },
  { label: "32000", max: 32000 },
];

const BETA_THINKING = "interleaved-thinking-2025-05-14";
const PROMPT = "Say hello in one word.";

// Both target models are adaptive-capable per OpenClaw's supportsAdaptiveThinking.
const supportsAdaptive = (id) =>
  /(opus-4-[678]|opus-4\.[678]|opus-4-8|sonnet-4-6|sonnet-4\.6)/.test(id);

/**
 * Build the payload frostclaw actually emits for (model, level, maxTokens).
 * Mirrors OpenClaw buildParams + frostclaw transforms.
 */
function buildPayload(model, level, requestedMax) {
  const thinkingActive = level !== "off";
  const payload = {
    model,
    max_tokens: requestedMax,
    stream: false,
    messages: [{ role: "user", content: PROMPT }],
  };

  if (!thinkingActive) {
    payload.thinking = { type: "disabled" };
  } else if (level === "enabled") {
    // Explicit budget path. frostclaw normalizeThinkingBudget overwrites budget
    // from levelBudget(level); "enabled" is not a named level → default 16000.
    payload.thinking = { type: "enabled", budget_tokens: 1024, display: "summarized" };
  } else if (supportsAdaptive(model)) {
    // OpenClaw emits adaptive for all on-levels on adaptive-capable models.
    payload.thinking = { type: "adaptive", display: "summarized" };
    payload.output_config = { effort: mapEffort(level) };
  } else {
    payload.thinking = { type: "enabled", budget_tokens: levelBudget(level), display: "summarized" };
  }

  // Apply frostclaw transforms exactly as the wrapStreamFn hook does.
  normalizeThinkingBudget(payload, level);
  clampMaxTokens(payload);
  return payload;
}

// OpenClaw mapThinkingLevelToEffort (no model.thinkingLevelMap → switch default)
function mapEffort(level) {
  switch (level) {
    case "minimal":
    case "low": return "low";
    case "medium": return "medium";
    case "high": return "high";
    case "max": return "max";
    default: return "high"; // adaptive
  }
}

async function call(payload, thinkingActive, modelReasoning = true) {
  const headers = {
    "Content-Type": "application/json",
    ...(thinkingActive && modelReasoning ? { "anthropic-beta": BETA_THINKING } : {}),
  };
  const start = Date.now();
  try {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(60000),
    });
    const elapsed = Date.now() - start;
    const text = await res.text();
    let parsed = null;
    try { parsed = JSON.parse(text); } catch {}
    if (!res.ok) {
      return { ok: false, status: res.status, elapsed, err: text.slice(0, 400) };
    }
    const content = Array.isArray(parsed?.content) ? parsed.content : [];
    return {
      ok: true,
      status: res.status,
      elapsed,
      stopReason: parsed?.stop_reason ?? "?",
      outputTokens: parsed?.usage?.output_tokens ?? "?",
      thinkingChars: content.filter((b) => b.type === "thinking")
        .reduce((n, b) => n + (b.thinking?.length ?? 0), 0),
      reply: content.filter((b) => b.type === "text").map((b) => b.text?.trim()).join(" ").slice(0, 40),
    };
  } catch (err) {
    return { ok: false, status: "ERR", elapsed: Date.now() - start, err: String(err).slice(0, 200) };
  }
}

const results = {}; // results[model][level][variant] = {...}
const errorLog = [];

console.log(`Endpoint: ${ENDPOINT}`);
console.log(`Start: ${new Date().toISOString()}\n`);

for (const model of MODELS) {
  results[model] = {};
  console.log(`\n=== ${model} ===`);
  for (const level of LEVELS) {
    results[model][level] = {};
    for (const variant of TOKEN_VARIANTS) {
      const payload = buildPayload(model, level, variant.max);
      const thinkingActive = level !== "off";
      const r = await call(payload, thinkingActive);
      const effMax = payload.max_tokens;
      const tType = payload.thinking?.type;
      const effort = payload.output_config?.effort;
      const budget = payload.thinking?.budget_tokens;
      const meta = `type=${tType}${effort ? `/effort=${effort}` : ""}${budget ? `/budget=${budget}` : ""} reqMax=${variant.max}→eff=${effMax}`;

      let mark, detail;
      if (r.ok) {
        // Detect unexpected: thinking active but zero thinking chars (Opus expected)
        const expectThink = thinkingActive;
        const gotThink = r.thinkingChars > 0;
        if (expectThink && !gotThink) {
          mark = "⚠️";
          detail = `no thinking block (think=${r.thinkingChars}) stop=${r.stopReason} out=${r.outputTokens} reply="${r.reply}"`;
        } else {
          mark = "✅";
          detail = `think=${r.thinkingChars}c stop=${r.stopReason} out=${r.outputTokens} reply="${r.reply}"`;
        }
      } else {
        mark = "❌";
        detail = `HTTP ${r.status}: ${r.err}`;
        errorLog.push(`[${model}] level=${level} ${variant.label} (${meta})\n    HTTP ${r.status}: ${r.err}`);
      }
      results[model][level][variant.label] = { mark, detail, meta, status: r.status, elapsed: r.elapsed };
      console.log(`  ${level.padEnd(9)} ${variant.label.padEnd(8)} ${mark} ${r.elapsed}ms  ${meta}\n      ${detail}`);
    }
  }
}

// Emit JSON for the report builder
const out = { results, errorLog, generatedAt: new Date().toISOString(), endpoint: ENDPOINT };
await Bun.write(process.env.MATRIX_JSON_OUT ?? "/tmp/thinking-matrix.json", JSON.stringify(out, null, 2));
console.log(`\nDone. JSON → ${process.env.MATRIX_JSON_OUT ?? "/tmp/thinking-matrix.json"}`);
