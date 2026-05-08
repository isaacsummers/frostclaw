#!/usr/bin/env node
/**
 * test-thinking-levels.mjs — Snowflake Cortex thinking level validation
 *
 * ╔══════════════════════════════════════════════════════════════════════════╗
 * ║  FINDINGS (last run 2026-05-05)                                         ║
 * ╠══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                          ║
 * ║  Beta flags (all frostclaw flags):                                       ║
 * ║    - All accepted (200 OK) on both Sonnet 4.6 and Opus 4.7              ║
 * ║    - Not required — enabled mode works with zero headers                 ║
 * ║                                                                          ║
 * ║  claude-sonnet-4-6:                                                      ║
 * ║    - enabled (budget_tokens): ✅ works, returns real thinking blocks     ║
 * ║      regardless of beta flags or trigger words                           ║
 * ║    - adaptive/effort=high: ✅ fires thinking for complex prompts without ║
 * ║      trigger words (361 chars on trick question). Trigger words          ║
 * ║      increase depth slightly (476 vs 361 chars) but not required.        ║
 * ║    - adaptive/effort=low: ⚠️  inconsistent — fires on some complex      ║
 * ║      prompts but misses simple ones; trigger words help but unreliable.  ║
 * ║    - "think carefully" / "think step by step" in prompt: increases       ║
 * ║      thinking at effort=high; marginally helps effort=low on trivial     ║
 * ║      prompts. Not needed at effort=high for non-trivial tasks.           ║
 * ║                                                                          ║
 * ║  claude-opus-4-7:                                                        ║
 * ║    - enabled (budget_tokens): ❌ 400 "invalid request" — unsupported    ║
 * ║      on Snowflake Cortex regardless of beta flags                        ║
 * ║    - adaptive: ✅ accepted, effort controls reply verbosity/length       ║
 * ║    - thinking blocks: NEVER returned on Opus via Snowflake Cortex        ║
 * ║    - trigger words: no effect on thinking blocks (none exist); do        ║
 * ║      affect reply verbosity at effort=high                               ║
 * ║                                                                          ║
 * ║  Practical guidance:                                                     ║
 * ║    - Guaranteed thinking  → Sonnet + enabled (medium/high)              ║
 * ║    - Contextual thinking  → Sonnet + adaptive/effort=high               ║
 * ║    - Opus reasoning       → adaptive only; opaque (no blocks exposed)   ║
 * ║    - Don't add trigger words to every spawn prompt                       ║
 * ║    - Beta headers: send them (frostclaw does), but they're not required  ║
 * ╚══════════════════════════════════════════════════════════════════════════╝
 *
 * Three test suites:
 *   1. Beta flag acceptance — which flags are valid/invalid, alone and combined
 *   2. Thinking modes × beta combos — what actually produces thinking blocks
 *   3. Adaptive vs disabled quality probe — does adaptive do anything real?
 */

const PAT      = process.env.SNOWFLAKE_PAT ?? process.env.SNOWFLAKE_CORTEX_API_KEY ?? "";
const BASE_URL = process.env.SNOWFLAKE_BASE_URL ?? "";
if (!PAT || !BASE_URL) { console.error("Missing SNOWFLAKE_PAT or SNOWFLAKE_BASE_URL"); process.exit(1); }

const ENDPOINT = `${BASE_URL}/api/v2/cortex/v1/messages`;

const MODELS = ["claude-sonnet-4-6", "claude-opus-4-7"];

// Known flags from frostclaw
const FLAG_ALWAYS_1  = "output-128k-2025-02-19";
const FLAG_ALWAYS_2  = "token-efficient-tools-2025-02-19";
const FLAG_THINK_1   = "interleaved-thinking-2025-05-14";
const FLAG_THINK_2   = "effort-2025-11-24";
const FLAG_THINK_3   = "tool-examples-2025-10-29";

const ALL_FLAGS = [FLAG_ALWAYS_1, FLAG_ALWAYS_2, FLAG_THINK_1, FLAG_THINK_2, FLAG_THINK_3];

// ─── HTTP helper ──────────────────────────────────────────────────────────────

async function call(model, body, betaFlags) {
  const headers = {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${PAT}`,
    "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    ...(betaFlags && betaFlags.length > 0 ? { "anthropic-beta": betaFlags.join(",") } : {}),
  };
  const start = Date.now();
  try {
    const res = await fetch(ENDPOINT, {
      method: "POST", headers, body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    });
    const elapsed = Date.now() - start;
    const text = await res.text();
    let parsed = null;
    try { parsed = JSON.parse(text); } catch {}
    if (!res.ok) return { ok: false, status: res.status, elapsed, err: text.slice(0, 250) };
    return {
      ok: true, status: res.status, elapsed,
      stopReason: parsed?.stop_reason ?? "?",
      inputTokens: parsed?.usage?.input_tokens ?? "?",
      outputTokens: parsed?.usage?.output_tokens ?? "?",
      thinkingBlocks: Array.isArray(parsed?.content)
        ? parsed.content.filter(b => b.type === "thinking")
        : [],
      textBlocks: Array.isArray(parsed?.content)
        ? parsed.content.filter(b => b.type === "text")
        : [],
    };
  } catch (err) {
    return { ok: false, status: "?", elapsed: Date.now() - start, err: String(err).slice(0, 150) };
  }
}

// ─── Formatting ───────────────────────────────────────────────────────────────

const pad  = (s, w) => String(s ?? "").padEnd(w);
const rpad = (s, w) => String(s ?? "").padStart(w);
const hr   = (w=90) => "─".repeat(w);

function resultLine(label, r, colW=42) {
  if (!r.ok) {
    return `  ${pad(label, colW)} ${rpad(r.status, 4)} ${rpad(r.elapsed+"ms", 7)}  ✗ ${r.err}`;
  }
  const thinking = r.thinkingBlocks.length > 0
    ? `YES (${r.thinkingBlocks.reduce((n,b)=>n+(b.thinking?.length??0),0)} chars)`
    : "no";
  const reply = r.textBlocks.map(b=>b.text?.trim().slice(0,50)).join(" ").slice(0,50);
  return `  ${pad(label, colW)} ${rpad(r.status, 4)} ${rpad(r.elapsed+"ms", 7)}  ${pad(thinking,20)}  ${reply}`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 1: Beta flag acceptance
// ═══════════════════════════════════════════════════════════════════════════════

async function suiteBetaFlags() {
  console.log("\n\n" + "═".repeat(90));
  console.log("SUITE 1 — Beta flag acceptance (disabled thinking, simple prompt)");
  console.log("═".repeat(90));
  console.log("Tests each flag alone and in combinations to find what Snowflake accepts.\n");

  const body = (model) => ({
    model,
    max_tokens: 64,
    thinking: { type: "disabled" },
    messages: [{ role: "user", content: "Reply with exactly: OK" }],
  });

  // Build test cases: each flag solo, all combos of thinking flags, no flags
  const betaCases = [
    { label: "no beta headers",       flags: [] },
    { label: FLAG_ALWAYS_1,           flags: [FLAG_ALWAYS_1] },
    { label: FLAG_ALWAYS_2,           flags: [FLAG_ALWAYS_2] },
    { label: FLAG_THINK_1,            flags: [FLAG_THINK_1] },
    { label: FLAG_THINK_2,            flags: [FLAG_THINK_2] },
    { label: FLAG_THINK_3,            flags: [FLAG_THINK_3] },
    { label: "always-1 + always-2",   flags: [FLAG_ALWAYS_1, FLAG_ALWAYS_2] },
    { label: "all ALWAYS",            flags: [FLAG_ALWAYS_1, FLAG_ALWAYS_2] },
    { label: "all THINKING",          flags: [FLAG_THINK_1, FLAG_THINK_2, FLAG_THINK_3] },
    { label: "ALL flags",             flags: ALL_FLAGS },
  ];

  for (const model of MODELS) {
    console.log(`\nModel: ${model}`);
    console.log(hr());
    console.log(`  ${pad("Beta flags sent", 42)} ${rpad("HTTP", 4)} ${rpad("ms", 7)}  result`);
    console.log(hr());
    for (const bc of betaCases) {
      const r = await call(model, body(model), bc.flags);
      console.log(resultLine(bc.label, r));
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 2: Thinking modes × beta header combinations
// ═══════════════════════════════════════════════════════════════════════════════

async function suiteThinkingModes() {
  console.log("\n\n" + "═".repeat(90));
  console.log("SUITE 2 — Thinking modes × beta combinations");
  console.log("═".repeat(90));
  console.log("Does the thinking mode + flags combo actually produce thinking blocks?\n");

  const BETA_COMBOS = [
    { label: "no headers",               flags: [] },
    { label: "ALWAYS only",              flags: [FLAG_ALWAYS_1, FLAG_ALWAYS_2] },
    { label: "ALWAYS + interleaved",     flags: [FLAG_ALWAYS_1, FLAG_ALWAYS_2, FLAG_THINK_1] },
    { label: "ALWAYS + effort",          flags: [FLAG_ALWAYS_1, FLAG_ALWAYS_2, FLAG_THINK_2] },
    { label: "ALWAYS + all THINKING",    flags: ALL_FLAGS },
  ];

  const THINK_MODES = [
    { label: "disabled",            body: (m) => ({ model: m, max_tokens: 512,  thinking: { type: "disabled" }, messages: [{role:"user",content:"Reply with exactly: OK"}] }) },
    { label: "enabled budget=4000", body: (m) => ({ model: m, max_tokens: 5024, thinking: { type: "enabled", budget_tokens: 4000 }, messages: [{role:"user",content:"Reply with exactly: OK"}] }) },
    { label: "enabled budget=8000", body: (m) => ({ model: m, max_tokens: 9024, thinking: { type: "enabled", budget_tokens: 8000 }, messages: [{role:"user",content:"Reply with exactly: OK"}] }) },
    { label: "adaptive effort=low", body: (m) => ({ model: m, max_tokens: 512,  thinking: { type: "adaptive" }, output_config: { effort: "low" },  messages: [{role:"user",content:"Reply with exactly: OK"}] }) },
    { label: "adaptive effort=high",body: (m) => ({ model: m, max_tokens: 512,  thinking: { type: "adaptive" }, output_config: { effort: "high" }, messages: [{role:"user",content:"Reply with exactly: OK"}] }) },
  ];

  for (const model of MODELS) {
    console.log(`\nModel: ${model}`);
    for (const tm of THINK_MODES) {
      console.log(`\n  thinking mode: ${tm.label}`);
      console.log(`  ${pad("Beta flags", 36)} ${rpad("HTTP",4)} ${rpad("ms",7)}  ${pad("thinking blocks?",20)}  reply`);
      console.log("  " + hr(86));
      for (const bc of BETA_COMBOS) {
        const r = await call(model, tm.body(model), bc.flags);
        console.log(resultLine(bc.label, r, 36));
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 3: Does adaptive actually do anything real?
// ═══════════════════════════════════════════════════════════════════════════════

async function suiteAdaptiveQuality() {
  console.log("\n\n" + "═".repeat(90));
  console.log("SUITE 3 — Adaptive vs disabled: does it actually reason differently?");
  console.log("═".repeat(90));
  console.log("Uses a reasoning-heavy prompt to see if thinking mode changes the answer quality.\n");

  const REASONING_PROMPT = "A farmer has 17 sheep. All but 9 die. How many sheep are left? Show your reasoning.";

  const cases = [
    { label: "disabled",             thinking: { type: "disabled" },                     outputConfig: null,            flags: [FLAG_ALWAYS_1, FLAG_ALWAYS_2] },
    { label: "enabled/medium",       thinking: { type: "enabled", budget_tokens: 8000 }, outputConfig: null,            flags: ALL_FLAGS },
    { label: "adaptive/effort=low",  thinking: { type: "adaptive" },                     outputConfig: { effort:"low" }, flags: ALL_FLAGS },
    { label: "adaptive/effort=high", thinking: { type: "adaptive" },                     outputConfig: { effort:"high"}, flags: ALL_FLAGS },
  ];

  for (const model of MODELS) {
    console.log(`\nModel: ${model}`);
    console.log(hr());
    for (const tc of cases) {
      const maxTok = tc.thinking.type === "enabled" ? tc.thinking.budget_tokens + 2048 : 2048;
      const body = {
        model,
        max_tokens: maxTok,
        thinking: tc.thinking,
        ...(tc.outputConfig ? { output_config: tc.outputConfig } : {}),
        messages: [{ role: "user", content: REASONING_PROMPT }],
      };
      const r = await call(model, body, tc.flags);
      if (!r.ok) {
        console.log(`  ${pad(tc.label, 28)}  ✗ HTTP ${r.status}: ${r.err}`);
      } else {
        const thinkLen = r.thinkingBlocks.reduce((n,b)=>n+(b.thinking?.length??0),0);
        const reply    = r.textBlocks.map(b=>b.text?.trim()).join(" ").slice(0, 120);
        console.log(`  ${pad(tc.label, 28)}  thinking=${thinkLen} chars  out=${r.outputTokens} tok  reply: ${reply}`);
      }
    }
  }
}

// ─── Run all suites ───────────────────────────────────────────────────────────

console.log("Snowflake Cortex — Comprehensive Thinking Level Test");
console.log(`Endpoint: ${ENDPOINT}`);
console.log(`Time: ${new Date().toISOString()}`);

await suiteBetaFlags();
await suiteThinkingModes();
await suiteAdaptiveQuality();

console.log("\n\nDone.\n");

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 4: Does adaptive need a "think" trigger in the prompt?
// ═══════════════════════════════════════════════════════════════════════════════

async function suiteAdaptiveTriggers() {
  console.log("\n\n" + "═".repeat(90));
  console.log("SUITE 4 — Does adaptive need a 'think' / 'think step by step' trigger?");
  console.log("═".repeat(90));
  console.log("Same prompt variants with and without explicit think instructions.\n");

  const PROMPTS = [
    // Simple prompt, no trigger
    { label: "simple, no trigger",          prompt: "What is 2+2?" },
    // Simple prompt + think trigger
    { label: "simple + 'think'",            prompt: "Think carefully. What is 2+2?" },
    { label: "simple + 'think step by step'",prompt: "What is 2+2? Think step by step." },
    // Medium complexity, no trigger
    { label: "medium, no trigger",          prompt: "What are the tradeoffs between REST and GraphQL APIs?" },
    // Medium complexity + trigger
    { label: "medium + 'think'",            prompt: "Think carefully. What are the tradeoffs between REST and GraphQL APIs?" },
    { label: "medium + 'think step by step'",prompt: "What are the tradeoffs between REST and GraphQL APIs? Think step by step." },
    // Trick question (needs reasoning), no trigger
    { label: "trick, no trigger",           prompt: "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?" },
    // Trick question + trigger
    { label: "trick + 'think'",             prompt: "Think carefully. A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?" },
    { label: "trick + 'think step by step'",prompt: "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost? Think step by step." },
  ];

  const MODES = [
    { label: "disabled",             thinking: { type: "disabled" },    outputConfig: null,            flags: [FLAG_ALWAYS_1, FLAG_ALWAYS_2] },
    { label: "adaptive/effort=low",  thinking: { type: "adaptive" },    outputConfig: { effort:"low" }, flags: ALL_FLAGS },
    { label: "adaptive/effort=high", thinking: { type: "adaptive" },    outputConfig: { effort:"high"}, flags: ALL_FLAGS },
    { label: "enabled/medium",       thinking: { type: "enabled", budget_tokens: 8000 }, outputConfig: null, flags: ALL_FLAGS },
  ];

  for (const model of MODELS) {
    console.log(`\nModel: ${model}`);
    console.log(hr());

    for (const pm of PROMPTS) {
      console.log(`\n  Prompt: "${pm.label}"`);
      console.log(`  ${pad("Mode", 24)} ${rpad("HTTP",4)} ${rpad("ms",7)}  ${pad("thinking blocks?",22)}  out tok  reply`);
      console.log("  " + "─".repeat(86));

      for (const mode of MODES) {
        if (model === "claude-opus-4-7" && mode.thinking.type === "enabled") {
          console.log(`  ${pad(mode.label, 24)}  (skipped — Opus doesn't support enabled)`);
          continue;
        }
        const maxTok = mode.thinking.type === "enabled" ? mode.thinking.budget_tokens + 2048 : 2048;
        const body = {
          model,
          max_tokens: maxTok,
          thinking: mode.thinking,
          ...(mode.outputConfig ? { output_config: mode.outputConfig } : {}),
          messages: [{ role: "user", content: pm.prompt }],
        };
        const r = await call(model, body, mode.flags);
        if (!r.ok) {
          console.log(`  ${pad(mode.label, 24)} ${rpad(r.status,4)} ${rpad(r.elapsed+"ms",7)}  ✗ ${r.err.slice(0,60)}`);
        } else {
          const thinkLen = r.thinkingBlocks.reduce((n,b)=>n+(b.thinking?.length??0),0);
          const thinking = thinkLen > 0 ? `YES (${thinkLen} chars)` : "no";
          const reply = r.textBlocks.map(b=>b.text?.trim()).join(" ").replace(/\n/g," ").slice(0,50);
          console.log(`  ${pad(mode.label, 24)} ${rpad(r.status,4)} ${rpad(r.elapsed+"ms",7)}  ${pad(thinking,22)}  ${rpad(r.outputTokens,7)}  ${reply}`);
        }
      }
    }
  }
}

await suiteAdaptiveTriggers();
console.log("\n\nFull run complete.\n");
