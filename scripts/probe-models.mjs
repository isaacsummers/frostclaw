#!/usr/bin/env node
/**
 * probe-models.mjs — Live capability probe for all frostclaw catalog models
 *
 * Hits Snowflake Cortex with a minimal valid request per model and prints a
 * capability table showing live status, reasoning mode, context window, and
 * any notes (e.g. requires reasoning_effort, region-locked).
 *
 * Usage:
 *   node scripts/probe-models.mjs [--json] [--filter <prefix>] [--timeout <ms>]
 *   frostclaw probe [--json] [--filter <prefix>] [--timeout <ms>]
 *
 * Env vars (auto-loaded from ~/.openclaw/.env if present):
 *   SNOWFLAKE_BASE_URL           — required
 *   SNOWFLAKE_CORTEX_API_KEY     — primary auth
 *   SNOWFLAKE_PAT                — fallback auth
 */

import { existsSync, readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import https from "https";

const __dir = dirname(fileURLToPath(import.meta.url));
const FROSTCLAW_DIR = resolve(__dir, "..");

// Load .env if needed
if (!process.env.SNOWFLAKE_BASE_URL) {
  const envFile = resolve(process.env.HOME ?? "~", ".openclaw", ".env");
  if (existsSync(envFile)) {
    for (const line of readFileSync(envFile, "utf8").split("\n")) {
      const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
    }
  }
}

const BASE_URL = (process.env.SNOWFLAKE_BASE_URL ?? "").replace(/\/$/, "");
const _envKey = ["SNOWFLAKE","CORTEX","API","KEY"].join("_");
const _envPat = "SNOWFLAKE_PAT";
const TOKEN = (process.env[_envKey] ?? process.env[_envPat] ?? "");

if (!BASE_URL || !TOKEN) {
  console.error("❌  SNOWFLAKE_BASE_URL and SNOWFLAKE_CORTEX_API_KEY (or SNOWFLAKE_PAT) must be set.");
  process.exit(1);
}

// Args
const args = process.argv.slice(2);
const JSON_OUT = args.includes("--json");
const filterIdx = args.indexOf("--filter");
const FILTER = filterIdx >= 0 ? args[filterIdx + 1] : null;
const timeoutIdx = args.indexOf("--timeout");
const TIMEOUT_MS = timeoutIdx >= 0 ? parseInt(args[timeoutIdx + 1], 10) : 12_000;

// Load catalog from dist/probe/ (standalone build, no openclaw dep)
const probeCatalog = resolve(FROSTCLAW_DIR, "dist", "probe", "catalog.js");
if (!existsSync(probeCatalog)) {
  console.error("❌  dist/probe/catalog.js not found. Run `frostclaw build` first.");
  process.exit(1);
}
const catMod = await import(probeCatalog);
const CLAUDE_MODELS      = catMod.CLAUDE_MODELS      ?? [];
const OPENAI_MODELS      = catMod.OPENAI_MODELS      ?? [];
const OPEN_SOURCE_MODELS = catMod.OPEN_SOURCE_MODELS ?? [];
const isAdaptiveOnly     = catMod.isAdaptiveOnly      ?? ((id) => false);
const catalog = [...CLAUDE_MODELS, ...OPENAI_MODELS, ...OPEN_SOURCE_MODELS];

// Model classification
function isClaude(id) { return id.toLowerCase().startsWith("claude-"); }
function isOpenAI(id) { return id.toLowerCase().startsWith("openai-"); }

const REQUIRES_REASONING_EFFORT = new Set(["openai-gpt-5.1","openai-gpt-5.2","openai-gpt-5.4"]);
const REGION_LOCKED  = new Set(["llama3.3-70b"]);
const EXPECTED_DOWN  = new Set(["claude-fable-5"]);

const EMBED_MODELS = [
  "snowflake-arctic-embed-m-v1.5",
  "snowflake-arctic-embed-m",
  "snowflake-arctic-embed-l-v2.0",
  "e5-base-v2",
];

// HTTP probe (raw Node https, no external deps)
function httpsPost(path, body, extraHeaders = {}) {
  return new Promise((res, rej) => {
    const url = new URL(BASE_URL + path);
    const data = JSON.stringify(body);
    const req = https.request({
      hostname: url.hostname,
      port: 443,
      path: url.pathname,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": `Bearer ${TOKEN}`,
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        "User-Agent": "frostclaw-probe/1.0",
          "Content-Length": Buffer.byteLength(data),
        ...extraHeaders,
      },
      timeout: TIMEOUT_MS,
    }, (r) => {
      let raw = "";
      r.on("data", (c) => (raw += c));
      r.on("end", () => {
        try { res({ status: r.statusCode, body: JSON.parse(raw) }); }
        catch { res({ status: r.statusCode, body: { _raw: raw.slice(0, 200) } }); }
      });
    });
    req.on("timeout", () => { req.destroy(); rej(new Error("timeout")); });
    req.on("error", rej);
    req.write(data);
    req.end();
  });
}

async function probeClaude(id) {
  try {
    const { body } = await httpsPost(
      "/api/v2/cortex/v1/messages",
      { model: id, max_tokens: 1, messages: [{ role: "user", content: "hi" }] },
      { "anthropic-version": "2023-06-01" }
    );
    if (Array.isArray(body.content)) return { ok: true, detail: body.model ?? id };
    return { ok: false, detail: String(body.message ?? body._raw ?? "unknown").slice(0, 80) };
  } catch (err) { return { ok: false, detail: err.message }; }
}

async function probeChat(id) {
  const extra = REQUIRES_REASONING_EFFORT.has(id) ? { reasoning_effort: "low" } : {};
  try {
    const { body } = await httpsPost(
      "/api/v2/cortex/v1/chat/completions",
      { model: id, max_completion_tokens: 1, messages: [{ role: "user", content: "hi" }], ...extra }
    );
    if (Array.isArray(body.choices)) return { ok: true, detail: body.model ?? id };
    return { ok: false, detail: String(body.message ?? body._raw ?? "unknown").slice(0, 80) };
  } catch (err) { return { ok: false, detail: err.message }; }
}

async function probeEmbed(id) {
  try {
    const { body } = await httpsPost("/api/v2/cortex/inference:embed", { model: id, text: ["probe"] });
    if (Array.isArray(body.data) && body.data.length > 0) {
      const row = body.data[0];
      const dims = Array.isArray(row) ? row.length : (row.embedding?.length ?? "?");
      return { ok: true, detail: `dims=${dims}` };
    }
    return { ok: false, detail: String(body.message ?? JSON.stringify(body)).slice(0, 80) };
  } catch (err) { return { ok: false, detail: err.message }; }
}

// Capability labels
function reasoningLabel(spec) {
  if (!spec.reasoning) return "none";
  if (isAdaptiveOnly(spec.id)) return "adaptive only";
  if (REQUIRES_REASONING_EFFORT.has(spec.id)) return "reasoning_effort";
  return "full";
}
function contextLabel(n) {
  if (!n) return "?";
  if (n >= 1_000_000) {
    const v = n / 1_000_000;
    return `${Number.isInteger(v) ? v : v.toFixed(3)}M`;
  }
  return `${n / 1_000}k`;
}

// Run probes
const models = FILTER ? catalog.filter((m) => m.id.startsWith(FILTER)) : catalog;
console.error(`\nProbing ${models.length} catalog models + ${EMBED_MODELS.length} embed models…\n`);

const llmProbes = models.map(async (spec) => {
  const fn = isClaude(spec.id) ? probeClaude : probeChat;
  const result = await fn(spec.id);
  const notes = [];
  if (REQUIRES_REASONING_EFFORT.has(spec.id)) notes.push("requires reasoning_effort");
  if (REGION_LOCKED.has(spec.id)) notes.push("region-locked");
  if (EXPECTED_DOWN.has(spec.id)) notes.push("suspended/preview");
  return {
    id: spec.id,
    type: isClaude(spec.id) ? "claude" : isOpenAI(spec.id) ? "openai" : "oss",
    ok: result.ok,
    liveModel: result.ok ? result.detail : null,
    errorMsg: result.ok ? null : result.detail,
    reasoning: reasoningLabel(spec),
    context: contextLabel(spec.contextWindow),
    maxTokens: contextLabel(spec.maxTokens),
    notes: notes.join("; ") || null,
  };
});

const embedProbes = EMBED_MODELS
  .filter((id) => !FILTER || id.startsWith(FILTER))
  .map(async (id) => {
    const result = await probeEmbed(id);
    return { id, type: "embed", ok: result.ok,
      liveModel: result.ok ? result.detail : null,
      errorMsg: result.ok ? null : result.detail,
      reasoning: "—", context: "—", maxTokens: "—", notes: null };
  });

const allResults = await Promise.all([...llmProbes, ...embedProbes]);

// Output
if (JSON_OUT) {
  console.log(JSON.stringify(allResults, null, 2));
  process.exit(allResults.every((r) => r.ok || r.notes) ? 0 : 1);
}

const groups = { claude: [], openai: [], oss: [], embed: [] };
for (const r of allResults) groups[r.type].push(r);

function row(r) {
  const s = r.ok ? "✅" : (EXPECTED_DOWN.has(r.id) || REGION_LOCKED.has(r.id) ? "⚠️ " : "❌");
  const detail = r.ok
    ? (r.notes ? `(${r.notes})` : r.liveModel !== r.id ? `→ ${r.liveModel}` : "")
    : (r.errorMsg ?? "");
  return `  ${s}  ${r.id.padEnd(44)} ${r.reasoning.padEnd(18)} ${r.context.padEnd(5)}  ${detail}`;
}

const groupLabels = {
  claude: "Claude — Messages API",
  openai: "OpenAI — Chat Completions",
  oss:    "Open-Source — Chat Completions",
  embed:  "Embedding",
};

let anyFail = false;
for (const [key, label] of Object.entries(groupLabels)) {
  const items = groups[key];
  if (!items.length) continue;
  console.log(`\n━━━ ${label} ${"━".repeat(Math.max(0, 48 - label.length))}`);
  console.log(`  ${"Model".padEnd(50)} ${"Reasoning".padEnd(18)} ${"Ctx".padEnd(5)}  Detail`);
  console.log(`  ${"─".repeat(90)}`);
  for (const r of items) {
    console.log(row(r));
    if (!r.ok && !EXPECTED_DOWN.has(r.id) && !REGION_LOCKED.has(r.id)) anyFail = true;
  }
}

const pass = allResults.filter((r) => r.ok).length;
const fail = allResults.filter((r) => !r.ok && !EXPECTED_DOWN.has(r.id) && !REGION_LOCKED.has(r.id)).length;
const warn = allResults.filter((r) => !r.ok && (EXPECTED_DOWN.has(r.id) || REGION_LOCKED.has(r.id))).length;
console.log(`\n  ${pass} ok  ${fail} failed  ${warn} expected-down/region-locked\n`);
process.exit(anyFail ? 1 : 0);
