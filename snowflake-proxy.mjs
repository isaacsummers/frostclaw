#!/usr/bin/env node

/**
 * Snowflake Cortex Unified Proxy
 *
 * Single proxy replacing both embed-proxy.mjs and cortex-proxy.mjs.
 * Handles all three Snowflake Cortex API surfaces on one port.
 *
 * Routes:
 *   POST /v1/embeddings             → Snowflake inference:embed  (with batching/coalescing)
 *   POST /api/v2/cortex/v1/messages  → Snowflake cortex/v1/messages  (Anthropic, with header normalisation)
 *   POST /api/v2/cortex/v1/chat/completions → Snowflake cortex/v1/chat/completions  (OpenAI-compat passthrough)
 *   GET  /health                      → {"status":"ok",...}
 *   *                                 → 404
 *
 * Auth translation (all routes):
 *   Client sends:  x-api-key: <anything> (or nothing)
 *   Proxy sends:   Authorization: Bearer ${SNOWFLAKE_CORTEX_API_KEY}
 *                  X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN
 *
 * Embedding batching/coalescing:
 *   Requests arriving within EMBED_COALESCE_MS (default 50ms) of each other
 *   are batched into a single Snowflake API call and fanned back to callers.
 *   EMBED_MAX_BATCH_TEXTS caps each batch for very large bursts.
 *
 * Environment (when run standalone):
 *   SNOWFLAKE_BASE_URL             - Snowflake account URL (required)
 *   SNOWFLAKE_CORTEX_API_KEY       - Programmatic Access Token (preferred)
 *   SNOWFLAKE_PAT                  - Alternative to SNOWFLAKE_CORTEX_API_KEY
 *   SNOWFLAKE_CORTEX_PROXY_PORT    - Port to listen on (default: 18790)
 *   EMBED_COALESCE_MS              - Batch window in ms (default: 50)
 *   EMBED_MAX_BATCH_TEXTS          - Max texts per batch (default: 64)
 */

import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

// ---------------------------------------------------------------------------
// undici Agent — long-lived dispatcher for outbound Snowflake streaming calls.
//
// bodyTimeout:0  prevents undici from dropping the connection mid-stream
//                during extended thinking phases where Snowflake emits no SSE
//                tokens for many seconds (default 300s is too short).
// keepAliveTimeout:95s must outlast the OpenClaw gateway-side undici Agent
//                (90s) so the proxy never closes a connection the gateway
//                still considers live, which would produce "terminated" errors.
// headersTimeout: 120s — non-streaming callers (e.g. graphiti-core) don't
//                receive response headers until Snowflake finishes the full
//                response. Complex structured-extraction prompts can take
//                30-60s, so 30s was too tight. 120s gives ample headroom.
// ---------------------------------------------------------------------------

let _snowflakeAgent = null;
// Node 24 ships undici 7 internally; openclaw bundles undici 8. Using an
// undici 8 Agent as a dispatcher with Node's globalThis.fetch (undici 7)
// causes "invalid content-length header" / "invalid onRequestStart method"
// errors due to API incompatibility. Fix: use the same undici module's own
// fetch() for upstream Snowflake calls so the Agent and fetch are always
// from the same version.
let _snowflakeFetch = globalThis.fetch;
try {
  const _require = createRequire(import.meta.url);
  // Prefer openclaw's bundled undici; fall back to the system package.
  let undici;
  try {
    undici = _require("/home/ubuntu/.npm-global/lib/node_modules/openclaw/node_modules/undici");
  } catch {
    undici = _require("undici");
  }
  _snowflakeAgent = new undici.Agent({
    headersTimeout: 120_000,
    bodyTimeout: 0,           // no timeout on response body stream
    keepAliveTimeout: 95_000, // outlast gateway-side Agent (90s)
    keepAliveMaxTimeout: 300_000,
  });
  // Use undici's own fetch so Agent and fetch share the same undici version.
  _snowflakeFetch = undici.fetch.bind(undici);
  console.log("[frostclaw-proxy] undici Agent configured (bodyTimeout=0, keepAlive=95s)");
} catch (e) {
  console.warn(`[frostclaw-proxy] undici not available, using default dispatcher: ${e.message}`);
}

// ---------------------------------------------------------------------------
// Shared constants (not config-dependent)
// ---------------------------------------------------------------------------

const DEFAULT_EMBED_MODEL = "snowflake-arctic-embed-m-v1.5";

const SUPPORTED_EMBED_MODELS = new Set([
  "snowflake-arctic-embed-m-v1.5",
  "snowflake-arctic-embed-m",
  "snowflake-arctic-embed-l-v2.0",
  "e5-base-v2",
  "multilingual-e5-large",
  "nv-embed-qa-4",
  "voyage-multimodal-3",
  "voyage-multilingual-2",
]);

// Headers to strip from client requests before forwarding (all lowercase)
const STRIP_HEADERS = new Set([
  "x-api-key",
  "host",
  "content-length",
  "transfer-encoding",
  "connection",
  "authorization",
  "x-snowflake-authorization-token-type",
]);

// ---------------------------------------------------------------------------
// response_format stripping + JSON prompt injection
// ---------------------------------------------------------------------------

const JSON_ONLY_SYSTEM_PROMPT =
  "You must respond with ONLY valid JSON. " +
  "Do not include any text before or after the JSON object. " +
  "Do not wrap the response in markdown code fences or backticks. " +
  "Do not add explanations, comments, or any prose. " +
  "Your entire response must be a single, complete, parseable JSON object.";

/**
 * Build the appropriate JSON system prompt based on the response_format type.
 * For json_schema, embeds the actual schema so the model knows the structure.
 * For json_object, returns the generic JSON-only instruction.
 *
 * @param {string} type - "json_object" or "json_schema"
 * @param {unknown} rf  - the original response_format value (before deletion)
 * @returns {string}
 */
function buildJsonPrompt(type, rf) {
  if (type === "json_schema") {
    const schema = rf?.json_schema?.schema;
    if (schema) {
      return (
        "You must respond with ONLY valid JSON conforming exactly to this schema:\n" +
        JSON.stringify(schema, null, 2) + "\n\n" +
        "Do not include any text before or after the JSON. " +
        "Do not wrap in markdown code fences or backticks. " +
        "No explanations, comments, or prose. " +
        "Your entire response must be a single, complete, parseable JSON object matching the schema above."
      );
    }
  }
  // json_object, or json_schema without a schema definition — use generic prompt.
  return JSON_ONLY_SYSTEM_PROMPT;
}

/**
 * Strip response_format from the body (Cortex rejects it for non-OpenAI models)
 * and inject a strong system prompt so the model still returns valid JSON.
 *
 * For json_schema, extracts the schema before stripping and embeds it in the
 * injected prompt so the model knows the exact structure expected.
 *
 * @param {Record<string,unknown>} body - parsed request body (mutated in place)
 * @returns {string|null} the stripped type ("json_object"/"json_schema") or null
 */
function stripResponseFormatAndInjectPrompt(body) {
  if (!("response_format" in body)) return null;
  const rf = body.response_format;
  const type = (rf && typeof rf === "object" && typeof rf.type === "string")
    ? rf.type : "json_object";

  // Only inject when the caller actually wanted JSON output.
  // response_format can also be { type: "text" }, in which case injecting a
  // JSON constraint would be actively wrong.
  const needsJson = type === "json_object" || type === "json_schema";

  // Extract schema (for json_schema) BEFORE deleting the field.
  const prompt = needsJson ? buildJsonPrompt(type, rf) : null;

  delete body.response_format;

  if (prompt && Array.isArray(body.messages)) {
    const sysIdx = body.messages.findIndex((m) => m && m.role === "system");
    if (sysIdx === -1) {
      body.messages = [{ role: "system", content: prompt }, ...body.messages];
    } else {
      const sys = { ...body.messages[sysIdx] };
      if (typeof sys.content === "string") {
        sys.content = prompt + "\n\n" + sys.content;
      } else if (Array.isArray(sys.content)) {
        sys.content = [{ type: "text", text: prompt }, ...sys.content];
      } else {
        sys.content = prompt;
      }
      const msgs = [...body.messages];
      msgs[sysIdx] = sys;
      body.messages = msgs;
    }
  }

  return type;
}

// ---------------------------------------------------------------------------
// Helpers (stateless, not config-dependent)
// ---------------------------------------------------------------------------

/**
 * Read the full request body as a string.
 * @param {import("node:http").IncomingMessage} req
 * @returns {Promise<string>}
 */
async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks).toString();
}

/**
 * Repair tool_use/tool_result pairing in an Anthropic Messages `messages` array
 * before it reaches Snowflake Cortex. Wire-format port of OpenClaw's
 * repairToolUseResultPairing (gateway-side `buildReplayPolicy`); since Honcho and
 * other Anthropic-SDK clients hit this proxy directly without OpenClaw's replay
 * pipeline, the repair must live here so the proxy is the single pre-Cortex
 * chokepoint.
 *
 * Cortex requires every assistant `tool_use` block to be matched by a
 * `tool_result` in the immediately-following user turn, in tool_use order, else:
 *   400 "Each 'toolUse' block must be accompanied with a matching 'toolResult' block."
 *
 * Guarantees: results reordered to tool_use order and consolidated into the user
 * turn right after the assistant; missing results synthesized (is_error
 * placeholder); orphan results (no matching prior tool_use) dropped; duplicate
 * results (same id) dropped; non-result content in the span preserved after.
 *
 * @param {any} messages - Anthropic Messages array (not mutated; new array on change)
 * @returns {{messages:any, changed:boolean}}
 */
function repairAnthropicToolPairing(messages) {
  if (!Array.isArray(messages)) return { messages, changed: false };

  const MISSING_TEXT = "[no tool result returned]";
  const blocksOf = (m) => (m && Array.isArray(m.content) ? m.content : null);
  const toolUsesOf = (m) => {
    if (!m || m.role !== "assistant") return [];
    const b = blocksOf(m);
    return b ? b.filter((x) => x && x.type === "tool_use" && typeof x.id === "string") : [];
  };
  const makeMissing = (id) => ({
    type: "tool_result",
    tool_use_id: id,
    content: MISSING_TEXT,
    is_error: true,
  });

  const out = [];
  let changed = false;
  const knownToolUseIds = new Set();

  for (let i = 0; i < messages.length; i += 1) {
    const msg = messages[i];
    const toolUses = toolUsesOf(msg);

    if (toolUses.length === 0) {
      // A user turn carrying tool_result blocks with no matching *known*
      // tool_use is an orphan span -> drop those blocks (keep other content).
      const b = blocksOf(msg);
      if (msg && msg.role === "user" && b && b.some((x) => x && x.type === "tool_result")) {
        const kept = b.filter((x) => !(x && x.type === "tool_result" && !knownToolUseIds.has(x.tool_use_id)));
        if (kept.length !== b.length) {
          changed = true;
          if (kept.length > 0) out.push({ ...msg, content: kept });
          continue; // drop the now-empty user message
        }
      }
      out.push(msg);
      continue;
    }

    const ids = toolUses.map((t) => t.id);
    ids.forEach((id) => knownToolUseIds.add(id));
    out.push(msg);

    // Gather tool_result blocks (first-seen wins) + non-result remainder from
    // the span up to the next assistant turn.
    const resultsById = new Map();
    const remainder = [];
    let j = i + 1;
    for (; j < messages.length; j += 1) {
      const next = messages[j];
      if (next && next.role === "assistant") break;
      const nb = blocksOf(next);
      if (next && next.role === "user" && nb) {
        for (const blk of nb) {
          if (blk && blk.type === "tool_result" && typeof blk.tool_use_id === "string") {
            if (!resultsById.has(blk.tool_use_id)) resultsById.set(blk.tool_use_id, blk);
            else changed = true; // duplicate -> dropped
          } else {
            remainder.push(blk);
          }
        }
      } else if (next && next.role === "user" && typeof next.content === "string") {
        remainder.push({ type: "text", text: next.content });
      } else if (next) {
        break;
      }
    }

    const idSet = new Set(ids);
    const resultBlocks = [];
    for (const id of ids) {
      const existing = resultsById.get(id);
      if (existing) resultBlocks.push(existing);
      else { resultBlocks.push(makeMissing(id)); changed = true; }
    }
    for (const [rid] of resultsById) {
      if (!idSet.has(rid)) changed = true; // result for a different turn -> dropped
    }
    const presentInIdOrder = ids.filter((id) => resultsById.has(id));
    const seenOrder = [...resultsById.keys()].filter((id) => idSet.has(id));
    if (seenOrder.some((id, k) => id !== presentInIdOrder[k])) changed = true; // reordered
    if (remainder.length > 0) changed = true;

    out.push({ role: "user", content: resultBlocks });
    if (remainder.length > 0) out.push({ role: "user", content: remainder });

    i = j - 1; // skip consumed span
  }

  return { messages: changed ? out : messages, changed };
}



// ---------------------------------------------------------------------------
// Factory — createProxyServer(config)
// ---------------------------------------------------------------------------

/**
 * Create a Snowflake Cortex proxy server.
 *
 * @param {{
 *   baseUrl: string,
 *   pat: string,
 *   port?: number,
 *   coalesceMs?: number,
 *   maxBatchTexts?: number,
 * }} config
 * @returns {{ server: import("node:http").Server, close: () => Promise<void> }}
 */
export function createProxyServer(config) {
  const {
    baseUrl: rawBaseUrl,
    pat,
    port = 0,
    coalesceMs = 50,
    maxBatchTexts = 64,
  } = config;

  const baseUrl = rawBaseUrl.replace(/\/$/, "");

  // ── Embedding queue state (per-instance) ──────────────────────────────────
  let pendingQueue = [];
  let flushTimer = null;

  async function flushQueue() {
    flushTimer = null;
    if (pendingQueue.length === 0) return;

    const batch = pendingQueue;
    pendingQueue = [];

    // Group by model
    const byModel = new Map();
    for (const item of batch) {
      const list = byModel.get(item.model) ?? [];
      list.push(item);
      byModel.set(item.model, list);
    }

    for (const [model, items] of byModel) {
      const allTexts = [];
      const mapping = [];
      for (let i = 0; i < items.length; i++) {
        for (let j = 0; j < items[i].texts.length; j++) {
          allTexts.push(items[i].texts[j]);
          mapping.push({ itemIndex: i, textIndex: j });
        }
      }

      const itemResults = items.map((item) =>
        new Array(item.texts.length).fill(null),
      );

      for (let offset = 0; offset < allTexts.length; offset += maxBatchTexts) {
        const chunkTexts = allTexts.slice(offset, offset + maxBatchTexts);
        const chunkMapping = mapping.slice(offset, offset + maxBatchTexts);

        let sfJson;
        try {
          const sfRes = await _snowflakeFetch(
            `${baseUrl}/api/v2/cortex/inference:embed`,
            {
              method: "POST",
              headers: {
                Authorization: `Bearer ${pat}`,
                "X-Snowflake-Authorization-Token-Type":
                  "PROGRAMMATIC_ACCESS_TOKEN",
                "Content-Type": "application/json",
                Accept: "application/json",
              },
              body: JSON.stringify({ text: chunkTexts, model }),
            },
          );

          if (!sfRes.ok) {
            const errBody = await sfRes.text().catch(() => "");
            const err = new Error(`Snowflake error ${sfRes.status}: ${errBody}`);
            const failedItemIndices = new Set(
              chunkMapping.map((m) => m.itemIndex),
            );
            for (const idx of failedItemIndices) items[idx].reject(err);
            continue;
          }

          sfJson = await sfRes.json();
        } catch (err) {
          const failedItemIndices = new Set(
            chunkMapping.map((m) => m.itemIndex),
          );
          for (const idx of failedItemIndices) items[idx].reject(err);
          continue;
        }

        for (let i = 0; i < sfJson.data.length; i++) {
          const raw = sfJson.data[i].embedding;
          const vec = Array.isArray(raw[0]) ? raw[0] : raw;
          const { itemIndex, textIndex } = chunkMapping[i];
          itemResults[itemIndex][textIndex] = vec;
        }
      }

      for (let i = 0; i < items.length; i++) {
        if (itemResults[i].every((v) => v !== null)) {
          items[i].resolve(itemResults[i]);
        } else {
          items[i].reject(
            new Error("Incomplete embedding results from Snowflake"),
          );
        }
      }
    }
  }

  function scheduleFlush() {
    const totalTexts = pendingQueue.reduce((s, it) => s + it.texts.length, 0);
    if (totalTexts >= maxBatchTexts) {
      if (flushTimer !== null) {
        clearTimeout(flushTimer);
        flushTimer = null;
      }
      setImmediate(flushQueue);
      return;
    }
    if (flushTimer === null) {
      flushTimer = setTimeout(flushQueue, coalesceMs);
    }
  }

  function embedTexts(texts, model) {
    return new Promise((resolve, reject) => {
      pendingQueue.push({ texts, model, resolve, reject });
      scheduleFlush();
    });
  }

  // ── Header helpers ─────────────────────────────────────────────────────────

  function buildForwardHeaders(incomingHeaders, bodyStr) {
    const headers = {};
    for (const [key, value] of Object.entries(incomingHeaders)) {
      const lower = key.toLowerCase();
      if (!STRIP_HEADERS.has(lower)) {
        headers[lower] = value;
      }
    }
    headers["authorization"] = `Bearer ${pat}`;
    headers["x-snowflake-authorization-token-type"] =
      "PROGRAMMATIC_ACCESS_TOKEN";
    headers["content-type"] = "application/json";
    headers["content-length"] = Buffer.byteLength(bodyStr).toString();
    return headers;
  }

  // ── Proxy helper ───────────────────────────────────────────────────────────

  async function proxyRequest(req, res, targetUrl, extraHeaders = {}) {
    const bodyStr = await readBody(req);

    let isStreaming = false;
    try {
      isStreaming = JSON.parse(bodyStr).stream === true;
    } catch {
      // non-JSON or unparseable — treat as non-streaming
    }

    const forwardHeaders = buildForwardHeaders(req.headers, bodyStr);
    for (const [k, v] of Object.entries(extraHeaders)) {
      if (!forwardHeaders[k]) forwardHeaders[k] = v;
    }

    const _t0 = Date.now();
    const fetchOpts = {
      method: "POST",
      headers: forwardHeaders,
      body: bodyStr,
    };
    // Attach the long-lived undici Agent for streaming requests so Snowflake's
    // extended thinking phases (no SSE tokens for many seconds) don't trigger
    // a bodyTimeout disconnect on the outbound Snowflake connection.
    if (isStreaming && _snowflakeAgent) {
      fetchOpts.dispatcher = _snowflakeAgent;
    }
    const sfRes = await _snowflakeFetch(targetUrl, fetchOpts);
    console.log(`[frostclaw-proxy] upstream ${targetUrl.split('/').pop()} → ${sfRes.status} (${Date.now() - _t0}ms, reqBytes=${bodyStr.length})`);

    // If upstream errored with 4xx/5xx on a "streaming" request, the body is
    // a small JSON error blob, not SSE. Buffer it so we can both log the
    // error detail and forward a properly-shaped response to the client.
    if (isStreaming && !sfRes.ok) {
      const errBody = await sfRes.text().catch(() => "");
      console.error(
        `[frostclaw-proxy] upstream error ${sfRes.status} on streaming request to ${targetUrl.split('/').pop()}: ${errBody.slice(0, 1000)}`,
      );
      const responseHeaders = { "content-type": "application/json" };
      for (const [key, value] of sfRes.headers.entries()) {
        const lower = key.toLowerCase();
        if (
          lower !== "content-length" &&
          lower !== "transfer-encoding" &&
          lower !== "content-encoding" &&
          lower !== "connection"
        ) {
          responseHeaders[lower] = value;
        }
      }
      responseHeaders["content-length"] = Buffer.byteLength(errBody).toString();
      res.writeHead(sfRes.status, responseHeaders);
      res.end(errBody);
      return;
    }

    if (isStreaming) {
      const responseHeaders = {
        "content-type": sfRes.headers.get("content-type") || "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      };
      for (const [key, value] of sfRes.headers.entries()) {
        const lower = key.toLowerCase();
        if (
          lower !== "content-length" &&
          lower !== "transfer-encoding" &&
          lower !== "connection" &&
          !responseHeaders[lower]
        ) {
          responseHeaders[lower] = value;
        }
      }
      res.writeHead(sfRes.status, responseHeaders);

      if (!sfRes.body) {
        res.end();
        return;
      }

      const reader = sfRes.body.getReader();
      // Cancel upstream reader if client disconnects mid-stream to avoid
      // leaking open Snowflake connections.
      const cancelReader = () => { reader.cancel().catch(() => {}); };
      req.socket?.once("close", cancelReader);
      req.socket?.once("error", cancelReader);
      const pump = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              res.end();
              break;
            }
            if (res.destroyed) {
              reader.cancel().catch(() => {});
              break;
            }
            const ok = res.write(value);
            if (!ok) {
              await new Promise((resolve) => res.once("drain", resolve));
            }
          }
        } finally {
          req.socket?.off("close", cancelReader);
          req.socket?.off("error", cancelReader);
        }
      };
      pump().catch((err) => {
        console.error("[frostclaw] Stream pump error:", err.message);
        if (!res.destroyed) res.end();
      });
    } else {
      const responseBody = await sfRes.text();
      const responseHeaders = { "content-type": "application/json" };
      for (const [key, value] of sfRes.headers.entries()) {
        const lower = key.toLowerCase();
        if (
          lower !== "content-length" &&
          lower !== "transfer-encoding" &&
          lower !== "content-encoding" &&
          lower !== "connection"
        ) {
          responseHeaders[lower] = value;
        }
      }
      responseHeaders["content-length"] =
        Buffer.byteLength(responseBody).toString();
      res.writeHead(sfRes.status, responseHeaders);
      res.end(responseBody);
    }
  }

  // ── HTTP server ────────────────────────────────────────────────────────────

  const server = createServer(async (req, res) => {
    console.log(`[frostclaw-proxy] ${req.method} ${req.url} pid=${req.socket.remoteAddress}:${req.socket.remotePort}`);
    // Health check
    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          status: "ok",
          provider: "snowflake-cortex",
          routes: ["embeddings", "v1/chat/completions", "api/v2/cortex/v1/messages", "api/v2/cortex/v1/chat/completions", "api/v2/cortex/inference:embed"],
          embedModels: [...SUPPORTED_EMBED_MODELS],
        }),
      );
      return;
    }

    // ── POST /v1/embeddings ────────────────────────────────────────────────
    if (req.method === "POST" && req.url === "/v1/embeddings") {
      try {
        const bodyStr = await readBody(req);
        const body = JSON.parse(bodyStr);

        const input = Array.isArray(body.input) ? body.input : [body.input];
        const model = body.model || DEFAULT_EMBED_MODEL;

        if (body.model && !SUPPORTED_EMBED_MODELS.has(body.model)) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: {
                message: `Unsupported embedding model: "${body.model}". Supported models: ${[...SUPPORTED_EMBED_MODELS].join(", ")}`,
                type: "invalid_request_error",
              },
            }),
          );
          return;
        }

        const embeddings = await embedTexts(input, model);

        const data = embeddings.map((vec, idx) => ({
          object: "embedding",
          embedding: vec,
          index: idx,
        }));

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            object: "list",
            data,
            model,
            usage: { prompt_tokens: 0, total_tokens: 0 },
          }),
        );
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({ error: { message: err.message, type: "proxy_error" } }),
        );
      }
      return;
    }

    // ── POST /api/v2/cortex/v1/messages ────────────────────────────────────
    if (req.method === "POST" && req.url === "/api/v2/cortex/v1/messages") {
      try {
        const rawBody = await readBody(req);
        let body;
        try {
          body = JSON.parse(rawBody);
        } catch {
          body = null;
        }

        // Two pre-Cortex repairs run here so the proxy is the single chokepoint
        // for everything that must happen before a request reaches Snowflake
        // Cortex — any Anthropic-SDK client (Honcho's dialectic, etc.) is made
        // safe regardless of whether it ran OpenClaw's gateway-side replay
        // pipeline.
        //
        // (a) eager_input_streaming scrub: Cortex's strict validator rejects
        //     tools[].custom.eager_input_streaming with
        //       400 ...eager_input_streaming: Extra inputs are not permitted
        //     Drop it; if `custom` becomes empty, drop the key too.
        // (b) tool_use/tool_result pairing repair: Cortex requires every
        //     assistant `tool_use` block to be matched by a `tool_result` in the
        //     immediately-following user turn (400 "Each 'toolUse' block must be
        //     accompanied with a matching 'toolResult' block"). This is the
        //     wire-format port of OpenClaw's repairToolUseResultPairing.
        let forwardBody = rawBody;
        if (body !== null && typeof body === "object") {
          let mutated = false;
          if (Array.isArray(body.tools)) {
            for (const tool of body.tools) {
              if (!tool || typeof tool !== "object") continue;
              const custom = tool.custom;
              if (!custom || typeof custom !== "object") continue;
              if (!("eager_input_streaming" in custom)) continue;
              delete custom.eager_input_streaming;
              if (Object.keys(custom).length === 0) delete tool.custom;
              mutated = true;
            }
          }
          const repaired = repairAnthropicToolPairing(body.messages);
          if (repaired.changed) {
            body.messages = repaired.messages;
            mutated = true;
            console.log("[frostclaw-proxy] messages: repaired tool_use/tool_result pairing");
          }
          if (mutated) forwardBody = JSON.stringify(body);
        }

        const syntheticReq = {
          headers: req.headers,
          [Symbol.asyncIterator]: async function* () {
            yield Buffer.from(forwardBody);
          },
        };

        await proxyRequest(
          syntheticReq,
          res,
          `${baseUrl}/api/v2/cortex/v1/messages`,
          { "anthropic-version": "2023-06-01" },
        );
      } catch (err) {
        console.error("[frostclaw] Messages proxy error:", err.message);
        if (!res.headersSent) {
          res.writeHead(500, { "Content-Type": "application/json" });
        }
        res.end(
          JSON.stringify({ error: { message: err.message, type: "proxy_error" } }),
        );
      }
      return;
    }

    // ── POST /api/v2/cortex/v1/chat/completions ──────────────────────────
    if (req.method === "POST" && req.url === "/api/v2/cortex/v1/chat/completions") {
      try {
        const rawBody = await readBody(req);
        let body;
        try {
          body = JSON.parse(rawBody);
        } catch {
          body = null;
        }

        // Normalise request body for all models:
        //   - strip "snowflake-cortex/" and "openai-" provider prefixes
        //   - rename max_tokens → max_completion_tokens (Snowflake chat completions
        //     uses the OpenAI v1 key; max_tokens is the Anthropic / legacy name)
        //
        // NOTE on tool calling: Snowflake's chat/completions endpoint accepts
        // OpenAI-format requests but, for Claude models, converts the tool history
        // into native Anthropic toolUse/toolResult blocks internally — and that
        // conversion mis-pairs parallel tool calls (400 "Each 'toolUse' block must
        // be accompanied with a matching 'toolResult' block"). Tool-using clients
        // should therefore use the Anthropic /api/v2/cortex/v1/messages surface
        // instead (Honcho's dialectic does). This surface is left as a plain
        // passthrough; the only OpenAI-surface callers here use structured output
        // (graphiti-core, Honcho deriver), which carry no tools.
        let forwardBody = rawBody;
        if (body !== null && typeof body === "object") {
          const originalModel = body.model;
          if (typeof body.model === "string") {
            body.model = body.model
              .replace(/^snowflake-cortex\//, "")
              .replace(/^openai\//, ""); // strip "openai/" provider prefix but keep "openai-" in model name
            // e.g. "openai/openai-gpt-4.1" → "openai-gpt-4.1" (correct Cortex name)
          }
          const isOpenAIModel = typeof body.model === "string" && body.model.startsWith("openai-");
          const promptChars = (body.messages ?? []).reduce((n, m) => n + (typeof m.content === "string" ? m.content.length : JSON.stringify(m.content).length), 0);
          console.log(`[frostclaw-proxy] chat/completions model: ${originalModel} → ${body.model} isOpenAI=${isOpenAIModel} promptChars=${promptChars}`);
          if ("max_tokens" in body) {
            body.max_completion_tokens = body.max_tokens;
            delete body.max_tokens;
          }
          // Strip response_format for non-OpenAI models — Snowflake Cortex rejects
          // both json_object and json_schema for Claude/Llama/Mistral models.
          // OpenAI models (openai-gpt-4.1 etc.) support response_format natively
          // so we pass it through for those. When stripping, inject a strong
          // system prompt so the model still returns valid JSON.
          if ("response_format" in body && !isOpenAIModel) {
            const stripped = stripResponseFormatAndInjectPrompt(body);
            console.log(`[frostclaw-proxy] stripped response_format type=${stripped} and injected JSON system prompt`);
          }
          forwardBody = JSON.stringify(body);
        }

        const syntheticReq = {
          headers: req.headers,
          [Symbol.asyncIterator]: async function* () {
            yield Buffer.from(forwardBody);
          },
        };

        await proxyRequest(
          syntheticReq,
          res,
          `${baseUrl}/api/v2/cortex/v1/chat/completions`,
        );
      } catch (err) {
        console.error("[frostclaw] Chat completions proxy error:", err.message);
        if (!res.headersSent) {
          res.writeHead(500, { "Content-Type": "application/json" });
        }
        res.end(
          JSON.stringify({ error: { message: err.message, type: "proxy_error" } }),
        );
      }
      return;
    }

    // ── POST /v1/chat/completions ──────────────────────────────────────────
    // OpenAI-compat alias — graphiti-core and other OpenAI-SDK clients append
    // /chat/completions to whatever base_url they're given. This aliases
    // /v1/chat/completions → the Snowflake cortex endpoint so they work with
    // OPENAI_API_URL=http://frostclaw-proxy:18790/v1
    if (req.method === "POST" && req.url === "/v1/chat/completions") {
      req.url = "/api/v2/cortex/v1/chat/completions";
      // Re-dispatch to the handler above by falling through — but since we
      // already exited that block, we inline the same logic here.
      try {
        const rawBody = await readBody(req);
        let body;
        try { body = JSON.parse(rawBody); } catch { body = null; }
        let forwardBody = rawBody;
        if (body !== null && typeof body === "object") {
          const originalModel = body.model;
          if (typeof body.model === "string") {
            body.model = body.model
              .replace(/^snowflake-cortex\//, "")
              .replace(/^openai\//, ""); // strip "openai/" provider prefix but keep "openai-" in model name
            // e.g. "openai/openai-gpt-4.1" → "openai-gpt-4.1" (correct Cortex name)
          }
          const isOpenAIModel = typeof body.model === "string" && body.model.startsWith("openai-");
          const promptChars = (body.messages ?? []).reduce((n, m) => n + (typeof m.content === "string" ? m.content.length : JSON.stringify(m.content).length), 0);
          console.log(`[frostclaw-proxy] chat/completions (v1 alias) model: ${originalModel} → ${body.model} isOpenAI=${isOpenAIModel} promptChars=${promptChars}`);
          if ("max_tokens" in body) {
            body.max_completion_tokens = body.max_tokens;
            delete body.max_tokens;
          }
          // Strip response_format for non-OpenAI models — Cortex rejects it for Claude/Llama/Mistral.
          // OpenAI models (openai-gpt-4.1) support it natively, so pass it through for those.
          // When stripping, inject a strong system prompt so the model still returns valid JSON.
          if ("response_format" in body && !isOpenAIModel) {
            const stripped = stripResponseFormatAndInjectPrompt(body);
            console.log(`[frostclaw-proxy] stripped response_format (v1 alias) type=${stripped} and injected JSON system prompt`);
          }
          forwardBody = JSON.stringify(body);
        }
        const syntheticReq = {
          headers: req.headers,
          [Symbol.asyncIterator]: async function* () { yield Buffer.from(forwardBody); },
        };
        await proxyRequest(syntheticReq, res, `${baseUrl}/api/v2/cortex/v1/chat/completions`);
      } catch (err) {
        console.error("[frostclaw] Chat completions (v1 alias) proxy error:", err.message);
        if (!res.headersSent) res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: err.message, type: "proxy_error" } }));
      }
      return;
    }

    // ── POST /api/v2/cortex/inference:embed ───────────────────────────────
    // Native passthrough for frostclaw's internal embed adapter. The body is
    // already in Snowflake's native shape ({ text, model }) — no translation
    // needed. Auth headers are injected by proxyRequest via buildForwardHeaders.
    if (req.method === "POST" && req.url === "/api/v2/cortex/inference:embed") {
      try {
        await proxyRequest(
          req,
          res,
          `${baseUrl}/api/v2/cortex/inference:embed`,
        );
      } catch (err) {
        console.error("[frostclaw] inference:embed proxy error:", err.message);
        if (!res.headersSent) {
          res.writeHead(500, { "Content-Type": "application/json" });
        }
        res.end(
          JSON.stringify({ error: { message: err.message, type: "proxy_error" } }),
        );
      }
      return;
    }

    // ── Catch-all 404 ──────────────────────────────────────────────────────
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        error:
          "Not found. Available routes: POST /v1/embeddings, POST /v1/chat/completions, POST /api/v2/cortex/v1/messages, POST /api/v2/cortex/v1/chat/completions, POST /api/v2/cortex/inference:embed, GET /health",
      }),
    );
  });

  // keepAliveTimeout must outlast both:
  //   (a) undici's 60s idle window — avoids "terminated" errors on keep-alive
  //       connection reuse from callers using undici/OpenAI SDK.
  //   (b) the OpenClaw gateway-side undici Agent keepAliveTimeout (90s) — if
  //       the proxy closes a connection before the gateway expects, the gateway
  //       gets a "terminated" error on the next reuse attempt.
  // 95s is the minimum safe value that outlasts both windows.
  server.keepAliveTimeout = 95_000;
  server.headersTimeout = 96_000;
  return {
    server,
    close() {
      return new Promise((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

// ---------------------------------------------------------------------------
// Standalone entry point — reads from env, starts server on configured port
// ---------------------------------------------------------------------------

// import.meta.main is a Deno/Bun convention — Node.js doesn't set it.
// Use process.argv[1] comparison instead (reliable on all Node versions).
const isMain = process.argv[1] && (
  process.argv[1] === fileURLToPath(import.meta.url) ||
  process.argv[1].includes("snowflake-proxy")
);
if (isMain) {
  const baseUrl = (process.env.SNOWFLAKE_BASE_URL || "").replace(/\/$/, "");
  const pat =
    process.env.SNOWFLAKE_CORTEX_API_KEY || process.env.SNOWFLAKE_PAT || "";

  if (!baseUrl || !pat) {
    console.error(
      "[frostclaw] Missing SNOWFLAKE_BASE_URL or SNOWFLAKE_CORTEX_API_KEY/SNOWFLAKE_PAT",
    );
    process.exit(1);
  }

  const port = parseInt(process.env.SNOWFLAKE_CORTEX_PROXY_PORT || "18790");
  const coalesceMs = parseInt(process.env.EMBED_COALESCE_MS || "50");
  const maxBatchTexts = parseInt(process.env.EMBED_MAX_BATCH_TEXTS || "64");

  const { server } = createProxyServer({ baseUrl, pat, port, coalesceMs, maxBatchTexts });
  server.listen(port, "127.0.0.1", () => {
    console.log(
      `[frostclaw] Snowflake unified proxy listening on http://127.0.0.1:${port} (coalesce=${coalesceMs}ms, maxBatch=${maxBatchTexts})`,
    );
  });
}
