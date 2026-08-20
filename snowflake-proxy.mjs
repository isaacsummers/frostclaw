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
 *   PROXY_MAX_RETRIES              - Max retries on rate-limit / transient errors (default: 3)
 *   PROXY_RETRY_BASE_DELAY         - Base delay in seconds for exponential backoff (default: 1.0)
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

// ---------------------------------------------------------------------------
// Retry config — reads from env at call time so changes take effect without
// restarting the proxy.
// ---------------------------------------------------------------------------

function getMaxRetries() {
  return Math.max(0, parseInt(process.env.PROXY_MAX_RETRIES ?? "3", 10) || 0);
}

function getRetryBaseDelayMs() {
  return Math.max(0, parseFloat(process.env.PROXY_RETRY_BASE_DELAY ?? "1.0") * 1000);
}

/**
 * Returns the wait time in ms before the next retry attempt.
 * Prefers the Retry-After response header (in seconds) when present;
 * falls back to exponential backoff: baseDelayMs * 2^attempt.
 *
 * @param {Headers} headers - Response headers from the upstream response
 * @param {number}  attempt - Zero-based attempt index (0 = first retry)
 * @param {number}  baseDelayMs - Base delay in ms
 * @returns {number} Wait time in ms
 */
function getRetryWaitMs(headers, attempt, baseDelayMs) {
  const retryAfter = headers.get("retry-after");
  if (retryAfter !== null && retryAfter !== undefined) {
    const seconds = parseFloat(retryAfter);
    if (!Number.isNaN(seconds) && seconds >= 0) {
      return seconds * 1000;
    }
  }
  return baseDelayMs * Math.pow(2, attempt);
}

/**
 * Returns true when the upstream HTTP status code is retryable.
 * Mirrors the logic in SnowClaw's retry.py and the plugin's fetch interceptor.
 *
 * @param {number} status
 * @param {string} body - Error response body text (used to detect throttled 400s)
 * @returns {boolean}
 */
function isRetryableStatus(status, body) {
  if (status === 429) return true;          // rate limited
  if (status === 503) return true;          // inference timeout
  if (status === 400 && body.toLowerCase().includes("throttled")) return true;
  // Snowflake connection-reset errors (TCP drop mid-stream or GOAWAY).
  // Error code 392606 = java.io.IOException: Connection reset by peer.
  // These are transient infrastructure errors and safe to retry.
  if (status === 500) {
    const b = body.toLowerCase();
    if (b.includes("392606") || b.includes("connection reset") || b.includes("goaway")) return true;
  }
  return false;
}

/**
 * Unwraps a fetch()-thrown error's cause chain into a single diagnostic
 * string. undici's top-level message is almost always the generic
 * "terminated" or "fetch failed" — the *actual* reason (socket reset vs
 * body timeout vs DNS vs TLS) lives in err.cause (and sometimes
 * err.cause.cause). Logging only err.message hides exactly the detail
 * needed to tell "Snowflake reset the connection" apart from "our own
 * client/timeout config did something wrong".
 *
 * @param {Error} err
 * @returns {string}
 */
function describeFetchError(err) {
  if (!err) return String(err);
  const parts = [err.message || String(err)];
  let cause = err.cause;
  let depth = 0;
  while (cause && depth < 3) {
    const code = cause.code ? `${cause.code}: ` : "";
    parts.push(`cause[${depth}]=${code}${cause.message || cause}`);
    cause = cause.cause;
    depth++;
  }
  return parts.join(" | ");
}

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
 /**
 * Inject a hardened JSON system prompt into the messages array when the
 * caller is requesting JSON output via response_format.
 *
 * Always injected regardless of model type — for OpenAI models this also
 * satisfies Snowflake Cortex's requirement that the messages contain the word
 * "json" when response_format is set. For non-OpenAI models, the field will
 * be stripped separately (Cortex rejects it), so the prompt is the only
 * structural-output constraint.
 *
 * For json_schema, extracts the schema before the field is deleted and embeds
 * it in the prompt so the model knows the exact structure expected.
 *
 * @param {Record<string,unknown>} body - parsed request body (mutated in place)
 * @returns {string|null} the format type ("json_object"/"json_schema") or null
 */
function injectJsonPromptFromResponseFormat(body) {
  if (!("response_format" in body)) return null;
  const rf = body.response_format;
  const type = (rf && typeof rf === "object" && typeof rf.type === "string")
    ? rf.type : "json_object";

  // Only inject when the caller actually wanted JSON output.
  const needsJson = type === "json_object" || type === "json_schema";
  if (!needsJson) return type;

  // Extract schema (for json_schema) while we still have the field.
  const prompt = buildJsonPrompt(type, rf);

  if (Array.isArray(body.messages)) {
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

/**
 * Strip response_format from the body and inject a hardened JSON prompt.
 *
 * Used for non-OpenAI models where Snowflake Cortex rejects response_format.
 * Injection is handled first via injectJsonPromptFromResponseFormat so the
 * schema is captured before the field is deleted.
 *
 * @param {Record<string,unknown>} body - parsed request body (mutated in place)
 * @returns {string|null} the stripped type or null
 */
function stripResponseFormatAndInjectPrompt(body) {
  const type = injectJsonPromptFromResponseFormat(body);
  if (type !== null) delete body.response_format;
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
        const embedMaxRetries = getMaxRetries();
        const embedBaseDelayMs = getRetryBaseDelayMs();
        let embedRetryCount = 0;
        let embedFailed = false;
        let embedFailErr = null;

        for (let embedAttempt = 0; embedAttempt <= embedMaxRetries; embedAttempt++) {
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
                ..._snowflakeAgent ? { dispatcher: _snowflakeAgent } : {},
              },
            );

            if (!sfRes.ok) {
              const errBody = await sfRes.text().catch(() => "");
              if (embedAttempt < embedMaxRetries && isRetryableStatus(sfRes.status, errBody)) {
                const waitMs = getRetryWaitMs(sfRes.headers, embedAttempt, embedBaseDelayMs);
                console.warn(`[frostclaw-proxy] retryable ${sfRes.status} from inference:embed — retry ${embedAttempt + 1}/${embedMaxRetries} in ${(waitMs / 1000).toFixed(1)}s`);
                await new Promise((resolve) => setTimeout(resolve, waitMs));
                embedRetryCount++;
                continue;
              }
              embedFailed = true;
              embedFailErr = new Error(`Snowflake error ${sfRes.status}: ${errBody}`);
              break;
            }

            sfJson = await sfRes.json();
            break;
          } catch (err) {
            // Connection reset before headers arrived (392606/GOAWAY/
            // "terminated") — this path previously had zero retry coverage,
            // unlike the streaming and now-fixed non-streaming proxyRequest
            // paths. Same root cause, same fix: retry on a fresh connection.
            const detail = describeFetchError(err);
            if (embedAttempt < embedMaxRetries) {
              const waitMs = getRetryWaitMs(new Headers(), embedAttempt, embedBaseDelayMs);
              console.warn(`[frostclaw-proxy] fetch error on inference:embed (${detail}) — retry ${embedAttempt + 1}/${embedMaxRetries} in ${(waitMs / 1000).toFixed(1)}s`);
              await new Promise((resolve) => setTimeout(resolve, waitMs));
              embedRetryCount++;
              continue;
            }
            console.error(`[frostclaw-proxy] fetch error on inference:embed after ${embedRetryCount} retries: ${detail}`);
            embedFailed = true;
            embedFailErr = err;
            break;
          }
        }

        if (embedFailed) {
          const failedItemIndices = new Set(
            chunkMapping.map((m) => m.itemIndex),
          );
          for (const idx of failedItemIndices) items[idx].reject(embedFailErr);
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

  // ── Streaming helper ────────────────────────────────────────────────────────
  //
  // Snowflake Cortex's backend intermittently resets the underlying TCP
  // connection mid-response (java.io.IOException: Connection reset by peer /
  // HTTP2 GOAWAY — see isRetryableStatus()'s 392606 handling). When that
  // reset happens *before* headers are sent, fetch() surfaces it as a
  // retryable HTTP error and the loop above already handles it. When it
  // happens *after* the SSE stream has started, undici's reader.read() throws
  // "terminated" instead — previously unretried, which is what surfaced to
  // OpenClaw as "Anthropic stream ended before message_stop" / "LLM request
  // failed."
  //
  // Fix: buffer the whole SSE response in memory instead of forwarding bytes
  // to the client as they arrive. That means nothing has been sent to the
  // real client yet if the connection drops, so we can transparently retry
  // the entire request — the same way the non-streaming path already retries
  // — and only flush the buffered bytes once we've confirmed a clean
  // terminal event. This route serves two SSE dialects with different
  // terminal markers: Anthropic Messages (message_stop) and OpenAI-compat
  // chat/completions ([DONE]) — see isAnthropicFormat below. Trade-off: the
  // client no longer sees token-by-token latency on a request that needed a
  // retry (it gets the full reply in one shot instead), but never sees a
  // truncated stream.
  async function proxyStreamingRequest(req, res, targetUrl, forwardHeaders, bodyStr, maxRetries, baseDelayMs, routeName) {
    // Anthropic Messages SSE terminates with a `message_stop` event.
    // OpenAI-compat chat/completions SSE terminates with `data: [DONE]`.
    // Using the wrong terminal marker means every completions stream gets
    // misdetected as "dropped mid-stream" and needlessly retried/502'd.
    const isAnthropicFormat = targetUrl.includes("/v1/messages");
    let retryCount = 0;
    let clientAborted = false;
    const onClientAbort = () => { clientAborted = true; };
    req.socket?.once("close", onClientAbort);
    req.socket?.once("error", onClientAbort);

    const sendJsonError = (status, message) => {
      if (res.headersSent || res.destroyed) { res.end(); return; }
      const body = JSON.stringify({ error: { type: "upstream_stream_error", message } });
      res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(body).toString() });
      res.end(body);
    };

    try {
      for (let attempt = 0; attempt <= maxRetries; attempt++) {
        if (clientAborted) return;
        const _t0 = Date.now();
        const fetchOpts = { method: "POST", headers: forwardHeaders, body: bodyStr };
        if (_snowflakeAgent) fetchOpts.dispatcher = _snowflakeAgent;

        let sfRes;
        try {
          sfRes = await _snowflakeFetch(targetUrl, fetchOpts);
        } catch (err) {
          const detail = describeFetchError(err);
          if (attempt < maxRetries) {
            const waitMs = getRetryWaitMs(new Headers(), attempt, baseDelayMs);
            console.warn(`[frostclaw-proxy] fetch error on streaming ${routeName} (${detail}) — retry ${attempt + 1}/${maxRetries} in ${(waitMs / 1000).toFixed(1)}s`);
            await new Promise((resolve) => setTimeout(resolve, waitMs));
            retryCount++;
            continue;
          }
          console.error(`[frostclaw-proxy] fetch error on streaming ${routeName} after ${retryCount} retries: ${detail}`);
          sendJsonError(502, `Snowflake fetch failed after ${retryCount} retries: ${err.message}`);
          return;
        }
        console.log(`[frostclaw-proxy] upstream ${routeName} → ${sfRes.status} (${Date.now() - _t0}ms, reqBytes=${bodyStr.length}${retryCount > 0 ? `, retry=${retryCount}` : ""})`);

        if (!sfRes.ok) {
          const errBody = await sfRes.text().catch(() => "");
          if (attempt < maxRetries && isRetryableStatus(sfRes.status, errBody)) {
            const waitMs = getRetryWaitMs(sfRes.headers, attempt, baseDelayMs);
            console.warn(`[frostclaw-proxy] retryable ${sfRes.status} from ${routeName} — waiting ${(waitMs / 1000).toFixed(1)}s before retry ${attempt + 1}/${maxRetries} (Retry-After: ${sfRes.headers.get("retry-after") ?? "none"})`);
            await new Promise((resolve) => setTimeout(resolve, waitMs));
            retryCount++;
            continue;
          }
          console.error(`[frostclaw-proxy] upstream error ${sfRes.status} on streaming request to ${routeName}${retryCount > 0 ? ` (after ${retryCount} retries)` : ""}: ${errBody.slice(0, 1000)}`);
          const responseHeaders = { "content-type": "application/json" };
          for (const [key, value] of sfRes.headers.entries()) {
            const lower = key.toLowerCase();
            if (!["content-length", "transfer-encoding", "content-encoding", "connection"].includes(lower)) {
              responseHeaders[lower] = value;
            }
          }
          if (retryCount > 0) responseHeaders["x-retry-count"] = String(retryCount);
          responseHeaders["content-length"] = Buffer.byteLength(errBody).toString();
          res.writeHead(sfRes.status, responseHeaders);
          res.end(errBody);
          return;
        }

        if (!sfRes.body) {
          res.writeHead(sfRes.status, { "content-type": "text/event-stream" });
          res.end();
          return;
        }

        // Buffer the full stream, watching for the terminal SSE event.
        const chunks = [];
        const decoder = new TextDecoder("utf-8");
        let decodedText = "";
        let streamErr = null;
        let bytesReceived = 0;
        const reader = sfRes.body.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            bytesReceived += value.byteLength;
            decodedText += decoder.decode(value, { stream: true });
          }
        } catch (err) {
          streamErr = err;
        } finally {
          reader.cancel().catch(() => {});
        }

        if (clientAborted) return;

        const sawTerminalEvent = isAnthropicFormat
          ? /event:\s*message_stop/.test(decodedText) || /"type"\s*:\s*"message_stop"/.test(decodedText)
          : /data:\s*\[DONE\]/.test(decodedText);

        if (!streamErr && sawTerminalEvent) {
          const responseHeaders = {
            "content-type": sfRes.headers.get("content-type") || "text/event-stream",
            "cache-control": "no-cache",
            connection: "keep-alive",
          };
          for (const [key, value] of sfRes.headers.entries()) {
            const lower = key.toLowerCase();
            if (!["content-length", "transfer-encoding", "connection"].includes(lower) && !responseHeaders[lower]) {
              responseHeaders[lower] = value;
            }
          }
          if (retryCount > 0) responseHeaders["x-retry-count"] = String(retryCount);
          res.writeHead(sfRes.status, responseHeaders);
          for (const chunk of chunks) res.write(chunk);
          res.end();
          return;
        }

        // Connection dropped before a terminal event — the mid-stream analog
        // of the 392606 "connection reset by peer" case above. Retry the
        // whole request if we have attempts left.
        const reason = streamErr
          ? describeFetchError(streamErr)
          : `stream ended without ${isAnthropicFormat ? "message_stop" : "[DONE]"} (bytesReceived=${bytesReceived})`;
        if (attempt < maxRetries) {
          const waitMs = getRetryWaitMs(new Headers(), attempt, baseDelayMs);
          console.warn(`[frostclaw-proxy] streaming ${routeName} dropped mid-stream (${reason}) — retry ${attempt + 1}/${maxRetries} in ${(waitMs / 1000).toFixed(1)}s`);
          await new Promise((resolve) => setTimeout(resolve, waitMs));
          retryCount++;
          continue;
        }

        console.error(`[frostclaw-proxy] streaming ${routeName} dropped mid-stream after ${retryCount} retries: ${reason}`);
        sendJsonError(502, `Snowflake Cortex connection dropped mid-stream after ${retryCount} retries: ${reason}`);
        return;
      }
    } finally {
      req.socket?.off("close", onClientAbort);
      req.socket?.off("error", onClientAbort);
    }
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

    const maxRetries = getMaxRetries();
    const baseDelayMs = getRetryBaseDelayMs();
    const routeName = targetUrl.split("/").pop();

    if (isStreaming) {
      return proxyStreamingRequest(req, res, targetUrl, forwardHeaders, bodyStr, maxRetries, baseDelayMs, routeName);
    }

    let sfRes = null;
    let retryCount = 0;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const _t0 = Date.now();
      const fetchOpts = {
        method: "POST",
        headers: forwardHeaders,
        body: bodyStr,
      };
      if (_snowflakeAgent) fetchOpts.dispatcher = _snowflakeAgent;

      // Non-streaming requests previously had no try/catch here: if Snowflake
      // reset the connection (392606/GOAWAY/"terminated") before headers
      // arrived, _snowflakeFetch() threw and the exception propagated
      // uncaught to the route handler's outer catch — an immediate 500 with
      // zero retries, unlike the streaming path which already retried this
      // exact failure mode. Wrap it the same way so non-streaming callers
      // (embeddings, graphiti-core structured-output calls, etc.) get the
      // same retry-on-connection-reset behavior.
      try {
        sfRes = await _snowflakeFetch(targetUrl, fetchOpts);
      } catch (err) {
        const detail = describeFetchError(err);
        if (attempt < maxRetries) {
          const waitMs = getRetryWaitMs(new Headers(), attempt, baseDelayMs);
          console.warn(`[frostclaw-proxy] fetch error on non-streaming ${routeName} (${detail}) — retry ${attempt + 1}/${maxRetries} in ${(waitMs / 1000).toFixed(1)}s`);
          await new Promise((resolve) => setTimeout(resolve, waitMs));
          retryCount++;
          continue;
        }
        console.error(`[frostclaw-proxy] fetch error on non-streaming ${routeName} after ${retryCount} retries: ${detail}`);
        res.writeHead(502, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: `Snowflake fetch failed after ${retryCount} retries: ${err.message}`, type: "proxy_error" } }));
        return;
      }
      console.log(`[frostclaw-proxy] upstream ${routeName} → ${sfRes.status} (${Date.now() - _t0}ms, reqBytes=${bodyStr.length}${retryCount > 0 ? `, retry=${retryCount}` : ""})`);

      if (sfRes.ok) break; // success — exit retry loop

      // Buffer the error body so we can inspect it for retry eligibility
      // (and forward it if we're not retrying).
      const errBody = await sfRes.text().catch(() => "");

      if (attempt < maxRetries && isRetryableStatus(sfRes.status, errBody)) {
        const waitMs = getRetryWaitMs(sfRes.headers, attempt, baseDelayMs);
        console.warn(
          `[frostclaw-proxy] retryable ${sfRes.status} from ${routeName} — waiting ${(waitMs / 1000).toFixed(1)}s before retry ${attempt + 1}/${maxRetries} (Retry-After: ${sfRes.headers.get("retry-after") ?? "none"})`,
        );
        await new Promise((resolve) => setTimeout(resolve, waitMs));
        retryCount++;
        continue;
      }

      // Non-retryable error (or retries exhausted) — forward error to client.
      console.error(
        `[frostclaw-proxy] upstream error ${sfRes.status} on non-streaming request to ${routeName}${retryCount > 0 ? ` (after ${retryCount} retries)` : ""}: ${errBody.slice(0, 1000)}`,
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
      if (retryCount > 0) responseHeaders["x-retry-count"] = String(retryCount);
      responseHeaders["content-length"] = Buffer.byteLength(errBody).toString();
      res.writeHead(sfRes.status, responseHeaders);
      res.end(errBody);
      return;
    }

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
    if (retryCount > 0) responseHeaders["x-retry-count"] = String(retryCount);
    res.writeHead(sfRes.status, responseHeaders);
    res.end(responseBody);
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
          routes: ["embeddings", "v1/chat/completions", "v1/models", "api/v2/cortex/v1/messages", "api/v2/cortex/v1/chat/completions", "api/v2/cortex/inference:embed"],
          embedModels: [...SUPPORTED_EMBED_MODELS],
        }),
      );
      return;
    }

    // ── GET /v1/models ────────────────────────────────────────────────────
    // Returns all Snowflake Cortex models in OpenAI list-models format so
    // clients that call GET /v1/models (e.g. FalkorDB text-to-cypher) get
    // the full model catalog instead of proxying to Snowflake's endpoint
    // (which only returns models with provider-specific availability).
    if (req.method === "GET" && (req.url === "/v1/models" || req.url === "/models")) {
      const now = Math.floor(Date.now() / 1000);
      const models = [
        // Claude models
        "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
        "claude-opus-4-5", "claude-sonnet-5", "claude-sonnet-4-6", "claude-4-sonnet", "claude-sonnet-4-5",
        "claude-haiku-4-5",
        // OpenAI models
        "openai-gpt-5.4", "openai-gpt-5.2", "openai-gpt-5.1", "openai-gpt-5",
        "openai-gpt-5-mini", "openai-gpt-5-nano", "openai-gpt-4.1",
        // Open-source models
        "deepseek-r1", "llama4-maverick", "llama3.1-405b", "llama3.1-70b",
        "llama3.1-8b", "llama3.3-70b", "mistral-large", "mistral-large2",
        "mistral-7b", "snowflake-llama-3.3-70b",
      ];
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        object: "list",
        data: models.map((id) => ({
          id,
          object: "model",
          created: now,
          owned_by: "snowflake-cortex",
        })),
      }));
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
        //     tools[].eager_input_streaming (a top-level tool field, NOT inside
        //     custom) with:
        //       400 ...eager_input_streaming: Extra inputs are not permitted
        //     Supported on direct Anthropic API, Bedrock, Vertex, Foundry —
        //     not Snowflake. Strip it unconditionally on this endpoint.
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
              if (!("eager_input_streaming" in tool)) continue;
              delete tool.eager_input_streaming;
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
          // For all models: inject a hardened JSON system prompt when response_format
          // is present — this satisfies Snowflake Cortex's requirement that messages
          // contain the word "json" when response_format is set (OpenAI models), and
          // provides a structural-output fallback (non-OpenAI models).
          // For non-OpenAI models: also strip response_format since Cortex rejects it
          // for Claude/Llama/Mistral. OpenAI models support it natively so we keep it.
          if ("response_format" in body) {
            if (isOpenAIModel) {
              const injected = injectJsonPromptFromResponseFormat(body);
              if (injected) console.log(`[frostclaw-proxy] injected JSON system prompt for OpenAI model, response_format type=${injected} kept`);
            } else {
              const stripped = stripResponseFormatAndInjectPrompt(body);
              if (stripped) console.log(`[frostclaw-proxy] stripped response_format type=${stripped} and injected JSON system prompt`);
            }
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
          // For all models: inject hardened JSON system prompt when response_format present.
          // For non-OpenAI models: also strip the field (Cortex rejects it for Claude/Llama/Mistral).
          // OpenAI models support response_format natively so we keep it.
          if ("response_format" in body) {
            if (isOpenAIModel) {
              const injected = injectJsonPromptFromResponseFormat(body);
              if (injected) console.log(`[frostclaw-proxy] injected JSON system prompt for OpenAI model (v1 alias), response_format type=${injected} kept`);
            } else {
              const stripped = stripResponseFormatAndInjectPrompt(body);
              if (stripped) console.log(`[frostclaw-proxy] stripped response_format (v1 alias) type=${stripped} and injected JSON system prompt`);
            }
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
