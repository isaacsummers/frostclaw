#!/usr/bin/env node

/**
 * Snowflake Cortex Unified Proxy
 *
 * Single proxy replacing both embed-proxy.mjs and cortex-proxy.mjs.
 * Handles all three Snowflake Cortex API surfaces on one port.
 *
 * Routes:
 *   POST /v1/embeddings       → Snowflake inference:embed  (with batching/coalescing)
 *   POST /v1/messages         → Snowflake cortex/v1/messages  (Anthropic, with header normalisation)
 *   POST /v1/chat/completions → Snowflake cortex/v1/chat/completions  (OpenAI-compat passthrough)
 *   GET  /health              → {"status":"ok",...}
 *   *                         → 404
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
 * Environment:
 *   SNOWFLAKE_BASE_URL             - Snowflake account URL (required)
 *   SNOWFLAKE_CORTEX_API_KEY       - Programmatic Access Token (preferred)
 *   SNOWFLAKE_PAT                  - Alternative to SNOWFLAKE_CORTEX_API_KEY
 *   SNOWFLAKE_CORTEX_PROXY_PORT    - Port to listen on (default: 18790)
 *   EMBED_COALESCE_MS              - Batch window in ms (default: 50)
 *   EMBED_MAX_BATCH_TEXTS          - Max texts per batch (default: 64)
 */

import { createServer } from "node:http";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const SNOWFLAKE_BASE_URL = (process.env.SNOWFLAKE_BASE_URL || "").replace(
  /\/$/,
  "",
);
const SNOWFLAKE_PAT =
  process.env.SNOWFLAKE_CORTEX_API_KEY || process.env.SNOWFLAKE_PAT;
const PORT = parseInt(process.env.SNOWFLAKE_CORTEX_PROXY_PORT || "18790");
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

const COALESCE_MS = parseInt(process.env.EMBED_COALESCE_MS || "50");
const MAX_BATCH_TEXTS = parseInt(process.env.EMBED_MAX_BATCH_TEXTS || "64");

if (!SNOWFLAKE_BASE_URL || !SNOWFLAKE_PAT) {
  console.error(
    "[frostclaw] Missing SNOWFLAKE_BASE_URL or SNOWFLAKE_CORTEX_API_KEY/SNOWFLAKE_PAT",
  );
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Embedding — request coalescing / batching
// ---------------------------------------------------------------------------

/** @type {Array<{ texts: string[], model: string, resolve: Function, reject: Function }>} */
let pendingQueue = [];
let flushTimer = null;

/**
 * Flush all pending embedding items as a single (or chunked) Snowflake API call.
 * Items with the same model are batched together.
 */
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
    // Flatten all texts, keeping track of which item/position each belongs to
    const allTexts = [];
    const mapping = []; // { itemIndex, textIndex }
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < items[i].texts.length; j++) {
        allTexts.push(items[i].texts[j]);
        mapping.push({ itemIndex: i, textIndex: j });
      }
    }

    // Allocate result buffers (null = not yet received)
    const itemResults = items.map((item) =>
      new Array(item.texts.length).fill(null),
    );

    // Split into chunks of MAX_BATCH_TEXTS and call Snowflake
    for (let offset = 0; offset < allTexts.length; offset += MAX_BATCH_TEXTS) {
      const chunkTexts = allTexts.slice(offset, offset + MAX_BATCH_TEXTS);
      const chunkMapping = mapping.slice(offset, offset + MAX_BATCH_TEXTS);

      let sfJson;
      try {
        const sfRes = await fetch(
          `${SNOWFLAKE_BASE_URL}/api/v2/cortex/inference:embed`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${SNOWFLAKE_PAT}`,
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

      // Distribute embeddings back to per-item result buffers
      for (let i = 0; i < sfJson.data.length; i++) {
        const raw = sfJson.data[i].embedding;
        const vec = Array.isArray(raw[0]) ? raw[0] : raw;
        const { itemIndex, textIndex } = chunkMapping[i];
        itemResults[itemIndex][textIndex] = vec;
      }
    }

    // Resolve or reject each item
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

/**
 * Schedule a flush. If the queue already has MAX_BATCH_TEXTS texts, flush immediately.
 */
function scheduleFlush() {
  const totalTexts = pendingQueue.reduce((s, it) => s + it.texts.length, 0);
  if (totalTexts >= MAX_BATCH_TEXTS) {
    if (flushTimer !== null) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
    setImmediate(flushQueue);
    return;
  }
  if (flushTimer === null) {
    flushTimer = setTimeout(flushQueue, COALESCE_MS);
  }
}

/**
 * Enqueue texts for embedding and return a Promise that resolves with vectors.
 * @param {string[]} texts
 * @param {string} model
 * @returns {Promise<number[][]>}
 */
function embedTexts(texts, model) {
  return new Promise((resolve, reject) => {
    pendingQueue.push({ texts, model, resolve, reject });
    scheduleFlush();
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/**
 * Build forwarded headers: strip client auth/connection headers, inject
 * Snowflake auth, normalise content-type and content-length.
 */
function buildForwardHeaders(incomingHeaders, bodyStr) {
  const headers = {};
  for (const [key, value] of Object.entries(incomingHeaders)) {
    const lower = key.toLowerCase();
    if (!STRIP_HEADERS.has(lower)) {
      headers[lower] = value;
    }
  }
  headers["authorization"] = `Bearer ${SNOWFLAKE_PAT}`;
  headers["x-snowflake-authorization-token-type"] =
    "PROGRAMMATIC_ACCESS_TOKEN";
  headers["content-type"] = "application/json";
  headers["content-length"] = Buffer.byteLength(bodyStr).toString();
  return headers;
}

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
 * Proxy a streaming or non-streaming request to a Snowflake target URL.
 */
async function proxyRequest(req, res, targetUrl, extraHeaders = {}) {
  const bodyStr = await readBody(req);

  // Determine if the client wants streaming
  let isStreaming = false;
  try {
    isStreaming = JSON.parse(bodyStr).stream === true;
  } catch {
    // non-JSON or unparseable — treat as non-streaming
  }

  const forwardHeaders = buildForwardHeaders(req.headers, bodyStr);
  // Allow caller to inject additional headers (e.g. anthropic-version)
  for (const [k, v] of Object.entries(extraHeaders)) {
    if (!forwardHeaders[k]) forwardHeaders[k] = v;
  }

  const sfRes = await fetch(targetUrl, {
    method: "POST",
    headers: forwardHeaders,
    body: bodyStr,
  });

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
    const pump = async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          res.end();
          break;
        }
        const ok = res.write(value);
        if (!ok) {
          await new Promise((resolve) => res.once("drain", resolve));
        }
      }
    };
    pump().catch((err) => {
      console.error("[frostclaw] Stream pump error:", err.message);
      res.end();
    });
  } else {
    const responseBody = await sfRes.text();
    const responseHeaders = { "content-type": "application/json" };
    for (const [key, value] of sfRes.headers.entries()) {
      const lower = key.toLowerCase();
      if (
        lower !== "content-length" &&
        lower !== "transfer-encoding" &&
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

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

const server = createServer(async (req, res) => {
  // Health check
  if (req.method === "GET" && req.url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        status: "ok",
        provider: "snowflake-cortex",
        routes: ["embeddings", "messages", "chat/completions"],
        embedModels: [...SUPPORTED_EMBED_MODELS],
      }),
    );
    return;
  }

  // ── POST /v1/embeddings ──────────────────────────────────────────────────
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

  // ── POST /v1/messages ────────────────────────────────────────────────────
  if (req.method === "POST" && req.url === "/v1/messages") {
    try {
      await proxyRequest(
        req,
        res,
        `${SNOWFLAKE_BASE_URL}/api/v2/cortex/v1/messages`,
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

  // ── POST /v1/chat/completions ────────────────────────────────────────────
  if (req.method === "POST" && req.url === "/v1/chat/completions") {
    try {
      // Read and parse the body so we can apply Snowflake-required fixes
      // before forwarding. These mirror the transforms frostclaw applies
      // internally via the SDK compat layer:
      //
      //   1. max_tokens → max_completion_tokens
      //      Snowflake's OpenAI-compat endpoint has deprecated max_tokens in
      //      favour of max_completion_tokens (matching the OpenAI v2 spec).
      //      Forwarding max_tokens produces a deprecation warning in the
      //      Snowflake response and may stop working in a future version.
      //
      //   2. Strip the "openai-" prefix from model IDs
      //      Client-facing model IDs use the "openai-" namespace prefix
      //      (e.g. "openai-gpt-5-mini") so OpenClaw can route them to this
      //      provider. Snowflake's endpoint only recognises the bare names
      //      without the prefix (e.g. "gpt-5-mini").
      const rawBody = await readBody(req);
      let body;
      try {
        body = JSON.parse(rawBody);
      } catch {
        // Unparseable body — forward as-is and let Snowflake return the error
        body = null;
      }

      let forwardBody = rawBody;
      if (body !== null && typeof body === "object") {
        // Fix 1: max_tokens → max_completion_tokens
        if ("max_tokens" in body) {
          body.max_completion_tokens = body.max_tokens;
          delete body.max_tokens;
        }

        // Fix 2: strip "openai-" prefix from model ID
        if (typeof body.model === "string" && body.model.startsWith("openai-")) {
          body.model = body.model.slice("openai-".length);
        }

        forwardBody = JSON.stringify(body);
      }

      // Inject the modified body back into a synthetic request-like object
      // so proxyRequest can forward it. Since proxyRequest calls readBody()
      // internally we wrap the already-read string as an async generator.
      const syntheticReq = {
        headers: req.headers,
        [Symbol.asyncIterator]: async function* () {
          yield Buffer.from(forwardBody);
        },
      };

      await proxyRequest(
        syntheticReq,
        res,
        `${SNOWFLAKE_BASE_URL}/api/v2/cortex/v1/chat/completions`,
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

  // ── Catch-all 404 ────────────────────────────────────────────────────────
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(
    JSON.stringify({
      error:
        "Not found. Available routes: POST /v1/embeddings, POST /v1/messages, POST /v1/chat/completions, GET /health",
    }),
  );
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(
    `[frostclaw] Snowflake unified proxy listening on http://127.0.0.1:${PORT} (coalesce=${COALESCE_MS}ms, maxBatch=${MAX_BATCH_TEXTS})`,
  );
});
