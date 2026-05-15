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
 * Environment (when run standalone):
 *   SNOWFLAKE_BASE_URL             - Snowflake account URL (required)
 *   SNOWFLAKE_CORTEX_API_KEY       - Programmatic Access Token (preferred)
 *   SNOWFLAKE_PAT                  - Alternative to SNOWFLAKE_CORTEX_API_KEY
 *   SNOWFLAKE_CORTEX_PROXY_PORT    - Port to listen on (default: 18790)
 *   EMBED_COALESCE_MS              - Batch window in ms (default: 50)
 *   EMBED_MAX_BATCH_TEXTS          - Max texts per batch (default: 64)
 */

import { createServer } from "node:http";

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
          const sfRes = await fetch(
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

  // ── HTTP server ────────────────────────────────────────────────────────────

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

    // ── POST /v1/messages ──────────────────────────────────────────────────
    if (req.method === "POST" && req.url === "/v1/messages") {
      try {
        const rawBody = await readBody(req);
        let body;
        try {
          body = JSON.parse(rawBody);
        } catch {
          body = null;
        }

        // Defensive scrub: drop `eager_input_streaming` from every tool's
        // `custom` object. Snowflake Cortex's strict validator (Haiku 4.5 and
        // other tiers) rejects the field with:
        //   400 tools.0.custom.eager_input_streaming: Extra inputs are not
        //       permitted
        // The plugin's onPayload hook already handles this when frostclaw is
        // in-process, but the standalone proxy may receive requests from any
        // Anthropic SDK client that still emits the field. Strip it here so
        // the proxy is safe regardless of caller. If the resulting `custom`
        // object is empty, drop the key so we never forward `"custom": {}`.
        let forwardBody = rawBody;
        if (body !== null && typeof body === "object" && Array.isArray(body.tools)) {
          let mutated = false;
          for (const tool of body.tools) {
            if (!tool || typeof tool !== "object") continue;
            const custom = tool.custom;
            if (!custom || typeof custom !== "object") continue;
            if (!("eager_input_streaming" in custom)) continue;
            delete custom.eager_input_streaming;
            if (Object.keys(custom).length === 0) delete tool.custom;
            mutated = true;
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

    // ── POST /v1/chat/completions ──────────────────────────────────────────
    if (req.method === "POST" && req.url === "/v1/chat/completions") {
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
        // Snowflake's /api/v2/cortex/v1/chat/completions handles ALL models
        // (Claude, OpenAI, Llama, …) natively in OpenAI format — no format
        // translation to/from Anthropic Messages format is required.
        let forwardBody = rawBody;
        if (body !== null && typeof body === "object") {
          if (typeof body.model === "string") {
            body.model = body.model
              .replace(/^snowflake-cortex\//, "")
              .replace(/^openai-/, "");
          }
          if ("max_tokens" in body) {
            body.max_completion_tokens = body.max_tokens;
            delete body.max_tokens;
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

    // ── Catch-all 404 ──────────────────────────────────────────────────────
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        error:
          "Not found. Available routes: POST /v1/embeddings, POST /v1/messages, POST /v1/chat/completions, GET /health",
      }),
    );
  });

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

if (import.meta.main || process.argv[1]?.includes("snowflake-proxy")) {
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
