#!/usr/bin/env node

/**
 * Snowflake Cortex Embedding Proxy
 *
 * Translates OpenAI-compatible embedding requests to Snowflake's native
 * embed API format. This is needed because OpenClaw's CLI tools (e.g.
 * `openclaw memory index`) don't load full plugin handlers, so they can't
 * use the native embedding adapter registered by the frostclaw plugin.
 *
 * The gateway process uses the native adapter directly via
 * registerMemoryEmbeddingProvider — this proxy is only needed for CLI tools.
 *
 * Format differences handled:
 *   Request:  OpenAI uses `input` (string|string[]) → Snowflake uses `text` (string[])
 *   Response: Snowflake wraps vectors in [[vec]] → OpenAI expects [vec]
 *
 * Concurrency:
 *   Requests arriving within COALESCE_MS (default 50ms) of each other are
 *   batched into a single Snowflake API call and fanned back to callers.
 *   This avoids N simultaneous Snowflake calls under burst load (e.g. 5-6
 *   agent sessions starting at once), while keeping latency low.
 *   MAX_BATCH_TEXTS caps each batch for very large bursts.
 *
 * Environment:
 *   SNOWFLAKE_BASE_URL          - Snowflake account URL (required)
 *   SNOWFLAKE_PAT               - Programmatic Access Token (preferred)
 *   SNOWFLAKE_CORTEX_API_KEY    - Alternative to SNOWFLAKE_PAT
 *   SNOWFLAKE_EMBED_PROXY_PORT  - Port to listen on (default: 18790)
 *   EMBED_COALESCE_MS           - Batch window in ms (default: 50)
 *   EMBED_MAX_BATCH_TEXTS       - Max texts per batch (default: 64)
 */

import { createServer } from "node:http";

const SNOWFLAKE_BASE_URL = process.env.SNOWFLAKE_BASE_URL;
const SNOWFLAKE_PAT =
  process.env.SNOWFLAKE_PAT || process.env.SNOWFLAKE_CORTEX_API_KEY;
const PORT = parseInt(process.env.SNOWFLAKE_EMBED_PROXY_PORT || "18790");
const DEFAULT_MODEL = "snowflake-arctic-embed-m-v1.5";
const COALESCE_MS = parseInt(process.env.EMBED_COALESCE_MS || "50");
const MAX_BATCH_TEXTS = parseInt(process.env.EMBED_MAX_BATCH_TEXTS || "64");

if (!SNOWFLAKE_BASE_URL || !SNOWFLAKE_PAT) {
  console.error(
    "Missing SNOWFLAKE_BASE_URL or SNOWFLAKE_PAT/SNOWFLAKE_CORTEX_API_KEY",
  );
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Request coalescing / batching
// ---------------------------------------------------------------------------

/** @type {Array<{ texts: string[], model: string, resolve: Function, reject: Function }>} */
let pendingQueue = [];
let flushTimer = null;

/**
 * Flush all pending items as a single (or chunked) Snowflake API call.
 * Items with the same model are batched together; different models each get
 * their own call (rare in practice — typically all use the same model).
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
          const err = new Error(
            `Snowflake error ${sfRes.status}: ${errBody}`,
          );
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
 *
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
        coalesceMs: COALESCE_MS,
        maxBatchTexts: MAX_BATCH_TEXTS,
        queueDepth: pendingQueue.length,
      }),
    );
    return;
  }

  // Only handle POST /v1/embeddings (OpenAI-compatible path)
  if (req.method !== "POST" || !req.url?.includes("/embeddings")) {
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Not found. Use POST /v1/embeddings" }));
    return;
  }

  try {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks).toString());

    const input = Array.isArray(body.input) ? body.input : [body.input];
    const model = body.model || DEFAULT_MODEL;

    // Enqueue and await coalesced batch result
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
      JSON.stringify({
        error: { message: err.message, type: "proxy_error" },
      }),
    );
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(
    `[frostclaw] Snowflake embed proxy listening on http://127.0.0.1:${PORT} (coalesce=${COALESCE_MS}ms, maxBatch=${MAX_BATCH_TEXTS})`,
  );
});
