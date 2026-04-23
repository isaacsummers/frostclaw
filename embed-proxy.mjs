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
 * Environment:
 *   SNOWFLAKE_BASE_URL          - Snowflake account URL (required)
 *   SNOWFLAKE_PAT               - Programmatic Access Token (preferred)
 *   SNOWFLAKE_CORTEX_API_KEY    - Alternative to SNOWFLAKE_PAT
 *   SNOWFLAKE_EMBED_PROXY_PORT  - Port to listen on (default: 18790)
 */

import { createServer } from "node:http";

const SNOWFLAKE_BASE_URL = process.env.SNOWFLAKE_BASE_URL;
const SNOWFLAKE_PAT =
  process.env.SNOWFLAKE_PAT || process.env.SNOWFLAKE_CORTEX_API_KEY;
const PORT = parseInt(process.env.SNOWFLAKE_EMBED_PROXY_PORT || "18790");
const DEFAULT_MODEL = "snowflake-arctic-embed-m-v1.5";

if (!SNOWFLAKE_BASE_URL || !SNOWFLAKE_PAT) {
  console.error(
    "Missing SNOWFLAKE_BASE_URL or SNOWFLAKE_PAT/SNOWFLAKE_CORTEX_API_KEY",
  );
  process.exit(1);
}

const server = createServer(async (req, res) => {
  // Health check
  if (req.method === "GET" && req.url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", provider: "snowflake-cortex" }));
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

    // Translate: OpenAI format → Snowflake format
    const input = Array.isArray(body.input) ? body.input : [body.input];
    const model = body.model || DEFAULT_MODEL;

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
        body: JSON.stringify({ text: input, model }),
      },
    );

    if (!sfRes.ok) {
      const errBody = await sfRes.text().catch(() => "");
      res.writeHead(sfRes.status, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: { message: errBody, type: "snowflake_error" },
        }),
      );
      return;
    }

    const sfJson = await sfRes.json();

    // Translate: Snowflake format → OpenAI format
    // Snowflake wraps vectors in [[vec]], OpenAI expects [vec]
    const data = sfJson.data.map((item) => ({
      object: "embedding",
      embedding: Array.isArray(item.embedding[0])
        ? item.embedding[0]
        : item.embedding,
      index: item.index,
    }));

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        object: "list",
        data,
        model: sfJson.model || model,
        usage: sfJson.usage || { prompt_tokens: 0, total_tokens: 0 },
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
    `[frostclaw] Snowflake embed proxy listening on http://127.0.0.1:${PORT}`,
  );
});
