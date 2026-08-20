import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { createServer } from "node:http";
import type { Server, IncomingMessage, ServerResponse } from "node:http";
import { createProxyServer } from "../snowflake-proxy.mjs";

// ---------------------------------------------------------------------------
// Helpers — mini in-process mock upstream
// ---------------------------------------------------------------------------

function createMockUpstream(): {
  server: Server;
  setHandler: (fn: (req: IncomingMessage, res: ServerResponse) => void) => void;
  url: () => string;
  close: () => Promise<void>;
} {
  let handler: (req: IncomingMessage, res: ServerResponse) => void = (_req, res) => {
    res.writeHead(500);
    res.end("no handler set");
  };

  const server = createServer((req, res) => handler(req, res));

  return {
    server,
    setHandler(fn) {
      handler = fn;
    },
    url() {
      const addr = server.address();
      if (!addr || typeof addr === "string") throw new Error("server not listening");
      return `http://127.0.0.1:${addr.port}`;
    },
    close() {
      return new Promise((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      );
    },
  };
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString();
}

// ---------------------------------------------------------------------------
// Suite setup — one mock upstream, one proxy, shared across all tests
// ---------------------------------------------------------------------------

let mock: ReturnType<typeof createMockUpstream>;
let proxy: ReturnType<typeof createProxyServer>;
let proxyPort: number;

beforeAll(async () => {
  mock = createMockUpstream();
  await new Promise<void>((resolve) => mock.server.listen(0, "127.0.0.1", resolve));

  proxy = createProxyServer({
    baseUrl: mock.url(),
    pat: "test-pat-token",
    coalesceMs: 5,
    maxBatchTexts: 64,
  });
  await new Promise<void>((resolve) => proxy.server.listen(0, "127.0.0.1", resolve));
  const addr = proxy.server.address();
  proxyPort = typeof addr === "object" && addr !== null ? addr.port : 0;
});

afterAll(async () => {
  await proxy.close();
  await mock.close();
});

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

test("GET /health returns 200 with embedModels array of 8", async () => {
  const res = await fetch(`http://127.0.0.1:${proxyPort}/health`);
  expect(res.status).toBe(200);
  const body = (await res.json()) as { status: string; embedModels: string[] };
  expect(body.status).toBe("ok");
  expect(Array.isArray(body.embedModels)).toBe(true);
  expect(body.embedModels).toHaveLength(8);
});

// ---------------------------------------------------------------------------
// POST /v1/embeddings
// ---------------------------------------------------------------------------

describe("POST /v1/embeddings", () => {
  test("happy path — returns OpenAI-compat shape and forwards auth headers", async () => {
    const capturedHeaders: Record<string, string> = {};
    mock.setHandler(async (req, res) => {
      for (const [k, v] of Object.entries(req.headers)) {
        if (typeof v === "string") capturedHeaders[k] = v;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: [{ embedding: [0.1, 0.2, 0.3] }] }));
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: ["hello world"], model: "snowflake-arctic-embed-m-v1.5" }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      object: string;
      data: Array<{ object: string; embedding: number[]; index: number }>;
      model: string;
      usage: { prompt_tokens: number; total_tokens: number };
    };
    expect(body.object).toBe("list");
    expect(Array.isArray(body.data)).toBe(true);
    expect(body.data[0].object).toBe("embedding");
    expect(body.data[0].embedding).toEqual([0.1, 0.2, 0.3]);
    expect(body.data[0].index).toBe(0);
    expect(body.model).toBe("snowflake-arctic-embed-m-v1.5");
    expect(body.usage).toBeTruthy();

    expect(capturedHeaders["authorization"]).toBe("Bearer test-pat-token");
    expect(capturedHeaders["x-snowflake-authorization-token-type"]).toBe(
      "PROGRAMMATIC_ACCESS_TOKEN",
    );
  });

  test("default model — request without model field uses snowflake-arctic-embed-m-v1.5", async () => {
    let requestedModel = "";
    mock.setHandler(async (req, res) => {
      const body = JSON.parse(await readBody(req)) as { model: string };
      requestedModel = body.model;
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: [{ embedding: [0.5] }] }));
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: ["hello"] }),
    });
    expect(res.status).toBe(200);
    expect(requestedModel).toBe("snowflake-arctic-embed-m-v1.5");
  });

  test("unknown model — returns 400", async () => {
    const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: ["hello"], model: "not-a-real-model" }),
    });
    expect(res.status).toBe(400);
    const body = (await res.json()) as { error: { message: string } };
    expect(body.error.message).toMatch(/unsupported/i);
  });

  test("nested vector unwrap — proxy unwraps [[0.1, 0.2]] to [0.1, 0.2]", async () => {
    mock.setHandler(async (_req, res) => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: [{ embedding: [[0.1, 0.2]] }] }));
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: ["hello"], model: "snowflake-arctic-embed-m-v1.5" }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { data: Array<{ embedding: number[] }> };
    expect(body.data[0].embedding).toEqual([0.1, 0.2]);
  });

  test("batching/coalescing — 3 concurrent requests result in 1 upstream call", async () => {
    let upstreamCallCount = 0;
    mock.setHandler(async (req, res) => {
      upstreamCallCount++;
      const body = JSON.parse(await readBody(req)) as { text: string[] };
      const data = body.text.map((_t: string, i: number) => ({ embedding: [0.1 * (i + 1)] }));
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data }));
    });

    upstreamCallCount = 0;
    const results = await Promise.all([
      fetch(`http://127.0.0.1:${proxyPort}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: ["text1"], model: "snowflake-arctic-embed-m-v1.5" }),
      }),
      fetch(`http://127.0.0.1:${proxyPort}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: ["text2"], model: "snowflake-arctic-embed-m-v1.5" }),
      }),
      fetch(`http://127.0.0.1:${proxyPort}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: ["text3"], model: "snowflake-arctic-embed-m-v1.5" }),
      }),
    ]);

    for (const r of results) {
      expect(r.status).toBe(200);
    }
    expect(upstreamCallCount).toBe(1);
  });

  test("upstream error — returns error status", async () => {
    mock.setHandler(async (_req, res) => {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ message: "unauthorized" }));
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: ["hello"], model: "snowflake-arctic-embed-m-v1.5" }),
    });
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test("connection reset before headers (non-streaming) — retries and recovers cleanly", async () => {
    // Root-cause gap found while investigating GOAWAY/connection-reset
    // reports: the streaming path already retried a connection dropped
    // before headers, but the non-streaming path had no try/catch around
    // the fetch() call at all — a reset here threw uncaught and skipped
    // retry entirely, unlike the streaming path. This reproduces that case
    // for a non-streaming caller (e.g. embeddings, graphiti-core).
    let attempt = 0;
    mock.setHandler(async (_req, res) => {
      attempt++;
      if (attempt === 1) {
        // Simulate Snowflake resetting the connection before any response.
        res.destroy();
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: [{ embedding: [0.1, 0.2], index: 0 }] }));
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: ["hello"], model: "snowflake-arctic-embed-m-v1.5" }),
    });

    expect(res.status).toBe(200);
    expect(attempt).toBe(2); // one dropped attempt, one clean retry
  });
});

// ---------------------------------------------------------------------------
// POST /v1/messages
// ---------------------------------------------------------------------------

describe("POST /v1/messages", () => {
  test("passthrough — correct auth headers, x-api-key not forwarded", async () => {
    const capturedHeaders: Record<string, string> = {};
    mock.setHandler(async (req, res) => {
      for (const [k, v] of Object.entries(req.headers)) {
        if (typeof v === "string") capturedHeaders[k] = v;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ id: "msg_test", type: "message" }));
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/v2/cortex/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": "client-key-should-not-forward",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { id: string; type: string };
    expect(body.id).toBe("msg_test");

    expect(capturedHeaders["authorization"]).toBe("Bearer test-pat-token");
    expect(capturedHeaders["x-snowflake-authorization-token-type"]).toBe(
      "PROGRAMMATIC_ACCESS_TOKEN",
    );
    expect(capturedHeaders["anthropic-version"]).toBe("2023-06-01");
    expect(capturedHeaders["x-api-key"]).toBeUndefined();
  });

  test("strips eager_input_streaming from tools[] top-level; preserves other fields", async () => {
    let upstreamBody: Record<string, unknown> = {};
    mock.setHandler(async (req, res) => {
      const chunks: Buffer[] = [];
      for await (const c of req) chunks.push(c as Buffer);
      upstreamBody = JSON.parse(Buffer.concat(chunks).toString());
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ id: "msg_test", type: "message" }));
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/v2/cortex/v1/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hi" }],
        tools: [
          {
            name: "lookup",
            description: "do a lookup",
            input_schema: { type: "object" },
            eager_input_streaming: true,
          },
          {
            name: "other",
            input_schema: { type: "object" },
          },
        ],
      }),
    });
    expect(res.status).toBe(200);

    const tools = upstreamBody.tools as Array<Record<string, unknown>>;
    expect(tools).toHaveLength(2);
    // eager_input_streaming stripped from tool[0], other fields preserved
    expect(tools[0]).toEqual({
      name: "lookup",
      description: "do a lookup",
      input_schema: { type: "object" },
    });
    // tool[1] had no eager_input_streaming — untouched
    expect(tools[1]).toEqual({ name: "other", input_schema: { type: "object" } });
  });

  test("upstream 4xx on streaming request — buffers error body and forwards as JSON", async () => {
    // Snowflake returns a JSON error body (not SSE) when it rejects a
    // streaming `/messages` request. The proxy must read that body, log it
    // for diagnostics, and forward it to the client with the original
    // status code instead of leaving the SSE pump to swallow it.
    mock.setHandler(async (_req, res) => {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          message:
            'invalid request parameters: "messages.76: `tool_use` ids were found without `tool_result` blocks immediately after"',
        }),
      );
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/v2/cortex/v1/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-5",
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    expect(res.status).toBe(400);
    expect(res.headers.get("content-type")).toContain("application/json");
    const body = (await res.json()) as { message: string };
    expect(body.message).toContain("tool_use");
  });

  test("connection dropped mid-stream (no message_stop) — retries and recovers cleanly", async () => {
    // Reproduces the real-world root cause: Snowflake Cortex's backend
    // resets the TCP connection mid-response (HTTP/2 GOAWAY / "connection
    // reset by peer", error 392606) after the SSE stream has already
    // started. undici surfaces that as a bare "terminated" error on
    // reader.read() — this must be retried transparently, not surfaced to
    // the client as a truncated/failed stream.
    let attempt = 0;
    mock.setHandler(async (_req, res) => {
      attempt++;
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.write('event: content_block_start\ndata: {"type":"content_block_start"}\n\n');
      if (attempt === 1) {
        // Simulate Snowflake killing the connection before message_stop.
        res.destroy();
        return;
      }
      res.write('event: message_stop\ndata: {"type":"message_stop"}\n\n');
      res.end();
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/v2/cortex/v1/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-5",
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    expect(res.status).toBe(200);
    expect(attempt).toBe(2); // one dropped attempt, one clean retry
    const body = await res.text();
    expect(body).toContain("message_stop");
    expect(res.headers.get("x-retry-count")).toBe("1");
  });

  test("connection dropped mid-stream, retries exhausted — surfaces a clean 502 (never a truncated stream)", async () => {
    // Speed up the exponential backoff for this test so exhausting all
    // retries doesn't blow the test timeout; getRetryWaitMs() reads this
    // env var fresh on every call, so this takes effect immediately.
    const prevBaseDelay = process.env.PROXY_RETRY_BASE_DELAY;
    process.env.PROXY_RETRY_BASE_DELAY = "0";
    try {
      mock.setHandler(async (_req, res) => {
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        res.write('event: content_block_start\ndata: {"type":"content_block_start"}\n\n');
        res.destroy();
      });

      const res = await fetch(`http://127.0.0.1:${proxyPort}/api/v2/cortex/v1/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          stream: true,
          messages: [{ role: "user", content: "hi" }],
        }),
      });

      expect(res.status).toBe(502);
      const body = (await res.json()) as { error: { message: string } };
      // Depending on exactly when the mock destroys the socket, this can
      // surface as either the mid-stream-drop path or the earlier
      // fetch-level connection-closed path — both are correct outcomes:
      // a clean JSON 502 after exhausting retries, never a truncated SSE
      // stream leaked to the client.
      expect(body.error.message).toMatch(/dropped mid-stream|socket connection was closed/i);
    } finally {
      if (prevBaseDelay === undefined) delete process.env.PROXY_RETRY_BASE_DELAY;
      else process.env.PROXY_RETRY_BASE_DELAY = prevBaseDelay;
    }
  });
});

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

describe("POST /api/v2/cortex/v1/chat/completions", () => {
  test("max_tokens rewrite and model prefix stripping", async () => {
    let upstreamBody: Record<string, unknown> = {};
    mock.setHandler(async (req, res) => {
      upstreamBody = JSON.parse(await readBody(req)) as Record<string, unknown>;
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ id: "chat_test", choices: [] }));
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/v2/cortex/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "openai-gpt-5-mini",
        max_tokens: 100,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(200);
    expect(upstreamBody["max_tokens"]).toBeUndefined();
    expect(upstreamBody["max_completion_tokens"]).toBe(100);
    expect(upstreamBody["model"]).toBe("openai-gpt-5-mini");
  });

  test("connection reset before headers (non-streaming) — retries and recovers cleanly", async () => {
    // proxyRequest()'s non-streaming loop previously called _snowflakeFetch()
    // with no try/catch: a connection reset before headers arrived (the same
    // 392606/GOAWAY failure mode already retried on the streaming path) threw
    // uncaught and skipped retry entirely, surfacing as an immediate 500 with
    // zero retries. This reproduces that case on a non-streaming request.
    let attempt = 0;
    mock.setHandler(async (_req, res) => {
      attempt++;
      if (attempt === 1) {
        res.destroy();
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ id: "chat_test", choices: [] }));
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/v2/cortex/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "openai-gpt-5-mini",
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    expect(res.status).toBe(200);
    expect(attempt).toBe(2); // one dropped attempt, one clean retry
    expect(res.headers.get("x-retry-count")).toBe("1");
  });

  test("streaming passthrough — content-type: text/event-stream", async () => {
    // Real Snowflake chat/completions streams terminate with OpenAI's
    // `data: [DONE]` marker (not Anthropic's message_stop). The proxy's
    // buffer-and-verify logic waits for that terminal event before
    // flushing, so the mock must send one to reflect real behavior.
    mock.setHandler(async (_req, res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.write('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n');
      res.write("data: [DONE]\n\n");
      res.end();
    });

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/v2/cortex/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "openai-gpt-5-mini",
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(200);
    const ct = res.headers.get("content-type") ?? "";
    expect(ct).toContain("text/event-stream");
    await res.text(); // consume body
  });
});

// ---------------------------------------------------------------------------
// POST /api/v2/cortex/inference:embed (native passthrough)
// ---------------------------------------------------------------------------

describe("POST /api/v2/cortex/inference:embed", () => {
  test("passes body through unchanged and injects auth headers", async () => {
    let capturedHeaders: Record<string, string> = {};
    let capturedBody: Record<string, unknown> = {};

    mock.setHandler(async (req, res) => {
      for (const [k, v] of Object.entries(req.headers)) {
        if (typeof v === "string") capturedHeaders[k] = v;
      }
      capturedBody = JSON.parse(await readBody(req)) as Record<string, unknown>;
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ data: [{ embedding: [0.9, 0.8, 0.7] }] }));
    });

    const nativePayload = {
      text: ["hello", "world"],
      model: "snowflake-arctic-embed-l-v2.0",
    };

    const res = await fetch(
      `http://127.0.0.1:${proxyPort}/api/v2/cortex/inference:embed`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-api-key": "should-be-stripped" },
        body: JSON.stringify(nativePayload),
      },
    );
    expect(res.status).toBe(200);

    // Body forwarded as-is (native shape: `text` not `input`)
    expect(capturedBody["text"]).toEqual(["hello", "world"]);
    expect(capturedBody["model"]).toBe("snowflake-arctic-embed-l-v2.0");
    expect(capturedBody).not.toHaveProperty("input");

    // Auth injected, client key stripped
    expect(capturedHeaders["authorization"]).toBe("Bearer test-pat-token");
    expect(capturedHeaders["x-snowflake-authorization-token-type"]).toBe(
      "PROGRAMMATIC_ACCESS_TOKEN",
    );
    expect(capturedHeaders["x-api-key"]).toBeUndefined();

    // Response forwarded
    const body = (await res.json()) as { data: Array<{ embedding: number[] }> };
    expect(body.data[0].embedding).toEqual([0.9, 0.8, 0.7]);
  });
});

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

test("POST /unknown-route returns 404", async () => {
  const res = await fetch(`http://127.0.0.1:${proxyPort}/unknown-route`, {
    method: "POST",
    body: "{}",
    headers: { "Content-Type": "application/json" },
  });
  expect(res.status).toBe(404);
});
