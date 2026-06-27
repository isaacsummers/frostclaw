/**
 * Snowflake Cortex — OpenClaw Plugin
 *
 * Routes Claude models to the Anthropic Messages API (`/messages`) and all
 * other models to the OpenAI-compatible Chat Completions API
 * (`/chat/completions`). Both endpoints live under the same Snowflake Cortex
 * gateway and share PAT authentication.
 *
 * Base URL: https://<account>.snowflakecomputing.com
 *
 * All payload transforms (tool stripping, max_tokens rewrite, tool-use repair)
 * are handled via SDK hooks and model compat flags — no raw JSON body patching.
 */

import {
  definePluginEntry,
  type ProviderWrapStreamFnContext,
  type ProviderNormalizeToolSchemasContext,
  type ProviderReplayPolicyContext,
  type ProviderNormalizeResolvedModelContext,
  type ProviderApplyConfigDefaultsContext,
} from "openclaw/plugin-sdk/plugin-entry";
import { resolveClaudeThinkingProfile } from "openclaw/plugin-sdk/provider-model-shared";
import type {
  EmbeddingProviderAdapter,
  EmbeddingProviderCreateOptions,
  EmbeddingInput,
  EmbeddingProviderCallOptions,
} from "openclaw/plugin-sdk/embedding-providers-25O-YtFs";
import { createProviderApiKeyAuthMethod } from "openclaw/plugin-sdk/provider-auth-api-key";
import type {
  ModelDefinitionConfig,
  ModelApi,
} from "openclaw/plugin-sdk/provider-model-types";
import {
  buildModelCatalog,
  findCatalogEntry,
  getCatalogBaseURL,
  isAdaptiveOnly,
} from "./src/catalog.js";
import { applyEagerInputStreamingStrip } from "./src/onpayload.js";

// ---------------------------------------------------------------------------
// Environment — lazy getters so env vars are read at call time, not import time
//
// Recognized variables:
//   SNOWFLAKE_CORTEX_API_KEY / SNOWFLAKE_PAT — Snowflake PAT used for auth
//   SNOWFLAKE_BASE_URL — real Snowflake account URL
//     (e.g. https://<acct>.snowflakecomputing.com). Used by snowflake-proxy.mjs
//     for upstream forwarding, and as the fallback model-request target when
//     no proxy override is set.
//   SNOWFLAKE_PROXY_BASE_URL — optional override pointing at the local proxy
//     (e.g. http://127.0.0.1:18790). When set, openclaw hits this URL directly
//     with no path suffix — the proxy owns the route shape and forwards to
//     Snowflake using its own SNOWFLAKE_BASE_URL.
// ---------------------------------------------------------------------------

function getApiKey(): string {
  return process.env.SNOWFLAKE_CORTEX_API_KEY ?? process.env.SNOWFLAKE_PAT ?? "";
}
function getBaseURL(): string {
  return process.env.SNOWFLAKE_PROXY_BASE_URL ?? process.env.SNOWFLAKE_BASE_URL ?? "";
}

// ---------------------------------------------------------------------------
// Structured debug logger — writes to stderr, which OpenClaw captures into
// its plugin log. Kept dependency-free and side-effect-only.
// ---------------------------------------------------------------------------

// Logging tiers:
//   log()      — debug/verbose. Normally routed to api.logger.debug() so it
//                only appears when OpenClaw's logging.level=debug (or lower).
//                FROSTCLAW_DEBUG=1 overrides this to api.logger.info() so
//                these lines surface at the default info level without touching
//                openclaw.json — useful for temporary tracing without a full
//                log-level change.
//   logWarn()  — operational signals (retry fired, error-path resolved).
//                Always emitted via api.logger.warn().
//   logError() — actual failures. Always emitted via api.logger.error().
//
// Before register() runs (module-load time), all three fall back to the
// matching console.* method so nothing is silently dropped.
type PluginLogger = {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
};

let _pluginLogger: PluginLogger | null = null;

// Numeric ordering for log levels — lower = more verbose.
const LOG_LEVEL_ORDER: Record<string, number> = {
  trace: 0, debug: 1, info: 2, warn: 3, error: 4, fatal: 5,
};

// The effective log level to use when FROSTCLAW_DEBUG=1. Computed once at
// registration time from both logging.level and logging.consoleLevel so debug
// messages surface on whichever output is most permissive.
let _debugForceLevel: "debug" | "info" | "warn" | "error" = "info";

/** Call once inside register(api) to wire in the OpenClaw-scoped logger. */
function setPluginLogger(logger: PluginLogger, cfg: { logging?: { level?: string; consoleLevel?: string } }): void {
  _pluginLogger = logger;
  // Pick the most permissive (lowest) of the two configured log levels so
  // FROSTCLAW_DEBUG forces debug output to wherever it will actually appear.
  const fileLevel = cfg?.logging?.level ?? "info";
  const consoleLevel = cfg?.logging?.consoleLevel ?? "info";
  const fileOrder = LOG_LEVEL_ORDER[fileLevel] ?? LOG_LEVEL_ORDER.info;
  const consoleOrder = LOG_LEVEL_ORDER[consoleLevel] ?? LOG_LEVEL_ORDER.info;
  const effectiveOrder = Math.min(fileOrder, consoleOrder);
  if (effectiveOrder <= LOG_LEVEL_ORDER.debug) {
    _debugForceLevel = "debug";
  } else if (effectiveOrder <= LOG_LEVEL_ORDER.info) {
    _debugForceLevel = "info";
  } else if (effectiveOrder <= LOG_LEVEL_ORDER.warn) {
    _debugForceLevel = "warn";
  } else {
    _debugForceLevel = "error";
  }
}

const DEBUG_ENABLED: boolean = ((): boolean => {
  const v = process.env.FROSTCLAW_DEBUG;
  if (!v) return false;
  const s = v.toLowerCase();
  return s !== "0" && s !== "false" && s !== "off" && s !== "";
})();

function log(event: string, data?: Record<string, unknown>): void {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  if (_pluginLogger) {
    if (DEBUG_ENABLED) {
      // Force to the most permissive level visible in either file or console
      // output, so debug messages surface wherever they can be seen.
      const method = _pluginLogger[_debugForceLevel];
      if (typeof method === "function") (method as (m: string) => void).call(_pluginLogger, line);
      else _pluginLogger.info(line);
    } else {
      _pluginLogger.debug?.(line);
    }
  } else if (DEBUG_ENABLED) {
    // Pre-registration fallback — only emit if debug is explicitly enabled.
    console.log(line);
  }
}

function logWarn(event: string, data?: Record<string, unknown>): void {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  if (_pluginLogger) {
    _pluginLogger.warn(line);
  } else {
    console.warn(line);
  }
}

function logError(event: string, data?: Record<string, unknown>): void {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  if (_pluginLogger) {
    _pluginLogger.error(line);
  } else {
    console.error(line);
  }
}

// ---------------------------------------------------------------------------
// Request/response debug logging — FROSTCLAW_DEBUG_REQUESTS=1 or "true"
// Read at call time so it can be toggled without restarting anything.
// ---------------------------------------------------------------------------

function isRequestDebugEnabled(): boolean {
  const v = process.env.FROSTCLAW_DEBUG_REQUESTS;
  if (!v) return false;
  const s = v.toLowerCase();
  return s === "1" || s === "true";
}

function assertConfig(): void {
  if (!getApiKey()) {
    throw new Error(
      "[snowflake-cortex] Missing auth token. " +
        "Set SNOWFLAKE_PAT (preferred) or SNOWFLAKE_CORTEX_API_KEY.",
    );
  }
  if (!getBaseURL()) {
    throw new Error(
      "[snowflake-cortex] Missing SNOWFLAKE_BASE_URL. " +
        "Expected: https://<account>.snowflakecomputing.com",
    );
  }
}

// ---------------------------------------------------------------------------
// Anthropic beta headers (anthropic-beta) — Snowflake Cortex constraints.
//
// Snowflake Cortex docs: "Only Bedrock-compatible anthropic-beta header
// values are supported." Cortex rejects unrecognized flags with a 400
// ("invalid beta flag" / "provider rejected the request schema"), so the
// matrix below tracks AWS Bedrock's published Anthropic beta flag list,
// not the broader anthropic.com flag set.
//
// Per Anthropic's Claude 4+ migration guide, several flags we used to send
// are now legacy or model-specific:
//
//   output-128k-2025-02-19          DROPPED.
//                                   Bedrock-listed for Claude 3.7 Sonnet only,
//                                   where it gated 64K → 128K output. On all
//                                   Claude 4+ models 128K output is native and
//                                   driven purely by max_tokens in the body
//                                   (catalog already sets maxTokens: 128_000).
//                                   Migration guide explicitly says:
//                                   "Remove ... output-128k-2025-02-19."
//                                   Sending it to Haiku 4.5 (16K output cap)
//                                   is meaningless. Drop universally.
//
//   token-efficient-tools-2025-02-19  DROPPED.
//                                   Bedrock-listed for Claude 3.7 Sonnet only.
//                                   On Claude 4+ token-efficient tool use is
//                                   built into the model. Migration guide:
//                                   "Remove ... token-efficient-tools-2025-02-19.
//                                   All Claude 4+ models have built-in
//                                   token-efficient tool use." Snowflake
//                                   already rejects it for Haiku.
//
//   effort-2025-11-24                DROPPED.
//                                   Bedrock-listed for Claude Opus 4.5 only.
//                                   GA on Opus 4.6/4.7 (header now no-op there).
//                                   We can't safely send Opus-only flags to
//                                   the Sonnet/Haiku/3.7 catalog entries that
//                                   share this code path; per-model gating
//                                   would require a model-id list and the
//                                   benefit (effort param on a single model
//                                   tier) doesn't justify the surface area.
//
//   tool-examples-2025-10-29         DROPPED.
//                                   Bedrock-listed for Claude Opus 4.5 only.
//                                   Same reasoning as effort-2025-11-24:
//                                   sending an Opus-only flag to Sonnet/
//                                   Haiku/3.7 risks rejection on a strict
//                                   provider, and the feature isn't wired
//                                   into our request shape anyway.
//
//   interleaved-thinking-2025-05-14  KEPT, thinking-only.
//                                   Bedrock-supported on Claude Sonnet 4.5
//                                   and Claude Haiku 4.5 (and Opus 4 family).
//                                   GA on Claude 4.6+ (header still accepted
//                                   but no-op there). Adaptive thinking on
//                                   Opus 4.7 turns it on automatically.
//                                   Only meaningful when thinking is active,
//                                   and only on reasoning-capable models —
//                                   gated on both conditions below.
//
// Buckets:
//   BETA_ALWAYS         — sent on every Claude request. Currently empty:
//                         every legacy "always-safe" flag turned out to be
//                         either model-specific or rendered redundant by the
//                         body's max_tokens.
//   BETA_THINKING       — sent only when extended thinking is active AND
//                         model.reasoning === true. Haiku (reasoning:false)
//                         and any other non-reasoning model never receive
//                         these.
// ---------------------------------------------------------------------------

/** Flags sent only when extended thinking is active on a reasoning-capable model. */
const BETA_THINKING = [
  "interleaved-thinking-2025-05-14",
];

// Pure payload transforms — extracted to src/transforms.ts for testability.
import {
  fixTrailingAssistant,
  fixEmptyTextBlocks,
  stripResponseFormat,
  injectJsonSystemPrompt,
  levelBudget,
  levelEffort,
  normalizeThinkingBudget,
  clampMaxTokens,
  stripEagerInputStreaming,
  stripDocumentBlocks,
  isClaudeModel,
} from "./src/transforms.js";
import { createAssistantMessageEventStream } from "./src/event-stream.js";

/**
 * Returns true for models that support tool calling on the OpenAI Chat
 * Completions path. Claude models are excluded here because they use the
 * Anthropic Messages API, which handles tools natively and independently.
 * Returning true for openai-* here ensures tools aren't stripped for
 * those models. Everything else (Llama, DeepSeek, Mistral, etc.) gets
 * tools stripped via normalizeToolSchemas.
 */
function modelSupportsTools(modelId: string): boolean {
  return modelId.toLowerCase().startsWith("openai-");
}

// Note: Snowflake Cortex only supports the 5-minute ephemeral TTL for prompt
// caching. Injecting ttl:"1h" is not supported and may cause unexpected
// behavior. Cache breakpoints are left as-is.


// ---------------------------------------------------------------------------
// Plugin entry
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Snowflake Cortex Embedding Provider
//
// Snowflake's embed API is NOT OpenAI-compatible:
//   - Request uses `text` (array) instead of `input`
//   - Response wraps vectors in an extra array: [[...]] instead of [...]
// So we use a custom adapter rather than the OpenAI embedding provider.
// ---------------------------------------------------------------------------

// Default model — cheapest, widely available, 768 dims
const DEFAULT_SNOWFLAKE_EMBED_MODEL = "snowflake-arctic-embed-m-v1.5";

async function snowflakeEmbed(
  texts: string[],
  model: string,
): Promise<number[][]> {
  const apiKey = getApiKey();
  const rawBaseUrl = process.env.SNOWFLAKE_PROXY_BASE_URL ?? process.env.SNOWFLAKE_BASE_URL ?? "";
  if (!apiKey || !rawBaseUrl) {
    throw new Error(
      "[snowflake-cortex] Missing SNOWFLAKE_BASE_URL or SNOWFLAKE_CORTEX_API_KEY",
    );
  }

  // Always call inference:embed natively — same code path regardless of
  // whether SNOWFLAKE_PROXY_BASE_URL is set. In proxy mode rawBaseUrl is
  // already the proxy base (e.g. http://127.0.0.1:18790/api/v2/cortex), so
  // this resolves to http://proxy/api/v2/cortex/inference:embed. In direct
  // mode it resolves to https://<acct>.snowflakecomputing.com/api/v2/cortex/inference:embed.
  const url = `${rawBaseUrl.replace(/\/$/, "")}/api/v2/cortex/inference:embed`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ text: texts, model }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `[snowflake-cortex] Embed request failed (${res.status}): ${body}`,
    );
  }

  const json = await res.json() as {
    data: Array<{ embedding: number[][] | number[]; index: number }>;
  };

  // Snowflake wraps each vector in an extra array: [[vec]] — flatten if needed
  return json.data
    .sort((a, b) => a.index - b.index)
    .map(({ embedding }) =>
      Array.isArray(embedding[0]) ? (embedding as number[][])[0] : (embedding as number[]),
    );
}

const snowflakeCortexEmbeddingAdapter: EmbeddingProviderAdapter = {
  id: "snowflake-cortex",
  defaultModel: DEFAULT_SNOWFLAKE_EMBED_MODEL,
  transport: "remote",

  async create(options: EmbeddingProviderCreateOptions) {
    const model = options.model || DEFAULT_SNOWFLAKE_EMBED_MODEL;
    const hasKey = !!getApiKey();
    const hasBaseUrl = !!getBaseURL();
    log("embedding.create", { model, hasKey, hasBaseUrl });

    if (!hasKey || !hasBaseUrl) {
      log("embedding.create returning null — missing config");
      return { provider: null };
    }

    return {
      provider: {
        id: "snowflake-cortex",
        model,
        maxInputTokens: 4096,
        embed: (input: EmbeddingInput, _options?: EmbeddingProviderCallOptions) => {
          const text = typeof input === "string" ? input : input.text;
          return snowflakeEmbed([text], model).then((v) => v[0]);
        },
        embedBatch: (inputs: EmbeddingInput[], _options?: EmbeddingProviderCallOptions) => {
          const texts = inputs.map((i) => (typeof i === "string" ? i : i.text));
          return snowflakeEmbed(texts, model);
        },
      },
    };
  },
};

export default definePluginEntry({
  id: "snowflake-cortex",
  name: "Snowflake Cortex",
  description:
    "Snowflake Cortex AI — routes Claude models to Anthropic Messages API " +
    "and all other models to OpenAI-compatible Chat Completions, both " +
    "behind PAT authentication.",

  register(api) {
    try {
      // Wire in the OpenClaw-scoped logger immediately so all subsequent
      // log calls route through it and respect openclaw.json logging.level
      // and logging.consoleLevel.
      setPluginLogger(api.logger, api.config);
      log("plugin registered");

      // Install fetch interceptor to transparently retry Snowflake empty-200
      // SSE responses and configure long-lived streaming connections.
      //
      // URL matching covers both Snowflake Cortex API paths:
      //   - /api/v2/cortex/v1/messages        (Anthropic Messages API, direct or via proxy)
      //   - /api/v2/cortex/v1/chat/completions (OpenAI Chat Completions API)
      // Both are POST endpoints that return text/event-stream.
      // Using full path strings rather than two separate includes() guards so
      // the match is unambiguous regardless of base URL configuration.
      //
      // The marker is versioned so a gateway hot-reload installs a fresh
      // interceptor rather than reusing a stale one from a previous plugin load.
      const FETCH_INTERCEPT_MARKER = Symbol.for("frostclaw.fetchIntercepted.v2");
      if (!(globalThis as Record<symbol, unknown>)[FETCH_INTERCEPT_MARKER]) {
        (globalThis as Record<symbol, unknown>)[FETCH_INTERCEPT_MARKER] = true;
        const originalFetch = globalThis.fetch;

        // Configure a long-lived dispatcher for Snowflake inference endpoints.
        // bodyTimeout:0 prevents undici from dropping the connection mid-stream
        // during extended thinking phases where Snowflake emits no SSE tokens
        // for many seconds. headersTimeout covers the initial connection setup.
        // keepAliveTimeout:90s outlasts the 60s undici idle window to avoid
        // "terminated" errors from keep-alive races on reused connections.
        let _snowflakeDispatcher: unknown = undefined;
        try {
          // undici ships with openclaw's node_modules. Use an opaque runtime
          // require (Function constructor) so bun's bundler does not statically
          // analyse or inline the module — which would pull in undici's optional
          // node:sqlite cache-store dep and fail the build.
          // Build a working require() without an `import` of node:module — bun's
          // bundler (default target) stubs `import {createRequire} from "node:module"`
          // to an empty object, and `new Function("p","return require(p)")` throws
          // "require is not defined" in ESM. process.getBuiltinModule is a runtime
          // accessor (Node 22+) the bundler cannot stub. The undici path is passed
          // at runtime so undici is never statically inlined (which would pull in
          // its optional node:sqlite cache-store dep and fail the build).
          const _nodeModule = (process as unknown as { getBuiltinModule: (m: string) => { createRequire: (u: string) => (p: string) => Record<string, unknown> } }).getBuiltinModule("module");
          const _runtimeRequire = _nodeModule.createRequire(import.meta.url);
          const undici = _runtimeRequire("/home/ubuntu/.npm-global/lib/node_modules/openclaw/node_modules/undici") as Record<string, unknown>;
          _snowflakeDispatcher = new undici.Agent({
            headersTimeout: 30_000,
            bodyTimeout: 0,          // no timeout on response body stream
            keepAliveTimeout: 90_000,
            keepAliveMaxTimeout: 300_000,
          });
          _pluginLogger?.info("[frostclaw:fetch] undici Agent dispatcher configured (bodyTimeout=0, keepAlive=90s)");
        } catch (e) {
          _pluginLogger?.warn(`[frostclaw:fetch] undici not available, using default dispatcher: ${e}`);
        }

        globalThis.fetch = async function frostclawFetch(input, init) {
          const url = typeof input === "string" ? input : (input as Request).url;
          const method = (init?.method ?? "GET").toUpperCase();
          const isSnowflakeInference =
            method === "POST" &&
            (url.includes("/api/v2/cortex/v1/messages") ||
             url.includes("/api/v2/cortex/v1/chat/completions"));
          if (!isSnowflakeInference) {
            return originalFetch(input, init);
          }

          // Attach the long-lived dispatcher for Snowflake inference calls.
          if (_snowflakeDispatcher) {
            init = { ...init, dispatcher: _snowflakeDispatcher } as typeof init;
          }

          // Detect SSE format: Anthropic emits content_block_start/message_stop;
          // OpenAI emits data: {choices:[{delta:{content:...}}]} / data: [DONE].
          const isAnthropicFormat = url.includes("/api/v2/cortex/v1/messages");

          // Fix auth headers for Snowflake Cortex requests that arrive via direct
          // model calls (e.g. the pdf tool's complete() path bypasses wrapStreamFn
          // and the Anthropic SDK sends x-api-key instead of Authorization: Bearer,
          // which Snowflake rejects with 401).
          // Only applies when x-api-key is present but Authorization is absent,
          // so normal agent turns — which already have correct headers from
          // wrapStreamFn — pass through unchanged.
          {
            const rawHeaders = init?.headers;
            const getHeader = (name: string): string | null => {
              if (!rawHeaders) return null;
              if (rawHeaders instanceof Headers) return rawHeaders.get(name);
              const rec = rawHeaders as Record<string, string>;
              const lower = name.toLowerCase();
              for (const key of Object.keys(rec)) {
                if (key.toLowerCase() === lower) return rec[key];
              }
              return null;
            };
            const xApiKey = getHeader("x-api-key");
            const authorization = getHeader("Authorization");
            if (xApiKey && !authorization) {
              const newHeaders: Record<string, string> = {};
              if (rawHeaders instanceof Headers) {
                rawHeaders.forEach((value, key) => {
                  if (key.toLowerCase() !== "x-api-key") newHeaders[key] = value;
                });
              } else {
                for (const [key, value] of Object.entries(rawHeaders as Record<string, string>)) {
                  if (key.toLowerCase() !== "x-api-key") newHeaders[key] = value as string;
                }
              }
              newHeaders["Authorization"] = `Bearer ${xApiKey}`;
              newHeaders["X-Snowflake-Authorization-Token-Type"] = "PROGRAMMATIC_ACCESS_TOKEN";
              init = { ...init, headers: newHeaders };
              _pluginLogger?.info("[frostclaw:fetch] patched x-api-key → Bearer auth for Snowflake Cortex direct call");
            }
          }

          // Ensure the body is reusable across retries.
          // The Anthropic SDK serializes bodies as JSON strings, but guard against
          // the stream case defensively.
          let bodyForRetry: BodyInit | null | undefined = init?.body;
          if (bodyForRetry instanceof ReadableStream) {
            const chunks: Uint8Array[] = [];
            const reader = (bodyForRetry as ReadableStream<Uint8Array>).getReader();
            while (true) {
              const { value, done } = await reader.read();
              if (done) break;
              if (value) chunks.push(value);
            }
            const total = chunks.reduce((a, c) => a + c.length, 0);
            const merged = new Uint8Array(total);
            let off = 0;
            for (const c of chunks) { merged.set(c, off); off += c.length; }
            bodyForRetry = merged;
            _pluginLogger?.info("[frostclaw:fetch] body was ReadableStream — buffered to Uint8Array for retry safety");
          }

          const FETCH_MAX_RETRIES = 2;
          for (let attempt = 0; attempt <= FETCH_MAX_RETRIES; attempt++) {
            const retryInit = bodyForRetry !== init?.body
              ? { ...init, body: bodyForRetry }
              : init;
            const response = await originalFetch(input, retryInit);

            // --- Non-2xx handling ---
            // Snowflake status codes that are retryable:
            //   400 "all requests were throttled by remote service"
            //   402 budget exceeded
            //   429 too many requests
            //   503 inference timed out
            // All other non-2xx pass through immediately (auth errors, bad schema, etc.)
            if (!response.ok) {
              const errorBody = await response.text().catch(() => "");
              const isThrottled400 = response.status === 400 &&
                errorBody.toLowerCase().includes("throttled");
              const isBudget402 = response.status === 402;
              const isRateLimit429 = response.status === 429;
              const isTimeout503 = response.status === 503;
              // HTTP/2 GOAWAY / Snowflake LB connection-cycle (error_code 392606).
              // Surfaces as a 5xx or 200-with-error-body; retry on fresh connection.
              const isGoaway = errorBody.includes("GOAWAY") || errorBody.includes("392606");
              const retryable = isThrottled400 || isBudget402 || isRateLimit429 || isTimeout503 || isGoaway;
              if (retryable && attempt < FETCH_MAX_RETRIES) {
                _pluginLogger?.warn(
                  `[frostclaw:fetch] retryable HTTP ${response.status} (attempt ${attempt + 1}/${FETCH_MAX_RETRIES + 1}), retrying... body=${errorBody.slice(0, 300)}`
                );
                await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
                continue;
              }
              _pluginLogger?.warn(
                `[frostclaw:fetch] non-2xx HTTP ${response.status}${retryable ? " (retries exhausted)" : " (non-retryable)"} body=${errorBody.slice(0, 300)}`
              );
              // Reconstruct the response so the body is still readable downstream.
              return new Response(errorBody, {
                status: response.status,
                statusText: response.statusText,
                headers: response.headers,
              });
            }

            const ct = response.headers.get("content-type") ?? "";
            if (!ct.includes("text/event-stream") || !response.body) {
              return response;
            }

            const chunks: Uint8Array[] = [];
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulated = "";
            let hasContentBlock = false;
            let hasMessageStop = false;
            let done = false;

            while (!done && !hasContentBlock && !hasMessageStop) {
              const { value, done: readerDone } = await reader.read();
              done = readerDone;
              if (value) {
                chunks.push(value);
                accumulated += decoder.decode(value, { stream: true });
                if (isAnthropicFormat) {
                  // Anthropic SSE: content_block_start indicates real content;
                  // message_stop without it is an empty response.
                  if (accumulated.includes("content_block_start")) hasContentBlock = true;
                  if (accumulated.includes("message_stop")) hasMessageStop = true;
                } else {
                  // OpenAI SSE: any delta with non-empty content is real;
                  // [DONE] without it is an empty response.
                  if (/"delta"\s*:\s*\{[^}]*"content"\s*:\s*"[^"]+"/.test(accumulated)) hasContentBlock = true;
                  if (accumulated.includes("[DONE]")) hasMessageStop = true;
                }
              }
            }

            if (hasContentBlock || !hasMessageStop) {
              const remaining = reader;
              const combined = new ReadableStream<Uint8Array>({
                async start(controller) {
                  for (const chunk of chunks) controller.enqueue(chunk);
                  while (true) {
                    const { value, done } = await remaining.read();
                    if (done) break;
                    if (value) controller.enqueue(value);
                  }
                  controller.close();
                },
              });
              return new Response(combined, {
                status: response.status,
                statusText: response.statusText,
                headers: response.headers,
              });
            }

            // Empty-stop detected (HTTP 200 with no content blocks) — log raw SSE for diagnosis
            reader.cancel();
            const rawSse = accumulated.replace(/\n/g, "\\n").slice(0, 800);
            if (attempt < FETCH_MAX_RETRIES) {
              _pluginLogger?.warn(
                `[frostclaw:fetch] empty-stop detected (attempt ${attempt + 1}/${FETCH_MAX_RETRIES + 1}), retrying... raw=${rawSse}`
              );
              await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
              continue;
            }

            _pluginLogger?.warn(
              `[frostclaw:fetch] empty-stop persists after ${FETCH_MAX_RETRIES} retries, passing through raw=${rawSse}`
            );
            const emptyStream = new ReadableStream<Uint8Array>({
              start(controller) {
                for (const chunk of chunks) controller.enqueue(chunk);
                controller.close();
              },
            });
            return new Response(emptyStream, {
              status: response.status,
              statusText: response.statusText,
              headers: response.headers,
            });
          }
          return originalFetch(input, init);
        };
        _pluginLogger?.info("[frostclaw:fetch] empty-200 retry interceptor installed");
      } else {
        _pluginLogger?.info("[frostclaw:fetch] interceptor already installed, skipping");
      }

      api.registerEmbeddingProvider(snowflakeCortexEmbeddingAdapter);
      api.registerProvider({
      id: "snowflake-cortex",
      label: "Snowflake Cortex",

      auth: [
        createProviderApiKeyAuthMethod({
          providerId: "snowflake-cortex",
          methodId: "snowflake-pat",
          label: "Snowflake PAT",
          hint: "Programmatic Access Token for Snowflake Cortex",
          optionKey: "snowflakePat",
          flagName: "--snowflake-pat",
          envVar: "SNOWFLAKE_CORTEX_API_KEY",
          promptMessage:
            "Enter your Snowflake Programmatic Access Token (PAT):",
        }),
      ],

      catalog: {
        runtimeAugment: true,
        run: async (ctx) => {
          try {
            const resolved = ctx.resolveProviderApiKey("snowflake-cortex");
            const resolvedKey = resolved.apiKey ?? getApiKey();
            const envKey = getApiKey();
            const baseURL = getBaseURL();

            log("catalog.run", {
              resolvedKeyPresent: !!resolved.apiKey,
              resolvedKeyLength: resolved.apiKey?.length ?? 0,
              envKeyPresent: !!envKey,
              envKeyLength: envKey.length,
              baseURL: baseURL || "(not set)",
            });

            if (!resolvedKey || !baseURL) {
              log("catalog.run returning null — missing config", {
                resolvedKey: !!resolvedKey,
                baseURL: !!baseURL,
              });
              return null;
            }

            const models = buildModelCatalog();
            log("catalog.run returning catalog", { modelCount: models.length });
            return {
              provider: {
                baseUrl: baseURL,
                apiKey: resolvedKey,
                api: "openai-completions" as ModelApi,
                authHeader: true,
                models,
              },
            };
          } catch (err) {
            logError("catalog.run ERROR", {
              error: String(err),
              stack: err instanceof Error ? err.stack : undefined,
            });
            throw err;
          }
        },
      },

      // -----------------------------------------------------------------------
      // Hook: Resolve model API family when catalog hasn't loaded yet.
      //
      // Returns the full catalog ModelDefinitionConfig when we recognize the
      // model id. This is important: openclaw's applyConfiguredProviderOverrides
      // (model-iYcWZTML.js:380-394) spreads the discoveredModel and pulls
      // `cost`, `maxTokens`, `contextWindow`, `reasoning`, `compat` directly
      // from it — fields we omit here simply never reach the downstream
      // pipeline, causing pi-ai's calculateCost to warn `model.cost missing`
      // and openclaw's resolveAnthropicMessagesMaxTokens to emit `max_tokens: 0`.
      //
      // For unknown ids we fall back to a minimal stub so the request doesn't
      // crash: openclaw's provider-stream reads `model.input.includes(...)`
      // and pi-ai's sdk reads `model.baseUrl.includes(...)` unconditionally.
      // Claude models support text + image; other unknown ids default to
      // text-only since dropping images is safer than crashing.
      // -----------------------------------------------------------------------
      resolveDynamicModel(ctx) {
        const modelId = ctx.modelId;
        if (!modelId) {
          log("resolveDynamicModel: no modelId");
          return null;
        }

        const catalogEntry = findCatalogEntry(modelId);
        if (catalogEntry) {
          log("resolveDynamicModel (catalog hit)", {
            modelId,
            api: catalogEntry.api,
            contextWindow: catalogEntry.contextWindow,
            maxTokens: catalogEntry.maxTokens,
            hasCost: !!catalogEntry.cost,
          });
          return catalogEntry;
        }

        const claude = isClaudeModel(modelId);
        const api: ModelApi = claude ? "anthropic-messages" : "openai-completions";
        const input: Array<"text" | "image"> = claude ? ["text", "image"] : ["text"];
        const baseUrl = getCatalogBaseURL();
        log("resolveDynamicModel (unknown id, minimal stub)", { modelId, api, input, baseUrl });
        return { id: modelId, name: modelId, api, input, baseUrl };
      },

      // -----------------------------------------------------------------------
      // Hook: Strip tools for models that don't support them
      // -----------------------------------------------------------------------
      normalizeToolSchemas(ctx: ProviderNormalizeToolSchemasContext) {
        if (!ctx.modelId) return ctx.tools;
        if (isClaudeModel(ctx.modelId)) {
          // Strip eager_input_streaming from tool schemas — Snowflake Cortex
          // rejects this field on Haiku (and may reject it on Sonnet in future).
          // OpenClaw never consumes this field client-side (FGTS uses the beta
          // header, not the per-tool field), so stripping is always safe.
          return ctx.tools.map((tool: Record<string, unknown>) => {
            const custom = tool.custom as Record<string, unknown> | undefined;
            if (!custom || !("eager_input_streaming" in custom)) return tool;
            const { eager_input_streaming: _dropped, ...rest } = custom;
            if (Object.keys(rest).length > 0) return { ...tool, custom: rest };
            const { custom: _c, ...toolWithoutCustom } = tool;
            return toolWithoutCustom;
          });
        }
        if (!modelSupportsTools(ctx.modelId)) return [];
        return ctx.tools;
      },

      // -----------------------------------------------------------------------
      // Hook: Inject Snowflake PAT header type and conditionally add thinking
      // beta flags based on ctx.thinkingLevel for Claude models.
      //
      // Beta flags are split: always-safe flags come from the catalog headers,
      // while thinking-specific flags (interleaved-thinking, effort,
      // tool-examples) are only added when the request uses thinking. This
      // prevents Snowflake Cortex from rejecting unknown beta flags with 400.
      // -----------------------------------------------------------------------
      wrapStreamFn(ctx: ProviderWrapStreamFnContext) {
        log("wrapStreamFn", {
          modelId: (ctx as unknown as Record<string, unknown>).modelId as string | undefined,
          thinkingLevel: ctx.thinkingLevel,
          thinkingActive: ctx.thinkingLevel !== undefined && ctx.thinkingLevel !== "off",
          hasStreamFn: !!ctx.streamFn,
        });
        if (!ctx.streamFn) return undefined;

        const inner = ctx.streamFn;

        // OpenClaw re-invokes wrapStreamFn(ctx) fresh per request, so
        // ctx.thinkingLevel is current here — not stale from registration time.
        // thinkingLevel must be read from the outer ctx; the inner StreamFn's
        // context param (messages, tools, systemPrompt) does not carry it.
        const thinkingActive =
          ctx.thinkingLevel !== undefined && ctx.thinkingLevel !== "off";
        const thinkingLevel = ctx.thinkingLevel;

        return (model, context, options) => {
          try {
            // Guard: two known openclaw/pi-ai sites dereference model fields
            // unconditionally and crash on undefined:
            //   1. provider-stream-CQzDRxyR.js:154 reads model.input.includes(...)
            //   2. pi-coding-agent/dist/core/sdk.js:31 reads model.baseUrl.includes(...)
            //      after `provider !== "openrouter"` (no short-circuit on baseUrl).
            // Patch both if missing so the downstream pipeline never sees an
            // illegal state.
            const modelObj = model as {
              id?: unknown;
              api?: unknown;
              input?: unknown;
              baseUrl?: unknown;
            } | undefined;
            if (modelObj && !Array.isArray(modelObj.input)) {
              const id = String(modelObj.id ?? "");
              const inferred: Array<"text" | "image"> = isClaudeModel(id)
                ? ["text", "image"]
                : ["text"];
              log("wrapStreamFn.inner: patching missing input", {
                modelId: id,
                inferred,
                priorInput: modelObj.input,
                keys: Object.keys(modelObj as Record<string, unknown>),
              });
              (modelObj as Record<string, unknown>).input = inferred;
            }
            if (modelObj && typeof modelObj.baseUrl !== "string") {
              const fallbackBaseUrl = getCatalogBaseURL();
              log("wrapStreamFn.inner: patching missing baseUrl", {
                modelId: String(modelObj.id ?? ""),
                fallbackBaseUrl,
                priorBaseUrl: modelObj.baseUrl,
              });
              (modelObj as Record<string, unknown>).baseUrl = fallbackBaseUrl;
            }
            log("wrapStreamFn.inner", {
              modelId: modelObj?.id,
              modelApi: modelObj?.api,
              modelInput: modelObj?.input,
              modelBaseUrl: modelObj?.baseUrl,
              modelKeys: modelObj
                ? Object.keys(modelObj as Record<string, unknown>)
                : undefined,
              hasContext: !!context,
              hasOptions: !!options,
              messageCount: Array.isArray((context as Record<string, unknown> | undefined)?.messages)
                ? ((context as Record<string, unknown>).messages as unknown[]).length
                : undefined,
            });
            const originalOnPayload = options?.onPayload;

            // Build per-request anthropic-beta header: start with any
            // catalog-level flags from the model definition, then append
            // thinking flags when the request actually uses thinking.
            const catalogBeta =
              (model as { headers?: Record<string, string> })?.headers?.[
                "anthropic-beta"
              ] ?? "";
            const betaFlags = catalogBeta ? [catalogBeta] : [];
            // Per-request, append BETA_THINKING only when extended thinking
            // is active and the model is reasoning-capable. Catalog headers
            // (BETA_ALWAYS) are seeded above; everything else lives here.
            // Haiku 4.5 (reasoning:false) never receives thinking flags.
            const modelSupportsReasoning =
              (modelObj as { reasoning?: boolean } | undefined)?.reasoning === true;
            if (thinkingActive && modelSupportsReasoning) {
              betaFlags.push(BETA_THINKING.join(","));
            }

            // Snowflake Cortex requires `Authorization: Bearer <PAT>` for
            // every request, including its Anthropic-Messages endpoint. The
            // SDK's Anthropic transport (provider-stream:335) instead sends
            // `x-api-key: <key>`, which Snowflake rejects with
            //   400 Bearer token is missing
            // So for Claude models we must attach the Bearer header ourselves.
            // Non-Claude (openai-completions) routes already get Bearer from
            // the SDK's default flow and need no patching here.
            const modelId = String((model as { id?: unknown })?.id ?? "");
            const isClaudeRoute = isClaudeModel(modelId);
            const optionsApiKey =
              (options as { apiKey?: unknown } | undefined)?.apiKey;
            const bearerKey =
              typeof optionsApiKey === "string" && optionsApiKey.length > 0
                ? optionsApiKey
                : getApiKey();
            const authHeader =
              isClaudeRoute && bearerKey
                ? { Authorization: `Bearer ${bearerKey}` }
                : {};

            // Log request info before calling inner() — stream-level debug logging
            if (isRequestDebugEnabled()) {
              const ts = new Date().toISOString();
              const thinkingInfo = thinkingActive ? `level=${thinkingLevel}` : "off";
              const ctxRecord = context as Record<string, unknown> | undefined;
              const msgCount = Array.isArray(ctxRecord?.messages)
                ? (ctxRecord!.messages as unknown[]).length
                : 0;
              const sysPrompt = typeof ctxRecord?.systemPrompt === "string"
                ? ctxRecord!.systemPrompt as string
                : "";
              const maxTok = (options as Record<string, unknown> | undefined)?.maxTokens;
              _pluginLogger?.info(
                `[frostclaw:debug] ${ts} → Snowflake | model=${modelId} | attempt=1 | messages=${msgCount} | maxTokens=${maxTok} | thinking=${thinkingInfo} | systemPromptChars=${sysPrompt.length}`
              );
            }

            const merged = {
              ...options,
              headers: {
                ...options?.headers,
                "X-Snowflake-Authorization-Token-Type":
                  "PROGRAMMATIC_ACCESS_TOKEN",
                ...authHeader,
                ...(betaFlags.length > 0
                  ? { "anthropic-beta": betaFlags.join(",") }
                  : {}),
              },
              onPayload: (payload: unknown, payloadModel: unknown) => {
                const payloadModelObj = payloadModel as { id?: unknown } | undefined;
                log("onPayload", {
                  payloadType: typeof payload,
                  isObject: payload !== null && typeof payload === "object",
                  payloadModelId: payloadModelObj?.id,
                  isClaudeModelResult: payload && typeof payload === "object"
                    ? isClaudeModel(String((model as { id?: unknown })?.id ?? ""))
                    : false,
                });
                if (
                  payload &&
                  typeof payload === "object" &&
                  isClaudeModel(String((model as { id?: unknown })?.id ?? ""))
                ) {
                  const record = payload as Record<string, unknown>;
                  if (Array.isArray(record.messages)) {
                    record.messages = fixTrailingAssistant(record.messages);
                    record.messages = fixEmptyTextBlocks(record.messages);
                    // Strip native PDF/document blocks — Snowflake Cortex rejects
                    // them with HTTP 401 even though it runs Claude models. The
                    // Anthropic document block type is not part of Snowflake's
                    // API surface. Each stripped block becomes a text placeholder.
                    record.messages = stripDocumentBlocks(record.messages);
                  }
                  // Defensive: strip `eager_input_streaming` from tool schemas.
                  // The catalog sets supportsEagerToolInputStreaming: false to
                  // prevent pi-ai's Anthropic provider from adding it, but we
                  // also scrub it here so any future SDK regression or alternate
                  // code path can't re-introduce a Cortex-fatal field.
                  stripEagerInputStreaming(record);
                  // Strip response_format — Snowflake Cortex rejects both
                  // json_object and json_schema response_format values with
                  // HTTP 400. When we strip it we inject a strong system
                  // prompt so the model still returns valid JSON without
                  // the API-level constraint.
                  const stripped = stripResponseFormat(record);
                  if (Array.isArray(record.messages)) {
                    record.messages = injectJsonSystemPrompt(
                      record.messages,
                      stripped,
                    );
                  }
                  normalizeThinkingBudget(
                    record,
                    thinkingLevel,
                    isAdaptiveOnly(String((model as { id?: unknown })?.id ?? "")),
                  );
                  clampMaxTokens(record);
                }
                const chained = (originalOnPayload as
                  | ((p: unknown, m: unknown) => unknown)
                  | undefined)?.(payload, payloadModel);
                // Must return the (possibly mutated) payload so OpenClaw
                // uses it. If a chained handler returned something, prefer
                // that; otherwise return our mutated payload directly.
                return chained !== undefined ? chained : payload;
              },
            };
            const streamResult = inner(model, context, merged as typeof options);
            // Snowflake returns HTTP 200 with empty content + stop_reason="stop"
            // when the account hits a concurrency/rate limit instead of returning
            // a proper 429. Detect this and retry transparently.
            //
            // Two invocation modes:
            //   (a) Promise path — streamFn returned a Promise (test mocks, some
            //       alternative transports). Await the result then inspect.
            //   (b) Stream path — streamFn returned an AssistantMessageEventStream
            //       (the real OpenClaw transport: always synchronous stream +
            //       background async HTTP task). Forward events in real-time; after
            //       the stream settles inspect the final message for empty stop.
            //
            // Empty-stop forms:
            //   1. content array completely empty  → pure Snowflake overload
            //   2. content only thinking blocks    → thinking-only overload signal
            const EMPTY_STOP_MAX_RETRIES = 2;
            // Backoff between retries (ms): 5s, 10s
            const RETRY_BACKOFF_MS = (attempt: number) => 5_000 * (attempt + 1);

            // Helper: is an error a retryable network/timeout error from Snowflake?
            function isRetryableError(err: unknown): boolean {
              if (!err) return false;
              const msg = String(err);
              // Anthropic SDK timeout
              if (msg.includes("APIConnectionTimeoutError") || msg.includes("Connection timeout")) return true;
              // Generic timeout strings from SDK / undici
              if (/timed?\s?out/i.test(msg)) return true;
              // Node.js fetch / undici network errors
              if (msg.includes("ECONNRESET") || msg.includes("ECONNREFUSED") || msg.includes("ENOTFOUND")) return true;
              if (msg.includes("UND_ERR_SOCKET") || msg.includes("UND_ERR_CONNECT_TIMEOUT") || msg.includes("UND_ERR_HEADERS_TIMEOUT") || msg.includes("UND_ERR_BODY_TIMEOUT")) return true;
              if (msg.includes("AbortError") || msg.includes("The operation was aborted")) return true;
              // Fetch-level
              if (msg.includes("network error") || msg.includes("fetch failed")) return true;
              // undici bare connection drop — surfaces as "TypeError: terminated"
              // when Snowflake closes a keep-alive connection mid-stream.
              if (/\bterminated\b/i.test(msg)) return true;
              // HTTP/2 GOAWAY — Snowflake's LB gracefully cycles the connection
              // (server-side, not a rate limit). Retry on a fresh connection.
              if (msg.includes("GOAWAY") || msg.includes("392606")) return true;
              return false;
            }

            // Helper: is a settled AssistantMessage an empty-stop that warrants retry?
            function isEmptyStop(
              msg: { stopReason?: unknown; content?: unknown } | undefined,
              attempt: number
            ): boolean {
              if (!msg || typeof msg !== "object") return false;
              // Accept stopReason="stop" OR stopReason=undefined/null:
              // Snowflake omits message_delta entirely when rate-limiting,
              // leaving stopReason undefined rather than "stop".
              const sr = msg.stopReason;
              if (sr !== "stop" && sr !== undefined && sr !== null) return false;
              if (attempt >= EMPTY_STOP_MAX_RETRIES) return false;
              if (!Array.isArray(msg.content)) return false;
              if (msg.content.length === 0) return true;
              // thinking-only
              return (msg.content as Array<{ type?: string }>).every(
                (blk) => blk?.type === "thinking"
              );
            }

            // (a) Promise path — has .then but NOT .result (AssistantMessageEventStream
            // has both; use .result presence to distinguish stream from plain Promise)
            if (
              streamResult &&
              typeof (streamResult as { then?: unknown }).then === "function" &&
              typeof (streamResult as { result?: unknown }).result !== "function"
            ) {
              return (async () => {
                let currentResult = streamResult as Promise<unknown>;
                for (let attempt = 0; attempt <= EMPTY_STOP_MAX_RETRIES; attempt++) {
                  if (attempt > 0 && isRequestDebugEnabled()) {
                    const ts = new Date().toISOString();
                    const ctxRecord2 = context as Record<string, unknown> | undefined;
                    const msgCount2 = Array.isArray(ctxRecord2?.messages)
                      ? (ctxRecord2!.messages as unknown[]).length : 0;
                    const maxTok2 = (options as Record<string, unknown> | undefined)?.maxTokens;
                    const thinkingInfo2 = thinkingActive ? `level=${thinkingLevel}` : "off";
                    const sysPrompt2 = typeof ctxRecord2?.systemPrompt === "string"
                      ? ctxRecord2!.systemPrompt as string : "";
                    _pluginLogger?.info(
                      `[frostclaw:debug] ${ts} → Snowflake | model=${modelId} | attempt=${attempt + 1} | messages=${msgCount2} | maxTokens=${maxTok2} | thinking=${thinkingInfo2} | systemPromptChars=${sysPrompt2.length}`
                    );
                  }
                  let value: unknown;
                  try {
                    value = await currentResult;
                  } catch (err) {
                    if (isRetryableError(err) && attempt < EMPTY_STOP_MAX_RETRIES) {
                      logWarn("wrapStreamFn.promise: retryable error from Snowflake, retrying", {
                        modelId,
                        attempt,
                        error: String(err),
                      });
                      await new Promise((r) => setTimeout(r, RETRY_BACKOFF_MS(attempt)));
                      const retryResult = inner(model, context, merged as typeof options);
                      if (retryResult && typeof (retryResult as { then?: unknown }).then === "function") {
                        currentResult = retryResult as Promise<unknown>;
                        continue;
                      }
                      return retryResult;
                    }
                    logError("wrapStreamFn.promise REJECTED", {
                      error: String(err),
                      stack: err instanceof Error ? err.stack : undefined,
                    });
                    throw err;
                  }
                  const valueObj = value as
                    | { errorMessage?: unknown; stopReason?: unknown; content?: unknown }
                    | undefined;
                  if (
                    valueObj &&
                    typeof valueObj === "object" &&
                    (valueObj.errorMessage || valueObj.stopReason === "error")
                  ) {
                    logWarn("wrapStreamFn.promise resolved with error", {
                      errorMessage: valueObj.errorMessage,
                      stopReason: valueObj.stopReason,
                    });
                  }
                  if (isEmptyStop(valueObj, attempt)) {
                    const isThinkingOnly =
                      Array.isArray(valueObj?.content) &&
                      (valueObj!.content as unknown[]).length > 0;
                    logWarn("wrapStreamFn.promise: empty/thinking-only stop from Snowflake, retrying", {
                      modelId,
                      attempt,
                      isThinkingOnly,
                      contentLength: Array.isArray(valueObj?.content) ? (valueObj!.content as unknown[]).length : 0,
                    });
                    const retryResult = inner(model, context, merged as typeof options);
                    if (retryResult && typeof (retryResult as { then?: unknown }).then === "function") {
                      currentResult = retryResult as Promise<unknown>;
                      continue;
                    }
                    // Retry returned a stream — fall through to stream path below
                    return retryResult;
                  }
                  if (isRequestDebugEnabled()) {
                    const ts = new Date().toISOString();
                    const finalObj = valueObj as { stopReason?: unknown; content?: unknown } | undefined;
                    const contentBlocks = Array.isArray(finalObj?.content) ? (finalObj!.content as unknown[]).length : 0;
                    _pluginLogger?.info(
                      `[frostclaw:debug] ${ts} ← Snowflake | model=${modelId} | attempt=${attempt + 1} | stop_reason=${finalObj?.stopReason} | content_blocks=${contentBlocks} | retry=false`
                    );
                  }
                  return value;
                }
              })() as typeof streamResult;
            }

            // (b) Stream path — AssistantMessageEventStream (real transport)
            // Forward events to a new outer stream as they arrive. After the
            // inner stream settles, check the final message for empty-stop and
            // retry by pumping a fresh inner stream into the same outer stream.
            if (
              streamResult &&
              typeof (streamResult as { result?: unknown })[Symbol.asyncIterator as unknown as string] === "function" &&
              typeof (streamResult as { result?: unknown }).result === "function"
            ) {
              const outerStream = createAssistantMessageEventStream();
              void (async () => {
                let currentStream = streamResult as AsyncIterable<unknown> & { result: () => Promise<unknown> };
                for (let attempt = 0; attempt <= EMPTY_STOP_MAX_RETRIES; attempt++) {
                  // Log each attempt request (attempt>0 = retry)
                  if (attempt > 0 && isRequestDebugEnabled()) {
                    const ts = new Date().toISOString();
                    const ctxRecord3 = context as Record<string, unknown> | undefined;
                    const msgCount3 = Array.isArray(ctxRecord3?.messages)
                      ? (ctxRecord3!.messages as unknown[]).length : 0;
                    const maxTok3 = (options as Record<string, unknown> | undefined)?.maxTokens;
                    const thinkingInfo3 = thinkingActive ? `level=${thinkingLevel}` : "off";
                    const sysPrompt3 = typeof ctxRecord3?.systemPrompt === "string"
                      ? ctxRecord3!.systemPrompt as string : "";
                    _pluginLogger?.info(
                      `[frostclaw:debug] ${ts} → Snowflake | model=${modelId} | attempt=${attempt + 1} | messages=${msgCount3} | maxTokens=${maxTok3} | thinking=${thinkingInfo3} | systemPromptChars=${sysPrompt3.length}`
                    );
                  }
                  // Buffer events only until we confirm the stream isn't empty.
                  // For non-empty responses: forward buffered header events then
                  // switch to pass-through. For empty responses: discard buffer
                  // and retry — no events reach the outer stream from this attempt.
                  const buffer: unknown[] = [];
                  let hasContent = false;
                  let sseSeq = 0;
                  let _dbgContentLen = 0;
                  try {
                    for await (const event of currentStream) {
                      const evType = (event as { type?: string })?.type;
                      // Stream-level debug logging
                      if (isRequestDebugEnabled()) {
                        const evObj = event as Record<string, unknown>;
                        if (evType === "message_start") {
                          const usage = evObj.usage as Record<string, unknown> | undefined;
                          _pluginLogger?.info(
                            `[frostclaw:debug] SSE[${sseSeq++}]: ${evType} | input_tokens=${usage?.input_tokens} | cache_read_input_tokens=${usage?.cache_read_input_tokens ?? 0} | cache_creation_input_tokens=${usage?.cache_creation_input_tokens ?? 0}`
                          );
                        } else if (evType === "content_block_delta") {
                          const delta = evObj.delta as Record<string, unknown> | undefined;
                          const deltaType = delta?.type as string | undefined;
                          if (typeof delta?.text === "string") _dbgContentLen += (delta.text as string).length;
                          if (typeof delta?.partial_json === "string") _dbgContentLen += (delta.partial_json as string).length;
                          if (typeof delta?.thinking === "string") _dbgContentLen += (delta.thinking as string).length;
                          _pluginLogger?.info(
                            `[frostclaw:debug] SSE[${sseSeq++}]: ${evType} | delta_type=${deltaType} | content_len=${_dbgContentLen}`
                          );
                        } else if (evType === "message_delta") {
                          const delta2 = evObj.delta as Record<string, unknown> | undefined;
                          const usage2 = evObj.usage as Record<string, unknown> | undefined;
                          _pluginLogger?.info(
                            `[frostclaw:debug] SSE[${sseSeq++}]: ${evType} | stop_reason=${delta2?.stop_reason} | output_tokens=${usage2?.output_tokens}`
                          );
                        } else if (evType === "message_stop") {
                          _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType}`);
                        } else if (evType === "content_block_start") {
                          const cb = evObj.content_block as Record<string, unknown> | undefined;
                          _pluginLogger?.info(
                            `[frostclaw:debug] SSE[${sseSeq++}]: ${evType} | block_type=${cb?.type}`
                          );
                        } else if (evType === "content_block_stop") {
                          _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType}`);
                        } else {
                          _pluginLogger?.info(`[frostclaw:debug] SSE[${sseSeq++}]: ${evType}`);
                        }
                      }
                      const isContentEvent =
                        evType === "text_start" ||
                        evType === "text_delta" ||
                        evType === "text_end" ||
                        evType === "toolcall_start" ||
                        evType === "toolcall_delta" ||
                        evType === "toolcall_end" ||
                        evType === "thinking_start" ||
                        evType === "thinking_delta" ||
                        evType === "thinking_end";
                      if (!hasContent && isContentEvent) {
                        // First real content — flush the buffer and go pass-through.
                        hasContent = true;
                        for (const buffered of buffer) {
                          (outerStream as unknown as { push: (e: unknown) => void }).push(buffered);
                        }
                        buffer.length = 0;
                      }
                      if (hasContent) {
                        (outerStream as unknown as { push: (e: unknown) => void }).push(event);
                      } else {
                        buffer.push(event);
                      }
                    }
                  } catch (err) {
                    // If no content has been emitted yet and the error is retryable,
                    // we can safely retry the whole request.
                    if (!hasContent && isRetryableError(err) && attempt < EMPTY_STOP_MAX_RETRIES) {
                      logWarn("wrapStreamFn.stream: retryable error from Snowflake (no content emitted), retrying", {
                        modelId,
                        attempt,
                        error: String(err),
                      });
                      buffer.length = 0;
                      await new Promise((r) => setTimeout(r, RETRY_BACKOFF_MS(attempt)));
                      const retryResult = inner(model, context, merged as typeof options);
                      if (
                        retryResult &&
                        typeof (retryResult as { result?: unknown })[Symbol.asyncIterator as unknown as string] === "function" &&
                        typeof (retryResult as { result?: unknown }).result === "function"
                      ) {
                        currentStream = retryResult as typeof currentStream;
                        continue;
                      }
                    }
                    logError("wrapStreamFn.stream ERROR", {
                      attempt,
                      error: String(err),
                      stack: err instanceof Error ? err.stack : undefined,
                    });
                    // Flush buffer so the outer stream doesn't hang, then end with error.
                    for (const buffered of buffer) {
                      (outerStream as unknown as { push: (e: unknown) => void }).push(buffered);
                    }
                    // Re-throw into outer stream by ending it; the outer caller
                    // will receive whatever partial message was built.
                    (outerStream as unknown as { end: (r?: unknown) => void }).end();
                    return;
                  }

                  // Stream finished — check final message.
                  let finalMsg: unknown;
                  try {
                    finalMsg = await currentStream.result();
                  } catch {
                    finalMsg = undefined;
                  }
                  const msg = finalMsg as { stopReason?: unknown; content?: unknown; usage?: unknown } | undefined;

                  // Result logging
                  if (isRequestDebugEnabled()) {
                    const ts = new Date().toISOString();
                    const contentBlocks = Array.isArray(msg?.content) ? (msg!.content as unknown[]).length : 0;
                    const isEmpty = !hasContent && isEmptyStop(msg, attempt);
                    const usage = msg?.usage as Record<string, unknown> | undefined;
                    _pluginLogger?.info(
                      `[frostclaw:debug] ${ts} ← Snowflake | model=${modelId} | attempt=${attempt + 1} | stop_reason=${msg?.stopReason} | content_blocks=${contentBlocks} | empty_stop=${isEmpty} | input_tokens=${usage?.inputTokens ?? usage?.input_tokens} | output_tokens=${usage?.outputTokens ?? usage?.output_tokens} | cache_read_input_tokens=${(usage?.cacheReadInputTokens ?? usage?.cache_read_input_tokens) ?? 0} | cache_creation_input_tokens=${(usage?.cacheCreationInputTokens ?? usage?.cache_creation_input_tokens) ?? 0}`
                    );
                  }

                  if (!hasContent && isEmptyStop(msg, attempt)) {
                    const isThinkingOnly =
                      Array.isArray(msg?.content) && (msg!.content as unknown[]).length > 0;
                    logWarn("wrapStreamFn.stream: empty/thinking-only stop from Snowflake, retrying", {
                      modelId,
                      attempt,
                      isThinkingOnly,
                    });
                    // Discard buffer — nothing was pushed to outerStream.
                    buffer.length = 0;
                    const retryResult = inner(model, context, merged as typeof options);
                    if (
                      retryResult &&
                      typeof (retryResult as { result?: unknown })[Symbol.asyncIterator as unknown as string] === "function" &&
                      typeof (retryResult as { result?: unknown }).result === "function"
                    ) {
                      currentStream = retryResult as typeof currentStream;
                      continue;
                    }
                    // Retry returned something unexpected — fall through and end.
                    (outerStream as unknown as { end: (r?: unknown) => void }).end(finalMsg);
                    return;
                  }

                  // Non-empty (or retry exhausted) — flush remaining buffer and end.
                  for (const buffered of buffer) {
                    (outerStream as unknown as { push: (e: unknown) => void }).push(buffered);
                  }
                  (outerStream as unknown as { end: (r?: unknown) => void }).end(finalMsg);
                  return;
                }
              })();
              return outerStream as typeof streamResult;
            }

            return streamResult;
          } catch (err) {
            logError("wrapStreamFn.inner ERROR", {
              error: String(err),
              stack: err instanceof Error ? err.stack : undefined,
            });
            throw err;
          }
        };
      },

      // -----------------------------------------------------------------------
      // Hook: Normalize the resolved model before the runner sees it.
      // We intentionally do NOT set requestTimeoutMs here. Setting it causes
      // openclaw's resolveLlmIdleTimeoutMs to fire the modelRequestTimeoutMs > 0
      // branch before the isLocalProviderBaseUrl short-circuit, wrapping every
      // stream call in streamWithIdleTimeout. If the run's AbortSignal is already
      // aborted (e.g. after a session-lock cascade), that wrapper immediately
      // aborts the stream controller and every fetch fails with AbortError at ~3ms.
      // Without requestTimeoutMs the idle-timeout path returns 0 for 127.0.0.1
      // URLs so no wrapping occurs. The run-level signal still provides
      // cancellation (180 s for cron jobs).
      // -----------------------------------------------------------------------
      normalizeResolvedModel: (_ctx: ProviderNormalizeResolvedModelContext) => {
        return { ..._ctx.model };
      },

      // -----------------------------------------------------------------------
      // Hook: Participate in config materialization.
      // -----------------------------------------------------------------------
      applyConfigDefaults: (_ctx: ProviderApplyConfigDefaultsContext) => {
        return null;
      },

      // -----------------------------------------------------------------------
      // Thinking profile: expose adaptive and all supported levels for Claude
      // models so the Control UI thinking dropdown shows them correctly.
      // -----------------------------------------------------------------------
      resolveThinkingProfile(ctx) {
        if (!ctx.modelId) return null;
        if (!isClaudeModel(ctx.modelId)) return null;
        // Strip the "snowflake-cortex/" prefix before calling the SDK function
        const bareId = ctx.modelId.replace(/^snowflake-cortex\//, "");
        return resolveClaudeThinkingProfile(bareId) ?? null;
      },

      // -----------------------------------------------------------------------
      // Replay policy for Claude on Snowflake Cortex.
      //
      // The policy flags below activate openclaw's runtime tool-call/tool-result
      // repair pipeline before the Anthropic Messages payload is serialized.
      // Without these flags, owned providers (like this one) skip the wrapper
      // that openclaw installs automatically for unowned Anthropic providers
      // via `buildUnownedProviderTransportReplayFallback`. Missing wrapper =
      // `tool_use` orphans (e.g. from session compaction or restart-mid-tool)
      // reach Snowflake's strict validator and 400 with "tool_use ids were
      // found without tool_result blocks immediately after".
      //
      // What each flag turns on (see selection-*.js wrapStreamFn registration):
      //   - sanitizeToolCallIds + toolCallIdMode: gates the wrapper that runs
      //     `sanitizeToolUseResultPairing` (drops dupes, moves displaced
      //     results adjacent, drops orphans).
      //   - preserveNativeAnthropicToolUseIds: keep `toolu_bdrk_*` IDs
      //     unchanged across the sanitizer (Bedrock-style native IDs).
      //   - repairToolUseResultPairing: actually do the pairing repair.
      //   - allowSyntheticToolResults: synthesize a placeholder tool_result
      //     when a tool_use has no matching result instead of dropping it.
      //   - validateAnthropicTurns: final pass that strips dangling tool_use
      //     blocks and merges consecutive user turns.
      //   - preserveSignatures: keep thinking-block signatures across replay
      //     so Bedrock-style provider-owned thinking continues to validate.
      // -----------------------------------------------------------------------
      buildReplayPolicy(ctx: ProviderReplayPolicyContext) {
        if (!ctx.modelId) return null;

        if (isClaudeModel(ctx.modelId)) {
          return {
            sanitizeToolCallIds: true,
            toolCallIdMode: "strict",
            preserveNativeAnthropicToolUseIds: true,
            repairToolUseResultPairing: true,
            allowSyntheticToolResults: true,
            validateAnthropicTurns: true,
            preserveSignatures: true,
          };
        }

        // Chat completions models: default policy is fine
        return null;
      },
    });
    } catch (err) {
      logError("register ERROR", {
        error: String(err),
        stack: err instanceof Error ? err.stack : undefined,
      });
      throw err;
    }
  },
});
