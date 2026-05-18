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
} from "openclaw/plugin-sdk/plugin-entry";
import { resolveClaudeThinkingProfile } from "openclaw/plugin-sdk/provider-model-shared";
import type {
  MemoryEmbeddingProviderAdapter,
  MemoryEmbeddingProviderCreateOptions,
} from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import { createProviderApiKeyAuthMethod } from "openclaw/plugin-sdk/provider-auth-api-key";
import type {
  ModelDefinitionConfig,
  ModelApi,
} from "openclaw/plugin-sdk/provider-model-types";
import {
  buildModelCatalog,
  findCatalogEntry,
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
//   log()      — debug/verbose, gated behind FROSTCLAW_DEBUG. Uses console.log
//                (stdout) so the Control UI shows these as info-level, not errors.
//                Stdout can buffer under heavy I/O; acceptable for verbose debug.
//   logWarn()  — operational signals always emitted (retry fired, error-path
//                resolved). Uses console.warn (stderr) — unbuffered, always
//                visible without enabling debug mode.
//   logError() — actual failures, always emitted. Uses console.error (stderr).
const DEBUG_ENABLED: boolean = ((): boolean => {
  const v = process.env.FROSTCLAW_DEBUG;
  if (!v) return false;
  const s = v.toLowerCase();
  return s !== "0" && s !== "false" && s !== "off" && s !== "";
})();

function log(event: string, data?: Record<string, unknown>): void {
  if (!DEBUG_ENABLED) return;
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  console.log(line);
}

function logWarn(event: string, data?: Record<string, unknown>): void {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  console.warn(line);
}

function logError(event: string, data?: Record<string, unknown>): void {
  const line = data ? `[snowflake-cortex] ${event} ${JSON.stringify(data)}` : `[snowflake-cortex] ${event}`;
  console.error(line);
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
  levelBudget,
  levelEffort,
  normalizeThinkingBudget,
  clampMaxTokens,
  stripEagerInputStreaming,
  isClaudeModel,
} from "./src/transforms.js";

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
  const baseUrl = getBaseURL();
  if (!apiKey || !baseUrl) {
    throw new Error(
      "[snowflake-cortex] Missing SNOWFLAKE_BASE_URL or SNOWFLAKE_CORTEX_API_KEY",
    );
  }

  const url = `${baseUrl}/api/v2/cortex/inference:embed`;
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

const snowflakeCortexEmbeddingAdapter: MemoryEmbeddingProviderAdapter = {
  id: "snowflake-cortex",
  defaultModel: DEFAULT_SNOWFLAKE_EMBED_MODEL,
  transport: "remote",
  // Low priority — only selected when explicitly configured
  autoSelectPriority: -1,

  async create(options: MemoryEmbeddingProviderCreateOptions) {
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
        embedQuery: (text: string) => snowflakeEmbed([text], model).then((v) => v[0]),
        embedBatch: (texts: string[]) => snowflakeEmbed(texts, model),
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
      // Silent on success to avoid per-load log spam (openclaw re-imports
      // this plugin frequently; debug output is gated to FROSTCLAW_DEBUG).
      log("plugin registered");
      api.registerMemoryEmbeddingProvider(snowflakeCortexEmbeddingAdapter);
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
        const baseUrl = getBaseURL();
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
              const fallbackBaseUrl = getBaseURL();
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
                  }
                  // Defensive: strip `eager_input_streaming` from tool schemas.
                  // The catalog sets supportsEagerToolInputStreaming: false to
                  // prevent pi-ai's Anthropic provider from adding it, but we
                  // also scrub it here so any future SDK regression or alternate
                  // code path can't re-introduce a Cortex-fatal field.
                  stripEagerInputStreaming(record);
                  normalizeThinkingBudget(record, thinkingLevel);
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
            // The openclaw runtime may invoke streamFn as streamSimple, which
            // returns a Promise. Wrap to observe rejection so we can capture
            // the stack trace for the "Cannot read properties of undefined"
            // crash that otherwise surfaces only as opaque text.
            if (
              streamResult &&
              typeof (streamResult as { then?: unknown }).then === "function"
            ) {
              // Retry wrapper: Snowflake returns HTTP 200 with empty content +
              // stop_reason="stop" when the account hits a concurrency/rate
              // limit instead of returning a proper 429. Retry transparently
              // up to EMPTY_STOP_MAX_RETRIES times before letting it through.
              const EMPTY_STOP_MAX_RETRIES = 2;
              return (async () => {
                let currentResult = streamResult as Promise<unknown>;
                for (let attempt = 0; attempt <= EMPTY_STOP_MAX_RETRIES; attempt++) {
                  let value: unknown;
                  try {
                    value = await currentResult;
                  } catch (err) {
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
                  // Detect empty stop — Snowflake overload signal.
                  // Two forms:
                  //   (a) content array is completely empty (content.length === 0)
                  //   (b) content array contains only thinking blocks with no text
                  //       output (thinking-only response: output ≈ 2 tokens,
                  //       assistantTexts: [] at OpenClaw level)
                  // Both are Snowflake's silent overload response instead of 429.
                  const isThinkingOnlyContent =
                    Array.isArray(valueObj?.content) &&
                    (valueObj!.content as unknown[]).length > 0 &&
                    (valueObj!.content as Array<{ type?: string }>).every(
                      (blk) => blk?.type === "thinking"
                    );
                  const isEmptyOrThinkingOnlyStop =
                    attempt < EMPTY_STOP_MAX_RETRIES &&
                    valueObj &&
                    typeof valueObj === "object" &&
                    valueObj.stopReason === "stop" &&
                    (
                      (Array.isArray(valueObj.content) && valueObj.content.length === 0) ||
                      isThinkingOnlyContent
                    );
                  if (isEmptyOrThinkingOnlyStop) {
                    logWarn("wrapStreamFn.promise: empty/thinking-only stop from Snowflake, retrying", {
                      modelId,
                      attempt,
                      isThinkingOnly: isThinkingOnlyContent,
                      contentLength: Array.isArray(valueObj?.content) ? (valueObj!.content as unknown[]).length : 0,
                    });
                    const retryResult = inner(model, context, merged as typeof options);
                    if (retryResult && typeof (retryResult as { then?: unknown }).then === "function") {
                      currentResult = retryResult as Promise<unknown>;
                      continue;
                    }
                    // Retry returned a stream — can't handle here, pass through
                    return retryResult;
                  }
                  return value;
                }
              })() as typeof streamResult;
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
