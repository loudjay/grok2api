import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Env } from "../env";
import { requireApiAuth } from "../auth";
import { getSettings, normalizeCfCookie } from "../settings";
import { isValidModel, MODEL_CONFIG } from "../grok/models";
import { extractContent, buildConversationPayload, sendConversationRequest } from "../grok/conversation";
import { uploadImage } from "../grok/upload";
import { getDynamicHeaders } from "../grok/headers";
import { createMediaPost, createPost } from "../grok/create";
import { createOpenAiStreamFromGrokNdjson, parseOpenAiFromGrokNdjson } from "../grok/processor";
import { addRequestLog } from "../repo/logs";
import { applyCooldown, recordTokenFailure, selectBestToken } from "../repo/tokens";
import type { ApiAuthInfo } from "../auth";
import { getApiKeyLimits } from "../repo/apiKeys";
import { localDayString, tryConsumeDailyUsage, tryConsumeDailyUsageMulti } from "../repo/apiKeyUsage";
import { nextLocalMidnightExpirationSeconds } from "../kv/cleanup";
import { nowMs } from "../utils/time";
import { arrayBufferToBase64 } from "../utils/base64";
import { upsertCacheRow } from "../repo/cache";

function openAiError(message: string, code: string): Record<string, unknown> {
  return { error: { message, type: "invalid_request_error", code } };
}

function getClientIp(req: Request): string {
  return (
    req.headers.get("CF-Connecting-IP") ||
    req.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() ||
    "0.0.0.0"
  );
}

async function mapLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = [];
  const queue = items.slice();
  const workers = Array.from({ length: Math.max(1, limit) }, async () => {
    while (queue.length) {
      const item = queue.shift() as T;
      results.push(await fn(item));
    }
  });
  await Promise.all(workers);
  return results;
}

export const openAiRoutes = new Hono<{ Bindings: Env; Variables: { apiAuth: ApiAuthInfo } }>();

openAiRoutes.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Authorization", "Content-Type"],
    allowMethods: ["GET", "POST", "OPTIONS"],
    maxAge: 86400,
  }),
);

openAiRoutes.use("/*", requireApiAuth);

function parseIntSafe(v: string | undefined, fallback: number): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.floor(n);
}

function quotaError(bucket: string): Record<string, unknown> {
  return openAiError(`Daily quota exceeded: ${bucket}`, "daily_quota_exceeded");
}

async function enforceQuota(args: {
  env: Env;
  apiAuth: ApiAuthInfo;
  model: string;
  kind: "chat" | "image" | "video";
  imageCount?: number;
}): Promise<{ ok: true } | { ok: false; resp: Response }> {
  const key = args.apiAuth.key;
  if (!key) return { ok: true };
  if (args.apiAuth.is_admin) return { ok: true };

  const limits = await getApiKeyLimits(args.env.DB, key);
  if (!limits) return { ok: true };

  const tz = parseIntSafe(args.env.CACHE_RESET_TZ_OFFSET_MINUTES, 480);
  const day = localDayString(nowMs(), tz);
  const atMs = nowMs();
  const jsonHeaders = { "content-type": "application/json; charset=utf-8" };

  if (args.model === "grok-4-heavy") {
    const ok = await tryConsumeDailyUsageMulti({
      db: args.env.DB,
      key,
      day,
      atMs,
      updates: [
        { field: "heavy_used", inc: 1, limit: limits.heavy_limit },
        { field: "chat_used", inc: 1, limit: limits.chat_limit },
      ],
    });
    if (!ok) return { ok: false, resp: new Response(JSON.stringify(quotaError("heavy/chat")), { status: 429, headers: jsonHeaders }) };
    return { ok: true };
  }

  if (args.kind === "video") {
    const ok = await tryConsumeDailyUsage({
      db: args.env.DB,
      key,
      day,
      atMs,
      field: "video_used",
      inc: 1,
      limit: limits.video_limit,
    });
    if (!ok) return { ok: false, resp: new Response(JSON.stringify(quotaError("video")), { status: 429, headers: jsonHeaders }) };
    return { ok: true };
  }

  if (args.kind === "image") {
    const inc = Math.max(1, Math.floor(Number(args.imageCount ?? 1) || 1));
    const ok = await tryConsumeDailyUsage({
      db: args.env.DB,
      key,
      day,
      atMs,
      field: "image_used",
      inc,
      limit: limits.image_limit,
    });
    if (!ok) return { ok: false, resp: new Response(JSON.stringify(quotaError("image")), { status: 429, headers: jsonHeaders }) };
    return { ok: true };
  }

  // chat
  const ok = await tryConsumeDailyUsage({
    db: args.env.DB,
    key,
    day,
    atMs,
    field: "chat_used",
    inc: 1,
    limit: limits.chat_limit,
  });
  if (!ok) return { ok: false, resp: new Response(JSON.stringify(quotaError("chat")), { status: 429, headers: jsonHeaders }) };
  return { ok: true };
}

function base64UrlEncodeString(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function encodeAssetPath(raw: string): string {
  try {
    const u = new URL(raw);
    return `u_${base64UrlEncodeString(u.toString())}`;
  } catch {
    const p = raw.startsWith("/") ? raw : `/${raw}`;
    return `p_${base64UrlEncodeString(p)}`;
  }
}

function toProxyUrl(baseUrl: string, path: string): string {
  return `${baseUrl.replace(/\/$/, "")}/images/${path}`;
}

type ImageResponseFormat = "url" | "base64" | "b64_json";

function resolveResponseFormat(raw: unknown, defaultMode: string): ImageResponseFormat | null {
  const fallback = String(defaultMode || "url").trim().toLowerCase();
  const candidate =
    typeof raw === "string" && raw.trim() ? raw.trim().toLowerCase() : fallback;
  if (candidate === "url" || candidate === "base64" || candidate === "b64_json") {
    return candidate;
  }
  return null;
}

function responseFieldName(format: ImageResponseFormat): ImageResponseFormat {
  return format;
}

function toBool(input: unknown): boolean {
  if (typeof input === "boolean") return input;
  if (typeof input === "number") return input === 1;
  if (typeof input !== "string") return false;
  const normalized = input.trim().toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes";
}

function normalizeGeneratedImageUrls(input: unknown): string[] {
  if (!Array.isArray(input)) return [];
  return input
    .filter((u): u is string => typeof u === "string")
    .map((u) => u.trim())
    .filter((u) => Boolean(u && u !== "/"));
}

function pickImageResults(images: string[], n: number): string[] {
  if (images.length >= n) {
    const pool = images.slice();
    const picked: string[] = [];
    while (picked.length < n && pool.length) {
      const idx = Math.floor(Math.random() * pool.length);
      const [item] = pool.splice(idx, 1);
      if (item) picked.push(item);
    }
    return picked;
  }
  const picked = images.slice();
  while (picked.length < n) picked.push("error");
  return picked;
}

function normalizeImageMime(mime: string): string {
  const m = (mime || "").trim().toLowerCase();
  if (m === "image/jpg") return "image/jpeg";
  return m;
}

function mimeFromFilename(filename: string): string | null {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".webp")) return "image/webp";
  return null;
}

async function fetchImageAsBase64(args: {
  rawUrl: string;
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
}): Promise<string> {
  let url: URL;
  try {
    url = new URL(args.rawUrl);
  } catch {
    const p = args.rawUrl.startsWith("/") ? args.rawUrl : `/${args.rawUrl}`;
    url = new URL(`https://assets.grok.com${p}`);
  }

  const headers = getDynamicHeaders(args.settings, url.pathname || "/");
  headers.Cookie = args.cookie;
  delete headers["Content-Type"];
  headers.Accept = "image/avif,image/webp,image/*,*/*;q=0.8";
  headers["Sec-Fetch-Dest"] = "image";
  headers["Sec-Fetch-Mode"] = "no-cors";
  headers["Sec-Fetch-Site"] = "same-site";
  headers.Referer = "https://grok.com/";

  const resp = await fetch(url.toString(), { method: "GET", headers, redirect: "follow" });
  if (!resp.ok) {
    const txt = await resp.text().catch(() => "");
    throw new Error(`Image download failed: ${resp.status} ${txt.slice(0, 200)}`);
  }
  return arrayBufferToBase64(await resp.arrayBuffer());
}

async function convertRawUrlByFormat(
  rawUrl: string,
  responseFormat: ImageResponseFormat,
  args: {
    baseUrl: string;
    cookie: string;
    settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  },
): Promise<string> {
  if (responseFormat === "url") {
    return toProxyUrl(args.baseUrl, encodeAssetPath(rawUrl));
  }
  return fetchImageAsBase64({ rawUrl, cookie: args.cookie, settings: args.settings });
}

async function collectImageUrls(resp: Response): Promise<string[]> {
  const text = await resp.text();
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const allUrls: string[] = [];
  for (const line of lines) {
    let data: any;
    try {
      data = JSON.parse(line);
    } catch {
      continue;
    }
    const err = data?.error;
    if (err?.message) throw new Error(String(err.message));
    const grok = data?.result?.response;
    const urls = normalizeGeneratedImageUrls(grok?.modelResponse?.generatedImageUrls);
    if (urls.length) allUrls.push(...urls);
  }
  return allUrls;
}

function buildImageSse(event: string, data: Record<string, unknown>): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function createImageEventStream(args: {
  upstream: Response;
  responseFormat: ImageResponseFormat;
  baseUrl: string;
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  n: number;
  onFinish?: (result: { status: number; duration: number }) => Promise<void> | void;
}): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const responseField = responseFieldName(args.responseFormat);
  const targetIndex = args.n === 1 ? Math.floor(Math.random() * 2) : null;

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const startedAt = Date.now();
      const body = args.upstream.body;
      if (!body) {
        if (args.onFinish) {
          await args.onFinish({ status: 500, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.close();
        return;
      }

      const reader = body.getReader();
      const finalImages: string[] = [];
      let buffer = "";
      let failed = false;
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value) continue;
          buffer += decoder.decode(value, { stream: true });
          let idx = buffer.indexOf("\n");
          while (idx >= 0) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line) {
              idx = buffer.indexOf("\n");
              continue;
            }

            let data: any;
            try {
              data = JSON.parse(line);
            } catch {
              idx = buffer.indexOf("\n");
              continue;
            }

            const err = data?.error;
            if (err?.message) throw new Error(String(err.message));

            const resp = data?.result?.response ?? {};
            const progressInfo = resp.streamingImageGenerationResponse;
            if (progressInfo) {
              const imageIndex = Number(progressInfo.imageIndex ?? 0);
              const progress = Number(progressInfo.progress ?? 0);
              if (args.n === 1 && imageIndex !== targetIndex) {
                idx = buffer.indexOf("\n");
                continue;
              }
              const outIndex = args.n === 1 ? 0 : imageIndex;
              controller.enqueue(
                encoder.encode(
                  buildImageSse("image_generation.partial_image", {
                    type: "image_generation.partial_image",
                    [responseField]: "",
                    index: outIndex,
                    progress,
                  }),
                ),
              );
            }

            const rawUrls = normalizeGeneratedImageUrls(resp?.modelResponse?.generatedImageUrls);
            if (rawUrls.length) {
              for (const rawUrl of rawUrls) {
                const converted = await convertRawUrlByFormat(rawUrl, args.responseFormat, {
                  baseUrl: args.baseUrl,
                  cookie: args.cookie,
                  settings: args.settings,
                });
                finalImages.push(converted);
              }
            }
            idx = buffer.indexOf("\n");
          }
        }

        for (let i = 0; i < finalImages.length; i++) {
          if (args.n === 1 && i !== targetIndex) continue;
          const outIndex = args.n === 1 ? 0 : i;
          controller.enqueue(
            encoder.encode(
              buildImageSse("image_generation.completed", {
                type: "image_generation.completed",
                [responseField]: finalImages[i] ?? "",
                index: outIndex,
                usage: {
                  total_tokens: 50,
                  input_tokens: 25,
                  output_tokens: 25,
                  input_tokens_details: { text_tokens: 5, image_tokens: 20 },
                },
              }),
            ),
          );
        }
        if (args.onFinish) {
          await args.onFinish({ status: 200, duration: (Date.now() - startedAt) / 1000 });
        }
      } catch (e) {
        failed = true;
        console.error("Image stream processing failed:", e);
        if (args.onFinish) {
          await args.onFinish({ status: 500, duration: (Date.now() - startedAt) / 1000 });
        }
        controller.error(e);
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
        if (!failed) controller.close();
      }
    },
  });
}

function imageResponseData(field: ImageResponseFormat, values: string[]) {
  return values.map((v) => ({ [field]: v }));
}

function getTokenSuffix(token: string): string {
  return token.length >= 6 ? token.slice(-6) : token;
}

function parseImageCount(input: unknown): number {
  const raw = Number(input ?? 1);
  if (!Number.isFinite(raw)) return 1;
  return Math.max(1, Math.min(10, Math.floor(raw)));
}

function parseImagePrompt(input: unknown): string {
  return String(input ?? "").trim();
}

function parseImageModel(input: unknown): string {
  return String(input ?? "grok-imagine-1.0").trim() || "grok-imagine-1.0";
}

function parseImageStream(input: unknown): boolean {
  return toBool(input);
}

function parseAllowedImageMime(file: File): string | null {
  const byMime = normalizeImageMime(String(file.type || ""));
  if (byMime === "image/png" || byMime === "image/jpeg" || byMime === "image/webp") return byMime;
  const byName = mimeFromFilename(String(file.name || ""));
  if (byName) return byName;
  return null;
}

function buildCookie(token: string, cf: string): string {
  return cf ? `sso-rw=${token};sso=${token};${cf}` : `sso-rw=${token};sso=${token}`;
}

async function runImageCall(args: {
  requestModel: string;
  prompt: string;
  fileIds: string[];
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  responseFormat: ImageResponseFormat;
  baseUrl: string;
}): Promise<string[]> {
  const { payload, referer } = buildConversationPayload({
    requestModel: args.requestModel,
    content: args.prompt,
    imgIds: args.fileIds,
    imgUris: [],
    settings: args.settings,
  });
  const upstream = await sendConversationRequest({
    payload,
    cookie: args.cookie,
    settings: args.settings,
    ...(referer ? { referer } : {}),
  });
  if (!upstream.ok) {
    const txt = await upstream.text().catch(() => "");
    throw new Error(`Upstream ${upstream.status}: ${txt.slice(0, 200)}`);
  }
  const rawUrls = await collectImageUrls(upstream);
  const converted = await Promise.all(
    rawUrls.map((rawUrl) =>
      convertRawUrlByFormat(rawUrl, args.responseFormat, {
        baseUrl: args.baseUrl,
        cookie: args.cookie,
        settings: args.settings,
      }),
    ),
  );
  return converted.filter(Boolean);
}

async function runImageStreamCall(args: {
  requestModel: string;
  prompt: string;
  fileIds: string[];
  cookie: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
}): Promise<Response> {
  const { payload, referer } = buildConversationPayload({
    requestModel: args.requestModel,
    content: args.prompt,
    imgIds: args.fileIds,
    imgUris: [],
    settings: args.settings,
  });
  return sendConversationRequest({
    payload,
    cookie: args.cookie,
    settings: args.settings,
    ...(referer ? { referer } : {}),
  });
}

function streamHeaders(): Record<string, string> {
  return {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
  };
}

function isValidImageModel(model: string): boolean {
  if (!isValidModel(model)) return false;
  const cfg = MODEL_CONFIG[model];
  return Boolean(cfg?.is_image_model);
}

function invalidResponseFormatMessage(): string {
  return "response_format must be one of [\"b64_json\", \"base64\", \"url\"]";
}

function invalidStreamNMessage(): string {
  return "Streaming is only supported when n=1 or n=2";
}

function imageUsagePayload(values: string[]) {
  return {
    total_tokens: 0 * values.filter((v) => v !== "error").length,
    input_tokens: 0,
    output_tokens: 0 * values.filter((v) => v !== "error").length,
    input_tokens_details: { text_tokens: 0, image_tokens: 0 },
  };
}

function createdTs(): number {
  return Math.floor(Date.now() / 1000);
}

function buildImageJsonPayload(field: ImageResponseFormat, values: string[]) {
  return {
    created: createdTs(),
    data: imageResponseData(field, values),
    usage: imageUsagePayload(values),
  };
}

async function recordImageLog(args: {
  env: Env;
  ip: string;
  model: string;
  start: number;
  keyName: string;
  status: number;
  tokenSuffix?: string;
  error: string;
}) {
  const duration = (Date.now() - args.start) / 1000;
  await addRequestLog(args.env.DB, {
    ip: args.ip,
    model: args.model,
    duration: Number(duration.toFixed(2)),
    status: args.status,
    key_name: args.keyName,
    token_suffix: args.tokenSuffix ?? "",
    error: args.error,
  });
}

function listImageFiles(form: FormData): File[] {
  return [...form.getAll("image"), ...form.getAll("image[]")].filter(
    (item): item is File => item instanceof File,
  );
}

function nonEmptyPromptOrError(prompt: string) {
  if (prompt) return null;
  return { message: "Missing 'prompt'", code: "missing_prompt" };
}

function invalidModelOrError(model: string) {
  if (!isValidModel(model)) return { message: `Model '${model}' not supported`, code: "model_not_supported" };
  if (!isValidImageModel(model)) return { message: `Model '${model}' is not an image model`, code: "invalid_model" };
  return null;
}

function baseUrlFromSettings(settingsBundle: Awaited<ReturnType<typeof getSettings>>, origin: string): string {
  return (settingsBundle.global.base_url ?? "").trim() || origin;
}

function imageCallPrompt(kind: "generation" | "edit", prompt: string): string {
  return kind === "edit" ? `Image Edit: ${prompt}` : `Image Generation: ${prompt}`;
}

function imageFormatDefault(settingsBundle: Awaited<ReturnType<typeof getSettings>>): string {
  return String(settingsBundle.global.image_mode ?? "url");
}

function parseResponseFormatOrError(raw: unknown, defaultMode: string) {
  const resolved = resolveResponseFormat(raw, defaultMode);
  if (!resolved) {
    return { error: { message: invalidResponseFormatMessage(), code: "invalid_response_format" } };
  }
  return { value: resolved };
}

openAiRoutes.get("/models", async (c) => {
  const ts = Math.floor(Date.now() / 1000);
  const data = Object.entries(MODEL_CONFIG).map(([id, cfg]) => ({
    id,
    object: "model",
    created: ts,
    owned_by: "x-ai",
    display_name: cfg.display_name,
    description: cfg.description,
    raw_model_path: cfg.raw_model_path,
    default_temperature: cfg.default_temperature,
    default_max_output_tokens: cfg.default_max_output_tokens,
    supported_max_output_tokens: cfg.supported_max_output_tokens,
    default_top_p: cfg.default_top_p,
  }));
  return c.json({ object: "list", data });
});

openAiRoutes.get("/models/:modelId", async (c) => {
  const modelId = c.req.param("modelId");
  if (!isValidModel(modelId)) return c.json(openAiError(`Model '${modelId}' not found`, "model_not_found"), 404);
  const cfg = MODEL_CONFIG[modelId]!;
  const ts = Math.floor(Date.now() / 1000);
  return c.json({
    id: modelId,
    object: "model",
    created: ts,
    owned_by: "x-ai",
    display_name: cfg.display_name,
    description: cfg.description,
    raw_model_path: cfg.raw_model_path,
    default_temperature: cfg.default_temperature,
    default_max_output_tokens: cfg.default_max_output_tokens,
    supported_max_output_tokens: cfg.supported_max_output_tokens,
    default_top_p: cfg.default_top_p,
  });
});

openAiRoutes.post("/chat/completions", async (c) => {
  const start = Date.now();
  const ip = getClientIp(c.req.raw);
  const keyName = c.get("apiAuth").name ?? "Unknown";

  const origin = new URL(c.req.url).origin;

  let requestedModel = "";
  try {
    const body = (await c.req.json()) as {
      model?: string;
      messages?: any[];
      stream?: boolean;
      video_config?: {
        aspect_ratio?: string;
        video_length?: number;
        resolution?: string;
        preset?: string;
      };
    };

    requestedModel = String(body.model ?? "");
    if (!requestedModel) return c.json(openAiError("Missing 'model'", "missing_model"), 400);
    if (!Array.isArray(body.messages)) return c.json(openAiError("Missing 'messages'", "missing_messages"), 400);
    if (!isValidModel(requestedModel))
      return c.json(openAiError(`Model '${requestedModel}' not supported`, "model_not_supported"), 400);

    const settingsBundle = await getSettings(c.env);
    const cfg = MODEL_CONFIG[requestedModel]!;

    const retryCodes = Array.isArray(settingsBundle.grok.retry_status_codes)
      ? settingsBundle.grok.retry_status_codes
      : [401, 429];

    const stream = Boolean(body.stream);
    const maxRetry = 3;
    let lastErr: string | null = null;

    // === Quota check (best-effort) ===
    // - heavy: consumes both heavy + chat
    // - image model: counts as 2 images per request (grok upstream emits up to 2)
    // - video model: 1 video per request
    // - others: 1 chat per request
    const quotaKind = cfg.is_video_model ? "video" : cfg.is_image_model ? "image" : "chat";
    const quota = await enforceQuota({
      env: c.env,
      apiAuth: c.get("apiAuth"),
      model: requestedModel,
      kind: quotaKind as any,
      ...(cfg.is_image_model ? { imageCount: 2 } : {}),
    });
    if (!quota.ok) return quota.resp;

    for (let attempt = 0; attempt < maxRetry; attempt++) {
      const chosen = await selectBestToken(c.env.DB, requestedModel);
      if (!chosen) return c.json(openAiError("No available token", "NO_AVAILABLE_TOKEN"), 503);

      const jwt = chosen.token;
      const cf = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");
      const cookie = cf ? `sso-rw=${jwt};sso=${jwt};${cf}` : `sso-rw=${jwt};sso=${jwt}`;

      const { content, images } = extractContent(body.messages as any);
      const isVideoModel = Boolean(cfg.is_video_model);
      const imgInputs = isVideoModel && images.length > 1 ? images.slice(0, 1) : images;

      try {
        const uploads = await mapLimit(imgInputs, 5, (u) => uploadImage(u, cookie, settingsBundle.grok));
        const imgIds = uploads.map((u) => u.fileId).filter(Boolean);
        const imgUris = uploads.map((u) => u.fileUri).filter(Boolean);

        let postId: string | undefined;
        if (isVideoModel) {
          if (imgUris.length) {
            const post = await createPost(imgUris[0]!, cookie, settingsBundle.grok);
            postId = post.postId || undefined;
          } else {
            const post = await createMediaPost(
              { mediaType: "MEDIA_POST_TYPE_VIDEO", prompt: content },
              cookie,
              settingsBundle.grok,
            );
            postId = post.postId || undefined;
          }
        }

        const { payload, referer } = buildConversationPayload({
          requestModel: requestedModel,
          content,
          imgIds,
          imgUris,
          ...(postId ? { postId } : {}),
          ...(isVideoModel && body.video_config ? { videoConfig: body.video_config } : {}),
          settings: settingsBundle.grok,
        });

        const upstream = await sendConversationRequest({
          payload,
          cookie,
          settings: settingsBundle.grok,
          ...(referer ? { referer } : {}),
        });

        if (!upstream.ok) {
          const txt = await upstream.text().catch(() => "");
          lastErr = `Upstream ${upstream.status}: ${txt.slice(0, 200)}`;
          await recordTokenFailure(c.env.DB, jwt, upstream.status, txt.slice(0, 200));
          await applyCooldown(c.env.DB, jwt, upstream.status);
          if (retryCodes.includes(upstream.status) && attempt < maxRetry - 1) continue;
          break;
        }

        if (stream) {
          const sse = createOpenAiStreamFromGrokNdjson(upstream, {
            cookie,
            settings: settingsBundle.grok,
            global: settingsBundle.global,
            origin,
            onFinish: async ({ status, duration }) => {
              await addRequestLog(c.env.DB, {
                ip,
                model: requestedModel,
                duration: Number(duration.toFixed(2)),
                status,
                key_name: keyName,
                token_suffix: jwt.slice(-6),
                error: status === 200 ? "" : "stream_error",
              });
            },
          });

          return new Response(sse, {
            status: 200,
            headers: {
              "Content-Type": "text/event-stream; charset=utf-8",
              "Cache-Control": "no-cache",
              Connection: "keep-alive",
              "X-Accel-Buffering": "no",
              "Access-Control-Allow-Origin": "*",
            },
          });
        }

        const json = await parseOpenAiFromGrokNdjson(upstream, {
          cookie,
          settings: settingsBundle.grok,
          global: settingsBundle.global,
          origin,
          requestedModel,
        });

        const duration = (Date.now() - start) / 1000;
        await addRequestLog(c.env.DB, {
          ip,
          model: requestedModel,
          duration: Number(duration.toFixed(2)),
          status: 200,
          key_name: keyName,
          token_suffix: jwt.slice(-6),
          error: "",
        });

        return c.json(json);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        lastErr = msg;
        await recordTokenFailure(c.env.DB, jwt, 500, msg);
        await applyCooldown(c.env.DB, jwt, 500);
        if (attempt < maxRetry - 1) continue;
      }
    }

    const duration = (Date.now() - start) / 1000;
    await addRequestLog(c.env.DB, {
      ip,
      model: requestedModel,
      duration: Number(duration.toFixed(2)),
      status: 500,
      key_name: keyName,
      token_suffix: "",
      error: lastErr ?? "unknown_error",
    });

    return c.json(openAiError(lastErr ?? "Upstream error", "upstream_error"), 500);
  } catch (e) {
    const duration = (Date.now() - start) / 1000;
    await addRequestLog(c.env.DB, {
      ip,
      model: requestedModel || "unknown",
      duration: Number(duration.toFixed(2)),
      status: 500,
      key_name: keyName,
      token_suffix: "",
      error: e instanceof Error ? e.message : String(e),
    });
    return c.json(openAiError("Internal error", "internal_error"), 500);
  }
});

openAiRoutes.post("/images/generations", async (c) => {
  const start = Date.now();
  const ip = getClientIp(c.req.raw);
  const keyName = c.get("apiAuth").name ?? "Unknown";
  const origin = new URL(c.req.url).origin;

  let requestedModel = "grok-imagine-1.0";
  try {
    const body = (await c.req.json()) as {
      prompt?: unknown;
      model?: unknown;
      n?: unknown;
      stream?: unknown;
      response_format?: unknown;
    };
    const prompt = parseImagePrompt(body.prompt);
    const promptErr = nonEmptyPromptOrError(prompt);
    if (promptErr) return c.json(openAiError(promptErr.message, promptErr.code), 400);

    requestedModel = parseImageModel(body.model);
    const modelErr = invalidModelOrError(requestedModel);
    if (modelErr) return c.json(openAiError(modelErr.message, modelErr.code), 400);

    const n = parseImageCount(body.n);
    const stream = parseImageStream(body.stream);
    if (stream && ![1, 2].includes(n)) {
      return c.json(openAiError(invalidStreamNMessage(), "invalid_stream_n"), 400);
    }

    const settingsBundle = await getSettings(c.env);
    const parsedResponseFormat = parseResponseFormatOrError(
      body.response_format,
      imageFormatDefault(settingsBundle),
    );
    if ("error" in parsedResponseFormat) {
      return c.json(
        openAiError(parsedResponseFormat.error.message, parsedResponseFormat.error.code),
        400,
      );
    }
    const responseFormat = parsedResponseFormat.value;
    const responseField = responseFieldName(responseFormat);
    const baseUrl = baseUrlFromSettings(settingsBundle, origin);
    const cf = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");

    const quota = await enforceQuota({
      env: c.env,
      apiAuth: c.get("apiAuth"),
      model: requestedModel,
      kind: "image",
      imageCount: n,
    });
    if (!quota.ok) return quota.resp;

    if (stream) {
      const chosen = await selectBestToken(c.env.DB, requestedModel);
      if (!chosen) return c.json(openAiError("No available token", "NO_AVAILABLE_TOKEN"), 503);
      const cookie = buildCookie(chosen.token, cf);

      const upstream = await runImageStreamCall({
        requestModel: requestedModel,
        prompt: imageCallPrompt("generation", prompt),
        fileIds: [],
        cookie,
        settings: settingsBundle.grok,
      });
      if (!upstream.ok) {
        const txt = await upstream.text().catch(() => "");
        await recordTokenFailure(c.env.DB, chosen.token, upstream.status, txt.slice(0, 200));
        await applyCooldown(c.env.DB, chosen.token, upstream.status);
        await recordImageLog({
          env: c.env,
          ip,
          model: requestedModel,
          start,
          keyName,
          status: upstream.status,
          tokenSuffix: getTokenSuffix(chosen.token),
          error: txt.slice(0, 200),
        });
        return c.json(openAiError(`Upstream ${upstream.status}`, "upstream_error"), 500);
      }

      const streamBody = createImageEventStream({
        upstream,
        responseFormat,
        baseUrl,
        cookie,
        settings: settingsBundle.grok,
        n,
        onFinish: async ({ status, duration }) => {
          await addRequestLog(c.env.DB, {
            ip,
            model: requestedModel,
            duration: Number(duration.toFixed(2)),
            status,
            key_name: keyName,
            token_suffix: getTokenSuffix(chosen.token),
            error: status === 200 ? "" : "stream_error",
          });
        },
      });
      return new Response(streamBody, { status: 200, headers: streamHeaders() });
    }

    const calls = Math.ceil(n / 2);
    const urlsNested = await mapLimit(Array.from({ length: calls }), 3, async () => {
      const chosen = await selectBestToken(c.env.DB, requestedModel);
      if (!chosen) throw new Error("No available token");
      const cookie = buildCookie(chosen.token, cf);
      try {
        return await runImageCall({
          requestModel: requestedModel,
          prompt: imageCallPrompt("generation", prompt),
          fileIds: [],
          cookie,
          settings: settingsBundle.grok,
          responseFormat,
          baseUrl,
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        await recordTokenFailure(c.env.DB, chosen.token, 500, msg.slice(0, 200));
        await applyCooldown(c.env.DB, chosen.token, 500);
        throw e;
      }
    });
    const urls = urlsNested.flat().filter(Boolean);
    const selected = pickImageResults(urls, n);

    await recordImageLog({
      env: c.env,
      ip,
      model: requestedModel,
      start,
      keyName,
      status: 200,
      error: "",
    });

    return c.json(buildImageJsonPayload(responseField, selected));
  } catch (e) {
    await recordImageLog({
      env: c.env,
      ip,
      model: requestedModel || "image",
      start,
      keyName,
      status: 500,
      error: e instanceof Error ? e.message : String(e),
    });
    return c.json(openAiError(e instanceof Error ? e.message : "Internal error", "internal_error"), 500);
  }
});

openAiRoutes.post("/images/edits", async (c) => {
  const start = Date.now();
  const ip = getClientIp(c.req.raw);
  const keyName = c.get("apiAuth").name ?? "Unknown";
  const origin = new URL(c.req.url).origin;
  const maxImageBytes = 50 * 1024 * 1024;

  let requestedModel = "grok-imagine-1.0";
  try {
    const form = await c.req.formData();
    const prompt = parseImagePrompt(form.get("prompt"));
    const promptErr = nonEmptyPromptOrError(prompt);
    if (promptErr) return c.json(openAiError(promptErr.message, promptErr.code), 400);

    requestedModel = parseImageModel(form.get("model"));
    const modelErr = invalidModelOrError(requestedModel);
    if (modelErr) return c.json(openAiError(modelErr.message, modelErr.code), 400);

    const n = parseImageCount(form.get("n"));
    const stream = parseImageStream(form.get("stream"));
    if (stream && ![1, 2].includes(n)) {
      return c.json(openAiError(invalidStreamNMessage(), "invalid_stream_n"), 400);
    }

    const files = listImageFiles(form);
    if (!files.length) return c.json(openAiError("Image is required", "missing_image"), 400);
    if (files.length > 16) {
      return c.json(openAiError("Too many images. Maximum is 16.", "invalid_image_count"), 400);
    }

    const settingsBundle = await getSettings(c.env);
    const parsedResponseFormat = parseResponseFormatOrError(
      form.get("response_format"),
      imageFormatDefault(settingsBundle),
    );
    if ("error" in parsedResponseFormat) {
      return c.json(
        openAiError(parsedResponseFormat.error.message, parsedResponseFormat.error.code),
        400,
      );
    }
    const responseFormat = parsedResponseFormat.value;
    const responseField = responseFieldName(responseFormat);
    const baseUrl = baseUrlFromSettings(settingsBundle, origin);

    const quota = await enforceQuota({
      env: c.env,
      apiAuth: c.get("apiAuth"),
      model: requestedModel,
      kind: "image",
      imageCount: n,
    });
    if (!quota.ok) return quota.resp;

    const chosen = await selectBestToken(c.env.DB, requestedModel);
    if (!chosen) return c.json(openAiError("No available token", "NO_AVAILABLE_TOKEN"), 503);
    const cf = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");
    const cookie = buildCookie(chosen.token, cf);

    const fileIds: string[] = [];
    for (const file of files) {
      const bytes = await file.arrayBuffer();
      if (bytes.byteLength <= 0) {
        return c.json(openAiError("File content is empty", "empty_file"), 400);
      }
      if (bytes.byteLength > maxImageBytes) {
        return c.json(openAiError("Image file too large. Maximum is 50MB.", "file_too_large"), 400);
      }

      const mime = parseAllowedImageMime(file);
      if (!mime) {
        return c.json(
          openAiError("Unsupported image type. Supported: png, jpg, webp.", "invalid_image_type"),
          400,
        );
      }

      const dataUrl = `data:${mime};base64,${arrayBufferToBase64(bytes)}`;
      const uploaded = await uploadImage(dataUrl, cookie, settingsBundle.grok);
      if (uploaded.fileId) fileIds.push(uploaded.fileId);
    }

    if (stream) {
      const upstream = await runImageStreamCall({
        requestModel: requestedModel,
        prompt: imageCallPrompt("edit", prompt),
        fileIds,
        cookie,
        settings: settingsBundle.grok,
      });
      if (!upstream.ok) {
        const txt = await upstream.text().catch(() => "");
        await recordTokenFailure(c.env.DB, chosen.token, upstream.status, txt.slice(0, 200));
        await applyCooldown(c.env.DB, chosen.token, upstream.status);
        await recordImageLog({
          env: c.env,
          ip,
          model: requestedModel,
          start,
          keyName,
          status: upstream.status,
          tokenSuffix: getTokenSuffix(chosen.token),
          error: txt.slice(0, 200),
        });
        return c.json(openAiError(`Upstream ${upstream.status}`, "upstream_error"), 500);
      }

      const streamBody = createImageEventStream({
        upstream,
        responseFormat,
        baseUrl,
        cookie,
        settings: settingsBundle.grok,
        n,
        onFinish: async ({ status, duration }) => {
          await addRequestLog(c.env.DB, {
            ip,
            model: requestedModel,
            duration: Number(duration.toFixed(2)),
            status,
            key_name: keyName,
            token_suffix: getTokenSuffix(chosen.token),
            error: status === 200 ? "" : "stream_error",
          });
        },
      });
      return new Response(streamBody, { status: 200, headers: streamHeaders() });
    }

    const calls = Math.ceil(n / 2);
    const urlsNested = await mapLimit(Array.from({ length: calls }), 3, async () => {
      return runImageCall({
        requestModel: requestedModel,
        prompt: imageCallPrompt("edit", prompt),
        fileIds,
        cookie,
        settings: settingsBundle.grok,
        responseFormat,
        baseUrl,
      });
    });
    const urls = urlsNested.flat().filter(Boolean);
    const selected = pickImageResults(urls, n);

    await recordImageLog({
      env: c.env,
      ip,
      model: requestedModel,
      start,
      keyName,
      status: 200,
      tokenSuffix: getTokenSuffix(chosen.token),
      error: "",
    });

    return c.json(buildImageJsonPayload(responseField, selected));
  } catch (e) {
    await recordImageLog({
      env: c.env,
      ip,
      model: requestedModel || "image",
      start,
      keyName,
      status: 500,
      error: e instanceof Error ? e.message : String(e),
    });
    return c.json(openAiError(e instanceof Error ? e.message : "Internal error", "internal_error"), 500);
  }
});

openAiRoutes.post("/uploads/image", async (c) => {
  try {
    const form = await c.req.formData();
    const file = form.get("file");
    if (!(file instanceof File)) return c.json(openAiError("Missing file", "missing_file"), 400);

    const mime = String(file.type || "application/octet-stream");
    if (!mime.toLowerCase().startsWith("image/"))
      return c.json(openAiError(`Unsupported mime: ${mime}`, "unsupported_file"), 400);

    const bytes = await file.arrayBuffer();
    const size = bytes.byteLength;
    const maxBytes = Math.min(25 * 1024 * 1024, Math.max(1, parseIntSafe(c.env.KV_CACHE_MAX_BYTES, 25 * 1024 * 1024)));
    if (size > maxBytes) return c.json(openAiError(`File too large (${size} > ${maxBytes})`, "file_too_large"), 413);

    const ext = (() => {
      const m = mime.toLowerCase();
      if (m === "image/png") return "png";
      if (m === "image/webp") return "webp";
      if (m === "image/gif") return "gif";
      if (m === "image/jpeg" || m === "image/jpg") return "jpg";
      return "jpg";
    })();

    const name = `upload-${crypto.randomUUID()}.${ext}`;
    const kvKey = `image/${name}`;

    const tz = parseIntSafe(c.env.CACHE_RESET_TZ_OFFSET_MINUTES, 480);
    const expiresAt = nextLocalMidnightExpirationSeconds(nowMs(), tz);

    await c.env.KV_CACHE.put(kvKey, bytes, {
      expiration: expiresAt,
      metadata: { contentType: mime, size },
    });

    const now = nowMs();
    await upsertCacheRow(c.env.DB, {
      key: kvKey,
      type: "image",
      size,
      content_type: mime,
      created_at: now,
      last_access_at: now,
      expires_at: expiresAt * 1000,
    });

    return c.json({
      url: `/images/${encodeURIComponent(name)}`,
      name,
      size_bytes: size,
    });
  } catch (e) {
    return c.json(openAiError(e instanceof Error ? e.message : "Internal error", "internal_error"), 500);
  }
});

openAiRoutes.options("/*", (c) => c.body(null, 204));
