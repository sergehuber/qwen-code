/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { getErrorStatus } from './errors.js';
import { isApiError, isStructuredError } from './quotaErrorDetection.js';

// ─── Shared helpers ───────────────────────────────────────────────────────────

/** Extracts the raw string message from an unknown error value, or null. */
function toRawMessage(error: unknown): string | null {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  return null;
}

/**
 * Tries to parse a JSON-embedded ApiError from a string (e.g. a streamed SSE
 * frame that opens with non-JSON text followed by `{...}`).
 * Returns null when the string contains no valid ApiError JSON.
 */
function tryParseApiError(s: string): unknown {
  const i = s.indexOf('{');
  if (i === -1) return null;
  try {
    const p = JSON.parse(s.substring(i)) as unknown;
    return isApiError(p) ? p : null;
  } catch {
    return null;
  }
}

// ─── Rate-limit detection ─────────────────────────────────────────────────────

// Known rate-limit error codes across providers.
// 429  - Standard HTTP "Too Many Requests" (DashScope TPM, OpenAI, etc.)
// 503  - Provider throttling/overload (treated as rate-limit for retry UI)
// 1302 - Z.AI GLM rate limit (https://docs.z.ai/api-reference/api-code)
// 1305 - DashScope/IdealTalk internal rate limit (issue #1918)
const RATE_LIMIT_ERROR_CODES = new Set([429, 503, 1302, 1305]);

export interface RetryInfo {
  /** Formatted error message for display, produced by parseAndFormatApiError. */
  message?: string;
  /** Current retry attempt (1-based). */
  attempt: number;
  /** Max retries allowed. */
  maxRetries: number;
  /** Delay in milliseconds before the retry happens. */
  delayMs: number;
  /** When called, resolves the delay promise early so the retry happens immediately. */
  skipDelay: () => void;
}

export interface RateLimitErrorDetails {
  statusCode?: number;
  providerCode?: string;
  providerMessage?: string;
  requestId?: string;
  transport: 'http' | 'sse' | 'unknown';
}

export interface RateLimitRetryDelayOptions {
  initialDelayMs: number;
  maxDelayMs: number;
  error?: unknown;
}

/**
 * Detects rate-limit / throttling errors.
 *
 * @param extraCodes - Additional error codes to treat as rate-limit errors,
 *   merged with the built-in set at call time (not mutating the default set).
 */
export function isRateLimitError(
  error: unknown,
  extraCodes?: readonly number[],
): boolean {
  const code = extractErrorCode(error);
  if (code === null) return false;
  if (RATE_LIMIT_ERROR_CODES.has(code)) return true;
  if (extraCodes?.includes(code)) return true;
  return false;
}

/**
 * Extracts structured diagnostic fields from known HTTP and SSE rate-limit
 * error shapes without changing retryability decisions.
 */
export function getRateLimitErrorDetails(
  error: unknown,
): RateLimitErrorDetails {
  const statusCode = getErrorStatus(error);
  const payload = getProviderErrorPayload(error);
  const message = getRawErrorMessage(error);
  const transport =
    message?.includes('event:error') || message?.includes('HTTP_STATUS/')
      ? 'sse'
      : statusCode !== undefined
        ? 'http'
        : 'unknown';

  return {
    ...(statusCode !== undefined ? { statusCode } : {}),
    ...(payload?.code !== undefined
      ? { providerCode: String(payload.code) }
      : {}),
    ...(payload?.message !== undefined
      ? { providerMessage: payload.message }
      : {}),
    ...(payload?.requestId !== undefined
      ? { requestId: payload.requestId }
      : {}),
    transport,
  };
}

/**
 * Calculates the stream-side rate-limit retry delay.
 *
 * Retry-After is treated as a provider-supplied minimum wait, but the final
 * delay is still capped by maxDelayMs so an interactive session cannot be
 * parked indefinitely by an oversized header.
 */
export function getRateLimitRetryDelayMs(
  attempt: number,
  options: RateLimitRetryDelayOptions,
): number {
  const normalizedAttempt = Math.max(1, attempt);
  const exponentialDelayMs =
    options.initialDelayMs * Math.pow(2, normalizedAttempt - 1);
  const retryAfterMs = getRetryAfterDelayMs(options.error);
  const delayMs =
    retryAfterMs === null
      ? exponentialDelayMs
      : Math.max(exponentialDelayMs, retryAfterMs);
  return Math.min(delayMs, options.maxDelayMs);
}

/**
 * Extracts the numeric error code from various error shapes.
 * Mirrors the same parsing patterns used by parseAndFormatApiError.
 */
function extractErrorCode(error: unknown): number | null {
  // ApiError (.error.code) — fall through when the code is not a finite number
  // (e.g. DashScope `"code":"Throttling.AllocationQuota"`) so later handlers
  // can still recover a status from `.status` or the message.
  if (isApiError(error)) {
    const n = Number(error.error.code);
    if (Number.isFinite(n) && n > 0) return n;
  }

  // JSON in string / Error.message — check BEFORE isStructuredError because
  // Error instances also satisfy isStructuredError (both have .message).
  const raw = toRawMessage(error);
  if (raw) {
    const embedded = tryParseApiError(raw);
    if (isApiError(embedded)) {
      const n = Number(embedded.error.code);
      if (Number.isFinite(n) && n > 0) return n;
    }
  }

  // StructuredError (.status) — plain objects from Gemini SDK.
  if (isStructuredError(error) && typeof error.status === 'number') {
    return error.status;
  }

  // HttpError (.status on Error subclass)
  if (error instanceof Error && 'status' in error) {
    const s = (error as { status?: unknown }).status;
    if (typeof s === 'number') return s;
  }

  // Final fallback: parses `HTTP_STATUS/NNN` out of streamed SSE error frames.
  return getErrorStatus(error) ?? null;
}

interface ProviderErrorPayload {
  code?: string | number;
  message?: string;
  requestId?: string;
}

function getProviderErrorPayload(error: unknown): ProviderErrorPayload | null {
  for (const payload of getJsonPayloads(error)) {
    if (typeof payload !== 'object' || payload === null) continue;

    const direct = payload as {
      code?: unknown;
      message?: unknown;
      request_id?: unknown;
      requestId?: unknown;
    };
    const nestedError = (payload as { error?: unknown }).error;
    const nested =
      typeof nestedError === 'object' && nestedError !== null
        ? (nestedError as {
            code?: unknown;
            message?: unknown;
            request_id?: unknown;
            requestId?: unknown;
          })
        : undefined;
    const source = nested ?? direct;
    const code =
      typeof source.code === 'string' || typeof source.code === 'number'
        ? source.code
        : undefined;
    const message =
      typeof source.message === 'string' ? source.message : undefined;
    const requestId =
      typeof source.request_id === 'string'
        ? source.request_id
        : typeof source.requestId === 'string'
          ? source.requestId
          : typeof direct.request_id === 'string'
            ? direct.request_id
            : typeof direct.requestId === 'string'
              ? direct.requestId
              : undefined;

    if (
      code !== undefined ||
      message !== undefined ||
      requestId !== undefined
    ) {
      return { code, message, requestId };
    }
  }

  if (isApiError(error)) {
    return {
      code: error.error.code,
      message: error.error.message,
    };
  }

  return null;
}

function getJsonPayloads(error: unknown): unknown[] {
  const message = getRawErrorMessage(error);
  if (!message) return [];

  const payloads: unknown[] = [];
  for (const line of message.split(/\r?\n/)) {
    if (!line.startsWith('data:')) continue;
    const data = line.slice('data:'.length).trim();
    if (!data || data === '[DONE]') continue;
    try {
      payloads.push(JSON.parse(data) as unknown);
    } catch {
      /* ignore invalid SSE data */
    }
  }

  if (payloads.length > 0) return payloads;

  const jsonStart = message.indexOf('{');
  const jsonEnd = message.lastIndexOf('}');
  if (jsonStart !== -1 && jsonEnd > jsonStart) {
    try {
      payloads.push(
        JSON.parse(message.slice(jsonStart, jsonEnd + 1)) as unknown,
      );
    } catch {
      /* ignore non-JSON message fragments */
    }
  }

  return payloads;
}

function getRawErrorMessage(error: unknown): string | null {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  return null;
}

function getRetryAfterDelayMs(error: unknown): number | null {
  const value =
    getHeaderValue(error, 'retry-after') ??
    getResponseHeaderValue(error, 'retry-after');
  if (value === null) return null;

  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return seconds * 1000;
  }

  const retryAtMs = Date.parse(value);
  if (!Number.isFinite(retryAtMs)) return null;

  const delayMs = retryAtMs - Date.now();
  return delayMs > 0 ? delayMs : 0;
}

function getHeaderValue(error: unknown, headerName: string): string | null {
  if (!hasHeaders(error)) return null;

  const { headers } = error;
  if (typeof headers.get === 'function') {
    const value = headers.get(headerName);
    return typeof value === 'string' ? value : null;
  }

  if (typeof headers !== 'object' || headers === null) return null;

  const lowerHeaderName = headerName.toLowerCase();
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() !== lowerHeaderName) continue;
    return typeof value === 'string' ? value : null;
  }

  return null;
}

function getResponseHeaderValue(
  error: unknown,
  headerName: string,
): string | null {
  if (!hasResponseHeaders(error)) return null;
  return getHeaderValue(error.response, headerName);
}

function hasHeaders(error: unknown): error is {
  headers: { get?: (name: string) => unknown } | Record<string, unknown>;
} {
  return (
    typeof error === 'object' &&
    error !== null &&
    'headers' in error &&
    error.headers != null
  );
}

function hasResponseHeaders(error: unknown): error is {
  response: {
    headers: { get?: (name: string) => unknown } | Record<string, unknown>;
  };
} {
  return (
    typeof error === 'object' &&
    error !== null &&
    'response' in error &&
    typeof error.response === 'object' &&
    error.response !== null &&
    'headers' in error.response &&
    error.response.headers != null
  );
}

// ─── Context-limit detection ──────────────────────────────────────────────────

// Substrings found in context-limit error messages across providers.
// Ordered from most specific to most generic.
// - OpenAI / OpenAI-compatible (Qwen, DashScope): "context_length_exceeded",
//   "maximum context length", "max_tokens is too large", "prompt_too_long"
// - Gemini: "context window", "token count exceeds", "request too large"
// - Anthropic: "prompt is too long", "input too long"
// - Qwen/local providers: "exceeds the available context size"
// - Generic: "token limit exceeded", "context limit"
const CONTEXT_LIMIT_FRAGMENTS = [
  'context_length_exceeded',
  'maximum context length',
  'exceeds the available context size',
  'context window',
  'token limit exceeded',
  'context limit',
  'prompt is too long',
  'input too long',
  'input token count',
  'request too large',
  'max_tokens is too large',
  'prompt_too_long',
] as const;

function matchesContextLimit(message: string): boolean {
  const lower = message.toLowerCase();
  return CONTEXT_LIMIT_FRAGMENTS.some((f) => lower.includes(f));
}

/**
 * Yields candidate message strings from an error in priority order so the
 * caller can stop at the first match without inspecting all error shapes.
 */
function* candidateMessages(error: unknown): Generator<string> {
  // Direct ApiError: { error: { message, status } }
  if (isApiError(error)) {
    yield error.error.message;
    if (typeof error.error.status === 'string') yield error.error.status;
    return; // fully handled; no further shapes apply
  }

  // Raw string / Error.message, possibly containing embedded JSON
  const raw = toRawMessage(error);
  if (raw) {
    yield raw;
    const embedded = tryParseApiError(raw);
    if (isApiError(embedded)) {
      yield embedded.error.message;
      if (typeof embedded.error.status === 'string')
        yield embedded.error.status;
    }
  }

  // Plain StructuredError (Gemini SDK plain object). Skip Error instances
  // because their .message was already yielded via toRawMessage above.
  if (isStructuredError(error) && !(error instanceof Error)) {
    yield error.message;
  }
}

/**
 * Returns the matching error message if `error` is a context-window overflow,
 * or `null` otherwise. Useful for logging the original API message.
 */
export function getContextLimitMessage(error: unknown): string | null {
  for (const msg of candidateMessages(error)) {
    if (matchesContextLimit(msg)) return msg;
  }
  return null;
}

/**
 * Returns `true` when `error` signals that the model's context / token window
 * was exceeded. Checks all common error shapes across providers.
 */
export function isContextLimitError(error: unknown): boolean {
  return getContextLimitMessage(error) !== null;
}
