export type ApiErrorCode =
  | "auth_failed"
  | "not_found"
  | "rate_limited"
  | "invalid_request"
  | "internal_error"
  | "upstream_unavailable"
  | "validation"
  | "network"
  | "unknown";

const KNOWN_CODES = new Set<string>([
  "auth_failed",
  "not_found",
  "rate_limited",
  "invalid_request",
  "internal_error",
  "upstream_unavailable",
]);

const DEFAULT_MESSAGES: Record<ApiErrorCode, string> = {
  auth_failed: "Incorrect email or password.",
  not_found: "That thread doesn't exist, or isn't yours.",
  rate_limited: "Too many attempts. Wait a moment and try again.",
  invalid_request: "That request wasn't valid.",
  internal_error: "Something went wrong on our end. Try again.",
  upstream_unavailable: "The service is temporarily unavailable. Try again shortly.",
  validation: "Check the highlighted fields and try again.",
  network: "Could not reach the server. Check your connection and try again.",
  unknown: "Something went wrong.",
};

export class ApiError extends Error {
  readonly status: number;
  readonly code: ApiErrorCode;

  constructor(status: number, code: ApiErrorCode, message?: string) {
    super(message ?? DEFAULT_MESSAGES[code]);
    this.status = status;
    this.code = code;
  }
}

interface ValidationDetailItem {
  msg?: unknown;
}

function firstValidationMessage(detail: unknown[]): string | undefined {
  const first = detail[0] as ValidationDetailItem | undefined;
  return typeof first?.msg === "string" ? first.msg : undefined;
}

export function parseError(status: number, body: unknown): ApiError {
  const detail =
    body !== null && typeof body === "object" && "detail" in body
      ? (body as { detail?: unknown }).detail
      : undefined;

  if (Array.isArray(detail)) {
    return new ApiError(status, "validation", firstValidationMessage(detail));
  }

  if (typeof detail === "string") {
    if (KNOWN_CODES.has(detail)) {
      return new ApiError(status, detail as ApiErrorCode);
    }
    return new ApiError(status, "invalid_request", detail);
  }

  return new ApiError(status, "unknown");
}
