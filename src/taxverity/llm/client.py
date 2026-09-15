"""Step 7.2 — the one place this project talks to an LLM.

A thin internal client, not a gateway: two OpenAI-compatible providers do not
justify LiteLLM or Portkey. Groq's `openai/gpt-oss-120b` answers; Google's
Gemini Flash is a single cross-vendor fallback for availability only.

Every knob here was measured in the Step 7.1 spike rather than assumed —
`reasoning_effort` cannot be turned off and bills against the completion cap,
so the default is the cheapest setting that exists and every cap must be sized
for reasoning plus answer.

`complete()` is non-streaming. `stream()` (Step 10.3) is a separate contract
because a stream cannot retry once a token has been shown: retries and the
fallback apply only before the first token, and a failure after it raises.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx2
from pydantic import BaseModel, ConfigDict

from taxverity.config import Settings
from taxverity.observability import get_logger, redact

logger = get_logger(__name__)

LLM_STAGE_VERSION = 1

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 0.5
MAX_RETRY_AFTER = 60.0

# Reasoning tokens are billed against this and cannot be switched off (Step
# 7.1), so a cap sized for the visible answer alone truncates it.
DEFAULT_MAX_COMPLETION_TOKENS = 2_048

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Base: what Phase 13's graph catches."""


class LLMUnavailable(LLMError):
    """Every provider failed for a reason that is not our request's fault."""


class LLMRequestError(LLMError):
    """The request itself was refused. Retrying or failing over repeats it."""


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    # The Settings field holding this provider's key, read through require().
    settings_key: str
    # Request fields this provider understands and the other may not.
    extras: Mapping[str, Any] = field(default_factory=dict)


GROQ = Provider(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-120b",
    settings_key="groq_api_key",
    extras={"reasoning_effort": "low"},
)

# Google's OpenAI-compatible surface, so the fallback is a base URL, a model and
# a key rather than a second SDK, a second request shape and a second parser.
# `reasoning_effort` is deliberately absent: it is a gpt-oss control.
GEMINI = Provider(
    name="gemini",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    # gemini-2.5-flash was retired for new users; the live API's own 404
    # names gemini-3.6-flash as its replacement (confirmed 2026-09-14).
    model="gemini-3.6-flash",
    settings_key="gemini_api_key",
)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Billed inside completion_tokens; reported separately because it is the
    # difference between a cap that fits the answer and one that truncates it.
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    provider: str
    model: str
    finish_reason: str
    usage: Usage
    # True when the fallback answered: the caller may want to say so, and the
    # traces in Step 7.4 must not record a degraded answer as a normal one.
    degraded: bool


class LLMClient:
    def __init__(
        self,
        primary: Provider,
        primary_key: str,
        *,
        fallback: Provider | None = None,
        fallback_key: str | None = None,
        http_client: httpx2.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ) -> None:
        if not primary_key:
            raise ValueError("primary_key must be non-empty")
        if fallback is not None and not fallback_key:
            raise ValueError("a fallback provider needs a key")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, not {max_attempts}")
        self._primary = primary
        self._fallback = fallback
        self._keys = {primary.name: primary_key}
        if fallback is not None and fallback_key is not None:
            self._keys[fallback.name] = fallback_key
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._owns_client = http_client is None
        self._client = http_client or httpx2.Client(timeout=timeout)
        # Accounting is in tokens, not currency: both tiers are free, and a
        # price table written now would be a number nobody measured.
        self.tokens_used: dict[str, int] = {}

    @property
    def primary(self) -> Provider:
        """Which provider this client asks first. Step 7.3's cache key covers
        it: an answer from one model is not an answer from another."""
        return self._primary

    _PRIMARY = GROQ
    _FALLBACK = GEMINI

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> LLMClient:
        """The fallback is configured only when its key is present.

        A missing fallback key is an ordinary state, not an error — it means
        this process has one provider and says so once, at construction.
        """
        fallback_key = (
            settings.gemini_api_key.get_secret_value()
            if cls._FALLBACK is GEMINI and settings.gemini_api_key
            else settings.groq_api_key.get_secret_value()
            if cls._FALLBACK is GROQ and settings.groq_api_key
            else None
        )
        if fallback_key is None:
            logger.warning(
                "no %s configured: %s answers with no cross-vendor fallback",
                f"TAXVERITY_{cls._FALLBACK.settings_key.upper()}",
                cls._PRIMARY.name,
            )
        return cls(
            cls._PRIMARY,
            settings.require(cls._PRIMARY.settings_key),
            fallback=cls._FALLBACK if fallback_key else None,
            fallback_key=fallback_key,
            **kwargs,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        response_format: Mapping[str, Any] | None = None,
        temperature: float | None = None,
    ) -> Completion:
        if not messages:
            raise ValueError("messages must be non-empty")
        # Rule 03, egress path 5. Applied here rather than at each call site so
        # the control holds structurally; redact() leaves statutory tokens and
        # ordinary amounts intact, so evidence text crosses unchanged.
        body: dict[str, Any] = {
            "messages": [
                {"role": m.role, "content": redact(m.content)} for m in messages
            ],
            "max_completion_tokens": max_completion_tokens,
        }
        if response_format is not None:
            body["response_format"] = dict(response_format)
        if temperature is not None:
            body["temperature"] = temperature

        started = time.perf_counter()
        try:
            completion = self._call(self._primary, body, degraded=False)
        except LLMUnavailable as primary_error:
            if self._fallback is None:
                raise
            logger.warning(
                "%s unavailable (%s); falling back to %s",
                self._primary.name,
                primary_error,
                self._fallback.name,
            )
            try:
                completion = self._call(self._fallback, body, degraded=True)
            except LLMUnavailable as fallback_error:
                raise LLMUnavailable(
                    f"{self._primary.name} failed ({primary_error}) and "
                    f"{self._fallback.name} failed ({fallback_error})"
                ) from fallback_error
        logger.info(
            "llm %s/%s answered in %.2fs: %d prompt + %d completion tokens "
            "(%d reasoning), finish_reason=%s%s",
            completion.provider,
            completion.model,
            time.perf_counter() - started,
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            completion.usage.reasoning_tokens,
            completion.finish_reason,
            ", degraded" if completion.degraded else "",
        )
        return completion

    def stream(
        self,
        messages: Sequence[Message],
        *,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        temperature: float | None = None,
    ) -> CompletionStream:
        if not messages:
            raise ValueError("messages must be non-empty")
        # Rule 03, egress path 5, exactly as complete() applies it.
        body: dict[str, Any] = {
            "messages": [
                {"role": m.role, "content": redact(m.content)} for m in messages
            ],
            "max_completion_tokens": max_completion_tokens,
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        return CompletionStream(self, body)

    def _open_stream(self, body: dict[str, Any]) -> _OpenStream:
        """Connect and read up to the first token, failing over if that fails."""
        try:
            return self._open_with_retries(self._primary, body, degraded=False)
        except LLMUnavailable as primary_error:
            if self._fallback is None:
                raise
            logger.warning(
                "%s stream unavailable (%s); falling back to %s",
                self._primary.name,
                primary_error,
                self._fallback.name,
            )
            try:
                return self._open_with_retries(self._fallback, body, degraded=True)
            except LLMUnavailable as fallback_error:
                raise LLMUnavailable(
                    f"{self._primary.name} failed ({primary_error}) and "
                    f"{self._fallback.name} failed ({fallback_error})"
                ) from fallback_error

    def _open_with_retries(
        self, provider: Provider, body: dict[str, Any], *, degraded: bool
    ) -> _OpenStream:
        payload = {**body, **provider.extras, "model": provider.model}
        headers = {"Authorization": f"Bearer {self._keys[provider.name]}"}
        url = f"{provider.base_url}/chat/completions"
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            delay = self._backoff_base * 2 ** (attempt - 1)
            request = self._client.build_request("POST", url, json=payload, headers=headers)
            response = None
            try:
                response = self._client.send(request, stream=True)
                if response.status_code in RETRYABLE_STATUS:
                    last = LLMUnavailable(f"{provider.name} returned {response.status_code}")
                    delay = _retry_after(response) or delay
                    response.close()
                elif response.status_code >= 400:
                    text = response.read().decode("utf-8", "replace")[:200]
                    response.close()
                    raise LLMRequestError(
                        f"{provider.name} returned {response.status_code}: {text}"
                    )
                else:
                    opened = _OpenStream(provider, degraded, response, response.iter_lines())
                    opened.read_until_first_token()
                    return opened
            except (httpx2.RequestError, httpx2.StreamError) as error:
                last = error
                if response is not None:
                    response.close()
            except LLMUnavailable as error:
                last = error
                if response is not None:
                    response.close()
            if attempt < self._max_attempts:
                logger.warning(
                    "%s stream failed before its first token (attempt %d/%d): %s; "
                    "retrying in %.2fs",
                    provider.name,
                    attempt,
                    self._max_attempts,
                    last,
                    delay,
                )
                time.sleep(delay)
        raise LLMUnavailable(
            f"{provider.name} stream failed after {self._max_attempts} attempts: {last}"
        ) from last

    def _record_usage(self, provider: Provider, usage: Usage) -> None:
        self.tokens_used[provider.name] = (
            self.tokens_used.get(provider.name, 0) + usage.total_tokens
        )

    def _call(
        self, provider: Provider, body: dict[str, Any], *, degraded: bool
    ) -> Completion:
        payload = {**body, **provider.extras, "model": provider.model}
        response_body = self._post(provider, payload)
        return self._parse(provider, response_body, degraded=degraded)

    def _post(self, provider: Provider, payload: dict[str, Any]) -> dict[str, Any]:
        # The payload carries the user's own words. It is never logged — not on
        # success, not on retry, not on failure.
        headers = {"Authorization": f"Bearer {self._keys[provider.name]}"}
        url = f"{provider.base_url}/chat/completions"
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            delay = self._backoff_base * 2 ** (attempt - 1)
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except httpx2.RequestError as error:
                last = error
            else:
                if response.status_code not in RETRYABLE_STATUS:
                    if response.status_code >= 400:
                        # Our request, not their availability: the same request
                        # is refused by a retry and by the other vendor too.
                        raise LLMRequestError(
                            f"{provider.name} returned {response.status_code}: "
                            f"{response.text[:200]}"
                        )
                    try:
                        return response.json()
                    except ValueError as error:
                        raise LLMUnavailable(
                            f"{provider.name} returned a body that is not JSON"
                        ) from error
                last = LLMUnavailable(
                    f"{provider.name} returned {response.status_code}"
                )
                delay = _retry_after(response) or delay
            if attempt < self._max_attempts:
                logger.warning(
                    "%s call failed (attempt %d/%d): %s; retrying in %.2fs",
                    provider.name,
                    attempt,
                    self._max_attempts,
                    last,
                    delay,
                )
                time.sleep(delay)
        raise LLMUnavailable(
            f"{provider.name} failed after {self._max_attempts} attempts: {last}"
        ) from last

    def _parse(
        self, provider: Provider, body: Mapping[str, Any], *, degraded: bool
    ) -> Completion:
        try:
            choice = body["choices"][0]
            text = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason") or "unknown"
        except (KeyError, IndexError, TypeError) as error:
            # A malformed body is an availability failure, not a bug of ours:
            # the caller degrades on it rather than seeing a bare KeyError.
            raise LLMUnavailable(
                f"malformed {provider.name} response: {error!r}"
            ) from error
        raw = body.get("usage") or {}
        details = raw.get("completion_tokens_details") or {}
        usage = Usage(
            prompt_tokens=int(raw.get("prompt_tokens", 0)),
            completion_tokens=int(raw.get("completion_tokens", 0)),
            reasoning_tokens=int(details.get("reasoning_tokens", 0)),
        )
        self.tokens_used[provider.name] = (
            self.tokens_used.get(provider.name, 0) + usage.total_tokens
        )
        return Completion(
            text=text,
            provider=provider.name,
            model=str(body.get("model") or provider.model),
            finish_reason=finish_reason,
            usage=usage,
            degraded=degraded,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _OpenStream:
    """One provider's server-sent-event response, read one chunk at a time."""

    def __init__(
        self,
        provider: Provider,
        degraded: bool,
        response: httpx2.Response,
        lines: Iterator[str],
    ) -> None:
        self.provider = provider
        self.degraded = degraded
        self.response = response
        self._lines = lines
        self.pending: list[str] = []
        self.done = False
        self.model = provider.model
        self.finish_reason = "unknown"
        self.usage = Usage()

    def read_until_first_token(self) -> None:
        while not self.pending and not self.done:
            self._read_chunk()

    def next_deltas(self) -> list[str]:
        if not self.pending and not self.done:
            self._read_chunk()
        deltas, self.pending = self.pending, []
        return deltas

    def _read_chunk(self) -> None:
        for line in self._lines:
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                self.done = True
                return
            self._parse(data)
            if self.pending:
                return
        self.done = True

    def _parse(self, data: str) -> None:
        try:
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
        except (ValueError, AttributeError) as error:
            raise LLMUnavailable(
                f"malformed {self.provider.name} stream chunk: {error!r}"
            ) from error
        self.model = str(chunk.get("model") or self.model)
        for choice in choices:
            content = (choice.get("delta") or {}).get("content")
            if content:
                self.pending.append(content)
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        # OpenAI-compatible providers put usage on the last chunk; Groq nests
        # it under x_groq.
        raw = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
        if raw:
            details = raw.get("completion_tokens_details") or {}
            self.usage = Usage(
                prompt_tokens=int(raw.get("prompt_tokens", 0)),
                completion_tokens=int(raw.get("completion_tokens", 0)),
                reasoning_tokens=int(details.get("reasoning_tokens", 0)),
            )


class CompletionStream:
    """Text deltas as they arrive. `completion` is set once the stream ends.

    Opening happens on first iteration, so constructing a stream sends nothing.
    """

    def __init__(self, client: LLMClient, body: dict[str, Any]) -> None:
        self._client = client
        self._body = body
        self._started = False
        self.completion: Completion | None = None

    def __iter__(self) -> Iterator[str]:
        if self._started:
            raise RuntimeError("a completion stream can be read only once")
        self._started = True
        started = time.perf_counter()
        opened = self._client._open_stream(self._body)
        parts: list[str] = []
        try:
            while True:
                try:
                    deltas = opened.next_deltas()
                except (httpx2.RequestError, httpx2.StreamError, LLMUnavailable) as error:
                    # Tokens are already on screen, so there is no retry and no
                    # fallback: a second answer would contradict the first.
                    raise LLMUnavailable(
                        f"{opened.provider.name} stream interrupted after the first "
                        f"token: {error}"
                    ) from error
                if not deltas and opened.done:
                    break
                for delta in deltas:
                    parts.append(delta)
                    yield delta
        finally:
            opened.response.close()
        self._client._record_usage(opened.provider, opened.usage)
        self.completion = Completion(
            text="".join(parts),
            provider=opened.provider.name,
            model=opened.model,
            finish_reason=opened.finish_reason,
            usage=opened.usage,
            degraded=opened.degraded,
        )
        logger.info(
            "llm stream %s/%s finished in %.2fs: %d prompt + %d completion tokens "
            "(%d reasoning), finish_reason=%s%s",
            opened.provider.name,
            opened.model,
            time.perf_counter() - started,
            opened.usage.prompt_tokens,
            opened.usage.completion_tokens,
            opened.usage.reasoning_tokens,
            opened.finish_reason,
            ", degraded" if opened.degraded else "",
        )


def _retry_after(response: httpx2.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return min(max(float(value), 0.0), MAX_RETRY_AFTER)
    except ValueError:
        return None
