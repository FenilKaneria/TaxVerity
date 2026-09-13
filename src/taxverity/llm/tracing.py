"""Step 7.4 — Langfuse tracing, the first entry on rule 03's egress list.

The LLM call itself never logs its body (Step 7.2). A trace is the opposite: it
exists to carry the prompt and the completion off this machine, which is the
whole reason `redact()` is applied here rather than left to a call site.

Hand-rolled over the documented ingestion envelope rather than the Langfuse SDK,
which would bring eight transitive packages — a second httpx major and the whole
OpenTelemetry stack — into an image ADR-076 keeps deliberately thin. The wire
format lives in `_events()` alone, so Langfuse v4's move to OTLP is one method.

Tracing is never allowed to break, slow or fail a call. An unconfigured process
traces nothing, a failed post is dropped with a WARNING, and no post is ever
retried: losing a trace is cheap, delaying an answer is not.
"""

from __future__ import annotations

import base64
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import httpx2

from taxverity.config import Settings
from taxverity.llm.client import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    Completion,
    Message,
)
from taxverity.observability import get_logger, redact

logger = get_logger(__name__)

TRACING_STAGE_VERSION = 1

INGESTION_PATH = "/api/public/ingestion"

# One attempt, short. A trace that misses is a trace; a retry is latency the
# user pays for on the answer path.
TRACE_TIMEOUT = 5.0

# The endpoint caps a batch at 3.5 MB. An evidence-pack prompt is a few tens of
# kilobytes, so a count is a sufficient proxy and needs no byte accounting.
MAX_BATCH_EVENTS = 20


@runtime_checkable
class Tracer(Protocol):
    def generation(
        self,
        name: str,
        messages: Sequence[Message],
        *,
        completion: Completion | None = None,
        model_parameters: Mapping[str, Any] | None = None,
        latency_s: float | None = None,
        error: str | None = None,
    ) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class NullTracer:
    """What an unconfigured process gets. Every call is a no-op."""

    def generation(self, name: str, messages: Sequence[Message], **_: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class LangfuseTracer:
    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
        *,
        http_client: httpx2.Client | None = None,
        timeout: float = TRACE_TIMEOUT,
        release: str | None = None,
        environment: str | None = None,
    ) -> None:
        if not (public_key and secret_key and host):
            raise ValueError(
                "a Langfuse tracer needs a public key, a secret key and a host"
            )
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._headers = {"Authorization": f"Basic {token}"}
        self._url = host.rstrip("/") + INGESTION_PATH
        self._release = release
        self._environment = environment
        self._owns_client = http_client is None
        self._client = http_client or httpx2.Client(timeout=timeout)
        self._batch: list[dict[str, Any]] = []
        self.dropped = 0

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> Tracer:
        """A process with no Langfuse configured traces nothing and says so once."""
        if not (
            settings.langfuse_public_key
            and settings.langfuse_secret_key
            and settings.langfuse_host
        ):
            logger.warning("no Langfuse configured: LLM calls are not traced")
            return NullTracer()
        return cls(
            settings.require("langfuse_public_key"),
            settings.require("langfuse_secret_key"),
            settings.require("langfuse_host"),
            **kwargs,
        )

    def generation(
        self,
        name: str,
        messages: Sequence[Message],
        *,
        completion: Completion | None = None,
        model_parameters: Mapping[str, Any] | None = None,
        latency_s: float | None = None,
        error: str | None = None,
    ) -> None:
        self._batch.extend(
            self._events(
                name,
                messages,
                completion=completion,
                model_parameters=model_parameters,
                latency_s=latency_s,
                error=error,
            )
        )
        if len(self._batch) >= MAX_BATCH_EVENTS:
            self.flush()

    def _events(
        self,
        name: str,
        messages: Sequence[Message],
        *,
        completion: Completion | None,
        model_parameters: Mapping[str, Any] | None,
        latency_s: float | None,
        error: str | None,
    ) -> list[dict[str, Any]]:
        now = time.time()
        started = now - latency_s if latency_s is not None else now
        # Rule 03, egress path 1. Applied on the way into the envelope so a
        # future caller cannot forget it; redact() leaves statutory tokens and
        # ordinary amounts intact, so evidence text crosses unchanged.
        prompt = [{"role": m.role, "content": redact(m.content)} for m in messages]
        output = redact(completion.text) if completion is not None else None
        trace_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {"stage_version": TRACING_STAGE_VERSION}
        if completion is not None:
            # A degraded answer came from the fallback vendor. A trace recording
            # it as an ordinary one misreports which model was measured.
            metadata["degraded"] = completion.degraded
            metadata["finish_reason"] = completion.finish_reason
            metadata["reasoning_tokens"] = completion.usage.reasoning_tokens
            metadata["provider"] = completion.provider
        trace: dict[str, Any] = {
            "id": trace_id,
            "name": name,
            "timestamp": _timestamp(started),
            "input": prompt,
            "output": output,
            "metadata": metadata,
        }
        generation: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "traceId": trace_id,
            "name": name,
            "startTime": _timestamp(started),
            "endTime": _timestamp(now),
            "input": prompt,
            "output": output,
            "metadata": metadata,
            "level": "ERROR" if error is not None else "DEFAULT",
        }
        if error is not None:
            generation["statusMessage"] = error
        if completion is not None:
            generation["model"] = completion.model
            generation["usage"] = {
                "promptTokens": completion.usage.prompt_tokens,
                "completionTokens": completion.usage.completion_tokens,
                "totalTokens": completion.usage.total_tokens,
            }
        if model_parameters:
            generation["modelParameters"] = dict(model_parameters)
        for body in (trace, generation):
            if self._release is not None:
                body["release"] = self._release
            if self._environment is not None:
                body["environment"] = self._environment
        return [
            _envelope("trace-create", trace),
            _envelope("generation-create", generation),
        ]

    def flush(self) -> None:
        if not self._batch:
            return
        batch, self._batch = self._batch, []
        try:
            response = self._client.post(
                self._url, json={"batch": batch}, headers=self._headers
            )
        except httpx2.RequestError as error:
            self.dropped += len(batch)
            # The batch is not requeued: an unreachable Langfuse would otherwise
            # grow this list for the life of the process.
            logger.warning("dropped %d trace events: %s", len(batch), error)
            return
        if response.status_code >= 400:
            self.dropped += len(batch)
            logger.warning(
                "dropped %d trace events: Langfuse returned %d",
                len(batch),
                response.status_code,
            )
            return
        # 207 is the documented partial success: the batch was accepted but some
        # of its events were not, and only the body says which.
        for rejected in _rejected(response):
            self.dropped += 1
            logger.warning("Langfuse rejected trace event %s", rejected)

    def close(self) -> None:
        self.flush()
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LangfuseTracer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TracedLLMClient:
    """Wraps a client, the same shape as CachedLLMClient and CachedReranker.

    A failed call is traced too, at ERROR, and then re-raised unchanged: the
    calls worth reading a trace for are usually the ones that did not work.
    """

    def __init__(self, inner: Any, tracer: Tracer, *, name: str = "llm") -> None:
        self._inner = inner
        self._tracer = tracer
        self._name = name

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        response_format: Mapping[str, Any] | None = None,
        temperature: float | None = None,
    ) -> Completion:
        parameters: dict[str, Any] = {"max_completion_tokens": max_completion_tokens}
        if response_format is not None:
            parameters["response_format"] = str(dict(response_format))
        if temperature is not None:
            parameters["temperature"] = temperature
        started = time.perf_counter()
        try:
            completion = self._inner.complete(
                messages,
                max_completion_tokens=max_completion_tokens,
                response_format=response_format,
                temperature=temperature,
            )
        except Exception as error:
            self._trace(
                messages,
                parameters,
                time.perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        self._trace(
            messages, parameters, time.perf_counter() - started, completion=completion
        )
        return completion

    def stream(
        self,
        messages: Sequence[Message],
        *,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Traced once the stream ends or fails, never per token."""
        parameters: dict[str, Any] = {
            "max_completion_tokens": max_completion_tokens,
            "stream": True,
        }
        if temperature is not None:
            parameters["temperature"] = temperature
        inner = self._inner.stream(
            messages, max_completion_tokens=max_completion_tokens, temperature=temperature
        )
        started = time.perf_counter()
        try:
            yield from inner
        except Exception as error:
            self._trace(
                messages,
                parameters,
                time.perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        # A cache hit replays stored text and carries no completion record.
        completion = getattr(inner, "completion", None)
        self._trace(messages, parameters, time.perf_counter() - started, completion=completion)

    def _trace(
        self,
        messages: Sequence[Message],
        parameters: Mapping[str, Any],
        latency_s: float,
        *,
        completion: Completion | None = None,
        error: str | None = None,
    ) -> None:
        # Tracing is observability, never a failure mode of the thing observed.
        try:
            self._tracer.generation(
                self._name,
                messages,
                completion=completion,
                model_parameters=parameters,
                latency_s=latency_s,
                error=error,
            )
        except Exception as failure:
            logger.warning("could not trace an llm call: %s", failure)

    def flush(self) -> None:
        self._tracer.flush()

    def close(self) -> None:
        self._tracer.close()

    def __enter__(self) -> TracedLLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _envelope(event_type: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "timestamp": _timestamp(time.time()),
        "body": dict(body),
    }


def _timestamp(epoch: float) -> str:
    moment = datetime.fromtimestamp(epoch, UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _rejected(response: httpx2.Response) -> list[str]:
    try:
        body = response.json()
    except ValueError:
        return []
    if not isinstance(body, Mapping):
        return []
    errors = body.get("errors") or []
    if not isinstance(errors, list):
        return []
    return [str(entry.get("id", entry))[:120] for entry in errors if entry]
