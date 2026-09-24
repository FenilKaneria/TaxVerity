"""Step 7.4 — Langfuse tracing, the first entry on rule 03's egress list.

The LLM call itself never logs its body (Step 7.2). A trace is the opposite: it
exists to carry the prompt and the completion off this machine, which is the
whole reason `redact()` is applied here rather than left to a call site.

Hand-rolled over the raw OTLP HTTP/JSON wire format rather than the Langfuse
SDK or an OpenTelemetry SDK, either of which would bring the whole
OpenTelemetry stack plus a second httpx major into an image ADR-076 keeps
deliberately thin. **Step 17.7 (ADR-111) migrated the endpoint from
Langfuse's legacy `/api/public/ingestion` batch-event envelope to
`/api/public/otel/v1/traces`** — the legacy endpoint is deprecated and
Langfuse Cloud drops it on its v4 upgrade (16 November 2026). One generation
is now one OTLP span, carrying Langfuse's `langfuse.observation.*` and
`langfuse.trace.*` attributes rather than a `trace-create`/`generation-create`
event pair; the wire format lives in `_span()` and `flush()` alone.

Tracing is never allowed to break, slow or fail a call. An unconfigured process
traces nothing, a failed post is dropped with a WARNING, and no post is ever
retried: losing a trace is cheap, delaying an answer is not.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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

TRACING_STAGE_VERSION = 2

OTLP_TRACES_PATH = "/api/public/otel/v1/traces"

# One attempt, short. A trace that misses is a trace; a retry is latency the
# user pays for on the answer path.
TRACE_TIMEOUT = 5.0

# The endpoint caps a batch at 3.5 MB. An evidence-pack prompt is a few tens of
# kilobytes, so a count is a sufficient proxy and needs no byte accounting.
MAX_BATCH_EVENTS = 20

# R21 Part B: a full batch is posted off the answer path. At most this many
# batches wait for the one sender thread; beyond that a batch is dropped and
# counted, never queued without bound against a slow or unreachable host.
MAX_PENDING_BATCHES = 2


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
        background: bool = False,
    ) -> None:
        """`background=True` (production, via `from_settings`) posts a full
        batch on a sender thread so no LLM-calling node waits on Langfuse;
        `flush()`/`close()` stay synchronous either way."""
        if not (public_key and secret_key and host):
            raise ValueError(
                "a Langfuse tracer needs a public key, a secret key and a host"
            )
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._headers = {
            "Authorization": f"Basic {token}",
            # Asks Langfuse Cloud to process the batch synchronously rather
            # than queueing it, so a trace is visible immediately.
            "x-langfuse-ingestion-version": "4",
        }
        self._url = host.rstrip("/") + OTLP_TRACES_PATH
        self._release = release
        self._environment = environment
        self._owns_client = http_client is None
        self._client = http_client or httpx2.Client(timeout=timeout)
        self._batch: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._sender = ThreadPoolExecutor(max_workers=1) if background else None
        self._pending = threading.BoundedSemaphore(MAX_PENDING_BATCHES)
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
        kwargs.setdefault("background", True)
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
        span = self._span(
            name,
            messages,
            completion=completion,
            model_parameters=model_parameters,
            latency_s=latency_s,
            error=error,
        )
        with self._lock:
            self._batch.append(span)
            if len(self._batch) < MAX_BATCH_EVENTS:
                return
            batch, self._batch = self._batch, []
        if self._sender is None:
            self._send(batch)
        elif self._pending.acquire(blocking=False):
            self._sender.submit(self._send_and_release, batch)
        else:
            self.dropped += len(batch)
            logger.warning("dropped %d trace spans: sender backlog full", len(batch))

    def _send_and_release(self, batch: list[dict[str, Any]]) -> None:
        try:
            self._send(batch)
        finally:
            self._pending.release()

    def _span(
        self,
        name: str,
        messages: Sequence[Message],
        *,
        completion: Completion | None,
        model_parameters: Mapping[str, Any] | None,
        latency_s: float | None,
        error: str | None,
    ) -> dict[str, Any]:
        now = time.time()
        started = now - latency_s if latency_s is not None else now
        # Rule 03, egress path 1. Applied on the way into the span so a
        # future caller cannot forget it; redact() leaves statutory tokens and
        # ordinary amounts intact, so evidence text crosses unchanged.
        prompt = [{"role": m.role, "content": redact(m.content)} for m in messages]
        output = redact(completion.text) if completion is not None else None
        metadata: dict[str, Any] = {"stage_version": TRACING_STAGE_VERSION}
        if completion is not None:
            # A degraded answer came from the fallback vendor. A trace recording
            # it as an ordinary one misreports which model was measured.
            metadata["degraded"] = completion.degraded
            metadata["finish_reason"] = completion.finish_reason
            metadata["reasoning_tokens"] = completion.usage.reasoning_tokens
            metadata["provider"] = completion.provider

        attributes = [
            _str_attr("langfuse.observation.type", "generation"),
            _str_attr("langfuse.trace.name", name),
            _str_attr("langfuse.observation.input", json.dumps(prompt)),
            _str_attr("langfuse.observation.metadata", json.dumps(metadata)),
        ]
        if output is not None:
            attributes.append(_str_attr("langfuse.observation.output", json.dumps(output)))
        if completion is not None:
            attributes.append(_str_attr("langfuse.observation.model.name", completion.model))
            attributes.append(
                _str_attr(
                    "langfuse.observation.usage_details",
                    json.dumps(
                        {
                            "input": completion.usage.prompt_tokens,
                            "output": completion.usage.completion_tokens,
                            "total": completion.usage.total_tokens,
                        }
                    ),
                )
            )
        if model_parameters:
            attributes.append(
                _str_attr(
                    "langfuse.observation.model.parameters",
                    json.dumps(dict(model_parameters)),
                )
            )
        if error is not None:
            attributes.append(_str_attr("langfuse.observation.status_message", error))

        return {
            "traceId": uuid.uuid4().hex,
            "spanId": uuid.uuid4().hex[:16],
            "name": name,
            "startTimeUnixNano": str(int(started * 1_000_000_000)),
            "endTimeUnixNano": str(int(now * 1_000_000_000)),
            "attributes": attributes,
            # OTLP status codes: 0 UNSET, 1 OK, 2 ERROR.
            "status": (
                {"code": 2, "message": error} if error is not None else {"code": 1}
            ),
        }

    def flush(self) -> None:
        with self._lock:
            if not self._batch:
                return
            batch, self._batch = self._batch, []
        self._send(batch)

    def _send(self, batch: list[dict[str, Any]]) -> None:
        resource_attributes = [_str_attr("service.name", "taxverity")]
        if self._release is not None:
            resource_attributes.append(_str_attr("service.version", self._release))
        if self._environment is not None:
            resource_attributes.append(
                _str_attr("deployment.environment.name", self._environment)
            )
        payload = {
            "resourceSpans": [
                {
                    "resource": {"attributes": resource_attributes},
                    "scopeSpans": [
                        {"scope": {"name": "taxverity.llm"}, "spans": batch}
                    ],
                }
            ]
        }
        try:
            response = self._client.post(
                self._url, json=payload, headers=self._headers
            )
        except httpx2.RequestError as error:
            self.dropped += len(batch)
            # The batch is not requeued: an unreachable Langfuse would otherwise
            # grow this list for the life of the process.
            logger.warning("dropped %d trace spans: %s", len(batch), error)
            return
        if response.status_code >= 400:
            self.dropped += len(batch)
            logger.warning(
                "dropped %d trace spans: Langfuse returned %d",
                len(batch),
                response.status_code,
            )
            return
        # OTLP's own partial-success shape: a 200 whose body still names some
        # spans the collector could not ingest.
        rejected = _rejected_span_count(response)
        if rejected:
            self.dropped += rejected
            logger.warning("Langfuse rejected %d trace spans", rejected)

    def close(self) -> None:
        if self._sender is not None:
            self._sender.shutdown(wait=True)
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


def _str_attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _rejected_span_count(response: httpx2.Response) -> int:
    try:
        body = response.json()
    except ValueError:
        return 0
    if not isinstance(body, Mapping):
        return 0
    partial = body.get("partialSuccess")
    if not isinstance(partial, Mapping):
        return 0
    try:
        return int(partial.get("rejectedSpans", 0) or 0)
    except (TypeError, ValueError):
        return 0
