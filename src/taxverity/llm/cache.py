"""Step 7.3 — an on-disk cache over the Step 7.2 client.

Its reason is affordability, not latency. Step 7.1 measured about 1.7
evidence-pack queries a minute on the free tier, so re-running an eval after a
prompt edit costs tens of minutes and the whole quota. A cached run costs
nothing and returns the same answers, which is also what makes an eval number
comparable across runs at all.

It is an offline tool. Nothing on a request path reads it: a served answer must
come from the model that is running now, not from whatever answered last week.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from taxverity.llm.client import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    LLM_STAGE_VERSION,
    Completion,
    LLMClient,
    Message,
)
from taxverity.observability import get_logger, redact

logger = get_logger(__name__)

# Bumping this invalidates every stored entry by design. Bump it when what the
# key covers changes, not for a cosmetic edit.
LLM_CACHE_VERSION = 1


class CachedLLMClient:
    """Read-through, write-through. A failure is never stored."""

    def __init__(self, inner: LLMClient, directory: Path) -> None:
        self._inner = inner
        self._directory = directory
        self.hits = 0
        self.misses = 0

    @property
    def directory(self) -> Path:
        return self._directory

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        response_format: Mapping[str, Any] | None = None,
        temperature: float | None = None,
    ) -> Completion:
        request = self._request(
            messages,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            temperature=temperature,
        )
        path = self._path(request)
        stored = _read(path, request)
        if stored is not None:
            self.hits += 1
            return stored
        self.misses += 1
        completion = self._inner.complete(
            messages,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            temperature=temperature,
        )
        _write(path, request, completion)
        return completion

    def _request(
        self,
        messages: Sequence[Message],
        *,
        max_completion_tokens: int,
        response_format: Mapping[str, Any] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        provider = self._inner.primary
        # redact() before the key, for the same two reasons CachedReranker does:
        # the key must describe what actually goes on the wire, and the stored
        # request lands on disk. It is applied again inside complete(), which is
        # harmless because a masked value contains nothing left to mask.
        return {
            "cache_version": LLM_CACHE_VERSION,
            "stage_version": LLM_STAGE_VERSION,
            "provider": provider.name,
            "model": provider.model,
            "extras": dict(provider.extras),
            "messages": [
                {"role": m.role, "content": redact(m.content)} for m in messages
            ],
            "max_completion_tokens": max_completion_tokens,
            "response_format": dict(response_format) if response_format else None,
            "temperature": temperature,
        }

    def _path(self, request: Mapping[str, Any]) -> Path:
        return self._directory / f"{_digest(request)}.json"


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _digest(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(request).encode("utf-8")).hexdigest()[:32]


def _read(path: Path, request: Mapping[str, Any]) -> Completion | None:
    """Anything unreadable is a miss. A cache must never break a run."""
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        logger.warning("unreadable llm cache entry %s: %s", path.name, error)
        return None
    # The stored request is what makes a hit checkable rather than trusted: a
    # digest collision or an edited file is a mismatch, not a wrong answer.
    if entry.get("request") != request:
        logger.warning("llm cache entry %s does not match its key", path.name)
        return None
    try:
        return Completion.model_validate(entry["completion"])
    except (KeyError, ValueError) as error:
        logger.warning("malformed llm cache entry %s: %s", path.name, error)
        return None


def _write(path: Path, request: Mapping[str, Any], completion: Completion) -> None:
    entry = {"request": request, "completion": completion.model_dump(mode="json")}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written whole and then moved into place: a run killed mid-write must not
    # leave a truncated entry that every later run has to discard.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(_canonical(entry) + "\n")
        os.replace(temporary, path)
    except OSError as error:
        logger.warning("could not store llm cache entry %s: %s", path.name, error)
        temporary.unlink(missing_ok=True)
