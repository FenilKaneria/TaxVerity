"""Step 4.3 — the embedding client: batching, retry, model-identity assertion."""

from __future__ import annotations

import logging

import httpx2
import pytest
from fastapi.testclient import TestClient

from taxverity.embedding.backends import (
    MAX_BATCH,
    STUB_DIM,
    STUB_MODEL_ID,
    EmbedKind,
    ModelInfo,
    StubEmbedder,
)
from taxverity.embedding.client import (
    EmbeddingClient,
    EmbeddingServiceError,
    EmbedKindError,
    ModelIdentityError,
)
from taxverity.embedding.service import create_app

BASE_URL = "http://embed.test"

CLIENT_LOGGER = "taxverity.embedding.client"

SERVED = ModelInfo(
    model_id="model-a", dim=4, revision="r1", runtime="torch", encoding="v1"
)
OTHER = ModelInfo(
    model_id="model-b", dim=4, revision="r1", runtime="torch", encoding="v1"
)

DOCUMENT = EmbedKind.DOCUMENT
QUERY = EmbedKind.QUERY


class Recorder:
    """A MockTransport handler that scripts responses and records requests."""

    def __init__(self, *, embed_responses=None, info=SERVED):
        self.info = info
        self.embed_responses = list(embed_responses or [])
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path == "/model-info":
            return httpx2.Response(200, json=self.info.model_dump())
        if self.embed_responses:
            scripted = self.embed_responses.pop(0)
            if isinstance(scripted, Exception):
                raise scripted
            if isinstance(scripted, httpx2.Response):
                return scripted
        body = httpx2.Response(200, content=request.read()).json()
        count = len(body["texts"])
        return httpx2.Response(
            200,
            json={
                "embeddings": [[1.0] + [0.0] * (self.info.dim - 1)] * count,
                "model": self.info.model_dump(),
                "kind": body["kind"],
            },
        )

    @property
    def embed_bodies(self) -> list[dict]:
        return [
            httpx2.Response(200, content=r.read()).json()
            for r in self.requests
            if r.url.path == "/embed"
        ]

    @property
    def embed_calls(self) -> list[list[str]]:
        return [body["texts"] for body in self.embed_bodies]


def make_client(handler: Recorder, **kwargs) -> EmbeddingClient:
    kwargs.setdefault("backoff_base", 0.0)
    http_client = httpx2.Client(
        base_url=BASE_URL, transport=httpx2.MockTransport(handler)
    )
    return EmbeddingClient(http_client=http_client, **kwargs)


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        formatter = logging.Formatter("%(levelname)s %(message)s")
        return "\n".join(formatter.format(r) for r in self.records)


@pytest.fixture
def client_logs():
    """Captures on the client's own logger, not via caplog.

    `configure_logging()` sets `propagate = False` on the `taxverity` root and
    clears its handlers, so caplog's root handler sees nothing and any
    assertion made through it is vacuous (Step 4.2 note 5).
    """
    logger = logging.getLogger(CLIENT_LOGGER)
    handler = CaptureHandler()
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# --- construction -------------------------------------------------------------


def test_identity_is_pinned_from_model_info():
    client = make_client(Recorder())
    assert client.model == SERVED


def test_model_info_is_fetched_exactly_once():
    handler = Recorder()
    make_client(handler)
    assert [r.url.path for r in handler.requests] == ["/model-info"]


def test_matching_expected_identity_is_accepted():
    client = make_client(Recorder(), expected=SERVED)
    assert client.model == SERVED


def test_mismatched_expected_identity_raises():
    with pytest.raises(ModelIdentityError, match="model-b"):
        make_client(Recorder(), expected=OTHER)


def test_a_differing_revision_alone_is_a_mismatch():
    # Same weights id, different weights. Two embedding spaces (ADR-026).
    other_revision = SERVED.model_copy(update={"revision": "r2"})
    with pytest.raises(ModelIdentityError):
        make_client(Recorder(), expected=other_revision)


def test_a_differing_runtime_alone_is_a_mismatch():
    # torch vs ONNX vs int8 change the numbers identical weights emit.
    other_runtime = SERVED.model_copy(update={"runtime": "onnx"})
    with pytest.raises(ModelIdentityError):
        make_client(Recorder(), expected=other_runtime)


def test_a_differing_encoding_alone_is_a_mismatch():
    # Same weights, same runtime, edited prompt recipe: a different embedding
    # space that the other four fields report as identical (ADR-069).
    other_encoding = SERVED.model_copy(update={"encoding": "v2"})
    with pytest.raises(ModelIdentityError):
        make_client(Recorder(), expected=other_encoding)


def test_identity_mismatch_is_not_retried():
    handler = Recorder()
    with pytest.raises(ModelIdentityError):
        make_client(handler, expected=OTHER, max_attempts=3)
    assert len(handler.requests) == 1


def test_batch_size_above_the_service_ceiling_is_refused():
    with pytest.raises(ValueError, match="batch_size"):
        make_client(Recorder(), batch_size=MAX_BATCH + 1)


def test_batch_size_below_one_is_refused():
    with pytest.raises(ValueError, match="batch_size"):
        make_client(Recorder(), batch_size=0)


def test_max_attempts_below_one_is_refused():
    with pytest.raises(ValueError, match="max_attempts"):
        make_client(Recorder(), max_attempts=0)


# --- batching -----------------------------------------------------------------


def test_a_short_input_is_one_request():
    handler = Recorder()
    client = make_client(handler, batch_size=4)
    client.embed(["a", "b"], DOCUMENT)
    assert handler.embed_calls == [["a", "b"]]


def test_a_long_input_is_split_and_order_is_preserved():
    handler = Recorder()
    client = make_client(handler, batch_size=3)
    texts = [f"t{i}" for i in range(7)]
    vectors = client.embed(texts, DOCUMENT)
    assert handler.embed_calls == [texts[0:3], texts[3:6], texts[6:7]]
    assert len(vectors) == 7


def test_an_exact_multiple_of_the_batch_size_makes_no_empty_request():
    handler = Recorder()
    client = make_client(handler, batch_size=2)
    client.embed(["a", "b", "c", "d"], DOCUMENT)
    assert handler.embed_calls == [["a", "b"], ["c", "d"]]


def test_empty_input_makes_no_request_at_all():
    handler = Recorder()
    client = make_client(handler)
    # The service refuses an empty batch with a 422 (min_length=1), so the
    # client must not send one.
    assert client.embed([], DOCUMENT) == []
    assert handler.embed_calls == []


def test_the_default_batch_size_is_the_service_ceiling():
    handler = Recorder()
    client = make_client(handler)
    client.embed([f"t{i}" for i in range(MAX_BATCH + 1)], DOCUMENT)
    assert [len(call) for call in handler.embed_calls] == [MAX_BATCH, 1]


# --- per-response identity and shape ------------------------------------------


def test_identity_changing_mid_session_raises():
    handler = Recorder(
        embed_responses=[
            httpx2.Response(
                200,
                json={
                    "embeddings": [[1.0, 0.0, 0.0, 0.0]],
                    "model": OTHER.model_dump(),
                    "kind": "document",
                },
            )
        ]
    )
    client = make_client(handler)
    with pytest.raises(ModelIdentityError, match="mid-session"):
        client.embed(["a"], DOCUMENT)


def test_a_wrong_vector_count_raises():
    handler = Recorder(
        embed_responses=[
            httpx2.Response(
                200,
                json={
                    "embeddings": [],
                    "model": SERVED.model_dump(),
                    "kind": "document",
                },
            )
        ]
    )
    client = make_client(handler)
    with pytest.raises(EmbeddingServiceError, match="asked for 1"):
        client.embed(["a"], DOCUMENT)


def test_a_width_contradicting_the_declared_dim_raises():
    handler = Recorder(
        embed_responses=[
            httpx2.Response(
                200,
                json={
                    "embeddings": [[1.0, 0.0]],
                    "model": SERVED.model_dump(),
                    "kind": "document",
                },
            )
        ]
    )
    client = make_client(handler)
    with pytest.raises(EmbeddingServiceError, match="dim=4"):
        client.embed(["a"], DOCUMENT)


# --- retry --------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_a_retryable_status_is_retried_then_succeeds(status):
    handler = Recorder(embed_responses=[httpx2.Response(status)])
    client = make_client(handler)
    assert len(client.embed(["a"], DOCUMENT)) == 1
    assert len(handler.embed_calls) == 2


def test_a_transport_error_is_retried():
    handler = Recorder(embed_responses=[httpx2.ConnectError("refused", request=None)])
    client = make_client(handler)
    assert len(client.embed(["a"], DOCUMENT)) == 1
    assert len(handler.embed_calls) == 2


def test_retries_are_bounded_and_then_raise():
    handler = Recorder(embed_responses=[httpx2.Response(503)] * 5)
    client = make_client(handler, max_attempts=3)
    with pytest.raises(EmbeddingServiceError, match="after 3 attempts"):
        client.embed(["a"], DOCUMENT)
    assert len(handler.embed_calls) == 3


def test_a_client_error_is_not_retried():
    # A 422 means the request itself is wrong; repeating it changes nothing.
    handler = Recorder(embed_responses=[httpx2.Response(422)] * 3)
    client = make_client(handler, max_attempts=3)
    with pytest.raises(EmbeddingServiceError):
        client.embed(["a"], DOCUMENT)
    assert len(handler.embed_calls) == 1


def test_every_retry_is_logged(client_logs):
    handler = Recorder(embed_responses=[httpx2.Response(503)])
    client = make_client(handler)
    client.embed(["a"], DOCUMENT)
    warnings = [r for r in client_logs.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "retrying" in warnings[0].getMessage()


def test_request_text_is_never_logged(client_logs):
    secret = "my salary is 1400000 and my PAN is ABCDE1234F"
    handler = Recorder(embed_responses=[httpx2.Response(503)])
    client = make_client(handler)
    client.embed([secret], DOCUMENT)
    # Proves the capture is live, so the absence assertions below cannot pass
    # by capturing nothing at all.
    assert "pinned to" in client_logs.text
    assert "1400000" not in client_logs.text
    assert "ABCDE1234F" not in client_logs.text


# --- kind (ADR-069) -----------------------------------------------------------


def test_the_requested_kind_is_sent_on_the_wire():
    handler = Recorder()
    client = make_client(handler)
    client.embed(["a"], QUERY)
    assert handler.embed_bodies == [{"texts": ["a"], "kind": "query"}]


def test_every_batch_of_one_call_carries_the_same_kind():
    handler = Recorder()
    client = make_client(handler, batch_size=2)
    client.embed(["a", "b", "c"], DOCUMENT)
    assert [body["kind"] for body in handler.embed_bodies] == ["document", "document"]


def test_a_service_applying_the_other_kind_raises():
    handler = Recorder(
        embed_responses=[
            httpx2.Response(
                200,
                json={
                    "embeddings": [[1.0, 0.0, 0.0, 0.0]],
                    "model": SERVED.model_dump(),
                    "kind": "document",
                },
            )
        ]
    )
    client = make_client(handler)
    with pytest.raises(EmbedKindError, match="kind=query"):
        client.embed(["a"], QUERY)


def test_a_kind_mismatch_is_not_retried():
    # The same service applies the same recipe on a repeat call, and the wrong
    # vector looks entirely well-formed.
    handler = Recorder(
        embed_responses=[
            httpx2.Response(
                200,
                json={
                    "embeddings": [[1.0, 0.0, 0.0, 0.0]],
                    "model": SERVED.model_dump(),
                    "kind": "document",
                },
            )
        ]
        * 3
    )
    client = make_client(handler, max_attempts=3)
    with pytest.raises(EmbedKindError):
        client.embed(["a"], QUERY)
    assert len(handler.embed_calls) == 1


def test_kind_is_required():
    client = make_client(Recorder())
    with pytest.raises(TypeError):
        client.embed(["a"])


# --- lifecycle and end-to-end -------------------------------------------------


def test_close_is_idempotent_via_the_context_manager():
    with make_client(Recorder()) as client:
        assert client.model == SERVED
    client.close()


def test_exactly_one_of_base_url_or_http_client_is_required():
    with pytest.raises(ValueError, match="exactly one"):
        EmbeddingClient()
    with pytest.raises(ValueError, match="exactly one"):
        EmbeddingClient(BASE_URL, http_client=httpx2.Client())


def test_a_borrowed_client_is_not_closed():
    handler = Recorder()
    http_client = httpx2.Client(
        base_url=BASE_URL, transport=httpx2.MockTransport(handler)
    )
    with EmbeddingClient(http_client=http_client, backoff_base=0.0):
        pass
    assert not http_client.is_closed


def test_against_the_real_service():
    """No mock: the client speaks to the actual ASGI app, in-process.

    The stub backend carries no semantics, so this asserts the wire contract
    and the identity pin, never a similarity. `TestClient` is used purely as a
    synchronous transport onto the app — starlette's ASGI transport is
    async-only and cannot drive a sync client.
    """
    app = create_app(StubEmbedder())
    with TestClient(app) as http_client:
        client = EmbeddingClient(
            http_client=http_client, batch_size=2, backoff_base=0.0
        )
        assert client.model.model_id == STUB_MODEL_ID
        assert client.model.dim == STUB_DIM
        vectors = client.embed(["alpha", "beta", "gamma"], DOCUMENT)
        assert len(vectors) == 3
        assert {len(v) for v in vectors} == {STUB_DIM}
        # Determinism, which the stub does guarantee: same text, same vector.
        assert client.embed(["alpha"], DOCUMENT)[0] == vectors[0]
        # And the kind travels the whole path — client, wire, backend, echo.
        assert client.embed(["alpha"], QUERY)[0] != vectors[0]
