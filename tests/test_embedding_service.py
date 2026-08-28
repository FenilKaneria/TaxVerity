import logging
import math
import threading
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from taxverity.embedding.backends import (
    STUB_DIM,
    STUB_MODEL_ID,
    STUB_REVISION,
    STUB_RUNTIME,
    Embedder,
    ModelInfo,
    StubEmbedder,
)
from taxverity.embedding.service import (
    MAX_BATCH,
    WARMUP_TEXT,
    Readiness,
    create_app,
)

READY_TIMEOUT = 5.0

SERVICE_LOGGER = "taxverity.embedding.service"


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        formatter = logging.Formatter("%(message)s")
        return "\n".join(formatter.format(r) for r in self.records)


@pytest.fixture
def service_logs():
    """Captures on the service's own logger, not via caplog.

    `configure_logging()` sets `propagate = False` on the `taxverity` root and
    clears its handlers, so caplog's root handler sees nothing and any
    assertion made through it is vacuous. Attaching to the child logger sits
    below both effects. It also sits below `RedactingFilter`, which is on the
    configured handler — so what is captured here is what the call site
    actually passed, which is the stronger thing to assert about.
    """
    logger = logging.getLogger(SERVICE_LOGGER)
    handler = CaptureHandler()
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def wait_until(client, predicate, timeout=READY_TIMEOUT):
    """Poll /ready until `predicate` holds. The warm-up runs concurrently with
    startup by design, so its completion is not observable at fixture time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get("/ready")
        if predicate(response):
            return response
        time.sleep(0.005)
    raise AssertionError(f"/ready never satisfied the predicate: {response.json()}")


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        wait_until(test_client, lambda r: r.status_code == 200)
        yield test_client


class FakeEmbedder:
    def __init__(self, dim: int = 3, model_id: str = "fake") -> None:
        self._info = ModelInfo(
            model_id=model_id, dim=dim, revision="abc123", runtime="fake-runtime"
        )
        self.calls: list[list[str]] = []

    def info(self) -> ModelInfo:
        return self._info

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0] + [0.0] * (self._info.dim - 1) for _ in texts]


# --- ModelInfo ---------------------------------------------------------------


def test_model_info_is_frozen():
    info = StubEmbedder().info()
    with pytest.raises(ValidationError):
        info.model_id = "other"


@pytest.mark.parametrize(
    "field, value",
    [("model_id", ""), ("dim", 0), ("dim", -1), ("revision", ""), ("runtime", "")],
)
def test_model_info_rejects_empty_identity(field, value):
    fields = {
        "model_id": "m",
        "dim": 4,
        "revision": "r",
        "runtime": "t",
        field: value,
    }
    with pytest.raises(ValueError):
        ModelInfo(**fields)


# --- StubEmbedder ------------------------------------------------------------


def test_stub_satisfies_the_embedder_protocol():
    assert isinstance(StubEmbedder(), Embedder)


def test_stub_info_matches_its_constants():
    info = StubEmbedder().info()
    assert info.model_id == STUB_MODEL_ID
    assert info.dim == STUB_DIM
    assert info.revision == STUB_REVISION
    assert info.runtime == STUB_RUNTIME


def test_stub_returns_one_vector_per_text_in_order():
    vectors = StubEmbedder().embed(["alpha", "beta", "alpha"])
    assert len(vectors) == 3
    assert all(len(v) == STUB_DIM for v in vectors)
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]


def test_stub_vectors_are_unit_length():
    for vector in StubEmbedder().embed(["alpha", "", "section 80C"]):
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-12)


def test_stub_is_deterministic_across_instances():
    assert StubEmbedder().embed(["section 22(2)"]) == StubEmbedder().embed(
        ["section 22(2)"]
    )


def test_stub_normalises_before_hashing():
    # Step 1.1: the corpus carries soft hyphens and NBSPs that read identically.
    assert StubEmbedder().embed(["sub­section"]) == StubEmbedder().embed(
        ["subsection"]
    )


def test_stub_embeds_an_empty_batch_to_an_empty_list():
    assert StubEmbedder().embed([]) == []


# --- Contract: /embed --------------------------------------------------------


def test_embed_returns_a_vector_per_text():
    embedder = FakeEmbedder(dim=3)
    with TestClient(create_app(embedder)) as client:
        wait_until(client, lambda r: r.status_code == 200)
        response = client.post("/embed", json={"texts": ["a", "b"]})
    assert response.status_code == 200
    body = response.json()
    assert body["embeddings"] == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert embedder.calls == [[WARMUP_TEXT], ["a", "b"]]


def test_embed_response_carries_model_identity(client):
    body = client.post("/embed", json={"texts": ["a"]}).json()
    assert body["model"] == {
        "model_id": STUB_MODEL_ID,
        "dim": STUB_DIM,
        "revision": STUB_REVISION,
        "runtime": STUB_RUNTIME,
    }


def test_embed_vector_width_matches_the_declared_dim(client):
    body = client.post("/embed", json={"texts": ["a", "bb"]}).json()
    assert all(len(vector) == body["model"]["dim"] for vector in body["embeddings"])


def test_embed_rejects_an_empty_batch(client):
    assert client.post("/embed", json={"texts": []}).status_code == 422


def test_embed_rejects_a_batch_over_the_limit(client):
    texts = [str(i) for i in range(MAX_BATCH + 1)]
    assert client.post("/embed", json={"texts": texts}).status_code == 422


def test_embed_accepts_a_batch_at_the_limit(client):
    texts = [str(i) for i in range(MAX_BATCH)]
    assert client.post("/embed", json={"texts": texts}).status_code == 200


def test_embed_rejects_a_missing_body(client):
    assert client.post("/embed", json={}).status_code == 422


def test_embed_rejects_unknown_fields(client):
    response = client.post("/embed", json={"texts": ["a"], "normalize": True})
    assert response.status_code == 422


def test_embed_rejects_a_non_string_text(client):
    assert client.post("/embed", json={"texts": [1]}).status_code == 422


def test_embed_is_deterministic_across_requests(client):
    first = client.post("/embed", json={"texts": ["section 22"]}).json()
    second = client.post("/embed", json={"texts": ["section 22"]}).json()
    assert first["embeddings"] == second["embeddings"]


# --- Contract: /model-info, /health, /ready ----------------------------------


def test_model_info_returns_all_four_identity_fields(client):
    body = client.get("/model-info").json()
    assert set(body) == {"model_id", "dim", "revision", "runtime"}


def test_model_info_reflects_the_injected_backend():
    with TestClient(create_app(FakeEmbedder(dim=7, model_id="other"))) as c:
        body = c.get("/model-info").json()
    assert body == {
        "model_id": "other",
        "dim": 7,
        "revision": "abc123",
        "runtime": "fake-runtime",
    }


def test_model_info_agrees_with_what_embed_reports(client):
    assert client.get("/model-info").json() == (
        client.post("/embed", json={"texts": ["a"]}).json()["model"]
    )


def test_health_is_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_is_true_once_warm(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True, "state": "ready"}


def test_health_and_ready_take_no_body(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_unknown_route_is_404(client):
    assert client.get("/embeddings").status_code == 404


# --- The service does not log request text -----------------------------------


def test_embed_does_not_log_request_text(service_logs):
    secret = "my salary is 1400000 and my PAN is ABCDE1234F"
    with TestClient(create_app()) as client:
        wait_until(client, lambda r: r.status_code == 200)
        client.post("/embed", json={"texts": [secret]})
    # Proves the capture is live, so the two absence assertions below cannot
    # pass by capturing nothing at all.
    assert "embed service starting" in service_logs.text
    assert "1400000" not in service_logs.text
    assert "ABCDE1234F" not in service_logs.text


# --- Step 4.2: warm-up and the readiness gate --------------------------------


class GatedEmbedder:
    """Holds the warm-up call open until released, so `pending` is observable.

    Only the warm-up text blocks: every other call is served normally, which is
    what lets a test assert that `/embed` stays open while `/ready` is closed.
    """

    def __init__(self, dim: int = 3) -> None:
        self._info = ModelInfo(
            model_id="gated", dim=dim, revision="r1", runtime="fake-runtime"
        )
        self.gate = threading.Event()
        self.calls: list[list[str]] = []

    def info(self) -> ModelInfo:
        return self._info

    def embed(self, texts):
        self.calls.append(list(texts))
        if list(texts) == [WARMUP_TEXT]:
            assert self.gate.wait(timeout=READY_TIMEOUT), "warm-up gate never released"
        return [[1.0] + [0.0] * (self._info.dim - 1) for _ in texts]


class FailingEmbedder:
    def __init__(self) -> None:
        self._info = ModelInfo(
            model_id="failing", dim=3, revision="r1", runtime="fake-runtime"
        )

    def info(self) -> ModelInfo:
        return self._info

    def embed(self, texts):
        raise RuntimeError("weights are not loadable")


class LyingEmbedder:
    """Declares one width and returns another — the skew the warm-up catches."""

    def __init__(self) -> None:
        self._info = ModelInfo(
            model_id="lying", dim=3, revision="r1", runtime="fake-runtime"
        )

    def info(self) -> ModelInfo:
        return self._info

    def embed(self, texts):
        return [[0.0, 1.0, 0.0, 0.0] for _ in texts]


def test_ready_is_503_before_warm_up_and_200_after():
    embedder = GatedEmbedder()
    try:
        with TestClient(create_app(embedder)) as client:
            pending = client.get("/ready")
            assert pending.status_code == 503
            assert pending.json() == {"ready": False, "state": "pending"}

            embedder.gate.set()
            warm = wait_until(client, lambda r: r.status_code == 200)
            assert warm.json() == {"ready": True, "state": "ready"}
    finally:
        embedder.gate.set()


def test_warm_up_embeds_the_warm_up_text_exactly_once():
    embedder = GatedEmbedder()
    try:
        with TestClient(create_app(embedder)) as client:
            embedder.gate.set()
            wait_until(client, lambda r: r.status_code == 200)
            client.get("/ready")
            client.get("/ready")
    finally:
        embedder.gate.set()
    assert embedder.calls == [[WARMUP_TEXT]]


def test_startup_does_not_block_on_the_warm_up():
    # The gate is never released before these calls: if the warm-up ran inside
    # startup, the app would not be answering anything yet.
    embedder = GatedEmbedder()
    try:
        with TestClient(create_app(embedder)) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/model-info").status_code == 200
            assert client.get("/ready").status_code == 503
            # Released before leaving: shutdown would otherwise sit behind the
            # worker thread still blocked inside the warm-up call.
            embedder.gate.set()
            wait_until(client, lambda r: r.status_code == 200)
    finally:
        embedder.gate.set()


def test_embed_is_served_while_the_warm_up_is_pending():
    # PENDING gates routing, not authorisation: a caller reaching the container
    # directly merely pays the first-pass cost the warm-up exists to move.
    embedder = GatedEmbedder()
    try:
        with TestClient(create_app(embedder)) as client:
            assert client.get("/ready").status_code == 503
            response = client.post("/embed", json={"texts": ["a"]})
            assert response.status_code == 200
            assert response.json()["embeddings"] == [[1.0, 0.0, 0.0]]
            embedder.gate.set()
            wait_until(client, lambda r: r.status_code == 200)
    finally:
        embedder.gate.set()


def test_ready_is_failed_when_the_warm_up_raises():
    with TestClient(create_app(FailingEmbedder())) as client:
        response = wait_until(client, lambda r: r.json()["state"] == "failed")
    assert response.status_code == 503
    assert response.json() == {"ready": False, "state": "failed"}


def test_ready_is_failed_when_the_backend_width_contradicts_its_declared_dim():
    with TestClient(create_app(LyingEmbedder())) as client:
        response = wait_until(client, lambda r: r.json()["state"] == "failed")
    assert response.status_code == 503


def test_embed_is_503_after_a_failed_warm_up():
    with TestClient(create_app(FailingEmbedder())) as client:
        wait_until(client, lambda r: r.json()["state"] == "failed")
        response = client.post("/embed", json={"texts": ["a"]})
    assert response.status_code == 503
    assert response.json()["detail"] == "embedding backend failed warm-up"


def test_health_stays_200_after_a_failed_warm_up():
    # A task that will never serve must stay diagnosable rather than look dead.
    with TestClient(create_app(FailingEmbedder())) as client:
        wait_until(client, lambda r: r.json()["state"] == "failed")
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/model-info").status_code == 200


def test_a_failed_warm_up_is_logged(service_logs):
    with TestClient(create_app(FailingEmbedder())) as client:
        wait_until(client, lambda r: r.json()["state"] == "failed")
    assert "warm-up failed" in service_logs.text
    assert any(record.exc_info for record in service_logs.records)


def test_readiness_states_are_exactly_three():
    assert [state.value for state in Readiness] == ["pending", "ready", "failed"]
