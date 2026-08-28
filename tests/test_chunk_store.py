import hashlib
import json

import pytest

from conftest import CHUNK_TEST_VERSION
from taxverity.chunking.models import Chunk
from taxverity.chunking.store import (
    CHUNK_MANIFEST_FILENAME,
    CHUNK_STAGE_VERSIONS,
    CHUNKS_FILENAME,
    StaleChunkStoreError,
    build_chunk_manifest,
    load_chunks,
    read_chunk_manifest,
    read_chunks_jsonl,
    to_json_line,
    write_chunk_manifest,
    write_chunks_jsonl,
)
from taxverity.corpus.nodes import NodeType

VERSION = "a" * 64
OTHER_VERSION = "b" * 64


def chunk(path, text, **fields):
    fields.setdefault("node_type", NodeType.SECTION)
    fields.setdefault("doc_id", "income-tax-act-2025")
    fields.setdefault("page_start", 0)
    fields.setdefault("page_end", 0)
    fields.setdefault("section_number", path.split("(")[0])
    fields.setdefault("parent_id", None)
    return Chunk.create(VERSION, path, text, **fields)


@pytest.fixture
def two_chunks():
    root = chunk("10", "10. Salary.\n(1) Anything paid.")
    child = chunk(
        "10(1)",
        "(1) Anything paid.",
        node_type=NodeType.SUBSECTION,
        parent_id=root.chunk_id,
        char_start=12,
        char_end=30,
        outgoing_refs=("11(2)",),
        defined_terms=("salary",),
    )
    return [root, child]


@pytest.fixture
def store(tmp_path, two_chunks):
    count, digest = write_chunks_jsonl(two_chunks, tmp_path / CHUNKS_FILENAME)
    write_chunk_manifest(
        build_chunk_manifest(two_chunks, VERSION, digest),
        tmp_path / CHUNK_MANIFEST_FILENAME,
    )
    assert count == 2
    return tmp_path


# --- serialisation -----------------------------------------------------------


def test_a_chunk_round_trips_through_a_json_line(two_chunks):
    for original in two_chunks:
        assert Chunk.model_validate_json(to_json_line(original)) == original


def test_the_line_is_canonical(two_chunks):
    line = to_json_line(two_chunks[0])
    assert list(json.loads(line)) == sorted(json.loads(line))
    assert ", " not in line and '": ' not in line


def test_non_ascii_statutory_text_survives_verbatim(tmp_path):
    original = chunk("10", "10. An em dash — and a rupee ₹1,00,000.")
    write_chunks_jsonl([original], tmp_path / CHUNKS_FILENAME)
    (restored,) = read_chunks_jsonl(tmp_path / CHUNKS_FILENAME)
    assert restored.text == original.text


def test_writing_twice_is_byte_identical(tmp_path, two_chunks):
    first, digest_one = write_chunks_jsonl(two_chunks, tmp_path / "one.jsonl")
    second, digest_two = write_chunks_jsonl(two_chunks, tmp_path / "two.jsonl")
    assert (first, digest_one) == (second, digest_two)
    assert (tmp_path / "one.jsonl").read_bytes() == (tmp_path / "two.jsonl").read_bytes()


def test_lines_are_lf_terminated_on_every_platform(tmp_path, two_chunks):
    write_chunks_jsonl(two_chunks, tmp_path / CHUNKS_FILENAME)
    assert b"\r\n" not in (tmp_path / CHUNKS_FILENAME).read_bytes()


def test_the_digest_is_over_the_bytes_actually_written(tmp_path, two_chunks):
    _, digest = write_chunks_jsonl(two_chunks, tmp_path / CHUNKS_FILENAME)
    assert digest == hashlib.sha256((tmp_path / CHUNKS_FILENAME).read_bytes()).hexdigest()


def test_blank_lines_are_skipped(tmp_path, two_chunks):
    path = tmp_path / CHUNKS_FILENAME
    write_chunks_jsonl(two_chunks, path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert len(list(read_chunks_jsonl(path))) == 2


# --- manifest ----------------------------------------------------------------


def test_the_manifest_counts_chunks_and_roots(two_chunks):
    manifest = build_chunk_manifest(two_chunks, VERSION, "d" * 64)
    assert (manifest.chunk_count, manifest.root_count) == (2, 1)
    assert manifest.corpus_version == VERSION
    assert manifest.stage_versions == CHUNK_STAGE_VERSIONS


def test_the_manifest_round_trips(tmp_path, two_chunks):
    manifest = build_chunk_manifest(two_chunks, VERSION, "d" * 64)
    write_chunk_manifest(manifest, tmp_path / CHUNK_MANIFEST_FILENAME)
    assert read_chunk_manifest(tmp_path / CHUNK_MANIFEST_FILENAME) == manifest


def test_the_manifest_does_not_carry_chunk_versions_into_corpus_version(two_chunks):
    """chunk ids derive from corpus_version, so it must not derive from them."""
    manifest = build_chunk_manifest(two_chunks, VERSION, "d" * 64)
    assert manifest.corpus_version == VERSION
    assert "chunker" not in manifest.corpus_version


# --- load_chunks -------------------------------------------------------------


def test_a_matching_store_loads(store, two_chunks):
    chunks, manifest = load_chunks(store, corpus_version=VERSION)
    assert list(chunks) == two_chunks
    assert manifest.chunk_count == 2


def test_it_loads_without_a_declared_corpus_version(store):
    chunks, _ = load_chunks(store)
    assert len(chunks) == 2


def test_a_store_built_for_another_corpus_is_refused(store):
    with pytest.raises(StaleChunkStoreError, match="mislabelled"):
        load_chunks(store, corpus_version=OTHER_VERSION)


def test_a_store_built_by_another_chunker_version_is_refused(store):
    path = store / CHUNK_MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stage_versions"]["chunker"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StaleChunkStoreError, match="chunking stages"):
        load_chunks(store, corpus_version=VERSION)


def test_a_truncated_store_is_refused(store):
    path = store / CHUNKS_FILENAME
    kept = path.read_text(encoding="utf-8").splitlines(keepends=True)[:1]
    path.write_text("".join(kept), encoding="utf-8", newline="")
    with pytest.raises(StaleChunkStoreError, match="truncated or edited"):
        load_chunks(store, corpus_version=VERSION)


def test_an_edited_store_is_refused(store):
    path = store / CHUNKS_FILENAME
    path.write_text(
        path.read_text(encoding="utf-8").replace("Anything paid", "Nothing paid"),
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(StaleChunkStoreError, match="truncated or edited"):
        load_chunks(store, corpus_version=VERSION)


def test_a_missing_store_names_the_script_that_builds_it(tmp_path):
    with pytest.raises(StaleChunkStoreError, match="build_chunks.py"):
        load_chunks(tmp_path)


def test_a_store_with_no_manifest_is_refused(tmp_path, two_chunks):
    write_chunks_jsonl(two_chunks, tmp_path / CHUNKS_FILENAME)
    with pytest.raises(StaleChunkStoreError, match=CHUNK_MANIFEST_FILENAME):
        load_chunks(tmp_path)


def test_a_tampered_text_cannot_keep_its_id(tmp_path, two_chunks):
    """Chunk validates its own id, so the store cannot smuggle altered text."""
    path = tmp_path / CHUNKS_FILENAME
    write_chunks_jsonl(two_chunks, path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("Anything paid", "Nothing paid"),
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(ValueError, match="chunk_id"):
        list(read_chunks_jsonl(path))


# --- the real corpus ---------------------------------------------------------


def test_the_whole_corpus_round_trips_through_the_store(tmp_path, chunks):
    count, digest = write_chunks_jsonl(chunks, tmp_path / CHUNKS_FILENAME)
    write_chunk_manifest(
        build_chunk_manifest(chunks, CHUNK_TEST_VERSION, digest),
        tmp_path / CHUNK_MANIFEST_FILENAME,
    )
    loaded, manifest = load_chunks(tmp_path, corpus_version=CHUNK_TEST_VERSION)
    assert count == len(chunks) == 7561
    assert manifest.root_count == 553
    assert loaded == tuple(chunks)


def test_the_corpus_store_is_byte_identical_across_two_writes(tmp_path, chunks):
    first, digest_one = write_chunks_jsonl(chunks, tmp_path / "one.jsonl")
    second, digest_two = write_chunks_jsonl(chunks, tmp_path / "two.jsonl")
    assert (first, digest_one) == (second, digest_two)
