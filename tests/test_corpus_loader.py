import json
from pathlib import Path

import pytest

from taxverity.config import Settings
from taxverity.corpus.loader import (
    BOLD_FLAG,
    ITALIC_FLAG,
    build_manifest,
    extract_pages,
    hash_file,
    is_furniture,
    normalise,
    page_artifact_from_dict,
    read_pages_jsonl,
    span_from_dict,
    to_json_line,
    write_pages_jsonl,
)
from taxverity.corpus.models import PageArtifact

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pages"
GOLDEN_PAGES = (0, 273, 333, 554, 665)

EXPECTED_PAGE_COUNT = 666
MIN_NON_EMPTY_RATIO = 0.99


def raw_span(
    text, *, font="LiberationSerif", size=8.486839294433594, flags=0, bbox=None
):
    return {
        "text": text,
        "font": font,
        "size": size,
        "flags": flags,
        "bbox": bbox or (1.23456, 2.34567, 3.45678, 4.56789),
    }


def corpus_pdf_or_skip() -> Path:
    settings = Settings(_env_file=None)
    if not settings.corpus_pdf.is_file():
        pytest.skip("corpus PDF not present in this working copy")
    return settings.corpus_pdf


# --- normalisation and furniture: pure, no corpus needed ---------------------


def test_normalise_strips_the_soft_hyphen_and_nbsp():
    assert normalise("473.\xad") == "473."
    assert normalise("(i)\xa0the valuation") == "(i) the valuation"


def test_footer_strings_are_furniture():
    assert is_furniture("Income Tax Department")
    assert is_furniture("Ministry of Finance, Government of India")
    assert is_furniture(
        "Downloaded/Printed on 5/22/26, 4:43 PM from www.incometaxindia.gov.in"
    )


def test_statutory_content_is_not_furniture():
    # NotoSans renders both the footer and some clause markers, so a font-based
    # filter would delete real content. These must survive a string-based one.
    for text in ("(", ")", "a", "b", "473.", "Income Tax Department of India Ltd"):
        assert not is_furniture(text), text


# --- span and page construction ---------------------------------------------


def test_span_flags_become_booleans():
    plain = span_from_dict(raw_span("x"))
    assert (plain.bold, plain.italic) == (False, False)
    bold = span_from_dict(raw_span("x", flags=BOLD_FLAG))
    assert (bold.bold, bold.italic) == (True, False)
    both = span_from_dict(raw_span("x", flags=BOLD_FLAG | ITALIC_FLAG))
    assert (both.bold, both.italic) == (True, True)


def test_span_numbers_are_rounded_for_stable_serialisation():
    span = span_from_dict(raw_span("x"))
    assert span.size == 8.49
    assert span.bbox == (1.23, 2.35, 3.46, 4.57)


def test_span_text_is_preserved_verbatim():
    span = span_from_dict(raw_span("473.\xad"))
    assert span.text == "473.\xad"


def test_blank_spans_are_dropped():
    blocks = [{"lines": [{"spans": [raw_span("real"), raw_span("   "), raw_span("")]}]}]
    artifact = page_artifact_from_dict(7, "real", blocks)
    assert [span.text for span in artifact.spans] == ["real"]
    assert artifact.page == 7


def test_blocks_without_lines_are_tolerated():
    artifact = page_artifact_from_dict(0, "", [{"type": 1, "image": b""}])
    assert artifact.spans == ()


def test_body_spans_excludes_furniture():
    blocks = [
        {
            "lines": [
                {
                    "spans": [
                        raw_span("statutory text"),
                        raw_span("Income Tax Department"),
                    ]
                }
            ]
        }
    ]
    artifact = page_artifact_from_dict(0, "x", blocks)
    assert len(artifact.spans) == 2
    assert [span.text for span in artifact.body_spans()] == ["statutory text"]


# --- serialisation determinism ----------------------------------------------


def test_json_line_is_stable_and_key_sorted():
    artifact = page_artifact_from_dict(
        1, "t", [{"lines": [{"spans": [raw_span("a")]}]}]
    )
    line = to_json_line(artifact)
    assert to_json_line(artifact) == line
    assert list(json.loads(line)) == sorted(json.loads(line))


def test_json_line_keeps_unicode_unescaped():
    artifact = page_artifact_from_dict(1, "₹ 100", [])
    assert "₹" in to_json_line(artifact)
    assert "\\u20b9" not in to_json_line(artifact)


def test_jsonl_round_trip_preserves_artifacts(tmp_path):
    artifacts = [
        page_artifact_from_dict(
            i, f"page {i}\xa0text", [{"lines": [{"spans": [raw_span("s")]}]}]
        )
        for i in range(3)
    ]
    destination = tmp_path / "pages.jsonl"
    count, digest = write_pages_jsonl(artifacts, destination)
    assert count == 3
    assert list(read_pages_jsonl(destination)) == artifacts
    assert write_pages_jsonl(artifacts, tmp_path / "again.jsonl")[1] == digest


def test_written_artifact_uses_lf_endings(tmp_path):
    destination = tmp_path / "pages.jsonl"
    write_pages_jsonl([page_artifact_from_dict(0, "x", [])], destination)
    assert b"\r\n" not in destination.read_bytes()


def test_hash_file_matches_a_known_digest(tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert hash_file(target) == expected


# --- golden fixtures: committed, so these run without the corpus PDF ---------


@pytest.mark.parametrize("page_number", GOLDEN_PAGES)
def test_golden_fixture_parses(page_number):
    path = FIXTURE_DIR / f"page_{page_number:04d}.json"
    artifact = PageArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    assert artifact.page == page_number
    assert artifact.spans


@pytest.mark.parametrize("page_number", GOLDEN_PAGES)
def test_golden_fixture_has_exactly_the_two_footer_spans(page_number):
    path = FIXTURE_DIR / f"page_{page_number:04d}.json"
    artifact = PageArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    assert sum(span.furniture for span in artifact.spans) == 2


def test_golden_fixture_preserves_the_lone_soft_hyphen():
    path = FIXTURE_DIR / "page_0554.json"
    artifact = PageArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    assert "\xad" in artifact.text
    assert any("\xad" in span.text for span in artifact.spans)


# --- against the real corpus: skipped when the PDF is absent -----------------


@pytest.fixture(scope="session")
def extracted_pages():
    """One walk of the 666-page corpus, shared by every test below that needs it."""
    return list(extract_pages(corpus_pdf_or_skip()))


@pytest.mark.parametrize("page_number", GOLDEN_PAGES)
def test_extraction_still_matches_the_golden_fixture(page_number, extracted_pages):
    artifact = next(a for a in extracted_pages if a.page == page_number)
    expected = (FIXTURE_DIR / f"page_{page_number:04d}.json").read_text(
        encoding="utf-8"
    )
    assert to_json_line(artifact) + "\n" == expected


def test_corpus_extracts_the_expected_page_count_and_is_almost_all_text(
    extracted_pages,
):
    assert len(extracted_pages) == EXPECTED_PAGE_COUNT
    non_empty = sum(1 for a in extracted_pages if a.char_count > 50)
    assert non_empty / len(extracted_pages) >= MIN_NON_EMPTY_RATIO


def test_re_extraction_is_byte_identical(tmp_path):
    pdf_path = corpus_pdf_or_skip()
    first = write_pages_jsonl(extract_pages(pdf_path), tmp_path / "a.jsonl")
    second = write_pages_jsonl(extract_pages(pdf_path), tmp_path / "b.jsonl")
    assert first == second
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_manifest_pins_the_source_and_the_artifact(tmp_path, extracted_pages):
    pdf_path = corpus_pdf_or_skip()
    count, digest = write_pages_jsonl(extracted_pages, tmp_path / "pages.jsonl")
    manifest = build_manifest(pdf_path, count, digest, {"extract": 1})
    assert manifest.page_count == EXPECTED_PAGE_COUNT
    assert manifest.artifact_sha256 == digest
    assert manifest.source_sha256 == hash_file(pdf_path)
    assert manifest.source_bytes == pdf_path.stat().st_size
    assert len(manifest.source_sha256) == 64
