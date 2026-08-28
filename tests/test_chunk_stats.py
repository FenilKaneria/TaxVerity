import pytest

from taxverity.chunking.models import Chunk
from taxverity.chunking.stats import (
    CONTEXT_TOKENS,
    LARGE_CHUNK_TOKENS,
    SAFETY_FACTOR,
    SMALL_CHUNK_TOKENS,
    Distribution,
    estimate_tokens,
    is_omitted_only,
    outlier_reason,
    percentile,
    summarise,
)
from taxverity.corpus.nodes import NodeType

VERSION = "a" * 64


def chunk(path, text, **fields):
    fields.setdefault("node_type", NodeType.SECTION)
    fields.setdefault("doc_id", "d")
    fields.setdefault("page_start", 0)
    fields.setdefault("page_end", 0)
    fields.setdefault("section_number", path.split("(")[0])
    fields.setdefault("parent_id", None)
    return Chunk.create(VERSION, path, text, **fields)


# --- estimate_tokens ---------------------------------------------------------


def test_words_and_punctuation_each_count_once():
    assert estimate_tokens("section 80C, and (a) more.") == 9


def test_whitespace_is_not_a_token():
    assert estimate_tokens("  a\n\n b \t") == 2


def test_an_empty_string_has_no_tokens():
    assert estimate_tokens("") == 0


def test_the_estimate_is_a_floor_not_a_ceiling():
    """A subword tokenizer splits long words further, never fewer -- which is
    why the context check multiplies by SAFETY_FACTOR rather than trusting
    this number directly."""
    assert estimate_tokens("incomprehensibility") == 1


# --- percentile and Distribution ---------------------------------------------


def test_a_percentile_is_a_value_some_element_actually_has():
    values = [1, 5, 9, 40]
    assert percentile(values, 0.5) in values


def test_the_top_percentile_never_runs_off_the_end():
    assert percentile([1, 2, 3], 1.0) == 3


def test_an_empty_percentile_is_zero_rather_than_an_error():
    assert percentile([], 0.5) == 0


def test_a_distribution_sorts_what_it_is_given():
    dist = Distribution.of([9, 1, 5])
    assert (dist.minimum, dist.maximum, dist.total, dist.count) == (1, 9, 15, 3)


def test_an_empty_distribution_is_all_zeroes():
    dist = Distribution.of([])
    assert (dist.count, dist.minimum, dist.p50, dist.maximum) == (0, 0, 0, 0)


# --- omission marks ----------------------------------------------------------


@pytest.mark.parametrize("text", ["(m) \n[***]", "(m) [***]", "[***]", "12. [***]"])
def test_a_clause_that_is_only_an_omission_mark_is_recognised(text):
    assert is_omitted_only(text)


@pytest.mark.parametrize("text", ["(m) income [***] from house property", "(m) wages;"])
def test_a_clause_with_real_text_beside_the_mark_is_not(text):
    assert not is_omitted_only(text)


# --- outlier attribution -----------------------------------------------------


def test_an_untrusted_chunk_is_named_as_untrusted_before_anything_else():
    assert "untrusted" in outlier_reason(chunk("2", "text"), untrusted={"2"})


def test_a_trusted_root_is_named_as_a_root():
    assert "root" in outlier_reason(chunk("9", "text"), untrusted=())


def test_a_trusted_intermediate_node_is_named_separately_from_a_root():
    child = chunk(
        "9(1)",
        "text",
        node_type=NodeType.SUBSECTION,
        parent_id="b" * 16,
    )
    assert outlier_reason(child, untrusted=()) != outlier_reason(chunk("9", "t"), ())


# --- summarise ---------------------------------------------------------------


def sample():
    root = chunk("9", "9. A section.\n(1) A sub-section about tax.\n(2) [***]")
    return [
        root,
        chunk(
            "9(1)",
            "(1) A sub-section about tax.",
            node_type=NodeType.SUBSECTION,
            parent_id=root.chunk_id,
            defined_terms=("tax",),
            outgoing_refs=("80C",),
        ),
        chunk(
            "9(2)",
            "(2) [***]",
            node_type=NodeType.SUBSECTION,
            parent_id=root.chunk_id,
        ),
    ]


def test_counts_and_grouping():
    stats = summarise(sample(), glossary_size=109)
    assert (stats.chunks, stats.roots) == (3, 1)
    assert dict(stats.by_type) == {NodeType.SECTION: 1, NodeType.SUBSECTION: 2}
    assert dict(stats.by_depth) == {0: 1, 1: 2}
    assert stats.glossary_terms == 109


def test_term_and_reference_coverage_is_counted_both_ways():
    stats = summarise(sample())
    assert (stats.chunks_without_terms, stats.chunks_with_refs) == (2, 1)
    assert stats.term_frequency == (("tax", 1),)
    assert stats.refs_per_chunk.maximum == 1


def test_the_omission_only_chunk_is_found():
    assert summarise(sample()).omitted_only == 1


def test_small_is_measured_over_the_body_not_the_breadcrumb():
    """embed_text() prepends the citation label, which would push a genuinely
    tiny clause over the threshold on provenance alone."""
    tiny = chunk(
        "Schedule XII(A8)",
        "8. Gold.",
        node_type=NodeType.SCHEDULE_PARAGRAPH,
        schedule_number="XII",
        section_number=None,
        parent_id="b" * 16,
        root_title="Minerals and ores for the purposes of section 44",
    )
    assert estimate_tokens(tiny.embed_text()) >= SMALL_CHUNK_TOKENS
    assert summarise([tiny]).small == 1


def test_tokens_are_measured_over_embed_text_because_that_is_what_is_embedded():
    only = sample()[1]
    stats = summarise([only])
    assert stats.tokens.maximum == estimate_tokens(only.embed_text())
    assert stats.tokens.maximum > estimate_tokens(only.text)


def test_a_chunk_over_the_threshold_is_named_in_full():
    big = chunk("9", "word " * (LARGE_CHUNK_TOKENS + 1))
    stats = summarise([big], untrusted={"9"})
    assert [outlier.node_path for outlier in stats.large] == ["9"]
    assert stats.large[0].characters == len(big.text)
    assert "untrusted" in stats.large[0].reason


def test_outliers_are_ordered_largest_first():
    small = chunk("9", "word " * (LARGE_CHUNK_TOKENS + 1))
    large = chunk("10", "word " * (LARGE_CHUNK_TOKENS * 2))
    tokens = [o.tokens for o in summarise([small, large]).large]
    assert tokens == sorted(tokens, reverse=True)


def test_the_context_check_applies_the_safety_factor():
    """A chunk under the raw context but over it once the proxy is doubled must
    be reported -- that is the whole point of the factor."""
    over = chunk("9", "word " * (CONTEXT_TOKENS // SAFETY_FACTOR + 10))
    stats = summarise([over])
    assert stats.over_context == 0
    assert stats.over_context_with_safety == 1
    assert not stats.fits_context


def test_a_corpus_that_fits_reports_that_it_fits():
    assert summarise(sample()).fits_context


def test_token_count_is_left_unset_on_every_chunk():
    """ADR-057: the proxy is reported, never persisted. Phase 4 fills this
    field with the chosen model's own tokenizer."""
    for chunk_ in sample():
        assert chunk_.token_count is None
