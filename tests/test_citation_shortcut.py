"""Step 3.5 — a citation in a query is an address: surface forms, false
positives, and the direct lookup they route to."""

from __future__ import annotations

import pytest

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodeType
from taxverity.evals.gold import QuerySlice
from taxverity.retrieval.base import Retriever, as_ranked_citations
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.citations import (
    CitationRetriever,
    ShortcutRetriever,
    extract_query_citations,
)

CORPUS_VERSION = "v" * 64


def chunk(node_path, text, title=None):
    schedule = node_path.startswith("Schedule")
    return Chunk.create(
        CORPUS_VERSION,
        node_path,
        text,
        parent_id=None if "(" not in node_path else "0" * 16,
        doc_id="income-tax-act-2025",
        node_type=NodeType.SUBSECTION if "(" in node_path else NodeType.SECTION,
        section_number=None if schedule else node_path.split("(")[0],
        schedule_number=node_path.split()[1].split("(")[0] if schedule else None,
        root_title=title,
        page_start=1,
        page_end=1,
    )


CORPUS = [
    chunk("22", "Deductions from income from house property.", "House property"),
    chunk("22(2)", "Interest on borrowed capital, up to two lakh rupees."),
    chunk("354A", "This section applies to a specified person."),
    chunk("206", "An untrusted section, chunked whole and never split."),
    chunk("Schedule XV", "Sums qualifying for deduction."),
]


@pytest.fixture(scope="module")
def retriever():
    return CitationRetriever(CORPUS)


def paths(results):
    return [result.chunk.node_path for result in results]


# --- surface forms -------------------------------------------------------


def test_a_bare_section_number():
    assert extract_query_citations("What does section 22 allow?") == ["22"]


def test_a_bracket_chain():
    assert extract_query_citations("Under section 22(2)(a), what is the cap?") == [
        "22(2)(a)"
    ]


def test_a_letter_suffixed_number():
    """354A and 354 are different sections; the suffix must survive."""
    assert extract_query_citations("section 354A") == ["354A"]


def test_a_lowercase_query():
    assert extract_query_citations("section 6(5) says what?") == ["6(5)"]


def test_the_prefix_style():
    assert extract_query_citations("clause (b) of section 80") == ["80(b)"]


def test_a_list_of_sections():
    assert extract_query_citations("sections 22, 23 and 24") == ["22", "23", "24"]


def test_a_list_inheriting_the_section_number():
    assert extract_query_citations("section 70(1)(a), (c) and (d)") == [
        "70(1)(a)",
        "70(1)(c)",
        "70(1)(d)",
    ]


def test_a_numeric_range():
    assert extract_query_citations("sections 28 to 31") == ["28", "29", "30", "31"]


def test_a_schedule():
    assert extract_query_citations("What is in Schedule XV?") == ["Schedule XV"]


def test_a_schedule_paragraph():
    assert extract_query_citations("paragraph 8 of Part A of Schedule XI") == [
        "Schedule XI(A8)"
    ]


@pytest.mark.parametrize(
    "query",
    [
        "what does s. 22 cover",
        "what does s.22 cover",
        "what does sec 22 cover",
        "what does sec. 22 cover",
        "what does u/s 22 cover",
    ],
)
def test_user_abbreviations(query):
    """The Act never abbreviates; users always do."""
    assert extract_query_citations(query) == ["22"]


def test_the_own_act_named_in_full():
    """The corpus scanner routes this outside the corpus, correctly for statute
    prose quoting the 1961 Act. In a query it means the opposite."""
    assert extract_query_citations("section 22 of the Income-tax Act, 2025") == ["22"]
    assert extract_query_citations("section 22 of the Income tax Act") == ["22"]


def test_duplicates_collapse_in_order_of_appearance():
    assert extract_query_citations("section 24, then section 22, then section 24") == [
        "24",
        "22",
    ]


# --- false positives -----------------------------------------------------


def test_prose_with_no_reference_yields_nothing():
    assert extract_query_citations("How many days make me a resident?") == []


def test_a_bare_number_is_not_a_citation():
    assert extract_query_citations("I earned 22 lakh rupees in 2025") == []


def test_another_act_is_not_looked_up():
    assert extract_query_citations("section 8 of the Companies Act, 2013") == []
    assert (
        extract_query_citations("section 16 of the Central Goods and Services Tax Act")
        == []
    )


def test_the_1961_act_is_not_this_corpus():
    assert extract_query_citations("section 80C of the Income-tax Act, 1961") == []


def test_an_abbreviation_without_a_number_is_not_expanded():
    assert extract_query_citations("Ms. Sharma sec of the trust") == []


def test_a_percentage_is_not_a_citation():
    assert extract_query_citations("Is the 30% standard deduction still allowed?") == []


# --- lookup --------------------------------------------------------------


def test_the_retriever_satisfies_the_protocol(retriever):
    assert isinstance(retriever, Retriever)


def test_an_exact_citation_returns_its_chunk(retriever):
    assert paths(retriever.search("section 22(2)", 5)) == ["22(2)"]


def test_an_unretrievable_depth_falls_back_to_its_nearest_ancestor(retriever):
    """ADR-056 prunes an untrusted subtree to its root, so 206(1)(m) names no
    chunk. The text containing it beats returning nothing."""
    results = retriever.search("section 206(1)(m)", 5)
    assert paths(results) == ["206"]
    assert results[0].score < 1.0


def test_an_exact_hit_outranks_an_ancestor_fallback(retriever):
    assert paths(retriever.search("sections 22(2) and 206(1)(m)", 5)) == [
        "22(2)",
        "206",
    ]


def test_a_citation_outside_the_corpus_returns_nothing(retriever):
    assert retriever.search("section 999", 5) == []


def test_a_query_with_no_citation_returns_nothing(retriever):
    """A shortcut, not a search: no lexical fallback lives here."""
    assert retriever.search("interest on borrowed capital", 5) == []


def test_the_same_chunk_is_not_returned_twice(retriever):
    """22(2) and 22(2)(z) both resolve to the 22(2) chunk."""
    assert paths(retriever.search("section 22(2) and section 22(2)(z)", 5)) == ["22(2)"]


def test_results_are_ranked_and_unique(retriever):
    results = retriever.search("sections 22, 22(2) and 206(1)(m)", 5)
    assert as_ranked_citations(results) == paths(results)


def test_k_bounds_the_results(retriever):
    assert len(retriever.search("sections 22, 22(2) and 354A", 2)) == 2


def test_k_must_be_positive(retriever):
    with pytest.raises(ValueError, match="at least 1"):
        retriever.search("section 22", 0)


# --- composition ---------------------------------------------------------


@pytest.fixture(scope="module")
def combined():
    return ShortcutRetriever(CitationRetriever(CORPUS), BM25Retriever(CORPUS))


def test_the_composite_satisfies_the_protocol(combined):
    assert isinstance(combined, Retriever)


def test_a_cited_chunk_is_injected_ahead_of_the_lexical_ranking(combined):
    assert paths(combined.search("section 354A specified person", 5))[0] == "354A"


def test_the_composite_falls_through_to_the_primary(combined):
    assert paths(combined.search("interest on borrowed capital", 3))[0] == "22(2)"


def test_the_composite_returns_no_chunk_twice(combined):
    results = combined.search("section 22 house property", 5)
    assert as_ranked_citations(results) == paths(results)
    assert len(paths(results)) == len(set(paths(results)))


def test_k_bounds_the_composite(combined):
    assert len(combined.search("section 22 deduction", 2)) == 2


# --- the real corpus -----------------------------------------------------


def test_every_gold_citation_query_resolves_to_a_chunk(gold, chunks):
    """The citation slice is exactly the slice this shortcut exists for."""
    retriever = CitationRetriever(chunks)
    cited = [q for q in gold if q.slice is QuerySlice.CITATION]
    assert cited
    for query in cited:
        assert paths(retriever.search(query.question, 10)), query.query_id


def test_no_negative_query_triggers_the_shortcut(gold, chunks):
    """A false positive here injects statutory text into a question the corpus
    cannot answer, which is worse than a miss."""
    retriever = CitationRetriever(chunks)
    for query in (q for q in gold if q.slice is QuerySlice.NEGATIVE):
        assert retriever.search(query.question, 10) == [], query.query_id
