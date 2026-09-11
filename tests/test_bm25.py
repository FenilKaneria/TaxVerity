"""Step 3.4 — the lexical baseline: tokenisation, scoring and ranking."""

from __future__ import annotations

import pytest

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodeType
from taxverity.evals.gold import QuerySlice
from taxverity.evals.metrics import CitationIndex, score_run
from taxverity.retrieval.base import Retriever, as_ranked_citations
from taxverity.retrieval.bm25 import K1, B, BM25Retriever, tokenize

CORPUS_VERSION = "v" * 64


def chunk(node_path, text, title=None):
    return Chunk.create(
        CORPUS_VERSION,
        node_path,
        text,
        parent_id=None if "(" not in node_path else "0" * 16,
        doc_id="income-tax-act-2025",
        node_type=NodeType.SUBSECTION if "(" in node_path else NodeType.SECTION,
        section_number=node_path.split("(")[0],
        root_title=title,
        page_start=1,
        page_end=1,
    )


CORPUS = [
    chunk(
        "22", "Income from buildings and land is chargeable to tax.", "House property"
    ),
    chunk("23", "The annual value of a property shall be determined as follows."),
    chunk("24", "A deduction of thirty per cent of the annual value is allowed."),
    chunk("354A", "This section applies to a specified person."),
    chunk("354", "This section applies to an unspecified body."),
]


@pytest.fixture(scope="module")
def retriever():
    return BM25Retriever(CORPUS)


def paths(results):
    return [result.chunk.node_path for result in results]


def test_tokenisation_keeps_a_letter_suffixed_number_whole():
    """354A and 354 are different sections. A tokenizer that splits the suffix
    makes them the same query."""
    assert tokenize("section 354A") == ["section", "354a"]
    assert tokenize("80C deduction") == ["80c", "deduction"]


def test_tokenisation_splits_a_citation_into_its_parts():
    assert tokenize("22(2)") == ["22", "2"]


def test_tokenisation_normalises_the_corpus_hyphen_and_nbsp():
    """Step 1.1 measured both in the Act; a raw casefold leaves them in place and
    the token never matches."""
    assert tokenize("sub\xadsection\xa0(1)") == ["subsection", "1"]


def test_the_retriever_satisfies_the_protocol(retriever):
    assert isinstance(retriever, Retriever)


def test_a_known_token_returns_its_chunk(retriever):
    assert paths(retriever.search("annual value", 1)) == ["23"]


def test_a_rare_token_outranks_a_common_one(retriever):
    """idf is the whole point: "deduction" appears once, "section" twice, and the
    query contains both."""
    assert paths(retriever.search("section deduction", 1)) == ["24"]


def test_a_suffixed_section_number_is_not_confused_with_its_base(retriever):
    assert paths(retriever.search("354A", 1)) == ["354A"]
    assert paths(retriever.search("354", 1)) == ["354"]


def test_the_title_is_searchable(retriever):
    """The breadcrumb is indexed, so a chunk is reachable by its own heading even
    when the word is absent from its text."""
    assert "house" not in CORPUS[0].text.casefold()
    assert paths(retriever.search("house property", 1)) == ["22"]


def test_k_bounds_the_results(retriever):
    assert len(retriever.search("the", 2)) == 2


def test_k_must_be_positive(retriever):
    with pytest.raises(ValueError, match="at least 1"):
        retriever.search("annual value", 0)


def test_a_query_matching_nothing_returns_nothing(retriever):
    assert retriever.search("cryptocurrency airdrop", 5) == []


def test_an_empty_query_returns_nothing(retriever):
    assert retriever.search("   ", 5) == []


def test_results_are_ranked_and_unique(retriever):
    results = retriever.search("annual value of the property deduction", 5)
    assert len(results) > 1
    assert as_ranked_citations(results) == paths(results)


def test_a_repeated_query_token_does_not_weight_it_twice(retriever):
    once = retriever.search("deduction annual", 5)
    twice = retriever.search("deduction deduction annual", 5)
    assert [(r.chunk.node_path, r.score) for r in once] == [
        (r.chunk.node_path, r.score) for r in twice
    ]


def test_no_score_is_negative(retriever):
    """The positive-idf variant. Robertson's original subtracts for a term in
    more than half the corpus, and "the" is in four of these five."""
    assert all(result.score > 0 for result in retriever.search("the", 5))


def test_the_ranking_is_reproducible(retriever):
    rebuilt = BM25Retriever(CORPUS)
    query = "annual value of the property"
    assert [(r.chunk.chunk_id, r.score) for r in retriever.search(query, 5)] == [
        (r.chunk.chunk_id, r.score) for r in rebuilt.search(query, 5)
    ]


def test_the_tokenizer_is_injectable():
    """Step 4.4b's ladder swaps tokenizers without forking the retriever. The
    default splits "2,00,000" into three tokens, none of them in this chunk."""
    corpus = [chunk("1", "A limit of 200000 rupees.")]

    def ungrouped(text):
        return tokenize(text.replace(",", ""))

    assert paths(BM25Retriever(corpus, tokenizer=ungrouped).search("2,00,000", 1)) == [
        "1"
    ]
    assert BM25Retriever(corpus).search("2,00,000", 1) == []


def test_the_defaults_are_the_step_4_4b_tuning():
    """ADR-077. A silent revert to Okapi's 0.75 would undo a measured gain."""
    assert (K1, B) == (1.5, 0.3)


def test_b_decides_between_term_frequency_and_length():
    """With b=0 length is ignored and the chunk saying "tax" twice wins; with
    b=1 its length counts against it and the short chunk wins."""
    corpus = [
        chunk("1", "tax"),
        chunk("2", "tax tax " + " ".join(f"filler{n}" for n in range(30))),
    ]
    assert paths(BM25Retriever(corpus, b=0.0).search("tax", 1)) == ["2"]
    assert paths(BM25Retriever(corpus, b=1.0).search("tax", 1)) == ["1"]


@pytest.mark.parametrize("kwargs", [{"k1": -1.0}, {"b": 1.5}, {"b": -0.1}])
def test_k1_and_b_are_validated(kwargs):
    with pytest.raises(ValueError):
        BM25Retriever(CORPUS, **kwargs)


def test_weighted_search_at_unit_weight_is_search(retriever):
    query = "annual value of the property deduction"
    weights = dict.fromkeys(tokenize(query), 1.0)
    assert [(r.chunk.chunk_id, r.score) for r in retriever.search(query, 5)] == [
        (r.chunk.chunk_id, r.score) for r in retriever.search_weighted(weights, 5)
    ]


def test_a_zero_weight_term_adds_nothing(retriever):
    plain = retriever.search("annual", 5)
    weighted = retriever.search_weighted({"annual": 1.0, "deduction": 0.0}, 5)
    assert [(r.chunk.chunk_id, r.score) for r in plain] == [
        (r.chunk.chunk_id, r.score) for r in weighted
    ]


def test_term_frequencies_cover_the_indexed_surface(retriever):
    """The breadcrumb is indexed, so its words count too."""
    assert retriever.term_frequencies(CORPUS[0])["house"] >= 1
    assert retriever.idf("house") > 0
    assert retriever.idf("never-indexed") == 0.0


def test_an_empty_corpus_does_not_divide_by_zero():
    assert BM25Retriever([]).search("anything", 5) == []


def test_the_index_covers_the_whole_chunk_set(chunks):
    assert len(BM25Retriever(chunks)) == len(chunks)


def test_a_quoted_statutory_phrase_finds_its_own_chunk(chunks):
    """The weakest possible retrieval claim, and the one that must hold: text
    lifted verbatim out of a chunk retrieves that chunk."""
    retriever = BM25Retriever(chunks)
    target = next(c for c in chunks if c.node_path == "22")
    assert "22" in paths(retriever.search(" ".join(target.text.split()[:25]), 5))


def test_the_baseline_beats_a_random_retriever_on_the_gold_set(gold, chunks):
    """Not a target — Step 3.6 owns the number. This only pins that the whole
    chain runs against the real corpus and clears chance by a wide margin."""
    retriever = BM25Retriever(chunks)
    answerable = [q for q in gold if q.slice is not QuerySlice.NEGATIVE]
    runs = {
        q.query_id: as_ranked_citations(retriever.search(q.question, 10))
        for q in answerable
    }
    report = score_run(gold, runs, 10, CitationIndex(chunks))
    assert report.overall["lenient"].recall > 0.30
