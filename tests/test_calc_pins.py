"""The provisions pinned ahead of the ranking for a calculation question
resolve to real corpus chunks and leave the production pack room for the
ranking itself."""

from __future__ import annotations

from taxverity.graph.build import PRODUCTION_EVIDENCE_BUDGET
from taxverity.graph.state import CALC_PIN_CITATIONS
from taxverity.retrieval.citations import EXACT_SCORE, CitationRetriever
from taxverity.retrieval.evidence import EvidencePacker


def test_every_pin_is_an_exact_corpus_chunk(chunks):
    lookup = CitationRetriever(chunks)
    for citation in CALC_PIN_CITATIONS:
        found = lookup.lookup(citation)
        assert found is not None, citation
        assert found.chunk.node_path == citation
        assert found.score == EXACT_SCORE


def test_the_pins_are_the_slab_rates_and_the_rebate(chunks):
    lookup = CitationRetriever(chunks)
    slabs = lookup.lookup("202(1)").chunk.text
    rebate = lookup.lookup("156").chunk.text
    assert "rate of tax given in the following Table" in slabs
    assert "Rs. 400001" in slabs
    assert "deduction of 100% of income-tax payable" in rebate


def test_the_pins_use_at_most_a_quarter_of_the_production_budget(chunks):
    lookup = CitationRetriever(chunks)
    pins = [lookup.lookup(citation) for citation in CALC_PIN_CITATIONS]
    pack = EvidencePacker(chunks, budget=PRODUCTION_EVIDENCE_BUDGET).pack(pins)
    assert [unit.citation for unit in pack.units] == list(CALC_PIN_CITATIONS)
    assert sum(unit.tokens for unit in pack.units) <= PRODUCTION_EVIDENCE_BUDGET // 4
