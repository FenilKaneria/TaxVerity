"""The compiled graph: state, nodes and edges wired together, plus `build_deps` for
production's own composition."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from typing import Any

import psycopg
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from taxverity.config import Settings
from taxverity.db.chunks import load_chunks_from_db
from taxverity.db.serving import ServedCorpus, resolve_serving
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.generation.generate import AnswerGenerator
from taxverity.graph import nodes
from taxverity.graph.state import CALC_PIN_CITATIONS, GraphDeps, GraphState
from taxverity.llm.client import (
    GEMINI,
    GROQ,
    GROQ_20B,
    OPENAI_GPT5,
    LLMClient,
    Provider,
)
from taxverity.llm.conversational import Conversationalist
from taxverity.llm.extract import FactExtractor
from taxverity.llm.tracing import LangfuseTracer, TracedLLMClient
from taxverity.reasoning.reason import Reasoner
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.bridge import BridgedRetriever, TermBridge, load_bridge_map
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.evidence import EvidencePacker
from taxverity.retrieval.fallback import FallbackRetriever
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.pgvector import PgVectorIndex
from taxverity.retrieval.rerank import CachedReranker, JinaReranker, RerankRetriever
from taxverity.safety.classifier import IntentClassifier, ScopeCategory
from taxverity.safety.evidence_gate import served_grounded_claims

GRAPH_BUILD_STAGE_VERSION = 11

PRODUCTION_EVIDENCE_BUDGET = 3_000

QUERY_EMBED_TIMEOUT = 5.0
QUERY_EMBED_ATTEMPTS = 2

_NODES = (
    "load_thread",
    "classify",
    "respond_fixed",
    "respond_conversational",
    "extract_facts",
    "merge_facts",
    "retrieve",
    "route_calc",
    "reason",
    "decide",
    "generate_verify",
    "retrieve_retry",
    "finalize",
)


def build_deps(
    settings: Settings, conn: psycopg.Connection
) -> tuple[GraphDeps, ServedCorpus]:
    embedding_set_id = settings.require("serving_embedding_set_id")
    served = resolve_serving(
        conn,
        corpus_version=settings.require("serving_corpus_version"),
        embedding_set_id=embedding_set_id,
    )
    chunks = load_chunks_from_db(conn, served.corpus_version)

    bm25 = BM25Retriever(chunks)
    # The query path must fail fast, not after the offline batch
    # defaults (120 s x 6 attempts) — a dense failure degrades to BM25 + the
    # citation shortcut, which is far better than a stalled turn.
    query_embedder = JinaAPIEmbedder.from_settings(
        settings, timeout=QUERY_EMBED_TIMEOUT, max_attempts=QUERY_EMBED_ATTEMPTS
    )
    dense = PgVectorIndex(conn, embedding_set_id, query_embedder)
    fusion = FallbackRetriever(FusionRetriever([dense, bm25]), bm25)
    bridge = TermBridge(load_bridge_map(), chunks)
    reranker = CachedReranker(JinaReranker.from_settings(settings))
    ranked = BridgedRetriever(bridge, RerankRetriever(fusion, reranker))
    citations = CitationRetriever(chunks)
    retriever = ShortcutRetriever(citations, ranked)
    calc_pins = tuple(
        found
        for citation in CALC_PIN_CITATIONS
        if (found := citations.lookup(citation))
    )

    by_path = {chunk.node_path: chunk for chunk in chunks}
    generation_primary, generation_fallback = generation_providers(settings)
    llm: object = LLMClient.from_settings(
        settings, primary=generation_primary, fallback=generation_fallback
    )
    llm = TracedLLMClient(llm, LangfuseTracer.from_settings(settings))

    deps = GraphDeps(
        conn=conn,
        chunks=by_path,
        retriever=retriever,
        packer=EvidencePacker(chunks, budget=PRODUCTION_EVIDENCE_BUDGET),
        classifier=IntentClassifier.from_settings(
            settings, cache=False, primary=GROQ_20B, fallback=GEMINI
        ),
        extractor=FactExtractor.from_settings(settings, cache=False),
        generator=AnswerGenerator(llm, by_path),
        conversational=Conversationalist.from_settings(settings, cache=False),
        reasoner=Reasoner.from_settings(settings, cache=False),
        calc_pins=calc_pins,
        citation_lookup=citations.lookup,
    )
    return deps, served


def generation_providers(settings: Settings) -> tuple[Provider, Provider]:
    """(primary, fallback) for generate + repair."""
    if settings.generation_llm == "openai":
        return OPENAI_GPT5, GROQ
    return GROQ, GEMINI


def _timed(name: str, fn: Callable[..., dict]) -> Callable[..., dict]:
    """Wraps a node with wall-clock timing for the trace panel (always visible, per user
    decision)."""

    def wrapper(state: GraphState, *args: Any, **kwargs: Any) -> dict:
        start = time.perf_counter()
        result = fn(state, *args, **kwargs)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        sub_trace = result.get("trace") or []
        return {**result, "trace": [*sub_trace, {"node": name, "ms": elapsed_ms}]}

    return wrapper


def build_graph(deps: GraphDeps) -> CompiledStateGraph:
    graph = StateGraph(GraphState)
    for name in _NODES:
        defer = name == "reason"
        graph.add_node(
            name, _timed(name, partial(getattr(nodes, name), deps=deps)), defer=defer
        )

    graph.add_edge(START, "load_thread")
    graph.add_edge("load_thread", "classify")
    graph.add_conditional_edges(
        "classify",
        _scope_branch,
        {
            "extract_facts": "extract_facts",
            "retrieve": "retrieve",
            "conversational": "respond_conversational",
            "refused": "respond_fixed",
        },
    )
    graph.add_edge("respond_fixed", "finalize")
    graph.add_edge("respond_conversational", "finalize")
    graph.add_edge("extract_facts", "merge_facts")
    graph.add_edge("merge_facts", "route_calc")
    graph.add_edge("route_calc", "reason")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "decide")
    graph.add_edge("decide", "generate_verify")
    graph.add_conditional_edges(
        "generate_verify",
        _retry_branch,
        {"retry": "retrieve_retry", "finalize": "finalize"},
    )
    graph.add_edge("retrieve_retry", "generate_verify")
    graph.add_edge("finalize", END)
    return graph.compile()


def _scope_branch(state: GraphState) -> list[str]:
    """`in_scope` fans out to two parallel branches — `extract_facts` (-> `merge_facts`
    -> `route_calc`) needs only the fact state, `retrieve` needs only
    `query`/`search_query` from `classify`, and neither reads the other's output."""
    category = state["category"]
    if category is ScopeCategory.IN_SCOPE:
        return ["extract_facts", "retrieve"]
    if category is ScopeCategory.CONVERSATIONAL:
        return ["conversational"]
    return ["refused"]


def _retry_branch(state: GraphState) -> str:
    if state.get("retried"):
        return "finalize"
    if served_grounded_claims(state.get("events", [])) > 0:
        return "finalize"
    return "retry"
