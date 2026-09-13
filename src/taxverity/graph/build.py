"""Steps 13.1 and 13.3 — the compiled graph: state, nodes and edges wired
together, plus `build_deps` for production's own composition.

Sync graph, since every client in this codebase is sync — `astream_events`
would need an async one for no benefit here. Streaming goes out through
`stream_mode="custom"` via `get_stream_writer()` inside each node (rule 04),
which is simpler than mapping LangGraph's own event stream onto this
project's SSE contract.

The scope branch is one conditional edge: `prohibited | out_of_scope |
adjacent` never reach retrieval, the calculator or the LLM (rule 03).
`route_calc`'s `TEXT_ONLY`/`INCOMPLETE`/`COMPUTE` split (PLAN 13.3) is not a
graph branch — it is handled inside `nodes.route_calc` by varying what
reaches `generate_verify`, since every route still answers through the same
generation-and-verification step.

The other conditional edge is Step 13.5's minimal corrective loop (ADR-033 as
amended by ADR-110): after `generate_verify`, `_retry_branch` sends the run
back to `retrieve_retry` (wider pool, `pack(expand=True)`) exactly once, when
the first pass served zero statute claims. `retrieve_retry` feeds back into
`generate_verify` rather than into `route_calc` — a retry changes only the
evidence pack, never the calculator's inputs. The retry is bounded to one
cycle by `state["retried"]`, checked in `_retry_branch` itself, not by any
counter on the node.
"""

from __future__ import annotations

from functools import partial

import psycopg
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import Settings
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.generation.generate import AnswerGenerator
from taxverity.graph import nodes
from taxverity.graph.state import GraphDeps, GraphState
from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import LLMClient
from taxverity.llm.extract import FactExtractor
from taxverity.llm.tracing import LangfuseTracer, TracedLLMClient
from taxverity.memory.contextualize import QueryContextualizer
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.bridge import BridgedRetriever, TermBridge, load_bridge_map
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.evidence import EvidencePacker
from taxverity.retrieval.fallback import FallbackRetriever
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.pgvector import PgVectorIndex
from taxverity.retrieval.rerank import CachedReranker, JinaReranker, RerankRetriever
from taxverity.safety.classifier import IntentClassifier, ScopeCategory
from taxverity.safety.evidence_gate import served_statute_claims

GRAPH_BUILD_STAGE_VERSION = 2

_NODES = (
    "load_thread",
    "contextualize",
    "classify",
    "respond_fixed",
    "extract_facts",
    "merge_facts",
    "retrieve",
    "route_calc",
    "generate_verify",
    "retrieve_retry",
    "finalize",
)


def build_deps(settings: Settings, conn: psycopg.Connection) -> GraphDeps:
    """Production's composition (ADR-086) over `PgVectorIndex`, per PLAN 13.2.

    One instance per process; `conn` is a single connection, adequate for
    Phase 13's own tests and scripts. Phase 14's API owns pooling one per
    request.
    """
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    chunks = load_chunks(settings.interim_dir, corpus_version=corpus_version)[0]
    embedding_set_id = settings.require("serving_embedding_set_id")

    bm25 = BM25Retriever(chunks)
    dense = PgVectorIndex(conn, embedding_set_id, JinaAPIEmbedder.from_settings(settings))
    fusion = FallbackRetriever(FusionRetriever([dense, bm25]), bm25)
    bridge = TermBridge(load_bridge_map(), chunks)
    reranker = CachedReranker(JinaReranker.from_settings(settings))
    ranked = BridgedRetriever(bridge, RerankRetriever(fusion, reranker))
    retriever = ShortcutRetriever(CitationRetriever(chunks), ranked)

    by_path = {chunk.node_path: chunk for chunk in chunks}
    llm: object = LLMClient.from_settings(settings)
    llm = CachedLLMClient(llm, settings.llm_cache_dir)
    llm = TracedLLMClient(llm, LangfuseTracer.from_settings(settings))

    return GraphDeps(
        conn=conn,
        chunks=by_path,
        retriever=retriever,
        packer=EvidencePacker(chunks),
        classifier=IntentClassifier.from_settings(settings),
        contextualizer=QueryContextualizer.from_settings(settings),
        extractor=FactExtractor.from_settings(settings),
        generator=AnswerGenerator(llm, by_path),
    )


def build_graph(deps: GraphDeps) -> CompiledStateGraph:
    graph = StateGraph(GraphState)
    for name in _NODES:
        graph.add_node(name, partial(getattr(nodes, name), deps=deps))

    graph.add_edge(START, "load_thread")
    graph.add_edge("load_thread", "contextualize")
    graph.add_edge("contextualize", "classify")
    graph.add_conditional_edges(
        "classify",
        _scope_branch,
        {"in_scope": "extract_facts", "refused": "respond_fixed"},
    )
    graph.add_edge("respond_fixed", "finalize")
    graph.add_edge("extract_facts", "merge_facts")
    graph.add_edge("merge_facts", "retrieve")
    graph.add_edge("retrieve", "route_calc")
    graph.add_edge("route_calc", "generate_verify")
    graph.add_conditional_edges(
        "generate_verify",
        _retry_branch,
        {"retry": "retrieve_retry", "finalize": "finalize"},
    )
    graph.add_edge("retrieve_retry", "generate_verify")
    graph.add_edge("finalize", END)
    return graph.compile()


def _scope_branch(state: GraphState) -> str:
    return "in_scope" if state["category"] is ScopeCategory.IN_SCOPE else "refused"


def _retry_branch(state: GraphState) -> str:
    """Step 13.5. Reached only from `in_scope` (the scope branch never lets a
    refused question this far), so no separate scope check is needed here."""
    if state.get("retried"):
        return "finalize"
    if served_statute_claims(state.get("events", [])) > 0:
        return "finalize"
    return "retry"
