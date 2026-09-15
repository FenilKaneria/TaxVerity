"""Steps 13.1 and 13.3 — the compiled graph: state, nodes and edges wired
together, plus `build_deps` for production's own composition.

Sync graph, since every client in this codebase is sync — `astream_events`
would need an async one for no benefit here. Streaming goes out through
`stream_mode="custom"` via `get_stream_writer()` inside each node (rule 04),
which is simpler than mapping LangGraph's own event stream onto this
project's SSE contract.

The scope branch is one conditional edge, now three-way: `prohibited |
out_of_scope | adjacent` never reach retrieval, the calculator or the LLM
(rule 03); `conversational` (advisor pivot, Step 5) skips retrieval and the
calculator but does reach one small guarded LLM call
(`llm/conversational.py`), never the generator/verifier.
`route_calc`'s `TEXT_ONLY`/`INCOMPLETE`/`COMPUTE` split (PLAN 13.3) is not a
graph branch — it is handled inside `nodes.route_calc` by varying what
reaches `generate_verify`, since every route still answers through the same
generation-and-verification step.

The other conditional edge is Step 13.5's minimal corrective loop (ADR-033 as
amended by ADR-110): after `generate_verify`, `_retry_branch` sends the run
back to `retrieve_retry` (wider pool, `pack(expand=True)`) exactly once, when
the first pass served zero grounded (statute/advice) claims. `retrieve_retry` feeds back into
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

from taxverity.config import Settings
from taxverity.db.chunks import load_chunks_from_db
from taxverity.db.serving import ServedCorpus, resolve_serving
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.generation.generate import AnswerGenerator
from taxverity.graph import nodes
from taxverity.graph.state import GraphDeps, GraphState
from taxverity.llm.client import GEMINI, GROQ_20B, LLMClient
from taxverity.llm.conversational import Conversationalist
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
from taxverity.safety.evidence_gate import served_grounded_claims

GRAPH_BUILD_STAGE_VERSION = 3

_NODES = (
    "load_thread",
    "contextualize",
    "classify",
    "respond_fixed",
    "respond_conversational",
    "extract_facts",
    "merge_facts",
    "retrieve",
    "route_calc",
    "generate_verify",
    "retrieve_retry",
    "finalize",
)


def build_deps(
    settings: Settings, conn: psycopg.Connection
) -> tuple[GraphDeps, ServedCorpus]:
    """Production's composition (ADR-086) over `PgVectorIndex`, per PLAN 13.2,
    widened at Phase 14 to hydrate its chunks from the database rather than a
    local `chunks.jsonl` — TaxVerity is deploy-only, and a production instance
    has no corpus files on disk (ADR-006's serving guard, Step 6.6).

    `resolve_serving()` runs first, per its own contract: an outage cannot
    stop this from refusing a schema or corpus this code cannot serve. One
    instance per process; `conn` is a single connection here, adequate for
    Phase 13's own tests and scripts. Phase 14's API builds this once at
    startup against one dedicated connection for `PgVectorIndex`, then swaps
    `GraphDeps.conn` per request with `dataclasses.replace` — the request
    connection never reaches the dense index, which does only read queries.
    """
    embedding_set_id = settings.require("serving_embedding_set_id")
    served = resolve_serving(
        conn,
        corpus_version=settings.require("serving_corpus_version"),
        embedding_set_id=embedding_set_id,
    )
    chunks = load_chunks_from_db(conn, served.corpus_version)

    bm25 = BM25Retriever(chunks)
    dense = PgVectorIndex(conn, embedding_set_id, JinaAPIEmbedder.from_settings(settings))
    fusion = FallbackRetriever(FusionRetriever([dense, bm25]), bm25)
    bridge = TermBridge(load_bridge_map(), chunks)
    reranker = CachedReranker(JinaReranker.from_settings(settings))
    ranked = BridgedRetriever(bridge, RerankRetriever(fusion, reranker))
    retriever = ShortcutRetriever(CitationRetriever(chunks), ranked)

    by_path = {chunk.node_path: chunk for chunk in chunks}
    llm: object = LLMClient.from_settings(settings)
    llm = TracedLLMClient(llm, LangfuseTracer.from_settings(settings))

    # R18 per-node routing (ADR-118, PLAN R18): `classify` and `contextualize`
    # moved to Groq's 20b model after clearing all three gates (streaming
    # correctness argued from `LineBuffer.feed()` plus a partial live check —
    # neither node streams, so gate 1 doesn't apply to them; a live two-arm
    # answer-quality check isn't meaningful for a schema-bound classification/
    # rewrite call the way it is for generation; the 20b eval re-run cleared
    # each node's own floor). `extract_facts` failed its 20b strict-rate floor
    # (0.857 < 0.88, `reports/extraction_eval_20b.md`) and stays on 120b.
    # `respond_conversational` has no eval set to gate it — it did not exist
    # when R18 was planned — and stays on 120b until one is built. Generation
    # (`AnswerGenerator`'s `llm` above) stays on 120b: gate 2 (the two-arm
    # benchmark) is blocked by exhausted Gemini free-tier quota this session,
    # so R18 for that node is still `Proposed`, not decided.
    deps = GraphDeps(
        conn=conn,
        chunks=by_path,
        retriever=retriever,
        packer=EvidencePacker(chunks),
        classifier=IntentClassifier.from_settings(
            settings, cache=False, primary=GROQ_20B, fallback=GEMINI
        ),
        contextualizer=QueryContextualizer.from_settings(
            settings, cache=False, primary=GROQ_20B, fallback=GEMINI
        ),
        extractor=FactExtractor.from_settings(settings, cache=False),
        generator=AnswerGenerator(llm, by_path),
        conversational=Conversationalist.from_settings(settings, cache=False),
    )
    return deps, served


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
        {
            "in_scope": "extract_facts",
            "conversational": "respond_conversational",
            "refused": "respond_fixed",
        },
    )
    graph.add_edge("respond_fixed", "finalize")
    graph.add_edge("respond_conversational", "finalize")
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
    category = state["category"]
    if category is ScopeCategory.IN_SCOPE:
        return "in_scope"
    if category is ScopeCategory.CONVERSATIONAL:
        return "conversational"
    return "refused"


def _retry_branch(state: GraphState) -> str:
    """Step 13.5. Reached only from `in_scope` (the scope branch never lets a
    refused question this far), so no separate scope check is needed here."""
    if state.get("retried"):
        return "finalize"
    if served_grounded_claims(state.get("events", [])) > 0:
        return "finalize"
    return "retry"
