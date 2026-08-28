from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from taxverity.corpus.loader import normalise
from taxverity.corpus.nodes import NodePath, NodeType

CHUNK_STAGE_VERSION = 1

CHUNK_ID_CHARS = 16
CHUNK_ID = re.compile(rf"^[0-9a-f]{{{CHUNK_ID_CHARS}}}$")

# ACT and CHAPTER exist to describe the document, not to be retrieved: neither
# is citable on its own, and NodePath refuses to start at either.
CHUNKABLE_TYPES = frozenset(NodeType) - {NodeType.ACT, NodeType.CHAPTER}

BREADCRUMB_SEPARATOR = " / "
TITLE_SEPARATOR = " — "

# A separator the corpus cannot contain, so that shifting a boundary between
# the three inputs cannot produce the same digest. The plan wrote plain
# concatenation; this is the same recipe made collision-safe.
ID_SEPARATOR = "\x00"


def compute_chunk_id(corpus_version: str, node_path: str, text: str) -> str:
    payload = ID_SEPARATOR.join((corpus_version, node_path, normalise(text)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CHUNK_ID_CHARS]


def build_citation_label(node_path: str, title: str | None) -> str:
    path = NodePath.parse(node_path)
    rendered = path.render()
    if path.components[0].type is NodeType.SECTION:
        rendered = f"Section {rendered}"
    return f"{rendered}{TITLE_SEPARATOR}{title}" if title else rendered


def build_breadcrumb(
    citation_label: str, chapter_numeral: str | None, chapter_title: str | None
) -> str:
    if not chapter_numeral:
        return citation_label
    chapter = f"Chapter {chapter_numeral}"
    if chapter_title:
        chapter = f"{chapter}{TITLE_SEPARATOR}{chapter_title}"
    return f"{chapter}{BREADCRUMB_SEPARATOR}{citation_label}"


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    parent_id: str | None
    doc_id: str
    corpus_version: str

    node_type: NodeType
    node_path: str
    section_number: str | None = None
    schedule_number: str | None = None
    root_title: str | None = None
    chapter_numeral: str | None = None
    chapter_title: str | None = None

    text: str
    page_start: int
    page_end: int
    char_start: int | None = None
    char_end: int | None = None

    outgoing_refs: tuple[str, ...] = ()
    defined_terms: tuple[str, ...] = ()
    token_count: int | None = None

    @field_validator("chunk_id", "parent_id")
    @classmethod
    def ids_are_chunk_ids(cls, value: str | None) -> str | None:
        if value is not None and not CHUNK_ID.match(value):
            raise ValueError(f"not a chunk id: {value!r}")
        return value

    @field_validator("doc_id", "corpus_version", "text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("node_type")
    @classmethod
    def type_is_chunkable(cls, value: NodeType) -> NodeType:
        if value not in CHUNKABLE_TYPES:
            raise ValueError(f"{value} is not a citable, retrievable unit")
        return value

    @field_validator("outgoing_refs")
    @classmethod
    def refs_are_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for ref in value:
            NodePath.parse(ref)
        return value

    @model_validator(mode="after")
    def path_agrees_with_its_root_fields(self) -> Chunk:
        path = NodePath.parse(self.node_path)
        root = path.components[0]
        if root.type is NodeType.SECTION:
            expected = (self.section_number, self.schedule_number)
        else:
            expected = (self.schedule_number, self.section_number)
        named, absent = expected
        if named != root.marker:
            raise ValueError(f"{self.node_path!r} does not name {named!r} as its root")
        if absent is not None:
            raise ValueError("a chunk has exactly one root, a section or a schedule")
        # A chunk with no parent is a root of the retrieval hierarchy; anything
        # deeper is a child and must name the parent it was split out of.
        if (path.depth == 1) != (self.parent_id is None):
            raise ValueError("parent_id is set for exactly the non-root chunks")
        return self

    @model_validator(mode="after")
    def spans_are_ordered(self) -> Chunk:
        if self.page_start < 0 or self.page_end < self.page_start:
            raise ValueError(f"bad page span: {self.page_start}..{self.page_end}")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end are set together or not at all")
        if self.char_start is not None:
            assert self.char_end is not None
            if self.char_start < 0 or self.char_end < self.char_start:
                raise ValueError(f"bad char span: {self.char_start}..{self.char_end}")
        if self.token_count is not None and self.token_count < 0:
            raise ValueError(f"bad token count: {self.token_count}")
        return self

    @model_validator(mode="after")
    def id_is_derived_from_its_inputs(self) -> Chunk:
        expected = compute_chunk_id(self.corpus_version, self.node_path, self.text)
        if self.chunk_id != expected:
            raise ValueError(f"chunk_id {self.chunk_id!r} is not {expected!r}")
        return self

    @classmethod
    def create(cls, corpus_version: str, node_path: str, text: str, **fields) -> Chunk:
        return cls(
            chunk_id=compute_chunk_id(corpus_version, node_path, text),
            corpus_version=corpus_version,
            node_path=node_path,
            text=text,
            **fields,
        )

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def citation_label(self) -> str:
        return build_citation_label(self.node_path, self.root_title)

    @property
    def breadcrumb(self) -> str:
        return build_breadcrumb(
            self.citation_label, self.chapter_numeral, self.chapter_title
        )

    def embed_text(self) -> str:
        """What actually gets embedded: the clause plus the hierarchy it sits in."""
        return f"{self.breadcrumb}\n{self.text}"
