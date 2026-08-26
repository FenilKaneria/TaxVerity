from __future__ import annotations

import re
from collections.abc import Iterator
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class NodeType(StrEnum):
    ACT = "act"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    CLAUSE = "clause"
    SUBCLAUSE = "subclause"
    ITEM = "item"
    SUBITEM = "subitem"
    SCHEDULE = "schedule"
    SCHEDULE_PARAGRAPH = "schedule_paragraph"


# Deliberately absent: PROVISO and EXPLANATION. Step 1.1 found zero
# "Provided ... that" constructions in the Act, and all five "Explanation"
# occurrences are references to the Income-tax Act, 1961.

# Which type sits at each depth below a root, used only when reading a citation
# string. Document parsing assigns types from sibling sequence instead: the
# marker alone cannot separate clause (i) from sub-clause roman (i).
# Step 1.6 measured five levels below a section — 9(8)(b)(i)(A)(I) is real — so
# ITEM and SUBITEM are not speculative. The Act names level 4 ("item (ii)") but
# never names level 5; SUBITEM is our label for an unnamed level that exists.
CITATION_DEPTH_TYPES = {
    NodeType.SECTION: (
        NodeType.SUBSECTION,
        NodeType.CLAUSE,
        NodeType.SUBCLAUSE,
        NodeType.ITEM,
        NodeType.SUBITEM,
    ),
    NodeType.SCHEDULE: (
        NodeType.SCHEDULE_PARAGRAPH,
        NodeType.CLAUSE,
        NodeType.SUBCLAUSE,
        NodeType.ITEM,
        NodeType.SUBITEM,
    ),
}

SECTION_MARKER = re.compile(r"^\d{1,3}[A-Z]{0,3}$")
CITATION = re.compile(
    r"^(?:(?P<schedule>Schedule\s+[IVXLCDM]+)|(?P<section>\d{1,3}[A-Z]{0,3}))"
    r"(?P<rest>(?:\([^()]+\))*)$"
)
BRACKETED = re.compile(r"\(([^()]+)\)")


class PathComponent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: NodeType
    marker: str

    @field_validator("marker")
    @classmethod
    def marker_is_bare(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("marker must not be empty")
        if any(char in stripped for char in "()"):
            raise ValueError(f"marker must not contain parentheses: {value!r}")
        return stripped

    @model_validator(mode="after")
    def section_markers_are_numeric_with_an_optional_suffix(self) -> PathComponent:
        if self.type is NodeType.SECTION and not SECTION_MARKER.match(self.marker):
            raise ValueError(f"not a section number: {self.marker!r}")
        return self

    def render(self) -> str:
        if self.type is NodeType.SECTION:
            return self.marker
        if self.type is NodeType.SCHEDULE:
            return f"Schedule {self.marker}"
        return f"({self.marker})"


class NodePath(BaseModel):
    model_config = ConfigDict(frozen=True)

    components: tuple[PathComponent, ...]

    @field_validator("components")
    @classmethod
    def must_start_at_a_citable_root(
        cls, value: tuple[PathComponent, ...]
    ) -> tuple[PathComponent, ...]:
        if not value:
            raise ValueError("a path needs at least one component")
        head = value[0].type
        if head not in CITATION_DEPTH_TYPES:
            raise ValueError(f"a path must start at a section or schedule, not {head}")
        if any(c.type in CITATION_DEPTH_TYPES for c in value[1:]):
            raise ValueError("only the first component may be a section or schedule")
        return value

    @classmethod
    def section(cls, marker: str) -> NodePath:
        return cls(components=(PathComponent(type=NodeType.SECTION, marker=marker),))

    @classmethod
    def schedule(cls, marker: str) -> NodePath:
        return cls(components=(PathComponent(type=NodeType.SCHEDULE, marker=marker),))

    @classmethod
    def parse(cls, text: str) -> NodePath:
        """Read a citation such as ``277(1)(i)``, typing components by depth."""
        match = CITATION.match(text.strip())
        if not match:
            raise ValueError(f"not a citation path: {text!r}")

        if match.group("schedule"):
            root_type = NodeType.SCHEDULE
            root_marker = match.group("schedule").split()[1]
        else:
            root_type = NodeType.SECTION
            root_marker = match.group("section")

        components = [PathComponent(type=root_type, marker=root_marker)]
        depth_types = CITATION_DEPTH_TYPES[root_type]
        markers = BRACKETED.findall(match.group("rest") or "")
        if len(markers) > len(depth_types):
            raise ValueError(f"citation nests deeper than supported: {text!r}")
        for depth, marker in enumerate(markers):
            components.append(PathComponent(type=depth_types[depth], marker=marker))
        return cls(components=tuple(components))

    def child(self, type: NodeType, marker: str) -> NodePath:
        return NodePath(
            components=(*self.components, PathComponent(type=type, marker=marker))
        )

    @property
    def parent(self) -> NodePath | None:
        if len(self.components) == 1:
            return None
        return NodePath(components=self.components[:-1])

    @property
    def depth(self) -> int:
        return len(self.components)

    def render(self) -> str:
        return "".join(component.render() for component in self.components)

    def __str__(self) -> str:
        return self.render()


class StatutoryNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: NodeType
    marker: str
    path: NodePath | None = None
    title: str | None = None
    text: str = ""
    chapter: str | None = None
    pages: tuple[int, ...] = ()
    children: tuple[StatutoryNode, ...] = ()

    @property
    def citation(self) -> str | None:
        return self.path.render() if self.path else None

    def walk(self) -> Iterator[StatutoryNode]:
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, citation: str) -> StatutoryNode | None:
        for node in self.walk():
            if node.citation == citation:
                return node
        return None

    def full_text(self) -> str:
        parts = [self.text] if self.text else []
        parts.extend(child.full_text() for child in self.children)
        return "\n".join(part for part in parts if part)
