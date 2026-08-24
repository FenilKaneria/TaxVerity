from pydantic import BaseModel, ConfigDict

# Rounding at the model boundary keeps serialisation byte-stable: PyMuPDF
# reports sizes like 8.486839294433594, whose repr can drift across versions.
SIZE_PRECISION = 2
BBOX_PRECISION = 2


class Span(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    font: str
    size: float
    bold: bool
    italic: bool
    bbox: tuple[float, float, float, float]
    furniture: bool


class PageArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int
    text: str
    spans: tuple[Span, ...]

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    def body_spans(self) -> tuple[Span, ...]:
        return tuple(span for span in self.spans if not span.furniture)


class CorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_name: str
    source_sha256: str
    source_bytes: int
    page_count: int
    extractor: str
    extractor_version: str
    stage_versions: dict[str, int]
    artifact_sha256: str
