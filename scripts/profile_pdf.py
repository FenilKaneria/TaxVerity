"""Step 1.1 corpus characterisation spike — a one-off investigation, not library code."""

import re
import statistics
import unicodedata
from collections import Counter, defaultdict

import pymupdf

from taxverity.config import Settings
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

BOLD_FLAG = 1 << 4
ITALIC_FLAG = 1 << 1

NEAR_EMPTY_CHARS = 50
EXPECTED_SECTIONS = 536

BODY_FONT_PREFIX = "LiberationSerif"
FURNITURE_FONT_PREFIX = "NotoSans"

SECTION_NUMBER_SPAN = re.compile(r"^(\d{1,3})\.$")
SECTION_LINE_START = re.compile(r"^\s*(\d{1,3})\.\s+\S", re.MULTILINE)

MARKERS = {
    "chapter": re.compile(r"^\s*CHAPTER\s+[IVXLCDM]+\b", re.MULTILINE),
    "schedule": re.compile(r"^\s*SCHEDULE\s+[IVXLCDM]+\b", re.MULTILINE),
    "subsection": re.compile(r"^\s*\(\d{1,3}\)\s", re.MULTILINE),
    "clause": re.compile(r"^\s*\([a-z]{1,2}\)\s", re.MULTILINE),
    "explanation": re.compile(r"^\s*Explanation\b", re.MULTILINE),
    "proviso": re.compile(r"Provided\s+(?:further\s+|also\s+)?that\b"),
}


def normalise(text):
    """Soft hyphens and NBSPs break otherwise-clean matches; see the report."""
    return text.replace("\xad", "").replace("\xa0", " ")


def scan(doc):
    pages = []
    fonts = Counter()
    sizes = Counter()
    marker_pages = defaultdict(list)
    odd_chars = Counter()
    furniture_strings = Counter()
    furniture_pages = 0
    bold_section_numbers = set()
    regex_section_numbers = set()

    for index, page in enumerate(doc):
        raw = page.get_text("text")
        text = normalise(raw)

        for char in raw:
            category = unicodedata.category(char)
            if char in "\n\r\t":
                continue
            if ord(char) > 127 or category in ("Cc", "Cf", "Co", "Cn", "Cs"):
                odd_chars[char] += 1

        saw_furniture = False
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    span_text = normalise(span["text"]).strip()
                    if not span_text:
                        continue
                    is_bold = bool(span["flags"] & BOLD_FLAG)
                    fonts[
                        (
                            span["font"],
                            round(span["size"], 1),
                            is_bold,
                            bool(span["flags"] & ITALIC_FLAG),
                        )
                    ] += len(span_text)
                    sizes[round(span["size"], 1)] += len(span_text)

                    if span["font"].startswith(FURNITURE_FONT_PREFIX):
                        if len(span_text) > 5:
                            furniture_strings[span_text] += 1
                            saw_furniture = True
                    elif span["font"].startswith(BODY_FONT_PREFIX) and is_bold:
                        match = SECTION_NUMBER_SPAN.match(span_text)
                        if match:
                            bold_section_numbers.add(int(match.group(1)))

        if saw_furniture:
            furniture_pages += 1

        regex_section_numbers.update(int(m) for m in SECTION_LINE_START.findall(text))

        for name, pattern in MARKERS.items():
            if pattern.search(text):
                marker_pages[name].append(index)

        pages.append(
            {
                "index": index,
                "chars": len(text.strip()),
                "images": len(page.get_images(full=True)),
            }
        )

        if index and index % 200 == 0:
            logger.info("profiled %d pages", index)

    return {
        "pages": pages,
        "fonts": fonts,
        "sizes": sizes,
        "marker_pages": marker_pages,
        "odd_chars": odd_chars,
        "furniture_strings": furniture_strings,
        "furniture_pages": furniture_pages,
        "bold_section_numbers": bold_section_numbers,
        "regex_section_numbers": regex_section_numbers,
    }


def section_cue_coverage(scanned):
    expected = set(range(1, EXPECTED_SECTIONS + 1))
    bold = scanned["bold_section_numbers"]
    regex = scanned["regex_section_numbers"]
    return {
        "bold": (len(bold & expected), sorted(expected - bold)),
        "regex": (len(regex & expected), sorted(expected - regex)),
        "union": (len((bold | regex) & expected), sorted(expected - (bold | regex))),
    }


def pick_sample_pages(pages, marker_pages):
    """Deterministic, and justified by the measurements rather than guessed."""
    non_empty = [p for p in pages if p["chars"] > NEAR_EMPTY_CHARS]
    picks = {}

    def add(label, index):
        if index is not None and 0 <= index < len(pages):
            picks.setdefault(index, label)

    add("first-page", 0)
    for marker in ("chapter", "subsection", "clause", "explanation", "schedule"):
        if marker_pages[marker]:
            add(f"first-{marker}", marker_pages[marker][0])
    if marker_pages["schedule"]:
        add("last-schedule", marker_pages["schedule"][-1])
    if non_empty:
        add("densest-page", max(non_empty, key=lambda p: p["chars"])["index"])
        add("sparsest-non-empty", min(non_empty, key=lambda p: p["chars"])["index"])
    add("midpoint", len(pages) // 2)
    add("last-page", len(pages) - 1)
    return dict(sorted(picks.items()))


def compare_extractors(pdf_path, page_indexes):
    try:
        import pdfplumber
    except ImportError:
        return None
    rows = []
    with pdfplumber.open(pdf_path) as plumber_doc, pymupdf.open(pdf_path) as mupdf_doc:
        for index, label in page_indexes.items():
            plumber_page = plumber_doc.pages[index]
            rows.append(
                {
                    "index": index,
                    "label": label,
                    "mupdf_chars": len(mupdf_doc[index].get_text("text").strip()),
                    "plumber_chars": len((plumber_page.extract_text() or "").strip()),
                    "tables": len(plumber_page.find_tables()),
                }
            )
    return rows


def write_sample_pages(pdf_path, page_indexes, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("*.txt"):
        existing.unlink()
    with pymupdf.open(pdf_path) as doc:
        for index, label in page_indexes.items():
            target = out_dir / f"page_{index:04d}_{label}.txt"
            target.write_text(doc[index].get_text("text"), encoding="utf-8")
    return out_dir


def render(doc, pdf_path, scanned, coverage, samples, compare):
    pages = scanned["pages"]
    fonts = scanned["fonts"]
    chars = [p["chars"] for p in pages]
    non_empty = [c for c in chars if c > NEAR_EMPTY_CHARS]
    total_font_chars = sum(fonts.values()) or 1

    out = [
        "# Corpus profile — Income-tax Act, 2025",
        "",
        "Auto-generated by `scripts/profile_pdf.py` (Step 1.1). Measurements only —",
        "the interpretation lives in the Step 1.1 entry in `PLAN.md`. Re-running",
        "regenerates this file.",
        "",
        "## Document",
        "",
        f"- Source: `{pdf_path.name}` ({pdf_path.stat().st_size / 1024**2:.1f} MiB)",
        f"- Pages: **{doc.page_count}**",
        f"- Format: {doc.metadata.get('format')}",
        f"- Encrypted: {doc.is_encrypted} · needs password: {bool(doc.needs_pass)}",
        f"- Producer: {doc.metadata.get('producer')!r}",
        f"- Bookmarks / outline entries: **{len(doc.get_toc())}**",
        "",
        "## Text layer",
        "",
        f"- Total extracted characters: **{sum(chars):,}**",
        f"- Pages with >{NEAR_EMPTY_CHARS} chars: **{len(non_empty)} / {len(pages)}** "
        f"({100 * len(non_empty) / max(len(pages), 1):.1f}%)",
        f"- Mean / median chars per page: {statistics.mean(non_empty):,.0f} / "
        f"{statistics.median(non_empty):,.0f}",
        f"- Min / max chars on a page: {min(non_empty):,} / {max(non_empty):,}",
        f"- Embedded images across the whole document: **{sum(p['images'] for p in pages)}**",
        "",
        "## Section-heading cues",
        "",
        f"Against the {EXPECTED_SECTIONS} sections the Act is known to contain, after",
        "normalising soft hyphens and non-breaking spaces.",
        "",
        "| Cue | Sections found | Missing |",
        "|---|---|---|",
    ]
    for name, (found, missing) in coverage.items():
        shown = ", ".join(str(m) for m in missing[:18]) or "none"
        out.append(f"| {name} | **{found} / {EXPECTED_SECTIONS}** | {shown} |")
    out += [
        "",
        "`bold` = a `LiberationSerif-Bold` span matching `<n>.`; `regex` = a",
        "line-start `<n>. ` match in the page text; `union` = either. The two cues",
        "fail on disjoint sets, which is the finding that matters for Step 1.5.",
        "",
        "## Font inventory",
        "",
        "| Font | Size | Bold | Italic | Chars | Share |",
        "|---|---|---|---|---|---|",
    ]
    for (name, size, bold, italic), count in fonts.most_common(15):
        out.append(
            f"| `{name}` | {size} | {'yes' if bold else ''} | {'yes' if italic else ''} "
            f"| {count:,} | {100 * count / total_font_chars:.1f}% |"
        )

    out += [
        "",
        "## Page furniture",
        "",
        f"- Pages carrying a `{FURNITURE_FONT_PREFIX}` running footer: "
        f"**{scanned['furniture_pages']} / {doc.page_count}**",
        "",
        "| Footer string | Pages |",
        "|---|---|",
    ]
    for text, count in scanned["furniture_strings"].most_common(5):
        out.append(f"| `{text[:60]}` | {count} |")
    out += [
        "",
        f"Note that `{FURNITURE_FONT_PREFIX}` is **not** exclusively furniture — it also",
        "renders some clause markers, so the footer must be stripped by string or",
        "position, never by font alone.",
        "",
        "## Non-ASCII and format characters",
        "",
        "| Codepoint | Name | Count |",
        "|---|---|---|",
    ]
    for char, count in scanned["odd_chars"].most_common(12):
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "<unnamed>"
        out.append(f"| U+{ord(char):04X} | {name} | {count:,} |")

    out += [
        "",
        "## Structural markers",
        "",
        "| Marker | Pages matched | First | Last |",
        "|---|---|---|---|",
    ]
    for name in MARKERS:
        hits = scanned["marker_pages"].get(name, [])
        out.append(
            f"| {name} | {len(hits)} | {hits[0] if hits else '—'} | {hits[-1] if hits else '—'} |"
        )

    if compare:
        out += [
            "",
            "## Extractor comparison — PyMuPDF vs pdfplumber",
            "",
            "| Page | Label | PyMuPDF chars | pdfplumber chars | pdfplumber tables |",
            "|---|---|---|---|---|",
        ]
        for row in compare:
            out.append(
                f"| {row['index']} | {row['label']} | {row['mupdf_chars']:,} "
                f"| {row['plumber_chars']:,} | {row['tables']} |"
            )

    out += [
        "",
        "## Sample pages written",
        "",
        "| Page | Label | Chars |",
        "|---|---|---|",
    ]
    for index, label in samples.items():
        out.append(f"| {index} | {label} | {pages[index]['chars']:,} |")

    return "\n".join(out) + "\n"


def main():
    configure_logging()
    settings = Settings()
    pdf_path = settings.resolve_corpus_pdf()
    logger.info("profiling %s", pdf_path)

    with pymupdf.open(pdf_path) as doc:
        scanned = scan(doc)
        coverage = section_cue_coverage(scanned)
        samples = pick_sample_pages(scanned["pages"], scanned["marker_pages"])
        compare = compare_extractors(pdf_path, samples)
        report = render(doc, pdf_path, scanned, coverage, samples, compare)

    sample_dir = write_sample_pages(
        pdf_path, samples, settings.interim_dir / "sample_pages"
    )
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.reports_dir / "corpus_profile.md"
    report_path.write_text(report, encoding="utf-8")

    logger.info("wrote %s", report_path)
    logger.info("wrote %d sample pages to %s", len(samples), sample_dir)


if __name__ == "__main__":
    main()
