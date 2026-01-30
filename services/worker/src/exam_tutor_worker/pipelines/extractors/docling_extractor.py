from __future__ import annotations

from dataclasses import dataclass

from docling.document_converter import DocumentConverter


@dataclass(frozen=True)
class ExtractedDocument:
    markdown: str


class DoclingExtractor:
    """Docling-based document extractor.

    Uses Docling's DocumentConverter and exports to Markdown.
    Docling supports local paths and URLs.
    """

    def __init__(self):
        self._converter = DocumentConverter()

    def extract(self, source: str) -> ExtractedDocument:
        result = self._converter.convert(source)
        md = result.document.export_to_markdown()
        return ExtractedDocument(markdown=md)
