"""Layout-aware text extraction using PyMuPDF4LLM."""

from dataclasses import dataclass

import pymupdf
import pymupdf4llm


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    image_count: int


def extract_pages(document: pymupdf.Document) -> list[PageText]:
    """Extract pages in layout-aware reading order and retain page metadata."""
    print(f"[text] extracting {len(document)} pages", flush=True)
    chunks = pymupdf4llm.to_markdown(document, page_chunks=True)
    if len(chunks) != len(document):
        raise RuntimeError(
            "PyMuPDF4LLM returned an unexpected number of page chunks: "
            f"{len(chunks)} for {len(document)} PDF pages"
        )
    pages: list[PageText] = []
    for index, page in enumerate(document):
        text = str(chunks[index].get("text", "")).strip()
        image_count = len(page.get_images(full=True))
        print(
            f"[text] page {index + 1}: {len(text)} characters, "
            f"{image_count} images",
            flush=True,
        )
        pages.append(
            PageText(
                page_number=index + 1,
                text=text,
                image_count=image_count,
            )
        )
    print("[text] PyMuPDF4LLM extraction complete", flush=True)
    return pages
