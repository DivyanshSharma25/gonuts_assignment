"""PDF page rendering for visual fallback extraction."""

import base64

import pymupdf


def render_page_base64(page: pymupdf.Page, dpi: int) -> str:
    """Render a page as a PNG data payload suitable for Ollama."""
    print(f"[image] rendering page {page.number + 1} at {dpi} DPI", flush=True)
    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    print(
        f"[image] page {page.number + 1} rendered: {len(image)} base64 characters",
        flush=True,
    )
    return image
