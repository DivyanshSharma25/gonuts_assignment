"""Deterministic Markdown output for extracted PDF pages."""

from pathlib import Path

from .text_extractor import PageText


def format_document(
    source: Path,
    pages: list[PageText],
    visual_text: dict[int, str],
    errors: dict[int, str],
) -> str:
    print(
        f"[markdown] formatting {source.name}: {len(pages)} pages, "
        f"{len(visual_text)} visual results, {len(errors)} errors",
        flush=True,
    )
    lines = [
        f"# {source.stem}",
        "",
        f"> Source PDF: `{source.name}`",
        "",
        "This file preserves page boundaries. Text marked as visual extraction was "
        "transcribed by the configured local VLM and must be reviewed before use as evidence.",
        "",
    ]
    for page in pages:
        lines.extend([f"## Page {page.page_number}", ""])
        if page.text:
            lines.extend(["### Text extraction", "", page.text, ""])
        else:
            lines.extend(["### Text extraction", "", "_No text layer was available._", ""])
        if page.page_number in visual_text:
            lines.extend(
                [
                    "### Visual extraction (local VLM)",
                    "",
                    visual_text[page.page_number],
                    "",
                ]
            )
        if page.page_number in errors:
            lines.extend(
                [
                    "### Visual extraction status",
                    "",
                    f"> VLM fallback failed: {errors[page.page_number]}",
                    "",
                ]
            )
    markdown = "\n".join(lines).rstrip() + "\n"
    print(
        f"[markdown] formatted {source.name}: {len(markdown)} characters",
        flush=True,
    )
    return markdown
