"""Hybrid PDF extraction: text layer first, local VLM for visual gaps."""

import json
from pathlib import Path

import pymupdf

from .config import LoaderConfig
from .gemini_vlm_client import GeminiVLM
from .image_extractor import render_page_base64
from .markdown_formatter import format_document
from .text_extractor import extract_pages
from .vlm_client import OllamaVLM, VLMError


class HybridExtractor:
    def __init__(self, config: LoaderConfig) -> None:
        self.config = config
        print(
            f"[loader] configured: input={config.input_dir}, "
            f"output={config.output_dir}, vlm={config.use_vlm}, "
            f"model={config.model}",
            flush=True,
        )
        if config.ocr_backend == "ollama":
            self.vlm = OllamaVLM(
                base_url=config.ollama_url,
                model=config.model,
                timeout=config.request_timeout,
            )
        elif config.ocr_backend == "gemini":
            self.vlm = GeminiVLM(
                model=config.gemini_model,
                api_key_env=config.gemini_api_key_env,
                timeout=config.request_timeout,
                max_retries=config.gemini_max_retries,
                retry_delay=config.gemini_retry_delay,
            )
        else:
            raise ValueError(
                f"Unsupported OCR backend {config.ocr_backend!r}; "
                "choose 'ollama' or 'gemini'"
            )

    def run(self) -> list[dict[str, object]]:
        print("[loader] starting PDF extraction", flush=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(self.config.input_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found in {self.config.input_dir}")
        print(f"[loader] found {len(pdfs)} PDF files", flush=True)

        manifest: list[dict[str, object]] = []
        for source in pdfs:
            manifest.append(self._process_pdf(source))
        manifest_path = self.config.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"[loader] manifest written to {manifest_path}", flush=True)
        print("[loader] PDF extraction complete", flush=True)
        return manifest

    def _process_pdf(self, source: Path) -> dict[str, object]:
        print(f"[loader] processing {source}", flush=True)
        visual_text: dict[int, str] = {}
        errors: dict[int, str] = {}
        with pymupdf.open(source) as document:
            pages = extract_pages(document)
            for page_info in pages:
                needs_visual = (
                    self.config.use_vlm
                    and (
                        page_info.image_count > 0
                        or len(page_info.text) < self.config.min_text_chars
                    )
                )
                if not needs_visual:
                    continue
                print(
                    f"[loader] page {page_info.page_number}: visual fallback required "
                    f"(text={len(page_info.text)}, images={page_info.image_count})",
                    flush=True,
                )
                try:
                    page = document[page_info.page_number - 1]
                    image = render_page_base64(page, self.config.render_dpi)
                    visual_text[page_info.page_number] = self.vlm.extract_page(
                        image, page_info.page_number
                    )
                except VLMError as exc:
                    errors[page_info.page_number] = str(exc)
                    print(
                        f"[loader] page {page_info.page_number}: visual fallback failed",
                        flush=True,
                    )

        output_path = self.config.output_dir / f"{source.stem}.md"
        output_path.write_text(
            format_document(source, pages, visual_text, errors),
            encoding="utf-8",
        )
        print(
            f"[loader] wrote {output_path} "
            f"(visual={len(visual_text)}, errors={len(errors)})",
            flush=True,
        )
        return {
            "source": str(source),
            "output": str(output_path),
            "pages": len(pages),
            "visual_pages": sorted(visual_text),
            "vlm_errors": errors,
            "status": "ok" if not errors else "partial",
        }
