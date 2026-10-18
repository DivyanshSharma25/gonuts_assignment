"""CLI for the standalone hybrid document loader."""

import argparse
import json
from pathlib import Path

from .config import LoaderConfig
from .hybrid_extractor import HybridExtractor


def main() -> int:
    print("[cli] starting document loader", flush=True)
    parser = argparse.ArgumentParser(
        description="Extract PDFs to Markdown using text plus a configurable vision OCR fallback."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("papers"),
        help="Directory containing source PDFs (default: papers)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/markdowns"),
        help="Directory for Markdown files and manifest (default: data/markdowns)",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="maternion/LightOnOCR-2")
    parser.add_argument(
        "--ocr-backend",
        choices=("ollama", "gemini"),
        default="ollama",
        help="Vision OCR backend (default: ollama).",
    )
    parser.add_argument("--gemini-model", default="gemini-3-flash-preview")
    parser.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--min-text-chars", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Only use the PDF text layer; do not contact Ollama.",
    )
    args = parser.parse_args()

    config = LoaderConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ocr_backend=args.ocr_backend,
        ollama_url=args.ollama_url,
        model=args.model,
        gemini_model=args.gemini_model,
        gemini_api_key_env=args.gemini_api_key_env,
        min_text_chars=args.min_text_chars,
        use_vlm=not args.no_vlm,
        render_dpi=args.dpi,
    )
    print(
        f"[cli] input={config.input_dir}, output={config.output_dir}, "
        f"vlm={'disabled' if args.no_vlm else 'enabled'}, backend={config.ocr_backend}",
        flush=True,
    )
    manifest = HybridExtractor(config).run()
    print(f"[cli] processed {len(manifest)} PDF files", flush=True)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
