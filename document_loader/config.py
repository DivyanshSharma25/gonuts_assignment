"""Configuration for the standalone document loading pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoaderConfig:
    input_dir: Path
    output_dir: Path
    ocr_backend: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    model: str = "maternion/LightOnOCR-2"
    gemini_model: str = "gemini-3-flash-preview"
    gemini_api_key_env: str = "GEMINI_API_KEY"
    gemini_max_retries: int = 3
    gemini_retry_delay: float = 2.0
    min_text_chars: int = 200
    use_vlm: bool = True
    render_dpi: int = 150
    request_timeout: float = 300.0
