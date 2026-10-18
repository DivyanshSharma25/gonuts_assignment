"""Small Ollama client for visual PDF-page transcription."""

from dataclasses import dataclass

import ollama


class VLMError(RuntimeError):
    """Raised when Ollama cannot return a valid visual extraction."""


@dataclass
class OllamaVLM:
    base_url: str
    model: str
    timeout: float = 120.0

    def __post_init__(self) -> None:
        self.client = ollama.Client(host=self.base_url, timeout=self.timeout)

    def extract_page(self, image_base64: str, page_number: int) -> str:
        print(
            f"[vlm] requesting page {page_number} from {self.base_url} "
            f"using model {self.model}",
            flush=True,
        )
        prompt = (
            "Transcribe the useful content visible on this PDF page into Markdown. "
            "Preserve wording, numbers, symbols, footnotes, and table relationships. "
            "Represent tables as Markdown tables when possible. Include chart labels "
            "and legends. Do not infer, summarize, correct, or invent anything. "
            "Return only the transcription. This is page "
            f"{page_number}."
        )
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_base64],
                    }
                ],
                stream=False,
                think=False,
                options={"temperature": 0},
            )
        except (ollama.RequestError, ollama.ResponseError) as exc:
            print(f"[vlm] page {page_number} request failed: {exc}", flush=True)
            raise VLMError(f"Ollama request failed for page {page_number}: {exc}") from exc

        content = response.message.content
        if not content or not content.strip():
            print(f"[vlm] page {page_number} returned empty content", flush=True)
            raise VLMError(f"Ollama returned no usable content for page {page_number}")
        print(
            f"[vlm] page {page_number} received {len(content.strip())} characters",
            flush=True,
        )
        return content.strip()
