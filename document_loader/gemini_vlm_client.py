"""Small Gemini client for visual PDF-page transcription."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .vlm_client import VLMError


@dataclass
class GeminiVLM:
    model: str
    api_key_env: str = "GEMINI_API_KEY"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 2.0

    def extract_page(self, image_base64: str, page_number: int) -> str:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must not be negative")
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise VLMError(
                f"Missing {self.api_key_env}; set it before using the Gemini OCR backend"
            )
        prompt = (
            "Transcribe the useful content visible on this PDF page into Markdown. "
            "Preserve wording, numbers, symbols, footnotes, and table relationships. "
            "Represent tables as Markdown tables when possible. Include chart labels "
            "and legends. Do not infer, summarize, correct, or invent anything. "
            "Return only the transcription. This is page "
            f"{page_number}."
        )
        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        print(
            f"[vlm] requesting page {page_number} from Gemini using model {self.model}",
            flush=True,
        )
        payload: dict[str, object] | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt >= self.max_retries:
                    raise VLMError(
                        f"Gemini request failed for page {page_number}: {exc}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else self.retry_delay * (2**attempt)
                except ValueError:
                    delay = self.retry_delay * (2**attempt)
                delay = max(0.0, min(delay, 60.0))
                print(
                    f"[vlm] page {page_number}: Gemini rate limited (429); "
                    f"retrying in {delay:g}s ({attempt + 1}/{self.max_retries})",
                    flush=True,
                )
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise VLMError(f"Gemini request failed for page {page_number}: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise VLMError(
                    f"Gemini returned an invalid response for page {page_number}: {exc}"
                ) from exc
        if payload is None:
            raise VLMError(f"Gemini returned no response for page {page_number}")
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMError(
                f"Gemini returned an invalid response for page {page_number}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise VLMError(
                f"Gemini returned an invalid response for page {page_number}: {exc}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise VLMError(f"Gemini returned no usable content for page {page_number}")
        print(
            f"[vlm] page {page_number} received {len(content.strip())} characters",
            flush=True,
        )
        return content.strip()
