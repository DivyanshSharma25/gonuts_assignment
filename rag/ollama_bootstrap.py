"""Start the local Ollama service before making normal client calls."""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OllamaProcesses:
    server: subprocess.Popen[str] | None = None

    def close(self) -> None:
        if self.server is not None and self.server.poll() is None:
            print(
                "[rag] leaving Ollama server running in its separate terminal",
                flush=True,
            )


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=1):
            return True
    except (OSError, urllib.error.URLError):
        return False


def start_ollama(url: str, timeout: float = 30) -> OllamaProcesses:
    processes = OllamaProcesses()
    if not _ready(url):
        print(
            "[rag] Ollama is not reachable; starting `ollama serve` "
            "in a separate Windows terminal",
            flush=True,
        )
        try:
            processes.server = subprocess.Popen(
                ["ollama", "serve"],
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        except OSError as exc:
            raise RuntimeError(
                "Could not start Ollama in a separate terminal. Install Ollama "
                "and ensure `ollama` is on PATH."
            ) from exc
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not _ready(url):
            if processes.server.poll() is not None:
                output = processes.server.stdout.read() if processes.server.stdout else ""
                raise RuntimeError(f"`ollama serve` exited before becoming ready: {output}")
            time.sleep(0.5)
        if not _ready(url):
            raise RuntimeError(f"Ollama did not become ready at {url} within {timeout:g}s.")
        print("[rag] Ollama server is ready in its separate terminal", flush=True)
    else:
        print("[rag] Ollama server is already running", flush=True)

    print(
        "[rag] Ollama API is ready; the retriever will load the model through "
        "the normal ollama.embed call",
        flush=True,
    )
    return processes
