"""Qdrant and BM25 hybrid retrieval over page-aware Markdown evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

import ollama
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi


PAGE_RE = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.%/-][a-z0-9]+)*", re.IGNORECASE)
DEFAULT_MAX_CHUNK_CHARS = 4000


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    paper: str
    page: int
    section: str | None
    extraction_type: str
    text: str
    source_markdown: str


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    paper: str
    page: int
    section: str | None
    extraction_type: str
    text: str
    source_markdown: str
    cosine_score: float
    bm25_score: float
    combined_score: float


def tokenize(text: str) -> list[str]:
    """Tokenize words, measurements, percentages, and statistical values."""
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _chunk_id(
    paper: str,
    page: int,
    section: str | None,
    extraction_type: str,
    text: str,
) -> str:
    identity = f"{paper}\0{page}\0{section or ''}\0{extraction_type}\0{text}"
    return str(uuid5(NAMESPACE_URL, identity))


def _split_block(text: str, max_chars: int) -> list[str]:
    """Split a block while enforcing a hard character limit."""
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        boundary = remaining.rfind("\n", 0, max_chars + 1)
        if boundary <= 0:
            boundary = remaining.rfind(" ", 0, max_chars + 1)
        if boundary <= 0:
            boundary = max_chars
        parts.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts


def _parse_page_blocks(
    markdown: str,
    paper: str,
    source: Path,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[EvidenceChunk]:
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than zero")
    page_matches = list(PAGE_RE.finditer(markdown))
    chunks: list[EvidenceChunk] = []
    for index, page_match in enumerate(page_matches):
        page = int(page_match.group(1))
        page_start = page_match.end()
        page_end = page_matches[index + 1].start() if index + 1 < len(page_matches) else len(markdown)
        page_text = markdown[page_start:page_end].strip()
        current_section: str | None = None
        blocks: list[str] = []
        for block in re.split(r"\n\s*\n", page_text):
            block = block.strip()
            if not block:
                continue
            heading = HEADING_RE.fullmatch(block)
            if heading:
                current_section = heading.group(2).strip()
                continue
            if block.startswith("### "):
                current_section = block[4:].strip()
                continue
            blocks.append(block)

        for block in blocks:
            extraction_type = "visual" if "### Visual extraction" in block else "text"
            clean_block = re.sub(r"^### (?:Text|Visual extraction).*?\n", "", block, count=1)
            clean_block = clean_block.strip()
            if not clean_block:
                continue
            for chunk_text in _split_block(clean_block, max_chunk_chars):
                chunk_id = _chunk_id(
                    paper,
                    page,
                    current_section,
                    extraction_type,
                    chunk_text,
                )
                chunks.append(
                    EvidenceChunk(
                        chunk_id=chunk_id,
                        paper=paper,
                        page=page,
                        section=current_section,
                        extraction_type=extraction_type,
                        text=chunk_text,
                        source_markdown=str(source),
                    )
                )
    return chunks


def load_chunks(
    markdown_dir: Path,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    seen_ids: set[str] = set()
    sources = sorted(markdown_dir.glob("*.md"))
    print(f"[rag] found {len(sources)} Markdown files", flush=True)
    for source in sources:
        file_chunks = _parse_page_blocks(
            source.read_text(encoding="utf-8"),
            source.stem,
            source,
            max_chunk_chars=max_chunk_chars,
        )
        unique_file_chunks: list[EvidenceChunk] = []
        for chunk in file_chunks:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            unique_file_chunks.append(chunk)
        chunks.extend(unique_file_chunks)
        print(
            f"[rag] parsed {source.name}: {len(unique_file_chunks)} chunks "
            f"({len(file_chunks) - len(unique_file_chunks)} duplicates skipped)",
            flush=True,
        )
    if not chunks:
        raise ValueError(f"No page-aware Markdown chunks found in {markdown_dir}")
    print(f"[rag] total chunks: {len(chunks)}", flush=True)
    return chunks


class HybridRetriever:
    """Persistent 50/50 cosine and BM25 retriever."""

    def __init__(
        self,
        qdrant_path: Path,
        bm25_path: Path,
        collection_name: str = "fractiva_evidence",
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.qdrant_path = qdrant_path
        self.bm25_path = bm25_path
        self.collection_name = collection_name
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.ollama = ollama.Client(host=ollama_url)
        self.client = QdrantClient(path=str(qdrant_path))
        self.chunks: dict[str, EvidenceChunk] = {}
        self.bm25: BM25Okapi | None = None
        self._warm_up_model()
        self._load_bm25()
        print(
            f"[rag] retriever ready; loaded {len(self.chunks)} BM25 chunks",
            flush=True,
        )

    def _warm_up_model(self) -> None:
        print(f"[rag] loading Ollama model {self.embedding_model!r}", flush=True)
        self.ollama.embed(
            model=self.embedding_model,
            input="warm up embedding model",
            keep_alive=-1,
        )
        print(f"[rag] Ollama model {self.embedding_model!r} is ready", flush=True)

    def _load_bm25(self) -> None:
        metadata_path = self.bm25_path.with_suffix(".json")
        if not self.bm25_path.exists() or not metadata_path.exists():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunk_list = [EvidenceChunk(**item) for item in metadata["chunks"]]
        self.chunks = {chunk.chunk_id: chunk for chunk in chunk_list}
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunk_list])
        print(f"[rag] restored BM25 metadata: {len(chunk_list)} chunks", flush=True)

    def _embed_one(self, text: str) -> list[float]:
        try:
            response = self.ollama.embed(
                model=self.embedding_model,
                input=text,
                keep_alive=-1,
            )
        except ollama.ResponseError as exc:
            sanitized = text.replace("|", "")
            if sanitized == text:
                raise
            try:
                response = self.ollama.embed(
                    model=self.embedding_model,
                    input=sanitized,
                    keep_alive=-1,
                )
            except ollama.ResponseError as sanitized_exc:
                raise RuntimeError(
                    "Ollama rejected an extracted text chunk while creating an "
                    f"embedding: {sanitized_exc}"
                ) from exc
        embeddings = response.get("embeddings")
        if not embeddings:
            raise RuntimeError(
                f"Ollama returned no embeddings for model {self.embedding_model!r}"
            )
        return embeddings[0]

    def index(
        self,
        markdown_dir: Path,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    ) -> int:
        chunks = load_chunks(markdown_dir, max_chunk_chars=max_chunk_chars)
        embeddings: list[list[float]] = []
        print(
            f"[rag] embedding {len(chunks)} chunks one by one",
            flush=True,
        )
        try:
            for index, chunk in enumerate(chunks, start=1):
                print(
                    f"[rag] embedding chunk {index}/{len(chunks)}",
                    flush=True,
                )
                try:
                    embeddings.append(self._embed_one(chunk.text))
                except Exception:
                    print(
                        f"[rag] failed chunk {index}/{len(chunks)}: "
                        f"paper={chunk.paper!r}, page={chunk.page}, "
                        f"section={chunk.section!r}",
                        flush=True,
                    )
                    print(f"[rag] failed chunk text: {chunk.text!r}", flush=True)
                    raise
        except Exception as exc:
            raise RuntimeError(
                f"Could not create embeddings with Ollama model {self.embedding_model!r}. "
                f"Endpoint: {self.ollama_url}. "
                "Confirm Ollama is running and the model is pulled. "
                f"Original error: {exc}"
            ) from exc
        if len(embeddings) != len(chunks):
            raise RuntimeError("Ollama returned a different number of embeddings than chunks")

        print(f"[rag] received {len(embeddings)} embeddings", flush=True)
        if self.client.collection_exists(self.collection_name):
            print(f"[rag] replacing Qdrant collection {self.collection_name}", flush=True)
            self.client.delete_collection(self.collection_name)
        print(f"[rag] creating Qdrant collection {self.collection_name}", flush=True)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE),
        )
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload=asdict(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        print(f"[rag] storing {len(points)} vectors in Qdrant", flush=True)
        self.client.upsert(collection_name=self.collection_name, points=points)

        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
        print("[rag] building BM25 index", flush=True)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25_path.write_bytes(
            hashlib.sha256("".join(chunk.chunk_id for chunk in chunks).encode()).digest()
        )
        self.bm25_path.with_suffix(".json").write_text(
            json.dumps({"chunks": [asdict(chunk) for chunk in chunks]}, indent=2),
            encoding="utf-8",
        )
        print(f"[rag] persisted BM25 metadata to {self.bm25_path.with_suffix('.json')}", flush=True)
        return len(chunks)

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        low, high = min(scores.values()), max(scores.values())
        if high == low:
            return {key: 1.0 for key in scores}
        return {key: (value - low) / (high - low) for key, value in scores.items()}

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int = 20,
        paper: str | None = None,
        allowed_papers: set[str] | None = None,
    ) -> list[RetrievalResult]:
        if self.bm25 is None or not self.chunks:
            raise RuntimeError("The index is empty. Run the index command first.")
        if paper and allowed_papers is not None:
            raise ValueError("Use either paper or allowed_papers, not both")
        paper_filter = allowed_papers if allowed_papers is not None else {paper} if paper else None
        if paper_filter is not None and not paper_filter:
            return []
        print("[rag] embedding query", flush=True)
        dense_vector = self._embed_one(query)
        dense_limit = max(candidate_k, len(self.chunks)) if paper_filter is not None else candidate_k
        print(f"[rag] querying Qdrant for {dense_limit} candidates", flush=True)
        dense_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            limit=dense_limit,
            with_payload=True,
        ).points
        dense_scores = {
            str(hit.id): float(hit.score)
            for hit in dense_hits
            if str(hit.id) in self.chunks
            and (paper_filter is None or hit.payload.get("paper") in paper_filter)
        }

        print(f"[rag] Qdrant returned {len(dense_scores)} candidates", flush=True)

        print("[rag] scoring BM25 candidates", flush=True)
        query_scores = self.bm25.get_scores(tokenize(query))
        bm25_scores = {
            chunk.chunk_id: float(query_scores[index])
            for index, chunk in enumerate(self.chunks.values())
            if paper_filter is None or chunk.paper in paper_filter
        }
        dense_norm = self._normalize(dense_scores)
        bm25_norm = self._normalize(bm25_scores)
        candidate_ids = set(dense_scores) | set(sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:candidate_k])
        ranked = sorted(
            candidate_ids,
            key=lambda chunk_id: 0.5 * dense_norm.get(chunk_id, 0.0) + 0.5 * bm25_norm.get(chunk_id, 0.0),
            reverse=True,
        )[:top_k]
        print(f"[rag] fused {len(candidate_ids)} candidates; returning {len(ranked)}", flush=True)
        return [
            RetrievalResult(
                **asdict(self.chunks[chunk_id]),
                cosine_score=dense_scores.get(chunk_id, 0.0),
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                combined_score=0.5 * dense_norm.get(chunk_id, 0.0) + 0.5 * bm25_norm.get(chunk_id, 0.0),
            )
            for chunk_id in ranked
        ]
