"""CLI for building and querying the persistent hybrid evidence index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .claims import load_claims, retrieve_claims
from .hybrid_retriever import DEFAULT_MAX_CHUNK_CHARS, HybridRetriever
from .ollama_bootstrap import start_ollama


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/query the Qdrant + BM25 evidence index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Embed and persist Markdown evidence.")
    index_parser.add_argument("--markdown-dir", type=Path, default=Path("data/markdowns"))
    index_parser.add_argument("--qdrant-path", type=Path, default=Path("data/qdrant"))
    index_parser.add_argument("--bm25-path", type=Path, default=Path("data/bm25/index.bin"))
    index_parser.add_argument("--ollama-url", default="http://localhost:11434")
    index_parser.add_argument("--model", default="nomic-embed-text")
    index_parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help=f"Hard maximum characters per chunk (default: {DEFAULT_MAX_CHUNK_CHARS}).",
    )

    search_parser = subparsers.add_parser("search", help="Run a hybrid evidence search.")
    search_parser.add_argument("query")
    search_parser.add_argument("--paper")
    search_parser.add_argument("--top-k", type=int, default=20)
    search_parser.add_argument("--qdrant-path", type=Path, default=Path("data/qdrant"))
    search_parser.add_argument("--bm25-path", type=Path, default=Path("data/bm25/index.bin"))
    search_parser.add_argument("--ollama-url", default="http://localhost:11434")
    search_parser.add_argument("--model", default="nomic-embed-text")

    claims_parser = subparsers.add_parser(
        "claims", help="Retrieve evidence for every claim in a JSON claims file."
    )
    claims_parser.add_argument("claims_file", type=Path)
    claims_parser.add_argument("--markdown-dir", type=Path, default=Path("data/markdowns"))
    claims_parser.add_argument("--output", type=Path, default=Path("outputs/retrieved_evidence.json"))
    claims_parser.add_argument("--top-k", type=int, default=20)
    claims_parser.add_argument("--candidate-k", type=int, default=20)
    claims_parser.add_argument("--qdrant-path", type=Path, default=Path("data/qdrant"))
    claims_parser.add_argument("--bm25-path", type=Path, default=Path("data/bm25/index.bin"))
    claims_parser.add_argument("--ollama-url", default="http://localhost:11434")
    claims_parser.add_argument("--model", default="nomic-embed-text")
    args = parser.parse_args()

    print(f"[rag] command={args.command}", flush=True)
    print(f"[rag] embedding model={args.model}", flush=True)
    print(f"[rag] qdrant path={args.qdrant_path}", flush=True)
    print(f"[rag] BM25 path={args.bm25_path}", flush=True)
    ollama_processes = start_ollama(args.ollama_url)
    try:
        retriever = HybridRetriever(
            qdrant_path=args.qdrant_path,
            bm25_path=args.bm25_path,
            ollama_url=args.ollama_url,
            embedding_model=args.model,
        )
        if args.command == "index":
            print(f"[rag] loading Markdown from {args.markdown_dir}", flush=True)
            count = retriever.index(
                args.markdown_dir,
                max_chunk_chars=args.max_chunk_chars,
            )
            print(f"Indexed {count} chunks into {args.qdrant_path}.")
            return 0

        if args.command == "claims":
            claims = load_claims(args.claims_file)
            evidence = retrieve_claims(
                claims,
                retriever,
                args.markdown_dir,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            print(f"[rag] wrote claim evidence to {args.output}", flush=True)
            return 0

        print(f"[rag] query={args.query!r}", flush=True)
        if args.paper:
            print(f"[rag] filtering paper={args.paper}", flush=True)
        print(f"[rag] retrieving top {args.top_k} results", flush=True)
        results = retriever.search(args.query, top_k=args.top_k, paper=args.paper)
        print(f"[rag] retrieved {len(results)} results", flush=True)
        print(json.dumps([result.__dict__ for result in results], indent=2))
        return 0
    finally:
        ollama_processes.close()


if __name__ == "__main__":
    raise SystemExit(main())
