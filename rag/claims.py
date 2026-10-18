"""Claim loading and claim-ID-driven hybrid evidence retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .hybrid_retriever import HybridRetriever, RetrievalResult


REQUIRED_CLAIM_FIELDS = {"claim_id", "text", "population", "molecule", "cited_paper"}


def load_claims(path: Path) -> list[dict[str, Any]]:
    """Load and validate the claims array used to build retrieval queries."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load claims file {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Claims file {path} must contain a JSON array")

    claims: list[dict[str, Any]] = []
    for index, claim in enumerate(payload):
        if not isinstance(claim, dict):
            raise ValueError(f"Claim at index {index} must be a JSON object")
        missing = REQUIRED_CLAIM_FIELDS - claim.keys()
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(f"Claim at index {index} is missing: {missing_fields}")
        if not all(isinstance(claim[field], str) and claim[field].strip() for field in REQUIRED_CLAIM_FIELDS):
            raise ValueError(f"Claim at index {index} has empty or non-string required fields")
        claims.append(claim)
    return claims


def matching_papers(claim_id: str, markdown_dir: Path) -> list[str]:
    """Return Markdown stems whose filenames contain the opaque claim ID."""
    normalized_id = claim_id.casefold()
    return sorted(
        source.stem
        for source in markdown_dir.glob("*.md")
        if normalized_id in source.stem.casefold()
    )


def claim_query(claim: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"Claim: {claim['text']}",
            f"Population: {claim['population']}",
            f"Molecule/comparator: {claim['molecule']}",
        )
    )


def retrieve_claims(
    claims: list[dict[str, Any]],
    retriever: HybridRetriever,
    markdown_dir: Path,
    top_k: int = 20,
    candidate_k: int = 20,
) -> list[dict[str, Any]]:
    """Retrieve evidence independently for each claim from matching papers only."""
    retrieved: list[dict[str, Any]] = []
    indexed_papers = {chunk.paper for chunk in retriever.chunks.values()}
    for claim in claims:
        matched_papers = matching_papers(claim["claim_id"], markdown_dir)
        allowed_papers = set(matched_papers) & indexed_papers
        results: list[RetrievalResult] = []
        if allowed_papers:
            results = retriever.search(
                claim_query(claim),
                top_k=top_k,
                candidate_k=candidate_k,
                allowed_papers=allowed_papers,
            )
        retrieved.append(
            {
                "claim": claim,
                "matched_papers": matched_papers,
                "indexed_papers": sorted(allowed_papers),
                "evidence": [asdict(result) for result in results],
                "reason": None
                if results
                else "No indexed evidence document matched this claim ID."
                if not allowed_papers
                else "No evidence chunks were retrieved from the matching documents.",
            }
        )
    return retrieved
