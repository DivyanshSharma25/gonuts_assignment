"""Claim verification entry point.

The LLM extracts and classifies evidence; this module validates its response
and the evidence span. Arithmetic remains a Python responsibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fitz

from llm_verifier import GeminiVerifier, LLMConfig, _fail_closed
from rag.claims import load_claims


def _load_evidence(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Evidence file {path} must contain a JSON array")
    return {
        item["claim"]["claim_id"]: item.get("evidence", [])
        for item in payload
        if isinstance(item, dict)
        and isinstance(item.get("claim"), dict)
        and isinstance(item["claim"].get("claim_id"), str)
    }


def _pdf_text(claim: dict[str, Any], papers_dir: Path) -> str | None:
    cited_name = Path(claim["cited_paper"]).name
    pdf = papers_dir / cited_name
    if not pdf.exists():
        return None
    with fitz.open(pdf) as document:
        return "\n".join(page.get_text() for page in document)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("claims_file", type=Path)
    parser.add_argument(
        "--claim-id",
        help="Verify only the claim with this opaque claim ID.",
    )
    parser.add_argument(
        "--evidence", type=Path, default=Path("outputs/retrieved_evidence.json")
    )
    parser.add_argument("--papers-dir", type=Path, default=Path("papers"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.json"))
    parser.add_argument("--log", type=Path, default=Path("outputs/llm_log.jsonl"))
    args = parser.parse_args()

    print(f"[verify] loading claims from {args.claims_file}", flush=True)
    claims = load_claims(args.claims_file)
    if args.claim_id:
        claims = [claim for claim in claims if claim["claim_id"] == args.claim_id]
        if not claims:
            parser.error(f"Claim ID {args.claim_id!r} was not found in {args.claims_file}")
    print(f"[verify] claims selected: {len(claims)}", flush=True)
    print(f"[verify] evidence file: {args.evidence}", flush=True)
    print(f"[verify] output file: {args.output}", flush=True)
    evidence = _load_evidence(args.evidence)
    verifier = GeminiVerifier(LLMConfig(), args.log)
    predictions = []
    for claim in claims:
        print(f"[verify] processing claim {claim['claim_id']}", flush=True)
        claim_evidence = evidence.get(claim["claim_id"], [])[:20]
        print(
            f"[verify] claim {claim['claim_id']}: passing "
            f"{len(claim_evidence)} retrieved chunks to LLM",
            flush=True,
        )
        if not claim_evidence:
            print(
                f"[verify] claim {claim['claim_id']}: no retrieved evidence",
                flush=True,
            )
            predictions.append(
                _fail_closed(claim["claim_id"], "No retrieved evidence was supplied.")
            )
            continue
        predictions.append(
            verifier.verify(claim, claim_evidence, _pdf_text(claim, args.papers_dir))
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"[verify] wrote predictions: {args.output}", flush=True)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
