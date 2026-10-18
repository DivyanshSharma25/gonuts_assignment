"""Gemini-backed, fail-closed claim verification."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


CHECK_NAMES = ("figure", "population", "drug_indication", "actually_stated")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class ChecksModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    figure: Literal["match", "mismatch", "unclear"]
    population: Literal["match", "mismatch", "unclear"]
    drug_indication: Literal["match", "mismatch", "unclear"]
    actually_stated: Literal["match", "mismatch", "unclear"]


class LLMResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["supported", "overstated", "cannot_verify"]
    checks: ChecksModel
    evidence_page: StrictInt | None
    evidence_span: StrictStr | None
    computed: dict[str, Any] | None
    reason: StrictStr = Field(min_length=1)


class PredictionModel(LLMResponseModel):
    claim_id: StrictStr


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gemini-3-flash-preview"
    api_key_env: str = "GEMINI_API_KEY"
    base_url: str = GEMINI_BASE_URL
    timeout: float = 90.0


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding process variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fail_closed(claim_id: str, reason: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "verdict": "cannot_verify",
        "checks": {name: "unclear" for name in CHECK_NAMES},
        "evidence_page": None,
        "evidence_span": None,
        "computed": None,
        "reason": reason,
    }


def _prompt(claim: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    schema = json.dumps(LLMResponseModel.model_json_schema(), ensure_ascii=False)
    return (
        "You verify one medical promotional claim against supplied source evidence. "
        "Return JSON only. The verdict MUST be exactly one of: supported, overstated, "
        "cannot_verify. Never use partially_supported or any other verdict. Do not "
        "calculate percentages, ratios, differences, or "
        "p-values. Extract raw numbers exactly as written; Python performs arithmetic "
        "afterward. Use 'unclear' when evidence is insufficient. "
        "A supported verdict requires all four checks to be match and an evidence "
        "span copied verbatim from the source evidence. "
        "The response MUST conform to this JSON Schema exactly:\n"
        f"{schema}\n\n"
        "The computed value may contain only raw extracted values and the claimed "
        "figure; do not put a calculated result in it.\n\n"
        f"CLAIM:\n{json.dumps(claim, ensure_ascii=False)}\n\n"
        f"SOURCE EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
    )


class GeminiVerifier:
    def __init__(self, config: LLMConfig, log_path: Path) -> None:
        self.config = config
        self.log_path = log_path

    def verify(
        self,
        claim: dict[str, Any],
        evidence: list[dict[str, Any]],
        pdf_text: str | None,
    ) -> dict[str, Any]:
        claim_id = str(claim["claim_id"])
        print(f"[llm] starting claim {claim_id}", flush=True)
        load_dotenv(Path(__file__).resolve().parent / ".env")
        api_key = os.getenv(self.config.api_key_env)
        if not api_key or api_key == "your-gemini-api-key-here":
            print(
                f"[llm] claim {claim_id}: missing usable {self.config.api_key_env}",
                flush=True,
            )
            return _fail_closed(
                claim_id,
                f"Missing a usable API key. Set {self.config.api_key_env} in .env "
                "or in the process environment.",
            )
        prompt = _prompt(claim, evidence)
        request_body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a conservative evidence verifier.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "claim_verification",
                    "strict": True,
                    "schema": LLMResponseModel.model_json_schema(),
                },
            },
        }
        print(
            f"[llm] claim {claim_id}: model={self.config.model}, "
            f"evidence_chunks={len(evidence)}",
            flush=True,
        )
        print(f"[llm] claim {claim_id}: final prompt BEGIN", flush=True)
        print(prompt, flush=True)
        print(f"[llm] claim {claim_id}: final prompt END", flush=True)
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claim_id": claim_id,
            "model": self.config.model,
            "request": request_body,
        }
        try:
            print(
                f"[llm] claim {claim_id}: sending request to {self.config.base_url}",
                flush=True,
            )
            raw_body = self._request(api_key, request_body)
            record["raw_response"] = raw_body
            print(f"[llm] claim {claim_id}: raw response BEGIN", flush=True)
            print(raw_body, flush=True)
            print(f"[llm] claim {claim_id}: raw response END", flush=True)
            response = json.loads(raw_body)
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as exc:
            print(f"[llm] claim {claim_id}: response parsing failed: {exc}", flush=True)
            record["validation_success"] = False
            record["error"] = str(exc)
            self._append_log(record)
            return _fail_closed(claim_id, f"LLM response could not be parsed: {exc}")
        try:
            validated = LLMResponseModel.model_validate(parsed)
        except ValidationError as exc:
            print(
                f"[llm] claim {claim_id}: Pydantic validation failed: {exc}",
                flush=True,
            )
            record["validation_success"] = False
            record["error"] = exc.errors()
            self._append_log(record)
            return _fail_closed(claim_id, "LLM response failed the required JSON schema.")

        print(
            f"[llm] claim {claim_id}: Pydantic validation passed; "
            f"verdict={validated.verdict}",
            flush=True,
        )
        span = validated.evidence_span
        if validated.verdict == "supported":
            if not span or not pdf_text or _normalise(span) not in _normalise(pdf_text):
                print(
                    f"[llm] claim {claim_id}: evidence span verification failed",
                    flush=True,
                )
                record["validation_success"] = False
                self._append_log(record)
                return _fail_closed(
                    claim_id,
                    "The proposed evidence span was not found verbatim in the cited PDF.",
                )
            if any(getattr(validated.checks, name) != "match" for name in CHECK_NAMES):
                print(
                    f"[llm] claim {claim_id}: supported verdict had non-matching checks",
                    flush=True,
                )
                record["validation_success"] = False
                self._append_log(record)
                return _fail_closed(
                    claim_id,
                    "Supported verdict did not satisfy all four matching checks.",
                )
        record["validation_success"] = True
        self._append_log(record)
        prediction = PredictionModel(
            claim_id=claim_id,
            **validated.model_dump(),
        )
        print(
            f"[llm] claim {claim_id}: final verdict={prediction.verdict}",
            flush=True,
        )
        return prediction.model_dump()

    def _request(self, api_key: str, body: dict[str, Any]) -> str:
        print(
            f"[llm] HTTP POST {self.config.base_url.rstrip('/')}/chat/completions "
            f"(model={self.config.model})",
            flush=True,
        )
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            return response.read().decode("utf-8")

    def _append_log(self, record: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
