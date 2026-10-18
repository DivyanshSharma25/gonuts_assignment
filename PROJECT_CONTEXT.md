# Fractiva Claim Verification Mini-Pipeline — Project Context

This document is the source of truth for architecture, domain rules, schemas, and engineering constraints. Implementation work must follow these rules. Do not invent additional verification criteria or relax fail-closed behavior.

---

## A. Project Goal

We are building an automated medical/pharma promotional claim verifier.

Promotional leave-behinds (notably `papers/SOLIQUA_LBL.pdf`) make statistical and clinical claims about drugs and cite clinical trial papers (`papers/*.pdf`). The pipeline:

1. **`verify.py`** takes a JSON list of claims (`data/claims.json` or any path passed on the CLI), checks each claim against its cited PDF(s), and writes structured verdicts to `outputs/predictions.json`.
2. **`eval.py`** compares `outputs/predictions.json` against manually curated ground-truth labels in `data/gold.json`.

The product is a *mini-pipeline*: extract evidence from cited papers, reconcile numbers in Python (never in the LLM), apply four verification checks, and emit a fail-closed verdict a human reviewer can act on.

---

## B. The Four Verification Criteria (`checks`)

For every claim, the verifier evaluates **four** dimensions. Each check is marked exactly one of:

- `"match"`
- `"mismatch"`
- `"unclear"`

| Key | Meaning |
|-----|---------|
| `figure` | The number/statistic in the claim reconciles with the paper's own data (after Python arithmetic on extracted raw numbers). |
| `population` | The study patient population matches the claim's population. |
| `drug_indication` | Same disease, molecule, form, dose, comparator, and time point as claimed. |
| `actually_stated` | The paper **explicitly** states the claim. Not an inference, extrapolation, pooled/generalized subgroup result, or marketing paraphrase. |

Do not add extra check keys. Do not omit any of the four.

---

## C. Verdict Rules ("Fail Closed" Priority)

Allowed verdicts: `"supported"` | `"overstated"` | `"cannot_verify"`.

### `supported`

ALL of the following must hold:

- All four checks are `"match"`.
- A verbatim `evidence_span` is present and programmatically verified as a (whitespace- and order-tolerant) substring of the source PDF.

### `overstated`

Any check is `"mismatch"` — the claim exaggerates or goes beyond what the data shows (wrong magnitude, wrong population, wrong indication/time point, or not actually stated).

### `cannot_verify`

Use when the source is silent, paywalled/abstract-only without the granular data, ambiguous, or a table cannot be cleanly parsed. Also use for missing API keys (see constraints), missing PDFs, and failed evidence-span verification.

### Golden Rule

**Never guess. Never give a claim the benefit of the doubt.**

A false `"supported"` is the worst possible error. When unsure, fail closed to `"cannot_verify"` with a clear `reason`.

Priority when combining signals:

1. If evidence is missing, ambiguous, unparseable, or the span cannot be verified → `cannot_verify` (do not claim `supported`).
2. If any check is `"mismatch"` and evidence is sufficient to justify that mismatch → `overstated`.
3. `supported` only in the strict all-match + verified-span case.

---

## D. Non-Negotiable Engineering Constraints

### 1. All Arithmetic in Python

An LLM may locate and label **raw numbers** from text/tables. It must **never** compute a percentage, difference, ratio, fold-change, or p-value comparison. Python code performs all math. Store raw extracted values and Python-computed results in `computed`.

### 2. Fail Closed

Missing evidence, ambiguous matches, or unparseable tables must default to `cannot_verify` with a clear `reason`. Do not emit `supported` on incomplete evidence.

### 3. Verbatim Evidence

`evidence_span` must be an **exact substring** appearing in the source PDF. Check this programmatically with whitespace- and order-tolerance (normalize whitespace; do not rewrite wording). If the span is not found in the PDF text, the claim cannot be `supported`.

### 4. Structured LLM Output

All LLM calls must request structured JSON. Validate every response in code (e.g. Pydantic). Invalid or schema-violating responses → treat as unverifiable, log the raw payload, fail closed.

### 5. Generalizability (Zero Hardcoding)

`verify.py` must accept **any** claims file path:

```text
python verify.py <path_to_claims.json>
```

It must run on unseen claims against the same papers **without** hardcoded rules for `C1`–`C7` (or any claim ID). Claim IDs are opaque identifiers.

### 6. Full LLM Audit Logging

Every LLM request (prompt + parameters) and raw response must be **appended** to `outputs/llm_log.jsonl` (JSON Lines, one object per call). Never overwrite the log as a single JSON document.

### 7. Graceful Missing API Key Handling

If the API key is missing, `verify.py` must either:

- exit with a clear message, or
- return `cannot_verify` for every claim,

and **never** crash with a raw traceback or return `supported`.

---

## E. Data Schemas

Use `null` where a field does not apply (e.g. no page, no computed comparison).

### Input Claim Schema (`data/claims.json`)

A JSON **array** of claim objects:

```json
{
  "claim_id": "C3",
  "text": "~2x greater HbA1c reduction vs basal insulin (iGlar), p<0.0001, to Week 30",
  "population": "T2DM inadequately controlled on oral agents",
  "molecule": "iGlarLixi (insulin glargine + lixisenatide)",
  "cited_paper": "papers/LixiLan-O_Rosenstock2016.pdf"
}
```

| Field | Type | Notes |
|-------|------|--------|
| `claim_id` | string | Opaque ID (e.g. `C3`). Do not special-case IDs. |
| `text` | string | Promotional claim wording. |
| `population` | string | Claimed patient population. |
| `molecule` | string | Claimed molecule / regimen. |
| `cited_paper` | string | Path to cited PDF, typically under `papers/`. |

### Output Prediction Schema (`outputs/predictions.json`) and Gold Schema (`data/gold.json`)

Both are JSON **arrays** of objects with the same shape. Gold is manually curated; predictions are written by `verify.py`.

```json
{
  "claim_id": "C3",
  "verdict": "overstated",
  "checks": {
    "figure": "mismatch",
    "population": "match",
    "drug_indication": "match",
    "actually_stated": "mismatch"
  },
  "evidence_page": 9,
  "evidence_span": "verbatim text from the PDF",
  "computed": {
    "claimed": "~2x",
    "from_source": [1.63, 1.34],
    "result": "1.22x"
  },
  "reason": "One or two sentences a human reviewer can act on."
}
```

| Field | Type | Notes |
|-------|------|--------|
| `claim_id` | string | Must match the input claim. |
| `verdict` | string | `supported` \| `overstated` \| `cannot_verify`. |
| `checks.figure` | string | `match` \| `mismatch` \| `unclear`. |
| `checks.population` | string | `match` \| `mismatch` \| `unclear`. |
| `checks.drug_indication` | string | `match` \| `mismatch` \| `unclear`. |
| `checks.actually_stated` | string | `match` \| `mismatch` \| `unclear`. |
| `evidence_page` | int or null | 1-based page if known. |
| `evidence_span` | string or null | Verbatim PDF substring; required for `supported`. |
| `computed` | object or null | Python math: claimed figure, raw source numbers, computed result. |
| `reason` | string | One or two sentences a human reviewer can act on. |

`computed` example fields (extensible, not hardcoded per claim):

- `claimed`: the figure as stated in the claim (string or number).
- `from_source`: raw numbers extracted from the paper (list of numbers).
- `result`: Python-computed comparison (e.g. `"1.22x"`).

---

## F. CLI and Artifacts (for later implementation)

| Artifact | Role |
|----------|------|
| `python verify.py data/claims.json` | Run verification; write `outputs/predictions.json`; append `outputs/llm_log.jsonl`. |
| `python eval.py` | Compare predictions vs `data/gold.json`; print/report metrics. |
| `papers/` | Source PDFs (leave-behind + clinical studies). Do not overwrite existing PDFs. |
| `data/claims.json` | Input claims. Do not overwrite if already present. |
| `data/gold.json` | Ground-truth labels (same schema as predictions). |
| `outputs/predictions.json` | Verifier output array. |
| `outputs/llm_log.jsonl` | Append-only LLM request/response audit log. |
| `WRITEUP.md` | Human report with the seven required headings. |

### Suggested `llm_log.jsonl` record (implementation later)

Each line is one JSON object, conceptually:

- timestamp
- claim_id (if applicable)
- model and request parameters
- prompt / messages
- raw response body
- validation success/failure

---

## G. Implementation Guardrails (scaffolding reminder)

This repository is scaffolded **without** functional verification, LLM parsing, or evaluation logic. Subsequent steps must:

- Keep arithmetic in Python.
- Fail closed.
- Avoid claim-ID-specific rules.
- Validate LLM JSON in code.
- Log every LLM call.
- Handle missing API keys without tracebacks or false `supported` verdicts.
