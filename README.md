# Claim Verification Mini-Pipeline

This is a Python pipeline for checking medical and pharmaceutical promotional
claims against evidence from clinical-trial PDFs.

It combines:

- PDF text extraction and optional visual OCR
- Page-aware Markdown generation
- Hybrid semantic and keyword evidence retrieval
- Structured LLM-based claim verification
- Verbatim evidence-span validation
- JSON predictions and LLM audit logs

## How it works

```text
PDF papers
   |
   v
Document extraction
   |
   v
Page-aware Markdown
   |
   v
Qdrant + BM25 hybrid index
   |
   v
Retrieved claim evidence
   |
   v
LLM claim verification
   |
   v
Structured predictions
```

Each claim is checked against four criteria:

1. **Figure** — the reported statistic agrees with the source.
2. **Population** — the study population matches the claim.
3. **Drug/indication** — the drug, disease, comparator, dose, and time point
   match.
4. **Actually stated** — the source explicitly supports the claim.

The verifier produces one of three verdicts:

- `supported`
- `overstated`
- `cannot_verify`

The pipeline is designed to fail closed when evidence is missing, ambiguous,
or cannot be verified.

## Requirements

- Python 3.10+
- Packages in [`requirements.txt`](requirements.txt)
- [Ollama](https://ollama.com/) for embeddings and local OCR
- `nomic-embed-text` Ollama model
- `maternion/LightOnOCR-2` Ollama model when visual OCR is enabled
- A `GEMINI_API_KEY` for Gemini OCR or claim verification

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pull the Ollama models:

```powershell
ollama pull nomic-embed-text
ollama pull materion/LightOnOCR-2
```

## Running the project

### 1. Extract PDFs

Place source PDFs in `papers/`, then run:

```powershell
python -m document_loader
```

The extracted Markdown files are written to `data/markdowns/`.

To extract only the PDF text layer:

```powershell
python -m document_loader --no-vlm
```

### 2. Build the evidence index

```powershell
python -m rag index
```

This creates:

- A Qdrant vector collection in `data/qdrant/`
- A BM25 index in `data/bm25/`

### 3. Search the evidence index

```powershell
python -m rag search "HbA1c reduction versus basal insulin"
```

To search a specific indexed paper:

```powershell
python -m rag search "hypoglycemia odds ratio" --paper C2C4C6
```

### 4. Retrieve evidence for claims

Claims are stored in [`data/claims.json`](data/claims.json). Retrieve evidence
for all claims with:

```powershell
python -m rag claims data/claims.json
```

The results are written to `outputs/retrieved_evidence.json`.

Each claim contains:

```json
{
  "claim_id": "C3",
  "text": "Claim wording",
  "population": "Study population",
  "molecule": "Drug or regimen",
  "cited_paper": "papers/source.pdf"
}
```

### 5. Verify claims

Set the Gemini API key:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

Run verification:

```powershell
python verify.py data/claims.json
```

The verifier writes:

- `outputs/predictions.json`
- `outputs/llm_log.jsonl`

To verify a single claim:

```powershell
python verify.py data/claims.json --claim-id C3
```

## Prediction format

```json
{
  "claim_id": "C3",
  "verdict": "supported",
  "checks": {
    "figure": "match",
    "population": "match",
    "drug_indication": "match",
    "actually_stated": "match"
  },
  "evidence_page": 9,
  "evidence_span": "Verbatim text from the source PDF",
  "computed": null,
  "reason": "Explanation for the verdict."
}
```

## Project structure

```text
document_loader/       PDF extraction and OCR
rag/                   Hybrid evidence retrieval
papers/                Source PDF documents
data/claims.json       Claims to verify
data/markdowns/        Extracted Markdown documents
data/qdrant/           Qdrant vector index
data/bm25/             BM25 index
outputs/               Evidence, predictions, and logs
verify.py              Claim verification entry point
eval.py                Evaluation entry point
llm_verifier.py        LLM response validation and verification logic
PROJECT_CONTEXT.md     Detailed project rules and schemas
WRITEUP.md             Project write-up template
```

## Notes

- Do not commit API keys or other credentials.
- Rebuild the indexes after changing source PDFs or extraction settings.
- Review OCR output before using it as evidence.
- The `eval.py` script is reserved for comparing predictions with curated
  labels in `data/gold.json`.
