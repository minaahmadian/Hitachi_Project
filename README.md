# Hitachi Safety Assessment Pipeline

An AI-assisted pre-ISA (Independent Safety Assessment) pipeline for railway software certification under **CEI EN 50128**. The system cross-checks declared requirement traces against test-design evidence, communication records, and regulatory rules, then produces an explainable release recommendation with full audit trail.

---

## What It Does

Given a set of project artifacts — a test-design Word document (RSSOM/FIT), a requirements trace CSV, structured test logs, email threads, and authorization records — the pipeline:

1. **Checks traceability** — verifies each declared requirement ID against the FIT corpus using exact line-local evidence and a semantic RAG fallback
2. **Scans governance** — detects derogation/waiver language in emails near anomaly-linked requirement IDs
3. **Triages communications** — an LLM flags suspicious patterns in the communication record
4. **Evaluates regulatory compliance** — applies CEI EN 50128 rule checks and retrieves the most relevant clauses from a local Qdrant index
5. **Compiles a pre-ISA report** — consolidates all findings into one explainable JSON bundle with per-anomaly verdicts and evidence citations
6. **Issues a release decision** — `GO` or `NO-GO` with full rationale

All outputs are persisted to `output/vdd_last_run.json` and optionally rendered as a Word VDD draft.

---

## Pipeline Architecture

```
RSSOM .docx ──► parse_docx ──► corpus ──► [Disk cache] ──► RSSOM RAG index
                                   │                              │
requirements_trace.csv ────────► GraphState ──────────────► traceability_matcher
test_logs.json ────────────────►   │                              │
email_threads.txt ─────────────►   │                         anomaly_envelope
authorizations.txt ────────────►   │                              │
                                   ▼                              ▼
                              formal_auditor          regulatory_assessor ◄── Qdrant (CEI EN 50128)
                                   │
                            derogation_context
                                   │
                            context_detective
                                   │
                            pre_isa_compiler ◄── all reports
                                   │
                            lead_assessor
                            ┌──────┴──────┐
                    vdd_last_run.json    VDD .docx
```

### Two RAG Retrieval Legs

| Leg | Index | Query source | Purpose |
|-----|-------|--------------|---------|
| **RSSOM traceability** | Requirement-centric in-memory index (disk-cached) | `requirement_id` + title → evidence-aware reranker | Semantic fallback when exact line-scan is weak |
| **Regulatory clauses** | Qdrant CEI EN 50128 collection | `anomaly_envelope.retrieval_query` → anomaly-type-aware fallback | Retrieve the most relevant standard clauses for current anomalies |

RSSOM retrieval uses **proximity-based verdict derivation** (searches for PASS/FAIL tokens within ±320 chars of the requirement ID) and a **post-retrieval reranker** (+0.30 for exact metadata match, +0.15 for ID in text). The index is **hash-keyed and cached to disk** — subsequent runs skip re-embedding entirely (~700ms → ~100ms).

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for the Qdrant regulatory clause index)
- Groq API key (for LLM and embeddings)

### Setup

```bash
git clone https://github.com/ManInBlackout/hitachi-ai-backend.git
cd hitachi-ai-backend
python -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then add GROQ_API_KEY
```

### Run the pipeline

```bash
export VDD_AUDIT=1
venv/bin/python main.py
```

Output: `output/vdd_last_run.json` + console summary.

### Run an eval scenario

```bash
export VDD_AUDIT=1
export REQUIREMENTS_TRACE_CSV=data/eval/scenarios/baseline_legacy_trace.csv
venv/bin/python main.py
venv/bin/python scripts/evaluate_against_golden.py \
  --audit output/vdd_last_run.json \
  --golden data/eval/golden/scenario_baseline.json
```

### Benchmark RSSOM retrieval quality

```bash
venv/bin/python scripts/benchmark_rssom_retrieval.py
```

Runs against a 12-query labeled set and reports MRR, Hit@K, Precision@K, Recall@K.

---

## Project Structure

```
agents/              LangGraph node implementations
  formal_auditor.py       LLM compliance analysis on RSSOM excerpt
  regulatory_assessor.py  CEI EN 50128 rule engine + anomaly-type-aware clause RAG
  context_detective.py    LLM communication triage
  lead_assessor.py        Final GO/NO-GO decision node

traceability/
  matcher.py              Deterministic CSV↔corpus↔log cross-check
  rssom_rag.py            Requirement-centric RSSOM vector index (cache + reranker)
  anomaly_envelope.py     Converts matcher output into RAG-ready retrieval query
  derogation_scan.py      Email/authorization derogation scanner

regulatory/
  rule_engine.py          CEI EN 50128 deterministic rule checks
  clause_retrieval.py     Qdrant-backed regulatory clause RAG
  build_regulatory_index.py  Index builder for the clause collection

core/
  project_ingestion.py    Data loading and corpus builder
  audit_export.py         Frozen JSON audit artifact writer
  rssom_requirements_trace.py  Extract/merge requirement IDs from RSSOM .docx

scripts/
  benchmark_rssom_retrieval.py   RSSOM retrieval quality benchmark (MRR, Hit@K)
  benchmark_regulatory_retrieval.py  Regulatory clause retrieval benchmark
  evaluate_against_golden.py     Eval accuracy script (deterministic + soft checks)
  run_eval_scenarios.sh          Run all three eval scenarios in sequence

data/
  RSSOM_APCS_FIT.docx            Primary test-design evidence document
  requirements_trace.csv         Declared requirement claims for the current run
  test_logs.json                 Structured test execution results
  email_threads.txt              Communication record (18 blocks)
  eval/scenarios/                Controlled input CSVs for each eval scenario
  eval/golden/                   Frozen expected outputs per scenario
  eval/retrieval/                Golden query sets for RAG benchmarking
  regulatory/                    CEI EN 50128 rule definitions

output/
  vdd_last_run.json              Latest pipeline audit artifact
  rag_cache/                     Hash-keyed RSSOM embedding cache files

docs/
  pipeline-overview.md           Full pipeline walkthrough with Mermaid diagrams
```

---

## Eval Accuracy

Accuracy in this project is **not classification accuracy**. It is the agreement rate between a pipeline audit JSON and frozen golden expectations, split into:

| Check type | What it compares | Counts toward pass |
|---|---|---|
| `expected_deterministic` | `matcher_report.status`, `pre_isa_report.overall`, etc. | Yes |
| `anomaly_requirement_id_checks` | `must_include` / `must_not_include` IDs in anomalies | Yes |
| `expected_soft` | LLM-sensitive outputs (assessor rationale, detective status) | Optional (`--strict-soft`) |

Four scenarios are provided:

| Scenario | CSV | Expected anomaly behavior |
|---|---|---|
| A | `scenario_a_hit_fake.csv` | `HIT-FAKE-999` triggers MISSING_TEST_DESIGN_EVIDENCE |
| B | `scenario_b_no_hit_fake.csv` | No anomaly cites `HIT-FAKE-999` |
| C | `scenario_c_extra_conflict_row.csv` | Both `HIT-FAKE-999` and `C6-CONFLICT-DEMO` in anomalies |
| Baseline | `baseline_legacy_trace.csv` | 24-row RSSOM-enriched trace; stable reference |

Run all three in sequence:

```bash
export VDD_AUDIT=1
chmod +x scripts/run_eval_scenarios.sh
./scripts/run_eval_scenarios.sh
```

---

## Retrieval Quality

| Metric | 5-query set (direct) | 12-query set (with paraphrases) |
|--------|----------------------|---------------------------------|
| MRR | 1.0 | 0.77 |
| Hit@1 | 1.0 | 0.58 |
| Hit@3 | 1.0 | 0.92 |
| Hit@5 | 1.0 | **1.0** |
| Recall@5 | 1.0 | **1.0** |

The gap between Hit@1 and Hit@5 on the harder set is caused by semantically identical requirement titles (e.g., multiple requirements share the phrase *"vehicle crowding levels visualization on RTM section"*). In the runtime pipeline this is resolved by the evidence-aware reranker, which uses the exact `requirement_id` to place the correct entry at rank 1.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for LLM and embeddings | **required** |
| `EMBEDDING_PROVIDER` | `groq`, `local`, or `hash` | `groq` |
| `RSSOM_RAG_CACHE_DIR` | Directory for RSSOM embedding cache | `output/rag_cache` |
| `RSSOM_RAG_TOP_K` | Top-k hits for RSSOM semantic retrieval | `3` |
| `RSSOM_RAG_ENTRY_CHUNK_SIZE` | Chunk size for requirement entries | `1400` |
| `AUDITOR_MAX_DOC_CHARS` | Max RSSOM chars sent to the auditor LLM | `12000` |
| `REGULATORY_RAG` | `auto`, `1` (force on), `0` (force off) | `auto` |
| `REGULATORY_RAG_TOP_K` | Top-k regulatory clause hits | `5` |
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `VDD_AUDIT` | Set to `1` to write `vdd_last_run.json` | off |
| `REQUIREMENTS_TRACE_CSV` | Override requirements CSV path | `data/requirements_trace.csv` |

---

## Regenerate or enrich the requirements trace from RSSOM

```bash
# Overwrite from RSSOM
venv/bin/python -m core.rssom_requirements_trace \
  --docx data/RSSOM_APCS_FIT.docx \
  --out data/requirements_trace.csv

# Merge: update titles from RSSOM, keep existing statuses and manual-only rows
venv/bin/python -m core.rssom_requirements_trace \
  --docx data/RSSOM_APCS_FIT.docx \
  --out data/requirements_trace.csv \
  --merge
```

---

## Docker (Qdrant for regulatory index)

```bash
# Start Qdrant
docker compose up -d

# Build the regulatory clause index (one-time)
venv/bin/python regulatory/build_regulatory_index.py
```

---

## Contributing

1. Create a feature branch: `git checkout -b feature/name`
2. Make changes and commit following existing message style
3. Open a Pull Request — main branch is protected

See [`docs/pipeline-overview.md`](docs/pipeline-overview.md) for a detailed walkthrough of the full pipeline and its RAG architecture.
