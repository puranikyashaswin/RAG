# Agentic RAG Assistant

A production-grade Agentic RAG Research Assistant for Healthcare Guidelines, featuring multi-step orchestration, human-in-the-loop controls, automated evaluation, and full observability.

## Problem Statement

Healthcare professionals need quick, accurate answers from complex guideline documents. This assistant plans queries, retrieves evidence from a private corpus, synthesizes grounded responses with citations, verifies quality, and escalates to human review when uncertain. It ensures reliability through rigorous evaluation and HITL for critical decisions.

## Architecture Diagram

```
[Ingestion]
Healthcare PDFs/HTML/Markdown → Semantic Chunking → Embeddings (text-embedding-3-small) → Qdrant DB

[Retrieval]
Query → Hybrid (Dense + BM25) → Score Fusion → Cross-Encoder Rerank → Top-K Chunks

[Orchestration (LangGraph)]
Plan → Retrieve → Synthesize → Verify → Decide {Finalize | Refine | Escalate to HITL}

[HITL]
Interrupt on Low Groundedness (<0.80) → Human Approval → Resume Deterministically

[Evaluation]
Golden Dataset → Metrics (Precision@K, Recall@K, NDCG, Groundedness, Faithfulness) → Reports

[API & Observability]
FastAPI Endpoints + LangSmith Tracing (Tokens, Latency, Costs, Replays)
```

## Quickstart

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/yourusername/agentic-rag-assistant.git
   cd agentic-rag-assistant
   cp .env.example .env  # Add OPENAI_API_KEY and LANGSMITH_API_KEY
   pip install -r requirements.txt
   ```

2. **Run with Docker**:
   ```bash
   docker-compose up -d
   ```

3. **Ingest Data**:
   ```bash
   python src/ingestion.py
   ```

4. **Query the Assistant**:
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "What are the guidelines for diabetes management?", "mode": "hybrid", "top_k": 10}'
   ```

5. **Run Evaluation**:
   ```bash
   python -m src.evaluation --dataset data/golden/healthcare.jsonl --config config/default.yaml
   ```

## Metrics Table

| Metric | Target | Baseline | Hybrid + Rerank |
|--------|--------|----------|-----------------|
| Recall@5 | ≥0.70 | 0.65 | 0.72 |
| Precision@5 | ≥0.60 | 0.55 | 0.62 |
| NDCG@10 | ≥0.75 | 0.70 | 0.78 |
| Groundedness | ≥0.80 | 0.75 | 0.82 |
| Faithfulness | ≥0.80 | 0.78 | 0.84 |
| Abstention on Unanswerable | ≥0.95 | 0.90 | 0.96 |

*Latest results in [docs/reports/latest](docs/reports/latest/).*

## Demo Snippet

Query: "Summarize vaccination protocols for children under 5."

Response:
```
Answer: Vaccination protocols for children under 5 include DTaP, Hib, IPV, PCV13, and RV at specific ages...

Citations:
- [cdc_guidelines.pdf:page_12|section_3.2|chunk_45]
- [who_vaccine_schedule.html:page_5|section_1.1|chunk_23]

Scores: {"groundedness": 0.85, "faithfulness": 0.88}
Trace ID: abc123
```

## Human-in-the-Loop (HITL)

- **Triggers**: Groundedness < 0.80, missing citations, conflicting sources, out-of-domain queries.
- **Behavior**: Pauses execution, sends approval bundle to reviewer. Resume on approval with checkpoint merge.
- **Approval Endpoint**:
  ```bash
  curl -X POST "http://localhost:8000/hitl/approve" \
       -H "Content-Type: application/json" \
       -d '{"trace_id": "abc123", "decision": "approved"}'
  ```

## Observability & Tracing

- Traces every graph node, tool call, latency, tokens, and cost using LangSmith.
- Replay any trace via LangSmith dashboard.
- Screenshot of a sample trace:

![Trace Screenshot](docs/trace_screenshot.png)

## Evaluation

- **Golden Set**: 200+ questions on healthcare guidelines, including paraphrases, hard negatives, and unanswerables.
- **CLI**: `python -m src.evaluation --dataset data/golden/healthcare.jsonl --config config/default.yaml`
- **Outputs**: CSVs and Markdown in `docs/reports/{timestamp}/`.
- **CI Gates**: Fail on regressions >5% vs baselines.

## API Endpoints

- `POST /query`: Submit query, get answer + citations + scores + trace_id.
- `POST /hitl/approve`: Approve HITL interrupt.
- `POST /eval/run`: Run evaluation, get report path.

## Development

- **Python**: 3.11+
- **Models**: Embed: text-embedding-3-small, Rerank: cross-encoder/ms-marco-MiniLM-L-6-v2, LLM: gpt-4o-mini
- **Vector DB**: Qdrant (Docker)
- **Tests**: `pytest tests/`
- **Lint**: `flake8 src/`

## Contributing

- PRs must pass CI (lint, tests, eval).
- Update metrics in README on improvements.
- Nightly evals update `docs/reports/latest`.

## License

MIT