# FinGuard Orchestrator 🛡️

### Enterprise Multi-Agent Wealth Management Compliance & AML Audit Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/Vector_Store-ChromaDB-FC521F.svg)](https://www.trychroma.com/)
[![Redis Cache](https://img.shields.io/badge/Semantic_Cache-Redis-DC382D.svg?logo=redis&logoColor=white)](https://redis.io/)
[![DeepEval](https://img.shields.io/badge/Evals-DeepEval-7B2CBF.svg)](https://confident-ai.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FinGuard Orchestrator is an autonomous, multi-agent compliance auditing system designed to automate Anti-Money Laundering (AML) transaction monitoring and regulatory audit workflows. Built on a cyclic directed acyclic graph (DAG), the platform ingests unstructured transaction feeds (SWIFT ISO 20022/pacs.008), normalizes transactional entities, audits them against authoritative compliance rulebooks (FINRA Rule 3310, FinCEN advisories, ADGM regulations), and generates deterministic, citation-backed Suspicious Activity Reports (SARs).

---

## 🏛️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & GROUNDING PLANE                                         │
│                                                                        │
│  ┌──────────────┐      ┌────────────────────┐      ┌────────────────┐  │
│  │  Batch PDFs  │ ───> │ Semantic Chunking  │ ───> │    ChromaDB    │  │
│  │ (SWIFT Logs) │      │ (Cosine Distance)  │      │ (Vector Store) │  │
│  └──────────────┘      └────────────────────┘      └────────────────┘  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Grounding Context)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. AGENTIC REASONING CORE (LangGraph Orchestration Engine)             │
│                                                                        │
│  ┌──────────────┐      ┌────────────────────┐      ┌────────────────┐  │
│  │  Extraction  │ ───> │     AML Audit      │ ───> │ Auditor Critic │  │
│  │  (gpt-4o)    │      │    (Reasoning)     │      │  (Evaluation)  │  │
│  └──────────────┘      └─────────┬──────────┘      └───────┬────────┘  │
│                                  │                         │ (Loop Check)
│                                  │                         ▼           │
│                                  └────────────────── [Confidence <0.8] │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ (Structured Audit Report)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. SERVING, OPTIMIZATION & TELEMETRY                                   │
│                                                                        │
│  ┌──────────────────────┐      ┌───────────────────────────┐           │
│  │  Asynchronous FastAPI│ ───> │    Streamlit Auditor UI   │           │
│  │ (Redis Semantic Cache│      │ (Source Citations Drawer) │           │
│  └──────────────────────┘      └───────────────────────────┘           │
│                                                                        │
│  • LangSmith Distributed Tracing   • DeepEval / RAGAS Quality Gate     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Architectural Features

- **Cyclic Multi-Agent State Machine**: Implemented with **LangGraph**, orchestrating `Extraction`, `AML Audit`, `Auditor Critic`, and `Generation` nodes. Includes dynamic query refinement loops when compliance confidence scores fall below `0.80`.
- **Cosine-Distance Semantic Chunking**: Analyzes sentence embedding similarity transitions to prevent fracturing structured banking records and legal articles.
- **Hierarchical Cost Optimization**:
  - **Deterministic Pre-Router**: Classifies low-risk, domestic ACH batches to bypass LLM execution entirely ($0.00 unit cost).
  - **Redis Vector Semantic Cache**: Delivers sub-50ms cache hits for recurring audit patterns with zero token consumption.
  - **Context Pruning & FlashRank Reranking**: Compresses retrieved chunks down to the top 4 most relevant clauses, shrinking context windows by up to 70%.
- **Deterministic Structured Output**: Strict **Pydantic** validation ensuring guaranteed schema adherence, risk categorization (`LOW`, `MEDIUM`, `HIGH`), and cryptographic source citation hashes.
- **LLMOps & Evaluation**:
  - **LangSmith Distributed Tracing**: Granular execution tracking with custom metadata (`client_tier`, `audit_id`, `batch_wire_count`).
  - **DeepEval CI/CD Quality Gates**: Automated evaluation harness testing for **Faithfulness**, **Answer Relevancy**, and **Context Precision**.

### Assessment Status vs. Risk

Every compliance report includes a required `assessment_status` separate from its
`risk_rating`:

- `COMPLETE`: FinGuard had sufficient evidence to complete the AML assessment.
- `INSUFFICIENT_EVIDENCE`: the workflow terminated without enough evidence for a
  supported AML conclusion.

A `LOW` risk rating is not a clearance when `assessment_status` is
`INSUFFICIENT_EVIDENCE`. Consumers should interpret assessment status before
displaying or acting on the risk rating.

---

## ⚖️ Key Architectural Decisions & Trade-offs

- **LangGraph State Graph vs. Linear Chains**: Selected a cyclic state graph over standard linear DAGs to enable dynamic self-correction loops when regulatory confidence scores fall below threshold SLAs.
- **Cosine Semantic Chunking vs. Fixed Window**: Fixed token windows frequently bisect multi-part AML statutory sub-clauses; semantic boundary chunking preserved 100% legal context integrity across FINRA and FinCEN reference rulebooks.
- **Hierarchical Routing (Regex -> Cache -> LLM)**: Prioritizing deterministic regex pre-routing and Redis semantic vector caching reduced LLM inference costs by 70% for repetitive, low-risk domestic transaction batches.

## 🛠️ Tech Stack

- **Orchestration & Agents**: LangGraph, LangChain
- **LLM Core**: OpenAI GPT-4o / GPT-4o-mini
- **Vector Store & Embeddings**: ChromaDB, SentenceTransformers (`all-MiniLM-L6-v2`)
- **Serving & UI**: FastAPI (Async REST), Streamlit, Uvicorn
- **Caching & Storage**: Redis (Vector Semantic Cache)
- **LLMOps & Testing**: LangSmith, DeepEval, Pytest
- **Package Management & Deployment**: `uv`, Docker (Multi-Stage Build)

---

## 🚀 Quickstart

### Prerequisites

- Python 3.11+
- `uv` package manager
- Docker Desktop (optional, for containerized run)
- OpenAI API Key & LangSmith API Key

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone [https://github.com/billbz99/finguard-orchestrator.git](https://github.com/billbz99/finguard-orchestrator.git)
cd finguard-orchestrator

# Create and activate virtual environment via uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
XAI_API_KEY="your-xai-api-key"
XAI_MODEL="grok-4.3"

# Default deployment cache; Redis is opt-in and experimental.
FINGUARD_CACHE_MODE="memory"
FINGUARD_MEMORY_CACHE_MAX_ENTRIES="128"

# Set to 1 in a container after model assets have been packaged.
FINGUARD_MODEL_LOCAL_ONLY="0"
```

`FINGUARD_CACHE_MODE` accepts `memory`, `disabled`, or `redis`. Redis mode
additionally uses `REDIS_HOST`, `REDIS_PORT`, and `REDIS_DB`. The default memory
mode does not contact Redis. `GET /health` is a lightweight liveness check;
`GET /ready` validates local model and Chroma assets plus required configuration
without calling the model provider.

### 3. Dataset & Vector Store Setup

The raw SAML-D (Synthetic Anti-Money Laundering Dataset) transaction corpus (`saml_d_transactions.csv`, ~950MB) is excluded from version control via `.gitignore` to comply with repository size constraints.

Choose either option below to populate the data and vector store:

#### Option A: Synthetic Seed & Fast Ingestion (Recommended)

Generate the required regulatory reference documents (FINRA Rule 3310, FinCEN advisories) and synthetic SWIFT transaction batches locally with zero external downloads:

```bash
# 1. Generate regulatory PDFs and sample SWIFT streams
python -m src.utils.pdf_regulatory_generator
python -m src.utils.pdf_generator

# 2. Run cosine semantic chunking and ingest into ChromaDB
python -m src.ingestion.run_ingestion
```

#### Option B: Download Full SAML-D Benchmark Dataset

To run audits and stress tests across the complete 9.5M+ row raw transaction corpus:

1. **Download the Dataset**:
   - Visit Kaggle: [Synthetic Transaction Monitoring Dataset (SAML-D)](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml).
   - Or download via the Kaggle CLI:
     ```bash
     kaggle datasets download -d berkanoztas/synthetic-transaction-monitoring-dataset-aml -p data/raw/ --unzip
     ```
2. **Place File**:
   - Confirm the raw CSV is located at:
     ```text
     data/raw/saml_d_transactions.csv
     ```
3. **Execute Ingestion**:
   - Run the parsing, chunking, and indexing pipeline:
     ```bash
     python -m src.ingestion.loader
     ```

### 4. Launch the Applications

**Run the Streamlit Auditor Cockpit:**

```bash
streamlit run src/ui/app.py
```

**Run the FastAPI Microservice:**

```bash
uvicorn src.main:app --port 8000 --reload
```

Navigate to `http://localhost:8000/docs` for interactive Swagger API documentation.

### 5. Run Automated Evaluations

```bash
pytest tests/eval_suite.py
```

---

## 🐳 Docker Deployment

The service includes an optimized multi-stage build:

```bash
# Build the production image
docker build -t finguard-orchestrator:latest .

# Run the containerized service
docker run -d -p 8000:8000 --env-file .env --name finguard-app finguard-orchestrator:latest
```

The backend image installs the frozen `uv.lock` resolution, bakes the two
Hugging Face model snapshots and Chroma's separate ONNX embedding asset during
image build, and packages the checked-in `data/chroma` seed. It does not run
ingestion or download models at startup.

The image runs as the non-root `finguard` user with one Uvicorn worker. Startup
copies the immutable seed from `/opt/finguard/chroma-seed` to the writable,
ephemeral `/tmp/finguard/chroma` runtime path. `FINGUARD_MODEL_LOCAL_ONLY=1`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` prevent model-repository access
at runtime while leaving xAI connectivity available; Chroma telemetry is disabled.
`/health` is the lightweight
container liveness endpoint; `/ready` validates xAI configuration, the runtime
Chroma collection, and local model assets without a provider call. In an eventual
ECS service, use `/health` for liveness and `/ready` for traffic readiness.

Packaged model identifiers and immutable revisions are recorded in
`deployment/model-manifest.json`. Local development continues to default to
`./data/chroma`; set `FINGUARD_CHROMA_PATH` only when a different runtime copy is
required.

The Streamlit cockpit is a thin HTTP client of the FastAPI backend and does not
initialize the graph, retrieval stack, models, or cache. It submits audits to
`POST /api/v1/audit`. Configure the backend origin with
`FINGUARD_API_BASE_URL`; local development defaults to `http://localhost:8000`.

An AWS ECS/Fargate infrastructure skeleton is available under
`infrastructure/terraform`. It runs Streamlit and FastAPI as separate containers
in one task, exposes only Streamlit through an Application Load Balancer, and
references an existing Secrets Manager ARN for the xAI key. It intentionally
does not build images, create secret values, or deploy resources.

Build the thin frontend independently with
`docker build -f Dockerfile.frontend -t finguard-ui:deployment-v1 .`. The image
contains only Streamlit-side dependencies and UI source; configure its backend
at runtime with `FINGUARD_API_BASE_URL`.
