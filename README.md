# FinGuard Orchestrator 🛡️

### Enterprise Multi-Agent Wealth Management Compliance & AML Audit Engine

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

---

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
OPENAI_API_KEY="your-openai-api-key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="finguard-orchestrator"
REDIS_HOST="localhost"
REDIS_PORT="6379"
```

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
