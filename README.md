# FinGuard Orchestrator 🛡️

### Enterprise Multi-Agent Wealth Management Compliance & AML Audit Engine

FinGuard Orchestrator is an autonomous, multi-agent compliance auditing system designed to automate Anti-Money Laundering (AML) transaction monitoring and regulatory audit workflows. Built on a cyclic directed acyclic graph (DAG), the platform ingests unstructured transaction feeds (SWIFT ISO 20022/pacs.008), normalizes transactional entities, audits them against authoritative compliance rulebooks (FINRA Rule 3310, FinCEN advisories, ADGM regulations), and generates deterministic, citation-backed Suspicious Activity Reports (SARs).

## 🏛️ System Architecture

┌────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & GROUNDING PLANE │
│ │
│ ┌──────────────┐ ┌────────────────────┐ ┌────────────────┐ │
│ │ Batch PDFs │ ───> │ Semantic Chunking │ ───> │ ChromaDB │ │
│ │ (SWIFT Logs) │ │ (Cosine Distance) │ │ (Vector Store) │ │
│ └──────────────┘ └────────────────────┘ └────────────────┘ │
└───────────────────────────────────┬────────────────────────────────────┘
│ (Grounding Context)
┌────────────────────────────────────────────────────────────────────────┐
│ 2. AGENTIC REASONING CORE (LangGraph Orchestration Engine) │
│ │
│ ┌──────────────┐ ┌────────────────────┐ ┌────────────────┐ │
│ │ Extraction │ ───> │ AML Audit │ ───> │ Auditor Critic │ │
│ │ (gpt-4o) │ │ (Reasoning) │ │ (Evaluation) │ │
│ └──────────────┘ └─────────┬──────────┘ └───────┬────────┘ │
│ │ │ (Loop Check)
│ │ ▼ │
│ └────────────────── [Confidence <0.8] │
└──────────────────────────────────┬─────────────────────────────────────┘
│ (Structured Audit Report)
┌────────────────────────────────────────────────────────────────────────┐
│ 3. SERVING, OPTIMIZATION & TELEMETRY │
│ │
│ ┌──────────────────────┐ ┌───────────────────────────┐ │
│ │ Asynchronous FastAPI│ ───> │ Streamlit Auditor UI │ │
│ │ (Redis Semantic Cache│ │ (Source Citations Drawer) │ │
│ └──────────────────────┘ └───────────────────────────┘ │
│ │
│ • LangSmith Distributed Tracing • DeepEval / RAGAS Quality Gate │
└────────────────────────────────────────────────────────────────────────┘

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

## 🛠️ Tech Stack

- **Orchestration & Agents**: LangGraph, LangChain
- **LLM Core**: OpenAI GPT-4o / GPT-4o-mini
- **Vector Store & Embeddings**: ChromaDB, SentenceTransformers (`all-MiniLM-L6-v2`)
- **Serving & UI**: FastAPI (Async REST), Streamlit, Uvicorn
- **Caching & Storage**: Redis (Vector Semantic Cache)
- **LLMOps & Testing**: LangSmith, DeepEval, Pytest
- **Package Management & Deployment**: `uv`, Docker (Multi-Stage Build)

## 🚀 Quickstart

### Prerequisites

- Python 3.11+
- `uv` package manager
- Docker Desktop (optional, for containerized run)
- OpenAI API Key & LangSmith API Key

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/finguard-orchestrator.git
cd finguard-orchestrator

# Create and activate virtual environment via uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```
