# FinGuard Orchestrator

### Agentic AML and compliance audit prototype

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/Vector_Store-ChromaDB-FC521F.svg)](https://www.trychroma.com/)
[![DeepEval](https://img.shields.io/badge/Evaluation-DeepEval-7B2CBF.svg)](https://confident-ai.com/)
[![AWS ECS](https://img.shields.io/badge/Deployment-AWS_ECS-FF9900.svg?logo=amazon-ecs&logoColor=white)](infrastructure/terraform/README.md)

FinGuard Orchestrator is a portfolio-scale AML engineering prototype that turns a
synthetic transaction audit request into a structured, evidence-grounded compliance
report. A deterministic pre-router handles eligible low-risk requests without an LLM;
suspicious or ambiguous requests enter a cyclic LangGraph state graph that extracts
facts, retrieves and reranks local regulatory context, produces an AML assessment, and
uses an auditor critic to decide whether to finish or perform a bounded refinement.

FinGuard produces audit-supporting analysis, not legal advice, regulatory certification,
or automatically filed Suspicious Activity Reports. The examples, fixtures, and
evaluation scenarios use synthetic/demo data rather than production banking or customer
data.

## Why this project matters

- Combines deterministic routing with bounded agentic escalation instead of sending
  every request to a model.
- Grounds AML analysis in local retrieval, cross-encoder reranking, and structured
  Pydantic contracts.
- Uses a critic loop and explicit insufficient-evidence semantics to test whether a
  conclusion is supportable.
- Includes semantic caching, request-scoped token/cost observability, and optional
  LangSmith tracing.
- Evaluates behavior with versioned synthetic scenarios, offline graph replay, routing
  assertions, and an opt-in real-model harness.
- Separates a thin Streamlit UI from the FastAPI/ML backend and demonstrates controlled
  AWS ECS/Fargate deployment through Terraform.

## System architecture

```text
Synthetic transaction / audit request
                  |
                  v
        Streamlit Auditor UI :8501
                  |
                  | HTTP
                  v
           FastAPI REST API :8000
                  |
          +-------+-------------------+
          |                           |
          v                           v
  Semantic cache lookup      Deterministic pre-router
          |                           |
          | hit                       +--> eligible low-risk path
          |                                (zero provider calls)
          v
  cached structured report            otherwise
                                      |
                                      v
                       Cyclic LangGraph agentic workflow
                       extraction -> AML analysis/retrieval
                                            |
                                            v
                                      auditor critic
                                      /            \
                         bounded refinement         generate
                                loop                   |
                                                       v
                                      Structured Pydantic audit report

Supporting components
  - embedded ChromaDB PersistentClient (no Chroma HTTP service)
  - local Chroma seed plus packaged embedding and reranking model assets
  - bounded in-memory semantic cache by default; Redis is opt-in/experimental
  - xAI chat model through an OpenAI-compatible API interface
  - request-scoped usage/cost telemetry and optional LangSmith tracing
```

The agentic graph is `extraction -> aml_audit -> auditor_critic`. The critic can route
back to `aml_audit` for additional regulatory context, subject to `max_loops`, or onward
to deterministic report generation. Missing transaction evidence is not repaired by
inventing facts; it terminates as insufficient evidence.

## Core behavior

### Retrieval-grounded agentic workflow

- `TransactionExtraction`, `AMLAssessment`, `CriticAssessment`, and
  `ComplianceReport` are validated Pydantic structures.
- Chroma runs in-process through `chromadb.PersistentClient`; it is not a separately
  exposed service.
- Retrieval uses Chroma's local embedding function, then a packaged
  `BAAI/bge-reranker-large` cross-encoder to rerank candidates.
- The critic distinguishes missing transaction data, missing regulatory context, and
  inconsistent analysis. Only missing regulatory context is eligible for retrieval
  refinement.
- Final reports retain source identifiers and separate assessment completeness from
  risk.

### Assessment status is not risk

Every current report includes `assessment_status` separately from `risk_rating`:

- `COMPLETE`: the workflow had enough evidence to reach a supported AML assessment.
- `INSUFFICIENT_EVIDENCE`: the workflow could not support the requested conclusion.

A `LOW` risk rating is not clearance when the assessment status is
`INSUFFICIENT_EVIDENCE`. Consumers must interpret assessment status first.

### Deterministic routing and semantic cache

The FastAPI request path checks the semantic cache, then evaluates the deterministic
pre-router before invoking LangGraph. Eligible deterministic requests and valid cache
hits bypass the provider. Their request telemetry therefore records zero logical LLM
calls and zero tokens.

`FINGUARD_CACHE_MODE=memory` is the default. It uses a bounded, process-local semantic
cache whose maximum entry count is controlled by
`FINGUARD_MEMORY_CACHE_MAX_ENTRIES`. `disabled` is supported, and `redis` remains an
optional experimental mode for environments that separately provide Redis. Redis is
not part of the validated AWS runtime and is not required to run FinGuard.

### Request-scoped LLM observability

API responses can include request-scoped telemetry for:

- logical, completed, and failed LLM calls;
- input, cached-input, output, reasoning, and total tokens when the provider reports
  them;
- per-call latency and overall API execution latency;
- semantic-cache status;
- estimated provider cost and pricing revision.

Cost estimation is configuration-driven. Rates are supplied through environment
variables and matched to the configured model; FinGuard does not embed provider prices
in the UI. The telemetry reports explicit states such as `estimated`,
`pricing_not_configured`,
`model_mismatch`, `usage_unavailable`, or `not_applicable` rather than inventing a cost.
LangSmith tracing can be enabled through the standard LangChain/LangSmith environment
configuration; tracing metadata excludes the full transaction payload and credentials.

## Evaluation strategy

Evaluation checks behavioral contracts, not merely whether a model returned text.

- `tests/scenarios/aml_golden/v1` contains a versioned manifest with 15 synthetic AML
  scenarios covering ordinary wires, structuring, missing facts, jurisdiction handling,
  conflicting evidence, hallucination traps, single refinement, and maximum-loop
  behavior.
- The offline runner executes the real LangGraph topology with deterministic fake LLM
  and retrieval adapters, validating extraction, routing, critic actions, retry counts,
  termination, prohibited outcomes, and structured reports without provider calls.
- Focused tests cover pre-routing, graph nodes, integration boundaries, report/schema
  contracts, runtime safety, observability, the Streamlit/API boundary, containers, and
  Terraform configuration.
- An explicitly opt-in real-model harness can run selected or all 15 scenarios against
  the configured xAI model with controlled retrieval and versioned result artifacts.
- A separate DeepEval quality example exercises faithfulness and answer relevancy. It is
  provider-backed and is not part of the default offline test run.

No perfect score or universal quality metric is claimed. Real-model evaluations may
incur provider cost and must be run deliberately.

```bash
# Default suite excludes the opt-in real-model marker
uv run pytest -q

# Deterministic golden-scenario replay
uv run pytest tests/test_golden_scenarios_offline.py -q

# Representative API/runtime boundaries
uv run pytest tests/test_graph_integration_offline.py tests/test_report_contract_offline.py -q
```

See the test markers in `pyproject.toml` before running provider-backed evaluation.

## Technology stack

- **Orchestration:** LangGraph and LangChain
- **Runtime LLM:** xAI through the OpenAI-compatible `ChatOpenAI` interface;
  `XAI_MODEL` selects the model, and the validated demo configuration used `grok-4.3`
- **Structured contracts:** Pydantic
- **Retrieval:** embedded ChromaDB, SentenceTransformers, and CrossEncoder reranking
- **Serving:** asynchronous FastAPI, Uvicorn, and a thin Streamlit HTTP client
- **Caching:** bounded in-memory semantic cache by default; optional experimental Redis
- **Observability and evaluation:** request-level usage/cost telemetry, optional
  LangSmith tracing, Pytest, and opt-in DeepEval/real-model checks
- **Packaging and deployment:** `uv`, separate Docker images, Amazon ECR, ECS/Fargate,
  Application Load Balancer, CloudWatch, Secrets Manager, and Terraform

## Local quickstart

### Prerequisites

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- An xAI API key supplied by the user for agentic requests; credentials are never
  included in the repository. Deterministic and offline tests do not require it.
- Docker Desktop only for container workflows
- Optional LangSmith credentials only when remote tracing is intentionally enabled

### Install

```bash
git clone https://github.com/billbz99/finguard-orchestrator.git
cd finguard-orchestrator
uv sync
```

Copy `.env.example` to `.env` and configure only the capabilities you intend to use:

```env
XAI_API_KEY=your_xai_api_key_here
XAI_MODEL=grok-4.3

FINGUARD_CACHE_MODE=memory
FINGUARD_MEMORY_CACHE_MAX_ENTRIES=128
FINGUARD_MODEL_LOCAL_ONLY=0
FINGUARD_CHROMA_PATH=./data/chroma
FINGUARD_API_BASE_URL=http://localhost:8000
```

Optional cost estimation uses `XAI_PRICE_MODEL`,
`XAI_INPUT_PRICE_PER_MILLION`, `XAI_OUTPUT_PRICE_PER_MILLION`,
`XAI_CACHED_INPUT_PRICE_PER_MILLION`, and `XAI_PRICING_REVISION`. Supply rates from a
verified provider source; leaving them unset keeps estimation explicitly unconfigured.

Redis mode additionally uses `REDIS_HOST`, `REDIS_PORT`, and `REDIS_DB`.

### Data and local retrieval

The repository uses synthetic transaction fixtures. A checked-in processed SWIFT sample
supports development, while raw SAML-D data and generated Chroma indexes are ignored.
To regenerate transaction samples from SAML-D, place
`saml_d_transactions.csv` under `data/raw/`; the transaction generator requires that
local source file.

```bash
# Generate the local regulatory PDF fixture
uv run python -m src.utils.pdf_regulatory_generator

# Optional: regenerate processed SWIFT samples from data/raw/saml_d_transactions.csv
uv run python -m src.utils.pdf_generator

# Build or refresh the ignored local Chroma index
uv run python -m src.ingestion.run_ingestion
```

The ingestion pipeline preserves transaction-record boundaries and uses semantic
chunking for regulatory documents. Generated raw data, indexes, databases, and model
artifacts must remain outside version control.

### Run locally

Start the API first, then the UI in a second terminal:

```bash
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
uv run streamlit run src/ui/app.py
```

- API documentation: `http://127.0.0.1:8000/docs`
- API liveness: `GET http://127.0.0.1:8000/health`
- API readiness: `GET http://127.0.0.1:8000/ready`
- Streamlit UI: `http://localhost:8501`

`/health` is a lightweight process liveness check. `/ready` verifies xAI configuration,
local retrieval assets, the Chroma collection, and cache readiness without calling the
provider.

## Container architecture

FinGuard uses separate backend and frontend images:

```bash
# Backend: FastAPI, graph, local ML assets, and Chroma seed
docker build -f Dockerfile -t finguard-api:demo .

# Frontend: thin Streamlit HTTP client only
docker build -f Dockerfile.frontend -t finguard-ui:demo .
```

Example local run:

```bash
docker network create finguard-demo
docker run --rm --name finguard-api --network finguard-demo \
  --env-file .env -p 8000:8000 finguard-api:demo
docker run --rm --name finguard-ui --network finguard-demo \
  -e FINGUARD_API_BASE_URL=http://finguard-api:8000 \
  -p 8501:8501 finguard-ui:demo
```

The backend image:

- installs the frozen `uv.lock` dependency resolution;
- runs as the non-root `finguard` user with one Uvicorn worker;
- packages the Chroma seed and pinned assets for semantic-cache embeddings, Chroma query
  embeddings, and cross-encoder reranking;
- copies the immutable Chroma seed to writable ephemeral storage at startup;
- enables Hugging Face/Transformers offline mode in the container while retaining HTTPS
  connectivity to xAI;
- uses `/health` for container liveness and `/ready` for application readiness.

Pinned model identifiers and revisions are recorded in
[`deployment/model-manifest.json`](deployment/model-manifest.json). The frontend image
contains only Streamlit-side code and locked UI dependencies; it does not initialize
LangGraph, Chroma, the semantic cache, or the backend ML models.

## Validated AWS demo architecture

The Terraform module under [`infrastructure/terraform`](infrastructure/terraform/README.md)
was deployed and validated as a controlled portfolio demonstration:

```text
Internet
   |
   v
Application Load Balancer :80
   |
   v
Streamlit container :8501
   |
   | localhost inside the shared Fargate task network
   v
FastAPI container :8000
   |
   +--> LangGraph workflow
   +--> embedded Chroma / packaged local model assets
   +--> xAI over outbound HTTPS
```

- One ECS/Fargate task contains separate frontend and backend containers.
- The ALB and task security groups expose Streamlit only; FastAPI port 8000 is not
  directly internet-accessible.
- Streamlit calls FastAPI at `127.0.0.1:8000` within the task.
- Private ECR repositories hold independently built, immutable frontend and backend
  images, and Terraform consumes digest-pinned image URIs.
- An externally managed Secrets Manager secret injects `XAI_API_KEY` into the backend.
  Terraform receives only the secret ARN, and the ECS execution role is scoped to read
  that secret.
- CloudWatch captures frontend/backend container logs while the runtime exists.
- The ECS service enables a deployment circuit breaker with rollback.
- Terraform manages the ephemeral demo VPC, networking, ALB, ECS, IAM execution role,
  and log groups; it does not create ECR repositories, images, or the secret value.

This is a cost-conscious portfolio/demo topology, not a bank-grade production design.
The validated runtime was intentionally torn down after demonstration; reusable ECR
images and the separately managed secret can be retained for a future controlled
redeployment.

## AWS demo deployment lifecycle

The AWS runtime is intended to be created on demand rather than operated continuously:

1. Confirm the required immutable frontend/backend images already exist in ECR.
2. Confirm the xAI credential is managed separately in Secrets Manager; do not retrieve
   or place its value in Terraform files.
3. Review the ignored `infrastructure/terraform/local.tfvars`, including digest-pinned
   image URIs and the existing secret ARN.
4. Run `terraform init` when initialization is required.
5. Create a plan with `terraform plan -var-file=local.tfvars`.
6. Review the complete plan, account, region, network exposure, and expected cost.
7. Apply only the reviewed plan.
8. Wait for ECS service stability and a healthy ALB target.
9. Validate Streamlit, `/health`, `/ready`, and an approved synthetic audit.
10. Demonstrate the application.
11. Create a saved teardown plan with
    `terraform plan -destroy -var-file=local.tfvars -out=deployment-runtime-destroy.tfplan`.
12. Review every destroy action and confirm ECR and the separately managed secret are
    absent.
13. Apply the exact saved plan with
    `terraform apply deployment-runtime-destroy.tfplan`.
14. Verify through AWS APIs that ECS/Fargate, ALB, and Terraform-managed runtime
    resources are gone.

Retaining ECR images and the externally managed secret allows the same immutable
application version to be redeployed without rebuilding. A recreated ALB can receive a
different DNS name. See the
[`controlled deployment runbook`](infrastructure/terraform/DEPLOYMENT.md) for the
detailed workflow.

## Security and scope

- FinGuard is a portfolio/demo AML engineering system using synthetic data, not a
  production banking workload.
- Runtime credentials are supplied through environment variables or AWS Secrets Manager
  and must never be committed.
- In the validated AWS topology, network controls expose only the Streamlit frontend;
  FastAPI and embedded Chroma are not directly internet-facing.
- Containerized embedding, reranking, and Chroma assets operate locally/offline where
  configured; only approved provider requests require outbound xAI connectivity.
- Deployment, model, dependency, and vulnerability risk still require environment-
  specific review. This repository does not claim production readiness, regulatory
  certification, or absence of vulnerabilities.

## Repository guide

- [`src/main.py`](src/main.py): FastAPI audit endpoint, cache/pre-router flow, and
  request observability
- [`src/graph`](src/graph): state, structured schemas, nodes, routing, and workflow
- [`src/ingestion`](src/ingestion): loaders, Chroma persistence, retrieval, and reranking
- [`src/ui`](src/ui): thin Streamlit client
- [`tests/scenarios/aml_golden/v1`](tests/scenarios/aml_golden/v1): versioned synthetic
  evaluation dataset
- [`deployment`](deployment): container startup, model packaging, and manifests
- [`infrastructure/terraform`](infrastructure/terraform/README.md): ECS/Fargate module
  and controlled deployment documentation
