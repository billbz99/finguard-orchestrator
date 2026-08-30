# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.8.15 AS uv

FROM python:3.11-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/finguard/models/huggingface \
    HOME=/opt/finguard/home

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./

# --frozen makes lockfile drift a build failure. The source tree is copied later,
# so the application itself is intentionally not installed as a wheel.
RUN uv sync --frozen --no-dev --no-install-project

COPY deployment/prepare_model_assets.py deployment/model-manifest.json /app/deployment/
RUN /app/.venv/bin/python /app/deployment/prepare_model_assets.py


FROM python:3.11-slim AS runtime

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/opt/finguard/home \
    HF_HOME=/opt/finguard/models/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=FALSE \
    FINGUARD_MODEL_LOCAL_ONLY=1 \
    FINGUARD_CACHE_MODE=memory \
    FINGUARD_CHROMA_SEED_PATH=/opt/finguard/chroma-seed \
    FINGUARD_CHROMA_PATH=/tmp/finguard/chroma

WORKDIR /app

RUN groupadd --system finguard \
    && useradd --system --gid finguard --home-dir /opt/finguard/home finguard \
    && mkdir -p /opt/finguard/home /opt/finguard/models /opt/finguard/chroma-seed /tmp/finguard \
    && chown -R finguard:finguard /opt/finguard /tmp/finguard

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=finguard:finguard /opt/finguard/models /opt/finguard/models
COPY --from=builder --chown=finguard:finguard /opt/finguard/home/.cache/chroma /opt/finguard/home/.cache/chroma
COPY --chown=finguard:finguard src /app/src
COPY --chown=finguard:finguard deployment /app/deployment
COPY --chown=finguard:finguard data/chroma /opt/finguard/chroma-seed

USER finguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

ENTRYPOINT ["python", "deployment/start_api.py"]
