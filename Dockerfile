FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra local-models

COPY src ./src

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r lean && useradd -r -g lean -u 1000 lean

WORKDIR /app

COPY --from=builder /app /app
COPY db ./db

USER lean

EXPOSE 8765

CMD ["/app/.venv/bin/python", "-m", "lean.mcp_server", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
