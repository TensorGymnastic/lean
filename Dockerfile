FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY db ./db
COPY supabase ./supabase

RUN uv sync --frozen --no-dev

EXPOSE 8765 8766

CMD ["uv", "run", "python", "-m", "lean.mcp_server", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
