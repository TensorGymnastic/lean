.PHONY: install hooks-install format format-check lint typecheck test verify \
        db-init db-reset ocr-health ingest-all ingest-one search \
        mcp-serve mcp-serve-http api-serve smoke build up down

install:
	uv sync --all-groups

hooks-install:
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy src/lean

test:
	uv run python3 -m pytest

verify: format-check lint typecheck test

db-init:
	for f in db/schemas/*.sql; do \
		echo "applying $$f..."; \
		psql "$(SUPABASE_DB_URL)" -f "$$f"; \
	done
	@echo "schema applied."

db-reset:
	supabase db reset

ocr-health:
	@./scripts/smoke-ocr.sh

ingest-all:
	uv run python -m lean.cli ingest data/*.pdf

ingest-one:
	@test -n "$(FILE)" || (echo "Usage: make ingest-one FILE=path/to.pdf" && exit 1)
	uv run python -m lean.cli ingest "$(FILE)"

search:
	@test -n "$(QUERY)" || (echo "Usage: make search QUERY='...'" && exit 1)
	uv run python -m lean.cli search "$(QUERY)"

mcp-serve:
	uv run python -m lean.mcp_server --transport stdio

mcp-serve-http:
	uv run python -m lean.mcp_server --transport http --port 8765

api-serve:
	uv run uvicorn lean.api.routes:app --reload --port 8766

smoke:
	./scripts/smoke-ocr.sh
	./scripts/smoke-pgvector.sh

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down
