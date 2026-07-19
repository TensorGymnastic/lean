.PHONY: install hooks-install format format-check lint typecheck test verify \
        verify-all db-init db-reset health ingest-all ingest-one search \
        mcp-serve mcp-serve-http api-serve build up down clean

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
	uv run python3 -m pytest -m 'not slow and not integration and not e2e' --cov=lean --cov-report=term-missing

verify: format-check lint typecheck test

verify-all: format-check lint typecheck
	uv run python3 -m pytest

CONFIG ?= configs/lean-pdf-lss.yaml

db-init:
	uv run lean --config $(CONFIG) db-init

db-reset:
	supabase db reset

health:
	uv run lean --config $(CONFIG) health

ingest-all:
	uv run lean --config $(CONFIG) ingest-directory --dir data --pattern "*.pdf"

ingest-one:
	@test -n "$(FILE)" || (echo "Usage: make ingest-one FILE=path/to.pdf" && exit 1)
	uv run lean --config $(CONFIG) ingest "$(FILE)"

search:
	@test -n "$(QUERY)" || (echo "Usage: make search QUERY='...'" && exit 1)
	uv run lean --config $(CONFIG) search "$(QUERY)"

mcp-serve:
	uv run lean --config $(CONFIG) mcp-serve --transport stdio

mcp-serve-http:
	uv run lean --config $(CONFIG) mcp-serve --transport http

api-serve:
	uv run lean --config $(CONFIG) api-serve

smoke: health

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
