.PHONY: install hooks-install format format-check lint typecheck test verify

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
