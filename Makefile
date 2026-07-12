.PHONY: install hooks-install format format-check lint typecheck test security deadcode deptry prepush verify run docker-build

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
	uv run mypy app tests

test:
	uv run python3 -m pytest

security:
	uv run pip-audit
	uv run bandit -q -r app

deadcode:
	uv run vulture app tests

deptry:
	uv run deptry .

prepush: format-check lint typecheck security deadcode deptry
	uv run python3 -m pytest -n auto --maxfail=1

verify: format-check lint typecheck test security deadcode deptry

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t starter-llm-service-fastapi .
