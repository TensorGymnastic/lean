# pydantic-settings

Lean's config layer. Reads `.env` (secrets) + `config.yaml` (app config),
with environment variables overriding YAML values.

## What lean uses

```python
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # Secrets from .env
    supabase_db_url: str = Field(description="Direct Postgres DSN")
    lean_mcp_api_key: str = Field(description="Bearer token for HTTP")

    # App config from config.yaml (with .env override via env var)
    ocr_base_url: str = _yaml.get("ocr", {}).get("base_url", "")
    search_top_k: int = _yaml.get("retrieval", {}).get("top_k", 5)

    @field_validator("lean_mcp_api_key")
    @classmethod
    def _validate_api_key(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError("must be at least 16 characters")
        return v

    @model_validator(mode="after")
    def _validate_cross_field(self) -> Settings:
        if self.chunk_hard_cap <= self.chunk_target_max:
            raise ValueError("hard_cap must be > target_max")
        return self
```

## Precedence (high → low)

1. **Environment variables** (e.g. `LEAN_MCP_API_KEY=...` exported)
2. **`.env` file** values
3. **`config.yaml` defaults** (read in module scope via `_load_yaml()`)

Note: lean loads YAML at import time and passes values to `Field`
defaults. Env vars override by matching the field name (case-insensitive).

## Validators

- **`field_validator("name")`** — single-field validation. Use for
  format checks (port range, API key length, etc.)
- **`model_validator(mode="after")`** — cross-field validation. Use for
  invariants like `hard_cap > target_max`.

Both raise `pydantic.ValidationError` on failure, which fails fast at
startup. See `src/lean/config/settings.py` for the full set.

## Gotchas

- **`extra="ignore"`** is set — unknown env vars are silently dropped.
  This means a typo'd env var won't error. Be explicit about expected
  vars in `.env.example`.
- **`@lru_cache` on `get_settings()`** — settings are evaluated once
  per process. Tests call `get_settings.cache_clear()` to force re-read.
- **YAML loaded at import time** — changing `config.yaml` requires a
  process restart, not a hot reload.
- **Sensitive defaults** — never put secrets in `config.yaml`. They go
  in `.env` (gitignored).

## Resources

- Docs: <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>
- lean config: `src/lean/config/settings.py` + `config.yaml`
