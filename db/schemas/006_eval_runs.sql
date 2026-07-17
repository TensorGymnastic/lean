CREATE TABLE IF NOT EXISTS eval_runs (
    id          uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    config      jsonb   NOT NULL DEFAULT '{}',
    hit_rate    float8  NOT NULL,
    mrr         float8  NOT NULL,
    ndcg        float8  NOT NULL DEFAULT 0,
    recall      float8  NOT NULL DEFAULT 0,
    mean_latency_ms integer NOT NULL,
    sample_count integer NOT NULL,
    k           integer NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_created_at ON eval_runs (created_at);
