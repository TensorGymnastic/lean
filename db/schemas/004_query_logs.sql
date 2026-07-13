CREATE TABLE IF NOT EXISTS query_logs (
    id          uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text  text    NOT NULL,
    k           integer NOT NULL,
    filters     jsonb   NOT NULL DEFAULT '{}',
    hit_chunk_ids  uuid[]   NOT NULL DEFAULT '{}',
    hit_scores     float8[] NOT NULL DEFAULT '{}',
    latency_ms  integer NOT NULL,
    agent_id    text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs (created_at);
