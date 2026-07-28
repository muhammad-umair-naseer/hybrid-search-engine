CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS docs;

CREATE TABLE docs (
    id    TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body  TEXT NOT NULL,
    -- Keyword side: a weighted tsvector (title matters more than body),
    -- maintained by Postgres so it can never drift from the text.
    tsv   tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(body, '')), 'B')
    ) STORED,
    -- Vector side: a 384-dim MiniLM embedding.
    embedding vector(384)
);

-- GIN index for full-text search.
CREATE INDEX docs_tsv_idx ON docs USING GIN (tsv);

-- NOTE: no HNSW index. At 5k docs, exact brute-force cosine kNN is fast and,
-- more importantly, EXACT — so the evaluation measures the fusion, not ANN
-- recall loss. Scaling up would add `USING hnsw (embedding vector_cosine_ops)`
-- and trade a little recall for speed (see README).
