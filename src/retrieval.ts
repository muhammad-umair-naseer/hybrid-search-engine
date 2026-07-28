import { query, toVectorLiteral } from "./db.js";

export interface Scored {
  id: string;
  score: number;
}

/**
 * Keyword retrieval via Postgres full-text search, ranked by `ts_rank_cd` (term
 * frequency + proximity / cover density) — a real lexical signal, though NOT
 * literal BM25 (that needs an extension).
 *
 * We build an OR query from the terms rather than using `plainto_tsquery`, which
 * ANDs them: a natural-language claim rarely shares *every* word with its
 * supporting document, so AND returns almost nothing. OR matches any term and
 * lets `ts_rank_cd` rank by how many/how closely they hit — the same "any term
 * contributes" behaviour BM25 has. Scores are small floats.
 */
export async function keywordSearch(q: string, limit: number): Promise<Scored[]> {
  const { rows } = await query<{ id: string; score: string }>(
    `WITH q AS (
        SELECT replace(plainto_tsquery('english', $1)::text, '&', '|')::tsquery AS query
     )
     SELECT d.id, ts_rank_cd(d.tsv, q.query) AS score
       FROM docs d, q
      WHERE q.query::text <> '' AND d.tsv @@ q.query
      ORDER BY score DESC
      LIMIT $2`,
    [q, limit],
  );
  return rows.map((r) => ({ id: r.id, score: Number(r.score) }));
}

/**
 * Vector retrieval via pgvector. `<=>` is cosine distance; we return cosine
 * similarity (1 - distance) in [~0, 1]. Exact brute-force kNN (no ANN index),
 * so results are the true nearest neighbours.
 */
export async function vectorSearch(qvec: number[], limit: number): Promise<Scored[]> {
  const { rows } = await query<{ id: string; score: string }>(
    `SELECT id, 1 - (embedding <=> $1::vector) AS score
       FROM docs
      WHERE embedding IS NOT NULL
      ORDER BY embedding <=> $1::vector
      LIMIT $2`,
    [toVectorLiteral(qvec), limit],
  );
  return rows.map((r) => ({ id: r.id, score: Number(r.score) }));
}
