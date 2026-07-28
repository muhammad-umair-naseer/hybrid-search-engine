import { loadQueries, loadQrels } from "./corpus.js";
import { searchAll, METHODS, type Method } from "./search.js";
import type { Scored } from "./retrieval.js";

export interface EvalResult {
  recallAt10: Record<Method, number>;
  perQuery: Array<{ qid: string; recall: Record<Method, number> }>;
  nQueries: number;
}

export function recallAt(ranked: Scored[], relevant: Set<string>, k: number): number {
  if (relevant.size === 0) return 0;
  let hit = 0;
  for (const { id } of ranked.slice(0, k)) if (relevant.has(id)) hit++;
  return hit / relevant.size;
}

/** Run every method over the SciFact test set and average recall@k. */
export async function evaluate(k = 10, poolK = 100): Promise<EvalResult> {
  const queries = loadQueries();
  const qrels = loadQrels();

  const sums: Record<Method, number> = { keyword: 0, vector: 0, hybrid: 0, naive: 0 };
  const perQuery: EvalResult["perQuery"] = [];

  for (const [qid, qtext] of queries) {
    const relevant = qrels.get(qid);
    if (!relevant || relevant.size === 0) continue;

    const all = await searchAll(qtext, poolK);
    const recall = {} as Record<Method, number>;
    for (const m of METHODS) {
      recall[m] = recallAt(all[m], relevant, k);
      sums[m] += recall[m];
    }
    perQuery.push({ qid, recall });
  }

  const n = perQuery.length;
  const recallAt10 = {} as Record<Method, number>;
  for (const m of METHODS) recallAt10[m] = sums[m] / n;

  return { recallAt10, perQuery, nQueries: n };
}

/**
 * Complementarity: how often each single method fails (recall 0) on a query the
 * OTHER single method gets (recall > 0). This is the whole reason to fuse —
 * keyword and vector fail on *different* queries.
 */
export function complementarity(result: EvalResult): {
  vectorSavesKeyword: number; // keyword=0, vector>0
  keywordSavesVector: number; // vector=0, keyword>0
} {
  let vectorSavesKeyword = 0;
  let keywordSavesVector = 0;
  for (const { recall } of result.perQuery) {
    if (recall.keyword === 0 && recall.vector > 0) vectorSavesKeyword++;
    if (recall.vector === 0 && recall.keyword > 0) keywordSavesVector++;
  }
  return { vectorSavesKeyword, keywordSavesVector };
}
