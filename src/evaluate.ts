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

function binom(n: number, i: number): number {
  let c = 1;
  for (let j = 0; j < i; j++) c = (c * (n - j)) / (j + 1);
  return c;
}

/**
 * Paired two-sided sign test on per-query recall for two methods. Ties (equal
 * recall) are dropped; the p-value asks how likely the observed win/loss split
 * is under the null "the two methods are equally good". Small margins over many
 * ties (e.g. hybrid vs a strong vector baseline) come out NOT significant —
 * which is the honest verdict, not something to hide.
 */
export function signTest(
  result: EvalResult,
  a: Method,
  b: Method,
): { aWins: number; bWins: number; ties: number; pValue: number } {
  let aWins = 0;
  let bWins = 0;
  let ties = 0;
  for (const { recall } of result.perQuery) {
    if (recall[a] > recall[b]) aWins++;
    else if (recall[b] > recall[a]) bWins++;
    else ties++;
  }
  const n = aWins + bWins;
  const k = Math.max(aWins, bWins);
  let tail = 0;
  for (let i = k; i <= n; i++) tail += binom(n, i);
  const pValue = n === 0 ? 1 : Math.min(1, 2 * tail * Math.pow(0.5, n));
  return { aWins, bWins, ties, pValue };
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
