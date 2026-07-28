import { embed } from "./embed.js";
import { keywordSearch, vectorSearch, hasContentTerms, type Scored } from "./retrieval.js";
import { fuseRRF, fuseNaive } from "./fusion.js";

export type Method = "keyword" | "vector" | "hybrid" | "naive";
export const METHODS: Method[] = ["keyword", "vector", "hybrid", "naive"];

/**
 * RRF weights as [keyword, vector], tuned on the SciFact TRAIN split by
 * src/scripts/tune.ts. Vector retrieval is the stronger signal on this corpus,
 * so keyword gets less weight — enough to rescue vector's misses without
 * demoting its hits.
 */
export const FUSION_WEIGHTS: [number, number] = [0.3, 1];

export interface RawLists {
  keyword: Scored[];
  vector: Scored[];
}

/** Embed the query and retrieve both lists (one embed + two DB round-trips). */
export async function retrieveLists(q: string, poolK = 100): Promise<RawLists> {
  const qvec = await embed(q);
  const [keyword, vector] = await Promise.all([
    keywordSearch(q, poolK),
    vectorSearch(qvec, poolK),
  ]);
  return { keyword, vector };
}

export interface AllRankings {
  keyword: Scored[];
  vector: Scored[];
  hybrid: Scored[]; // weighted RRF
  naive: Scored[]; // raw score addition
}

/** All four rankings for one query. */
export async function searchAll(
  q: string,
  poolK = 100,
  weights: [number, number] = FUSION_WEIGHTS,
): Promise<AllRankings> {
  // A contentless query has no meaningful answer — don't let the vector side
  // return neighbours of a meaningless embedding and pass noise off as results.
  if (!(await hasContentTerms(q))) {
    return { keyword: [], vector: [], hybrid: [], naive: [] };
  }
  const { keyword, vector } = await retrieveLists(q, poolK);
  return {
    keyword,
    vector,
    hybrid: fuseRRF([keyword, vector], weights),
    naive: fuseNaive(keyword, vector),
  };
}

export async function search(
  q: string,
  method: Method,
  topN = 10,
  poolK = 100,
): Promise<Scored[]> {
  const all = await searchAll(q, poolK);
  return all[method].slice(0, topN);
}
