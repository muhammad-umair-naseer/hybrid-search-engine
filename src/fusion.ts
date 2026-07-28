import type { Scored } from "./retrieval.js";

/**
 * The hard problem: keyword `ts_rank_cd` scores and cosine similarities live on
 * different, incomparable scales, so you cannot just combine the raw numbers.
 *
 * Two ways to combine ranked lists, one good and one deliberately broken.
 */

export const RRF_K = 60;

/**
 * Reciprocal Rank Fusion — the chosen method. It throws the raw scores away and
 * fuses by RANK: a document's contribution from a list is w / (k + rank), where
 * w is that list's weight. Because only ranks matter, the scale mismatch simply
 * never arises. `k` dampens the influence of low ranks (k=60 is the common
 * default).
 *
 * Weights matter when one retriever is much stronger than the other: equal
 * weights let the weaker list demote the stronger list's correct hits. The
 * weight is a single scalar, tuned on the TRAIN split (never the test set) —
 * see src/scripts/tune.ts.
 */
export function fuseRRF(
  lists: Scored[][],
  weights?: number[],
  k: number = RRF_K,
): Scored[] {
  const w = weights ?? lists.map(() => 1);
  const scores = new Map<string, number>();
  lists.forEach((list, li) => {
    list.forEach((item, i) => {
      const rank = i + 1;
      scores.set(item.id, (scores.get(item.id) ?? 0) + w[li]! / (k + rank));
    });
  });
  return toSorted(scores);
}

/**
 * Naive raw-score addition — the broken baseline. It adds the keyword score and
 * the cosine similarity directly. Whichever score type has the larger magnitude
 * dominates the sum, so one signal effectively drowns out the other and the
 * fused ranking collapses toward a single method. The PROVE phase shows recall
 * drop when this replaces RRF.
 */
export function fuseNaive(keyword: Scored[], vector: Scored[]): Scored[] {
  const scores = new Map<string, number>();
  for (const { id, score } of keyword) scores.set(id, (scores.get(id) ?? 0) + score);
  for (const { id, score } of vector) scores.set(id, (scores.get(id) ?? 0) + score);
  return toSorted(scores);
}

function toSorted(scores: Map<string, number>): Scored[] {
  return [...scores.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score);
}
