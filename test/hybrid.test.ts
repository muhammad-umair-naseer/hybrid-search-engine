import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { evaluate, complementarity, type EvalResult } from "../src/evaluate.js";
import { query, close } from "../src/db.js";

/**
 * THE PROOF. Runs every method over the SciFact test set and asserts the hybrid
 * fusion beats both single methods on recall@10, that keyword and vector each
 * fail on queries the other catches, and that naive score addition degrades.
 *
 * Requires the corpus to be loaded (`npm run db:reset && npm run load`); if it
 * isn't, the tests skip with a clear message rather than hard-failing.
 */

let loaded = false;
let result: EvalResult | null = null;

beforeAll(async () => {
  const { rows } = await query<{ n: string }>("SELECT count(embedding) AS n FROM docs");
  loaded = Number(rows[0]?.n ?? 0) >= 5000;
  if (loaded) result = await evaluate(10, 100);
}, 180_000);

afterAll(async () => {
  await close();
});

describe("hybrid search recall@10 (SciFact)", () => {
  it("hybrid (RRF) beats keyword-only AND vector-only", (ctx) => {
    if (!loaded || !result) return ctx.skip();
    const r = result.recallAt10;
    expect(r.hybrid).toBeGreaterThan(r.keyword);
    expect(r.hybrid).toBeGreaterThan(r.vector);
  });

  it("keyword and vector each fail on queries the other catches (complementarity)", (ctx) => {
    if (!loaded || !result) return ctx.skip();
    const c = complementarity(result);
    // Both directions must be non-zero — that is *why* fusion helps.
    expect(c.vectorSavesKeyword).toBeGreaterThan(0);
    expect(c.keywordSavesVector).toBeGreaterThan(0);
  });

  it("naive score addition degrades vs RRF (the incompatible-scales failure)", (ctx) => {
    if (!loaded || !result) return ctx.skip();
    expect(result.recallAt10.hybrid).toBeGreaterThan(result.recallAt10.naive);
  });
});
