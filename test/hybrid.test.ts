import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { evaluate, complementarity, signTest, type EvalResult } from "../src/evaluate.js";
import { query, close } from "../src/db.js";

/**
 * THE PROOF. Runs every method over the SciFact test set and asserts what is
 * *statistically* real:
 *   - hybrid RRF significantly beats keyword-only and naive score-addition
 *     (large, sign-test-significant gaps);
 *   - hybrid is never worse than the strong vector-only baseline (the gain over
 *     vector alone is within noise on this corpus — we assert "not worse" and
 *     report the sign test rather than overclaiming a 1-query win);
 *   - keyword and vector each fail on queries the other catches (why we fuse).
 *
 * Requires the FULL corpus loaded. If the table is empty the tests skip (fresh
 * checkout has no committed corpus); a PARTIAL load is a hard failure, not a
 * skip, so we never silently evaluate on an incomplete index.
 */

const MIN_DOCS = 5000;
let loaded = false;
let result: EvalResult | null = null;

beforeAll(async () => {
  const { rows } = await query<{ total: string; embedded: string }>(
    "SELECT count(*) AS total, count(embedding) AS embedded FROM docs",
  );
  const total = Number(rows[0]?.total ?? 0);
  const embedded = Number(rows[0]?.embedded ?? 0);

  if (total === 0) {
    loaded = false; // fresh checkout, no corpus — skip
    return;
  }
  if (embedded !== total || total < MIN_DOCS) {
    throw new Error(
      `corpus incompletely loaded (${embedded}/${total} embedded) — run: npm run db:reset && npm run load`,
    );
  }
  loaded = true;
  result = await evaluate(10, 100);
}, 180_000);

afterAll(async () => {
  await close();
});

describe("hybrid search recall@10 (SciFact)", () => {
  it("hybrid significantly beats keyword-only", (ctx) => {
    if (!loaded || !result) return ctx.skip();
    const t = signTest(result, "hybrid", "keyword");
    expect(t.aWins).toBeGreaterThan(t.bWins);
    expect(t.pValue).toBeLessThan(0.05);
  });

  it("hybrid significantly beats naive score addition (the incompatible-scales failure)", (ctx) => {
    if (!loaded || !result) return ctx.skip();
    expect(result.recallAt10.hybrid).toBeGreaterThan(result.recallAt10.naive);
    const t = signTest(result, "hybrid", "naive");
    expect(t.aWins).toBeGreaterThan(t.bWins);
    expect(t.pValue).toBeLessThan(0.05);
  });

  it("hybrid is never worse than the strong vector-only baseline", (ctx) => {
    if (!loaded || !result) return ctx.skip();
    expect(result.recallAt10.hybrid).toBeGreaterThanOrEqual(result.recallAt10.vector);
  });

  it("keyword and vector each fail on queries the other catches (complementarity)", (ctx) => {
    if (!loaded || !result) return ctx.skip();
    const c = complementarity(result);
    expect(c.vectorSavesKeyword).toBeGreaterThan(0);
    expect(c.keywordSavesVector).toBeGreaterThan(0);
  });
});
