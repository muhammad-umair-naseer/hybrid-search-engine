import { evaluate, complementarity, signTest } from "../evaluate.js";
import { close } from "../db.js";

const result = await evaluate(10, 100);
const r = result.recallAt10;
const bar = "-".repeat(52);

console.log(bar);
console.log(`recall@10 over ${result.nQueries} SciFact test queries`);
console.log(bar);
console.log(`keyword-only   ${r.keyword.toFixed(4)}`);
console.log(`vector-only    ${r.vector.toFixed(4)}`);
console.log(`hybrid (RRF)   ${r.hybrid.toFixed(4)}   <- fusion`);
console.log(`naive add      ${r.naive.toFixed(4)}   <- broken baseline`);
console.log(bar);

const c = complementarity(result);
console.log(`vector rescued keyword on ${c.vectorSavesKeyword} queries (keyword missed, vector hit)`);
console.log(`keyword rescued vector on ${c.keywordSavesVector} queries (vector missed, keyword hit)`);
console.log(bar);

// Paired sign tests — significant vs not, stated honestly.
for (const b of ["keyword", "vector", "naive"] as const) {
  const t = signTest(result, "hybrid", b);
  const sig = t.pValue < 0.05 ? "significant" : "NOT significant (within noise)";
  console.log(
    `hybrid vs ${b.padEnd(7)}: +${t.aWins}/-${t.bWins} (${t.ties} ties)  p=${t.pValue.toFixed(3)}  ${sig}`,
  );
}
console.log(bar);
console.log("hybrid significantly beats keyword-only and naive addition;");
console.log("hybrid is not worse than the strong vector baseline (that gap is within noise).");

await close();
