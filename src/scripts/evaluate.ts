import { evaluate, complementarity } from "../evaluate.js";
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
console.log(
  r.hybrid > r.keyword && r.hybrid > r.vector
    ? "hybrid beats BOTH single methods."
    : "hybrid did NOT beat both — investigate.",
);
console.log(r.hybrid > r.naive ? "RRF beats naive addition." : "naive did not degrade — investigate.");

await close();
