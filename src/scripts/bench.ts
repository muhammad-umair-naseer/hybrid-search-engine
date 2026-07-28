import { performance } from "node:perf_hooks";
import { embed } from "../embed.js";
import { keywordSearch, vectorSearch } from "../retrieval.js";
import { fuseRRF } from "../fusion.js";
import { FUSION_WEIGHTS } from "../search.js";
import { close } from "../db.js";

const q = "stem cells can be guided to differentiate into neurons";
const N = 50;

async function timeIt(fn: () => Promise<unknown>): Promise<number> {
  await fn(); // warm
  const t0 = performance.now();
  for (let i = 0; i < N; i++) await fn();
  return (performance.now() - t0) / N;
}

const embedMs = await timeIt(() => embed(q));
const qvec = await embed(q);
const kwMs = await timeIt(() => keywordSearch(q, 100));
const vecMs = await timeIt(() => vectorSearch(qvec, 100));
const fuseMs = await timeIt(async () => {
  const [k, v] = await Promise.all([keywordSearch(q, 100), vectorSearch(qvec, 100)]);
  fuseRRF([k, v], FUSION_WEIGHTS);
});

const bar = "-".repeat(52);
console.log(bar);
console.log(`per-query latency (5183 docs, mean of ${N})`);
console.log(bar);
console.log(`embed query          ${embedMs.toFixed(1)} ms`);
console.log(`keyword (FTS)        ${kwMs.toFixed(1)} ms`);
console.log(`vector kNN (exact)   ${vecMs.toFixed(1)} ms`);
console.log(`retrieve both + fuse ${fuseMs.toFixed(1)} ms`);
console.log(`hybrid end-to-end   ~${(embedMs + fuseMs).toFixed(1)} ms`);
console.log(bar);

await close();
