import { loadQueries, loadQrels } from "../corpus.js";
import { retrieveLists, type RawLists } from "../search.js";
import { fuseRRF } from "../fusion.js";
import { recallAt } from "../evaluate.js";
import { close } from "../db.js";

/**
 * Tune the single RRF keyword weight on the TRAIN split (never the test set),
 * so the reported test result is not tuned on itself.
 */
const queries = loadQueries("train");
const qrels = loadQrels("train");

const cached: Array<{ lists: RawLists; relevant: Set<string> }> = [];
let n = 0;
for (const [qid, qtext] of queries) {
  const relevant = qrels.get(qid);
  if (!relevant || relevant.size === 0) continue;
  cached.push({ lists: await retrieveLists(qtext, 100), relevant });
  if (++n % 100 === 0) console.error(`  retrieved ${n} train queries...`);
}

const candidates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0];
console.log(`tuning keyword weight on ${cached.length} train queries (vector weight = 1)\n`);
console.log("keyword_weight   train recall@10");

let best = { kw: 1, recall: -1 };
for (const kw of candidates) {
  let sum = 0;
  for (const { lists, relevant } of cached) {
    sum += recallAt(fuseRRF([lists.keyword, lists.vector], [kw, 1]), relevant, 10);
  }
  const recall = sum / cached.length;
  console.log(`   ${kw.toFixed(2)}             ${recall.toFixed(4)}`);
  if (recall > best.recall) best = { kw, recall };
}
console.log(`\nbest keyword weight (train): ${best.kw}  ->  recall@10 ${best.recall.toFixed(4)}`);
console.log("set FUSION_WEIGHTS = [<this>, 1] in src/search.ts");

await close();
