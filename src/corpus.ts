import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

/**
 * SciFact (BEIR) loader. The corpus ships as JSONL + a qrels TSV of relevance
 * judgements, so our recall numbers are comparable to published baselines.
 */

const DATA_DIR = fileURLToPath(new URL("../data/scifact/", import.meta.url));

export interface Doc {
  id: string;
  title: string;
  text: string;
}

function readJsonl(path: string): Record<string, unknown>[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

export function loadCorpus(): Doc[] {
  return readJsonl(DATA_DIR + "corpus.jsonl").map((d) => ({
    id: String(d._id),
    title: String(d.title ?? ""),
    text: String(d.text ?? ""),
  }));
}

export type Split = "test" | "train";

/** queryId -> query text, restricted to queries that have judgements in `split`. */
export function loadQueries(split: Split = "test"): Map<string, string> {
  const relevant = loadQrels(split);
  const all = new Map<string, string>();
  for (const q of readJsonl(DATA_DIR + "queries.jsonl")) {
    all.set(String(q._id), String(q.text ?? ""));
  }
  const out = new Map<string, string>();
  for (const qid of relevant.keys()) {
    const text = all.get(qid);
    if (text) out.set(qid, text);
  }
  return out;
}

/** queryId -> set of relevant docIds (qrels/<split>.tsv, score > 0) */
export function loadQrels(split: Split = "test"): Map<string, Set<string>> {
  const rows = readFileSync(DATA_DIR + `qrels/${split}.tsv`, "utf8")
    .split("\n")
    .slice(1) // header: query-id  corpus-id  score
    .filter((line) => line.trim().length > 0);

  const qrels = new Map<string, Set<string>>();
  for (const row of rows) {
    const [qid, docId, score] = row.split("\t");
    if (!qid || !docId) continue;
    if (Number(score) <= 0) continue;
    (qrels.get(qid) ?? qrels.set(qid, new Set()).get(qid)!).add(docId);
  }
  return qrels;
}
