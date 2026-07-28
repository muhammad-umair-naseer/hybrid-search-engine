import { pool, close, toVectorLiteral } from "../db.js";
import { embedBatch } from "../embed.js";
import { loadCorpus } from "../corpus.js";

/**
 * Load the SciFact corpus: insert every doc, then embed in batches and fill the
 * vector column. `npm run db:reset` first.
 */
const BATCH = 64;

const docs = loadCorpus();
console.log(`loading ${docs.length} docs...`);

// Insert text first (fast), embeddings second so a slow model run doesn't hold
// a transaction open.
for (let i = 0; i < docs.length; i += 500) {
  const chunk = docs.slice(i, i + 500);
  const values: string[] = [];
  const params: unknown[] = [];
  chunk.forEach((d, j) => {
    const b = j * 3;
    values.push(`($${b + 1}, $${b + 2}, $${b + 3})`);
    params.push(d.id, d.title, d.text);
  });
  await pool.query(
    `INSERT INTO docs (id, title, body) VALUES ${values.join(",")}
     ON CONFLICT (id) DO NOTHING`,
    params,
  );
}
console.log("text inserted; embedding...");

const t0 = Date.now();
let done = 0;
for (let i = 0; i < docs.length; i += BATCH) {
  const chunk = docs.slice(i, i + BATCH);
  const vecs = await embedBatch(chunk.map((d) => `${d.title}. ${d.text}`));
  // Update each doc's embedding.
  await Promise.all(
    chunk.map((d, j) =>
      pool.query("UPDATE docs SET embedding = $1::vector WHERE id = $2", [
        toVectorLiteral(vecs[j]!),
        d.id,
      ]),
    ),
  );
  done += chunk.length;
  if (done % 512 === 0 || done === docs.length) {
    const rate = done / ((Date.now() - t0) / 1000);
    console.log(`  embedded ${done}/${docs.length} (${rate.toFixed(0)} docs/s)`);
  }
}

const { rows } = await pool.query(
  "SELECT count(*) AS n, count(embedding) AS embedded FROM docs",
);
console.log(`done: ${rows[0].n} docs, ${rows[0].embedded} embedded`);
await close();
