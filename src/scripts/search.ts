import { search, type Method } from "../search.js";
import { query, close } from "../db.js";

/**
 *   npm run search -- how do vaccines reduce infection risk
 *   METHOD=keyword npm run search -- <query>   (keyword|vector|hybrid|naive)
 */
const q = process.argv.slice(2).join(" ").trim() || "how do vaccines reduce infection risk";
const method = (process.env.METHOD as Method) || "hybrid";

const results = await search(q, method, 10);
const ids = results.map((r) => r.id);
const { rows } = await query<{ id: string; title: string }>(
  "SELECT id, title FROM docs WHERE id = ANY($1)",
  [ids],
);
const titleById = new Map(rows.map((r) => [r.id, r.title]));

console.log(`\nquery: "${q}"    method: ${method}\n`);
results.forEach((r, i) => {
  console.log(
    `${String(i + 1).padStart(2)}. [${r.score.toFixed(4)}] ${(titleById.get(r.id) ?? "").slice(0, 90)}`,
  );
});

await close();
