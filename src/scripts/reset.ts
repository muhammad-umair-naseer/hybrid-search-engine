import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { pool, close } from "../db.js";

// Apply schema.sql (drops + recreates the docs table and indexes).
const sql = readFileSync(fileURLToPath(new URL("../schema.sql", import.meta.url)), "utf8");
await pool.query(sql);
console.log("schema applied: docs table + tsv GIN index ready");
await close();
