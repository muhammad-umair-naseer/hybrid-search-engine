import "dotenv/config";
import pg from "pg";

// node-postgres reads PGHOST/PGPORT/PGUSER/PGDATABASE/PGPASSWORD from the env.
export const pool = new pg.Pool();

export async function query<T extends pg.QueryResultRow = pg.QueryResultRow>(
  text: string,
  params: unknown[] = [],
): Promise<pg.QueryResult<T>> {
  return pool.query<T>(text, params);
}

/** pgvector accepts an array as the text literal '[a,b,c]'. */
export function toVectorLiteral(v: number[]): string {
  return `[${v.join(",")}]`;
}

export async function close(): Promise<void> {
  await pool.end();
}
