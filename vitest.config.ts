import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    // The recall evaluation runs hundreds of queries against Postgres.
    testTimeout: 120_000,
    hookTimeout: 120_000,
    // One DB, so don't run test files in parallel against it.
    fileParallelism: false,
  },
});
