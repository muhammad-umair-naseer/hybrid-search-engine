import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

/**
 * Local sentence embeddings via transformers.js — no API key, fully
 * reproducible. all-MiniLM-L6-v2 maps text to a 384-dim unit vector; because
 * vectors are L2-normalized, cosine similarity is just their dot product, and
 * pgvector's cosine distance operator works directly.
 */
export const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
export const EMBED_DIM = 384;

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;

function getExtractor(): Promise<FeatureExtractionPipeline> {
  extractorPromise ??= pipeline("feature-extraction", MODEL_ID);
  return extractorPromise;
}

export async function embed(text: string): Promise<number[]> {
  const [vec] = await embedBatch([text]);
  return vec!;
}

export async function embedBatch(texts: string[]): Promise<number[][]> {
  const extractor = await getExtractor();
  const output = await extractor(texts, { pooling: "mean", normalize: true });
  const [n, dim] = output.dims as [number, number];
  const data = output.data as Float32Array;
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    out.push(Array.from(data.subarray(i * dim, (i + 1) * dim)));
  }
  return out;
}
