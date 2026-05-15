import { createLLMClient } from './llmClient'
import { embedTexts } from './localEmbeddings'
import { cosineSimilarity } from './similarity'
import type { LLMConfig } from '@/types'

export async function scoreSimilarityPairs(
  pairs: [string, string][],
  llmConfig: LLMConfig
): Promise<number[]> {
  const client = createLLMClient(llmConfig)

  const prompt = `Score the semantic similarity of each text pair on a scale of 0.0 to 1.0.
Return ONLY a JSON array of numbers, one per pair, in order.

Pairs:
${pairs.map(([a, b], i) => `${i + 1}. A: "${a.slice(0, 300)}" | B: "${b.slice(0, 300)}"`).join('\n')}`

  const result = await client.complete(null, prompt, 256)
  const match = result.text.match(/\[[\d.,\s]+\]/)
  return match ? JSON.parse(match[0]) : pairs.map(() => 0.5)
}

export async function rankChunksByQuery(
  query: string,
  chunks: { index: number; text: string }[],
  topK: number,
  llmConfig: LLMConfig
): Promise<{ index: number; score: number }[]> {
  const client = createLLMClient(llmConfig)

  const prompt = `Query: "${query}"

Rank the following chunks by relevance to the query. Return a JSON array of objects with "index" and "score" (0.0–1.0), sorted by score descending, top ${topK} only.

Chunks:
${chunks.map(c => `[${c.index}] ${c.text.slice(0, 400)}`).join('\n\n')}`

  const result = await client.complete(null, prompt, 512)
  const match = result.text.match(/\[[\s\S]*?\]/)
  try {
    return match ? JSON.parse(match[0]) : []
  } catch {
    return []
  }
}

export async function rankChunksByEmbedding(
  query: string,
  chunks: { index: number; text: string }[],
  topK: number
): Promise<{ index: number; score: number }[]> {
  const texts = [query, ...chunks.map(c => c.text)]
  const embeddings = await embedTexts(texts)
  const queryVec = embeddings[0]

  return chunks
    .map((c, i) => ({ index: c.index, score: cosineSimilarity(queryVec, embeddings[i + 1]) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
}
