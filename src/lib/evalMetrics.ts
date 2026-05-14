import { createLLMClient } from './llmClient'
import { estimateTokens, tokensToUsd } from './tokenCounter'
import type { Chunk, EvalResult, LLMConfig } from '@/types'

const STOPWORDS = new Set([
  'a','an','the','and','or','but','in','on','at','to','for','of','with',
  'is','was','are','were','be','been','being','have','has','had','do','does',
  'did','will','would','could','should','may','might','shall','can','need',
  'it','its','this','that','these','those','i','you','he','she','we','they',
])

async function scoreChunkSelfContainedness(
  chunk: Chunk,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<number> {
  const client = createLLMClient(llmConfig)
  const result = await client.complete(
    null,
    `Rate how self-contained this text chunk is on a scale of 0.0–1.0.
A score of 1.0 means the chunk is fully understandable without any surrounding context. A score of 0.0 means it is incomprehensible without context.
Return only a JSON object: { "score": float, "reason": string }

Chunk:
"${chunk.text.slice(0, 800)}"`,
    128
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens
  const match = result.text.match(/\{[\s\S]*?\}/)
  try { return match ? (JSON.parse(match[0]).score ?? 0.5) : 0.5 } catch { return 0.5 }
}

async function scoreBoundaryCoherence(
  chunkA: Chunk,
  chunkB: Chunk,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<number> {
  const client = createLLMClient(llmConfig)
  const result = await client.complete(
    null,
    `Given these two adjacent text segments, rate how natural the split point is on a scale of 0.0–1.0. A score of 1.0 means the boundary falls at a perfect semantic break. A score of 0.0 means it cuts mid-thought.
Return only: { "score": float, "reason": string }

Segment A (end): "...${chunkA.text.slice(-300)}"
Segment B (start): "${chunkB.text.slice(0, 300)}..."`,
    128
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens
  const match = result.text.match(/\{[\s\S]*?\}/)
  try { return match ? (JSON.parse(match[0]).score ?? 0.5) : 0.5 } catch { return 0.5 }
}

function computeTokenEfficiency(chunks: Chunk[]): number {
  let useful = 0, total = 0
  for (const chunk of chunks) {
    const words = chunk.text.toLowerCase().split(/\s+/)
    useful += words.filter(w => !STOPWORDS.has(w) && w.length > 1).length
    total += words.length
  }
  return total === 0 ? 0 : useful / total
}

function stdDev(values: number[]): number {
  if (values.length === 0) return 0
  const mean = values.reduce((s, v) => s + v, 0) / values.length
  return Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length)
}

function sampleIndices(total: number, maxSample: number): number[] {
  if (total <= maxSample) return Array.from({ length: total }, (_, i) => i)
  const step = total / maxSample
  return Array.from({ length: maxSample }, (_, i) => Math.floor(i * step))
}

export async function computeEvalMetrics(
  strategyId: string,
  chunks: Chunk[],
  llmConfig: LLMConfig,
  sampleSize = 10,
  onProgress?: (done: number, total: number) => void
): Promise<EvalResult & { totalInputTokens: number; totalOutputTokens: number }> {
  const selfSampleIdx = sampleIndices(chunks.length, sampleSize)
  const boundarySampleIdx = sampleIndices(Math.max(0, chunks.length - 1), 8)
  const usage = { input: 0, output: 0 }
  const totalCalls = selfSampleIdx.length + boundarySampleIdx.length
  let done = 0

  const selfScores: number[] = []
  for (const idx of selfSampleIdx) {
    await new Promise(r => setTimeout(r, 200))
    selfScores.push(await scoreChunkSelfContainedness(chunks[idx], llmConfig, usage))
    onProgress?.(++done, totalCalls)
  }

  const boundaryScores: number[] = []
  for (const idx of boundarySampleIdx) {
    await new Promise(r => setTimeout(r, 200))
    boundaryScores.push(await scoreBoundaryCoherence(chunks[idx], chunks[idx + 1], llmConfig, usage))
    onProgress?.(++done, totalCalls)
  }

  const tokenCounts = chunks.map(c => c.tokens)
  const avgTokens = tokenCounts.reduce((s, v) => s + v, 0) / (tokenCounts.length || 1)

  return {
    strategyId,
    avgSelfContainedness: selfScores.reduce((s, v) => s + v, 0) / (selfScores.length || 1),
    avgBoundaryCoherence: boundaryScores.reduce((s, v) => s + v, 0) / (boundaryScores.length || 1),
    tokenEfficiency: computeTokenEfficiency(chunks),
    chunkCount: chunks.length,
    avgTokens,
    stdDevTokens: stdDev(tokenCounts),
    estimatedIndexCost: tokensToUsd(chunks.reduce((s, c) => s + estimateTokens(c.embeddingInput ?? c.text), 0), 0),
    totalInputTokens: usage.input,
    totalOutputTokens: usage.output,
  }
}
