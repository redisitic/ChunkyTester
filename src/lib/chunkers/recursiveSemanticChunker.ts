import { createLLMClient } from '../llmClient'
import { estimateTokens } from '../tokenCounter'
import { scoreSimilarityPairs } from '../embeddings'
import type { Chunk, ChunkResult, DocumentStructure, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

const SEPARATORS = ['\n\n', '. ', ' ']

function recursiveSplit(text: string, maxChars: number, sepIdx = 0): string[] {
  if (text.length <= maxChars || sepIdx >= SEPARATORS.length) return [text]
  const sep = SEPARATORS[sepIdx]
  const parts = text.split(sep)
  const chunks: string[] = []
  let current = ''
  for (const part of parts) {
    const candidate = current ? current + sep + part : part
    if (candidate.length <= maxChars) {
      current = candidate
    } else {
      if (current) chunks.push(current)
      current = part.length > maxChars ? recursiveSplit(part, maxChars, sepIdx + 1).join('') : part
    }
  }
  if (current) chunks.push(current)
  return chunks
}

async function askCoherence(
  chunk: string,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<string[]> {
  const sentences = chunk.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 10)
  if (sentences.length <= 2) return [chunk]

  const client = createLLMClient(llmConfig)
  const result = await client.complete(
    null,
    `Given this text, return the index (0-based) of the sentence that has the weakest semantic connection to its neighbors. Return only a JSON object: { "index": number }\n\nText sentences:\n${sentences.map((s, i) => `${i}: ${s}`).join('\n')}`,
    64
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens

  const match = result.text.match(/\{[\s\S]*?\}/)
  let splitIdx = Math.floor(sentences.length / 2)
  try { if (match) splitIdx = JSON.parse(match[0]).index ?? splitIdx } catch { /* use default */ }
  splitIdx = Math.max(1, Math.min(splitIdx, sentences.length - 1))
  return [sentences.slice(0, splitIdx).join(' '), sentences.slice(splitIdx).join(' ')]
}

const recursiveSemanticChunker: ChunkerDefinition = {
  id: 'recursive_semantic',
  label: 'Recursive + Semantic Hybrid',
  description: 'Splits recursively then uses Claude to merge over-split pairs and re-split incoherent chunks.',
  paperRef: 'LangChain SemanticChunker docs',
  configSchema: [
    { key: 'targetChunkSize', label: 'Target Chunk Size (chars)', description: 'Target character count per chunk', type: 'slider', min: 200, max: 2000, step: 100, default: 600 },
    { key: 'coherenceThreshold', label: 'Coherence Threshold', description: 'Chunks below this score are re-split (0–1)', type: 'slider', min: 0, max: 1, step: 0.05, default: 0.75 },
    { key: 'mergeThreshold', label: 'Merge Threshold', description: 'Adjacent chunks above this score are merged (0–1)', type: 'slider', min: 0, max: 1, step: 0.05, default: 0.92 },
  ],
  defaultConfig: { targetChunkSize: 600, coherenceThreshold: 0.75, mergeThreshold: 0.92 },

  async run(text, _structure, config, llmConfig): Promise<ChunkResult> {
    const start = Date.now()
    const usage = { input: 0, output: 0 }
    let apiCallCount = 0

    const targetSize = config.targetChunkSize as number
    const coherenceThreshold = config.coherenceThreshold as number
    const mergeThreshold = config.mergeThreshold as number

    let segments = recursiveSplit(text, targetSize)

    if (segments.length > 1) {
      const pairs: [string, string][] = segments.slice(0, -1).map((s, i) => [s, segments[i + 1]])
      const scores = await scoreSimilarityPairs(pairs, llmConfig)
      apiCallCount++
      usage.input += estimateTokens(pairs.map(p => p[0] + p[1]).join(''))
      usage.output += scores.length * 4

      const merged: string[] = []
      let i = 0
      while (i < segments.length) {
        if (i < scores.length && scores[i] >= mergeThreshold) {
          merged.push(segments[i] + ' ' + segments[i + 1])
          i += 2
        } else {
          merged.push(segments[i])
          i++
        }
      }
      segments = merged
    }

    const finalSegments: string[] = []
    for (const seg of segments) {
      const [score] = await scoreSimilarityPairs([[seg, seg]], llmConfig)
      apiCallCount++
      if (score < coherenceThreshold) {
        finalSegments.push(...await askCoherence(seg, llmConfig, usage))
        apiCallCount++
      } else {
        finalSegments.push(seg)
      }
      await new Promise(r => setTimeout(r, 200))
    }

    const client = createLLMClient(llmConfig)
    const chunks: Chunk[] = finalSegments.map((seg, i) => {
      const idx = text.indexOf(seg)
      return { index: i, text: seg, start: idx === -1 ? 0 : idx, end: idx === -1 ? seg.length : idx + seg.length, tokens: estimateTokens(seg) }
    })

    return {
      strategyId: 'recursive_semantic',
      strategyLabel: 'Recursive + Semantic Hybrid',
      chunks,
      durationMs: Date.now() - start,
      apiCallCount,
      estimatedCost: usage.input * client.costPerInputToken + usage.output * client.costPerOutputToken,
    }
  },
}

export default recursiveSemanticChunker
