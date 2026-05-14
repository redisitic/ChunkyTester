import { estimateTokens } from '../tokenCounter'
import { scoreSimilarityPairs } from '../embeddings'
import { percentile } from '../similarity'
import { createLLMClient } from '../llmClient'
import type { Chunk, ChunkResult, DocumentStructure, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

function splitSentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(s => s.length > 0)
}

const kamradtSemanticChunker: ChunkerDefinition = {
  id: 'kamradt_semantic',
  label: 'Kamradt Semantic Chunker',
  description: "Splits at the Nth percentile of similarity drops between adjacent sentences — adapts to document vocabulary rather than a fixed threshold.",
  paperRef: 'Greg Kamradt, "5 Levels of Text Splitting" notebook, 2024',
  configSchema: [
    { key: 'breakpointPercentile', label: 'Breakpoint Percentile', description: 'Top N% of similarity drops become chunk boundaries', type: 'slider', min: 50, max: 99, step: 1, default: 85 },
    { key: 'minChunkSentences', label: 'Min Chunk Sentences', description: 'Minimum sentences per chunk', type: 'slider', min: 1, max: 10, step: 1, default: 2 },
  ],
  defaultConfig: { breakpointPercentile: 85, minChunkSentences: 2 },

  async run(text, _structure, config, llmConfig): Promise<ChunkResult> {
    const start = Date.now()
    const breakpointPercentile = config.breakpointPercentile as number
    const minSentences = config.minChunkSentences as number
    const sentences = splitSentences(text)

    if (sentences.length <= 1) {
      return { strategyId: 'kamradt_semantic', strategyLabel: 'Kamradt Semantic Chunker', chunks: [{ index: 0, text, start: 0, end: text.length, tokens: estimateTokens(text) }], durationMs: Date.now() - start, apiCallCount: 0, estimatedCost: 0 }
    }

    const pairs: [string, string][] = sentences.slice(0, -1).map((s, i) => [s, sentences[i + 1]])
    const similarities = await scoreSimilarityPairs(pairs, llmConfig)
    const drops = similarities.map(s => 1 - s)
    const threshold = percentile(drops, breakpointPercentile)

    const boundaries = new Set<number>()
    drops.forEach((drop, i) => { if (drop >= threshold) boundaries.add(i + 1) })

    const chunkSentences: string[][] = []
    let current: string[] = []
    sentences.forEach((s, i) => {
      current.push(s)
      if (boundaries.has(i + 1) && current.length >= minSentences) {
        chunkSentences.push(current)
        current = []
      }
    })
    if (current.length > 0) {
      if (chunkSentences.length > 0 && current.length < minSentences) {
        chunkSentences[chunkSentences.length - 1].push(...current)
      } else {
        chunkSentences.push(current)
      }
    }

    let cursor = 0
    const chunks: Chunk[] = chunkSentences.map((sentGroup, i) => {
      const chunkText = sentGroup.join(' ')
      const idx = text.indexOf(chunkText, cursor)
      const chunkStart = idx === -1 ? cursor : idx
      cursor = chunkStart + chunkText.length
      return { index: i, text: chunkText, start: chunkStart, end: chunkStart + chunkText.length, tokens: estimateTokens(chunkText) }
    })

    const client = createLLMClient(llmConfig)
    const inputTokens = estimateTokens(pairs.map(p => p[0] + p[1]).join(''))
    return {
      strategyId: 'kamradt_semantic',
      strategyLabel: 'Kamradt Semantic Chunker',
      chunks,
      durationMs: Date.now() - start,
      apiCallCount: 1,
      estimatedCost: inputTokens * client.costPerInputToken + similarities.length * 4 * client.costPerOutputToken,
    }
  },
}

export default kamradtSemanticChunker
