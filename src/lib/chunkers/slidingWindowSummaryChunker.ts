import { createLLMClient } from '../llmClient'
import { estimateTokens } from '../tokenCounter'
import type { Chunk, ChunkResult, DocumentStructure, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

function slidingWindows(text: string, chunkSize: number, overlap: number): { text: string; start: number }[] {
  const windows: { text: string; start: number }[] = []
  let offset = 0
  while (offset < text.length) {
    windows.push({ text: text.slice(offset, offset + chunkSize), start: offset })
    if (offset + chunkSize >= text.length) break
    offset += chunkSize - overlap
  }
  return windows
}

const slidingWindowSummaryChunker: ChunkerDefinition = {
  id: 'sliding_window_summary',
  label: 'Sliding Window + LLM Summaries',
  description: 'Each chunk gets an LLM-generated summary prepended as a context header for richer retrieval signal.',
  paperRef: 'Anthropic cookbook: "Contextual Retrieval", 2024',
  configSchema: [
    { key: 'chunkSize', label: 'Chunk Size (chars)', description: 'Character size of each window', type: 'slider', min: 200, max: 2000, step: 100, default: 800 },
    { key: 'overlap', label: 'Overlap (chars)', description: 'Character overlap between windows', type: 'slider', min: 0, max: 400, step: 50, default: 100 },
    { key: 'summaryMaxTokens', label: 'Summary Max Words', description: 'Maximum words in each chunk summary', type: 'slider', min: 10, max: 80, step: 5, default: 40 },
  ],
  defaultConfig: { chunkSize: 800, overlap: 100, summaryMaxTokens: 40 },

  async run(text, _structure, config, llmConfig): Promise<ChunkResult> {
    const start = Date.now()
    const client = createLLMClient(llmConfig)
    let totalInput = 0, totalOutput = 0

    const chunkSize = config.chunkSize as number
    const overlap = config.overlap as number
    const summaryMaxWords = config.summaryMaxTokens as number
    const windows = slidingWindows(text, chunkSize, overlap)

    const results = await Promise.allSettled(
      windows.map(async (w, i) => {
        if (i > 0) await new Promise(r => setTimeout(r, i * 200))
        const result = await client.complete(
          null,
          `In one sentence of at most ${summaryMaxWords} words, summarize what this text is about. Return only the summary, no preamble.\n\n${w.text}`,
          80
        )
        totalInput += result.inputTokens
        totalOutput += result.outputTokens
        return { ...w, summary: result.text.trim(), index: i }
      })
    )

    const chunks: Chunk[] = []
    for (const result of results) {
      if (result.status === 'fulfilled') {
        const { text: chunkText, start: chunkStart, summary, index } = result.value
        chunks.push({
          index, text: chunkText,
          start: chunkStart, end: chunkStart + chunkText.length,
          tokens: estimateTokens(chunkText),
          summary,
          embeddingInput: summary + '\n\n' + chunkText,
        })
      }
    }
    chunks.sort((a, b) => a.index - b.index)

    return {
      strategyId: 'sliding_window_summary',
      strategyLabel: 'Sliding Window + LLM Summaries',
      chunks,
      durationMs: Date.now() - start,
      apiCallCount: windows.length,
      estimatedCost: totalInput * client.costPerInputToken + totalOutput * client.costPerOutputToken,
    }
  },
}

export default slidingWindowSummaryChunker
