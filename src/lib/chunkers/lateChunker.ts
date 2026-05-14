import { createLLMClient } from '../llmClient'
import { estimateTokens } from '../tokenCounter'
import type { Chunk, ChunkResult, DocumentStructure, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

const SYSTEM = `You are a document chunking assistant. You have read the full document and must identify chunk boundaries that minimize context dependency. Each chunk must be maximally self-contained given that you have seen the full document. Boundaries are determined by document-level coherence, not local text features.

Rules:
- Each chunk should be around {maxChunkTokens} tokens
- Use up to {contextWindowSize} tokens of surrounding context for boundary decisions
- Each chunk must be independently retrievable
- Return ONLY a JSON array: [{ "chunk": string, "contextDependencies": string[] }]`

const lateChunker: ChunkerDefinition = {
  id: 'late_chunking',
  label: 'Late Chunking',
  description: 'Embeds the full passage first so chunk boundaries reflect document-level coherence, not local text features.',
  paperRef: 'Günther et al., "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models", 2024',
  configSchema: [
    { key: 'maxChunkTokens', label: 'Max Chunk Tokens', description: 'Maximum tokens per chunk', type: 'slider', min: 64, max: 512, step: 32, default: 256 },
    { key: 'contextWindowSize', label: 'Context Window Size (tokens)', description: 'Surrounding context used for boundary decisions', type: 'slider', min: 128, max: 1024, step: 64, default: 512 },
  ],
  defaultConfig: { maxChunkTokens: 256, contextWindowSize: 512 },

  async run(text, _structure, config, llmConfig): Promise<ChunkResult> {
    const start = Date.now()
    const client = createLLMClient(llmConfig)
    const maxChunkTokens = config.maxChunkTokens as number
    const contextWindowSize = config.contextWindowSize as number

    const system = SYSTEM.replace('{maxChunkTokens}', String(maxChunkTokens)).replace('{contextWindowSize}', String(contextWindowSize))
    const result = await client.complete(system, `Chunk this text:\n\n${text}`, 4096)

    const match = result.text.match(/\[[\s\S]*\]/)
    let parsed: { chunk: string; contextDependencies: string[] }[] = []
    try { if (match) parsed = JSON.parse(match[0]) } catch { /* empty */ }

    const chunks: Chunk[] = parsed.map((item, i) => {
      const idx = text.indexOf(item.chunk)
      return {
        index: i, text: item.chunk,
        start: idx === -1 ? 0 : idx,
        end: idx === -1 ? item.chunk.length : idx + item.chunk.length,
        tokens: estimateTokens(item.chunk),
        contextDependencies: item.contextDependencies,
      }
    })

    return {
      strategyId: 'late_chunking',
      strategyLabel: 'Late Chunking',
      chunks,
      durationMs: Date.now() - start,
      apiCallCount: 1,
      estimatedCost: result.inputTokens * client.costPerInputToken + result.outputTokens * client.costPerOutputToken,
    }
  },
}

export default lateChunker
