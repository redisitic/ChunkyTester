import { createLLMClient } from '../llmClient'
import { estimateTokens } from '../tokenCounter'
import type { Chunk, ChunkResult, DocumentStructure, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

const SYSTEM_PROMPT = `You are a document chunking assistant for RAG systems.
Decompose the provided text into atomic propositions.
Rules:
- Each proposition must be a single, self-contained factual statement
- Resolve all pronouns and coreferences (e.g. "it", "they", "the company")
- Each proposition must be independently retrievable and understandable
- Do not add information not present in the source text
Return ONLY a JSON array: { "proposition": string, "sourceSpan": [startChar, endChar] }[]`

const agenticChunker: ChunkerDefinition = {
  id: 'agentic',
  label: 'Agentic Proposition Chunker',
  description: 'Decomposes text into atomic, self-contained factual propositions. Highest quality for dense knowledge retrieval.',
  paperRef: 'Chen et al., "Dense X Retrieval", 2023',
  configSchema: [
    { key: 'maxPropositionTokens', label: 'Max Proposition Tokens', description: 'Soft token cap per proposition', type: 'slider', min: 32, max: 256, step: 16, default: 64 },
    { key: 'decontextualize', label: 'Decontextualize', description: 'Resolve pronouns and implicit references', type: 'toggle', default: true },
  ],
  defaultConfig: { maxPropositionTokens: 64, decontextualize: true },

  async run(text, _structure, config, llmConfig): Promise<ChunkResult> {
    const start = Date.now()
    const client = createLLMClient(llmConfig)
    const maxTokens = config.maxPropositionTokens as number
    const decontextualize = config.decontextualize as boolean

    const user = decontextualize
      ? `Decompose this text into atomic propositions. Resolve all pronouns and implicit references. Each proposition should be at most ${maxTokens} tokens.\n\n${text}`
      : `Decompose this text into atomic propositions. Each proposition should be at most ${maxTokens} tokens.\n\n${text}`

    const result = await client.streamComplete(SYSTEM_PROMPT, user, 8192)

    const match = result.text.match(/\[[\s\S]*\]/)
    let parsed: { proposition: string; sourceSpan: [number, number] }[] = []
    try { if (match) parsed = JSON.parse(match[0]) } catch { /* empty */ }

    const chunks: Chunk[] = parsed.map((item, i) => ({
      index: i,
      text: item.proposition,
      start: item.sourceSpan?.[0] ?? 0,
      end: item.sourceSpan?.[1] ?? item.proposition.length,
      tokens: estimateTokens(item.proposition),
      rationale: decontextualize ? 'Pronouns and references resolved' : undefined,
    }))

    return {
      strategyId: 'agentic',
      strategyLabel: 'Agentic Proposition Chunker',
      chunks,
      durationMs: Date.now() - start,
      apiCallCount: 1,
      estimatedCost: result.inputTokens * client.costPerInputToken + result.outputTokens * client.costPerOutputToken,
    }
  },
}

export default agenticChunker
