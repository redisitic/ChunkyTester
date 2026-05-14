import { createLLMClient } from '../llmClient'
import { estimateTokens } from '../tokenCounter'
import type { Chunk, ChunkResult, DocumentStructure, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

async function semanticSplit(
  text: string,
  maxTokens: number,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<string[]> {
  const client = createLLMClient(llmConfig)
  const result = await client.complete(
    null,
    `Split this text into chunks of at most ${maxTokens} tokens each at natural semantic boundaries. Return ONLY a JSON array of strings.\n\n${text}`,
    256
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens
  const match = result.text.match(/\[[\s\S]*\]/)
  try { return match ? JSON.parse(match[0]) : [text] } catch { return [text] }
}

const structureAwareChunker: ChunkerDefinition = {
  id: 'structure_aware',
  label: 'Document-Structure-Aware',
  description: 'Uses heading hierarchy to define chunk boundaries, never splitting a section unless it exceeds the max size.',
  paperRef: 'LlamaIndex Hierarchical Node Parser docs',
  configSchema: [
    { key: 'maxChunkTokens', label: 'Max Chunk Tokens', description: 'Maximum tokens per chunk before semantic split', type: 'slider', min: 128, max: 1024, step: 64, default: 512 },
    { key: 'minChunkTokens', label: 'Min Chunk Tokens', description: 'Sections smaller than this are merged with the next sibling', type: 'slider', min: 20, max: 300, step: 10, default: 100 },
    { key: 'headingLevels', label: 'Max Heading Level to Split', description: 'Heading levels 1 through N create boundaries', type: 'slider', min: 1, max: 6, step: 1, default: 3 },
  ],
  defaultConfig: { maxChunkTokens: 512, minChunkTokens: 100, headingLevels: 3 },

  async run(text, structure, config, llmConfig): Promise<ChunkResult> {
    const start = Date.now()
    const usage = { input: 0, output: 0 }
    let apiCallCount = 0

    const maxTokens = config.maxChunkTokens as number
    const minTokens = config.minChunkTokens as number
    const maxLevel = config.headingLevels as number
    const headings = structure.headings.filter(h => h.level <= maxLevel)

    if (headings.length === 0) {
      const { default: recursive } = await import('./recursiveSemanticChunker')
      return recursive.run(text, structure, { targetChunkSize: maxTokens * 4, coherenceThreshold: 0.75, mergeThreshold: 0.92 }, llmConfig)
    }

    const sections: { text: string }[] = []
    for (let i = 0; i < headings.length; i++) {
      const sectionText = text.slice(headings[i].start, headings[i + 1]?.start ?? text.length).trim()
      if (sectionText) sections.push({ text: sectionText })
    }

    const merged: { text: string }[] = []
    let buffer = ''
    for (const section of sections) {
      if (estimateTokens(section.text) < minTokens) {
        buffer += ' ' + section.text
      } else {
        if (buffer) { merged.push({ text: buffer.trim() }); buffer = '' }
        merged.push({ text: section.text })
      }
    }
    if (buffer) merged.push({ text: buffer.trim() })

    const finalSegments: string[] = []
    for (const section of merged) {
      if (estimateTokens(section.text) > maxTokens) {
        finalSegments.push(...await semanticSplit(section.text, maxTokens, llmConfig, usage))
        apiCallCount++
      } else {
        finalSegments.push(section.text)
      }
      await new Promise(r => setTimeout(r, 200))
    }

    const client = createLLMClient(llmConfig)
    const chunks: Chunk[] = finalSegments.map((seg, i) => {
      const idx = text.indexOf(seg)
      return { index: i, text: seg, start: idx === -1 ? 0 : idx, end: idx === -1 ? seg.length : idx + seg.length, tokens: estimateTokens(seg) }
    })

    return {
      strategyId: 'structure_aware',
      strategyLabel: 'Document-Structure-Aware',
      chunks,
      durationMs: Date.now() - start,
      apiCallCount,
      estimatedCost: usage.input * client.costPerInputToken + usage.output * client.costPerOutputToken,
    }
  },
}

export default structureAwareChunker
