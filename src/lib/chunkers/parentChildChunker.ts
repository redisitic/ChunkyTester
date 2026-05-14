import { estimateTokens } from '../tokenCounter'
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

const parentChildChunker: ChunkerDefinition = {
  id: 'parent_child',
  label: 'Parent-Child (Small-to-Big)',
  description: 'Two-level chunking: small child chunks for precise retrieval, large parent chunks for LLM generation context.',
  paperRef: 'LlamaIndex "Small-to-Big" retrieval pattern',
  configSchema: [
    { key: 'parentChunkTokens', label: 'Parent Chunk Tokens', description: 'Token size of parent chunks', type: 'slider', min: 256, max: 1024, step: 64, default: 512 },
    { key: 'childChunkTokens', label: 'Child Chunk Tokens', description: 'Token size of child chunks', type: 'slider', min: 32, max: 256, step: 16, default: 128 },
    { key: 'childOverlap', label: 'Child Overlap (tokens)', description: 'Overlap between adjacent child chunks', type: 'slider', min: 0, max: 64, step: 8, default: 16 },
  ],
  defaultConfig: { parentChunkTokens: 512, childChunkTokens: 128, childOverlap: 16 },

  async run(text, _structure, config, _llmConfig: LLMConfig): Promise<ChunkResult> {
    const start = Date.now()
    const CHARS_PER_TOKEN = 4
    const parentMaxChars = (config.parentChunkTokens as number) * CHARS_PER_TOKEN
    const childMaxChars = (config.childChunkTokens as number) * CHARS_PER_TOKEN
    const overlapChars = (config.childOverlap as number) * CHARS_PER_TOKEN

    const parentTexts = recursiveSplit(text, parentMaxChars)
    const chunks: Chunk[] = []
    let globalIndex = 0

    for (let pi = 0; pi < parentTexts.length; pi++) {
      const parentText = parentTexts[pi]
      const parentStart = text.indexOf(parentText)
      const parentChunkIndex = globalIndex++

      const childTexts: string[] = []
      let offset = 0
      while (offset < parentText.length) {
        childTexts.push(parentText.slice(offset, offset + childMaxChars))
        offset += childMaxChars - overlapChars
        if (offset + childMaxChars > parentText.length && offset < parentText.length) {
          childTexts.push(parentText.slice(offset))
          break
        }
      }

      const childIndices: number[] = []
      for (const childText of childTexts) {
        const ci = globalIndex++
        childIndices.push(ci)
        const childStart = parentStart === -1 ? 0 : parentStart + parentText.indexOf(childText)
        chunks.push({
          index: ci,
          text: childText,
          start: childStart,
          end: childStart + childText.length,
          tokens: estimateTokens(childText),
          parentIndex: parentChunkIndex,
        })
      }

      chunks.splice(
        chunks.findIndex(c => c.index === parentChunkIndex) === -1 ? chunks.length : 0,
        0,
        {
          index: parentChunkIndex,
          text: parentText,
          start: parentStart === -1 ? 0 : parentStart,
          end: parentStart === -1 ? parentText.length : parentStart + parentText.length,
          tokens: estimateTokens(parentText),
          children: childIndices,
        }
      )
    }

    chunks.sort((a, b) => a.index - b.index)

    return {
      strategyId: 'parent_child',
      strategyLabel: 'Parent-Child (Small-to-Big)',
      chunks,
      durationMs: Date.now() - start,
      apiCallCount: 0,
      estimatedCost: 0,
    }
  },
}

export default parentChildChunker
