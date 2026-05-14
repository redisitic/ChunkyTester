import type { ChunkResult, DocumentStructure, ConfigField, LLMConfig } from '@/types'

export interface ChunkerDefinition {
  id: string
  label: string
  description: string
  paperRef?: string
  configSchema: ConfigField[]
  defaultConfig: Record<string, number | boolean>
  run: (
    text: string,
    structure: DocumentStructure,
    config: Record<string, number | boolean>,
    llmConfig: LLMConfig
  ) => Promise<ChunkResult>
}
