export type LLMProvider = 'anthropic' | 'gemini' | 'ollama'

export interface LLMConfig {
  provider: LLMProvider
  anthropicKey?: string
  geminiKey?: string
  geminiModel?: string
  ollamaBaseUrl?: string
  ollamaModel?: string
}

export interface Chunk {
  index: number
  text: string
  start: number
  end: number
  tokens: number
  summary?: string
  contextDependencies?: string[]
  rationale?: string
  parentIndex?: number
  children?: number[]
  embeddingInput?: string
}

export interface ChunkResult {
  strategyId: string
  strategyLabel: string
  chunks: Chunk[]
  durationMs: number
  apiCallCount: number
  estimatedCost: number
  error?: string
}

export interface EvalResult {
  strategyId: string
  avgSelfContainedness: number
  avgBoundaryCoherence: number
  tokenEfficiency: number
  retrievalPrecision?: number
  retrievalRecall?: number
  chunkCount: number
  avgTokens: number
  stdDevTokens: number
  estimatedIndexCost: number
}

export interface QueryResult {
  strategyId: string
  rankedChunks: { chunk: Chunk; score: number; rank: number }[]
}

export interface DocumentStructure {
  headings: { level: number; text: string; start: number; end: number }[]
  pages?: { number: number; start: number; end: number }[]
}

export interface DocumentState {
  filename: string
  fullText: string
  structure: DocumentStructure
  selectedStart: number
  selectedEnd: number
}

export interface ConfigField {
  key: string
  label: string
  description: string
  type: 'slider' | 'toggle'
  min?: number
  max?: number
  step?: number
  default: number | boolean
}

export interface Strategy {
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
