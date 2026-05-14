import { useState, useCallback, useRef } from 'react'
import { getChunker } from '@/lib/chunkerRegistry'
import type { ChunkResult, DocumentStructure, LLMConfig } from '@/types'

interface ColumnState {
  strategyId: string
  config: Record<string, number | boolean>
  result: ChunkResult | null
  running: boolean
  progress: number
  error: string | null
}

interface UseChunkingReturn {
  columns: ColumnState[]
  addColumn: (strategyId: string) => void
  removeColumn: (index: number) => void
  setColumnStrategy: (index: number, strategyId: string) => void
  setColumnConfig: (index: number, key: string, value: number | boolean) => void
  runColumn: (index: number, text: string, structure: DocumentStructure, llmConfig: LLMConfig) => Promise<void>
  runAll: (text: string, structure: DocumentStructure, llmConfig: LLMConfig) => Promise<void>
}

function makeDefaultColumn(strategyId: string): ColumnState {
  const chunker = getChunker(strategyId)
  return {
    strategyId,
    config: chunker ? { ...chunker.defaultConfig } : {},
    result: null,
    running: false,
    progress: 0,
    error: null,
  }
}

function cacheKey(text: string, strategyId: string, config: Record<string, number | boolean>): string {
  return `${strategyId}::${JSON.stringify(config)}::${text.length}::${text.slice(0, 64)}`
}

export function useChunking(): UseChunkingReturn {
  const [columns, setColumns] = useState<ColumnState[]>([
    makeDefaultColumn('recursive_semantic'),
    makeDefaultColumn('late_chunking'),
  ])
  const cache = useRef(new Map<string, ChunkResult>())

  const updateColumn = useCallback((index: number, patch: Partial<ColumnState>) => {
    setColumns(prev => prev.map((col, i) => i === index ? { ...col, ...patch } : col))
  }, [])

  const addColumn = useCallback((strategyId: string) => {
    setColumns(prev => [...prev, makeDefaultColumn(strategyId)])
  }, [])

  const removeColumn = useCallback((index: number) => {
    setColumns(prev => prev.filter((_, i) => i !== index))
  }, [])

  const setColumnStrategy = useCallback((index: number, strategyId: string) => {
    const chunker = getChunker(strategyId)
    setColumns(prev => prev.map((col, i) =>
      i === index
        ? { ...col, strategyId, config: chunker ? { ...chunker.defaultConfig } : {}, result: null, error: null }
        : col
    ))
  }, [])

  const setColumnConfig = useCallback((index: number, key: string, value: number | boolean) => {
    setColumns(prev => prev.map((col, i) =>
      i === index ? { ...col, config: { ...col.config, [key]: value }, result: null } : col
    ))
  }, [])

  const runColumn = useCallback(async (
    index: number,
    text: string,
    structure: DocumentStructure,
    llmConfig: LLMConfig
  ) => {
    const col = columns[index]
    const chunker = getChunker(col.strategyId)
    if (!chunker) return

    const key = cacheKey(text, col.strategyId, col.config)
    const cached = cache.current.get(key)
    if (cached) {
      updateColumn(index, { result: cached, error: null })
      return
    }

    updateColumn(index, { running: true, progress: 0, error: null, result: null })
    try {
      const result = await chunker.run(text, structure, col.config, llmConfig)
      cache.current.set(key, result)
      updateColumn(index, { running: false, result, progress: 100 })
    } catch (err) {
      updateColumn(index, {
        running: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        progress: 0,
      })
    }
  }, [columns, updateColumn])

  const runAll = useCallback(async (text: string, structure: DocumentStructure, llmConfig: LLMConfig) => {
    await Promise.allSettled(
      columns.map((_, i) => runColumn(i, text, structure, llmConfig))
    )
  }, [columns, runColumn])

  return { columns, addColumn, removeColumn, setColumnStrategy, setColumnConfig, runColumn, runAll }
}
