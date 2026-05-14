import { useState, useCallback } from 'react'
import { computeEvalMetrics } from '@/lib/evalMetrics'
import { rankChunksByQuery } from '@/lib/embeddings'
import type { ChunkResult, EvalResult, QueryResult, Chunk, LLMConfig } from '@/types'

interface EvalState {
  results: Record<string, EvalResult>
  running: boolean
  progress: number
  total: number
  error: string | null
}

interface QueryState {
  query: string
  results: Record<string, QueryResult>
  running: boolean
  relevanceJudgments: Record<string, Record<number, boolean>>
}

interface UseEvalReturn {
  evalState: EvalState
  queryState: QueryState
  runEval: (chunkResults: ChunkResult[], llmConfig: LLMConfig, sampleSize?: number) => Promise<void>
  setQuery: (query: string) => void
  runQuery: (chunkResults: ChunkResult[], llmConfig: LLMConfig, topK?: number) => Promise<void>
  judgeRelevance: (strategyId: string, chunkIndex: number, relevant: boolean) => void
  precisionAt: (strategyId: string, k: number) => number | null
}

export function useEval(): UseEvalReturn {
  const [evalState, setEvalState] = useState<EvalState>({
    results: {},
    running: false,
    progress: 0,
    total: 0,
    error: null,
  })

  const [queryState, setQueryState] = useState<QueryState>({
    query: '',
    results: {},
    running: false,
    relevanceJudgments: {},
  })

  const runEval = useCallback(async (
    chunkResults: ChunkResult[],
    llmConfig: LLMConfig,
    sampleSize = 10
  ) => {
    setEvalState(prev => ({ ...prev, running: true, error: null, progress: 0, total: chunkResults.length }))

    const allResults: Record<string, EvalResult> = {}
    let done = 0

    await Promise.allSettled(
      chunkResults.map(async cr => {
        try {
          const result = await computeEvalMetrics(
            cr.strategyId,
            cr.chunks,
            llmConfig,
            sampleSize,
            (callsDone, callsTotal) => {
              setEvalState(prev => ({
                ...prev,
                progress: Math.round(((done + callsDone / callsTotal) / chunkResults.length) * 100),
              }))
            }
          )
          allResults[cr.strategyId] = result
        } catch (err) {
          allResults[cr.strategyId] = {
            strategyId: cr.strategyId,
            avgSelfContainedness: 0,
            avgBoundaryCoherence: 0,
            tokenEfficiency: 0,
            chunkCount: cr.chunks.length,
            avgTokens: 0,
            stdDevTokens: 0,
            estimatedIndexCost: 0,
          }
        } finally {
          done++
        }
      })
    )

    setEvalState(prev => ({ ...prev, running: false, progress: 100, results: allResults }))
  }, [])

  const setQuery = useCallback((query: string) => {
    setQueryState(prev => ({ ...prev, query }))
  }, [])

  const runQuery = useCallback(async (
    chunkResults: ChunkResult[],
    llmConfig: LLMConfig,
    topK = 5
  ) => {
    setQueryState(prev => ({ ...prev, running: true }))
    const allResults: Record<string, QueryResult> = {}

    await Promise.allSettled(
      chunkResults.map(async cr => {
        const childChunks = cr.chunks.filter(c => c.parentIndex === undefined || c.children !== undefined
          ? c.children !== undefined
          : true
        )

        const candidates = childChunks.length > 0
          ? childChunks.map(c => ({ index: c.index, text: c.embeddingInput ?? c.text }))
          : cr.chunks.map(c => ({ index: c.index, text: c.embeddingInput ?? c.text }))

        const ranked = await rankChunksByQuery(
          queryState.query,
          candidates,
          topK,
          llmConfig
        )

        allResults[cr.strategyId] = {
          strategyId: cr.strategyId,
          rankedChunks: ranked.slice(0, topK).map((r, i) => {
            const chunk = cr.chunks.find(c => c.index === r.index) as Chunk
            return { chunk, score: r.score, rank: i + 1 }
          }).filter(r => r.chunk != null),
        }
      })
    )

    setQueryState(prev => ({ ...prev, running: false, results: allResults }))
  }, [queryState.query])

  const judgeRelevance = useCallback((strategyId: string, chunkIndex: number, relevant: boolean) => {
    setQueryState(prev => ({
      ...prev,
      relevanceJudgments: {
        ...prev.relevanceJudgments,
        [strategyId]: {
          ...(prev.relevanceJudgments[strategyId] ?? {}),
          [chunkIndex]: relevant,
        },
      },
    }))
  }, [])

  const precisionAt = useCallback((strategyId: string, k: number): number | null => {
    const judgments = queryState.relevanceJudgments[strategyId]
    const ranked = queryState.results[strategyId]?.rankedChunks
    if (!judgments || !ranked) return null

    const topK = ranked.slice(0, k)
    const judged = topK.filter(r => judgments[r.chunk.index] !== undefined)
    if (judged.length < Math.min(3, k)) return null

    const relevant = judged.filter(r => judgments[r.chunk.index] === true).length
    return relevant / topK.length
  }, [queryState])

  return { evalState, queryState, runEval, setQuery, runQuery, judgeRelevance, precisionAt }
}
