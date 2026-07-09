import { useState, useCallback } from 'react'
import { Plus, Trash2, Play, Loader2, Download, ChevronDown, ChevronRight, Target } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { embedChunksVoyage, rankChunksByVoyageEmbedding, rankChunksByEmbedding, rankChunksByQuery } from '@/lib/embeddings'
import { computeQueryMetrics, aggregateBenchmarkMetrics } from '@/lib/benchmarkMetrics'
import { estimateVoyageCost } from '@/lib/voyageEmbeddings'
import type { StrategyBenchmarkResult, RankedResult } from '@/lib/benchmarkMetrics'
import type { ChunkResult, LLMConfig, EmbeddingMode } from '@/types'

// ---- Types ----

export interface BenchmarkQuery {
  id: string
  text: string
  /** Map: strategyId → array of relevant chunk indices (gold labels) */
  goldLabels: Record<string, number[]>
}

interface ChunkLabelState {
  /** strategyId → Set of chunk indices marked relevant */
  labels: Record<string, Set<number>>
}

interface Props {
  chunkResults: ChunkResult[]
  embeddingMode: EmbeddingMode
  voyageKey: string
  llmConfig: LLMConfig
}

// ---- Helpers ----

function newQuery(): BenchmarkQuery {
  return { id: crypto.randomUUID(), text: '', goldLabels: {} }
}

function fmt(n: number, dec = 3): string {
  return n.toFixed(dec)
}

function metricColor(v: number): string {
  if (v >= 0.7) return 'text-emerald-600 dark:text-emerald-400'
  if (v >= 0.4) return 'text-amber-600 dark:text-amber-400'
  return 'text-rose-600 dark:text-rose-400'
}

// ---- Component ----

export function BenchmarkPanel({ chunkResults, embeddingMode, voyageKey, llmConfig }: Props) {
  const [queries, setQueries] = useState<BenchmarkQuery[]>([newQuery()])
  const [labelStates, setLabelStates] = useState<ChunkLabelState[]>([{ labels: {} }])
  const [expandedQuery, setExpandedQuery] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [progressLabel, setProgressLabel] = useState('')
  const [results, setResults] = useState<StrategyBenchmarkResult[]>([])
  const [error, setError] = useState<string | null>(null)
  const [costEstimate, setCostEstimate] = useState<number | null>(null)

  const hasChunks = chunkResults.length > 0
  const validQueries = queries.filter((q, i) => q.text.trim() && hasGoldLabels(labelStates[i]))

  function hasGoldLabels(ls: ChunkLabelState): boolean {
    return Object.values(ls.labels).some(s => s.size > 0)
  }

  // ---- Query management ----

  function addQuery() {
    setQueries(prev => [...prev, newQuery()])
    setLabelStates(prev => [...prev, { labels: {} }])
  }

  function removeQuery(idx: number) {
    setQueries(prev => prev.filter((_, i) => i !== idx))
    setLabelStates(prev => prev.filter((_, i) => i !== idx))
  }

  function updateQueryText(idx: number, text: string) {
    setQueries(prev => prev.map((q, i) => i === idx ? { ...q, text } : q))
  }

  function toggleLabel(queryIdx: number, strategyId: string, chunkIndex: number) {
    setLabelStates(prev => {
      const next = [...prev]
      const ls = { labels: { ...next[queryIdx].labels } }
      const set = new Set(ls.labels[strategyId] ?? [])
      if (set.has(chunkIndex)) set.delete(chunkIndex)
      else set.add(chunkIndex)
      ls.labels[strategyId] = set
      next[queryIdx] = ls
      return next
    })
  }

  // ---- Cost estimate ----

  function updateCostEstimate() {
    if (embeddingMode !== 'voyage') { setCostEstimate(null); return }
    const allChunkTexts = chunkResults.flatMap(cr => cr.chunks.map(c => c.text))
    const queryTexts = queries.filter(q => q.text.trim()).map(q => q.text)
    const cost = estimateVoyageCost([...allChunkTexts, ...queryTexts])
    setCostEstimate(cost)
  }

  // ---- Run benchmark ----

  const runBenchmark = useCallback(async () => {
    if (!hasChunks || validQueries.length === 0) return
    setRunning(true)
    setError(null)
    setResults([])

    const totalSteps = validQueries.length * chunkResults.length
    let done = 0

    try {
      // Pre-embed all chunks once per strategy (only for voyage mode)
      const chunkEmbeddingsMap: Record<string, number[][]> = {}
      if (embeddingMode === 'voyage') {
        for (const cr of chunkResults) {
          setProgressLabel(`Embedding chunks: ${cr.strategyLabel}…`)
          const candidates = cr.chunks.map(c => ({ index: c.index, text: c.embeddingInput ?? c.text }))
          chunkEmbeddingsMap[cr.strategyId] = await embedChunksVoyage(candidates, voyageKey)
          setProgress(Math.round((Object.keys(chunkEmbeddingsMap).length / chunkResults.length) * 30))
        }
      }

      const allStrategyMetrics: Map<string, ReturnType<typeof computeQueryMetrics>[]> = new Map(
        chunkResults.map(cr => [cr.strategyId, []])
      )

      for (let qi = 0; qi < validQueries.length; qi++) {
        const query = validQueries[qi]
        const ls = labelStates[queries.indexOf(query)]

        setProgressLabel(`Query ${qi + 1}/${validQueries.length}: "${query.text.slice(0, 40)}…"`)

        for (const cr of chunkResults) {
          const candidates = cr.chunks.map(c => ({ index: c.index, text: c.embeddingInput ?? c.text }))
          const goldForStrategy = Array.from(ls.labels[cr.strategyId] ?? new Set<number>())

          let ranked: RankedResult[]

          if (embeddingMode === 'voyage') {
            const vr = await rankChunksByVoyageEmbedding(query.text, candidates, candidates.length, voyageKey)
            ranked = vr.map(x => ({ chunkIndex: x.index, score: x.score }))
          } else if (embeddingMode === 'local') {
            try {
              const r = await rankChunksByEmbedding(query.text, candidates, candidates.length)
              ranked = r.map(x => ({ chunkIndex: x.index, score: x.score }))
            } catch {
              const r = await rankChunksByQuery(query.text, candidates, candidates.length, llmConfig)
              ranked = r.map(x => ({ chunkIndex: x.index, score: x.score }))
            }
          } else {
            const r = await rankChunksByQuery(query.text, candidates, candidates.length, llmConfig)
            ranked = r.map(x => ({ chunkIndex: x.index, score: x.score }))
          }

          const finalRanked: RankedResult[] = ranked

          const qm = computeQueryMetrics(query.text, finalRanked, goldForStrategy)
          allStrategyMetrics.get(cr.strategyId)!.push(qm)

          done++
          setProgress(30 + Math.round((done / totalSteps) * 70))
        }
      }

      // Aggregate
      const aggregated: StrategyBenchmarkResult[] = chunkResults.map(cr =>
        aggregateBenchmarkMetrics(
          cr.strategyId,
          cr.strategyLabel,
          allStrategyMetrics.get(cr.strategyId) ?? []
        )
      ).sort((a, b) => b.composite - a.composite)

      setResults(aggregated)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark failed')
    } finally {
      setRunning(false)
      setProgress(100)
      setProgressLabel('')
    }
  }, [chunkResults, validQueries, labelStates, queries, embeddingMode, voyageKey, llmConfig])

  // ---- Export ----

  function exportResults() {
    const data = {
      timestamp: new Date().toISOString(),
      embeddingMode,
      model: embeddingMode === 'voyage' ? 'voyage-finance-2' : embeddingMode,
      queries: validQueries.map(q => q.text),
      results,
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `benchmark-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ---- Render ----

  if (!hasChunks) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        Run at least one chunking strategy first.
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-sm font-semibold">Benchmark Harness</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Define queries + gold-label relevant chunks per strategy, then run to get MRR / NDCG / Precision / Recall across all queries.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {embeddingMode === 'voyage' && (
            <Badge variant="outline" className="text-[10px] font-mono gap-1">
              <Target className="w-2.5 h-2.5" />
              voyage-finance-2
            </Badge>
          )}
          {embeddingMode !== 'voyage' && (
            <Badge variant="outline" className="text-[10px]">
              {embeddingMode === 'local' ? 'all-MiniLM-L6-v2' : 'LLM ranking'}
            </Badge>
          )}
          {results.length > 0 && (
            <Button variant="outline" size="sm" onClick={exportResults} className="h-7 text-xs gap-1">
              <Download className="w-3 h-3" />
              Export
            </Button>
          )}
        </div>
      </div>

      {/* Voyage key warning */}
      {embeddingMode === 'voyage' && !voyageKey && (
        <div className="rounded-md border border-amber-200 bg-amber-50 dark:bg-amber-950/20 dark:border-amber-800 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
          ⚠ Voyage API key not set — open Settings to enter it before running.
        </div>
      )}

      {/* Query editor */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Test Queries</span>
          <Button variant="outline" size="sm" onClick={addQuery} className="h-7 text-xs gap-1">
            <Plus className="w-3 h-3" /> Add Query
          </Button>
        </div>

        {queries.map((q, qi) => {
          const ls = labelStates[qi]
          const totalLabeled = Object.values(ls.labels).reduce((s, set) => s + set.size, 0)
          const isExpanded = expandedQuery === q.id

          return (
            <Card key={q.id} className="overflow-hidden">
              <div className="flex items-center gap-2 p-3">
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground shrink-0"
                  onClick={() => setExpandedQuery(isExpanded ? null : q.id)}
                >
                  {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                </button>
                <Input
                  placeholder={`Query ${qi + 1}: e.g. "What is the net interest margin?"`}
                  value={q.text}
                  onChange={e => updateQueryText(qi, e.target.value)}
                  className="flex-1 h-8 text-sm"
                />
                <Badge variant={totalLabeled > 0 ? 'default' : 'secondary'} className="text-[10px] shrink-0">
                  {totalLabeled} labeled
                </Badge>
                {queries.length > 1 && (
                  <button
                    type="button"
                    className="text-muted-foreground hover:text-destructive shrink-0"
                    onClick={() => removeQuery(qi)}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              {/* Gold label editor — expanded */}
              {isExpanded && q.text.trim() && (
                <div className="border-t px-3 pb-3 pt-2 flex flex-col gap-3">
                  <p className="text-xs text-muted-foreground">
                    Click chunks to mark them as <strong>relevant</strong> for this query. Each strategy is labeled independently.
                  </p>
                  {chunkResults.map(cr => {
                    const labeled = ls.labels[cr.strategyId] ?? new Set<number>()
                    return (
                      <div key={cr.strategyId} className="flex flex-col gap-1.5">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium">{cr.strategyLabel}</span>
                          <Badge variant="secondary" className="text-[10px]">{labeled.size} relevant</Badge>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {cr.chunks.map(chunk => {
                            const isRelevant = labeled.has(chunk.index)
                            return (
                              <button
                                key={chunk.index}
                                type="button"
                                title={chunk.text.slice(0, 200)}
                                onClick={() => toggleLabel(qi, cr.strategyId, chunk.index)}
                                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono border transition-colors ${
                                  isRelevant
                                    ? 'bg-emerald-100 border-emerald-400 text-emerald-800 dark:bg-emerald-900/40 dark:border-emerald-600 dark:text-emerald-300'
                                    : 'bg-muted border-border text-muted-foreground hover:border-foreground/30'
                                }`}
                              >
                                #{chunk.index}
                                <span className="max-w-[120px] truncate normal-case font-sans">
                                  {chunk.text.slice(0, 30)}
                                </span>
                              </button>
                            )
                          })}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}

              {isExpanded && !q.text.trim() && (
                <p className="px-3 pb-3 text-xs text-muted-foreground">Enter a query above to label chunks.</p>
              )}
            </Card>
          )
        })}
      </div>

      {/* Cost estimate + run button */}
      <div className="flex items-center gap-3 flex-wrap">
        {costEstimate !== null && (
          <span className="text-xs text-muted-foreground">
            Est. Voyage cost: ~${costEstimate.toFixed(5)}
          </span>
        )}
        <Button
          onClick={() => { updateCostEstimate(); runBenchmark() }}
          disabled={running || validQueries.length === 0 || (embeddingMode === 'voyage' && !voyageKey)}
          className="gap-2"
          size="sm"
        >
          {running
            ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />{progressLabel || 'Running…'}</>
            : <><Play className="w-3.5 h-3.5" />Run Benchmark ({validQueries.length} {validQueries.length === 1 ? 'query' : 'queries'})</>
          }
        </Button>
      </div>

      {running && <Progress value={progress} className="h-1" />}

      {error && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold">Results</h3>
            <Badge variant="secondary" className="text-[10px]">{validQueries.length} queries</Badge>
          </div>

          {/* Summary table */}
          <div className="overflow-x-auto rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Strategy</TableHead>
                  <TableHead className="text-right">P@1</TableHead>
                  <TableHead className="text-right">P@3</TableHead>
                  <TableHead className="text-right">P@5</TableHead>
                  <TableHead className="text-right">R@5</TableHead>
                  <TableHead className="text-right">R@10</TableHead>
                  <TableHead className="text-right">MRR</TableHead>
                  <TableHead className="text-right">NDCG@5</TableHead>
                  <TableHead className="text-right">Composite</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.map((r, i) => (
                  <TableRow key={r.strategyId} className={i === 0 ? 'bg-emerald-50/50 dark:bg-emerald-950/20' : ''}>
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        {i === 0 && <Badge className="text-[10px]">Best</Badge>}
                        {r.strategyLabel}
                      </div>
                    </TableCell>
                    <TableCell className={`text-right font-mono text-xs ${metricColor(r.avgPrecisionAt1)}`}>{fmt(r.avgPrecisionAt1)}</TableCell>
                    <TableCell className={`text-right font-mono text-xs ${metricColor(r.avgPrecisionAt3)}`}>{fmt(r.avgPrecisionAt3)}</TableCell>
                    <TableCell className={`text-right font-mono text-xs ${metricColor(r.avgPrecisionAt5)}`}>{fmt(r.avgPrecisionAt5)}</TableCell>
                    <TableCell className={`text-right font-mono text-xs ${metricColor(r.avgRecallAt5)}`}>{fmt(r.avgRecallAt5)}</TableCell>
                    <TableCell className={`text-right font-mono text-xs ${metricColor(r.avgRecallAt10)}`}>{fmt(r.avgRecallAt10)}</TableCell>
                    <TableCell className={`text-right font-mono text-xs ${metricColor(r.meanReciprocalRank)}`}>{fmt(r.meanReciprocalRank)}</TableCell>
                    <TableCell className={`text-right font-mono text-xs ${metricColor(r.avgNdcgAt5)}`}>{fmt(r.avgNdcgAt5)}</TableCell>
                    <TableCell className={`text-right font-mono text-sm font-semibold ${metricColor(r.composite)}`}>{fmt(r.composite)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          {/* Per-query breakdown */}
          {results.map(r => (
            r.perQueryMetrics.length > 1 && (
              <div key={r.strategyId} className="flex flex-col gap-2">
                <span className="text-xs font-medium text-muted-foreground">{r.strategyLabel} — per query</span>
                <div className="overflow-x-auto rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Query</TableHead>
                        <TableHead className="text-right">P@5</TableHead>
                        <TableHead className="text-right">R@5</TableHead>
                        <TableHead className="text-right">MRR</TableHead>
                        <TableHead className="text-right">NDCG@5</TableHead>
                        <TableHead className="text-right">Relevant</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {r.perQueryMetrics.map((qm, qi) => (
                        <TableRow key={qi}>
                          <TableCell className="text-xs max-w-[200px] truncate">{qm.query}</TableCell>
                          <TableCell className={`text-right font-mono text-xs ${metricColor(qm.precisionAt5)}`}>{fmt(qm.precisionAt5)}</TableCell>
                          <TableCell className={`text-right font-mono text-xs ${metricColor(qm.recallAt5)}`}>{fmt(qm.recallAt5)}</TableCell>
                          <TableCell className={`text-right font-mono text-xs ${metricColor(qm.mrr)}`}>{fmt(qm.mrr)}</TableCell>
                          <TableCell className={`text-right font-mono text-xs ${metricColor(qm.ndcgAt5)}`}>{fmt(qm.ndcgAt5)}</TableCell>
                          <TableCell className="text-right font-mono text-xs text-muted-foreground">{qm.numRelevant}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            )
          ))}
        </div>
      )}
    </div>
  )
}
