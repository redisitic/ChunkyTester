import { useState, useEffect } from 'react'
import type React from 'react'
import { Loader2, Check, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { subscribeEmbeddingStatus } from '@/lib/localEmbeddings'
import type { EmbeddingStatus } from '@/lib/localEmbeddings'
import type { ChunkResult, QueryResult } from '@/types'
import type { EmbeddingMode } from '@/components/SettingsDrawer'

interface Props {
  chunkResults: ChunkResult[]
  queryState: {
    query: string
    results: Record<string, QueryResult>
    running: boolean
    relevanceJudgments: Record<string, Record<number, boolean>>
  }
  embeddingMode: EmbeddingMode
  onQueryChange: (q: string) => void
  onRun: () => void
  onJudge: (strategyId: string, chunkIndex: number, relevant: boolean) => void
  precisionAt: (strategyId: string, k: number) => number | null
}

export function QuerySimulator({ chunkResults, queryState, embeddingMode, onQueryChange, onRun, onJudge, precisionAt }: Props) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [embStatus, setEmbStatus] = useState<EmbeddingStatus>('idle')
  const [embProgress, setEmbProgress] = useState(0)

  useEffect(() => {
    if (embeddingMode !== 'local') return
    return subscribeEmbeddingStatus((status, progress) => {
      setEmbStatus(status)
      setEmbProgress(progress)
    })
  }, [embeddingMode])

  const hasResults = Object.keys(queryState.results).length > 0

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-2">
        <Input
          placeholder="Enter a test query…"
          value={queryState.query}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => onQueryChange(e.target.value)}
          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter' && !queryState.running) onRun() }}
          className="flex-1"
        />
        <Button
          onClick={onRun}
          disabled={queryState.running || !queryState.query.trim() || chunkResults.length === 0 || (embeddingMode === 'local' && embStatus === 'loading')}
        >
          {queryState.running ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Simulate Retrieval'}
        </Button>
      </div>

      {embeddingMode === 'local' && embStatus === 'loading' && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span>Loading embedding model (Xenova/all-MiniLM-L6-v2)… {embProgress > 0 ? `${Math.round(embProgress)}%` : ''}</span>
          </div>
          <Progress value={embProgress} className="h-1" />
        </div>
      )}

      {embeddingMode === 'local' && embStatus === 'error' && (
        <p className="text-xs text-destructive">
          Embedding model failed to load. Retrieval will fall back to LLM ranking.
        </p>
      )}

      {embeddingMode === 'local' && embStatus === 'ready' && (
        <div className="flex items-center gap-1.5">
          <Badge variant="outline" className="text-[10px] font-mono">all-MiniLM-L6-v2</Badge>
          <span className="text-xs text-muted-foreground">cosine similarity · local</span>
        </div>
      )}

      {embeddingMode === 'llm' && (
        <div className="flex items-center gap-1.5">
          <Badge variant="outline" className="text-[10px]">LLM ranking</Badge>
          <span className="text-xs text-muted-foreground">relevance scored via prompt</span>
        </div>
      )}

      {hasResults && (
        <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${Math.min(chunkResults.length, 3)}, minmax(0,1fr))` }}>
          {chunkResults.map(cr => {
            const qr = queryState.results[cr.strategyId]
            if (!qr) return null
            const p3 = precisionAt(cr.strategyId, 3)
            const p5 = precisionAt(cr.strategyId, 5)

            return (
              <div key={cr.strategyId} className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{cr.strategyLabel}</span>
                  <div className="flex gap-1">
                    {p3 !== null && <Badge variant="outline" className="text-[10px]">P@3: {p3.toFixed(2)}</Badge>}
                    {p5 !== null && <Badge variant="outline" className="text-[10px]">P@5: {p5.toFixed(2)}</Badge>}
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                  {qr.rankedChunks.map(({ chunk, score, rank }) => {
                    const key = `${cr.strategyId}-${chunk.index}`
                    const judgment = queryState.relevanceJudgments[cr.strategyId]?.[chunk.index]
                    const isExpanded = expanded[key]

                    const displayChunk = cr.strategyId === 'parent_child' && chunk.parentIndex !== undefined
                      ? cr.chunks.find(c => c.index === chunk.parentIndex)
                      : null

                    return (
                      <Card key={chunk.index} className="p-3 flex flex-col gap-2">
                        <div className="flex items-center gap-2">
                          <Badge className="text-[10px] h-4">#{rank}</Badge>
                          <span className="text-xs font-mono text-muted-foreground">
                            score: {score.toFixed(3)}
                          </span>
                          <div className="ml-auto flex gap-1">
                            <Button
                              size="icon"
                              variant={judgment === true ? 'default' : 'outline'}
                              className="h-5 w-5"
                              onClick={() => onJudge(cr.strategyId, chunk.index, true)}
                            >
                              <Check className="w-3 h-3" />
                            </Button>
                            <Button
                              size="icon"
                              variant={judgment === false ? 'destructive' : 'outline'}
                              className="h-5 w-5"
                              onClick={() => onJudge(cr.strategyId, chunk.index, false)}
                            >
                              <X className="w-3 h-3" />
                            </Button>
                          </div>
                        </div>

                        <p
                          className="text-xs font-mono leading-relaxed cursor-pointer"
                          onClick={() => setExpanded(prev => ({ ...prev, [key]: !isExpanded }))}
                        >
                          {isExpanded ? chunk.text : chunk.text.slice(0, 200) + (chunk.text.length > 200 ? '…' : '')}
                        </p>

                        {displayChunk && (
                          <div className="rounded-md bg-muted p-2 text-xs font-mono leading-relaxed">
                            <p className="text-[10px] text-muted-foreground mb-1 font-sans">Parent context sent to LLM:</p>
                            {displayChunk.text.slice(0, 300)}{displayChunk.text.length > 300 ? '…' : ''}
                          </div>
                        )}
                      </Card>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
