import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { MetricRadar } from './MetricRadar'
import { ComparisonTable } from './ComparisonTable'
import { QuerySimulator } from './QuerySimulator'
import { BenchmarkPanel } from './BenchmarkPanel'
import type { ChunkResult, EvalResult, QueryResult, LLMConfig, EmbeddingMode } from '@/types'

interface Props {
  chunkResults: ChunkResult[]
  evalState: {
    results: Record<string, EvalResult>
    running: boolean
    progress: number
    error: string | null
  }
  queryState: {
    query: string
    results: Record<string, QueryResult>
    running: boolean
    relevanceJudgments: Record<string, Record<number, boolean>>
  }
  embeddingMode: EmbeddingMode
  voyageKey: string
  llmConfig: LLMConfig
  onRunEval: () => void
  onQueryChange: (q: string) => void
  onRunQuery: () => void
  onJudge: (strategyId: string, chunkIndex: number, relevant: boolean) => void
  precisionAt: (strategyId: string, k: number) => number | null
}

export function EvalPanel({
  chunkResults,
  evalState,
  queryState,
  embeddingMode,
  voyageKey,
  llmConfig,
  onRunEval,
  onQueryChange,
  onRunQuery,
  onJudge,
  precisionAt,
}: Props) {
  const evalResults = Object.values(evalState.results)
  const hasChunks = chunkResults.length > 0

  return (
    <div className="border-t pt-4">
      <Tabs defaultValue="metrics">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
          <TabsList>
            <TabsTrigger value="metrics">Automated Metrics</TabsTrigger>
            <TabsTrigger value="query">Query Simulator</TabsTrigger>
            <TabsTrigger value="benchmark" className="gap-1.5">
              Benchmark
              <Badge variant="secondary" className="text-[9px] px-1 py-0 h-3.5">NEW</Badge>
            </TabsTrigger>
          </TabsList>

          <Button
            variant="outline"
            size="sm"
            onClick={onRunEval}
            disabled={!hasChunks || evalState.running}
          >
            {evalState.running ? (
              <><Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />Evaluating… {evalState.progress}%</>
            ) : 'Run Evaluation'}
          </Button>
        </div>

        {evalState.running && (
          <Progress value={evalState.progress} className="h-1 mb-4" />
        )}

        <TabsContent value="metrics" className="mt-0">
          {evalResults.length === 0 && !evalState.running && (
            <p className="text-sm text-muted-foreground text-center py-8">
              Run strategies above, then click "Run Evaluation" to score them.
            </p>
          )}
          {evalResults.length > 0 && (
            <div className="flex flex-col gap-6">
              <MetricRadar evalResults={evalResults} />
              <ComparisonTable evalResults={evalResults} />
            </div>
          )}
        </TabsContent>

        <TabsContent value="query" className="mt-0">
          {!hasChunks && (
            <p className="text-sm text-muted-foreground text-center py-8">
              Run at least one strategy first.
            </p>
          )}
          {hasChunks && (
            <QuerySimulator
              chunkResults={chunkResults}
              queryState={queryState}
              embeddingMode={embeddingMode}
              onQueryChange={onQueryChange}
              onRun={onRunQuery}
              onJudge={onJudge}
              precisionAt={precisionAt}
            />
          )}
        </TabsContent>

        <TabsContent value="benchmark" className="mt-0">
          <BenchmarkPanel
            chunkResults={chunkResults}
            embeddingMode={embeddingMode}
            voyageKey={voyageKey}
            llmConfig={llmConfig}
          />
        </TabsContent>
      </Tabs>
    </div>
  )
}
