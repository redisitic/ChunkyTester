import { Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useState } from 'react'
import type { ChunkResult, EvalResult, QueryResult } from '@/types'

interface Props {
  chunkResults: ChunkResult[]
  evalResults: EvalResult[]
  queryResults: Record<string, QueryResult>
}

function compositeScore(r: EvalResult): number {
  const hasPrecision = r.retrievalPrecision !== undefined
  const count = hasPrecision ? 4 : 3
  return (r.avgSelfContainedness + r.avgBoundaryCoherence + r.tokenEfficiency + (r.retrievalPrecision ?? 0)) / count
}

function exportJson(chunkResults: ChunkResult[], evalResults: EvalResult[], queryResults: Record<string, QueryResult>) {
  const blob = new Blob(
    [JSON.stringify({ chunkResults, evalResults, queryResults }, null, 2)],
    { type: 'application/json' }
  )
  download(blob, 'chunky-tester-export.json')
}

function exportCsv(chunkResults: ChunkResult[], evalResults: EvalResult[]) {
  const rows: string[] = ['strategy,chunk_index,tokens,self_containedness_score,boundary_score,text']
  for (const cr of chunkResults) {
    const er = evalResults.find(e => e.strategyId === cr.strategyId)
    for (const chunk of cr.chunks) {
      const sc = er?.avgSelfContainedness?.toFixed(3) ?? ''
      const bc = er?.avgBoundaryCoherence?.toFixed(3) ?? ''
      const text = JSON.stringify(chunk.text.replace(/\n/g, ' '))
      rows.push(`${cr.strategyId},${chunk.index},${chunk.tokens},${sc},${bc},${text}`)
    }
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
  download(blob, 'chunky-tester-export.csv')
}

function exportMarkdown(chunkResults: ChunkResult[], evalResults: EvalResult[]) {
  const sorted = [...evalResults].sort((a, b) => compositeScore(b) - compositeScore(a))
  const top3 = sorted.slice(0, 3)

  const lines: string[] = [
    '# ChunkyTester — Chunking Strategy Report',
    '',
    '## Metric Comparison',
    '',
    '| Strategy | Self-Contained | Boundary | Token Eff. | Chunks | Composite |',
    '|---|---|---|---|---|---|',
    ...sorted.map(r =>
      `| ${r.strategyId} | ${r.avgSelfContainedness.toFixed(2)} | ${r.avgBoundaryCoherence.toFixed(2)} | ${r.tokenEfficiency.toFixed(2)} | ${r.chunkCount} | ${compositeScore(r).toFixed(2)} |`
    ),
    '',
    '## Top 3 Recommendations',
    '',
    ...top3.map((r, i) => [
      `### ${i + 1}. ${r.strategyId.replace(/_/g, ' ')}`,
      `- Composite score: **${compositeScore(r).toFixed(3)}**`,
      `- Self-containedness: ${r.avgSelfContainedness.toFixed(3)}`,
      `- Boundary coherence: ${r.avgBoundaryCoherence.toFixed(3)}`,
      `- Chunk count: ${r.chunkCount} · Avg tokens: ${r.avgTokens.toFixed(0)}`,
      '',
    ].join('\n')),
    '## Chunk Stats',
    '',
    ...chunkResults.map(cr => `- **${cr.strategyLabel}**: ${cr.chunks.length} chunks, ${cr.durationMs}ms, ~$${cr.estimatedCost.toFixed(4)}`),
  ]

  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' })
  download(blob, 'chunky-tester-report.md')
}

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function ExportButton({ chunkResults, evalResults, queryResults }: Props) {
  const [format, setFormat] = useState<'json' | 'csv' | 'md'>('json')

  function handleExport() {
    if (format === 'json') exportJson(chunkResults, evalResults, queryResults)
    else if (format === 'csv') exportCsv(chunkResults, evalResults)
    else exportMarkdown(chunkResults, evalResults)
  }

  const disabled = chunkResults.length === 0

  return (
    <div className="flex items-center gap-1">
      <Select value={format} onValueChange={v => setFormat(v as typeof format)}>
        <SelectTrigger className="h-8 w-24 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="json">JSON</SelectItem>
          <SelectItem value="csv">CSV</SelectItem>
          <SelectItem value="md">Markdown</SelectItem>
        </SelectContent>
      </Select>
      <Button variant="outline" size="sm" className="h-8 gap-1.5" onClick={handleExport} disabled={disabled}>
        <Download className="w-3.5 h-3.5" />
        Export
      </Button>
    </div>
  )
}
