import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import type { EvalResult } from '@/types'

interface Props {
  evalResults: EvalResult[]
}

function fmt(n: number | undefined, decimals = 2): string {
  return n !== undefined ? n.toFixed(decimals) : '—'
}

function compositeScore(r: EvalResult): number {
  const precision = r.retrievalPrecision ?? 0
  const hasPrecision = r.retrievalPrecision !== undefined
  const count = hasPrecision ? 4 : 3
  return (r.avgSelfContainedness + r.avgBoundaryCoherence + r.tokenEfficiency + (hasPrecision ? precision : 0)) / count
}

export function ComparisonTable({ evalResults }: Props) {
  if (evalResults.length === 0) return null

  const sorted = [...evalResults].sort((a, b) => compositeScore(b) - compositeScore(a))

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Strategy</TableHead>
            <TableHead className="text-right">Self-Contained</TableHead>
            <TableHead className="text-right">Boundary</TableHead>
            <TableHead className="text-right">Token Eff.</TableHead>
            <TableHead className="text-right">Retrieval P</TableHead>
            <TableHead className="text-right">Chunks</TableHead>
            <TableHead className="text-right">Avg Tokens</TableHead>
            <TableHead className="text-right">Composite</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((r, i) => (
            <TableRow key={r.strategyId}>
              <TableCell className="font-medium">
                <div className="flex items-center gap-2">
                  {i === 0 && <Badge variant="default" className="text-[10px]">Top</Badge>}
                  {r.strategyId.replace(/_/g, ' ')}
                </div>
              </TableCell>
              <TableCell className="text-right font-mono text-sm">{fmt(r.avgSelfContainedness)}</TableCell>
              <TableCell className="text-right font-mono text-sm">{fmt(r.avgBoundaryCoherence)}</TableCell>
              <TableCell className="text-right font-mono text-sm">{fmt(r.tokenEfficiency)}</TableCell>
              <TableCell className="text-right font-mono text-sm">{fmt(r.retrievalPrecision)}</TableCell>
              <TableCell className="text-right font-mono text-sm">{r.chunkCount}</TableCell>
              <TableCell className="text-right font-mono text-sm">{fmt(r.avgTokens, 0)}</TableCell>
              <TableCell className="text-right font-mono text-sm font-semibold">{fmt(compositeScore(r))}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
