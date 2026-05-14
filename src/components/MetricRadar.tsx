import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { EvalResult } from '@/types'

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4', '#f97316']

interface Props {
  evalResults: EvalResult[]
}

export function MetricRadar({ evalResults }: Props) {
  if (evalResults.length === 0) return null

  const axes = [
    { key: 'avgSelfContainedness', label: 'Self-Containedness' },
    { key: 'avgBoundaryCoherence', label: 'Boundary Coherence' },
    { key: 'tokenEfficiency', label: 'Token Efficiency' },
    { key: 'retrievalPrecision', label: 'Retrieval Precision' },
    { key: 'chunkUniformity', label: 'Chunk Uniformity' },
  ]

  const data = axes.map(axis => {
    const point: Record<string, number | string> = { axis: axis.label }
    evalResults.forEach(r => {
      if (axis.key === 'chunkUniformity') {
        const maxStd = Math.max(...evalResults.map(e => e.stdDevTokens), 1)
        point[r.strategyId] = parseFloat((1 - r.stdDevTokens / maxStd).toFixed(3))
      } else {
        point[r.strategyId] = parseFloat(((r[axis.key as keyof EvalResult] as number) ?? 0).toFixed(3))
      }
    })
    return point
  })

  return (
    <ResponsiveContainer width="100%" height={320}>
      <RadarChart data={data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11 }} />
        <PolarRadiusAxis angle={90} domain={[0, 1]} tick={{ fontSize: 9 }} tickCount={3} />
        {evalResults.map((r, i) => (
          <Radar
            key={r.strategyId}
            name={r.strategyId.replace(/_/g, ' ')}
            dataKey={r.strategyId}
            stroke={COLORS[i % COLORS.length]}
            fill={COLORS[i % COLORS.length]}
            fillOpacity={0.1}
            strokeWidth={2}
          />
        ))}
        <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
      </RadarChart>
    </ResponsiveContainer>
  )
}
