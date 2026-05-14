import { Loader2, X, ExternalLink } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { ChunkViewer } from './ChunkViewer'
import { CHUNKERS } from '@/lib/chunkerRegistry'
import type { ChunkResult } from '@/types'

interface ColumnState {
  strategyId: string
  config: Record<string, number | boolean>
  result: ChunkResult | null
  running: boolean
  progress: number
  error: string | null
}

interface Props {
  column: ColumnState
  passageText: string
  onStrategyChange: (id: string) => void
  onConfigChange: (key: string, value: number | boolean) => void
  onRun: () => void
  onRemove: () => void
  canRemove: boolean
}

export function StrategyColumn({
  column,
  passageText,
  onStrategyChange,
  onConfigChange,
  onRun,
  onRemove,
  canRemove,
}: Props) {
  const chunker = CHUNKERS.find(c => c.id === column.strategyId)

  return (
    <Card className="flex flex-col gap-3 p-4 min-w-0">
      <div className="flex items-center gap-2">
        <Select value={column.strategyId} onValueChange={v => v !== null && onStrategyChange(v)}>
          <SelectTrigger className="flex-1 h-8 text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CHUNKERS.map(c => (
              <SelectItem key={c.id} value={c.id}>{c.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        {canRemove && (
          <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={onRemove}>
            <X className="w-4 h-4" />
          </Button>
        )}
      </div>

      <div className="flex flex-wrap gap-1">
        <Badge variant="secondary" className="text-[10px]">LLM</Badge>
        {column.strategyId === 'structure_aware' && (
          <Badge variant="outline" className="text-[10px]">Needs Structure</Badge>
        )}
        {chunker?.paperRef && (
          <Tooltip>
            <TooltipTrigger render={<Badge variant="outline" className="text-[10px] cursor-help gap-1" />}>
              <ExternalLink className="w-2.5 h-2.5" />
              Paper
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-xs text-xs">
              {chunker.paperRef}
            </TooltipContent>
          </Tooltip>
        )}
      </div>

      {chunker && (
        <p className="text-xs text-muted-foreground leading-relaxed">{chunker.description}</p>
      )}

      <Separator />

      {chunker && chunker.configSchema.length > 0 && (
        <div className="flex flex-col gap-3">
          {chunker.configSchema.map(field => (
            <div key={field.key} className="flex flex-col gap-1.5">
              <Tooltip>
                <TooltipTrigger render={<div className="flex items-center justify-between cursor-help" />}>
                    <span className="text-xs font-medium">{field.label}</span>
                    {field.type === 'slider' && (
                      <span className="text-xs font-mono text-muted-foreground">
                        {column.config[field.key] ?? field.default}
                      </span>
                    )}
                </TooltipTrigger>
                <TooltipContent side="right" className="max-w-xs text-xs">
                  {field.description}
                </TooltipContent>
              </Tooltip>

              {field.type === 'slider' && (
                <Slider
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={[column.config[field.key] as number ?? field.default as number]}
                  onValueChange={v => onConfigChange(field.key, Array.isArray(v) ? v[0] : v as number)}
                  className="w-full"
                />
              )}

              {field.type === 'toggle' && (
                <button
                  type="button"
                  role="switch"
                  aria-checked={!!(column.config[field.key] ?? field.default)}
                  onClick={() => onConfigChange(field.key, !(column.config[field.key] ?? field.default))}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                    (column.config[field.key] ?? field.default) ? 'bg-primary' : 'bg-input'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-background shadow transition-transform ${
                      (column.config[field.key] ?? field.default) ? 'translate-x-4' : 'translate-x-0.5'
                    }`}
                  />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2">
        <Button
          className="flex-1 h-8 text-sm"
          onClick={onRun}
          disabled={column.running || !passageText}
        >
          {column.running ? (
            <><Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />Running…</>
          ) : 'Run'}
        </Button>

        {column.result && (
          <Tooltip>
            <TooltipTrigger render={<Badge variant="outline" className="text-[10px] cursor-help shrink-0" />}>
              ~${column.result.estimatedCost.toFixed(4)}
            </TooltipTrigger>
            <TooltipContent side="bottom" className="text-xs">
              {column.result.apiCallCount} API call{column.result.apiCallCount !== 1 ? 's' : ''} ·{' '}
              {column.result.durationMs}ms · {column.result.chunks.length} chunks
            </TooltipContent>
          </Tooltip>
        )}
      </div>

      {column.running && (
        <Progress value={column.progress} className="h-1" />
      )}

      {column.error && (
        <Alert variant="destructive">
          <AlertDescription className="text-xs">{column.error}</AlertDescription>
        </Alert>
      )}

      {column.result && !column.running && (
        <>
          <Separator />
          <div className="flex gap-2 flex-wrap">
            <Badge variant="secondary" className="text-[10px]">{column.result.chunks.length} chunks</Badge>
            <Badge variant="secondary" className="text-[10px]">{column.result.durationMs}ms</Badge>
          </div>
          <ChunkViewer result={column.result} passageText={passageText} />
        </>
      )}
    </Card>
  )
}
