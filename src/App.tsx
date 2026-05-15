import { useState, useEffect } from 'react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { FileUpload } from '@/components/FileUpload'
import { PassageSelector } from '@/components/PassageSelector'
import { StrategyColumn } from '@/components/StrategyColumn'
import { EvalPanel } from '@/components/EvalPanel'
import { SettingsDrawer, loadSettings } from '@/components/SettingsDrawer'
import { ExportButton } from '@/components/ExportButton'
import { useDocument } from '@/hooks/useDocument'
import { useChunking } from '@/hooks/useChunking'
import { useEval } from '@/hooks/useEval'
import { estimateTokens } from '@/lib/tokenCounter'
import { CHUNKERS } from '@/lib/chunkerRegistry'
import type { AppSettings } from '@/components/SettingsDrawer'
import type { EvalResult, LLMConfig } from '@/types'

const PASSAGE_WARN_CHARS = 6000
const MAX_COLUMNS = 4

function buildLLMConfig(settings: AppSettings): LLMConfig {
  return {
    provider: settings.provider,
    anthropicKey: settings.anthropicKey,
    geminiKey: settings.geminiKey,
    geminiModel: settings.geminiModel,
    ollamaBaseUrl: settings.ollamaBaseUrl,
    ollamaModel: settings.ollamaModel,
  }
}

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(loadSettings)
  const [warnDialogOpen, setWarnDialogOpen] = useState(false)
  const [pendingRun, setPendingRun] = useState<number | 'all' | null>(null)

  const { step, docState, loading, error, loadFile, setSelection, advanceTo, reset } = useDocument()
  const { columns, addColumn, removeColumn, setColumnStrategy, setColumnConfig, runColumn, runAll } = useChunking()
  const { evalState, queryState, runEval, setQuery, runQuery, judgeRelevance, precisionAt } = useEval()

  useEffect(() => {
    document.documentElement.classList.toggle('dark', settings.theme === 'dark')
  }, [settings.theme])

  function patchSettings(patch: Partial<AppSettings>) {
    setSettings(prev => ({ ...prev, ...patch }))
  }

  const passage = docState
    ? docState.fullText.slice(docState.selectedStart, docState.selectedEnd)
    : ''

  const passageTokens = estimateTokens(passage)
  const completedResults = columns.map(c => c.result).filter(Boolean) as NonNullable<typeof columns[0]['result']>[]

  function guardedRun(index: number | 'all') {
    if (passage.length > PASSAGE_WARN_CHARS) {
      setPendingRun(index)
      setWarnDialogOpen(true)
    } else {
      executeRun(index)
    }
  }

  function executeRun(index: number | 'all') {
    if (!docState) return
    const llmConfig = buildLLMConfig(settings)
    if (index === 'all') {
      runAll(passage, docState.structure, llmConfig)
    } else {
      runColumn(index, passage, docState.structure, llmConfig)
    }
  }

  function confirmWarnRun() {
    setWarnDialogOpen(false)
    if (pendingRun !== null) executeRun(pendingRun)
    setPendingRun(null)
  }

  if (step === 'upload') {
    return (
      <TooltipProvider>
        <div className="min-h-screen bg-background">
          <FileUpload
            onFile={loadFile}
            docState={docState}
            loading={loading}
            error={error}
            onContinue={() => advanceTo('select')}
          />
        </div>
      </TooltipProvider>
    )
  }

  if (step === 'select') {
    return (
      <TooltipProvider>
        <div className="min-h-screen bg-background">
          <div className="flex items-center justify-between px-4 py-2 border-b">
            <button
              type="button"
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              onClick={reset}
            >
              ← Upload new file
            </button>
            <SettingsDrawer settings={settings} onSettingsChange={patchSettings} />
          </div>
          {docState && (
            <PassageSelector
              docState={docState}
              onSelectionChange={setSelection}
              onContinue={() => advanceTo('compare')}
            />
          )}
        </div>
      </TooltipProvider>
    )
  }

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background flex flex-col">
        <header className="border-b px-4 py-2 flex items-center gap-3 flex-wrap">
          <button
            type="button"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors shrink-0"
            onClick={() => advanceTo('select')}
          >
            ← Back
          </button>
          <Separator orientation="vertical" className="h-4" />
          <span className="text-sm font-medium truncate max-w-xs">{docState?.filename}</span>
          <div className="flex gap-2 flex-wrap">
            <Badge variant="secondary" className="text-[10px]">{passage.length.toLocaleString()} chars selected</Badge>
            <Badge variant="secondary" className="text-[10px]">~{passageTokens.toLocaleString()} tokens</Badge>
          </div>
          {settings.provider === 'anthropic' && !settings.anthropicKey && (
            <Alert variant="destructive" className="py-1 px-3 h-7 flex items-center ml-2">
              <AlertDescription className="text-xs">No API key set — open Settings</AlertDescription>
            </Alert>
          )}
          <div className="ml-auto flex items-center gap-2">
            <ExportButton
              chunkResults={completedResults}
              evalResults={Object.values(evalState.results) as EvalResult[]}
              queryResults={queryState.results}
            />
            <SettingsDrawer settings={settings} onSettingsChange={patchSettings} />
          </div>
        </header>

        <div className="flex-1 overflow-x-auto">
          <div
            className="grid gap-3 p-4 min-w-0"
            style={{
              gridTemplateColumns: `repeat(${Math.min(columns.length + 1, MAX_COLUMNS)}, minmax(280px, 1fr))`,
            }}
          >
            {columns.map((col, i) => (
              <StrategyColumn
                key={i}
                column={col}
                passageText={passage}
                onStrategyChange={id => setColumnStrategy(i, id)}
                onConfigChange={(key, val) => setColumnConfig(i, key, val)}
                onRun={() => guardedRun(i)}
                onRemove={() => removeColumn(i)}
                canRemove={columns.length > 1}
              />
            ))}

            {columns.length < MAX_COLUMNS && (
              <div className="flex flex-col gap-2 p-4 border-2 border-dashed border-border rounded-lg items-center justify-center min-h-40">
                <p className="text-sm text-muted-foreground">Add Strategy</p>
                <div className="flex flex-col gap-1.5 w-full max-w-[180px]">
                  {CHUNKERS.filter(c => !columns.some(col => col.strategyId === c.id)).map(c => (
                    <Button
                      key={c.id}
                      variant="outline"
                      size="sm"
                      className="text-xs w-full"
                      onClick={() => addColumn(c.id)}
                    >
                      + {c.label}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {completedResults.length > 0 && (
            <div className="px-4 pb-4">
              <EvalPanel
                chunkResults={completedResults}
                evalState={evalState}
                queryState={queryState}
                embeddingMode={settings.embeddingMode}
                onRunEval={() => runEval(completedResults, buildLLMConfig(settings), settings.evalSampleSize)}
                onQueryChange={setQuery}
                onRunQuery={() => runQuery(completedResults, buildLLMConfig(settings), settings.topK, settings.embeddingMode)}
                onJudge={judgeRelevance}
                precisionAt={precisionAt}
              />
            </div>
          )}
        </div>

        <Dialog open={warnDialogOpen} onOpenChange={setWarnDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Large passage selected</DialogTitle>
              <DialogDescription>
                You have selected {passage.length.toLocaleString()} characters (~{passageTokens.toLocaleString()} tokens).
                LLM strategies will be slower and consume more API credits.
                Estimated input cost: ~${(passageTokens * 3 / 1_000_000).toFixed(4)} per strategy.
              </DialogDescription>
            </DialogHeader>
            <div className="flex justify-end gap-2 mt-2">
              <Button variant="outline" onClick={() => setWarnDialogOpen(false)}>Cancel</Button>
              <Button onClick={confirmWarnRun}>Proceed anyway</Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  )
}
