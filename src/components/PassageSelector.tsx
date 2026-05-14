import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { estimateTokens } from '@/lib/tokenCounter'
import type { DocumentState } from '@/types'

const PASSAGE_WARN_CHARS = 6000

interface Props {
  docState: DocumentState
  onSelectionChange: (start: number, end: number) => void
  onContinue: () => void
}

export function PassageSelector({ docState, onSelectionChange, onContinue }: Props) {
  const { fullText, structure, selectedStart, selectedEnd } = docState
  const [start, setStart] = useState(selectedStart)
  const [end, setEnd] = useState(selectedEnd)
  const textAreaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    onSelectionChange(start, end)
  }, [start, end, onSelectionChange])

  function handleSelect() {
    const el = textAreaRef.current
    if (!el) return
    const s = el.selectionStart
    const e = el.selectionEnd
    if (s !== e) {
      setStart(s)
      setEnd(e)
    }
  }

  function selectAll() {
    setStart(0)
    setEnd(fullText.length)
  }

  function selectPage(pageIndex: number) {
    const page = structure.pages?.[pageIndex]
    if (!page) return
    setStart(page.start)
    setEnd(page.end)
  }

  function selectHeading(headingIndex: number) {
    const h = structure.headings[headingIndex]
    if (!h) return
    const next = structure.headings[headingIndex + 1]
    setStart(h.start)
    setEnd(next ? next.start : fullText.length)
  }

  const selectionLength = end - start
  const tokens = estimateTokens(fullText.slice(start, end))
  const wordCount = fullText.slice(start, end).trim().split(/\s+/).filter(Boolean).length

  return (
    <div className="flex flex-col gap-4 p-4 max-w-4xl mx-auto w-full">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Select Passage</h2>
          <p className="text-sm text-muted-foreground">Drag to select a region, or use the quick-select buttons</p>
        </div>
        <Button onClick={onContinue} disabled={selectionLength === 0}>
          Compare Strategies →
        </Button>
      </div>

      <div className="flex flex-wrap gap-2 items-center">
        <Button variant="outline" size="sm" onClick={selectAll}>Select All</Button>

        {structure.pages && structure.pages.length > 0 && (
          <>
            <Separator orientation="vertical" className="h-5" />
            <span className="text-xs text-muted-foreground">Pages:</span>
            {structure.pages.slice(0, 8).map((p, i) => (
              <Button key={i} variant="outline" size="sm" onClick={() => selectPage(i)}>
                p.{p.number}
              </Button>
            ))}
          </>
        )}

        {structure.headings.length > 0 && (
          <>
            <Separator orientation="vertical" className="h-5" />
            <span className="text-xs text-muted-foreground">Sections:</span>
            {structure.headings.slice(0, 6).map((h, i) => (
              <Button
                key={i}
                variant="outline"
                size="sm"
                onClick={() => selectHeading(i)}
                title={h.text}
              >
                {h.text.slice(0, 20)}{h.text.length > 20 ? '…' : ''}
              </Button>
            ))}
          </>
        )}
      </div>

      <div className="flex gap-3 text-sm">
        <Badge variant="secondary">{wordCount.toLocaleString()} words</Badge>
        <Badge variant="secondary">{selectionLength.toLocaleString()} chars</Badge>
        <Badge variant="secondary">~{tokens.toLocaleString()} tokens</Badge>
      </div>

      {selectionLength > PASSAGE_WARN_CHARS && (
        <Alert>
          <AlertDescription>
            Selection is {selectionLength.toLocaleString()} characters. LLM strategies will be slower and consume more API credits (~${(tokens * 3 / 1_000_000).toFixed(4)} input cost est.).
          </AlertDescription>
        </Alert>
      )}

      <ScrollArea className="h-[480px] w-full rounded-md border">
        <textarea
          ref={textAreaRef}
          value={fullText}
          readOnly
          onMouseUp={handleSelect}
          onKeyUp={handleSelect}
          className="w-full h-full min-h-[480px] p-4 font-mono text-xs resize-none bg-transparent outline-none leading-relaxed"
          style={{ whiteSpace: 'pre-wrap' }}
        />
      </ScrollArea>

      <p className="text-xs text-muted-foreground">
        Selected: chars {start.toLocaleString()}–{end.toLocaleString()}
      </p>
    </div>
  )
}
