import { useState } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import type { Chunk, ChunkResult } from '@/types'

const PALETTE = [
  'bg-blue-100 dark:bg-blue-900/40',
  'bg-green-100 dark:bg-green-900/40',
  'bg-amber-100 dark:bg-amber-900/40',
  'bg-purple-100 dark:bg-purple-900/40',
  'bg-rose-100 dark:bg-rose-900/40',
  'bg-cyan-100 dark:bg-cyan-900/40',
  'bg-orange-100 dark:bg-orange-900/40',
  'bg-teal-100 dark:bg-teal-900/40',
]

const BORDER_PALETTE = [
  'border-blue-400',
  'border-green-400',
  'border-amber-400',
  'border-purple-400',
  'border-rose-400',
  'border-cyan-400',
  'border-orange-400',
  'border-teal-400',
]

interface Props {
  result: ChunkResult
  passageText: string
}

function ChunkDetail({ chunk }: { chunk: Chunk }) {
  return (
    <div className="flex flex-col gap-3 text-sm">
      {chunk.summary && (
        <div className="rounded-md bg-muted p-2">
          <p className="text-xs font-semibold text-muted-foreground mb-1">Summary</p>
          <p>{chunk.summary}</p>
        </div>
      )}
      {chunk.contextDependencies && chunk.contextDependencies.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-muted-foreground mb-1">Context Dependencies</p>
          <ul className="list-disc list-inside space-y-0.5">
            {chunk.contextDependencies.map((d, i) => <li key={i}>{d}</li>)}
          </ul>
        </div>
      )}
      {chunk.rationale && (
        <div>
          <p className="text-xs font-semibold text-muted-foreground mb-1">Rationale</p>
          <p>{chunk.rationale}</p>
        </div>
      )}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>Tokens</span><span className="font-mono text-foreground">{chunk.tokens}</span>
        <span>Start</span><span className="font-mono text-foreground">{chunk.start}</span>
        <span>End</span><span className="font-mono text-foreground">{chunk.end}</span>
        {chunk.parentIndex !== undefined && (
          <><span>Parent chunk</span><span className="font-mono text-foreground">#{chunk.parentIndex}</span></>
        )}
        {chunk.children && (
          <><span>Children</span><span className="font-mono text-foreground">{chunk.children.join(', ')}</span></>
        )}
      </div>
      <div className="rounded-md bg-muted p-2 font-mono text-xs leading-relaxed whitespace-pre-wrap">
        {chunk.text}
      </div>
    </div>
  )
}

export function ChunkViewer({ result, passageText }: Props) {
  const [selected, setSelected] = useState<Chunk | null>(null)

  const isParentChild = result.strategyId === 'parent_child'
  const parents = isParentChild ? result.chunks.filter(c => c.children !== undefined) : []
  const children = isParentChild ? result.chunks.filter(c => c.parentIndex !== undefined) : []
  const displayChunks = isParentChild ? children : result.chunks

  return (
    <div className="flex flex-col gap-3">
      <div className="relative rounded-md border bg-muted/30 p-3 font-mono text-xs leading-relaxed overflow-hidden">
        <div className="relative">
          {passageText.split('').length > 0 && (
            <div className="relative">
              {isParentChild ? (
                <div className="flex flex-col gap-1">
                  {parents.map(parent => {
                    const pi = parents.indexOf(parent)
                    const color = PALETTE[pi % PALETTE.length]
                    const border = BORDER_PALETTE[pi % BORDER_PALETTE.length]
                    const parentChildren = children.filter(c => c.parentIndex === parent.index)
                    return (
                      <div key={parent.index} className={`rounded border-2 ${border} p-1`}>
                        <div className="flex gap-1 flex-wrap">
                          {parentChildren.map(child => (
                            <span
                              key={child.index}
                              className={`${color} rounded px-1 cursor-pointer hover:opacity-80`}
                              onClick={() => setSelected(child)}
                            >
                              {child.text.slice(0, 60)}{child.text.length > 60 ? '…' : ''}
                            </span>
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {displayChunks.map(chunk => (
                    <span
                      key={chunk.index}
                      className={`${PALETTE[chunk.index % PALETTE.length]} rounded px-1 cursor-pointer hover:opacity-80`}
                      onClick={() => setSelected(chunk)}
                    >
                      {chunk.text.slice(0, 80)}{chunk.text.length > 80 ? '…' : ''}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <ScrollArea className="h-64">
        <div className="flex flex-col gap-2 pr-2">
          {displayChunks.map(chunk => (
            <div
              key={chunk.index}
              className={`rounded-md border p-2 cursor-pointer hover:bg-muted/50 transition-colors ${BORDER_PALETTE[chunk.index % BORDER_PALETTE.length]}`}
              onClick={() => setSelected(chunk)}
            >
              {chunk.summary && (
                <p className="text-xs font-semibold text-muted-foreground mb-1 italic">{chunk.summary}</p>
              )}
              <p className="text-xs font-mono leading-relaxed line-clamp-3">{chunk.text}</p>
              <div className="flex gap-2 mt-1.5 flex-wrap">
                <Badge variant="outline" className="text-[10px] h-4"># {chunk.index}</Badge>
                <Badge variant="outline" className="text-[10px] h-4">{chunk.tokens} tok</Badge>
                {chunk.parentIndex !== undefined && (
                  <Badge variant="secondary" className="text-[10px] h-4">child of #{chunk.parentIndex}</Badge>
                )}
                {chunk.children && (
                  <Badge variant="secondary" className="text-[10px] h-4">{chunk.children.length} children</Badge>
                )}
                {chunk.contextDependencies && chunk.contextDependencies.length > 0 && (
                  <Badge variant="secondary" className="text-[10px] h-4">{chunk.contextDependencies.length} deps</Badge>
                )}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      <Dialog open={selected !== null} onOpenChange={() => setSelected(null)}>
        <DialogContent className="max-w-lg max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Chunk #{selected?.index}</DialogTitle>
          </DialogHeader>
          {selected && <ChunkDetail chunk={selected} />}
        </DialogContent>
      </Dialog>
    </div>
  )
}
