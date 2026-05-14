import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { FileText, Upload } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { estimateTokens } from '@/lib/tokenCounter'
import type { DocumentState } from '@/types'

interface Props {
  onFile: (file: File) => Promise<void>
  docState: DocumentState | null
  loading: boolean
  error: string | null
  onContinue: () => void
}

export function FileUpload({ onFile, docState, loading, error, onContinue }: Props) {
  const onDrop = useCallback((accepted: File[]) => {
    if (accepted[0]) onFile(accepted[0])
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'], 'text/html': ['.html', '.htm'] },
    maxFiles: 1,
    disabled: loading,
  })

  return (
    <div className="flex flex-col items-center gap-6 py-16 px-4 max-w-xl mx-auto">
      <div className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight mb-2">ChunkyTester</h1>
        <p className="text-muted-foreground text-sm">Compare production RAG chunking strategies side by side</p>
      </div>

      <Card
        {...getRootProps()}
        className={`w-full border-2 border-dashed cursor-pointer transition-colors p-10 flex flex-col items-center gap-3 ${
          isDragActive ? 'border-primary bg-muted/40' : 'border-border hover:border-primary/50'
        } ${loading ? 'opacity-50 pointer-events-none' : ''}`}
      >
        <input {...getInputProps()} />
        <Upload className="w-8 h-8 text-muted-foreground" />
        <p className="text-sm text-center text-muted-foreground">
          {isDragActive ? 'Drop it here' : 'Drag & drop a PDF or HTML file, or click to browse'}
        </p>
        <div className="flex gap-2">
          <Badge variant="outline">.pdf</Badge>
          <Badge variant="outline">.html</Badge>
          <Badge variant="outline">.htm</Badge>
        </div>
      </Card>

      {error && (
        <Alert variant="destructive" className="w-full">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {loading && (
        <p className="text-sm text-muted-foreground animate-pulse">Extracting document…</p>
      )}

      {docState && !loading && (
        <Card className="w-full p-4 flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-muted-foreground" />
            <span className="font-medium text-sm truncate">{docState.filename}</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
            <span>Characters</span>
            <span className="text-foreground font-mono">{docState.fullText.length.toLocaleString()}</span>
            <span>Est. tokens</span>
            <span className="text-foreground font-mono">{estimateTokens(docState.fullText).toLocaleString()}</span>
            {docState.structure.pages && (
              <>
                <span>Pages</span>
                <span className="text-foreground font-mono">{docState.structure.pages.length}</span>
              </>
            )}
            <span>Headings detected</span>
            <span className="text-foreground font-mono">{docState.structure.headings.length}</span>
          </div>
          <Button className="w-full mt-1" onClick={onContinue}>
            Continue →
          </Button>
        </Card>
      )}
    </div>
  )
}
