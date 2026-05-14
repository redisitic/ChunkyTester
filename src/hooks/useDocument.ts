import { useState, useCallback } from 'react'
import { extractPdf } from '@/lib/extractors/pdfExtractor'
import { extractHtml } from '@/lib/extractors/htmlExtractor'
import type { DocumentState } from '@/types'

type Step = 'upload' | 'select' | 'compare'

interface UseDocumentReturn {
  step: Step
  docState: DocumentState | null
  loading: boolean
  error: string | null
  loadFile: (file: File) => Promise<void>
  setSelection: (start: number, end: number) => void
  advanceTo: (step: Step) => void
  reset: () => void
}

export function useDocument(): UseDocumentReturn {
  const [step, setStep] = useState<Step>('upload')
  const [docState, setDocState] = useState<DocumentState | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadFile = useCallback(async (file: File) => {
    setLoading(true)
    setError(null)
    try {
      const ext = file.name.split('.').pop()?.toLowerCase()
      const { fullText, structure } =
        ext === 'pdf'
          ? await extractPdf(file)
          : await extractHtml(file)

      setDocState({
        filename: file.name,
        fullText,
        structure,
        selectedStart: 0,
        selectedEnd: fullText.length,
      })
      setStep('select')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to extract document')
    } finally {
      setLoading(false)
    }
  }, [])

  const setSelection = useCallback((start: number, end: number) => {
    setDocState(prev => prev ? { ...prev, selectedStart: start, selectedEnd: end } : prev)
  }, [])

  const advanceTo = useCallback((target: Step) => {
    setStep(target)
  }, [])

  const reset = useCallback(() => {
    setStep('upload')
    setDocState(null)
    setError(null)
  }, [])

  return { step, docState, loading, error, loadFile, setSelection, advanceTo, reset }
}
