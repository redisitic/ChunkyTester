import * as pdfjsLib from 'pdfjs-dist'
import type { DocumentStructure } from '@/types'

interface FontInfo {
  size: number
  text: string
}

function inferHeadingLevel(fontSize: number, bodySize: number): number | null {
  const ratio = fontSize / bodySize
  if (ratio >= 1.8) return 1
  if (ratio >= 1.4) return 2
  if (ratio >= 1.15) return 3
  return null
}

export async function extractPdf(file: File): Promise<{
  fullText: string
  structure: DocumentStructure
}> {
  const arrayBuffer = await file.arrayBuffer()
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise

  const pages: { number: number; start: number; end: number }[] = []
  const headings: DocumentStructure['headings'] = []
  let fullText = ''
  const fontSizes: number[] = []

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i)
    const content = await page.getTextContent()

    const pageStart = fullText.length
    const items = content.items as Array<{ str: string; transform: number[]; height: number }>

    const fontInfos: FontInfo[] = items.map(item => ({
      size: item.height,
      text: item.str,
    }))

    fontInfos.forEach(f => { if (f.size > 0) fontSizes.push(f.size) })

    const pageText = items.map(item => item.str).join(' ')
    fullText += pageText
    if (i < pdf.numPages) fullText += '\n\n'

    pages.push({ number: i, start: pageStart, end: fullText.length })
  }

  const sortedSizes = [...fontSizes].sort((a, b) => a - b)
  const bodySize = sortedSizes[Math.floor(sortedSizes.length * 0.5)] || 12

  let offset = 0
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i)
    const content = await page.getTextContent()
    const items = content.items as Array<{ str: string; height: number }>

    for (const item of items) {
      const level = inferHeadingLevel(item.height, bodySize)
      if (level && item.str.trim().length > 2) {
        headings.push({
          level,
          text: item.str.trim(),
          start: offset,
          end: offset + item.str.length,
        })
      }
      offset += item.str.length + 1
    }
    if (i < pdf.numPages) offset += 1
  }

  return { fullText, structure: { headings, pages } }
}
