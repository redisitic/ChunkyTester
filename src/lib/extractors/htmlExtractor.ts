import type { DocumentStructure } from '@/types'

const HEADING_TAGS = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'] as const

function walkNode(node: Node, parts: string[], offsets: Map<Node, number>): void {
  offsets.set(node, parts.join('').length)
  if (node.nodeType === Node.TEXT_NODE) {
    parts.push(node.textContent ?? '')
  } else {
    for (const child of Array.from(node.childNodes)) {
      walkNode(child, parts, offsets)
    }
  }
}

export function extractHtml(file: File): Promise<{
  fullText: string
  structure: DocumentStructure
}> {
  return file.text().then(raw => {
    const doc = new DOMParser().parseFromString(raw, 'text/html')

    doc.querySelectorAll('script, style, noscript, head').forEach(el => el.remove())

    const body = doc.body ?? doc.documentElement
    const parts: string[] = []
    const offsets = new Map<Node, number>()
    walkNode(body, parts, offsets)
    const fullText = parts.join('')

    const headings: DocumentStructure['headings'] = []

    body.querySelectorAll(HEADING_TAGS.join(',')).forEach(el => {
      const level = parseInt(el.tagName[1], 10)
      const start = offsets.get(el) ?? 0
      const text = el.textContent?.trim() ?? ''
      headings.push({ level, text, start, end: start + text.length })
    })

    headings.sort((a, b) => a.start - b.start)

    return { fullText, structure: { headings } }
  })
}
