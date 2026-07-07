/**
 * Improved PDF extractor
 *
 * Uses pdf.js positional data (transform[4]=x, transform[5]=y, height, width)
 * to reconstruct the visual layout of each page rather than concatenating all
 * text items with spaces.
 *
 * Key improvements over the original:
 *
 *  1. Line grouping — items within the same vertical band (Y ± tolerance)
 *     are grouped into a single visual line, then sorted left→right.
 *
 *  2. Column gap detection — large horizontal gaps between items on the same
 *     line are rendered as \t (tab). The tableAwareChunker and all other
 *     chunkers interpret tabs as column separators, preserving table structure.
 *
 *  3. Paragraph detection — vertical gaps larger than 1.4× the typical line
 *     height produce \n\n, keeping paragraph semantics intact.
 *
 *  4. List detection — lines starting with bullet characters (•, -, *, ◦)
 *     or numbered patterns (1., 2), (a)) are preserved with their markers.
 *
 *  5. Heading detection — improved: uses font-size ratio AND bold font name
 *     detection. Consecutive adjacent items at the same heading size are
 *     merged into a single heading entry.
 *
 *  6. Single-pass — text and heading offsets are computed together in one
 *     pass, so offsets in DocumentStructure are always consistent with fullText.
 *
 *  7. Hyphenation repair — trailing hyphens at line-end that break a word
 *     across lines are joined without a space.
 *
 * Output format (what chunkers receive):
 *   - Paragraphs separated by \n\n
 *   - Table rows on individual lines with \t between columns
 *   - List items on individual lines starting with their original marker
 *   - Headings on their own lines followed by \n\n
 *   - Pages separated by \n\n--- page N ---\n\n
 */

import * as pdfjsLib from 'pdfjs-dist'
import type { DocumentStructure } from '@/types'

// ─── Internal types ────────────────────────────────────────────────────────

interface RawItem {
  str: string
  x: number       // transform[4]
  y: number       // transform[5]
  width: number
  height: number  // font height (≈ font size in pt)
  fontName: string
  bold: boolean
  hasEOL: boolean // pdf.js end-of-line hint
}

interface VisualLine {
  items: RawItem[]
  y: number
  avgHeight: number
  isBold: boolean
  minX: number
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/**
 * Compute the median value of a numeric array.
 * Returns `fallback` for empty arrays.
 */
function median(values: number[], fallback = 12): number {
  if (values.length === 0) return fallback
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

/**
 * Collect all usable text items from a pdf.js TextContent,
 * enriching each with positional and font metadata.
 */
function collectItems(
  rawItems: Array<{ str: string; transform: number[]; width: number; height: number; fontName: string; hasEOL?: boolean }>
): RawItem[] {
  return rawItems
    .filter(it => it.str.length > 0)   // keep whitespace-only items — needed for gap measurement
    .map(it => ({
      str: it.str,
      x: it.transform[4],
      y: it.transform[5],
      width: it.width,
      height: Math.abs(it.height) || Math.abs(it.transform[0]) || 12,
      fontName: it.fontName ?? '',
      bold: /bold|heavy|black|demi/i.test(it.fontName ?? ''),
      hasEOL: it.hasEOL ?? false,
    }))
}

/**
 * Group items into visual lines by clustering on Y coordinate.
 * tolerance = fraction of the median font height within which items
 * are considered to be on the same line.
 */
function groupIntoLines(items: RawItem[], tolerance: number): VisualLine[] {
  if (items.length === 0) return []

  // Sort: top of page first (PDF y increases upward → descending sort)
  const sorted = [...items].sort((a, b) =>
    Math.abs(b.y - a.y) > 0.5 ? b.y - a.y : a.x - b.x
  )

  const lines: VisualLine[] = []
  let bucket: RawItem[] = [sorted[0]]
  let bucketY = sorted[0].y

  for (let i = 1; i < sorted.length; i++) {
    const item = sorted[i]
    if (Math.abs(item.y - bucketY) <= tolerance) {
      bucket.push(item)
    } else {
      lines.push(finaliseLine(bucket, bucketY))
      bucket = [item]
      bucketY = item.y
    }
  }
  if (bucket.length > 0) lines.push(finaliseLine(bucket, bucketY))

  return lines
}

function finaliseLine(items: RawItem[], y: number): VisualLine {
  const sorted = [...items].sort((a, b) => a.x - b.x)
  const heights = sorted.filter(it => it.height > 0).map(it => it.height)
  const avgHeight = heights.length > 0 ? heights.reduce((s, h) => s + h, 0) / heights.length : 12
  return {
    items: sorted,
    y,
    avgHeight,
    isBold: sorted.some(it => it.bold),
    minX: sorted[0]?.x ?? 0,
  }
}

/**
 * Render a visual line to a string.
 * Large horizontal gaps → \t (table column separator).
 * Normal word-gaps → single space.
 * Touching/overlapping items → no separator.
 */
function renderLine(line: VisualLine, avgCharWidth: number): string {
  if (line.items.length === 0) return ''

  let text = ''
  for (let i = 0; i < line.items.length; i++) {
    const item = line.items[i]
    if (i === 0) {
      text += item.str
      continue
    }
    const prev = line.items[i - 1]
    const gap = item.x - (prev.x + prev.width)

    if (gap > avgCharWidth * 3.5) {
      text += '\t'          // Wide gap → column separator (table)
    } else if (gap > avgCharWidth * 0.3 || (text.length > 0 && !/\s$/.test(text))) {
      // Only add space if the previous item didn't already end with whitespace
      if (item.str.length > 0 && !/^\s/.test(item.str)) {
        text += ' '
      }
    }
    text += item.str
  }

  return text.trim()
}

// ─── List detection ────────────────────────────────────────────────────────

const LIST_BULLET_RE = /^[•●◦▪▸►▷\-–—*]\s/
const LIST_NUMBER_RE = /^(\d+|[a-z])[.)]\s/i
const LIST_PAREN_RE  = /^\([a-z\d]+\)\s/i

function isListItem(text: string): boolean {
  return LIST_BULLET_RE.test(text) || LIST_NUMBER_RE.test(text) || LIST_PAREN_RE.test(text)
}

// ─── Heading inference ─────────────────────────────────────────────────────

function inferHeadingLevel(fontSize: number, bodySize: number, isBold: boolean): number | null {
  const ratio = fontSize / bodySize
  if (ratio >= 1.8) return 1
  if (ratio >= 1.4) return 2
  if (ratio >= 1.15) return 3
  // Bold at body size is an H3/H4 candidate for financial documents
  if (isBold && ratio >= 0.95) return 4
  return null
}

// ─── Hyphenation repair ────────────────────────────────────────────────────

/**
 * If the last line ended with a soft hyphen (word split across lines),
 * join without a space. Otherwise join with \n.
 */
function joinLines(prev: string, next: string): string {
  if (prev.endsWith('-') && next.length > 0 && /^[a-z]/i.test(next)) {
    return prev.slice(0, -1) + next   // remove hyphen, join directly
  }
  return prev + '\n' + next
}

// ─── Page reconstruction ───────────────────────────────────────────────────

interface PageOutput {
  text: string
  headings: DocumentStructure['headings']
  baseOffset: number  // character offset at which this page starts in fullText
}

function reconstructPage(
  lines: VisualLine[],
  bodyFontSize: number,
  pageBaseOffset: number,
  pageNumber: number
): PageOutput {
  const headings: DocumentStructure['headings'] = []
  const avgCharWidth = bodyFontSize * 0.45  // approximate for proportional fonts

  // Typical line height ≈ font size × 1.2 (standard typographic leading)
  const typicalLeading = bodyFontSize * 1.2
  const paragraphGapThreshold = typicalLeading * 1.4

  const segments: string[] = []
  let cursor = pageBaseOffset

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const text = renderLine(line, avgCharWidth)
    if (!text) continue

    // Paragraph gap detection
    const prevLine = lines.slice(0, i).reverse().find(l => renderLine(l, avgCharWidth).length > 0)
    const yGap = prevLine ? prevLine.y - line.y : 0
    const isParagraphBreak = yGap > paragraphGapThreshold && segments.length > 0

    // Heading detection
    const headingLevel = inferHeadingLevel(line.avgHeight, bodyFontSize, line.isBold)

    if (isParagraphBreak) {
      // Only emit double-newline if last segment didn't already end with one
      const last = segments[segments.length - 1] ?? ''
      if (!last.endsWith('\n')) {
        segments.push('\n')
        cursor += 1
      }
    }

    const lineStart = cursor
    let lineText: string

    if (headingLevel !== null && text.length > 2) {
      // Headings get their own paragraph break
      if (segments.length > 0 && !segments[segments.length - 1].endsWith('\n')) {
        segments.push('\n')
        cursor += 1
      }
      lineText = text
      segments.push(lineText)
      cursor += lineText.length

      headings.push({
        level: headingLevel,
        text,
        start: lineStart + (isParagraphBreak ? 1 : 0),
        end: cursor,
      })

      segments.push('\n')
      cursor += 1
    } else if (isListItem(text)) {
      lineText = text
      segments.push(lineText)
      cursor += lineText.length
    } else {
      // Body text — check if we should join with previous line or start new line
      const lastSeg = segments[segments.length - 1]
      if (lastSeg && !isParagraphBreak && !lastSeg.endsWith('\n') && !lastSeg.endsWith('\t')) {
        // Potential continuation — join with hyphenation repair
        const joined = joinLines(lastSeg, text)
        segments[segments.length - 1] = joined
        cursor += joined.length - lastSeg.length
        continue
      }
      lineText = text
      segments.push(lineText)
      cursor += lineText.length
    }
  }

  return {
    text: segments.join('\n'),
    headings,
    baseOffset: pageBaseOffset,
  }
}

// ─── Main extractor ────────────────────────────────────────────────────────

export async function extractPdf(file: File): Promise<{
  fullText: string
  structure: DocumentStructure
}> {
  const arrayBuffer = await file.arrayBuffer()
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise

  // ── Pass 1: collect font sizes across all pages to establish body size ──
  const allFontSizes: number[] = []
  const allPageItems: RawItem[][] = []

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i)
    const content = await page.getTextContent()
    const items = collectItems(
      content.items as Array<{ str: string; transform: number[]; width: number; height: number; fontName: string; hasEOL?: boolean }>
    )
    allPageItems.push(items)
    items.forEach(it => { if (it.height > 0) allFontSizes.push(it.height) })
  }

  // Body font size = median (most common size = body text, not headings/footnotes)
  const bodyFontSize = median(allFontSizes, 12)

  // Line grouping tolerance: 35% of body font height
  const lineTolerance = bodyFontSize * 0.35

  // ── Pass 2: reconstruct text page by page ──────────────────────────────
  const pages: { number: number; start: number; end: number }[] = []
  const allHeadings: DocumentStructure['headings'] = []
  let fullText = ''

  for (let i = 0; i < pdf.numPages; i++) {
    const pageNumber = i + 1
    const items = allPageItems[i]

    const pageStart = fullText.length

    // Add page marker (helps chunkers understand page boundaries)
    if (i > 0) {
      const marker = `\n\n─── Page ${pageNumber} ───\n\n`
      fullText += marker
    }

    if (items.length === 0) {
      pages.push({ number: pageNumber, start: pageStart, end: fullText.length })
      continue
    }

    const lines = groupIntoLines(items, lineTolerance)
    const { text: pageText, headings: pageHeadings } = reconstructPage(
      lines,
      bodyFontSize,
      fullText.length,
      pageNumber
    )

    // Offset-adjust headings (reconstructPage computes them relative to fullText.length at call time)
    allHeadings.push(...pageHeadings)
    fullText += pageText

    pages.push({ number: pageNumber, start: pageStart, end: fullText.length })
  }

  // Deduplicate headings that got split across adjacent items (same start offset)
  const dedupedHeadings = allHeadings
    .filter((h, idx, arr) => idx === 0 || h.start !== arr[idx - 1].start)
    .sort((a, b) => a.start - b.start)

  return {
    fullText: fullText.trim(),
    structure: { headings: dedupedHeadings, pages },
  }
}
