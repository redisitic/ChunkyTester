/**
 * Table-Aware Chunker — critical for finance documents
 *
 * Financial documents (10-Ks, earnings reports, balance sheets) contain
 * large amounts of tabular data. Generic text chunkers treat tables as
 * prose and destroy their structure — splitting mid-row, separating
 * headers from data, or interleaving table rows with surrounding text.
 *
 * Algorithm:
 *  1. Detect table regions in plain text by looking for lines with
 *     consistent columnar structure (tab/multi-space separation, number
 *     density, repeating column counts)
 *  2. Treat each detected table as an atomic unit:
 *       - If table fits within maxChars → single chunk, header always included
 *       - If table exceeds maxChars → split by row groups of TABLE_ROWS_PER_CHUNK,
 *         with the header row prepended to every sub-chunk
 *  3. Non-table text between tables is chunked by paragraph (double newline),
 *     with small paragraphs merged up to maxChars
 *  4. Adjacent text+table regions are kept together when small enough
 *
 * Table detection heuristics (works on PDF-extracted plain text):
 *   - ≥ 3 consecutive lines each with ≥ 2 tab characters OR ≥ 3 multi-spaces
 *   - OR ≥ 3 consecutive lines where ≥ 40% of characters are digits/$/%/,/.
 *   - OR consistent column count across ≥ 3 lines (within ±1)
 *
 * Env alignment:
 *   TABLE_ROWS_PER_CHUNK=25
 *   MAX_CHUNK_CHARS=3500
 */

import { estimateTokens } from '../tokenCounter'
import type { Chunk, ChunkResult } from '@/types'
import type { ChunkerDefinition } from './types'

// ---- Table detection ----

function isTableLike(line: string): boolean {
  if (line.trim().length === 0) return false
  const tabs = (line.match(/\t/g) ?? []).length
  if (tabs >= 2) return true
  const multiSpaces = (line.match(/  {2,}/g) ?? []).length
  if (multiSpaces >= 2) return true
  const numericChars = (line.match(/[\d$%,.()\-]/g) ?? []).length
  if (numericChars / line.length >= 0.4 && line.trim().length > 5) return true
  return false
}

function detectColumnCount(line: string): number {
  if (line.includes('\t')) return line.split('\t').filter(c => c.trim()).length
  return line.trim().split(/\s{2,}/).filter(c => c.trim()).length
}

interface Region {
  type: 'text' | 'table'
  lines: string[]
  startOffset: number
  endOffset: number
}

function detectRegions(text: string): Region[] {
  const lines = text.split('\n')
  const regions: Region[] = []
  let i = 0
  let charOffset = 0
  const lineOffsets: number[] = []

  // Pre-compute character offsets for each line
  let off = 0
  for (const line of lines) {
    lineOffsets.push(off)
    off += line.length + 1 // +1 for \n
  }

  while (i < lines.length) {
    // Look ahead 3 lines to detect table start
    let tableScore = 0
    const colCounts: number[] = []
    for (let j = i; j < Math.min(i + 4, lines.length); j++) {
      if (isTableLike(lines[j])) tableScore++
      colCounts.push(detectColumnCount(lines[j]))
    }
    const colConsistent = colCounts.length >= 2 &&
      Math.max(...colCounts) - Math.min(...colCounts) <= 1 &&
      colCounts[0] >= 2

    if (tableScore >= 3 || (tableScore >= 2 && colConsistent)) {
      // Consume table lines
      const tableLines: string[] = []
      const tableStart = lineOffsets[i]
      while (i < lines.length && (isTableLike(lines[i]) || lines[i].trim() === '')) {
        tableLines.push(lines[i])
        i++
      }
      // Trim trailing blank lines from table
      while (tableLines.length > 0 && tableLines[tableLines.length - 1].trim() === '') {
        tableLines.pop()
        i--
      }
      const tableEnd = i < lines.length ? lineOffsets[i] : text.length
      if (tableLines.length > 0) {
        regions.push({ type: 'table', lines: tableLines, startOffset: tableStart, endOffset: tableEnd })
      }
    } else {
      // Consume text lines until next table region
      const textLines: string[] = []
      const textStart = lineOffsets[i]
      const lookAhead = 3
      while (i < lines.length) {
        let upcomingTableScore = 0
        for (let j = i; j < Math.min(i + lookAhead, lines.length); j++) {
          if (isTableLike(lines[j])) upcomingTableScore++
        }
        if (upcomingTableScore >= 3) break
        textLines.push(lines[i])
        i++
      }
      const textEnd = i < lines.length ? lineOffsets[i] : text.length
      if (textLines.some(l => l.trim())) {
        regions.push({ type: 'text', lines: textLines, startOffset: textStart, endOffset: textEnd })
      }
    }
    charOffset = i < lines.length ? lineOffsets[i] : text.length
  }

  return regions
}

// ---- Table chunking ----

function chunkTable(
  tableLines: string[],
  regionStart: number,
  maxChars: number,
  rowsPerChunk: number,
  startIndex: number
): Chunk[] {
  if (tableLines.length === 0) return []

  // Detect header: first non-blank line, or lines before the first data row
  const headerLines: string[] = []
  let dataStart = 0
  for (let i = 0; i < Math.min(3, tableLines.length); i++) {
    const line = tableLines[i]
    const numericRatio = (line.match(/[\d$%,.]/g) ?? []).length / Math.max(line.length, 1)
    if (numericRatio < 0.25 && line.trim().length > 0) {
      headerLines.push(line)
      dataStart = i + 1
    } else {
      break
    }
  }

  const header = headerLines.join('\n')
  const dataLines = tableLines.slice(dataStart)
  const tableText = tableLines.join('\n')

  // Fits in one chunk
  if (tableText.length <= maxChars) {
    const chunkStart = regionStart
    return [{
      index: startIndex,
      text: tableText,
      start: chunkStart,
      end: chunkStart + tableText.length,
      tokens: estimateTokens(tableText),
      rationale: `Table chunk (${tableLines.length} rows)`,
    }]
  }

  // Split into row groups, always prepend header
  const chunks: Chunk[] = []
  let cursor = regionStart + headerLines.reduce((s, l) => s + l.length + 1, 0)

  for (let r = 0; r < dataLines.length; r += rowsPerChunk) {
    const rowGroup = dataLines.slice(r, r + rowsPerChunk)
    const rowText = rowGroup.join('\n')
    const chunkText = header ? `${header}\n${rowText}` : rowText
    const chunkStart = cursor - (header ? header.length + 1 : 0)
    chunks.push({
      index: startIndex + chunks.length,
      text: chunkText,
      start: Math.max(regionStart, chunkStart),
      end: cursor + rowText.length,
      tokens: estimateTokens(chunkText),
      rationale: `Table rows ${r + 1}–${Math.min(r + rowsPerChunk, dataLines.length)} of ${dataLines.length}`,
    })
    cursor += rowText.length + 1
  }

  return chunks
}

// ---- Text chunking ----

function chunkTextRegion(
  regionText: string,
  regionStart: number,
  maxChars: number,
  startIndex: number
): Chunk[] {
  const paragraphs = regionText.split(/\n\n+/).map(p => p.trim()).filter(p => p.length > 0)
  const segments: string[] = []
  let buffer = ''
  for (const para of paragraphs) {
    if (buffer && buffer.length + para.length + 2 <= maxChars) {
      buffer += '\n\n' + para
    } else {
      if (buffer) segments.push(buffer)
      buffer = para.length > maxChars ? para.slice(0, maxChars) : para
    }
  }
  if (buffer) segments.push(buffer)

  let cursor = regionStart
  return segments.map((seg, i) => {
    const idx = regionText.indexOf(seg, cursor - regionStart)
    const chunkStart = idx === -1 ? cursor : regionStart + idx
    cursor = chunkStart + seg.length
    return {
      index: startIndex + i,
      text: seg,
      start: chunkStart,
      end: chunkStart + seg.length,
      tokens: estimateTokens(seg),
    }
  })
}

// ---- Chunker definition ----

const tableAwareChunker: ChunkerDefinition = {
  id: 'table_aware',
  label: 'Table-Aware',
  description:
    'Detects tabular regions in financial documents and keeps them intact. ' +
    'Tables are chunked by row-group with headers always prepended. ' +
    'Prose between tables is chunked by paragraph. Critical for 10-Ks, earnings reports, and balance sheets.',
  paperRef: 'Finance RAG best practice — table integrity for structured financial data',
  configSchema: [
    {
      key: 'maxChunkChars',
      label: 'Max Chunk Size (chars)',
      description: 'Hard limit per chunk for prose regions',
      type: 'slider',
      min: 500,
      max: 5000,
      step: 100,
      default: 3500,
    },
    {
      key: 'rowsPerChunk',
      label: 'Table Rows per Chunk',
      description: 'Max data rows per chunk when splitting large tables',
      type: 'slider',
      min: 5,
      max: 50,
      step: 5,
      default: 25,
    },
  ],
  defaultConfig: { maxChunkChars: 3500, rowsPerChunk: 25 },

  async run(text, _structure, config): Promise<ChunkResult> {
    const start = Date.now()
    const maxChunkChars = config.maxChunkChars as number
    const rowsPerChunk  = config.rowsPerChunk as number

    const regions = detectRegions(text)
    const chunks: Chunk[] = []

    for (const region of regions) {
      const regionText = region.lines.join('\n')
      if (region.type === 'table') {
        const tableChunks = chunkTable(
          region.lines, region.startOffset, maxChunkChars, rowsPerChunk, chunks.length
        )
        chunks.push(...tableChunks)
      } else {
        const textChunks = chunkTextRegion(
          regionText, region.startOffset, maxChunkChars, chunks.length
        )
        chunks.push(...textChunks)
      }
    }

    // Re-index sequentially
    chunks.forEach((c, i) => { c.index = i })

    // Fallback if no regions detected (e.g. pure prose)
    if (chunks.length === 0) {
      chunks.push({ index: 0, text, start: 0, end: text.length, tokens: estimateTokens(text) })
    }

    return {
      strategyId: 'table_aware',
      strategyLabel: 'Table-Aware',
      chunks,
      durationMs: Date.now() - start,
      apiCallCount: 0,
      estimatedCost: 0,
    }
  },
}

export default tableAwareChunker
