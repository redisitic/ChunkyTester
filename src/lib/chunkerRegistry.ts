import recursiveSemanticChunker from './chunkers/recursiveSemanticChunker'
import lateChunker from './chunkers/lateChunker'
import structureAwareChunker from './chunkers/structureAwareChunker'
import parentChildChunker from './chunkers/parentChildChunker'
import slidingWindowSummaryChunker from './chunkers/slidingWindowSummaryChunker'
import kamradtSemanticChunker from './chunkers/kamradtSemanticChunker'
import agenticChunker from './chunkers/agenticChunker'
import tableAwareChunker from './chunkers/tableAwareChunker'
import contextualRetrievalChunker from './chunkers/contextualRetrievalChunker'
import raptorChunker from './chunkers/raptorChunker'
import type { ChunkerDefinition } from './chunkers/types'

export const CHUNKERS: ChunkerDefinition[] = [
  // ── Production-quality strategies (recommended) ──────────────────────────
  tableAwareChunker,          // finance-critical: keeps tables intact
  contextualRetrievalChunker, // Anthropic 2024: document-level context prefix
  raptorChunker,              // Stanford ICLR 2024: multi-level summary tree
  recursiveSemanticChunker,   // sentence-pair similarity + merge/re-split
  lateChunker,                // full-document context for boundary decisions
  structureAwareChunker,      // heading hierarchy boundaries
  parentChildChunker,         // small retrieval + large context pattern
  agenticChunker,             // atomic propositions (Dense X Retrieval)

  // ── Legacy / baseline ───────────────────────────────────────────────────
  slidingWindowSummaryChunker, // local window summary (predates contextual retrieval)
  kamradtSemanticChunker,      // percentile-based similarity drops — legacy baseline
]

export function getChunker(id: string): ChunkerDefinition | undefined {
  return CHUNKERS.find(c => c.id === id)
}
