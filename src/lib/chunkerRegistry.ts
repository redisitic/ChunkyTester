import recursiveSemanticChunker from './chunkers/recursiveSemanticChunker'
import lateChunker from './chunkers/lateChunker'
import structureAwareChunker from './chunkers/structureAwareChunker'
import parentChildChunker from './chunkers/parentChildChunker'
import slidingWindowSummaryChunker from './chunkers/slidingWindowSummaryChunker'
import kamradtSemanticChunker from './chunkers/kamradtSemanticChunker'
import agenticChunker from './chunkers/agenticChunker'
import type { ChunkerDefinition } from './chunkers/types'

export const CHUNKERS: ChunkerDefinition[] = [
  recursiveSemanticChunker,
  lateChunker,
  structureAwareChunker,
  parentChildChunker,
  slidingWindowSummaryChunker,
  kamradtSemanticChunker,
  agenticChunker,
]

export function getChunker(id: string): ChunkerDefinition | undefined {
  return CHUNKERS.find(c => c.id === id)
}
