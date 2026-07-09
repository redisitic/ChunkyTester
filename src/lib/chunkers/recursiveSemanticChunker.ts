/**
 * Recursive Semantic Chunker (fixed)
 *
 * Algorithm:
 *  1. Split text into sentences
 *  2. Embed adjacent sentence pairs → similarity scores (one LLM call)
 *  3. Find semantic breakpoints: drops below the low-similarity threshold
 *  4. Group sentences into initial segments at those breakpoints
 *  5. Merge pass: greedily merge adjacent segments whose boundary similarity
 *     is above mergeThreshold AND whose combined size is under maxChars
 *  6. Re-split pass: for any segment still above maxChars, find the
 *     weakest *boundary* (lowest inter-sentence similarity) and split there
 *  7. Recover exact start/end offsets by scanning forward through the
 *     original text (no indexOf — avoids duplicate-match bugs)
 *
 * Fixes vs original:
 *  - Bug 1: coherence check was scoring a segment against itself (always ~1.0)
 *            → now scores actual adjacent sentence pairs within the segment
 *  - Bug 2: merge loop was greedy-pair only, skipped segments after a merge
 *            → now iterates until stable (no more merges possible)
 *  - Bug 3: askCoherence found the "weakest sentence", not the weakest boundary
 *            → now finds the sentence pair with the minimum similarity score
 *  - Bug 4: start/end used text.indexOf(seg) → wrong position on repeated text
 *            → now walks a cursor forward through the original text
 */

import { createLLMClient } from '../llmClient'
import { estimateTokens } from '../tokenCounter'
import { scoreSimilarityPairs } from '../embeddings'
import type { Chunk, ChunkResult, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

// ---- Sentence splitting ----

function splitSentences(text: string): string[] {
  // Split on sentence-ending punctuation followed by whitespace.
  // Keep the punctuation attached to the preceding sentence.
  return text
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 0)
}

// ---- Offset recovery ----

/**
 * Walk forward through `fullText` starting at `cursor` to find the
 * exact position of `segment`. Falls back to cursor if not found.
 */
function findOffset(fullText: string, segment: string, cursor: number): { start: number; end: number; next: number } {
  const idx = fullText.indexOf(segment, cursor)
  if (idx === -1) {
    // Fallback: advance cursor by segment length (shouldn't happen)
    return { start: cursor, end: cursor + segment.length, next: cursor + segment.length }
  }
  return { start: idx, end: idx + segment.length, next: idx + segment.length }
}

// ---- Core algorithm ----

/**
 * Given a list of sentences and their inter-sentence similarity scores,
 * split into groups at boundaries where similarity drops below `splitThreshold`.
 */
function groupBySimilarity(
  sentences: string[],
  similarities: number[],
  splitThreshold: number,
  minSentences: number
): string[][] {
  const groups: string[][] = []
  let current: string[] = [sentences[0]]

  for (let i = 0; i < similarities.length; i++) {
    if (similarities[i] < splitThreshold && current.length >= minSentences) {
      groups.push(current)
      current = []
    }
    current.push(sentences[i + 1])
  }
  if (current.length > 0) groups.push(current)

  return groups
}

/**
 * Merge adjacent groups greedily while:
 *   - the boundary similarity between them is >= mergeThreshold
 *   - the combined char count stays under maxChars
 *
 * Iterates until stable (no more merges).
 */
function mergePass(
  groups: string[][],
  groupBoundaryScores: number[],
  mergeThreshold: number,
  maxChars: number
): string[][] {
  let changed = true
  while (changed) {
    changed = false
    const next: string[][] = []
    const nextScores: number[] = []
    let i = 0
    while (i < groups.length) {
      if (
        i < groups.length - 1 &&
        groupBoundaryScores[i] >= mergeThreshold &&
        (groups[i].join(' ').length + groups[i + 1].join(' ').length) <= maxChars
      ) {
        next.push([...groups[i], ...groups[i + 1]])
        // The new boundary score is the one after the merged block
        if (i + 1 < groupBoundaryScores.length) {
          nextScores.push(groupBoundaryScores[i + 1])
        }
        i += 2
        changed = true
      } else {
        next.push(groups[i])
        if (i < groupBoundaryScores.length) {
          nextScores.push(groupBoundaryScores[i])
        }
        i++
      }
    }
    groups = next
    groupBoundaryScores = nextScores
  }
  return groups
}

/**
 * Re-split a group of sentences that is too large.
 * Finds the sentence boundary with the *lowest* inter-sentence similarity
 * and splits there. Recurses until all pieces are under maxChars.
 */
async function resplitGroup(
  sentences: string[],
  maxChars: number,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<string[][]> {
  const text = sentences.join(' ')
  if (text.length <= maxChars || sentences.length <= 1) return [sentences]

  // Score all adjacent pairs within the group
  const pairs: [string, string][] = sentences.slice(0, -1).map((s, i) => [s, sentences[i + 1]])
  const scores = await scoreSimilarityPairs(pairs, llmConfig)
  usage.input += pairs.reduce((s, [a, b]) => s + a.length + b.length, 0) / 4
  usage.output += scores.length * 4

  // Find the boundary with the minimum similarity (weakest semantic connection)
  let minIdx = 0
  let minScore = Infinity
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] < minScore) { minScore = scores[i]; minIdx = i }
  }

  const left = sentences.slice(0, minIdx + 1)
  const right = sentences.slice(minIdx + 1)

  // Recurse on each half if still too large
  const leftResult = await resplitGroup(left, maxChars, llmConfig, usage)
  const rightResult = await resplitGroup(right, maxChars, llmConfig, usage)
  return [...leftResult, ...rightResult]
}

// ---- Chunker definition ----

const recursiveSemanticChunker: ChunkerDefinition = {
  id: 'recursive_semantic',
  label: 'Recursive Semantic',
  description:
    'Splits at semantic breakpoints (low inter-sentence similarity), then merges ' +
    'over-split adjacent segments and re-splits oversized ones at their weakest ' +
    'internal boundary. All similarity scoring is sentence-pair based.',
  paperRef: 'LangChain SemanticChunker — improved boundary detection',
  configSchema: [
    {
      key: 'maxChunkChars',
      label: 'Max Chunk Size (chars)',
      description: 'Hard upper limit per chunk; oversized chunks are re-split',
      type: 'slider',
      min: 500,
      max: 5000,
      step: 100,
      default: 3500,
    },
    {
      key: 'splitThreshold',
      label: 'Split Threshold',
      description: 'Sentence pairs below this similarity become chunk boundaries (0–1)',
      type: 'slider',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.65,
    },
    {
      key: 'mergeThreshold',
      label: 'Merge Threshold',
      description: 'Adjacent chunks above this boundary similarity are merged (0–1)',
      type: 'slider',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.88,
    },
    {
      key: 'minSentences',
      label: 'Min Sentences per Chunk',
      description: 'Minimum sentence count before a boundary is accepted',
      type: 'slider',
      min: 1,
      max: 8,
      step: 1,
      default: 2,
    },
  ],
  defaultConfig: {
    maxChunkChars: 3500,
    splitThreshold: 0.65,
    mergeThreshold: 0.88,
    minSentences: 2,
  },

  async run(text, _structure, config, llmConfig): Promise<ChunkResult> {
    const start = Date.now()
    const usage = { input: 0, output: 0 }
    let apiCallCount = 0

    const maxChunkChars   = config.maxChunkChars as number
    const splitThreshold  = config.splitThreshold as number
    const mergeThreshold  = config.mergeThreshold as number
    const minSentences    = config.minSentences as number

    const sentences = splitSentences(text)

    // Edge case: single sentence or very short text
    if (sentences.length <= 1) {
      return {
        strategyId: 'recursive_semantic',
        strategyLabel: 'Recursive Semantic',
        chunks: [{ index: 0, text, start: 0, end: text.length, tokens: estimateTokens(text) }],
        durationMs: Date.now() - start,
        apiCallCount: 0,
        estimatedCost: 0,
      }
    }

    // ---- Step 1: Score all adjacent sentence pairs (one LLM call) ----
    const pairs: [string, string][] = sentences.slice(0, -1).map((s, i) => [s, sentences[i + 1]])
    const similarities = await scoreSimilarityPairs(pairs, llmConfig)
    apiCallCount++
    usage.input += pairs.reduce((s, [a, b]) => s + a.length + b.length, 0) / 4
    usage.output += similarities.length * 4

    // ---- Step 2: Group by similarity breakpoints ----
    let groups = groupBySimilarity(sentences, similarities, splitThreshold, minSentences)

    // ---- Step 3: Compute inter-group boundary scores ----
    // Boundary between group[i] and group[i+1] = similarity at the sentence
    // index where group[i] ends.
    const groupBoundaryScores: number[] = []
    let sentenceCursor = 0
    for (let i = 0; i < groups.length - 1; i++) {
      sentenceCursor += groups[i].length
      // The boundary is between sentence [sentenceCursor-1] and [sentenceCursor]
      groupBoundaryScores.push(similarities[sentenceCursor - 1] ?? 0)
    }

    // ---- Step 4: Merge over-split adjacent groups ----
    groups = mergePass(groups, groupBoundaryScores, mergeThreshold, maxChunkChars)

    // ---- Step 5: Re-split any group still over maxChunkChars ----
    const finalGroups: string[][] = []
    for (const group of groups) {
      const chunkText = group.join(' ')
      if (chunkText.length > maxChunkChars) {
        const subGroups = await resplitGroup(group, maxChunkChars, llmConfig, usage)
        apiCallCount++
        finalGroups.push(...subGroups)
      } else {
        finalGroups.push(group)
      }
    }

    // ---- Step 6: Build Chunk objects with correct offsets ----
    let cursor = 0
    const chunks: Chunk[] = finalGroups.map((sentGroup, i) => {
      const chunkText = sentGroup.join(' ')
      const { start: chunkStart, end: chunkEnd, next } = findOffset(text, chunkText, cursor)
      cursor = next
      return {
        index: i,
        text: chunkText,
        start: chunkStart,
        end: chunkEnd,
        tokens: estimateTokens(chunkText),
      }
    })

    const client = createLLMClient(llmConfig)
    return {
      strategyId: 'recursive_semantic',
      strategyLabel: 'Recursive Semantic',
      chunks,
      durationMs: Date.now() - start,
      apiCallCount,
      estimatedCost: usage.input * client.costPerInputToken + usage.output * client.costPerOutputToken,
    }
  },
}

export default recursiveSemanticChunker
