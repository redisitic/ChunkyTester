/**
 * RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval
 *
 * Sarthi et al., Stanford, 2024 (https://arxiv.org/abs/2401.18059)
 *
 * Motivation:
 *   Standard chunking only captures local semantics. For complex financial
 *   documents (10-Ks, analyst reports), a user asking "What were the key
 *   risk factors driving the revenue decline?" needs both:
 *     (a) the specific sentence mentioning a 12% YoY decline
 *     (b) the executive summary contextualising it across all segments
 *   A flat chunk index cannot serve both. RAPTOR builds a tree.
 *
 * Algorithm:
 *  1. Leaf chunking: split into base chunks (paragraph-aware, fixed-size)
 *  2. Embedding: score pairwise similarity between all leaf chunks
 *     (one LLM call, scored as similarity matrix)
 *  3. Clustering: greedy agglomerative clustering of similar leaf chunks
 *     — chunks with similarity >= clusterThreshold merge into one cluster
 *     — each cluster has a max size (maxClusterChunks)
 *  4. Summarisation: one LLM call per cluster → cluster summary node
 *  5. Recursion: if summary count > 1, repeat steps 2–4 on the summaries
 *     (up to maxLevels recursion depth)
 *  6. Output: ALL nodes (leaves + all summary levels) are returned as chunks,
 *     tagged with their tree level and parent indices
 *     → The retriever searches across all levels simultaneously
 *
 * The browser implementation uses the existing LLM-based scoreSimilarityPairs
 * for clustering similarity (no local embedding model required).
 *
 * Cost: O(N) similarity calls + O(N/k) summarisation calls per level,
 * where k = avg cluster size. Typically 2–3 levels for a 20-chunk document.
 */

import { createLLMClient } from '../llmClient'
import { estimateTokens } from '../tokenCounter'
import { scoreSimilarityPairs } from '../embeddings'
import type { Chunk, ChunkResult, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

// ---- Base splitting ----

function splitToParagraphChunks(text: string, maxChars: number): { text: string; start: number; end: number }[] {
  const paragraphs = text.split(/\n\n+/)
  const result: { text: string; start: number; end: number }[] = []
  let buffer = ''
  let bufferStart = 0
  let cursor = 0

  for (const para of paragraphs) {
    const paraStart = text.indexOf(para, cursor)
    if (paraStart === -1) { cursor += para.length; continue }

    if (buffer && buffer.length + para.length + 2 > maxChars) {
      result.push({ text: buffer.trim(), start: bufferStart, end: bufferStart + buffer.trim().length })
      buffer = para
      bufferStart = paraStart
    } else {
      if (!buffer) bufferStart = paraStart
      buffer = buffer ? buffer + '\n\n' + para : para
    }
    cursor = paraStart + para.length
  }
  if (buffer.trim()) result.push({ text: buffer.trim(), start: bufferStart, end: bufferStart + buffer.trim().length })
  return result
}

// ---- Similarity matrix (efficient: only score adjacent + sampled distant pairs) ----

async function buildSimilarityMatrix(
  texts: string[],
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<number[][]> {
  const n = texts.length
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))

  // Always set diagonal to 1
  for (let i = 0; i < n; i++) matrix[i][i] = 1

  if (n <= 1) return matrix

  // Score adjacent pairs (always needed)
  const pairs: [string, string][] = []
  const pairIndices: [number, number][] = []

  for (let i = 0; i < n - 1; i++) {
    pairs.push([texts[i], texts[i + 1]])
    pairIndices.push([i, i + 1])
  }

  // Score window-2 pairs for better clustering signal
  for (let i = 0; i < n - 2; i++) {
    pairs.push([texts[i], texts[i + 2]])
    pairIndices.push([i, i + 2])
  }

  // Cap total pairs to avoid huge prompts
  const maxPairs = 30
  const sampledPairs = pairs.slice(0, maxPairs)
  const sampledIndices = pairIndices.slice(0, maxPairs)

  if (sampledPairs.length > 0) {
    const scores = await scoreSimilarityPairs(sampledPairs, llmConfig)
    usage.input += sampledPairs.reduce((s, [a, b]) => s + a.length + b.length, 0) / 4
    usage.output += scores.length * 4
    for (let k = 0; k < sampledIndices.length; k++) {
      const [i, j] = sampledIndices[k]
      matrix[i][j] = scores[k]
      matrix[j][i] = scores[k]
    }
  }

  return matrix
}

// ---- Greedy agglomerative clustering ----

interface Cluster {
  indices: number[]
  avgSimilarity: number
}

function greedyCluster(
  n: number,
  matrix: number[][],
  similarityThreshold: number,
  maxClusterSize: number
): Cluster[] {
  const assigned = new Set<number>()
  const clusters: Cluster[] = []

  for (let i = 0; i < n; i++) {
    if (assigned.has(i)) continue

    const cluster: number[] = [i]
    assigned.add(i)

    // Greedily add the most similar unassigned neighbours
    for (let j = i + 1; j < n && cluster.length < maxClusterSize; j++) {
      if (assigned.has(j)) continue
      // Must be similar to the LAST element in the cluster (preserves order)
      if (matrix[cluster[cluster.length - 1]][j] >= similarityThreshold) {
        cluster.push(j)
        assigned.add(j)
      }
    }

    const avgSim = cluster.length === 1
      ? 1
      : cluster.slice(0, -1).reduce((s, idx, k) => s + matrix[idx][cluster[k + 1]], 0) / (cluster.length - 1)

    clusters.push({ indices: cluster, avgSimilarity: avgSim })
  }

  return clusters
}

// ---- Summarise a cluster ----

async function summariseCluster(
  clusterTexts: string[],
  level: number,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<string> {
  const client = createLLMClient(llmConfig)
  const combined = clusterTexts.join('\n\n---\n\n')
  const result = await client.complete(
    null,
    `You are building a hierarchical index for a financial document RAG system.
The following ${clusterTexts.length} text ${clusterTexts.length === 1 ? 'passage' : 'passages'} ${level === 1 ? 'are from the source document' : 'are summaries from the previous level of abstraction'}.

Write a comprehensive summary that:
- Captures all key financial figures, ratios, and trends mentioned
- Identifies the companies, time periods, and financial metrics involved
- Preserves specificity (keep numbers, percentages, named entities)
- Is self-contained and retrieval-friendly

Target length: 150–250 words.

Text${clusterTexts.length > 1 ? 's' : ''}:
${combined.slice(0, 6000)}`,
    400
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens
  return result.text.trim()
}

// ---- RAPTOR tree construction ----

interface TreeNode {
  text: string
  level: number                // 0 = leaf, 1 = first-level summary, etc.
  childIndices: number[]       // indices of nodes at the level below
  start: number
  end: number
}

async function buildRaptorTree(
  leaves: { text: string; start: number; end: number }[],
  llmConfig: LLMConfig,
  usage: { input: number; output: number },
  clusterThreshold: number,
  maxClusterSize: number,
  maxLevels: number
): Promise<{ nodes: TreeNode[]; apiCalls: number }> {
  let apiCalls = 0
  const allNodes: TreeNode[] = []

  // Level 0: leaves
  const leafNodes: TreeNode[] = leaves.map(l => ({
    text: l.text,
    level: 0,
    childIndices: [],
    start: l.start,
    end: l.end,
  }))
  allNodes.push(...leafNodes)

  let currentLevelTexts = leaves.map(l => l.text)
  let currentLevelNodeIndices = leafNodes.map((_, i) => i) // indices in allNodes
  let level = 1

  while (currentLevelTexts.length > 1 && level <= maxLevels) {
    // Build similarity matrix for current level
    const matrix = await buildSimilarityMatrix(currentLevelTexts, llmConfig, usage)
    apiCalls++

    // Cluster
    const clusters = greedyCluster(currentLevelTexts.length, matrix, clusterThreshold, maxClusterSize)

    if (clusters.length >= currentLevelTexts.length) {
      // No merging happened — stop recursing
      break
    }

    // Summarise each cluster
    const summaryNodes: TreeNode[] = []
    for (const cluster of clusters) {
      const clusterTexts = cluster.indices.map(i => currentLevelTexts[i])
      const childAllNodeIndices = cluster.indices.map(i => currentLevelNodeIndices[i])

      const summaryText = await summariseCluster(clusterTexts, level, llmConfig, usage)
      apiCalls++

      const summaryNode: TreeNode = {
        text: summaryText,
        level,
        childIndices: childAllNodeIndices,
        start: allNodes[childAllNodeIndices[0]].start,
        end: allNodes[childAllNodeIndices[childAllNodeIndices.length - 1]].end,
      }
      allNodes.push(summaryNode)
      summaryNodes.push(summaryNode)
    }

    currentLevelTexts = summaryNodes.map(n => n.text)
    currentLevelNodeIndices = summaryNodes.map((_, i) => allNodes.length - summaryNodes.length + i)
    level++

    // Throttle
    await new Promise(r => setTimeout(r, 300))
  }

  return { nodes: allNodes, apiCalls }
}

// ---- Chunker definition ----

const raptorChunker: ChunkerDefinition = {
  id: 'raptor',
  label: 'RAPTOR',
  description:
    'Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., Stanford 2024). ' +
    'Builds a hierarchy: leaf chunks → clustered summaries → higher-level summaries. ' +
    'All levels are indexed — retrieval finds both specific facts and broad context. ' +
    'Best for long, complex financial documents.',
  paperRef: 'Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", ICLR 2024',
  configSchema: [
    {
      key: 'maxLeafChars',
      label: 'Max Leaf Chunk Size (chars)',
      description: 'Size of base leaf chunks before tree construction',
      type: 'slider',
      min: 500,
      max: 3500,
      step: 100,
      default: 1500,
    },
    {
      key: 'clusterThreshold',
      label: 'Cluster Similarity Threshold',
      description: 'Min similarity for two chunks to merge into one cluster (0–1)',
      type: 'slider',
      min: 0.4,
      max: 0.95,
      step: 0.05,
      default: 0.65,
    },
    {
      key: 'maxClusterSize',
      label: 'Max Cluster Size (chunks)',
      description: 'Maximum leaf/summary nodes per cluster before summarisation',
      type: 'slider',
      min: 2,
      max: 8,
      step: 1,
      default: 4,
    },
    {
      key: 'maxLevels',
      label: 'Max Tree Levels',
      description: 'Maximum recursion depth (2–3 is usually sufficient)',
      type: 'slider',
      min: 1,
      max: 4,
      step: 1,
      default: 2,
    },
  ],
  defaultConfig: {
    maxLeafChars: 1500,
    clusterThreshold: 0.65,
    maxClusterSize: 4,
    maxLevels: 2,
  },

  async run(text, _structure, config, llmConfig): Promise<ChunkResult> {
    const startTs = Date.now()
    const client = createLLMClient(llmConfig)
    const usage = { input: 0, output: 0 }

    const maxLeafChars      = config.maxLeafChars as number
    const clusterThreshold  = config.clusterThreshold as number
    const maxClusterSize    = config.maxClusterSize as number
    const maxLevels         = config.maxLevels as number

    // ---- Step 1: Leaf chunks ----
    const leaves = splitToParagraphChunks(text, maxLeafChars)

    if (leaves.length === 0) {
      return {
        strategyId: 'raptor',
        strategyLabel: 'RAPTOR',
        chunks: [{ index: 0, text, start: 0, end: text.length, tokens: estimateTokens(text) }],
        durationMs: Date.now() - startTs,
        apiCallCount: 0,
        estimatedCost: 0,
      }
    }

    // ---- Steps 2–5: Build tree ----
    const { nodes, apiCalls } = await buildRaptorTree(
      leaves, llmConfig, usage, clusterThreshold, maxClusterSize, maxLevels
    )

    // ---- Step 6: Convert all nodes to Chunk objects ----
    // All levels are output — retrieval searches all of them.
    // Level tag is encoded in the rationale field.
    const levelLabels: Record<number, string> = { 0: 'Leaf', 1: 'L1 Summary', 2: 'L2 Summary', 3: 'L3 Summary', 4: 'L4 Summary' }

    const chunks: Chunk[] = nodes.map((node, i) => ({
      index: i,
      text: node.text,
      start: node.start,
      end: node.end,
      tokens: estimateTokens(node.text),
      rationale: `${levelLabels[node.level] ?? `L${node.level} Summary`}${node.childIndices.length > 0 ? ` (covers ${node.childIndices.length} child nodes)` : ''}`,
      // Encode parent relationship: summary nodes reference their children
      children: node.level > 0 ? node.childIndices : undefined,
    }))

    return {
      strategyId: 'raptor',
      strategyLabel: 'RAPTOR',
      chunks,
      durationMs: Date.now() - startTs,
      apiCallCount: apiCalls,
      estimatedCost: usage.input * client.costPerInputToken + usage.output * client.costPerOutputToken,
    }
  },
}

export default raptorChunker
