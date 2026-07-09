/**
 * RAPTOR-KG Hybrid Chunker
 *
 * Combines two strategies into a single pipeline:
 *
 *   1. RAPTOR (Sarthi et al., Stanford ICLR 2024)
 *      Builds a recursive summary tree so retrieval can find both granular
 *      facts (leaf chunks) and broad context (L1/L2 summary nodes).
 *
 *   2. Agentic KG Propositions (extended from Chen et al., Dense X, 2023)
 *      For every node in the tree, extract relational triples in the form:
 *        (Subject [type]) → [predicate] → (Object [type]) | period?
 *      These triples serve two purposes:
 *        a) embeddingInput — linearized triples are richer signal for Voyage
 *           than raw prose; decontextualized and entity-dense.
 *        b) summary (JSON) — structured payload ready for direct ingestion
 *           into Neo4j / RDF / Wikidata / any KG tool.
 *
 * Pipeline:
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │  1. Split text → leaf chunks (paragraph-aware, fixed-size)  │
 *   │  2. Extract KG triples for ALL leaves in parallel            │
 *   │  3. Score similarity using triple text (denser signal)       │
 *   │  4. Greedy cluster → one LLM summary per cluster            │
 *   │  5. Extract merged triples for each summary node             │
 *   │  6. Recurse (up to maxLevels)                                │
 *   │  7. Return ALL nodes — leaves + every summary level          │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * Chunk fields used:
 *   text           → original prose (unchanged, shown in UI)
 *   embeddingInput → linearized triples (sent to Voyage/embedding model)
 *   summary        → JSON-stringified KGTriple[] (KG export payload)
 *   rationale      → tree level label (Leaf / L1 Summary / L2 Summary)
 *   children       → indices of child nodes (RAPTOR tree structure)
 *
 * KG triple schema:
 *   {
 *     s:      string   — subject entity name
 *     sType:  string   — entity type (company | person | product | geography |
 *                        financial_metric | regulation | event)
 *     p:      string   — snake_case predicate (e.g. has_revenue, grew_by,
 *                        acquired, is_subsidiary_of, reported_for)
 *     o:      string   — object entity or value
 *     oType:  string   — object type (financial_value | ratio | time_period |
 *                        company | person | boolean | geography)
 *     period?: string  — temporal qualifier (e.g. "Q3 2024", "FY2023")
 *   }
 */

import { createLLMClient } from '../llmClient'
import { estimateTokens } from '../tokenCounter'
import { scoreSimilarityPairs } from '../embeddings'
import type { Chunk, ChunkResult, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

// ─── KG triple type ───────────────────────────────────────────────────────────

export interface KGTriple {
  s: string       // subject entity
  sType: string   // subject type
  p: string       // snake_case predicate
  o: string       // object entity or value
  oType: string   // object type
  period?: string // temporal qualifier (optional)
}

// ─── Base splitting ───────────────────────────────────────────────────────────

function splitToParagraphChunks(
  text: string,
  maxChars: number
): { text: string; start: number; end: number }[] {
  const paragraphs = text.split(/\n\n+/)
  const result: { text: string; start: number; end: number }[] = []
  let buffer = ''
  let bufferStart = 0
  let cursor = 0

  for (const para of paragraphs) {
    const paraStart = text.indexOf(para, cursor)
    if (paraStart === -1) { cursor += para.length; continue }

    if (buffer && buffer.length + para.length + 2 > maxChars) {
      const t = buffer.trim()
      result.push({ text: t, start: bufferStart, end: bufferStart + t.length })
      buffer = para
      bufferStart = paraStart
    } else {
      if (!buffer) bufferStart = paraStart
      buffer = buffer ? buffer + '\n\n' + para : para
    }
    cursor = paraStart + para.length
  }
  if (buffer.trim()) {
    const t = buffer.trim()
    result.push({ text: t, start: bufferStart, end: bufferStart + t.length })
  }
  return result
}

// ─── KG triple extraction ─────────────────────────────────────────────────────

const TRIPLE_SYSTEM = `You are a financial knowledge graph extraction engine.
Extract factual relationships from financial text as compact triples.

Triple schema:
  s     — subject entity name (company, person, product, market, regulation)
  sType — one of: company | person | product | geography | financial_metric | regulation | event
  p     — snake_case predicate. Use ONLY from this list:
            has_revenue | has_net_income | has_ebitda | has_gross_profit | has_operating_income
            has_margin | has_eps | has_pe_ratio | has_debt | has_cash | has_assets | has_liabilities
            grew_by | declined_by | guided_for | beat_estimate | missed_estimate
            acquired | divested | merged_with | is_subsidiary_of | competes_with
            employs | headquartered_in | operates_in | listed_on
            issued | raised_capital | paid_dividend | repurchased_shares
            has_credit_rating | has_outlook | reported_by | covers_period
  o     — the object value or entity name (be specific: "$4.2B", "12.3%", "Apple Inc.")
  oType — one of: financial_value | ratio | time_period | company | person | geography | boolean | rating
  period — (optional) temporal context string, e.g. "Q3 2024", "FY2023", "H1 2025"

Rules:
  - Resolve ALL pronouns to their named referent
  - Include a period whenever a time frame is mentioned
  - Only extract facts explicitly stated — no inferences
  - Skip vague statements without a specific value or named entity
  - Maximum 10 triples per chunk

Return ONLY a valid JSON array with no surrounding text.`

async function extractTriples(
  chunkText: string,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<KGTriple[]> {
  const client = createLLMClient(llmConfig)
  const result = await client.complete(
    TRIPLE_SYSTEM,
    `Extract KG triples from this financial text:\n\n${chunkText.slice(0, 3000)}`,
    512
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens

  const match = result.text.match(/\[[\s\S]*\]/)
  try {
    if (match) return JSON.parse(match[0]) as KGTriple[]
  } catch { /* fallback below */ }
  return []
}

/**
 * Merge triples from multiple child nodes into a deduplicated set.
 * For summary nodes: ask the LLM to aggregate child triples, removing
 * duplicates and re-stating any with updated temporal context.
 */
async function mergeTriples(
  childTripleSets: KGTriple[][],
  summaryText: string,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<KGTriple[]> {
  // Flatten and deduplicate by (s, p, o) key
  const seen = new Map<string, KGTriple>()
  for (const set of childTripleSets) {
    for (const t of set) {
      const key = `${t.s}|${t.p}|${t.o}`
      if (!seen.has(key)) seen.set(key, t)
    }
  }
  const deduped = Array.from(seen.values())

  // If deduped set is small enough, just return it directly
  if (deduped.length <= 12) return deduped

  // Otherwise ask LLM to compress into top-10 most significant triples
  const client = createLLMClient(llmConfig)
  const result = await client.complete(
    TRIPLE_SYSTEM,
    `The following are aggregated KG triples from multiple related chunks.
Select the 10 most significant and non-redundant triples, then add any new
triples from the summary text below that aren't already covered.

Existing triples:
${JSON.stringify(deduped.slice(0, 20), null, 2)}

Summary text for new triples:
${summaryText.slice(0, 1500)}

Return ONLY a valid JSON array of KGTriple objects.`,
    512
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens

  const match = result.text.match(/\[[\s\S]*\]/)
  try {
    if (match) return JSON.parse(match[0]) as KGTriple[]
  } catch { /* fallback */ }
  return deduped.slice(0, 10)
}

// ─── Linearize triples → embeddingInput ──────────────────────────────────────

/**
 * Convert KG triples into a compact, human-readable string for embedding.
 * Format: (Subject [type]) → [predicate] → (Object [type]) | period
 *
 * This is much richer signal for Voyage than raw prose because:
 *   - Named entities are explicit and decontextualized
 *   - Relations are typed (not implicit in word order)
 *   - Every fact is independently retrievable
 */
function linearizeTriples(triples: KGTriple[]): string {
  return triples
    .map(t => {
      const head = `(${t.s} [${t.sType}])`
      const rel  = `[${t.p}]`
      const tail = `(${t.o} [${t.oType}])`
      const time = t.period ? ` | ${t.period}` : ''
      return `${head} → ${rel} → ${tail}${time}`
    })
    .join('\n')
}

// ─── Similarity matrix ────────────────────────────────────────────────────────

async function buildSimilarityMatrix(
  texts: string[],
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<number[][]> {
  const n = texts.length
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let i = 0; i < n; i++) matrix[i][i] = 1
  if (n <= 1) return matrix

  const pairs: [string, string][] = []
  const indices: [number, number][] = []

  for (let i = 0; i < n - 1; i++) {
    pairs.push([texts[i], texts[i + 1]])
    indices.push([i, i + 1])
  }
  for (let i = 0; i < n - 2; i++) {
    pairs.push([texts[i], texts[i + 2]])
    indices.push([i, i + 2])
  }

  const capped = pairs.slice(0, 30)
  const cappedIdx = indices.slice(0, 30)

  if (capped.length > 0) {
    const scores = await scoreSimilarityPairs(capped, llmConfig)
    usage.input += capped.reduce((s, [a, b]) => s + a.length + b.length, 0) / 4
    usage.output += scores.length * 4
    for (let k = 0; k < cappedIdx.length; k++) {
      const [i, j] = cappedIdx[k]
      matrix[i][j] = scores[k]
      matrix[j][i] = scores[k]
    }
  }
  return matrix
}

// ─── Greedy clustering ────────────────────────────────────────────────────────

function greedyCluster(
  n: number,
  matrix: number[][],
  threshold: number,
  maxSize: number
): number[][] {
  const assigned = new Set<number>()
  const clusters: number[][] = []

  for (let i = 0; i < n; i++) {
    if (assigned.has(i)) continue
    const cluster = [i]
    assigned.add(i)
    for (let j = i + 1; j < n && cluster.length < maxSize; j++) {
      if (!assigned.has(j) && matrix[cluster[cluster.length - 1]][j] >= threshold) {
        cluster.push(j)
        assigned.add(j)
      }
    }
    clusters.push(cluster)
  }
  return clusters
}

// ─── Cluster summarisation ────────────────────────────────────────────────────

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
The following ${clusterTexts.length} ${level === 1 ? 'source passages' : 'summaries from the previous abstraction level'} are thematically related.

Write a comprehensive summary that:
- Captures all key financial figures, ratios, and trends
- Names all companies, people, time periods, and metrics explicitly
- Is self-contained and independently retrievable
- Target: 150–250 words

${level === 1 ? 'Source passages' : 'Passages'}:
${combined.slice(0, 6000)}`,
    400
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens
  return result.text.trim()
}

// ─── Tree node ────────────────────────────────────────────────────────────────

interface HybridNode {
  text: string
  triples: KGTriple[]
  level: number
  childIndices: number[]
  start: number
  end: number
}

// ─── Main tree builder ────────────────────────────────────────────────────────

async function buildHybridTree(
  leaves: { text: string; start: number; end: number }[],
  llmConfig: LLMConfig,
  usage: { input: number; output: number },
  clusterThreshold: number,
  maxClusterSize: number,
  maxLevels: number,
  parallelExtractions: number
): Promise<{ nodes: HybridNode[]; apiCalls: number }> {
  let apiCalls = 0
  const allNodes: HybridNode[] = []

  // ── Step 1: Extract triples for all leaves in parallel batches ──
  const leafTriples: KGTriple[][] = new Array(leaves.length).fill([])
  for (let b = 0; b < leaves.length; b += parallelExtractions) {
    const batch = leaves.slice(b, b + parallelExtractions)
    const results = await Promise.all(
      batch.map(leaf => extractTriples(leaf.text, llmConfig, usage))
    )
    results.forEach((t, j) => { leafTriples[b + j] = t })
    apiCalls += batch.length
    if (b + parallelExtractions < leaves.length) {
      await new Promise(r => setTimeout(r, 200))
    }
  }

  // ── Step 2: Build leaf nodes ──
  const leafNodes: HybridNode[] = leaves.map((leaf, i) => ({
    text: leaf.text,
    triples: leafTriples[i],
    level: 0,
    childIndices: [],
    start: leaf.start,
    end: leaf.end,
  }))
  allNodes.push(...leafNodes)

  // ── Step 3: Recursive clustering & summarisation ──
  // Use LINEARIZED TRIPLES for similarity scoring — denser semantic signal
  let currentTexts = leafNodes.map(n =>
    n.triples.length > 0 ? linearizeTriples(n.triples) : n.text
  )
  let currentNodeIndices = leafNodes.map((_, i) => i)
  let level = 1

  while (currentTexts.length > 1 && level <= maxLevels) {
    // Similarity on triple-linearized text
    const matrix = await buildSimilarityMatrix(currentTexts, llmConfig, usage)
    apiCalls++

    const clusters = greedyCluster(currentTexts.length, matrix, clusterThreshold, maxClusterSize)
    if (clusters.length >= currentTexts.length) break  // no merging → stop

    const summaryNodes: HybridNode[] = []

    // Process each cluster: summarise prose + merge triples
    const clusterJobs = clusters.map(async cluster => {
      const clusterTexts = cluster.map(i => {
        const nodeIdx = currentNodeIndices[i]
        return allNodes[nodeIdx].text
      })
      const childAllNodeIndices = cluster.map(i => currentNodeIndices[i])

      // Summarise prose
      const summaryText = await summariseCluster(clusterTexts, level, llmConfig, usage)
      apiCalls++

      // Merge triples from children
      const childTripleSets = childAllNodeIndices.map(idx => allNodes[idx].triples)
      const mergedTriples = await mergeTriples(childTripleSets, summaryText, llmConfig, usage)
      apiCalls++

      return { summaryText, mergedTriples, childAllNodeIndices }
    })

    // Run in batches to respect rate limits
    for (let b = 0; b < clusterJobs.length; b += parallelExtractions) {
      const batchResults = await Promise.all(
        clusterJobs.slice(b, b + parallelExtractions)
      )
      for (const { summaryText, mergedTriples, childAllNodeIndices } of batchResults) {
        const summaryNode: HybridNode = {
          text: summaryText,
          triples: mergedTriples,
          level,
          childIndices: childAllNodeIndices,
          start: allNodes[childAllNodeIndices[0]].start,
          end: allNodes[childAllNodeIndices[childAllNodeIndices.length - 1]].end,
        }
        allNodes.push(summaryNode)
        summaryNodes.push(summaryNode)
      }
    }

    // Next level uses triple-linearized summary text
    currentTexts = summaryNodes.map(n =>
      n.triples.length > 0 ? linearizeTriples(n.triples) : n.text
    )
    currentNodeIndices = summaryNodes.map((_, i) => allNodes.length - summaryNodes.length + i)
    level++

    await new Promise(r => setTimeout(r, 300))
  }

  return { nodes: allNodes, apiCalls }
}

// ─── Chunker definition ───────────────────────────────────────────────────────

const raptorKGChunker: ChunkerDefinition = {
  id: 'raptor_kg',
  label: 'RAPTOR + KG Propositions',
  description:
    'RAPTOR hierarchical tree with KG-ready triple extraction on every node. ' +
    'Leaves and summaries carry (Subject → predicate → Object) triples ' +
    'as embedding input (for precise Voyage retrieval) and JSON metadata ' +
    '(for direct Neo4j / RDF / KG ingestion). ' +
    'Clustering is driven by triple similarity — denser semantic signal than raw prose.',
  paperRef:
    'Sarthi et al., RAPTOR (ICLR 2024) × Chen et al., Dense X Retrieval (2023)',
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
      description: 'Min triple-similarity for two chunks to cluster together (0–1)',
      type: 'slider',
      min: 0.4,
      max: 0.95,
      step: 0.05,
      default: 0.65,
    },
    {
      key: 'maxClusterSize',
      label: 'Max Cluster Size',
      description: 'Maximum nodes per cluster before summarisation',
      type: 'slider',
      min: 2,
      max: 8,
      step: 1,
      default: 4,
    },
    {
      key: 'maxLevels',
      label: 'Max Tree Levels',
      description: 'Recursion depth for RAPTOR tree (2 is usually sufficient)',
      type: 'slider',
      min: 1,
      max: 4,
      step: 1,
      default: 2,
    },
    {
      key: 'parallelExtractions',
      label: 'Parallel Triple Extractions',
      description: 'How many chunks to extract triples from simultaneously',
      type: 'slider',
      min: 1,
      max: 8,
      step: 1,
      default: 4,
    },
  ],
  defaultConfig: {
    maxLeafChars: 1500,
    clusterThreshold: 0.65,
    maxClusterSize: 4,
    maxLevels: 2,
    parallelExtractions: 4,
  },

  async run(text, _structure, config, llmConfig): Promise<ChunkResult> {
    const startTs = Date.now()
    const client = createLLMClient(llmConfig)
    const usage = { input: 0, output: 0 }

    const maxLeafChars       = config.maxLeafChars as number
    const clusterThreshold   = config.clusterThreshold as number
    const maxClusterSize     = config.maxClusterSize as number
    const maxLevels          = config.maxLevels as number
    const parallelExtractions = config.parallelExtractions as number

    // ── Leaf splitting ──
    const leaves = splitToParagraphChunks(text, maxLeafChars)

    if (leaves.length === 0) {
      return {
        strategyId: 'raptor_kg',
        strategyLabel: 'RAPTOR + KG Propositions',
        chunks: [{ index: 0, text, start: 0, end: text.length, tokens: estimateTokens(text) }],
        durationMs: Date.now() - startTs,
        apiCallCount: 0,
        estimatedCost: 0,
      }
    }

    // ── Build hybrid tree ──
    const { nodes, apiCalls } = await buildHybridTree(
      leaves, llmConfig, usage,
      clusterThreshold, maxClusterSize, maxLevels, parallelExtractions
    )

    // ── Convert to Chunk objects ──
    const levelLabels: Record<number, string> = {
      0: 'Leaf',
      1: 'L1 Summary',
      2: 'L2 Summary',
      3: 'L3 Summary',
      4: 'L4 Summary',
    }

    const chunks: Chunk[] = nodes.map((node, i) => {
      const linearized = linearizeTriples(node.triples)
      return {
        index: i,
        text: node.text,
        // embeddingInput: linearized triples — richer signal for Voyage
        embeddingInput: linearized || node.text,
        // summary: JSON triples — ready for KG ingestion
        summary: node.triples.length > 0
          ? JSON.stringify(node.triples, null, 2)
          : undefined,
        start: node.start,
        end: node.end,
        tokens: estimateTokens(node.text),
        rationale: `${levelLabels[node.level] ?? `L${node.level}`}${node.childIndices.length > 0 ? ` · ${node.childIndices.length} child nodes` : ` · ${node.triples.length} triples`}`,
        children: node.level > 0 ? node.childIndices : undefined,
      }
    })

    return {
      strategyId: 'raptor_kg',
      strategyLabel: 'RAPTOR + KG Propositions',
      chunks,
      durationMs: Date.now() - startTs,
      apiCallCount: apiCalls,
      estimatedCost: usage.input * client.costPerInputToken + usage.output * client.costPerOutputToken,
    }
  },
}

export default raptorKGChunker
