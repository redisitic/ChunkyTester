/**
 * Contextual Retrieval Chunker (Anthropic, 2024 — pure implementation)
 *
 * This is the *actual* Contextual Retrieval as described in Anthropic's
 * research, NOT the Sliding Window + Summary already in this app.
 *
 * The key distinction:
 *   - Sliding Window Summary → generates a summary of the local window of text
 *   - Contextual Retrieval  → generates document-level context that situates
 *     each chunk within the WHOLE document: "what section is this from, what
 *     document is this about, what is the broader topic being discussed?"
 *
 * Algorithm:
 *  1. Generate a concise document-level summary (one LLM call for the whole doc)
 *  2. Split text into base chunks using fixed-size splitting at paragraph
 *     boundaries (no LLM, cheap and fast)
 *  3. For each chunk, make one LLM call to generate a short "situating context"
 *     that answers: "What is this chunk about in the context of the full document?"
 *  4. Prepend the context to the chunk text as the embeddingInput:
 *       "[context]\n\n[original chunk text]"
 *     The STORED chunk text is unchanged — only the embedding input is enriched.
 *
 * This approach is Anthropic's recommended pattern for improving retrieval
 * precision in RAG systems. The document summary step amortises over all chunks.
 *
 * Cost: 1 doc-summary call + N per-chunk context calls (parallelised).
 * For a finance report with ~20 chunks: ~21 LLM calls total.
 *
 * Reference: https://www.anthropic.com/news/contextual-retrieval
 */

import { createLLMClient } from '../llmClient'
import { estimateTokens } from '../tokenCounter'
import type { Chunk, ChunkResult, LLMConfig } from '@/types'
import type { ChunkerDefinition } from './types'

// ---- Base splitting (paragraph-aware fixed-size) ----

function splitToParagraphChunks(text: string, maxChars: number): { text: string; start: number }[] {
  const paragraphs = text.split(/\n\n+/)
  const chunks: { text: string; start: number }[] = []
  let buffer = ''
  let bufferStart = 0
  let cursor = 0

  for (const para of paragraphs) {
    const paraLen = para.length
    const gap = text.indexOf(para, cursor) - cursor
    const paraStart = cursor + gap

    if (buffer && buffer.length + paraLen + 2 > maxChars) {
      chunks.push({ text: buffer.trim(), start: bufferStart })
      buffer = para
      bufferStart = paraStart
    } else {
      if (!buffer) bufferStart = paraStart
      buffer = buffer ? buffer + '\n\n' + para : para
    }
    cursor = paraStart + paraLen
  }
  if (buffer.trim()) chunks.push({ text: buffer.trim(), start: bufferStart })
  return chunks
}

// ---- Context generation ----

async function generateDocSummary(
  text: string,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<string> {
  const client = createLLMClient(llmConfig)
  // Use first 8000 chars for summary to stay within token limits
  const excerpt = text.length > 8000 ? text.slice(0, 8000) + '\n[document continues…]' : text
  const result = await client.complete(
    null,
    `You are preparing to index this financial document for retrieval. Write a 3–5 sentence summary that describes:
- What type of document this is (e.g. annual report, earnings release, 10-K, research note)
- The company or companies involved
- The time period covered
- The main financial topics discussed

Return only the summary, no preamble.

Document:
${excerpt}`,
    256
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens
  return result.text.trim()
}

async function generateChunkContext(
  docSummary: string,
  chunkText: string,
  llmConfig: LLMConfig,
  usage: { input: number; output: number }
): Promise<string> {
  const client = createLLMClient(llmConfig)
  const result = await client.complete(
    null,
    `Document summary:
${docSummary}

Given the above document summary, write a short context (2–3 sentences maximum) that situates the following chunk within the document. Explain what section it comes from, what financial topic it covers, and what makes it relevant for retrieval. Do not repeat the chunk content verbatim — describe its position and significance.

Return only the context, no preamble.

Chunk:
${chunkText.slice(0, 1200)}`,
    150
  )
  usage.input += result.inputTokens
  usage.output += result.outputTokens
  return result.text.trim()
}

// ---- Chunker definition ----

const contextualRetrievalChunker: ChunkerDefinition = {
  id: 'contextual_retrieval',
  label: 'Contextual Retrieval',
  description:
    'Anthropic\'s Contextual Retrieval (2024): generates document-level context for each chunk ' +
    'and prepends it to the embedding input. The stored chunk text is unchanged — only retrieval ' +
    'is enriched. One doc-summary call + one context call per chunk.',
  paperRef: 'Anthropic, "Contextual Retrieval", Sept 2024 — anthropic.com/news/contextual-retrieval',
  configSchema: [
    {
      key: 'maxChunkChars',
      label: 'Max Chunk Size (chars)',
      description: 'Maximum characters per base chunk before context is added',
      type: 'slider',
      min: 500,
      max: 5000,
      step: 100,
      default: 3500,
    },
    {
      key: 'contextMaxSentences',
      label: 'Context Length (sentences)',
      description: 'Target sentence count for the situating context prefix',
      type: 'slider',
      min: 1,
      max: 5,
      step: 1,
      default: 2,
    },
    {
      key: 'parallelContextCalls',
      label: 'Parallel Context Calls',
      description: 'How many per-chunk context calls to run in parallel (higher = faster, more rate-limit risk)',
      type: 'slider',
      min: 1,
      max: 10,
      step: 1,
      default: 4,
    },
  ],
  defaultConfig: { maxChunkChars: 3500, contextMaxSentences: 2, parallelContextCalls: 4 },

  async run(text, _structure, config, llmConfig): Promise<ChunkResult> {
    const startTs = Date.now()
    const client = createLLMClient(llmConfig)
    const usage = { input: 0, output: 0 }
    let apiCallCount = 0

    const maxChunkChars      = config.maxChunkChars as number
    const parallelCalls      = config.parallelContextCalls as number

    // ---- Step 1: Base split ----
    const baseChunks = splitToParagraphChunks(text, maxChunkChars)

    if (baseChunks.length === 0) {
      return {
        strategyId: 'contextual_retrieval',
        strategyLabel: 'Contextual Retrieval',
        chunks: [{ index: 0, text, start: 0, end: text.length, tokens: estimateTokens(text) }],
        durationMs: Date.now() - startTs,
        apiCallCount: 0,
        estimatedCost: 0,
      }
    }

    // ---- Step 2: Document-level summary ----
    const docSummary = await generateDocSummary(text, llmConfig, usage)
    apiCallCount++

    // ---- Step 3: Per-chunk context (parallelised in batches) ----
    const contexts: string[] = new Array(baseChunks.length).fill('')

    for (let batchStart = 0; batchStart < baseChunks.length; batchStart += parallelCalls) {
      const batch = baseChunks.slice(batchStart, batchStart + parallelCalls)
      const batchContexts = await Promise.all(
        batch.map(chunk => generateChunkContext(docSummary, chunk.text, llmConfig, usage))
      )
      batchContexts.forEach((ctx, j) => { contexts[batchStart + j] = ctx })
      apiCallCount += batch.length
      // Small rate-limit buffer between batches
      if (batchStart + parallelCalls < baseChunks.length) {
        await new Promise(r => setTimeout(r, 300))
      }
    }

    // ---- Step 4: Assemble chunks ----
    const chunks: Chunk[] = baseChunks.map((base, i) => {
      const context = contexts[i]
      const embeddingInput = context
        ? `${context}\n\n${base.text}`
        : base.text

      return {
        index: i,
        text: base.text,                 // stored text — unchanged
        embeddingInput,                  // enriched embedding input
        start: base.start,
        end: base.start + base.text.length,
        tokens: estimateTokens(base.text),
        summary: context,                // exposed in ChunkViewer as "context"
      }
    })

    return {
      strategyId: 'contextual_retrieval',
      strategyLabel: 'Contextual Retrieval',
      chunks,
      durationMs: Date.now() - startTs,
      apiCallCount,
      estimatedCost: usage.input * client.costPerInputToken + usage.output * client.costPerOutputToken,
    }
  },
}

export default contextualRetrievalChunker
