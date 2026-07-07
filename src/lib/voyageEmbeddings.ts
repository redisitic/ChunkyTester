/**
 * Voyage AI embedding client
 * Calls https://api.voyageai.com/v1/embeddings directly from the browser.
 * CORS is supported by Voyage AI for browser requests.
 *
 * Env alignment:
 *   VOYAGE_MODEL=voyage-finance-2
 *   VOYAGE_BATCH_SIZE=64
 *   EMBEDDING_DIMENSIONS=1024
 */

const VOYAGE_API_URL = 'https://api.voyageai.com/v1/embeddings'
const VOYAGE_MODEL = 'voyage-finance-2'
const BATCH_SIZE = 64          // matches VOYAGE_BATCH_SIZE=64
const EXPECTED_DIMS = 1024     // matches EMBEDDING_DIMENSIONS=1024

// --- In-memory embedding cache keyed by (text, inputType) ---
const embeddingCache = new Map<string, number[]>()

function cacheKey(text: string, inputType: 'document' | 'query'): string {
  return `${inputType}::${text}`
}

async function embedBatch(
  texts: string[],
  voyageKey: string,
  inputType: 'document' | 'query'
): Promise<number[][]> {
  const res = await fetch(VOYAGE_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${voyageKey}`,
    },
    body: JSON.stringify({
      input: texts,
      model: VOYAGE_MODEL,
      input_type: inputType,
    }),
  })

  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Voyage AI error ${res.status}: ${body}`)
  }

  const data = await res.json()
  // Voyage returns data.data[] sorted by index
  const sorted = (data.data as { index: number; embedding: number[] }[])
    .sort((a, b) => a.index - b.index)

  return sorted.map(item => {
    if (item.embedding.length !== EXPECTED_DIMS) {
      console.warn(`Voyage returned ${item.embedding.length} dims, expected ${EXPECTED_DIMS}`)
    }
    return item.embedding
  })
}

/**
 * Embed an array of texts using voyage-finance-2.
 * Results are cached in memory — repeated calls with the same text are free.
 *
 * @param texts       Array of strings to embed
 * @param voyageKey   Voyage AI API key
 * @param inputType   'document' for corpus chunks, 'query' for search queries
 */
export async function embedTextsVoyage(
  texts: string[],
  voyageKey: string,
  inputType: 'document' | 'query' = 'document'
): Promise<number[][]> {
  if (!voyageKey) throw new Error('Voyage API key is required. Set it in Settings → Voyage AI Key.')

  const results: number[][] = new Array(texts.length)
  const toFetch: { originalIndex: number; text: string }[] = []

  // Check cache first
  for (let i = 0; i < texts.length; i++) {
    const key = cacheKey(texts[i], inputType)
    const cached = embeddingCache.get(key)
    if (cached) {
      results[i] = cached
    } else {
      toFetch.push({ originalIndex: i, text: texts[i] })
    }
  }

  // Fetch in batches
  for (let start = 0; start < toFetch.length; start += BATCH_SIZE) {
    const batch = toFetch.slice(start, start + BATCH_SIZE)
    const embeddings = await embedBatch(batch.map(b => b.text), voyageKey, inputType)
    for (let j = 0; j < batch.length; j++) {
      const { originalIndex, text } = batch[j]
      embeddingCache.set(cacheKey(text, inputType), embeddings[j])
      results[originalIndex] = embeddings[j]
    }
  }

  return results
}

/** Clear the in-memory cache (e.g. when the API key changes) */
export function clearVoyageCache(): void {
  embeddingCache.clear()
}

/** Cost estimate: $0.00012 per 1K tokens (rough chars/4 approximation) */
export function estimateVoyageCost(texts: string[]): number {
  const totalChars = texts.reduce((s, t) => s + t.length, 0)
  const estimatedTokens = totalChars / 4
  return (estimatedTokens / 1000) * 0.00012
}
