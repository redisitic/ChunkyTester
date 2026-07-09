/**
 * Retrieval benchmark metrics
 *
 * Computes per-query and aggregate metrics over a set of ranked results
 * against gold-label relevant chunk indices.
 *
 * Metrics:
 *   - Precision@K   — fraction of top-K results that are relevant
 *   - Recall@K      — fraction of all relevant items found in top-K
 *   - MRR           — Mean Reciprocal Rank (position of first relevant hit)
 *   - NDCG@K        — Normalized Discounted Cumulative Gain
 */

export interface RankedResult {
  /** chunk index within the strategy's chunk array */
  chunkIndex: number
  score: number
}

export interface QueryMetrics {
  query: string
  precisionAt1: number
  precisionAt3: number
  precisionAt5: number
  recallAt5: number
  recallAt10: number
  mrr: number
  ndcgAt5: number
  numRelevant: number
}

export interface StrategyBenchmarkResult {
  strategyId: string
  strategyLabel: string
  perQueryMetrics: QueryMetrics[]
  // Macro-averaged across all queries
  avgPrecisionAt1: number
  avgPrecisionAt3: number
  avgPrecisionAt5: number
  avgRecallAt5: number
  avgRecallAt10: number
  meanReciprocalRank: number
  avgNdcgAt5: number
  /** Composite score for ranking (equal weight) */
  composite: number
}

// ---- Core metric functions ----

function precisionAtK(ranked: RankedResult[], relevantSet: Set<number>, k: number): number {
  const topK = ranked.slice(0, k)
  if (topK.length === 0) return 0
  const hits = topK.filter(r => relevantSet.has(r.chunkIndex)).length
  return hits / k
}

function recallAtK(ranked: RankedResult[], relevantSet: Set<number>, k: number): number {
  if (relevantSet.size === 0) return 0
  const topK = ranked.slice(0, k)
  const hits = topK.filter(r => relevantSet.has(r.chunkIndex)).length
  return hits / relevantSet.size
}

function reciprocalRank(ranked: RankedResult[], relevantSet: Set<number>): number {
  for (let i = 0; i < ranked.length; i++) {
    if (relevantSet.has(ranked[i].chunkIndex)) return 1 / (i + 1)
  }
  return 0
}

function dcg(ranked: RankedResult[], relevantSet: Set<number>, k: number): number {
  let score = 0
  const topK = ranked.slice(0, k)
  for (let i = 0; i < topK.length; i++) {
    const rel = relevantSet.has(topK[i].chunkIndex) ? 1 : 0
    score += rel / Math.log2(i + 2) // i+2 because log2(1) = 0
  }
  return score
}

function ndcgAtK(ranked: RankedResult[], relevantSet: Set<number>, k: number): number {
  const actualDcg = dcg(ranked, relevantSet, k)
  // Ideal DCG: all relevant items ranked first
  const idealRanked: RankedResult[] = Array.from(relevantSet).map((idx, i) => ({
    chunkIndex: idx,
    score: relevantSet.size - i,
  }))
  const idealDcg = dcg(idealRanked, relevantSet, k)
  return idealDcg === 0 ? 0 : actualDcg / idealDcg
}

// ---- Per-query computation ----

export function computeQueryMetrics(
  query: string,
  ranked: RankedResult[],
  goldLabelIndices: number[]
): QueryMetrics {
  const relevantSet = new Set(goldLabelIndices)
  return {
    query,
    precisionAt1: precisionAtK(ranked, relevantSet, 1),
    precisionAt3: precisionAtK(ranked, relevantSet, 3),
    precisionAt5: precisionAtK(ranked, relevantSet, 5),
    recallAt5: recallAtK(ranked, relevantSet, 5),
    recallAt10: recallAtK(ranked, relevantSet, 10),
    mrr: reciprocalRank(ranked, relevantSet),
    ndcgAt5: ndcgAtK(ranked, relevantSet, 5),
    numRelevant: goldLabelIndices.length,
  }
}

// ---- Aggregate across queries ----

function avg(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((s, v) => s + v, 0) / values.length
}

export function aggregateBenchmarkMetrics(
  strategyId: string,
  strategyLabel: string,
  perQueryMetrics: QueryMetrics[]
): StrategyBenchmarkResult {
  const avgP1  = avg(perQueryMetrics.map(q => q.precisionAt1))
  const avgP3  = avg(perQueryMetrics.map(q => q.precisionAt3))
  const avgP5  = avg(perQueryMetrics.map(q => q.precisionAt5))
  const avgR5  = avg(perQueryMetrics.map(q => q.recallAt5))
  const avgR10 = avg(perQueryMetrics.map(q => q.recallAt10))
  const mrr    = avg(perQueryMetrics.map(q => q.mrr))
  const ndcg5  = avg(perQueryMetrics.map(q => q.ndcgAt5))

  const composite = avg([avgP5, avgR5, mrr, ndcg5])

  return {
    strategyId,
    strategyLabel,
    perQueryMetrics,
    avgPrecisionAt1: avgP1,
    avgPrecisionAt3: avgP3,
    avgPrecisionAt5: avgP5,
    avgRecallAt5: avgR5,
    avgRecallAt10: avgR10,
    meanReciprocalRank: mrr,
    avgNdcgAt5: ndcg5,
    composite,
  }
}
