import { pipeline, env } from '@huggingface/transformers'

env.allowLocalModels = false

export type EmbeddingStatus = 'idle' | 'loading' | 'ready' | 'error'

let pipelineInstance: Awaited<ReturnType<typeof pipeline>> | null = null
let currentStatus: EmbeddingStatus = 'idle'
let loadProgress = 0
let loadPromise: Promise<Awaited<ReturnType<typeof pipeline>>> | null = null
const statusListeners = new Set<(status: EmbeddingStatus, progress: number) => void>()

function broadcast(status: EmbeddingStatus, progress = 0) {
  currentStatus = status
  loadProgress = progress
  for (const fn of statusListeners) fn(status, progress)
}

export function getEmbeddingStatus(): { status: EmbeddingStatus; progress: number } {
  return { status: currentStatus, progress: loadProgress }
}

export function subscribeEmbeddingStatus(
  fn: (status: EmbeddingStatus, progress: number) => void
): () => void {
  statusListeners.add(fn)
  fn(currentStatus, loadProgress)
  return () => { statusListeners.delete(fn) }
}

const MODEL = 'Xenova/all-MiniLM-L6-v2'

async function loadPipeline() {
  broadcast('loading', 0)
  try {
    pipelineInstance = await pipeline('feature-extraction', MODEL, {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      progress_callback: (info: any) => {
        if (typeof info.progress === 'number') {
          broadcast('loading', info.progress)
        }
      },
    })
    broadcast('ready', 100)
    return pipelineInstance
  } catch (err) {
    broadcast('error', 0)
    throw err
  }
}

async function ensurePipeline() {
  if (pipelineInstance) return pipelineInstance
  if (!loadPromise) {
    loadPromise = loadPipeline().catch(err => {
      loadPromise = null
      throw err
    })
  }
  return loadPromise
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  const extractor = await ensurePipeline()
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const output = await (extractor as any)(texts, { pooling: 'mean', normalize: true })
  return output.tolist() as number[][]
}
