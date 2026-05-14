import Anthropic from '@anthropic-ai/sdk'
import type { LLMConfig } from '@/types'

export interface CompletionResult {
  text: string
  inputTokens: number
  outputTokens: number
}

export interface LLMClient {
  complete(system: string | null, user: string, maxTokens: number): Promise<CompletionResult>
  streamComplete(system: string | null, user: string, maxTokens: number, onChunk?: (chunk: string) => void): Promise<CompletionResult>
  readonly costPerInputToken: number
  readonly costPerOutputToken: number
  readonly label: string
}

// --- Anthropic ---

function createAnthropicClient(config: LLMConfig): LLMClient {
  const client = new Anthropic({ apiKey: config.anthropicKey, dangerouslyAllowBrowser: true })
  const model = 'claude-sonnet-4-20250514'

  return {
    costPerInputToken: 3 / 1_000_000,
    costPerOutputToken: 15 / 1_000_000,
    label: `Claude (${model})`,

    async complete(system, user, maxTokens) {
      const response = await client.messages.create({
        model,
        max_tokens: maxTokens,
        ...(system ? { system } : {}),
        messages: [{ role: 'user', content: user }],
      })
      return {
        text: response.content[0].type === 'text' ? response.content[0].text : '',
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      }
    },

    async streamComplete(system, user, maxTokens, onChunk) {
      let text = ''
      let inputTokens = 0
      let outputTokens = 0

      const stream = await client.messages.create({
        model,
        max_tokens: maxTokens,
        ...(system ? { system } : {}),
        messages: [{ role: 'user', content: user }],
        stream: true,
      })

      for await (const event of stream) {
        if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
          text += event.delta.text
          onChunk?.(event.delta.text)
        }
        if (event.type === 'message_start') inputTokens = event.message.usage?.input_tokens ?? 0
        if (event.type === 'message_delta') outputTokens = event.usage?.output_tokens ?? 0
      }

      return { text, inputTokens, outputTokens }
    },
  }
}

// --- Gemini ---

const GEMINI_COSTS: Record<string, { input: number; output: number }> = {
  'gemini-2.0-flash':   { input: 0.10  / 1_000_000, output: 0.40  / 1_000_000 },
  'gemini-1.5-flash':   { input: 0.075 / 1_000_000, output: 0.30  / 1_000_000 },
  'gemini-1.5-pro':     { input: 1.25  / 1_000_000, output: 5.00  / 1_000_000 },
}

function createGeminiClient(config: LLMConfig): LLMClient {
  const model = config.geminiModel ?? 'gemini-2.0-flash'
  const costs = GEMINI_COSTS[model] ?? GEMINI_COSTS['gemini-2.0-flash']
  const base = `https://generativelanguage.googleapis.com/v1beta/models/${model}`

  function buildBody(system: string | null, user: string, maxTokens: number) {
    const body: Record<string, unknown> = {
      contents: [{ role: 'user', parts: [{ text: user }] }],
      generationConfig: { maxOutputTokens: maxTokens },
    }
    if (system) body.systemInstruction = { parts: [{ text: system }] }
    return body
  }

  return {
    costPerInputToken: costs.input,
    costPerOutputToken: costs.output,
    label: `Gemini (${model})`,

    async complete(system, user, maxTokens) {
      const res = await fetch(`${base}:generateContent?key=${config.geminiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildBody(system, user, maxTokens)),
      })
      if (!res.ok) throw new Error(`Gemini error ${res.status}: ${await res.text()}`)
      const data = await res.json()
      return {
        text: data.candidates?.[0]?.content?.parts?.[0]?.text ?? '',
        inputTokens: data.usageMetadata?.promptTokenCount ?? 0,
        outputTokens: data.usageMetadata?.candidatesTokenCount ?? 0,
      }
    },

    async streamComplete(system, user, maxTokens, onChunk) {
      const res = await fetch(`${base}:streamGenerateContent?key=${config.geminiKey}&alt=sse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildBody(system, user, maxTokens)),
      })
      if (!res.ok) throw new Error(`Gemini stream error ${res.status}: ${await res.text()}`)

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''
      let inputTokens = 0
      let outputTokens = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        for (const line of decoder.decode(value).split('\n')) {
          if (!line.startsWith('data: ')) continue
          const json = line.slice(6)
          if (json === '[DONE]') continue
          try {
            const data = JSON.parse(json)
            const chunk = data.candidates?.[0]?.content?.parts?.[0]?.text ?? ''
            if (chunk) { accumulated += chunk; onChunk?.(chunk) }
            if (data.usageMetadata) {
              inputTokens = data.usageMetadata.promptTokenCount ?? 0
              outputTokens = data.usageMetadata.candidatesTokenCount ?? 0
            }
          } catch { /* skip malformed */ }
        }
      }

      return { text: accumulated, inputTokens, outputTokens }
    },
  }
}

// --- Ollama ---

function isQwen3Model(model: string): boolean {
  return /qwen3/i.test(model)
}

function createOllamaClient(config: LLMConfig): LLMClient {
  const baseUrl = (config.ollamaBaseUrl ?? 'http://localhost:11434').replace(/\/$/, '')
  const model = config.ollamaModel ?? 'llama3.2'
  const noThink = isQwen3Model(model)

  function buildMessages(system: string | null, user: string) {
    const msgs: { role: string; content: string }[] = []
    if (system) msgs.push({ role: 'system', content: system })
    msgs.push({ role: 'user', content: user })
    return msgs
  }

  function buildBody(system: string | null, user: string, maxTokens: number, stream: boolean) {
    const body: Record<string, unknown> = {
      model,
      messages: buildMessages(system, user),
      stream,
      options: { num_predict: maxTokens },
    }
    if (noThink) body.think = false
    return body
  }

  return {
    costPerInputToken: 0,
    costPerOutputToken: 0,
    label: `Ollama (${model})`,

    async complete(system, user, maxTokens) {
      const res = await fetch(`${baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildBody(system, user, maxTokens, false)),
      })
      if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`)
      const data = await res.json()
      return {
        text: data.message?.content ?? '',
        inputTokens: data.prompt_eval_count ?? 0,
        outputTokens: data.eval_count ?? 0,
      }
    },

    async streamComplete(system, user, maxTokens, onChunk) {
      const res = await fetch(`${baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildBody(system, user, maxTokens, true)),
      })
      if (!res.ok) throw new Error(`Ollama stream error ${res.status}: ${await res.text()}`)

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''
      let inputTokens = 0
      let outputTokens = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        for (const line of decoder.decode(value).split('\n').filter(l => l.trim())) {
          try {
            const data = JSON.parse(line)
            const chunk = data.message?.content ?? ''
            if (chunk) { accumulated += chunk; onChunk?.(chunk) }
            if (data.done) {
              inputTokens = data.prompt_eval_count ?? 0
              outputTokens = data.eval_count ?? 0
            }
          } catch { /* skip */ }
        }
      }

      return { text: accumulated, inputTokens, outputTokens }
    },
  }
}

// --- Factory ---

export function createLLMClient(config: LLMConfig): LLMClient {
  switch (config.provider) {
    case 'gemini': return createGeminiClient(config)
    case 'ollama': return createOllamaClient(config)
    default:       return createAnthropicClient(config)
  }
}
