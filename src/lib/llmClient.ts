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

// --- OpenAI ---

const OPENAI_COSTS: Record<string, { input: number; output: number }> = {
  'gpt-5.5':      { input: 5.00 / 1_000_000, output: 30.00 / 1_000_000 },
  'gpt-5.4':      { input: 2.50 / 1_000_000, output: 15.00 / 1_000_000 },
  'gpt-5.4-mini': { input: 0.75 / 1_000_000, output: 4.50  / 1_000_000 },
  'gpt-5.4-nano': { input: 0.20 / 1_000_000, output: 1.25  / 1_000_000 },
}

interface OpenAIResponseContent {
  type?: string
  text?: string
}

interface OpenAIResponseOutput {
  type?: string
  content?: OpenAIResponseContent[]
}

interface OpenAIUsage {
  input_tokens?: number
  output_tokens?: number
}

interface OpenAIResponse {
  output_text?: string
  output?: OpenAIResponseOutput[]
  usage?: OpenAIUsage
}

function getOpenAIText(data: OpenAIResponse): string {
  if (data.output_text) return data.output_text
  return data.output
    ?.flatMap(item => item.content ?? [])
    .filter(content => content.type === 'output_text' || content.text)
    .map(content => content.text ?? '')
    .join('') ?? ''
}

function createOpenAIClient(config: LLMConfig): LLMClient {
  const model = config.openaiModel ?? 'gpt-5.4-mini'
  const costs = OPENAI_COSTS[model] ?? OPENAI_COSTS['gpt-5.4-mini']
  const base = 'https://api.openai.com/v1/responses'

  function buildBody(system: string | null, user: string, maxTokens: number, stream: boolean) {
    return {
      model,
      input: [{ role: 'user', content: user }],
      max_output_tokens: maxTokens,
      ...(system ? { instructions: system } : {}),
      ...(stream ? { stream: true } : {}),
    }
  }

  function headers() {
    return {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${config.openaiKey}`,
    }
  }

  return {
    costPerInputToken: costs.input,
    costPerOutputToken: costs.output,
    label: `OpenAI (${model})`,

    async complete(system, user, maxTokens) {
      const res = await fetch(base, {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify(buildBody(system, user, maxTokens, false)),
      })
      if (!res.ok) throw new Error(`OpenAI error ${res.status}: ${await res.text()}`)
      const data = await res.json() as OpenAIResponse
      return {
        text: getOpenAIText(data),
        inputTokens: data.usage?.input_tokens ?? 0,
        outputTokens: data.usage?.output_tokens ?? 0,
      }
    },

    async streamComplete(system, user, maxTokens, onChunk) {
      const res = await fetch(base, {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify(buildBody(system, user, maxTokens, true)),
      })
      if (!res.ok) throw new Error(`OpenAI stream error ${res.status}: ${await res.text()}`)

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let accumulated = ''
      let inputTokens = 0
      let outputTokens = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() ?? ''

        for (const event of events) {
          const dataLine = event
            .split('\n')
            .find(line => line.startsWith('data: '))
          if (!dataLine) continue

          const json = dataLine.slice(6)
          if (json === '[DONE]') continue

          try {
            const data = JSON.parse(json)
            if (data.type === 'response.output_text.delta' && data.delta) {
              accumulated += data.delta
              onChunk?.(data.delta)
            }
            const usage = data.response?.usage ?? data.usage
            if (usage) {
              inputTokens = usage.input_tokens ?? inputTokens
              outputTokens = usage.output_tokens ?? outputTokens
            }
          } catch { /* skip malformed */ }
        }
      }

      return { text: accumulated, inputTokens, outputTokens }
    },
  }
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
    case 'openai': return createOpenAIClient(config)
    case 'gemini': return createGeminiClient(config)
    case 'ollama': return createOllamaClient(config)
    default:       return createAnthropicClient(config)
  }
}
