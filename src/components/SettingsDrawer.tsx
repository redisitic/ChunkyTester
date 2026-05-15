import { useState } from 'react'
import { Settings, Eye, EyeOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { LLMProvider } from '@/types'

const STORAGE = {
  provider:      'rag_portal_provider',
  anthropicKey:  'rag_portal_apikey',
  geminiKey:     'rag_portal_gemini_key',
  geminiModel:   'rag_portal_gemini_model',
  ollamaUrl:     'rag_portal_ollama_url',
  ollamaModel:   'rag_portal_ollama_model',
  embeddingMode: 'rag_portal_embedding_mode',
}

const GEMINI_MODELS = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']

export type EmbeddingMode = 'local' | 'llm'

export interface AppSettings {
  provider: LLMProvider
  anthropicKey: string
  geminiKey: string
  geminiModel: string
  ollamaBaseUrl: string
  ollamaModel: string
  evalSampleSize: number
  topK: number
  theme: 'light' | 'dark'
  embeddingMode: EmbeddingMode
}

interface Props {
  settings: AppSettings
  onSettingsChange: (patch: Partial<AppSettings>) => void
}

function KeyInput({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder: string }) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative">
      <input
        type={show ? 'text' : 'password'}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full h-9 rounded-md border border-input bg-background px-3 pr-9 text-sm font-mono outline-none focus:ring-2 focus:ring-ring"
      />
      <button type="button" className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" onClick={() => setShow(v => !v)}>
        {show ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
      </button>
    </div>
  )
}

export function SettingsDrawer({ settings, onSettingsChange }: Props) {
  function persist(key: keyof typeof STORAGE, value: string) {
    localStorage.setItem(STORAGE[key], value)
  }

  function handleProvider(v: LLMProvider) {
    persist('provider', v)
    onSettingsChange({ provider: v })
  }

  return (
    <Sheet>
      <SheetTrigger render={<Button variant="ghost" size="icon" className="h-8 w-8" />}>
        <Settings className="w-4 h-4" />
      </SheetTrigger>
      <SheetContent side="right" className="w-80 flex flex-col gap-5 overflow-y-auto px-4">
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
        </SheetHeader>

        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium">LLM Provider</label>
          <Select value={settings.provider} onValueChange={v => v && handleProvider(v as LLMProvider)}>
            <SelectTrigger className="h-8 text-sm w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="anthropic">Anthropic (Claude)</SelectItem>
              <SelectItem value="gemini">Google Gemini</SelectItem>
              <SelectItem value="ollama">Ollama (local)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {settings.provider === 'anthropic' && (
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">Anthropic API Key</label>
            <KeyInput
              value={settings.anthropicKey}
              placeholder="sk-ant-…"
              onChange={v => { persist('anthropicKey', v); onSettingsChange({ anthropicKey: v }) }}
            />
            <p className="text-xs text-muted-foreground font-mono">claude-sonnet-4-20250514 · $3/$15 per M tokens</p>
            <Alert className="py-2">
              <AlertDescription className="text-xs">Stored in browser only. Never sent to any server other than api.anthropic.com.</AlertDescription>
            </Alert>
          </div>
        )}

        {settings.provider === 'gemini' && (
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Gemini API Key</label>
              <KeyInput
                value={settings.geminiKey}
                placeholder="AIza…"
                onChange={v => { persist('geminiKey', v); onSettingsChange({ geminiKey: v }) }}
              />
              <Alert className="py-2">
                <AlertDescription className="text-xs">Stored in browser only. Never sent to any server other than generativelanguage.googleapis.com.</AlertDescription>
              </Alert>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Model</label>
              <Select value={settings.geminiModel} onValueChange={v => v && (persist('geminiModel', v), onSettingsChange({ geminiModel: v }))}>
                <SelectTrigger className="h-8 text-sm w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {GEMINI_MODELS.map(m => <SelectItem key={m} value={m}>{m}</SelectItem>)}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {settings.geminiModel === 'gemini-2.0-flash' && '$0.10/$0.40 per M tokens'}
                {settings.geminiModel === 'gemini-1.5-flash' && '$0.075/$0.30 per M tokens'}
                {settings.geminiModel === 'gemini-1.5-pro'   && '$1.25/$5.00 per M tokens'}
              </p>
            </div>
          </div>
        )}

        {settings.provider === 'ollama' && (
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Ollama Base URL</label>
              <input
                type="text"
                value={settings.ollamaBaseUrl}
                onChange={e => { persist('ollamaUrl', e.target.value); onSettingsChange({ ollamaBaseUrl: e.target.value }) }}
                placeholder="http://localhost:11434"
                className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm font-mono outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Model</label>
              <input
                type="text"
                value={settings.ollamaModel}
                onChange={e => { persist('ollamaModel', e.target.value); onSettingsChange({ ollamaModel: e.target.value }) }}
                placeholder="llama3.2"
                className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm font-mono outline-none focus:ring-2 focus:ring-ring"
              />
              <p className="text-xs text-muted-foreground">Free (runs locally) · Ensure Ollama is running and CORS is enabled.</p>
            </div>
          </div>
        )}

        <Separator />

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Eval Sample Size</label>
            <span className="text-sm font-mono text-muted-foreground">{settings.evalSampleSize}</span>
          </div>
          <Slider min={3} max={20} step={1} value={[settings.evalSampleSize]} onValueChange={v => onSettingsChange({ evalSampleSize: Array.isArray(v) ? v[0] : v as number })} />
          <p className="text-xs text-muted-foreground">Chunks sampled per strategy for LLM eval</p>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Retrieval Top-K</label>
            <span className="text-sm font-mono text-muted-foreground">{settings.topK}</span>
          </div>
          <Slider min={1} max={10} step={1} value={[settings.topK]} onValueChange={v => onSettingsChange({ topK: Array.isArray(v) ? v[0] : v as number })} />
        </div>

        <Separator />

        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium">Retrieval Embeddings</label>
          <Select
            value={settings.embeddingMode}
            onValueChange={v => {
              if (v) {
                localStorage.setItem(STORAGE.embeddingMode, v)
                onSettingsChange({ embeddingMode: v as EmbeddingMode })
              }
            }}
          >
            <SelectTrigger className="h-8 text-sm w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="local">Local (transformers.js · free · in-browser)</SelectItem>
              <SelectItem value="llm">LLM ranking (uses API credits)</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {settings.embeddingMode === 'local'
              ? 'Xenova/all-MiniLM-L6-v2 · ~23 MB · downloaded once, cached by browser'
              : 'LLM scores chunks by relevance via prompt — no local model required'}
          </p>
        </div>

        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Theme</label>
          <Button variant="outline" size="sm" onClick={() => onSettingsChange({ theme: settings.theme === 'light' ? 'dark' : 'light' })}>
            {settings.theme === 'light' ? 'Switch to Dark' : 'Switch to Light'}
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  )
}

export function loadSettings(): AppSettings {
  return {
    provider:      (localStorage.getItem(STORAGE.provider) as LLMProvider) ?? 'anthropic',
    anthropicKey:  localStorage.getItem(STORAGE.anthropicKey) ?? '',
    geminiKey:     localStorage.getItem(STORAGE.geminiKey) ?? '',
    geminiModel:   localStorage.getItem(STORAGE.geminiModel) ?? 'gemini-2.0-flash',
    ollamaBaseUrl: localStorage.getItem(STORAGE.ollamaUrl) ?? 'http://localhost:11434',
    ollamaModel:   localStorage.getItem(STORAGE.ollamaModel) ?? 'llama3.2',
    evalSampleSize: 10,
    topK: 5,
    theme: 'light',
    embeddingMode: (localStorage.getItem(STORAGE.embeddingMode) as EmbeddingMode) ?? 'local',
  }
}

export const API_KEY_STORAGE = STORAGE.anthropicKey
