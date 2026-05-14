# ChunkyTester

A browser-based evaluation tool for comparing RAG chunking strategies before committing to one in production. Upload a document, select a passage, run multiple strategies in parallel, and measure which produces the best chunks for retrieval.

No server required. All processing runs in the browser. API calls go directly from your browser to the LLM provider.

---

## What it does

Chunking is one of the most consequential decisions in a RAG pipeline and one of the least tested. Most teams pick a strategy once and never revisit it. ChunkyTester makes it easy to run several strategies against your actual documents and compare them on metrics that matter for retrieval quality.

---

## Strategies

| Strategy | Description | Reference |
|---|---|---|
| Recursive Semantic | Recursive splits with LLM coherence scoring; merges over-split pairs and re-splits incoherent chunks | LangChain SemanticChunker |
| Late Chunking | Boundaries determined by full-document context rather than local text features | Gunther et al., 2024 |
| Structure-Aware | Uses heading hierarchy (H1-H6) to define boundaries; never splits a section unless it exceeds the token limit | LlamaIndex Hierarchical Node Parser |
| Parent-Child | Two levels: small child chunks for retrieval precision, large parent chunks returned to the LLM | LlamaIndex Small-to-Big pattern |
| Sliding Window + Summary | Each chunk gets an LLM-generated summary prepended to its embedding input for richer retrieval signal | Anthropic Contextual Retrieval, 2024 |
| Kamradt Semantic | Splits at the Nth percentile of similarity drops between adjacent sentences; threshold adapts to the document | Greg Kamradt, 2024 |
| Agentic Proposition | Decomposes text into atomic, self-contained factual statements with coreference resolution | Chen et al., Dense X Retrieval, 2023 |

---

## Evaluation

After running strategies, the eval panel scores each one across:

- **Self-Containedness** -- LLM-judged score (0-1) for how well each chunk stands alone without surrounding context
- **Boundary Coherence** -- LLM-judged score (0-1) for how natural each split point is
- **Token Efficiency** -- ratio of meaningful tokens to total tokens, computed client-side
- **Retrieval Precision** -- Precision@3 and Precision@5 based on your manual relevance judgments after query simulation

Results are shown as a radar chart and a sortable comparison table with a composite score.

---

## Query Simulator

Enter a test query and the tool ranks each strategy's chunks by relevance using the selected LLM. For Parent-Child, it matches against child chunks and shows the parent chunk that would be sent to the LLM. Mark results as relevant or not relevant to compute precision scores.

---

## Supported Providers

| Provider | Models | Cost |
|---|---|---|
| Anthropic | claude-sonnet-4-20250514 | $3 / $15 per million tokens (input/output) |
| Google Gemini | gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro | varies by model |
| Ollama | any locally installed model | free |

API keys are stored in `localStorage` only and are never sent to any server other than the provider's own API endpoint.

Ollama requires CORS to be enabled: set `OLLAMA_ORIGINS=*` before starting the server.

For Qwen3 models on Ollama, the `think` flag is automatically set to `false` to suppress extended reasoning chains.

---

## Getting Started

### Prerequisites

- Node.js 18 or later
- An API key for Anthropic, Google Gemini, or a running Ollama instance

### Install and run

```bash
git clone https://github.com/redisitic/ChunkyTester.git
cd ChunkyTester
npm install
npm run dev
```

Open the settings panel (gear icon, top right) and enter your API key before running any strategy.

### Build

```bash
npm run build
```

Output is written to `dist/`.

---

## Deployment

A GitHub Actions workflow is included at `.github/workflows/deploy.yml`. It builds the app and deploys to GitHub Pages on every push to `main`.

To activate: go to your repository Settings, open the Pages section, and set the source to GitHub Actions.

---

## Export

Results can be exported in three formats:

- **JSON** -- full chunk results, eval scores, and query results
- **CSV** -- one row per chunk with strategy, token count, and eval scores
- **Markdown** -- comparison table with a top-3 recommendation based on composite score

---

## Input formats

- PDF (`.pdf`) -- text and heading structure extracted via pdf.js; heading levels inferred from font size
- HTML (`.html`, `.htm`) -- parsed via DOMParser; heading hierarchy extracted from H1-H6 tags

---

## Notes

- Passages longer than 6000 characters will show a cost and time estimate before any strategy is run
- Results are cached in component state by a hash of the passage, strategy, and config; changing any config value invalidates the cache
- The sliding window summary strategy makes one LLM call per chunk; large passages will produce many calls
- The agentic chunker uses streaming to handle large JSON outputs
