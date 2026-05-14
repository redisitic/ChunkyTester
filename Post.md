Most RAG pipelines fail at retrieval, not generation. The chunk is the unit of retrieval. If it's wrong, nothing downstream fixes it.

The problem: everyone picks a chunking strategy once, based on a blog post or a default setting, and never revisits it. Recursive character splitting. 512 tokens. Ship it.

So I built ChunkyTester — a browser tool that lets you run 7 production chunking strategies side by side on your actual documents and measure which one works.

The strategies aren't toy examples:
- **Recursive Semantic** — recursive splits with LLM coherence scoring to merge/resplit at real boundaries
- **Late Chunking** — boundaries set by document-level attention, not local text features (Jina AI, 2024)
- **Structure-Aware** — respects your heading hierarchy; never splits a section mid-thought
- **Parent-Child** — small child chunks for precision retrieval, large parents sent to the LLM (LlamaIndex pattern)
- **Sliding Window + Summary** — each chunk gets an LLM-generated context header prepended to the embedding input
- **Kamradt Semantic** — splits at the Nth percentile of similarity drops, so thresholds adapt to your document's vocabulary
- **Agentic Proposition** — decomposes text into atomic, self-contained factual statements (Chen et al., Dense X Retrieval)

After running them, you get automated eval scores: self-containedness, boundary coherence, token efficiency, a query simulator with relevance judging, and Precision@K. Export as JSON, CSV, or a Markdown report.

Runs entirely in the browser. No server. Works with Claude, Gemini, or a local Ollama model.

The point isn't to declare a winner. It's to make the decision visible before you commit to an architecture.

#RAG #LLM #AIEngineering #RetrievalAugmentedGeneration #VectorSearch #LLMOps #AITools #OpenSource #MachineLearning #NLP
