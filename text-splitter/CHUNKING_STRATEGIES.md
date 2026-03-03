# Chunking Strategies Status

This document tracks strategy decisions for `pgai-text-splitter` and the current
implementation status.

## Implemented

### CharacterTextSplitter

- LangChain-style single-separator splitting.
- Supports regex separator mode.
- Supports `chunk_size`, `chunk_overlap`, and configurable `length_fn`.

### RecursiveCharacterTextSplitter

- LangChain-style recursive separator fallback (`\n\n` -> `\n` -> ` ` -> char fallback).
- Supports overlap and configurable `length_fn`.

### SentenceChunker

- Sentence-aware splitting with configurable delimiters.
- Greedy sentence packing with overlap backtracking.
- Supports `min_characters_per_sentence` and `min_sentences_per_chunk`.
- Supports configurable `length_fn`.

### SemchunkSplitter (semchunk-inspired)

- Recursive semantic separator hierarchy with punctuation reattachment.
- Longest-sequence preference for newline/tab/space delimiters.
- Expanded delimiter families:
  - sentence terminators
  - clause separators
  - bracket boundaries
  - quote boundaries
  - sentence interrupters
  - symbol boundaries
  - word joiners
- Adaptive merge path:
  - running chars-per-unit estimate
  - binary search for max fitting span
  - memoized range/text length calls
- Exposes `memoize` flag to enable/disable caching.
- Exposes `strict_mode` for denser delimiter precedence rules.
- Supports overlap and configurable `length_fn`.

### SemanticChunker (embedding-boundary splitter)

- Sentence windows + embedding similarity between adjacent windows.
- Savitzky-Golay smoothing (window 5, poly 2).
- Local minima boundary detection.
- Optional skip-window reconnection for tangential aside patterns.
- Size enforcement with sentence-preserving fallback splitting.
- Optional `embedding_fn`; falls back to greedy sentence chunking when absent.

## Cross-cutting Implemented Features

- Configurable length measurement across splitters via `LengthFn`.
- Default character-based measurement (`char_len`).
- Realistic fixture-based integration tests using upstream `pgai` examples.

## Benchmarks

Criterion benchmark target is available:

```bash
cargo bench -p pgai-text-splitter --bench splitters_bench
```

It compares Character, Recursive, Sentence, Semchunk, and Semantic splitters on:

- `examples/summarize_article.sql`
- `examples/embeddings_from_documents/documents/pgai.md`

Includes token-like (word-count) semchunk profiles with `memoize=true/false` for
adaptive merge cache impact measurement.

Includes real-tokenizer (`tiktoken-rs` / `cl100k_base`) semchunk profiles with
`memoize=true/false` for cache impact under token-counted sizing.

### Baseline Snapshot (local)

Approximate medians from one local run (`warm-up=0.5s`, `measurement=1.0s`):

- Markdown (`pgai.md`):
  - `character`: ~33.6 us
  - `recursive`: ~301 us
  - `semchunk`: ~633 us
  - `semchunk_tiktoken_memo`: ~47.2 ms
  - `semchunk_tiktoken_no_memo`: ~69.8 ms
  - `semantic`: ~6.56 ms
- SQL (`summarize_article.sql`):
  - `character`: ~6.37 us
  - `recursive`: ~140 us
  - `semchunk`: ~91.1 us
  - `semchunk_tiktoken_memo`: ~5.24 ms
  - `semchunk_tiktoken_no_memo`: ~7.28 ms
  - `semantic`: ~265 us

Interpretation:

- `memoize=true` materially improves semchunk under real tokenizer counting.
- tokenizer-backed sizing is significantly slower than char/word-count sizing,
  but now directly measurable and tunable.

## Remaining Gaps

### 1. Full semchunk parity

Current `SemchunkSplitter` is semchunk-inspired but not strict parity with the
full upstream delimiter taxonomy and all precedence nuances.

## Strategies Intentionally Not Implemented

- FastChunker (byte-level/SIMD-only tradeoff)
- LateChunker (full-document embedding prepass)
- NeuralChunker (specialized model dependency)
- SlumberChunker (LLM-at-chunk-time cost)
- TableChunker (niche; can be added later)
- AST CodeChunker (low current priority for text-column workloads)

## Suggested Next Steps

1. Add optional strict semchunk mode with a full delimiter table and precedence map.
2. Calibrate default semchunk tokenizer chunk sizes/overlap using benchmark outputs
   from markdown+SQL corpora.
