# MemSearch Memory Provider

Semantic long-term memory backed by [Milvus](https://milvus.io/) vector storage via [MemSearch](https://zilliztech.github.io/memsearch/).

MemSearch indexes markdown knowledge bases into Milvus with hybrid search (dense vector + BM25 RRF), giving Hermes persistent cross-session recall with progressive disclosure (search → expand → transcript).

## Requirements

- `pip install memsearch` — Milvus-backed knowledge retrieval CLI and Python library
- Embedding API key (OpenAI by default, or configure local/onnx for no API needed)
- Milvus Lite uses a local file (`~/.memsearch/milvus.db`) — no server required

## Setup

```bash
# Interactive setup (recommended)
hermes memory setup    # select memsearch, configure API key

# OR manual configuration
hermes config set memory.provider memsearch
echo 'OPENAI_API_KEY=sk-...' >> ~/.hermes/.env

# Initialize MemSearch config
memsearch config init
memsearch config set milvus.uri ~/.memsearch/milvus.db
memsearch config set embedding.provider openai
```

## Architecture Overview

### Three-Layer Progressive Disclosure

| Layer | Tool | Depth | Use Case |
|-------|------|-------|----------|
| L1 — Search | `memsearch_recall` | Snippet + score | Find relevant facts quickly |
| L2 — Expand | `memsearch_expand` | Full section | Need more context around a result |
| L3 — Transcript | `memsearch expand --full` | Full document | Deep dive into source material |

### Auto-Behavior

- **`prefetch(query)`** — Before each API call, recall top-k relevant chunks and inject as context
- **`sync_turn(user, asst)`** — After each turn, queue markdown for background indexing (daemon thread, non-blocking)
- **`on_session_end(messages)`** — Flush pending turns, run `memsearch compact` to summarize
- **`on_memory_write(action, target, content)`** — Mirror built-in memory tool writes to MemSearch index
- **`on_pre_compress(messages)`** — Extract key topics from messages about to be compressed

### Single-Provider Rule

Only ONE external memory provider can be active at a time, enforced by `MemoryManager`. Setting `memory.provider = memsearch` disables Honcho, Supermemory, etc.

## Tools Provided

### `memsearch_recall`

Semantic search across all indexed memory.

```json
{
  "query": "deployment process for staging",
  "top_k": 5
}
```

Returns: ranked list of chunks with score, source, heading, chunk_hash.

### `memsearch_expand`

Expand a chunk to show full section context. Use when a search result snippet isn't enough.

```json
{
  "chunk_hash": "abc123def456",
  "lines": 50
}
```

Returns: expanded section text with surrounding context.

### `memsearch_ingest`

Manually index a file or directory into semantic memory.

```json
{
  "path": "~/projects/docs/",
  "force": false
}
```

Content-hash dedup: unchanged files skip re-indexing.

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `embedding_provider` | openai, google, voyage, jina, mistral, ollama, local, onnx | `openai` |
| `milvus_uri` | Milvus connection URI | `~/.memsearch/milvus.db` |
| `collection` | Milvus collection for Hermes memory | `hermes_memory` |
| `auto_ingest` | Auto-index conversation turns | `true` |
| `auto_compact` | Run compact summary at session end | `true` |
| `max_recall_results` | Max results from semantic search | `5` |
| `context_budget_tokens` | Token budget for prefetch context | `800` |
| `compact_model` | LLM model for compact summaries | (provider default) |
| `index_paths` | Comma-separated paths to auto-index on init | `""` |
| `sync_mode` | daemon, direct, or skip | `daemon` |

## CLI Commands

```bash
hermes memsearch status           # Show index statistics
hermes memsearch index <path>     # Index a file or directory
hermes memsearch reset            # Drop all indexed data
hermes memsearch config            # Show MemSearch configuration
```

## Pitfalls

1. **`sync_turn()` MUST be non-blocking** — daemon thread pattern, never block the agent loop
2. **Milvus Lite requires no server** — `~/.memsearch/milvus.db` is a local file
3. **Embedding provider must match** — search uses the same provider/model as indexing
4. **Profile isolation** — use `hermes_home` kwarg from `initialize()`, not `~/.hermes`
5. **`is_available()` must not make network calls** — only check package + env vars
6. **Content hash dedup** — memsearch handles this; re-indexing unchanged files is a no-op
7. **subprocess calls to memsearch CLI** — all operations use CLI for isolation and simplicity