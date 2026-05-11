"""MemSearch memory plugin — MemoryProvider for semantic long-term recall via Milvus.

Provides cross-session semantic memory with hybrid search (dense vector + BM25 RRF),
progressive disclosure (recall → expand → transcript), auto-ingest of conversation
turns, and compact summarization via the MemSearch CLI and Python library.

Config in $HERMES_HOME/memsearch_config.json or config.yaml:
  plugins:
    memsearch:
      milvus_uri: ~/.memsearch/milvus.db
      embedding_provider: openai
      collection: hermes_memory
      auto_ingest: true
      auto_compact: true
      max_recall_results: 10
      context_budget_tokens: 800
      index_paths: ""

Requires:
  - pip install memsearch
  - Embedding API key (OpenAI, Google, etc.) or local/onnx provider
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "milvus_uri": "~/.memsearch/milvus.db",
    "embedding_provider": "openai",
    "embedding_model": "",
    "collection": "hermes_memory",
    "auto_ingest": True,
    "auto_compact": True,
    "compact_model": "",
    "max_recall_results": 5,
    "context_budget_tokens": 800,
    "index_paths": "",
    "sync_mode": "daemon",
}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "memsearch_recall",
    "description": (
        "Semantic search across indexed memory. Returns ranked excerpts from past "
        "conversations, notes, and documents. Use for finding specific facts, "
        "decisions, or context from earlier sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results (default 5, max 20).",
            },
        },
        "required": ["query"],
    },
}

EXPAND_SCHEMA = {
    "name": "memsearch_expand",
    "description": (
        "Expand a memory chunk to show full section context. Use after memsearch_recall "
        "when a search result snippet is not enough — shows the complete heading section "
        "from the original document. Progressive disclosure: recall → expand → transcript."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chunk_hash": {
                "type": "string",
                "description": "The chunk hash from a memsearch_recall result to expand.",
            },
            "lines": {
                "type": "integer",
                "description": "Number of context lines around the chunk (default: full section).",
            },
        },
        "required": ["chunk_hash"],
    },
}

INGEST_SCHEMA = {
    "name": "memsearch_ingest",
    "description": (
        "Index a file or directory into semantic memory. Markdown files are chunked, "
        "embedded, and stored in Milvus for future recall. Only new/changed content "
        "is indexed (content-hash dedup)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File or directory path to index.",
            },
            "force": {
                "type": "boolean",
                "description": "Re-index everything, even unchanged content (default: false).",
            },
        },
        "required": ["path"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_plugin_config(hermes_home: str = "") -> dict:
    """Load config from $HERMES_HOME/memsearch_config.json or config.yaml."""
    from hermes_constants import get_hermes_home
    _home = hermes_home or str(get_hermes_home())

    # 1. Try native JSON config
    config_path = Path(_home) / "memsearch_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 2. Try config.yaml under plugins.memsearch
    try:
        import yaml
        from hermes_cli.config import cfg_get
        yaml_path = Path(_home) / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8-sig") as f:
                all_config = yaml.safe_load(f) or {}
            plugin_cfg = cfg_get(all_config, "plugins", "memsearch", default={})
            if plugin_cfg:
                return plugin_cfg
    except Exception:
        pass

    return {}


def _real_home() -> str:
    """Return the real user home, even inside a sandboxed HERMES_HOME.

    When Hermes runs the gateway, HOME may be set to a profile sandbox
    (e.g., ``~/.hermes/profiles/<name>/home``).  Milvus and other data
    stores should use the *real* home so that the CLI and the agent
    share the same database.
    """
    # Priority: SUDO_USER home > passwd DB home > $HOME
    import pwd
    for candidate in (
        os.environ.get("SUDO_HOME"),
        os.environ.get("HOME"),
    ):
        if candidate and candidate != os.path.expanduser("~/.hermes"):
            # Skip if it looks like a Hermes sandbox (under profiles/*/home)
            if "/profiles/" not in candidate or "/home" not in candidate:
                return candidate
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        return os.path.expanduser("~")


def _expand_paths(cfg: dict) -> dict:
    """Expand ``~`` in path values to the real (non-sandboxed) home."""
    real_home = _real_home()
    path_keys = {"milvus_uri", "index_paths"}
    for key in path_keys & cfg.keys():
        val = cfg[key]
        if isinstance(val, str) and "~" in val:
            cfg[key] = val.replace("~", real_home, 1)
    return cfg


def _default_config() -> dict:
    """Return a fresh copy of defaults merged with any saved config."""
    cfg = dict(_DEFAULTS)
    saved = _load_plugin_config()
    if saved:
        cfg.update(saved)
    cfg = _expand_paths(cfg)
    return cfg


# ---------------------------------------------------------------------------
# MemSearchMemoryProvider
# ---------------------------------------------------------------------------

class MemSearchMemoryProvider(MemoryProvider):
    """MemSearch-backed semantic memory with hybrid search and auto-ingest."""

    def __init__(self, config: dict | None = None):
        self._config = config or _default_config()
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._is_primary: bool = True
        self._memsearch_available: bool = False
        self._daemon_thread: threading.Thread | None = None
        self._pending_turns: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "memsearch"

    # -----------------------------------------------------------------------
    # Core lifecycle
    # -----------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if memsearch CLI is installed and embedding API key is set.

        Does NOT make network calls — only checks package and env vars.
        """
        try:
            import memsearch  # noqa: F401
        except ImportError:
            return False
        provider = self._config.get("embedding_provider", "openai")
        if provider == "openai":
            return bool(os.environ.get("OPENAI_API_KEY", ""))
        elif provider == "google":
            return bool(os.environ.get("GOOGLE_API_KEY", ""))
        elif provider in ("local", "onnx", "ollama"):
            return True  # no API key needed
        # Default: check for OPENAI_API_KEY as fallback
        return bool(os.environ.get("OPENAI_API_KEY", ""))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize MemSearch for a session.

        kwargs always includes:
          - hermes_home (str): Active HERMES_HOME path
          - platform (str): "cli", "telegram", etc.
          - agent_context (str): "primary", "subagent", "cron", or "flush"
        """
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))

        # Skip writes for non-primary contexts (cron, subagent)
        self._is_primary = kwargs.get("agent_context", "primary") == "primary"

        # Reload config from hermes_home (expands ~ to real home)
        saved = _load_plugin_config(self._hermes_home)
        if saved:
            self._config.update(saved)
        self._config = _expand_paths(self._config)

        # Verify memsearch is importable
        try:
            import memsearch  # noqa: F401
            self._memsearch_available = True
        except ImportError:
            self._memsearch_available = False
            logger.warning("MemSearch package not installed — memory features disabled")

        # Auto-index configured paths on init (only for primary sessions)
        if self._config.get("auto_ingest", True) and self._is_primary:
            index_paths = self._config.get("index_paths", "")
            if index_paths:
                for path_str in index_paths.split(","):
                    path_str = path_str.strip()
                    if path_str:
                        expanded = path_str.replace("$HERMES_HOME", self._hermes_home)
                        expanded = expanded.replace("${HERMES_HOME}", self._hermes_home)
                        expanded = os.path.expanduser(expanded)
                        if Path(expanded).exists():
                            self._index_path(expanded)

        logger.info(
            "MemSearch initialized (session=%s, primary=%s, available=%s)",
            session_id, self._is_primary, self._memsearch_available,
        )

    def shutdown(self) -> None:
        """Flush pending turns and shut down."""
        if self._pending_turns:
            self._flush_turns()
        logger.info("MemSearch shutdown (session=%s)", self._session_id)

    # -----------------------------------------------------------------------
    # Context injection
    # -----------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        """Static description for the system prompt."""
        if not self._memsearch_available:
            return ""
        collection = self._config.get("collection", "hermes_memory")
        try:
            result = subprocess.run(
                ["memsearch", "stats", "--collection", collection],
                capture_output=True, text=True, timeout=10,
            )
            # Parse text output (memsearch stats doesn't have --json-output)
            count = 0
            output = (result.stdout or "") + (result.stderr or "")
            for line in output.strip().split("\n"):
                # Match patterns like "Total chunks: 42" or "chunks: 42"
                if "chunk" in line.lower():
                    try:
                        # Extract number from line like "Total chunks: 42"
                        num_part = line.split(":")[-1].strip()
                        count = int(num_part)
                    except (ValueError, IndexError):
                        pass
            if count == 0:
                return (
                    "# MemSearch Memory\n"
                    "Active. Empty index — conversations will be auto-indexed.\n"
                    "Use memsearch_recall to search memory, memsearch_ingest to add files."
                )
            return (
                f"# MemSearch Memory\n"
                f"Active. {count} chunks indexed with hybrid semantic search.\n"
                f"Use memsearch_recall for semantic search, memsearch_expand for full context, "
                f"memsearch_ingest to add files."
            )
        except Exception:
            return (
                "# MemSearch Memory\n"
                "Active. Use memsearch_recall to search, memsearch_ingest to index files."
            )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memory before each API call."""
        if not self._memsearch_available or not query or len(query) < 5:
            return ""
        max_results = int(self._config.get("max_recall_results", 5))
        budget = int(self._config.get("context_budget_tokens", 800))
        results = self._search(query, top_k=max_results)
        if not results:
            return ""

        # Format results within token budget (≈4 chars per token)
        max_chars = budget * 4
        lines: list[str] = []
        total_chars = 0
        for r in results:
            score = r.get("score", 0)
            source = r.get("source", "")
            heading = r.get("heading", "")
            content = r.get("content", "")[:500]
            chunk_hash = r.get("chunk_hash", "")
            line = f"- [{score:.2f}] {heading} ({source})"
            if chunk_hash:
                line += f" [ref={chunk_hash[:12]}]"
            line += f"\n  {content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        if not lines:
            return ""
        return "## MemSearch Recall\n" + "\n".join(lines)

    # -----------------------------------------------------------------------
    # Turn sync and session lifecycle
    # -----------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Queue a completed turn for background indexing.

        MUST be non-blocking per the MemoryProvider contract.
        """
        if not self._is_primary or not self._config.get("auto_ingest", True):
            return
        if not self._memsearch_available:
            return
        if not user_content or not assistant_content:
            return
        with self._lock:
            self._pending_turns.append((user_content, assistant_content))
        # Flush in background daemon thread
        if self._daemon_thread is None or not self._daemon_thread.is_alive():
            self._daemon_thread = threading.Thread(target=self._background_sync, daemon=True)
            self._daemon_thread.start()

    def _background_sync(self) -> None:
        """Background worker: debounce then flush pending turns."""
        time.sleep(2)  # debounce
        self._flush_turns()

    def _flush_turns(self) -> None:
        """Write pending turns to markdown, then index via memsearch CLI."""
        with self._lock:
            turns = list(self._pending_turns)
            self._pending_turns.clear()

        if not turns:
            return

        # Write turns as markdown to hermes_home/memory/
        memory_dir = Path(self._hermes_home) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        md_path = memory_dir / f"{ts}-{self._session_id[:8]}.md"

        lines: list[str] = []
        for user, assistant in turns:
            lines.append(f"## User\n\n{user}\n")
            lines.append(f"## Assistant\n\n{assistant}\n")

        # Append (don't overwrite — multiple flushes per session)
        with open(md_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Index the file
        self._index_path(str(md_path))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Flush all pending turns and optionally run compact."""
        self._flush_turns()
        if self._config.get("auto_compact", True) and self._is_primary:
            self._run_compact()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract key insights before context compression discards messages."""
        if not messages:
            return ""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return ""
        # Summarize last few user messages for preservation
        topics: set[str] = set()
        for m in user_msgs[-5:]:
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > 10:
                first_sentence = content.split(".")[0][:100]
                topics.add(first_sentence)
        if not topics:
            return ""
        return "MemSearch context from compressed messages: " + "; ".join(topics)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in memory writes to MemSearch index."""
        if not content or action == "remove":
            return
        if not self._memsearch_available:
            return
        memory_dir = Path(self._hermes_home) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        md_path = memory_dir / f"memory-{target}-{action}.md"
        heading = f"# Memory {action}: {target}"
        md_path.write_text(f"{heading}\n\n{content}\n", encoding="utf-8")
        self._index_path(str(md_path))

    # -----------------------------------------------------------------------
    # Tool schemas and dispatch
    # -----------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, EXPAND_SCHEMA, INGEST_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "memsearch_recall":
            return self._handle_recall(args)
        elif tool_name == "memsearch_expand":
            return self._handle_expand(args)
        elif tool_name == "memsearch_ingest":
            return self._handle_ingest(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _handle_recall(self, args: dict) -> str:
        query = args.get("query", "")
        top_k = min(int(args.get("top_k", 5)), 20)
        if not query:
            return tool_error("query is required")
        results = self._search(query, top_k=top_k)
        if not results:
            return json.dumps({"results": [], "count": 0, "message": "No results found"})
        formatted = []
        for r in results:
            formatted.append({
                "content": r.get("content", "")[:500],
                "source": r.get("source", ""),
                "heading": r.get("heading", ""),
                "chunk_hash": r.get("chunk_hash", ""),
                "score": r.get("score", 0),
            })
        return json.dumps({"results": formatted, "count": len(formatted)})

    def _handle_expand(self, args: dict) -> str:
        chunk_hash = args.get("chunk_hash", "")
        if not chunk_hash:
            return tool_error("chunk_hash is required")
        lines = args.get("lines")
        collection = self._config.get("collection", "hermes_memory")
        cmd = ["memsearch", "expand", chunk_hash, "--collection", collection, "--json-output"]
        if lines:
            cmd.extend(["--lines", str(lines)])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return tool_error(f"Expand failed: {result.stderr.strip()}")
        except Exception as e:
            return tool_error(f"Expand error: {e}")

    def _handle_ingest(self, args: dict) -> str:
        path = args.get("path", "")
        if not path:
            return tool_error("path is required")
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            return tool_error(f"Path not found: {path}")
        force = args.get("force", False)
        self._index_path(str(path_obj), force=force)
        return json.dumps({"status": "indexed", "path": str(path_obj), "force": force})

    # -----------------------------------------------------------------------
    # Config schema
    # -----------------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/.memsearch/milvus.db"
        return [
            {
                "key": "embedding_provider",
                "description": "Embedding provider (openai, google, voyage, jina, mistral, ollama, local, onnx)",
                "default": "openai",
                "choices": ["openai", "google", "voyage", "jina", "mistral", "ollama", "local", "onnx"],
            },
            {
                "key": "milvus_uri",
                "description": "Milvus connection URI (local file or remote server)",
                "default": _default_db,
            },
            {
                "key": "collection",
                "description": "Milvus collection name for Hermes memory",
                "default": "hermes_memory",
            },
            {
                "key": "api_key",
                "description": "Embedding API key",
                "secret": True,
                "required": True,
                "env_var": "OPENAI_API_KEY",
                "url": "https://platform.openai.com/api-keys",
            },
            {
                "key": "auto_ingest",
                "description": "Auto-index conversation turns as they happen",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "auto_compact",
                "description": "Run compact summary at session end",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "max_recall_results",
                "description": "Max results from semantic search",
                "default": "10",
            },
            {
                "key": "index_paths",
                "description": "Comma-separated paths to auto-index on init (e.g. ~/.hermes/skills/)",
                "default": "",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to memsearch_config.json, native settings via CLI."""
        config_path = Path(hermes_home) / "memsearch_config.json"
        # Save our config (excluding secrets)
        our_config = {k: v for k, v in values.items() if k not in ("api_key",)}
        config_path.write_text(json.dumps(our_config, indent=2), encoding="utf-8")
        # Also configure memsearch CLI for native settings
        key_map = {
            "milvus_uri": ("milvus", "uri"),
            "embedding_provider": ("embedding", "provider"),
            "embedding_model": ("embedding", "model"),
            "collection": ("milvus", "collection"),
        }
        for key, val in values.items():
            if key == "api_key":
                continue
            toml_key = key_map.get(key)
            if toml_key:
                try:
                    subprocess.run(
                        ["memsearch", "config", "set", f"{toml_key[0]}.{toml_key[1]}", str(val)],
                        capture_output=True, timeout=10,
                    )
                except Exception:
                    logger.debug("memsearch config set %s failed (non-critical)", key)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _index_path(self, path: str, force: bool = False) -> None:
        """Index a path via memsearch CLI (non-blocking)."""
        collection = self._config.get("collection", "hermes_memory")
        cmd = ["memsearch", "index", path, "--collection", collection]
        if force:
            cmd.append("--force")
        provider = self._config.get("embedding_provider", "openai")
        cmd.extend(["--provider", provider])
        model = self._config.get("embedding_model", "")
        if model:
            cmd.extend(["--model", model])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                logger.info("MemSearch indexed %s: %s", path, result.stdout.strip())
            else:
                logger.warning("MemSearch index failed for %s: %s", path, result.stderr.strip()[:200])
        except Exception as e:
            logger.warning("MemSearch index error for %s: %s", path, e)

    def _search(self, query: str, top_k: int = 5) -> list:
        """Run memsearch search and return parsed results."""
        collection = self._config.get("collection", "hermes_memory")
        cmd = [
            "memsearch", "search", query,
            "--top-k", str(top_k),
            "--collection", collection,
            "--json-output",
        ]
        provider = self._config.get("embedding_provider", "openai")
        cmd.extend(["--provider", provider])
        model = self._config.get("embedding_model", "")
        if model:
            cmd.extend(["--model", model])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except Exception as e:
            logger.debug("MemSearch search failed: %s", e)
        return []

    def _run_compact(self) -> None:
        """Run memsearch compact to summarize indexed chunks."""
        collection = self._config.get("collection", "hermes_memory")
        cmd = ["memsearch", "compact", "--collection", collection]
        provider = self._config.get("embedding_provider", "openai")
        cmd.extend(["--provider", provider])
        compact_model = self._config.get("compact_model", "")
        if compact_model:
            cmd.extend(["--llm-model", compact_model])
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            logger.info("MemSearch compact completed")
        except Exception as e:
            logger.debug("MemSearch compact failed (non-critical): %s", e)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the MemSearch memory provider with the plugin system.

    Called by the memory plugin discovery mechanism when this provider
    is selected via ``memory.provider = memsearch`` in config.yaml.
    """
    config = _default_config()
    provider = MemSearchMemoryProvider(config=config)
    ctx.register_memory_provider(provider)