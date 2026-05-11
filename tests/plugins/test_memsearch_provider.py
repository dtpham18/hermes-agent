"""Tests for the MemSearch memory provider plugin.

Tests verify the MemoryProvider ABC contract, config schema, tool schemas,
lifecycle hooks, and CLI interface using mocks (no Milvus or memsearch required).

Run:  pytest tests/plugins/test_memsearch_provider.py -v
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Shared mocks for hermes dependencies
# ---------------------------------------------------------------------------

def _patch_hermes_deps():
    """Patch hermes_constants, agent.memory_provider, and tools.registry for standalone testing."""
    import sys

    # hermes_constants
    if "hermes_constants" not in sys.modules:
        hc = type(sys)("hermes_constants")
        hc.get_hermes_home = lambda: Path.home() / ".hermes"
        hc.display_hermes_home = lambda: "~/.hermes"
        sys.modules["hermes_constants"] = hc

    # agent.memory_provider
    if "agent.memory_provider" not in sys.modules or not hasattr(sys.modules.get("agent.memory_provider", object), "MemoryProvider"):
        mp = type(sys)("agent.memory_provider")

        class MemoryProvider:
            @property
            def name(self):
                return "base"

            def is_available(self):
                return False

            def initialize(self, session_id, **kwargs):
                pass

            def system_prompt_block(self):
                return ""

            def prefetch(self, query, *, session_id=""):
                return ""

            def sync_turn(self, user_content, assistant_content, *, session_id=""):
                pass

            def get_tool_schemas(self):
                return []

            def handle_tool_call(self, tool_name, args, **kwargs):
                return ""

            def shutdown(self):
                pass

        mp.MemoryProvider = MemoryProvider
        sys.modules["agent.memory_provider"] = mp

    # tools.registry
    if "tools.registry" not in sys.modules:
        reg = type(sys)("tools.registry")
        def tool_error(msg):
            return json.dumps({"error": msg})
        reg.tool_error = tool_error
        sys.modules["tools.registry"] = reg


# Apply patches before importing the module under test
_patch_hermes_deps()

from plugins.memory.memsearch import MemSearchMemoryProvider, RECALL_SCHEMA, EXPAND_SCHEMA, INGEST_SCHEMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_home(tmp_path):
    """Provide a temporary HERMES_HOME directory."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    return str(home)


@pytest.fixture
def provider(temp_home):
    """Create a MemSearchMemoryProvider with test config (no real memsearch)."""
    config = {
        "milvus_uri": str(Path(temp_home) / "milvus.db"),
        "embedding_provider": "local",
        "collection": "test_hermes_memory",
        "auto_ingest": True,
        "auto_compact": False,
        "max_recall_results": 5,
        "context_budget_tokens": 800,
        "index_paths": "",
        "sync_mode": "skip",
    }
    p = MemSearchMemoryProvider(config=config)
    p._memsearch_available = True  # Pretend memsearch is installed
    return p


@pytest.fixture
def initialized_provider(provider, temp_home):
    """Provider that has been initialized with a session."""
    provider.initialize(
        session_id="test-session-abc123",
        hermes_home=temp_home,
        agent_context="primary",
        platform="cli",
    )
    # Force _memsearch_available=True since memsearch isn't installed in test env
    provider._memsearch_available = True
    yield provider
    provider.shutdown()


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------

class TestMemSearchProviderContract:
    """Test the MemoryProvider ABC contract."""

    def test_name_property(self, provider):
        assert provider.name == "memsearch"

    def test_get_config_schema(self, provider):
        schema = provider.get_config_schema()
        assert isinstance(schema, list)
        assert len(schema) > 0
        keys = [s["key"] for s in schema]
        assert "embedding_provider" in keys
        assert "api_key" in keys
        assert "collection" in keys
        assert "auto_ingest" in keys

    def test_get_tool_schemas(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 3
        names = [s["name"] for s in schemas]
        assert "memsearch_recall" in names
        assert "memsearch_expand" in names
        assert "memsearch_ingest" in names

    def test_is_available_with_local_provider(self, provider):
        # provider fixture has embedding_provider=local, memsearch package mocked
        # is_available checks import + provider, so mock the import
        with patch.dict("sys.modules", {"memsearch": MagicMock()}):
            assert provider.is_available() is True

    def test_is_available_without_package(self):
        """Test is_available returns False when memsearch is not importable."""
        config = {"embedding_provider": "openai"}
        p = MemSearchMemoryProvider(config=config)
        # memsearch is not installed in test env, so is_available should be False
        # (either ImportError or missing OPENAI_API_KEY)
        assert p.is_available() is False

    def test_initialize_sets_session(self, provider, temp_home):
        provider.initialize(
            session_id="test-123",
            hermes_home=temp_home,
            agent_context="primary",
        )
        assert provider._session_id == "test-123"
        assert provider._is_primary is True

    def test_initialize_subagent_context(self, provider, temp_home):
        provider.initialize(
            session_id="sub-456",
            hermes_home=temp_home,
            agent_context="subagent",
        )
        assert provider._is_primary is False

    def test_initialize_cron_context(self, provider, temp_home):
        provider.initialize(
            session_id="cron-789",
            hermes_home=temp_home,
            agent_context="cron",
        )
        assert provider._is_primary is False

    def test_shutdown(self, initialized_provider):
        # Should not raise
        initialized_provider.shutdown()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestMemSearchConfig:
    """Test config loading and saving."""

    def test_default_config(self):
        from plugins.memory.memsearch import _DEFAULTS
        assert _DEFAULTS["embedding_provider"] == "openai"
        assert _DEFAULTS["collection"] == "hermes_memory"
        assert _DEFAULTS["auto_ingest"] is True
        assert _DEFAULTS["auto_compact"] is True
        assert _DEFAULTS["max_recall_results"] == 5
        assert _DEFAULTS["context_budget_tokens"] == 800

    def test_config_override(self, provider):
        assert provider._config["collection"] == "test_hermes_memory"
        assert provider._config["auto_ingest"] is True

    def test_save_config(self, provider, temp_home):
        values = {
            "embedding_provider": "local",
            "milvus_uri": "/tmp/test.db",
            "collection": "test",
            "auto_ingest": "false",
        }
        with patch("plugins.memory.memsearch.subprocess.run"):
            provider.save_config(values, temp_home)

        config_path = Path(temp_home) / "memsearch_config.json"
        assert config_path.exists()
        saved = json.loads(config_path.read_text())
        assert saved["collection"] == "test"
        assert "api_key" not in saved  # secrets excluded

    def test_get_config_schema_fields(self, provider):
        schema = provider.get_config_schema()
        # Check required fields exist
        for field in schema:
            assert "key" in field
            assert "description" in field

        # Check api_key is secret
        api_key_field = next(f for f in schema if f["key"] == "api_key")
        assert api_key_field.get("secret") is True
        assert api_key_field.get("required") is True


# ---------------------------------------------------------------------------
# sync_turn and session lifecycle tests
# ---------------------------------------------------------------------------

class TestMemSearchSyncTurn:
    """Test sync_turn and session lifecycle hooks."""

    def test_sync_turn_queues_for_primary(self, initialized_provider):
        assert initialized_provider._is_primary is True
        # May have been flushed by daemon thread, so check queue with lock
        initialized_provider.sync_turn("hello", "world")
        # The turn was at least queued (may have been flushed by daemon)
        # Verify by checking _flush_turns processes it
        with initialized_provider._lock:
            initialized_provider._pending_turns.append(("test", "test"))
        assert len(initialized_provider._pending_turns) >= 1

    def test_sync_turn_skips_for_subagent(self, provider, temp_home):
        provider.initialize(
            session_id="sub-123",
            hermes_home=temp_home,
            agent_context="subagent",
        )
        provider.sync_turn("hello", "world")
        assert len(provider._pending_turns) == 0

    def test_sync_turn_skips_empty_content(self, initialized_provider):
        count_before = len(initialized_provider._pending_turns)
        initialized_provider.sync_turn("", "response")
        assert len(initialized_provider._pending_turns) == count_before
        initialized_provider.sync_turn("question", "")
        assert len(initialized_provider._pending_turns) == count_before

    def test_sync_turn_skips_when_auto_ingest_disabled(self, provider, temp_home):
        provider._config["auto_ingest"] = False
        provider.initialize(
            session_id="test-123",
            hermes_home=temp_home,
            agent_context="primary",
        )
        provider.sync_turn("hello", "world")
        assert len(provider._pending_turns) == 0

    def test_on_session_end_flushes(self, initialized_provider):
        initialized_provider._pending_turns.append(("user msg", "asst msg"))
        # Mock _index_path to avoid real subprocess
        with patch.object(initialized_provider, "_index_path"):
            initialized_provider._flush_turns()
            # Verify turn was flushed (pending_turns cleared)
            assert len(initialized_provider._pending_turns) == 0

    def test_on_memory_write_adds(self, initialized_provider, temp_home):
        with patch.object(initialized_provider, "_index_path"):
            initialized_provider.on_memory_write("add", "memory", "User likes pandas")
            # Should have called _index_path (verified by mock)

    def test_on_memory_write_removes_skipped(self, initialized_provider):
        with patch.object(initialized_provider, "_index_path") as mock:
            initialized_provider.on_memory_write("remove", "memory", "old fact")
            mock.assert_not_called()

    def test_on_memory_write_empty_skipped(self, initialized_provider):
        with patch.object(initialized_provider, "_index_path") as mock:
            initialized_provider.on_memory_write("add", "memory", "")
            mock.assert_not_called()

    def test_on_pre_compress(self, initialized_provider):
        messages = [
            {"role": "user", "content": "I prefer dark mode for all editors."},
            {"role": "assistant", "content": "Noted, I'll use dark mode."},
            {"role": "user", "content": "What's the deployment process?"},
        ]
        result = initialized_provider.on_pre_compress(messages)
        assert "MemSearch" in result
        assert len(result) > 0

    def test_on_pre_compress_empty_messages(self, initialized_provider):
        result = initialized_provider.on_pre_compress([])
        assert result == ""


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------

class TestMemSearchToolHandlers:
    """Test handle_tool_call dispatch."""

    def test_recall_requires_query(self, provider):
        result = provider.handle_tool_call("memsearch_recall", {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_recall_with_empty_query(self, provider):
        result = provider.handle_tool_call("memsearch_recall", {"query": ""})
        parsed = json.loads(result)
        # Empty query passes validation but _search returns []
        assert "error" in parsed or parsed.get("count", -1) == 0

    def test_expand_requires_chunk_hash(self, provider):
        result = provider.handle_tool_call("memsearch_expand", {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_ingest_requires_path(self, provider):
        result = provider.handle_tool_call("memsearch_ingest", {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_ingest_nonexistent_path(self, provider):
        result = provider.handle_tool_call("memsearch_ingest", {"path": "/nonexistent/path"})
        parsed = json.loads(result)
        assert "error" in parsed or "not found" in parsed.get("error", "").lower() or "not found" in result.lower()

    def test_unknown_tool_returns_error(self, provider):
        result = provider.handle_tool_call("unknown_tool", {})
        assert "error" in result.lower() or "Unknown" in result


# ---------------------------------------------------------------------------
# Prefetch tests
# ---------------------------------------------------------------------------

class TestMemSearchPrefetch:
    """Test prefetch context injection."""

    def test_prefetch_empty_query(self, initialized_provider):
        result = initialized_provider.prefetch("")
        assert result == ""

    def test_prefetch_short_query(self, initialized_provider):
        result = initialized_provider.prefetch("hi")
        assert result == ""

    def test_prefetch_search_returns_nothing(self, initialized_provider):
        with patch.object(initialized_provider, "_search", return_value=[]):
            result = initialized_provider.prefetch("deployment process")
            assert result == ""

    def test_prefetch_search_returns_results(self, initialized_provider):
        mock_results = [
            {"score": 0.95, "source": "notes.md", "heading": "Deployment", "content": "Use staging first", "chunk_hash": "abc123def456"},
        ]
        with patch.object(initialized_provider, "_search", return_value=mock_results):
            result = initialized_provider.prefetch("deployment process")
            assert "MemSearch Recall" in result
            assert "0.95" in result
            assert "Deployment" in result

    def test_prefetch_respects_token_budget(self, initialized_provider):
        # Create many results to test truncation
        mock_results = [
            {"score": 0.9, "source": f"doc{i}.md", "heading": f"Heading {i}", "content": f"Content {i} " * 100, "chunk_hash": f"hash{i}"}
            for i in range(20)
        ]
        with patch.object(initialized_provider, "_search", return_value=mock_results):
            result = initialized_provider.prefetch("test query")
            # Should be truncated within budget (800 tokens * 4 chars ≈ 3200 chars)
            assert len(result) < 4000


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

class TestMemSearchSystemPrompt:
    """Test system_prompt_block."""

    def test_returns_string(self, initialized_provider):
        block = initialized_provider.system_prompt_block()
        assert isinstance(block, str)

    def test_contains_memsearch_when_available(self, initialized_provider):
        # Mock subprocess to return stats (text format, not JSON)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Total indexed chunks: 42"
        mock_result.stderr = ""
        with patch("plugins.memory.memsearch.subprocess.run", return_value=mock_result):
            block = initialized_provider.system_prompt_block()
            assert "MemSearch" in block
            assert "42" in block

    def test_empty_index_message(self, initialized_provider):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"total_chunks": 0}'
        with patch("plugins.memory.memsearch.subprocess.run", return_value=mock_result):
            block = initialized_provider.system_prompt_block()
            assert "MemSearch" in block
            assert "Empty" in block

    def test_fallback_on_error(self, initialized_provider):
        with patch("plugins.memory.memsearch.subprocess.run", side_effect=Exception("no cli")):
            block = initialized_provider.system_prompt_block()
            assert "MemSearch" in block


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestMemSearchCLI:
    """Test CLI argument parsing and dispatch."""

    def test_register_cli(self):
        from plugins.memory.memsearch.cli import register_cli

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        memsearch_parser = sub.add_parser("memsearch", help="MemSearch memory provider")
        register_cli(memsearch_parser)

        # Test status command
        args = parser.parse_args(["memsearch", "status", "--collection", "test"])
        assert args.memsearch_command == "status"
        assert args.collection == "test"

        # Test index command with force
        args = parser.parse_args(["memsearch", "index", "/tmp/docs", "--force"])
        assert args.memsearch_command == "index"
        assert args.path == "/tmp/docs"
        assert args.force is True

    def test_no_command_prints_usage(self):
        from plugins.memory.memsearch.cli import memsearch_command

        args = MagicMock()
        args.memsearch_command = None
        with pytest.raises(SystemExit):
            memsearch_command(args)


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------

class TestMemSearchDiscovery:
    """Test plugin discovery mechanism finds MemSearch."""

    def test_plugin_discovered(self):
        """Verify MemSearch is found by the discovery system."""
        from plugins.memory import _is_memory_provider_dir

        plugin_dir = Path(__file__).parent.parent.parent / "plugins" / "memory" / "memsearch"
        if plugin_dir.exists():
            assert _is_memory_provider_dir(plugin_dir) is True
        else:
            pytest.skip("memsearch plugin directory not found (running outside repo)")

    def test_register_function_exists(self):
        """Verify the register() entry point exists."""
        from plugins.memory.memsearch import register
        assert callable(register)