"""CLI commands for the MemSearch memory provider plugin.

Provides subcommands for managing the MemSearch index:
  hermes memsearch status   — Show index statistics
  hermes memsearch index     — Index a file or directory
  hermes memsearch reset     — Drop all indexed data
  hermes memsearch config    — Show MemSearch configuration
"""

import subprocess
import sys


def memsearch_command(args):
    """Handler dispatched by argparse for 'hermes memsearch' subcommands."""
    sub = getattr(args, "memsearch_command", None)

    if sub == "status":
        collection = getattr(args, "collection", "hermes_memory")
        cmd = ["memsearch", "stats", "--collection", collection, "--json-output"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr, file=sys.stderr)

    elif sub == "index":
        path = getattr(args, "path", None)
        if not path:
            print("Usage: hermes memsearch index <path>", file=sys.stderr)
            sys.exit(1)
        collection = getattr(args, "collection", "hermes_memory")
        cmd = ["memsearch", "index", path, "--collection", collection]
        if getattr(args, "force", False):
            cmd.append("--force")
        provider = getattr(args, "provider", "openai")
        cmd.extend(["--provider", provider])
        model = getattr(args, "model", None)
        if model:
            cmd.extend(["--model", model])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            sys.exit(result.returncode)

    elif sub == "reset":
        collection = getattr(args, "collection", "hermes_memory")
        yes = getattr(args, "yes", False)
        cmd = ["memsearch", "reset", "--collection", collection]
        if yes:
            cmd.append("--yes")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr, file=sys.stderr)

    elif sub == "config":
        cmd = ["memsearch", "config", "list", "--resolved"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr, file=sys.stderr)

    else:
        print("Usage: hermes memsearch <status|index|reset|config>", file=sys.stderr)
        print()
        print("Commands:")
        print("  status   Show MemSearch index statistics")
        print("  index    Index a file or directory into MemSearch")
        print("  reset    Drop all indexed data")
        print("  config   Show MemSearch configuration")
        sys.exit(1)


def register_cli(subparser) -> None:
    """Build the hermes memsearch argparse tree.

    Called by the CLI plugin discovery system when this module is found
    in the plugin directory and the user invokes 'hermes memsearch'.
    """
    subs = subparser.add_subparsers(dest="memsearch_command")

    # status
    status_p = subs.add_parser("status", help="Show MemSearch index statistics")
    status_p.add_argument("--collection", default="hermes_memory", help="Collection name")

    # index
    index_p = subs.add_parser("index", help="Index a file or directory into MemSearch")
    index_p.add_argument("path", help="File or directory to index")
    index_p.add_argument("--force", action="store_true", help="Re-index everything")
    index_p.add_argument("--collection", default="hermes_memory", help="Collection name")
    index_p.add_argument("--provider", default="openai", help="Embedding provider")
    index_p.add_argument("--model", default=None, help="Embedding model (empty = default)")

    # reset
    reset_p = subs.add_parser("reset", help="Drop all indexed data")
    reset_p.add_argument("--collection", default="hermes_memory", help="Collection name")
    reset_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # config
    subs.add_parser("config", help="Show MemSearch configuration")

    subparser.set_defaults(func=memsearch_command)