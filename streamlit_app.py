"""
Streamlit UI is disabled for now. Use the CLI instead (from repo root)::

    $env:PYTHONPATH = "src"
    python -m cli.rag ingest --html Apple --strategy semantic
    python -m cli.rag query "What are the main risk factors?" -k 5

To re-enable a Streamlit UI later, restore the previous implementation from git history.
"""

from __future__ import annotations

import sys


def main() -> None:
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    main()
