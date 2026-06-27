"""
AIGuru Legal RAG - Build BM25 Index Script
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_indexer import main

if __name__ == "__main__":
    main()
