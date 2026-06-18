from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"

# Cấu hình đường dẫn cho Retrieval / Indexing
BM25_INDEX_DIR = STORAGE_DIR / "bm25"
BM25_CORPUS_FILE = BM25_INDEX_DIR / "corpus.pkl"
BM25_CHUNK_ID_MAP_FILE = BM25_INDEX_DIR / "chunk_id_map.json"
BM25_TOKENIZER_CONFIG_FILE = BM25_INDEX_DIR / "tokenizer_config.json"
CHUNKS_FILE = STORAGE_DIR / "chunks.jsonl"
