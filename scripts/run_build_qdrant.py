"""
AIGuru Legal RAG - Build Qdrant Index Script
"""

from src.config.logger import logger
import sys
import json
import argparse
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.paths import CHUNKS_FILE
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.retrieval.vector_db import VectorDBManager, chunks_to_nodes
from src.config.setting import settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device for embedding model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for indexing")
    args = parser.parse_args()
    
    logger.info("="* 60)
    logger.info("PHASE 2: BUILDING QDRANT INDEX")
    logger.info("="* 60)
    
    logger.info(f"\n[1/3] Loading chunks from {CHUNKS_FILE}...")
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} chunks")
    
    logger.info(f"\n[2/3] Initializing embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding.huggingface,
        device=args.device,
        max_length=1024,
    )
    logger.info(f"Model loaded: {settings.embedding.huggingface} on {args.device}")
    
    logger.info(f"\n[3/3] Building Qdrant index...")
    vector_db = VectorDBManager(embedding_model=embed_model)
    
    logger.info(f"Converting chunks to LlamaIndex nodes...")
    nodes = chunks_to_nodes(chunks)
    logger.info(f"Created {len(nodes)} nodes")
    
    logger.info(f"\nAdding nodes to Qdrant (this will take time on {args.device})...")
    vector_db.add_documents(nodes, show_progress=True, batch_size=args.batch_size)
    
    logger.info("\n"+ "="* 60)
    logger.info("QDRANT INDEX BUILD COMPLETE")
    logger.info("="* 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.info(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
