"""
AIGuru Legal RAG - Simplified BM25 Indexer

Simplified version inspired by reference implementation.
Uses simple whitespace tokenization to avoid Vietnamese tokenizer hangs.
"""

from __future__ import annotations
from src.config.logger import logger

import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.retrieval.config import (
    BM25_B,
    BM25_CHUNK_ID_MAP_FILE,
    BM25_CORPUS_FILE,
    BM25_INDEX_DIR,
    BM25_K1,
    CHUNKS_FILE,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LEGAL_PHRASES = sorted([
    # SME core
    "doanh nghiệp nhỏ và vừa", "hỗ trợ doanh nghiệp", "hộ kinh doanh",
    "doanh nghiệp siêu nhỏ", "doanh nghiệp vừa", "doanh nghiệp nhỏ",
    "khởi nghiệp sáng tạo", "vốn điều lệ", "đăng ký kinh doanh",
    "đăng ký doanh nghiệp", "giấy chứng nhận đăng ký",
    # Labor
    "người lao động", "hợp đồng lao động", "người sử dụng lao động",
    "thời giờ làm việc", "tiền lương", "an toàn lao động",
    "kỷ luật lao động", "sa thải", "nghỉ phép",
    # Insurance
    "bảo hiểm xã hội", "bảo hiểm thất nghiệp", "bảo hiểm y tế",
    # Tax
    "thuế giá trị gia tăng", "thuế thu nhập doanh nghiệp",
    "thuế thu nhập cá nhân", "quản lý thuế", "khai thuế", "nộp thuế",
    "chậm nộp thuế", "hoàn thuế", "miễn thuế", "giảm thuế",
    "hóa đơn điện tử", "hóa đơn chứng từ",
    # Accounting
    "báo cáo tài chính", "kế toán", "kiểm toán",
    # IP
    "sở hữu trí tuệ", "nhãn hiệu", "bản quyền", "sáng chế",
    "kiểu dáng công nghiệp", "chỉ dẫn địa lý",
    # Contract / Commerce
    "hợp đồng", "hợp đồng mua bán", "hợp đồng dịch vụ",
    "thương mại điện tử", "giao dịch điện tử",
    # Administrative
    "xử phạt vi phạm hành chính", "khắc phục hậu quả",
    "vi phạm hành chính", "biện pháp khắc phục",
    # Land / Facilities
    "mặt bằng sản xuất", "đất đai", "thuê đất", "quyền sử dụng đất",
    # Credit
    "bảo lãnh tín dụng", "quỹ bảo lãnh", "tín dụng",
    # Bidding
    "đấu thầu", "nhà thầu",
    # Legal structure
    "nghị định", "thông tư", "quyết định", "nghị quyết",
    "văn bản pháp luật", "quy định pháp luật",
    # Legal actions
    "cấp phép", "thu hồi", "đình chỉ", "tạm đình chỉ",
    "giấy phép", "chứng chỉ hành nghề",
], key=len, reverse=True)  # Longest phrases first for greedy matching

# Cache availability of underthesea for Vietnamese word segmentation
_UNDERTHESEA_AVAILABLE: bool | None = False


def _try_underthesea_tokenize(text: str) -> list[str] | None:
    """Try to use underthesea for proper Vietnamese word segmentation."""
    return None


def ensure_dirs() -> None:
    """Create BM25 index directory if it doesn't exist."""
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_chunks() -> List[Dict[str, Any]]:
    """Load all chunks from chunks.jsonl."""
    chunks = []
    logger.info(f"Loading chunks from {CHUNKS_FILE}...")
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
                if line_no % 10000 == 0:
                    logger.info(f" Loaded {line_no} chunks...")
            except json.JSONDecodeError as exc:
                logger.info(f"Warning: Skip line {line_no} - {exc}")
    
    logger.info(f"Loaded {len(chunks)} chunks total")
    return chunks


def legal_tokenize(text: str) -> List[str]:
    """Vietnamese legal tokenizer: underthesea → regex legal phrases fallback."""
    lowered = text.lower()
    # Always extract legal document IDs (e.g. 04/2017/QH14)
    legal_ids = re.findall(r"\b\d{1,3}/\d{4}/[\wđ-]+\b", lowered)
    # Always extract legal phrase bigrams
    phrases = [phrase.replace("", "_") for phrase in LEGAL_PHRASES if phrase in lowered]

    # Try underthesea for proper Vietnamese segmentation (works on Colab)
    segmented = _try_underthesea_tokenize(text)
    if segmented is not None:
        return legal_ids + segmented + phrases

    # Fallback: regex word extraction + adjacent bigrams
    words = re.findall(r"[0-9a-zà-ỹđ]+", lowered)
    words = [word for word in words if len(word) > 1 or word.isdigit()]
    # Generate adjacent bigrams to approximate compound words
    bigrams = [f"{words[i]}_{words[i+1]}"for i in range(len(words) - 1)]
    return legal_ids + words + bigrams + phrases


def tokenize_corpus(chunks: List[Dict[str, Any]], batch_size: int = 5000) -> List[List[str]]:
    """
    Tokenize corpus with progress tracking.
    
    Process in batches to show progress and avoid memory issues.
    """
    corpus_tokens = []
    total = len(chunks)
    
    logger.info(f"\nTokenizing {total} chunks with simple tokenizer...")
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        for chunk in batch:
            text = chunk.get("text", "")
            try:
                tokens = legal_tokenize(text)
                corpus_tokens.append(tokens)
            except Exception as exc:
                logger.info(f"Warning: Tokenization failed for chunk {chunk.get('chunk_id')} - {exc}")
                corpus_tokens.append([])  # Empty tokens for failed chunks
        
        processed = min(i + batch_size, total)
        logger.info(f" Tokenized {processed}/{total} chunks ({processed*100//total}%)")
    
    logger.info(f"Tokenization complete: {len(corpus_tokens)} documents")
    return corpus_tokens


def build_bm25_index(corpus_tokens: List[List[str]], k1: float = BM25_K1, b: float = BM25_B):
    """
    Build BM25 index from tokenized corpus.
    
    Returns BM25 object from rank_bm25 library.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "rank_bm25 not installed. Install: pip install rank-bm25"
        )
    
    logger.info(f"\nBuilding BM25 index (k1={k1}, b={b})...")
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    logger.info(f"BM25 index built: {len(corpus_tokens)} documents")
    return bm25


def save_bm25_artifacts(
    bm25,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Save BM25 corpus and metadata.
    
    Returns report dict.
    """
    # Save BM25 object (corpus + parameters)
    logger.info(f"\nSaving BM25 corpus to {BM25_CORPUS_FILE}...")
    with BM25_CORPUS_FILE.open("wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 corpus ({BM25_CORPUS_FILE.stat().st_size // 1024} KB)")
    
    # Save chunk_id mapping (position → chunk_id)
    chunk_id_map = [chunk["chunk_id"] for chunk in chunks]
    logger.info(f"\nSaving chunk ID map to {BM25_CHUNK_ID_MAP_FILE}...")
    with BM25_CHUNK_ID_MAP_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunk_id_map, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved chunk ID map ({len(chunk_id_map)} IDs)")
    
    # Save tokenizer config
    from src.retrieval.config import BM25_TOKENIZER_CONFIG_FILE
    tokenizer_config = {
        "tokenizer": "legal_regex_phrases",
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "corpus_size": len(chunks),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"\nSaving tokenizer config to {BM25_TOKENIZER_CONFIG_FILE}...")
    with BM25_TOKENIZER_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved tokenizer config")
    
    report = {
        "status": "success",
        "corpus_size": len(chunks),
        "tokenizer": "legal_regex_phrases",
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "artifacts": {
            "corpus_file": str(BM25_CORPUS_FILE),
            "chunk_id_map_file": str(BM25_CHUNK_ID_MAP_FILE),
            "tokenizer_config_file": str(BM25_TOKENIZER_CONFIG_FILE),
        },
    }
    return report


def build_bm25() -> Dict[str, Any]:
    """
    Main function: Build BM25 index cho toàn bộ corpus.
    
    Steps:
    1. Load chunks từ chunks.jsonl
    2. Tokenize corpus (simple whitespace)
    3. Build BM25 index
    4. Save artifacts
    
    Returns report dict.
    """
    ensure_dirs()
    
    logger.info("="* 60)
    logger.info("PHASE 2: BM25 INDEXING (SIMPLIFIED)")
    logger.info("="* 60)
    
    # Step 1: Load chunks
    logger.info("\n[1/3] Loading chunks...")
    chunks = load_chunks()
    
    if len(chunks) == 0:
        raise ValueError("No chunks found. Run Phase 1 first.")
    
    # Step 2: Tokenize corpus
    logger.info(f"\n[2/3] Tokenizing corpus...")
    corpus_tokens = tokenize_corpus(chunks, batch_size=5000)
    
    # Step 3: Build BM25 index
    logger.info(f"\n[3/3] Building BM25 index...")
    bm25 = build_bm25_index(corpus_tokens, k1=BM25_K1, b=BM25_B)
    
    # Step 4: Save artifacts
    logger.info("\n"+ "="* 60)
    logger.info("SAVING ARTIFACTS")
    logger.info("="* 60)
    report = save_bm25_artifacts(bm25, chunks)
    
    logger.info("\n"+ "="* 60)
    logger.info("BM25 INDEXING COMPLETE")
    logger.info("="* 60)
    logger.info(f"Corpus size: {report['corpus_size']}")
    logger.info(f"Tokenizer: {report['tokenizer']}")
    logger.info(f"Output: {BM25_INDEX_DIR}")
    
    return report


def main() -> None:
    """Entry point."""
    try:
        report = build_bm25()
        logger.info("\n BM25 index ready for retrieval")
    except Exception as exc:
        logger.info(f"\n Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
