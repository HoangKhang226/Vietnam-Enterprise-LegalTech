from src.config.logger import logger
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from typing import Dict, List
from src.retrieval.vector_db import VectorDBManager
from src.config.setting import settings

# Import LlamaIndex evaluation modules
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation.retrieval.metrics import MRR, HitRate

import json
from pathlib import Path

# Load ground truth từ file
DATASET_PATH = Path(__file__).parent / "dataset_evaluate_rag.json"

def load_ground_truth() -> Dict[str, List[str]]:
    if not DATASET_PATH.exists():
        logger.info(f"Lỗi: Không tìm thấy file {DATASET_PATH}")
        return {}
        
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return {item["query"]: item["expected_doc_ids"] for item in data}

def evaluate_retriever_no_llm():
    logger.info("Khởi tạo Retrieval Pipeline để đánh giá...")
    
    # Bỏ qua LLM, load Embedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embedding_model = HuggingFaceEmbedding(model_name=settings.embedding.huggingface)
    
    vector_db = VectorDBManager(embedding_model)

    # Khởi tạo các Metric Deterministic (Không dùng LLM)
    metrics = [
        HitRate(), # Tính tỷ lệ câu trả lời lấy được ít nhất 1 Document đúng
        MRR()      # Mean Reciprocal Rank: Thứ hạng trung bình của Document đúng đầu tiên
    ]
    
    logger.info("Bắt đầu đánh giá với tập Ground Truth...")
    results = []
    
    ground_truth = load_ground_truth()
    if not ground_truth:
        return
        
    for query, expected_ids in ground_truth.items():
        logger.info(f"Đang đánh giá câu hỏi: '{query}'")
        
        # Gọi retriever thực tế
        retrieved_nodes = vector_db.retrieve_with_rerank(query, retrieve_top_k=20, rerank_top_n=3)
        retrieved_ids = [node.metadata.get("doc_id", node.id_) for node in retrieved_nodes]
        
        # Tính toán Hit Rate (Có trúng không?)
        is_hit = any(expected_id in retrieved_ids for expected_id in expected_ids)
        
        # Tính toán Reciprocal Rank
        rank = 0
        for i, ret_id in enumerate(retrieved_ids):
            if ret_id in expected_ids:
                rank = 1.0 / (i + 1)
                break
                
        results.append({
            "Query": query,
            "Expected": expected_ids,
            "Retrieved Top 5": retrieved_ids,
            "Hit Rate": 1.0 if is_hit else 0.0,
            "MRR": rank
        })

    # Tổng hợp kết quả
    df = pd.DataFrame(results)
    
    logger.info("\n"+ "="*50)
    logger.info("KẾT QUẢ ĐÁNH GIÁ (DETERMINISTIC EVALUATION)")
    logger.info("="*50)
    logger.info(df.to_string(index=False))
    
    logger.info("\nTổng quan toàn hệ thống:")
    logger.info(f"Average Hit Rate (Top-5): {df['Hit Rate'].mean():.2f}")
    logger.info(f"Mean Reciprocal Rank (MRR): {df['MRR'].mean():.2f}")

if __name__ == "__main__":
    evaluate_retriever_no_llm()
