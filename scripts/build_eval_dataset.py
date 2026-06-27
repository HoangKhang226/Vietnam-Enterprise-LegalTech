from src.config.logger import logger
import json
import time
from pathlib import Path
from src.retrieval.vector_db import VectorDBManager
from src.config.setting import settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def build_dataset():
    input_file = Path("d:/Project/Legal AI/data/R2AIStage1DATA.json")
    output_file = Path("d:/Project/Legal AI/data/eval_dataset.json")
    
    logger.info("1. Đang tải model HuggingFaceEmbedding (BGE-M3)...")
    embedding_model = HuggingFaceEmbedding(model_name=settings.embedding.huggingface)
    
    logger.info("2. Đang kết nối Qdrant Vector DB...")
    vector_db = VectorDBManager(embedding_model)
    
    # Dùng Vector Retriever thuần túy (không dùng Reranker) để tiết kiệm thời gian chạy trên CPU
    index = vector_db.get_index()
    if not index:
        logger.info("Lỗi: Không lấy được Index!")
        return
        
    fast_retriever = index.as_retriever(similarity_top_k=2)
    
    with open(input_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)
        
    # Lấy 100 câu hỏi đầu tiên
    samples = all_data[:100]
    eval_dataset = []
    
    logger.info(f"3. Bắt đầu tự động dán nhãn (Auto-labeling) cho {len(samples)} câu hỏi...")
    start_time = time.time()
    
    for i, item in enumerate(samples):
        query = item["question"]
        try:
            nodes = fast_retriever.retrieve(query)
            
            # Lấy top 1 doc_id làm Ground Truth giả định (Silver Standard)
            expected_ids = []
            if nodes:
                # Lấy doc_id từ metadata của Node tốt nhất
                doc_id = nodes[0].metadata.get("doc_id")
                if doc_id and doc_id not in expected_ids:
                    expected_ids.append(doc_id)
            
            eval_dataset.append({
                "query": query,
                "expected_doc_ids": expected_ids
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Đã xử lý {i + 1}/100 câu hỏi...")
                
        except Exception as e:
            logger.info(f"Lỗi ở câu {i}: {e}")
            
    # Ghi đè vào file eval_dataset.json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, ensure_ascii=False, indent=2)
        
    elapsed = time.time() - start_time
    logger.info(f"\n Đã tạo thành công bộ dataset 100 câu tại: {output_file}")
    logger.info(f"⏱ Thời gian chạy: {elapsed:.1f} giây")

if __name__ == "__main__":
    build_dataset()
