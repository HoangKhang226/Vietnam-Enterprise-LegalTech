from src.config.logger import logger
import json
import os
import sys
import types
from pathlib import Path
from datasets import Dataset

# --- MONKEY PATCH CHO LỖI RAGAS ---
class DummyVertexAI: pass

# Inject into chat_models
import langchain_community.chat_models
dummy_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_vertexai_module.ChatVertexAI = DummyVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertexai_module

# Inject into llms without overwriting the whole module!
import langchain_community.llms
langchain_community.llms.VertexAI = DummyVertexAI
# ----------------------------------

from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# LlamaIndex & System imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.vector_db import VectorDBManager
from src.config.setting import settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def main():
    logger.info(f"Khởi tạo RAGAS Evaluator với Local Ollama ({settings.ollama.model})...")
    
    # RAGAS yêu cầu Langchain LLM và Embeddings để làm Giám khảo (Judge)
    judge_llm = Ollama(model=settings.ollama.model, base_url=settings.ollama.base_url)
    judge_embeddings = OllamaEmbeddings(model=settings.ollama.embed_model, base_url=settings.ollama.base_url)
    
    logger.info("Khởi tạo Hybrid Retriever...")
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding.huggingface)
    vector_db = VectorDBManager(embed_model)
    
    # Load test dataset gốc
    dataset_path = Path(__file__).parent / "dataset_evaluate_rag.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
        
    # CHẠY TOÀN BỘ DATASET BÊN TRONG (100 CÂU)
    test_data = full_data
    
    # ==========================================
    # 1. RETRIEVAL CHECKPOINT (Tìm kiếm tài liệu)
    # ==========================================
    checkpoint_file = Path("storage/rag_retrieval_checkpoint.json")
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    retrieved_data = {}
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            retrieved_data = json.load(f)
            logger.info(f"Đã khôi phục {len(retrieved_data)} câu hỏi từ Checkpoint Retrieval...")

    # Lọc ra những câu chưa được tìm kiếm
    pending_items = [item for item in test_data if item["query"] not in retrieved_data]
    
    if pending_items:
        logger.info(f"Đang truy xuất tài liệu cho {len(pending_items)} câu hỏi mới (Chạy SONG SONG)...")
        MAX_WORKERS = 1
        import concurrent.futures
        
        def retrieve_for_item(item):
            q = item["query"]
            expected_ids = item["expected_doc_ids"]
            nodes = vector_db.retrieve_with_rerank(q, retrieve_top_k=20, rerank_top_n=3)
            contexts = [f"Mã văn bản (Doc ID): {n.metadata.get('doc_id', 'Unknown')}\nNội dung: {n.node.text}"for n in nodes]
            gt_string = item.get("expected_answer", f"Các mã văn bản đúng là: {', '.join(expected_ids)}")
            return q, gt_string, contexts

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {executor.submit(retrieve_for_item, item): item for item in pending_items}
            
            completed = 0
            total = len(pending_items)
            
            for future in concurrent.futures.as_completed(future_to_item):
                q, gt_string, contexts = future.result()
                retrieved_data[q] = {
                    "gt_string": gt_string,
                    "contexts": contexts
                }
                # Lưu file liên tục sau mỗi câu để chống mất mát
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(retrieved_data, f, ensure_ascii=False, indent=2)
                
                completed += 1
                logger.info(f"[{completed}/{total}] Đã tìm xong tài liệu cho câu: {q[:50]}...")

    # ==========================================
    # 2. RAGAS BATCHING CHECKPOINT (Chấm điểm)
    # ==========================================
    questions = []
    ground_truths = []
    contexts_list = []
    
    # Gom lại data theo đúng thứ tự
    for item in test_data:
        q = item["query"]
        if q in retrieved_data:
            questions.append(q)
            ground_truths.append(retrieved_data[q]["gt_string"])
            contexts_list.append(retrieved_data[q]["contexts"])

    from ragas.run_config import RunConfig
    import pandas as pd
    
    run_config = RunConfig(timeout=600, max_retries=2, max_workers=2)
    
    BATCH_SIZE = 10 # Cứ 10 câu lưu 1 lần
    total_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE
    
    final_csv_path = Path("storage/ragas_evaluation_results.csv")
    
    evaluated_questions = set()
    if final_csv_path.exists():
        old_df = pd.read_csv(final_csv_path)
        evaluated_questions = set(old_df["user_input"].tolist())
        logger.info(f"Đã khôi phục điểm của {len(evaluated_questions)} câu từ file CSV...")
    
    logger.info(f"\n Bắt đầu chấm {len(questions)} câu bằng RAGAS (Chia thành {total_batches} Batch)...")
    
    for i in range(total_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(questions))
        
        batch_q = questions[start_idx:end_idx]
        
        # Nếu batch này đã được chấm hết thì bỏ qua
        if all(q in evaluated_questions for q in batch_q):
            continue
            
        logger.info(f"\n Đang chấm Batch {i+1}/{total_batches}...")
        
        batch_dict = {
            "user_input": batch_q,
            "retrieved_contexts": contexts_list[start_idx:end_idx],
            "reference": ground_truths[start_idx:end_idx]
        }
        dataset = Dataset.from_dict(batch_dict)
        
        # Bắt đầu gọi Ollama/Gemini làm Giám khảo
        result = evaluate(
            dataset,
            metrics=[context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config
        )
        
        # Ghi đè hoặc nối thêm vào file CSV
        batch_df = result.to_pandas()
        if not final_csv_path.exists():
            batch_df.to_csv(final_csv_path, index=False, encoding='utf-8-sig')
        else:
            # Chỉ lọc các câu chưa có để nối thêm
            batch_df = batch_df[~batch_df['user_input'].isin(list(evaluated_questions))]
            batch_df.to_csv(final_csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            
        evaluated_questions.update(batch_q)
        logger.info(f"Đã lưu kết quả Batch {i+1} vào {final_csv_path}")

    logger.info("\n=======================================================")
    logger.info(f"🎉 ĐÁNH GIÁ HOÀN TẤT Toàn Bộ Hệ Thống RAG!")
    logger.info(f"📁 Bạn có thể tải file kết quả tại: {final_csv_path}")
    logger.info("=======================================================")

if __name__ == "__main__":
    main()
