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
    print(f"🚀 Khởi tạo RAGAS Evaluator với Local Ollama ({settings.ollama.model})...")
    
    # RAGAS yêu cầu Langchain LLM và Embeddings để làm Giám khảo (Judge)
    judge_llm = Ollama(model=settings.ollama.model, base_url=settings.ollama.base_url)
    judge_embeddings = OllamaEmbeddings(model=settings.ollama.embed_model, base_url=settings.ollama.base_url)
    
    print("📦 Khởi tạo Hybrid Retriever...")
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding.huggingface)
    vector_db = VectorDBManager(embed_model)
    
    # Load test dataset gốc
    dataset_path = Path(__file__).parent / "dataset_evaluate_rag.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
        
    # TEST THỬ 1 CÂU THEO YÊU CẦU CỦA USER
    test_data = full_data[:1]
    
    questions = []
    ground_truths = []
    contexts_list = []
    
    print(f"🔍 Đang truy xuất tài liệu cho {len(test_data)} câu hỏi benchmark (Chạy SONG SONG)...")
    
    import concurrent.futures

    def retrieve_for_item(item):
        q = item["query"]
        expected_ids = item["expected_doc_ids"]
        # Gọi Hybrid Retriever
        nodes = vector_db.retrieve_with_rerank(q, retrieve_top_k=20, rerank_top_n=3)
        # MẸO: Đưa doc_id vào text của context
        contexts = [f"Mã văn bản (Doc ID): {n.metadata.get('doc_id', 'Unknown')}\nNội dung: {n.node.text}" for n in nodes]
        gt_string = item.get("expected_answer", f"Các mã văn bản đúng là: {', '.join(expected_ids)}")
        return q, gt_string, contexts

    # Cấu hình số luồng chạy song song (Tùy thuộc vào cấu hình Colab, ví dụ: 4)
    MAX_WORKERS = 10
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(retrieve_for_item, test_data))
        
    for q, gt_string, contexts in results:
        questions.append(q)
        ground_truths.append(gt_string)
        contexts_list.append(contexts)
        print(f"- Câu hỏi: {q} | Đã lấy {len(contexts)} chunks")
        
    # Prepare HuggingFace Dataset format cho Ragas 0.4.x
    data_dict = {
        "user_input": questions,
        "retrieved_contexts": contexts_list,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)
    
    from ragas.run_config import RunConfig
    # Cấu hình max_workers cho Ragas (Giám khảo LLM chấm điểm song song)
    run_config = RunConfig(timeout=600, max_retries=2, max_workers=MAX_WORKERS)
    
    print("\n📊 Bắt đầu chấm điểm bằng RAGAS...")
    # Tính điểm context precision và context recall
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config
    )
    
    print("\n==================================")
    print("✅ KẾT QUẢ ĐÁNH GIÁ RAGAS (1 CÂU):")
    print(result)
    print("==================================")

if __name__ == "__main__":
    main()
