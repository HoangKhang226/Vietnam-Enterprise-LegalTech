import logging
from typing import List, Dict, Any
from src.agents.state import ChatState
from src.retrieval.vector_db import VectorDBManager
from src.retrieval.retriever import RetrieverEngine

logger = logging.getLogger(__name__)

class LocalRetriever:
    """
    Singleton để quản lý instance của VectorDB và RetrieverEngine,
    tránh khởi tạo lại Qdrant và CrossEncoder nhiều lần gây tốn RAM và thời gian.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Khởi tạo LocalRetriever Singleton...")
            cls._instance = super(LocalRetriever, cls).__new__(cls)
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                from src.config.setting import settings
                
                embed_model = HuggingFaceEmbedding(model_name=settings.embedding.huggingface)
                vector_db = VectorDBManager(embedding_model=embed_model)
                cls._instance.engine = RetrieverEngine(vector_db=vector_db)
                logger.info("Khởi tạo RetrieverEngine thành công.")
            except Exception as e:
                logger.error(f"Lỗi khởi tạo RetrieverEngine: {e}")
                cls._instance.engine = None
        return cls._instance

def retriever_node(state: ChatState) -> ChatState:
    """
    Retriever Node (Hybrid Search + Reranking)
    Duyệt qua danh sách sub_queries (nếu có) hoặc question gốc để tìm kiếm ngữ cảnh.
    Lọc bỏ các chunk bị trùng lặp và format lại cho LLM.
    """
    logger.info("--- ĐANG XỬ LÝ RETRIEVER NODE ---")
    
    retriever = LocalRetriever().engine
    if not retriever:
        logger.error("Retriever chưa được khởi tạo. Bỏ qua tác vụ tìm kiếm.")
        return state
        
    queries_to_run = []
    
    # 1. Thu thập câu hỏi cần tìm kiếm
    if state.get("sub_queries") and len(state["sub_queries"]) > 0:
        logger.info(f"Phát hiện {len(state['sub_queries'])} sub_queries từ Decomposer Node. Sẽ chạy Hybrid Search đa luồng.")
        queries_to_run = state["sub_queries"]
    else:
        logger.info("Không có sub_queries, chạy Hybrid Search cho câu hỏi gốc.")
        queries_to_run = [state.get("question", "")]
        
    all_nodes = []
    seen_ids = set()
    
    # 2. Chạy truy xuất và Deduplication
    for q in queries_to_run:
        if not q.strip():
            continue
        logger.info(f"Đang tìm kiếm cho: '{q}'")
        nodes = retriever.retrieve(q)
        
        for n in nodes:
            if n.node_id not in seen_ids:
                seen_ids.add(n.node_id)
                all_nodes.append(n)
                
    if not all_nodes:
        logger.warning("Không tìm thấy bất kỳ ngữ cảnh nào trong Vector DB.")
        state["contexts"] = []
        state["relevant_docs"] = []
        return state
        
    # 3. Re-sort toàn cục dựa vào final score sau khi gom các sub_queries lại
    all_nodes.sort(key=lambda x: x.score, reverse=True)
    
    # Tùy chọn: Bạn có thể cắt lại Top K ở đây một lần nữa nếu số lượng tổng sau khi gộp quá lớn
    # all_nodes = all_nodes[:retriever.top_k]
    
    # 4. Format Context chuẩn bị cho Generator
    formatted_context_str = retriever.format_contexts(all_nodes)
    
    # 5. Cập nhật State
    state["contexts"] = [
        {
            "node_id": n.node_id,
            "score": n.score,
            "metadata": n.metadata,
            "text": n.get_content()
        } for n in all_nodes
    ]
    
    # Truyền string đã được format đẹp đẽ vào danh sách relevant_docs cho Generator dùng
    state["relevant_docs"] = [formatted_context_str]
    
    logger.info(f"Retriever Node hoàn tất: Thu thập được {len(all_nodes)} chunks ngữ cảnh độc lập.")
    
    return state
