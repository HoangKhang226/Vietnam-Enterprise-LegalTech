from typing import TypedDict, List, Optional, Dict, Any

class ChatState(TypedDict):
    # --- Input ---
    question: str                          # Câu hỏi hiện tại của người dùng
    chat_history: List[dict]               # Lịch sử hội thoại [{role, content}, ...]
    
    # --- Decomposition (MỚI) ---
    query_type: str                        # "simple_query"| "complex_query"| "chitchat"
    sub_queries: List[str]                 # Các câu hỏi nhỏ được tách ra từ câu gốc
    
    # --- Retrieval ---
    contexts: List[dict]                   # Danh sách KnowledgeChunk tìm được (DB + Web)
    relevant_articles: List[str]           # formatted_article từ eligible chunks
    relevant_docs: List[str]               # formatted_doc (deduplicated)
    web_search_used: bool                  # Cờ đánh dấu hệ thống đã dùng Web Search
    
    # --- Generation ---
    draft_answer: str                      # Câu trả lời nháp từ LLM
    
    # --- Auditing ---
    hallucinated_articles: List[str]       # Điều bị phát hiện bịa
    final_answer: str                      # Câu trả lời sau khi rà soát
    citations: List[dict]                  # Metadata nguồn để UI hiển thị
