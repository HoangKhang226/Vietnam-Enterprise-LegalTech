from src.config.logger import logger
import logging
from src.agents.state import ChatState
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)

def web_search_node(state: ChatState) -> ChatState:
    """
    Web Search Node (Fallback)
    Được kích hoạt khi Router quyết định tra cứu, nhưng VectorDB nội bộ
    không có kết quả hoặc kết quả có độ tin cậy quá thấp.
    Sử dụng Tavily API (Tối ưu cho RAG Agent).
    """
    logger.info("--- ĐANG XỬ LÝ WEB SEARCH NODE ---")
    
    if state.get("query_type") == "chitchat":
        return state
        
    contexts = state.get("contexts", [])
    
    # Quyết định có cần Search Web không?
    needs_search = False
    if not contexts:
        needs_search = True
    else:
        # Nếu điểm cao nhất của các chunk nội bộ < 0.4 -> Ngữ cảnh yếu, cần search web bổ trợ
        best_score = max([c.get("score", 0) for c in contexts]) if contexts else 0
        if best_score < 0.4:
            needs_search = True
            
    if not needs_search:
        logger.info("Ngữ cảnh từ VectorDB nội bộ đã đủ độ tin cậy. Bỏ qua Web Search.")
        return state
        
    logger.info("Ngữ cảnh nội bộ yếu hoặc trống. Đang kích hoạt Tavily Search API...")
    state["web_search_used"] = True
    
    import os
    if not os.environ.get("TAVILY_API_KEY"):
        logger.warning("Không tìm thấy TAVILY_API_KEY. Bỏ qua Web Search.")
        return state
    
    try:
        # Tavily trả về cấu trúc mảng dict gọn gàng
        search = TavilySearchResults(max_results=3)
        query = state.get("question", "")
        
        web_result = search.invoke({"query": query + "quy định pháp luật Việt Nam hiện hành"})
        
        # Nối kết quả web vào chung với relevant_docs
        web_docs = state.get("relevant_docs", [])
        if not web_docs:
            web_docs = []
            
        web_str = f"=== TÀI LIỆU TỪ TAVILY WEB SEARCH ===\n{web_result}\n"
        web_docs.append(web_str)
        state["relevant_docs"] = web_docs
        
        logger.info("Hoàn tất thu thập dữ liệu từ Tavily.")
    except Exception as e:
        logger.error(f"Lỗi khi gọi Tavily Web Search: {e}")
        
    return state
