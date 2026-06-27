import logging
from fastapi import APIRouter, Depends, HTTPException
from src.api.models.chat import ChatRequest, ChatResponse, Citation
from src.api.dependencies import get_graph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest, graph: CompiledStateGraph = Depends(get_graph)):
    """
    API xử lý câu hỏi của người dùng, đi qua luồng Multi-Agent (LangGraph)
    trả về đáp án và trích dẫn.
    """
    try:
        # 1. Chuẩn bị đầu vào cho LangGraph
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
        state_input = {
            "question": request.message,
            "chat_history": history_dicts,
        }
        
        logger.info(f"Nhận câu hỏi (Session: {request.session_id}): {request.message}")
        
        # 2. Chạy luồng
        result_state = graph.invoke(state_input)
        final_answer = result_state.get("final_answer", "")
        
        # 3. Gom citations từ metadata của các chunks
        contexts = result_state.get("contexts", [])
        citations = []
        seen_articles = set()
        
        for c in contexts:
            metadata = c.get("metadata", {})
            doc_id = metadata.get("doc_id", "Unknown")
            article = metadata.get("article", "Unknown")
            unique_key = f"{doc_id}|{article}"
            
            if unique_key not in seen_articles:
                seen_articles.add(unique_key)
                text_preview = c.get("text", "")[:300] + "..." if c.get("text") else ""
                
                citations.append(Citation(
                    article=article,
                    doc_id=doc_id,
                    doc_title=metadata.get("title", "Văn bản pháp luật"),
                    text_preview=text_preview,
                    score=round(c.get("score", 0.0), 3)
                ))
        
        # 4. Trả kết quả JSON
        return ChatResponse(
            answer=final_answer,
            citations=citations
        )
        
    except Exception as e:
        logger.error(f"Lỗi API chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xử lý câu hỏi")
