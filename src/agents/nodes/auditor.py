import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.agents.state import ChatState
from src.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

class AuditResult(BaseModel):
    is_safe: bool = Field(description="True nếu draft_answer an toàn. False nếu phát hiện trích dẫn Điều luật/số hiệu bịa đặt không có trong CƠ SỞ DỮ LIỆU.")
    hallucinated_citations: List[str] = Field(description="Danh sách các câu trích dẫn bịa đặt (nếu có). Trống nếu an toàn.")
    corrected_answer: str = Field(description="Câu trả lời đã được chỉnh sửa loại bỏ các phần bịa đặt (chỉ cần điền nếu is_safe=False).")

def auditor_node(state: ChatState) -> ChatState:
    """
    Auditor Node:
    Đóng vai trò "Thanh tra" rà soát lỗi Hallucination của Generator.
    Kiểm tra xem các trích dẫn trong câu trả lời có khớp với Context không.
    """
    logger.info("--- ĐANG XỬ LÝ AUDITOR NODE ---")
    
    query_type = state.get("query_type", "chitchat")
    draft_answer = state.get("draft_answer", "")
    relevant_docs = state.get("relevant_docs", [])
    
    if query_type == "chitchat" or not draft_answer:
        logger.info("Bỏ qua rà soát vì là chitchat hoặc không có bản nháp.")
        state["final_answer"] = draft_answer
        return state
        
    context_str = "\n".join(relevant_docs) if relevant_docs else "Không có tài liệu tham chiếu."
    
    from src.agents.prompts import AUDITOR_SYSTEM_PROMPT
    
    parser = JsonOutputParser(pydantic_object=AuditResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", AUDITOR_SYSTEM_PROMPT),
        ("human", "CƠ SỞ DỮ LIỆU THAM CHIẾU:\n{context}\n\nBẢN NHÁP CỦA AI:\n{draft_answer}")
    ])
    
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    # Sử dụng model có temperature=0 để rà soát logic khắt khe nhất
    llm = get_llm_provider(purpose="classifier")  
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "context": context_str,
            "draft_answer": draft_answer
        })
        
        is_safe = result.get("is_safe", True)
        
        if is_safe:
            logger.info("Auditor xác nhận: Câu trả lời an toàn, KHÔNG phát hiện Hallucination.")
            state["final_answer"] = draft_answer
        else:
            hallucinated = result.get("hallucinated_citations", [])
            logger.warning(f"Auditor phát hiện Hallucination (Bịa luật): {hallucinated}")
            state["hallucinated_articles"] = hallucinated
            state["final_answer"] = result.get("corrected_answer", draft_answer)
            logger.info("Đã thay thế final_answer bằng bản sửa lỗi an toàn.")
            
    except Exception as e:
        logger.error(f"Lỗi khi Auditor hoạt động: {e}. Bỏ qua kiểm tra.")
        state["final_answer"] = draft_answer
        
    return state
