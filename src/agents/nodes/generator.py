import logging
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import ChatState
from src.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

def generator_node(state: ChatState) -> ChatState:
    """
    Generator Node
    Phân luồng tư duy sinh đáp án (3-Tier Strategy):
    1. Nếu chitchat: Dùng kiến thức cá nhân bình thường.
    2. Nếu legal query (RAG): Ép khuôn trả lời chặt chẽ dựa trên Context từ DB + Web.
    """
    logger.info("--- ĐANG XỬ LÝ GENERATOR NODE ---")
    
    question = state.get("question", "")
    query_type = state.get("query_type", "chitchat")
    relevant_docs = state.get("relevant_docs", [])
    web_search_used = state.get("web_search_used", False)
    
    llm = get_llm_provider(purpose="rag")
    
    from src.agents.prompts import GENERATOR_CHITCHAT_PROMPT, GENERATOR_RAG_PROMPT

    if query_type == "chitchat":
        logger.info("Luồng CHITCHAT: Dùng kiến thức cá nhân của LLM để giao tiếp.")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERATOR_CHITCHAT_PROMPT),
            ("human", "{question}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"question": question})
        state["draft_answer"] = response.content
        state["final_answer"] = response.content
        return state

    # Nếu là RAG (simple_query hoặc complex_query)
    logger.info("Luồng RAG: Sinh câu trả lời dựa trên Context (VectorDB / Web Search).")
    context_str = "\n".join(relevant_docs) if relevant_docs else "Không có tài liệu tham chiếu."

    web_warning = "LƯU Ý: Một phần dữ liệu được lấy từ Web Search. Hãy chèn câu cảnh báo '(Nguồn tham khảo từ Internet)' vào cuối đáp án." if web_search_used else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATOR_RAG_PROMPT),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "context": context_str,
        "question": question,
        "web_search_warning": web_warning
    })
    
    # Gán vào draft_answer để Auditor rà soát tiếp
    state["draft_answer"] = response.content
    state["final_answer"] = response.content  # Mặc định gán trước
    
    logger.info("Generator Node hoàn tất việc sinh câu trả lời nháp.")
    return state
