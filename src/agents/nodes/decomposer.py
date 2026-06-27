from src.config.logger import logger
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.agents.state import ChatState
from src.llm.factory import get_llm_provider
from src.agents.prompts import DECOMPOSER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class SubQueries(BaseModel):
    queries: List[str] = Field(description="Mảng chứa 2-4 câu hỏi nhỏ được chia tách từ câu hỏi gốc.")

def decomposer_node(state: ChatState) -> ChatState:
    logger.info("--- ĐANG XỬ LÝ DECOMPOSER NODE ---")
    
    question = state.get("question", "")
    
    if state.get("query_type") != "complex_query":
        logger.info("Không phải complex_query, giữ nguyên câu hỏi.")
        state["sub_queries"] = []
        return state
        
    logger.info(f"Đang băm nhỏ câu hỏi phức tạp: '{question}'")
    
    parser = JsonOutputParser(pydantic_object=SubQueries)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", DECOMPOSER_SYSTEM_PROMPT),
        ("human", "Câu hỏi gốc: {question}")
    ])
    
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    
    llm = get_llm_provider(purpose="rag")
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"question": question})
        sub_queries = result.get("queries", [])
        
        if not sub_queries:
            logger.warning("LLM trả về mảng queries rỗng, dùng câu hỏi gốc.")
            sub_queries = [question]
            
        logger.info(f"Đã phân rã thành {len(sub_queries)} sub-queries:")
        for idx, sq in enumerate(sub_queries):
            logger.info(f" {idx+1}. {sq}")
            
        state["sub_queries"] = sub_queries
        
    except Exception as e:
        logger.error(f"Lỗi khi Decomposer phân tích JSON: {e}")
        state["sub_queries"] = [question]
        
    return state
