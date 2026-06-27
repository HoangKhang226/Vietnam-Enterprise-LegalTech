"""
Legal AI - CLI Chat Testing Pipeline (LangGraph Version)
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Thêm đường dẫn project vào PATH
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from src.agents.graph import graph_app
from src.agents.state import ChatState
from src.config.logger import logger

def main():
    logger.info("\n"+ "="*60)
    logger.info("LEGAL AI - LANGGRAPH CHAT PIPELINE")
    logger.info("="*60)
    
    logger.info("\n HỆ THỐNG ĐÃ SẴN SÀNG! (Gõ 'exit'hoặc 'quit'để thoát)")
    logger.info("-"* 60)
    
    # Giả lập lịch sử hội thoại
    chat_history = []
    
    while True:
        try:
            query = input("\n Câu hỏi của bạn: ")
        except (KeyboardInterrupt, EOFError):
            logger.info("Tạm biệt!")
            break
            
        if query.lower() in ['exit', 'quit', 'q']:
            logger.info("Tạm biệt!")
            break
            
        if not query.strip():
            continue
            
        logger.info("\n Đang kích hoạt LangGraph Agents...")
        
        # Khởi tạo trạng thái ban đầu
        initial_state = {
            "question": query,
            "chat_history": chat_history,
            "query_type": "",
            "sub_queries": [],
            "contexts": [],
            "relevant_articles": [],
            "relevant_docs": [],
            "web_search_used": False,
            "draft_answer": "",
            "hallucinated_articles": [],
            "final_answer": "",
            "citations": []
        }
        
        try:
            # Chạy qua luồng đồ thị
            final_state = graph_app.invoke(initial_state)
            
            logger.info("\n AI Trả lời:")
            sys.stdout.write(final_state.get("final_answer", ""))
            sys.stdout.write("\n")
            sys.stdout.flush()
            
            # Cập nhật lịch sử
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": final_state.get("final_answer", "")})
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình xử lý: {e}")

if __name__ == "__main__":
    main()

