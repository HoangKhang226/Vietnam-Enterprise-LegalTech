import logging
from langgraph.graph import StateGraph, END

# Import các thành phần của Agent
from src.agents.state import ChatState
from src.agents.nodes.router import router_node
from src.agents.nodes.decomposer import decomposer_node
from src.agents.nodes.retriever_node import retriever_node
from src.agents.nodes.web_search import web_search_node
from src.agents.nodes.generator import generator_node
# from src.agents.nodes.auditor import auditor_node

logger = logging.getLogger(__name__)

def route_decision(state: ChatState):
    """
    Hàm phân luồng (Conditional Edge) đứng ngay sau Router Node.
    Quyết định sẽ gửi state đi đâu tiếp theo dựa trên nhãn query_type.
    """
    q_type = state.get("query_type", "chitchat")
    logger.info(f"LangGraph Routing: Điều hướng theo luồng [{q_type.upper()}]")
    
    if q_type == "complex_query":
        return "decomposer"
    elif q_type == "simple_query":
        return "retriever"
    else:
        # Nếu là chitchat, bỏ qua hết các khâu tìm kiếm DB, nhảy thẳng đến LLM
        return "generator"

def build_graph():
    """
    Hàm khởi tạo và biên dịch StateGraph (Đồ thị trạng thái).
    Kết nối toàn bộ Nodes lại với nhau thành luồng nghiệp vụ RAG.
    """
    logger.info("Đang khởi tạo LangGraph Workflow...")
    
    # Khởi tạo đồ thị với khung State
    workflow = StateGraph(ChatState)
    
    # 1. Đăng ký các Nodes vào mạng lưới
    workflow.add_node("router", router_node)
    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generator", generator_node)
    # workflow.add_node("auditor", auditor_node)  # Đã tắt theo yêu cầu user
    
    # 2. Cổng vào (Entry Point): Mọi tin nhắn đều phải qua trạm kiểm soát Router
    workflow.set_entry_point("router")
    
    # 3. Ngã rẽ sau khi Router phân loại
    workflow.add_conditional_edges(
        "router", 
        route_decision, 
        {
            "decomposer": "decomposer",
            "retriever": "retriever",
            "generator": "generator"
        }
    )
    
    # 4. Các đường nối tuần tự (Edges)
    workflow.add_edge("decomposer", "retriever")      # Băm nhỏ xong thì đi tìm kiếm
    workflow.add_edge("retriever", "web_search")      # Tìm DB xong thì rà soát xem có cần Web Search không
    workflow.add_edge("web_search", "generator")      # Gom đủ đồ nghề rồi thì đưa cho LLM chém gió
    workflow.add_edge("generator", END)               # LLM chém gió xong -> KẾT THÚC (Bỏ qua Auditor)
    
    # 5. Đóng gói và biên dịch (Compile)
    app = workflow.compile()
    logger.info("LangGraph đã biên dịch thành công!")
    return app

# Khởi tạo biến toàn cục để các file khác (ví dụ: main.py, streamlit.py) có thể import và gọi graph_app.invoke(state)
graph_app = build_graph()
