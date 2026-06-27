import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import các routes
from src.api.routes import health, chat

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal AI API",
        description="API Backend xử lý truy vấn pháp lý sử dụng LangGraph đa tác tử",
        version="1.0.0"
    )

    # Cấu hình CORS để Frontend (Streamlit/React) gọi được API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Đăng ký các endpoints vào app
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    
    return app

# Khởi tạo ứng dụng chính
app = create_app()
