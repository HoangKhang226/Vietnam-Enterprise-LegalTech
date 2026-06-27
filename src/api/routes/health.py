from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["System"])
def health_check():
    """
    API để Frontend hoặc Load Balancer kiểm tra trạng thái sống còn của Backend.
    """
    return {
        "status": "ok", 
        "message": "Legal AI API is running",
        "version": "1.0.0"
    }
