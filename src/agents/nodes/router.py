import os
import joblib
import logging
from src.config.setting import settings
from src.agents.state import ChatState
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

class LocalRouterClassifier:
    """
    Singleton Class quản lý mô hình SVM Classifier cục bộ.
    Giúp tránh việc phải load lại model HuggingFace và SVM mỗi lần chạy Router.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Khởi tạo LocalRouterClassifier...")
            cls._instance = super(LocalRouterClassifier, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance
        
    def _init_model(self):
        try:
            # Load Embedding Model (Dùng chung với cấu hình RAG)
            self.embed_model = HuggingFaceEmbedding(model_name=settings.embedding.huggingface)
            
            # Load SVM Model đã được huấn luyện
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "storage", "router_svm_model.joblib")
            data = joblib.load(os.path.abspath(model_path))
            
            self.clf = data['classifier']
            self.le = data['label_encoder']
            logger.info("Tải thành công mô hình Router SVM!")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình Router SVM: {e}")
            raise e

    def predict(self, text: str) -> str:
        # Nhúng câu hỏi thành vector
        vec = self.embed_model.get_text_embedding(text)
        
        # Dự đoán nhãn
        pred = self.clf.predict([vec])
        prob = self.clf.predict_proba([vec])
        
        label = self.le.inverse_transform(pred)[0]
        confidence = max(prob[0])
        
        logger.info(f"Router Classifier => Nhãn: {label} | Tự tin: {confidence:.2f}")
        return label

def router_node(state: ChatState) -> ChatState:
    """
    Router Node: Phân loại câu hỏi của người dùng thành chitchat, simple_query, hoặc complex_query
    """
    question = state.get("question", "")
    logger.info(f"--- ĐANG XỬ LÝ ROUTER NODE --- Câu hỏi: {question}")
    
    if not question:
        state["query_type"] = "chitchat"
        return state
        
    classifier = LocalRouterClassifier()
    query_type = classifier.predict(question)
    
    # Cập nhật state
    state["query_type"] = query_type
    
    return state
