import json
import os
import sys
import time
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Thêm path gốc của project để import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Lấy embedding model từ LlamaIndex (để đồng nhất với RAG)
from src.config.setting import settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def main():
    print("1. Đang tải Dataset...")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "router_dataset.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    print(f"Tổng số mẫu: {len(texts)}")
    
    print("2. Đang khởi tạo Embedding Model (BAAI/bge-m3)...")
    start_time = time.time()
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding.huggingface)
    
    print("3. Đang nhúng văn bản thành Vectors (Embedding)... Quá trình này có thể mất chút thời gian.")
    # Nhúng hàng loạt để lấy vectors
    X = []
    for i, text in enumerate(texts):
        # embed_query sẽ trả về vector 1D
        vec = embed_model.get_text_embedding(text)
        X.append(vec)
        if (i+1) % 20 == 0:
            print(f"  Đã nhúng {i+1}/{len(texts)} câu...")
            
    X = np.array(X)
    
    print("4. Chuẩn bị Labels...")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Chia tập train/test để kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print("5. Huấn luyện mô hình SVM Classification...")
    # Sử dụng SVC với kernel linear hoặc rbf. Linear thường rất hiệu quả với embedding cao chiều.
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    print("6. Đánh giá mô hình...")
    y_pred = clf.predict(X_test)
    print("\nBÁO CÁO KẾT QUẢ PHÂN LOẠI (CLASSIFICATION REPORT):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Lưu mô hình
    print("7. Đang lưu mô hình (Model + LabelEncoder)...")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "storage"), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), "..", "storage", "router_svm_model.joblib")
    
    # Lưu dưới dạng dictionary để dễ dùng
    export_data = {
        "classifier": clf,
        "label_encoder": le
    }
    joblib.dump(export_data, model_path)
    
    print(f"Đã lưu mô hình thành công tại: {model_path}")
    print(f"Tổng thời gian huấn luyện: {time.time() - start_time:.2f} giây")

if __name__ == "__main__":
    main()
