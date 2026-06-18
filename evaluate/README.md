# Hướng dẫn Đánh giá Hệ thống Truy xuất RAG (Retrieval Evaluation Guide)

Tài liệu này giải thích cách hoạt động của hai script đánh giá lõi: `evaluate_retrieval.py` (Đánh giá thuật toán thuần) và `evaluate_rag.py` (Đánh giá ngữ nghĩa bằng LLM-as-a-judge). Đây là các module độc lập được thiết kế chuẩn MLOps để kiểm định năng lực bóc tách và truy xuất pháp lý (Legal Retrieval) của hệ thống.

---

## 1. Cơ chế Truy xuất (Hybrid Search + Cross-Encoder Reranking)

Mục đích tối thượng của pipeline Retrieval này là giải quyết đặc thù "ngôn ngữ mạng nhện" của pháp luật Việt Nam (Luật -> Nghị định -> Thông tư). Việc truy xuất phải bắt được đúng nhánh văn bản liên quan.

Hệ thống hoạt động theo mô hình **Wide Retrieval, Deep Reranking**:
- **Wide Retrieval (Lưới rộng):** Kết hợp BM25 (chống sai lệch từ khóa cứng như số Nghị định) và Vector Embedding (bắt ý nghĩa ngữ nghĩa) để vớt lên Top-20 Chunks tiềm năng nhất.
- **Deep Reranking (Tinh chỉnh sâu):** Dùng mô hình Cross-Encoder (`BAAI/bge-reranker-v2-m3`) để chấm điểm chéo (cross-attention) giữa Câu hỏi và từng Chunk, từ đó lọc ra Top-3 Chunks tinh túy nhất.

> **💡 Lưu ý về Metadata `doc_id`:** Hệ thống Qdrant lưu trữ siêu dữ liệu (metadata) theo định danh **Mã văn bản** (Ví dụ: `123/2020/NĐ-CP`), thay vì số Điều/Khoản. Do đó, việc đánh giá Retrieval sẽ căn cứ vào việc hệ thống có lôi lên đúng Mã văn bản chứa câu trả lời hay không. Điều này đảm bảo tính vĩ mô và sự toàn vẹn của cấu trúc Pháp luật.

---

## 2. Các chỉ số Đánh giá (Metrics)

### A. Nhóm Deterministic (Đánh giá Thuật toán - Nhanh, Không cần LLM)
Nhóm chỉ số này nằm trong script `evaluate_retrieval.py`. Nó đánh giá độ chính xác tuyệt đối bằng toán học.

*   **Hit Rate (Tỷ lệ trúng đích)**
    *   *Ý nghĩa:* Tỷ lệ phần trăm các câu hỏi mà hệ thống lôi lên được **ít nhất một** tài liệu chính xác nằm trong Top-K.
    *   *Công thức:* Tổng số truy vấn có kết quả đúng / Tổng số truy vấn.
    *   *Mục tiêu:* Càng gần 1.0 (100%) càng tốt. Đảm bảo hệ thống không bị "mù" thông tin.

*   **MRR (Mean Reciprocal Rank - Thứ hạng trung bình ngược)**
    *   *Ý nghĩa:* Đánh giá khả năng xếp hạng của Reranker. Tài liệu đúng xuất hiện càng sớm (Top 1, Top 2), điểm càng cao.
    *   *Công thức:* `MRR = 1 / Rank` (Ví dụ: Tài liệu đúng nằm ở Top 1 -> Điểm 1.0; nằm ở Top 3 -> Điểm 0.33).
    *   *Mục tiêu:* Đẩy MRR tiệm cận 1.0, chứng tỏ Reranker hoạt động xuất sắc trong việc đưa thông tin quan trọng lên đầu tiên để mớm cho LLM.

### B. Nhóm Semantic (Đánh giá Ngữ nghĩa chuyên sâu bằng RAGAS)
Nhóm chỉ số này nằm trong script `evaluate_rag.py`. Nó dùng mô hình `Qwen3:8b` đóng vai trò Giám khảo (LLM-as-a-judge) để đọc hiểu từng Chunk.

*   **Context Precision (Độ chính xác của Ngữ cảnh)**
    *   *Ý nghĩa:* Đánh giá Tỷ lệ tín hiệu / Nhiễu (Signal-to-Noise Ratio). Liệu hệ thống có lôi lên toàn rác (chunks không liên quan) làm tốn Token của LLM không?
    *   *Cách LLM chấm:* Nó đọc từng chunk trong Top-3 và tự hỏi: "Đoạn văn này có giúp trả lời câu hỏi không?". Chunk nào vô dụng sẽ bị trừ điểm.

*   **Context Recall (Độ bao phủ của Ngữ cảnh)**
    *   *Ý nghĩa:* Hệ thống có lôi lên **ĐỦ** thông tin không? Có bị thiếu sót ý nào để hình thành một câu trả lời hoàn hảo không?
    *   *Cách LLM chấm:* Nó đối chiếu câu trả lời gốc (Ground Truth) với các chunks được lôi lên xem có ý nào trong đáp án gốc mà các chunks chưa đề cập tới hay không.

---

## 3. Cấu trúc Ground Truth (Tập dữ liệu Vàng)

Mọi đánh giá đều dựa trên bộ dữ liệu 100 câu hỏi tại `dataset_evaluate_rag.json`.
Bộ dữ liệu này được xây dựng chuẩn MLOps với sự đối chiếu chéo đa tầng văn bản (Cross-Reference Documentation).

Ví dụ cấu trúc:
```json
{
  "query": "Chứng từ điện tử trong quản lý thuế được định nghĩa là gì?",
  "expected_doc_ids": [
    "38/2019/QH14",
    "123/2020/NĐ-CP",
    "78/2021/TT-BTC",
    "19/2021/TT-BTC"
  ]
}
```
Hệ thống tính điểm thành công khi Reranker lôi lên được các Chunks có chứa các `doc_id` tương ứng với chuỗi phân cấp Pháp lý này.

---

## 4. Hướng dẫn Chạy (How to Run)

Mở terminal trong môi trường ảo (venv) và chạy:

**1. Đánh giá Tốc độ cao (Hit Rate & MRR):**
```bash
python evaluate/evaluate_retrieval.py
```
*Phù hợp để test nhanh mỗi khi bạn tinh chỉnh thông số (K của BM25, K của Vector) hoặc thay đổi trọng số (alpha) của Hybrid Search.*

**2. Đánh giá Chuyên sâu (Ngữ cảnh RAGAS):**
```bash
python evaluate/evaluate_rag.py
```
*Phù hợp để nghiệm thu cuối ngày. Script này sử dụng LLM nội bộ nên sẽ cần thời gian chạy lâu hơn (đã cài đặt timeout an toàn 600s để tránh quá tải RAM).*
