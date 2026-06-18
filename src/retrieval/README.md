# 📚 Module Retrieval (`src/retrieval`)

> **Mô tả:** Đây là "bộ não tìm kiếm" (Search Engine) của toàn bộ dự án Legal AI. Tầng Retrieval có nhiệm vụ nhận câu hỏi của người dùng và lục tìm trong cơ sở dữ liệu (hàng nghìn văn bản luật) để bốc ra đúng Điều khoản/Nghị định/Luật cần thiết, hỗ trợ cho LLM trả lời.

---

## 1. Cấu Trúc Thư Mục

Thư mục `src/retrieval` được thiết kế theo hướng module hoá, chia tách rõ ràng trách nhiệm của việc xử lý từ vựng, quản lý CSDL, và thuật toán tìm kiếm:

```text
src/retrieval/
├── __init__.py          # Đánh dấu module Python
├── bm25_indexer.py      # Bộ Tokenizer đặc trị cho pháp luật Việt Nam (Lexical Search)
├── vector_db.py         # Quản lý kết nối Qdrant Database và khởi tạo Hybrid Retriever
└── retriever.py         # Engine tìm kiếm chính (Cross-Encoder, Lexical Boost, Thresholding)
```

---

## 2. Giải Thích Chi Tiết Từng File

### 2.1. `bm25_indexer.py` (Vietnamese Legal Tokenizer)

Đây là một "bộ lọc từ vựng" được thiết kế riêng cho ngôn ngữ pháp lý Việt Nam, phục vụ trực tiếp cho thuật toán tìm kiếm từ khóa BM25.

- **Vấn đề:** Các thư viện tách từ tiếng Việt thông thường (như `underthesea`) dễ gây lỗi hoặc treo hệ thống khi chạy lượng dữ liệu lớn. Nếu không tách từ, chữ "bảo hiểm" và "xã hội" sẽ bị tách rời khiến độ chính xác giảm sút.
- **Giải pháp:** Sử dụng Regular Expression (Regex) để gom các cụm từ quan trọng lại với nhau.
- **Chức năng chính:**
  - Nhận diện các ký hiệu pháp lý cứng (VD: `04/2017/QH14`).
  - Gộp các từ ghép pháp lý (Ví dụ: `doanh nghiệp nhỏ và vừa` -> `doanh_nghiệp_nhỏ_và_vừa`) dựa vào bộ từ điển `LEGAL_PHRASES`.
  - Giúp thuật toán BM25 bắt chính xác tuyệt đối các thực thể pháp lý.

### 2.2. `vector_db.py` (Vector Database Manager)

File này chịu trách nhiệm giao tiếp trực tiếp với cơ sở dữ liệu Vector (Sử dụng Qdrant).

- **Chức năng chính:**
  - Kết nối tới Qdrant (lưu trữ embedded trên ổ cứng để truy xuất offline).
  - Tải mô hình nhúng `BAAI/bge-m3` để mã hóa các câu hỏi/chunk thành các vector nhiều chiều (1024-dim).
  - Khởi tạo **Hybrid Retriever** thông qua LlamaIndex: Nó trộn bộ tìm kiếm Vector (`VectorStoreIndex`) với bộ tìm kiếm BM25 (`BM25Retriever.from_defaults`) để tạo ra một cấu trúc truy xuất lai.

### 2.3. `retriever.py` (Retriever Engine)

Đây là trái tim của hệ thống Retrieval, nơi chứa logic xử lý tìm kiếm cuối cùng. Nó kết hợp kết quả từ Qdrant và đẩy qua một loạt các thuật toán gạn lọc nâng cao.

- **Chức năng chính:**
  - **Nhận query:** Xử lý câu hỏi truyền vào.
  - **Cross-Encoder Reranking:** Sử dụng `BAAI/bge-reranker-v2-m3` để "đọc" lại cặp (Câu hỏi - Câu trả lời) và chấm điểm độ liên quan từ 0 đến 1 một cách chi tiết (thay vì chỉ đo khoảng cách vector).
  - **Lexical Boost:** Thuật toán cộng điểm thưởng nếu mã văn bản (doc_id) hoặc số điều (article) nằm ngay trong câu hỏi.
  - **Dynamic Thresholding:** Cắt và lọc kết quả đầu ra dựa trên điểm số (bỏ các chunk rác).
  - **Context Formatting:** Hàm `format_contexts` dùng để nhóm các chunk cùng một văn bản lại với nhau và trình bày đẹp mắt cho LLM đọc.

---

## 3. Cách Hoạt Động Của Tầng Này (Workflow)

Khi người dùng nhập một câu hỏi, quá trình xử lý ở tầng Retrieval sẽ diễn ra theo 5 bước (Pipeline):

### Bước 1: Khởi tạo và Vector hóa (Embedding & Lexical)

Câu hỏi được đẩy vào `vector_db.py`. Tại đây, câu hỏi được nhân bản làm 2 bản:

- **Bản 1 (Vector):** Đi qua mô hình `bge-m3` biến thành dãy số 1024 chiều để tìm kiếm theo "ngữ nghĩa" ngầm định.
- **Bản 2 (Lexical):** Đi qua `bm25_indexer.py` để tách thành các cụm từ khóa pháp lý (VD: "chậm", "nộp_thuế").

### Bước 2: Truy xuất kết hợp (Hybrid Search & RRF)

Qdrant sẽ trả về 2 danh sách top các đoạn luật (chunks):

- Top 50 từ Vector Search (Bắt ý nghĩa tốt).
- Top 50 từ BM25 Search (Bắt từ khóa tốt).
  Hệ thống dùng thuật toán **Reciprocal Rank Fusion (RRF)** để trộn 2 bảng xếp hạng này lại, lấy ra một danh sách hợp nhất (vd: 30 chunks tốt nhất).

### Bước 3: Reranking và Lexical Boost (Chấm điểm lại)

30 chunks này tiếp tục được đưa vào `retriever.py`. Mô hình Cross-Encoder sẽ đọc lại từng chunk và chấm lại điểm số từ đầu (Rerank Score).
Lúc này, thuật toán **Lexical Boost** được kích hoạt:

- `Final Score = 0.8 * Rerank Score + 0.2 * RRF Score + Điểm thưởng Lexical`
  _(Nếu user hỏi "Theo điều 4 luật doanh nghiệp", chunk nào có metadata `doc_id` là luật doanh nghiệp và `article` là điều 4 sẽ được cộng thêm điểm khổng lồ)._

### Bước 4: Lọc Ngưỡng Động (Dynamic Thresholding)

Hệ thống sẽ không lấy cứng Top 5 hay Top 10 như các hệ thống RAG cùi bắp, mà dùng **Ngưỡng động**:

- Nó sẽ thu gom tất cả các chunk có điểm `>= 0.5` (Độ tin cậy cao).
- Nếu tìm thấy quá ít (dưới 3 chunk), nó tự động hạ ngưỡng xuống `0.3` (Mức an toàn) để cố vét cạn kiến thức.
- Bước này giúp hệ thống tối ưu hóa chỉ số **Recall**, thà bắt nhầm còn hơn bỏ sót, vì LLM ở tầng sau sẽ lo việc đọc hiểu và loại trừ.

### Bước 5: Định dạng Ngữ cảnh (Context Formatting)

Các chunk sống sót cuối cùng sẽ được đưa qua hàm `format_contexts()`. Hàm này gom các chunk thuộc cùng một văn bản lại với nhau, đánh header rõ ràng (VD: `[VĂN BẢN 1]: Luật Doanh nghiệp...`). Định dạng này giúp LLM đọc không bị loạn số hiệu và có thể trích dẫn (Citation) chuẩn xác ra giao diện UI.

Kết quả cuối cùng là một khối văn bản chứa các Điều luật "thơm ngon" nhất, sẵn sàng được nạp cho LLM ở tầng Generation (Generator Node).
