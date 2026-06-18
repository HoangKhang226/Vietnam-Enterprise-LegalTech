# Phase 1: Data Ingestion & Structural Chunking

> Biến dữ liệu thô (Luật, Pháp điển, Án lệ) thành Knowledge Chunks có cấu trúc. **"Garbage in, garbage out"** — cắt sai dữ liệu → LLM trả lời sai.

---

## 1. Thu Thập Dữ Liệu (Collection)

### Nguồn dữ liệu

| Nguồn | Dataset HuggingFace | Config | Nội dung |
|-------|---------------------|--------|----------|
| Pháp điển | `tmquan/phapdien-moj-gov-vn` | `articles` | Luật, Nghị định, Thông tư |
| Án lệ | `tmquan/anle-toaan-gov-vn` | default | Bản án tiền lệ |

### Bài học từ cuộc thi AIGuru

Dataset `tmquan` có cấu trúc phức tạp, **không phải dạng text phẳng**:

- **Pháp điển**: trường chứa nội dung là `content_text`, KHÔNG phải `text` hay `content`. Metadata nguồn nằm trong `source_note_text` và `source_links`.
- **Án lệ**: trường chứa nội dung là `markdown`, KHÔNG phải `text`. Có thêm `applied_article_number` để tham chiếu.

> Collector phải adapt theo cấu trúc thực tế, dùng danh sách field ưu tiên riêng cho từng loại dataset.

### Lọc miền SME

Chỉ index các văn bản thuộc lĩnh vực: Luật Doanh nghiệp, Lao động, Thuế (TNDN/VAT/TNCN), BHXH, Thương mại, SHTT, Kế toán, và các Nghị định/Thông tư hướng dẫn. Sử dụng hệ thống **SME Keywords scoring** (high + medium weight) để đánh giá mức độ liên quan.

---

## 2. Structural Chunking

Sử dụng **Regex + Heuristics** thay vì cắt theo độ dài ký tự cố định.

### Quy tắc

| Loại văn bản | Đơn vị Chunk | Ghi chú |
|-------------|--------------|---------|
| Luật / Nghị định / Thông tư | Từng **Điều** | Regex: `Điều\s+\d+[a-zA-Z]?\.` |
| Điều quá dài (> 8000 ký tự) | Tách theo **Khoản** | Regex: `^\s*\d+\.\s+` |
| Bảng biểu trong Điều | Gộp vào chunk Điều đó | Markdown Table |
| Án lệ | Từng **vụ án** = 1 chunk | Gộp toàn bộ nội dung markdown |

### Phục hồi Metadata từ `source_note`

```text
"Điều 4 Luật số 04/2017/QH14 ngày 12/06/2017"
         ↓ parse ↓
doc_id = "04/2017/QH14",  article_number = "Điều 4",  doc_type = "Luật"
```

Regex doc_id: `\b\d{1,4}/\d{4}/(?:QH\d+|NĐ-CP|[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)+)\b`

---

## 3. Metadata Schema (KnowledgeChunk)

```json
{
  "chunk_id": "04_2017_QH14_Dieu_4_Khoan_1",
  "text": "1. Doanh nghiệp nhỏ và vừa bao gồm...",
  "metadata": {
    "doc_id": "04/2017/QH14",
    "doc_type": "Luật",
    "doc_title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
    "article_number": "Điều 4",
    "formatted_doc": "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ DNNVV",
    "formatted_article": "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ DNNVV|Điều 4",
    "source": "phapdien",
    "sme_score": 6.0,
    "source_note": "Điều 4 Luật số 04/2017/QH14"
  }
}
```

> **`formatted_doc`**: `<mã VB>|<Loại VB> <Mã VB> <Trích yếu>`
> **`formatted_article`**: `<formatted_doc>|<Điều X>`

---

## 4. Validation & Quality Gates

| Check | Điều kiện | Hành động nếu fail |
|-------|-----------|---------------------|
| `chunk_id` không rỗng | Bắt buộc | Ghi vào `metadata_errors.jsonl` |
| `text` ≥ 80 ký tự | Lọc chunk quá ngắn | Loại bỏ |
| `doc_id` match regex | Định dạng chuẩn | Warning |
| Không trùng `chunk_id` | Unique constraint | Append suffix `_dup_N` |

### Output

```text
data/knowledge_store/
├── chunks.jsonl            # Mỗi dòng = 1 JSON hoàn chỉnh
├── chunk_stats.json        # Thống kê tổng hợp
└── metadata_errors.jsonl   # Lỗi cần review
```
