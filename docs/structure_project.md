# Cấu Trúc Thư Mục Dự Án — Legal AI (Vietnamese Legal RAG Chatbot)

Dựa trên kinh nghiệm từ pipeline Legal RAG cuộc thi AIGuru và thiết kế Multi-Agent của dự án FinSight AI, hệ thống được tổ chức theo chuẩn MLOps và Microservices. Tách biệt rõ ràng: Thu thập (Ingestion), Tra cứu lai (Hybrid Retrieval), Đa tác tử (Multi-Agent), Backend API, và Giao diện Chatbot.

```text
Legal-AI/
│
├── config/                         # Cấu hình môi trường và tham số pipeline
│   ├── logging.yaml                # Cấu hình log hệ thống (format, level, rotation)
│   └── setting.yaml                # Thông số hệ thống: retrieval thresholds, model paths, API keys
│
├── data/                           # Dữ liệu cục bộ (Chặn đẩy lên Git qua .gitignore)
│   ├── raw_data/                   # File văn bản gốc thu thập (JSONL từ HuggingFace)
│   │   ├── legal_docs_raw.jsonl    # Pháp điển (Luật, Nghị định, Thông tư)
│   │   ├── precedents_raw.jsonl    # Án lệ
│   │   └── collection_report.json  # Báo cáo trạng thái thu thập
│   ├── knowledge_store/            # File JSONL đã qua Chunking + Metadata
│   │   ├── chunks.jsonl            # Mỗi dòng = 1 KnowledgeChunk hoàn chỉnh
│   │   ├── chunk_stats.json        # Thống kê: tổng chunks, doc_type phân bổ, lỗi
│   │   └── metadata_errors.jsonl   # Các chunk bị lỗi metadata (để debug)
│   ├── bm25_index/                 # Artifacts của BM25 Sparse Search
│   │   ├── corpus.pkl              # BM25Okapi object đã build
│   │   ├── chunk_id_map.json       # Mapping: vị trí corpus → chunk_id
│   │   └── tokenizer_config.json   # Tham số tokenizer (k1, b, loại tokenizer)
│
├── storage/                        # Thư mục lưu trữ Vector DB có sẵn (kế thừa từ AIGuru)
│   ├── aiguru_legal/               # Collection Qdrant chứa embeddings
│   └── qdrant_data/                # Config/Meta data của Qdrant
│
├── deployment/                     # Đóng gói và triển khai
│   ├── Dockerfile.api              # Container cho Backend FastAPI + LLM
│   ├── Dockerfile.ui               # Container cho Frontend Streamlit
│   └── docker-compose.yml          # Điều phối cụm: API, UI, Qdrant
│
├── docs/                           # Tài liệu thiết kế hệ thống
│   ├── plan.md                     # Bối cảnh, kiến trúc, tech stack, timeline
│   ├── structure_project.md        # Cấu trúc thư mục (File này)
│   ├── phase_1_ingestion.md        # Thiết kế thu thập và chunking
│   ├── phase_2_retrieval.md        # Thiết kế Hybrid Search
│   ├── phase_3_agents.md           # Thiết kế LangGraph Multi-Agent
│   └── phase_4_serving.md          # Thiết kế API, UI, Deployment
│
├── scripts/                        # Entry-point scripts (CLI)
│   ├── run_collect.py              # Chạy thu thập dữ liệu từ HuggingFace
│   ├── run_chunk.py                # Chạy Structural Chunking
│   ├── run_build_bm25.py           # Build BM25 index
│   ├── run_build_qdrant.py         # Build Qdrant dense index
│   └── run_pipeline.py             # Chạy toàn bộ pipeline end-to-end
│
├── app.py                          # File chạy chính kết hợp Backend + UI Streamlit
│
├── src/                            # MÃ NGUỒN CHÍNH
│   │
│   ├── config/                     # CẤU HÌNH VÀ LOGGING
│   │   ├── __init__.py
│   │   ├── setting.py              # Config TỔNG cho toàn dự án (Pydantic Settings)
│   │   └── logger.py               # Khởi tạo structured loggers
│   │
│   ├── core/                       # CÁC TIỆN ÍCH CỐT LÕI DÙNG CHUNG
│   │   ├── __init__.py
│   │   └── paths.py                # Định nghĩa tất cả path dùng chung (PROJECT_ROOT, ...)
│   │
│   ├── utils/                      # CÁC TIỆN ÍCH KHÁC
│   │   └── __init__.py
│   │
│   ├── llm/                        # TẦNG GIAO TIẾP LLM VÀ EMBEDDING
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base class
│   │   ├── factory.py              # LLMFactory (Gemini/Ollama, định tuyến theo purpose)
│   │   ├── embedding.py            # EmbeddingFactory
│   │   └── providers/              # Các client cụ thể
│   │       ├── __init__.py
│   │       ├── gemini_client.py
│   │       └── ollama_client.py
│   │
│   ├── ingestion/                  # (Phase 1) TẦNG TIỀN XỬ LÝ VÀ CHUNKING
│   │   ├── __init__.py
│   │   ├── collect.py              # Tải dữ liệu từ HuggingFace, normalize fields
│   │   ├── chunk.py                # Structural Chunking (tách theo Điều/Khoản)
│   │   └── metadata_schema.py      # Dataclass: ChunkMetadata, KnowledgeChunk
│   │
│   ├── retrieval/                  # (Phase 2) TẦNG TRA CỨU PHÁP LÝ (HYBRID SEARCH)
│   │   ├── __init__.py
│   │   ├── bm25_indexer.py         # Build BM25 corpus + Legal Phrase tokenizer
│   │   ├── vector_db.py            # Quản lý Qdrant: create collection, add nodes, query
│   │   ├── reranker.py             # Cross-encoder scoring (BGE Reranker)
│   │   └── retriever.py            # Kết hợp BM25 + Dense + RRF + Reranker → ScoredChunk[]
│   │
│   ├── agents/                     # (Phase 3) TẦNG ĐIỀU PHỐI (LANGGRAPH MULTI-AGENT)
│   │   ├── __init__.py
│   │   ├── state.py                # Định nghĩa ChatState (TypedDict) luân chuyển qua graph
│   │   ├── graph.py                # Biên dịch StateGraph: nối nodes + conditional edges
│   │   ├── prompts.py              # System Prompt, Answer Format, Warning template
│   │   └── nodes/                  # Nơi chứa các Node thực thi
│   │       ├── __init__.py
│   │       ├── router.py           # Phân loại câu hỏi (legal_query vs. chitchat)
│   │       ├── retriever_node.py   # Kích hoạt Hybrid Search, đẩy context vào State
│   │       ├── generator.py        # Gọi LLM sinh câu trả lời (batched / streaming)
│   │       └── auditor.py          # Post-processing: anti-hallucination + citation fallback
│   │
│   ├── api/                        # (Phase 4) TẦNG BACKEND PHỤC VỤ CHATBOT
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app factory (lifespan, CORS, exception handlers)
│   │   ├── dependencies.py         # Dependency injection (graph instance, retriever singleton)
│   │   ├── models/                 # Pydantic schemas
│   │   │   ├── chat.py             # ChatRequest, ChatResponse, Citation
│   │   │   └── health.py           # HealthCheckResponse
│   │   └── routes/
│   │       ├── chat.py             # POST /api/v1/chat — stream answer via SSE
│   │       └── health.py           # GET /api/v1/health — system status
│   │
│   └── ui/                         # (Phase 4) GIAO DIỆN CHATBOT (STREAMLIT)
│       ├── app.py                  # Main entry: st.chat_input + st.chat_message
│       ├── api_client.py           # HTTP client gọi FastAPI (SSE consumer)
│       └── components/
│           ├── citation_card.py    # st.expander hiển thị toàn văn Điều luật nguồn
│           └── sidebar.py          # Cấu hình model, session management
│
├── tests/                          # Kiểm thử
│   ├── test_chunking.py            # Validate chunk output format
│   ├── test_retrieval.py           # Retrieval quality checks
│   └── test_pipeline.py            # End-to-end smoke tests
│
├── pyproject.toml                  # Package metadata (editable install: pip install -e .)
├── requirements.txt                # Danh sách thư viện
├── .env                            # Biến môi trường bảo mật (API keys)
└── .gitignore                      # Loại trừ data/, .venv/, __pycache__/, .env
```

---

## Giải Thích Luồng Xử Lý (Data Flow)

### Luồng Offline (Chạy 1 lần khi cập nhật dữ liệu)

```text
scripts/run_collect.py
    → src/ingestion/collect.py      → data/raw_data/*.jsonl

scripts/run_chunk.py
    → src/ingestion/chunk.py        → data/knowledge_store/chunks.jsonl

scripts/run_build_bm25.py
    → src/retrieval/bm25_indexer.py → data/bm25_index/corpus.pkl

scripts/run_build_qdrant.py
    → src/retrieval/vector_db.py    → storage/aiguru_legal/
```

### Luồng Online (Khi người dùng tương tác)

1. **User** nhập câu hỏi trên Streamlit (`src/ui/app.py`).
2. **api_client.py** gửi HTTP POST tới FastAPI (`src/api/routes/chat.py`).
3. **FastAPI** khởi chạy đồ thị LangGraph (`src/agents/graph.py`):
   - **Router Node** → phân loại câu hỏi.
   - **Retriever Node** → gọi `src/retrieval/retriever.py` (BM25 + Dense + RRF + Reranker).
   - **Generator Node** → gọi LLM qua `src/core/llm_factory.py`.
   - **Auditor Node** → lọc hallucination, gắn cảnh báo, format citations.
4. **FastAPI** stream từng token qua **SSE** về Streamlit.
5. **Streamlit** hiển thị câu trả lời + Citation Expanders (toàn văn Điều luật nguồn).
