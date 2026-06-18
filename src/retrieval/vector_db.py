"""
LlamaIndex Qdrant VectorDBManager (Ported from Chat With Data)
"""
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.llms import MockLLM
from src.core.paths import STORAGE_DIR
from src.config.setting import settings
from src.config.logger import logger
from src.retrieval.bm25_indexer import legal_tokenize

class VectorDBManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.collection_name = settings.storage.collection_name
        self._qdrant_path = STORAGE_DIR / "qdrant_data"
        self._qdrant_path.mkdir(parents=True, exist_ok=True)
        self.persist_dir = STORAGE_DIR / self.collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._db_client = QdrantClient(path=str(self._qdrant_path))
        self._index = None

    def _get_embedding_dimension(self) -> int:
        return 1024  # bge-m3 dimension

    def _collection_exists(self) -> bool:
        existing = [c.name for c in self._db_client.get_collections().collections]
        return self.collection_name in existing

    def _get_storage_context(self) -> StorageContext:
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore

        if not self._collection_exists():
            embed_dim = self._get_embedding_dimension()
            self._db_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}'")

        vector_store = QdrantVectorStore(
            client=self._db_client,
            collection_name=self.collection_name,
        )

        docstore_path = self.persist_dir / "docstore.json"
        if docstore_path.exists():
            try:
                return StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=str(self.persist_dir),
                )
            except Exception as e:
                logger.warning(f"Failed to load storage context: {e}")

        return StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore(),
        )

    def get_index(self):
        if self._index is not None:
            return self._index
        try:
            storage_context = self._get_storage_context()
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=storage_context.vector_store,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )
            return self._index
        except Exception as e:
            logger.warning(f"Error loading index: {e}")
            return None

    def get_hybrid_retriever(self, similarity_top_k: int = 10, num_queries: int = 1):
        index = self.get_index()
        if index is None:
            logger.error("Cannot build hybrid retriever — index is empty.")
            return None

        # Lấy nodes trực tiếp từ storage_context vì VectorStoreIndex có thể không giữ docstore
        storage_context = self._get_storage_context()
        all_nodes = list(storage_context.docstore.docs.values())
        
        if not all_nodes:
            logger.warning("Docstore is empty - BM25 has no data.")
            return index.as_retriever(similarity_top_k=similarity_top_k)

        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        
        # Initialize BM25 with our custom legal tokenizer
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=similarity_top_k,
            tokenizer=legal_tokenize
        )

        logger.info("Initialising Hybrid Retriever (BM25 + Vector / RRF)...")
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=num_queries,
            mode="reciprocal_rerank",
            use_async=False,
            llm=MockLLM()
        )
        return hybrid_retriever

    def retrieve_with_rerank(self, query: str, retrieve_top_k: int = 25, rerank_top_n: int = 3):
        """
        Thực hiện retrieve số lượng lớn (top_k), sau đó dùng Cross-Encoder Reranker 
        để chọn lọc lại những nodes chất lượng nhất (top_n) dựa trên ngữ nghĩa chéo.
        """
        logger.info(f"Retrieving top {retrieve_top_k} nodes...")
        retriever = self.get_hybrid_retriever(similarity_top_k=retrieve_top_k)
        nodes = retriever.retrieve(query)
        
        logger.info(f"Reranking top {rerank_top_n} nodes using {settings.rag.reranker}...")
        from llama_index.core.postprocessor import SentenceTransformerRerank
        from llama_index.core import QueryBundle
        
        reranker = SentenceTransformerRerank(
            model=settings.rag.reranker,
            top_n=rerank_top_n
        )
        
        reranked_nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query))
        return reranked_nodes
