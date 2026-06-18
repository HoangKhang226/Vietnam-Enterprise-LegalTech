"""Legal AI Hybrid Retriever with Cross-Encoder Reranking"""
import os
import math
from src.config.logger import logger
from src.config.setting import settings

def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)

class RetrieverEngine:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.top_k = settings.retrieval.top_k
        self.retrieve_k = self.top_k * 5
        
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder: {settings.rag.reranker}")
            self.cross_encoder = CrossEncoder(settings.rag.reranker)
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder: {e}")
            self.cross_encoder = None
            
        self.fusion_retriever = self.vector_db.get_hybrid_retriever(
            similarity_top_k=self.retrieve_k,
            num_queries=1
        )

    def retrieve(self, query: str):
        if not self.fusion_retriever:
            logger.warning("Retriever is not initialized (no index found).")
            return []
            
        # 1. RRF Retrieval (Vector + BM25) via LlamaIndex
        logger.info(f"Querying: {query}")
        nodes_with_scores = self.fusion_retriever.retrieve(query)
        
        if not nodes_with_scores:
            return []
            
        # 2. CrossEncoder Rerank
        if self.cross_encoder is None:
            return nodes_with_scores[:self.top_k]
            
        pairs = [[query, node.get_content()] for node in nodes_with_scores]
        logits = self.cross_encoder.predict(pairs)
        
        # 3. Combine scores & Lexical Boost
        max_rrf = max([node.score or 1.0 for node in nodes_with_scores]) or 1.0
        query_lower = query.lower()
        
        for node, logit in zip(nodes_with_scores, logits):
            rerank_score = sigmoid(float(logit))
            rrf_score = (node.score or 0.0) / max_rrf
            
            # Lexical Boost
            lexical_boost = 0.0
            doc_id = node.metadata.get("doc_id", "").lower()
            article = node.metadata.get("article", "").lower()
            
            if doc_id and doc_id in query_lower:
                lexical_boost += 0.1
            if article and article in query_lower:
                lexical_boost += 0.05
                
            node.score = 0.8 * rerank_score + 0.2 * rrf_score + lexical_boost
            
        # 4. Sort by final hybrid score
        nodes_with_scores.sort(key=lambda x: x.score, reverse=True)
        
        # 5. Dynamic Thresholding (Tối ưu Recall theo Phase 2)
        SAFE_THRESHOLD = 0.3
        HIGH_CONF_THRESHOLD = 0.5
        MAX_ARTICLES = 10
        MIN_HIGH_CONF = 3
        
        final_nodes = [n for n in nodes_with_scores if n.score >= HIGH_CONF_THRESHOLD]
        
        if len(final_nodes) < MIN_HIGH_CONF:
            final_nodes = [n for n in nodes_with_scores if n.score >= SAFE_THRESHOLD]
            
        if len(final_nodes) < MIN_HIGH_CONF:
            final_nodes = nodes_with_scores[:MIN_HIGH_CONF]
            
        return final_nodes[:MAX_ARTICLES]
        
    def format_contexts(self, nodes_with_scores) -> str:
        """
        Context Formatting (Chống xung đột số hiệu)
        Group chunks theo doc_id -> render headers [VĂN BẢN N]
        """
        from collections import defaultdict
        
        # Nhóm các nodes theo doc_id
        grouped = defaultdict(list)
        for node in nodes_with_scores:
            doc_id = node.metadata.get("doc_id", "UNKNOWN")
            grouped[doc_id].append(node)
            
        formatted_str = "=== CƠ SỞ DỮ LIỆU THAM CHIẾU ===\n\n"
        
        for idx, (doc_id, nodes) in enumerate(grouped.items(), 1):
            # Sort theo article_number (nếu có thể)
            nodes.sort(key=lambda n: n.metadata.get("article", ""))
            
            title = nodes[0].metadata.get("title", "Văn bản pháp luật")
            formatted_str += f"[VĂN BẢN {idx}]: {doc_id} - {title}\n"
            
            for node in nodes:
                article = node.metadata.get("article", "Quy định")
                chunk_text = node.get_content().strip()
                source_id = node.metadata.get("source_id", f"{doc_id}|{article}")
                
                formatted_str += f"- Nội dung {article}: {chunk_text}\n"
                formatted_str += f"  [TRÍCH DẪN HỢP LỆ]: {source_id}\n\n"
                
        return formatted_str.strip()
