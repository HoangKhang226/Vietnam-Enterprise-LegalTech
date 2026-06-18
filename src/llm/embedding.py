from typing import Union
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from src.config.setting import settings
from src.config.logger import logger


class EmbeddingFactory:
    """Factory for creating LangChain Embeddings."""
    
    def __init__(self):
        self.google_model = settings.embedding.google
        self.ollama_model = settings.embedding.ollama
        self.ollama_base_url = settings.ollama.base_url
        self.google_api_key = settings.google_api_key

    def get_embedding(self, provider: str = None) -> Embeddings:
        """Return a LangChain-compatible embedding model instance for the given provider."""
        if provider is None:
            # Fallback to general embedding provider if not specified
            provider = settings.embedding.provider.lower()
        else:
            provider = provider.lower()

        # Handle provider aliases
        if provider == "gemini":
            provider = "google"

        if provider == "ollama":
            logger.debug(f"Initializing LangChain Ollama embeddings (model: {self.ollama_model})")
            return OllamaEmbeddings(
                model=self.ollama_model, 
                base_url=self.ollama_base_url
            )
        elif provider == "google":
            logger.debug(f"Initializing LangChain Google embeddings (model: {self.google_model})")
            return GoogleGenerativeAIEmbeddings(
                model=f"models/{self.google_model}", 
                google_api_key=self.google_api_key
            )
        else:
            raise ValueError(
                f"Provider {provider} not supported. Use 'google' or 'ollama'."
            )
