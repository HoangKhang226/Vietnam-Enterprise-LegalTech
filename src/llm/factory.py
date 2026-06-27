from typing import Literal, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from src.config.setting import settings
from src.config.logger import logger


class LLMFactory:
    """Factory for creating pre-configured LangChain Chat Models by purpose."""

    @staticmethod
    def create_client(
        purpose: Literal["summary", "rag", "classifier"],
        provider: str = None,
    ) -> BaseChatModel:
        if provider is None:
            provider = settings.graph_provider.lower()
        else:
            provider = provider.lower()

        if provider == "google":
            provider = "gemini"

        # Determine temperature based on purpose
        if purpose == "summary":
            temperature = 0.7
            model_config_name = "summary_model"
        elif purpose in ("rag", "classifier"):
            temperature = 0.0
            model_config_name = "model"
        else:
            logger.error(f"Unsupported LLM purpose: '{purpose}'")
            raise ValueError(f"LLM purpose not supported: {purpose}")

        if provider == "gemini":
            model = getattr(settings.gemini, model_config_name)
            logger.debug(f"Initializing LangChain Google GenAI for {purpose.upper()} (model: {model}, temp: {temperature})")
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=settings.google_api_key
            )
        elif provider == "ollama":
            model = getattr(settings.ollama, model_config_name)
            logger.debug(f"Initializing LangChain Ollama for {purpose.upper()} (model: {model}, temp: {temperature})")
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=settings.ollama.base_url
            )
        else:
            raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'gemini'or 'ollama'.")

def get_llm_provider(purpose: str = "rag", provider: str = None) -> BaseChatModel:
    """Convenience helper to get a pre-configured LangChain LLM instance."""
    return LLMFactory.create_client(purpose=purpose, provider=provider)
