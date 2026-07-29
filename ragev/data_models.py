"""
RAGEv Data Models.

Pydantic models defining the RAG pipeline configuration space,
including generation models, retrieval strategies, and reranking options.
"""
from typing import Literal
from pydantic import BaseModel


# --- Generation Models ---

class GPT432kModel(BaseModel):
    """GPT-4 32k via Azure OpenAI."""
    provider: str = "azure_openai"
    name: str = "gpt432k"
    mode: str = "chat"


class LLama3Model(BaseModel):
    """LLaMA-3.3-70B-Instruct."""
    provider: str = "openai_api_compatible"
    name: str = "llama-3.3-70b-instruct"
    mode: str = "chat"


class MistralSmallModel(BaseModel):
    """Mistral-Small-3-24B (Mixture-of-Experts)."""
    provider: str = "openai_api_compatible"
    name: str = "mistral-small-3-24b"
    mode: str = "chat"


# --- Dataset Configuration ---

class DatasetConfig(BaseModel):
    """Configuration specifying which indexed document collections to query."""
    datasets: list[str] = []


# --- Reranking Models ---

class CohereRerankingModel(BaseModel):
    """Cohere Rerank English v3.0."""
    reranking_provider_name: str = "cohere"
    reranking_model_name: str = "rerank-english-v3.0"


class RewriteRerankingModel(BaseModel):
    """Query rewriting strategy for reranking."""
    reranking_provider_name: str = "rewrite"
    reranking_model_name: str = "vast"


# --- Retrieval Configuration ---

class RetrieverResourcesModel(BaseModel):
    """Controls whether retriever resources (source chunks) are returned."""
    enabled: bool = True


class RetrievalModel(BaseModel):
    """Retrieval pipeline configuration."""
    search_method: Literal[
        "hybrid_search",
        "semantic_search",
        "full_text_search",
        "individual_hybrid",
        "colbert_search",
        "udapr_search",
    ]
    reranking_enable: bool = False
    reranking_model: None | CohereRerankingModel | RewriteRerankingModel = None
    top_k: int = 3
    score_threshold_enabled: bool = False
    score_threshold: float = 0.0


# --- Main Configuration ---

class VastConfig(BaseModel):
    """Complete RAG pipeline configuration for a single experiment."""
    model: GPT432kModel | LLama3Model | MistralSmallModel
    dataset_configs: DatasetConfig
    retrieval_model: RetrievalModel
    retriever_resource: RetrieverResourcesModel
