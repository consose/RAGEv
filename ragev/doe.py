"""
RAGEv Design of Experiments (DOE).

Generates the full factorial experimental design and creates
pipeline configurations for systematic RAG evaluation.
"""
import os
import itertools
import pandas as pd

from ragev.data_models import (
    GPT432kModel, LLama3Model, MistralSmallModel,
    DatasetConfig, CohereRerankingModel, RewriteRerankingModel,
    RetrievalModel, RetrieverResourcesModel, VastConfig,
)
from ragev.config import settings


# Experimental factors and their mnemonic codes for file naming
FACTORS = [
    "chunk strategy",
    "pipeline",
    "embedding",
    "model",
    "#chunks",
    "relevance threshold",
    "reranker",
]

MNEMONICS = ["chnk", "ppln", "embmdl", "genmdl", "nchnk", "th", "rnk"]

# Full factorial levels for the main experiments
# Modify these to run specific subsets of the design
LEVELS = [
    ("amalia", "phages"),                         # Chunk strategy / corpus
    ("txt", "vctr", "hbrd", "clbrt", "udapr"),   # Pipeline
    ("ada2",),                                     # Embedding model
    ("gpt4", "llama3", "nous"),                   # Generation model
    ("3", "10"),                                   # Number of retrieved chunks
    ("0", "5"),                                    # Relevance threshold (0=none, 5=0.5)
    ("n", "y", "r"),                              # Reranker (n=none, y=Cohere, r=Rewrite)
]


def create_fname(params: tuple) -> tuple[str, str]:
    """Create standardised filenames for experiment outputs.

    Args:
        params: Tuple of factor level values.

    Returns:
        Tuple of (pickle_filename, log_filename).
    """
    plist = list(params)
    # Add reasoning model tag (fixed as gpt4 for all experiments)
    plist[2] += "_rsnmdl-gpt4"
    base = "_".join(f"{MNEMONICS[i]}-{par}" for i, par in enumerate(plist))
    return f"{base}.pickle", f"{base}.log"


def load_dataset(dsname: str) -> tuple:
    """Load a Q&A dataset for evaluation.

    Args:
        dsname: Filename of the dataset CSV (e.g., 'ds-amalia-clean.csv').

    Returns:
        Tuple of (paper_list, dataframe).
    """
    ppr_lst = os.listdir(os.path.join(settings.PAPERSDIR, "/"))
    df = pd.read_csv(os.path.join(settings.DATADIR, dsname))
    return ppr_lst, df


def get_experiments() -> list[tuple]:
    """Generate all experiment configurations from the factorial design.

    Returns:
        List of tuples, each representing one experimental configuration.
    """
    return list(itertools.product(*LEVELS))


def create_config(
    chunk_strategy: str,
    pipeline: str,
    embedding: str,
    model: str,
    n_chunks: str,
    rel_th: str,
    reranker: str,
    datasets: list[str] | None = None,
) -> VastConfig:
    """Create a complete RAG pipeline configuration.

    Args:
        chunk_strategy: Corpus identifier ('amalia' for VHT, 'phages' for AMR).
        pipeline: Retrieval pipeline code ('txt', 'vctr', 'hbrd', 'clbrt', 'udapr').
        embedding: Embedding model code ('ada2').
        model: Generation model code ('gpt4', 'llama3', 'nous').
        n_chunks: Number of chunks to retrieve ('3' or '10').
        rel_th: Relevance threshold code ('0' for none, '5' for 0.5).
        reranker: Reranker code ('n' for none, 'y' for Cohere, 'r' for Rewrite).
        datasets: Optional override for dataset IDs.

    Returns:
        VastConfig object ready for API submission.
    """
    # Dataset ID mapping (platform-specific collection identifiers)
    # These IDs correspond to indexed collections on the RAG platform.
    # The source data can be downloaded from:
    # VHT: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DIGLIFE/OPEN/RagEv-Bench/ds-VHT.csv
    # AMR: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DIGLIFE/OPEN/RagEv-Bench/ds-AMR.csv
    ds_map = {
        ("amalia", "ada2"): ["YOUR-VHT-DATASET-ID"],   # VHT collection
        ("phages", "ada2"): ["YOUR-AMR-DATASET-ID"],   # AMR collection
    }

    ds_name = datasets if datasets else ds_map[(chunk_strategy, embedding)]

    # Pipeline mapping
    pip_map = {
        "txt": "full_text_search",
        "vctr": "semantic_search",
        "hbrd": "hybrid_search",
        "shy": "individual_hybrid",
        "clbrt": "colbert_search",
        "udapr": "udapr_search",
    }

    # Model mapping
    model_map = {
        "gpt4": GPT432kModel(),
        "llama3": LLama3Model(),
        "nous": MistralSmallModel(),
    }

    # Reranker mapping
    rer_map = {
        "n": None,
        "y": CohereRerankingModel(),
        "r": RewriteRerankingModel(),
    }

    return VastConfig(
        model=model_map[model],
        dataset_configs=DatasetConfig(datasets=ds_name),
        retrieval_model=RetrievalModel(
            search_method=pip_map[pipeline],
            reranking_enable=(reranker != "n"),
            reranking_model=rer_map[reranker],
            top_k=int(n_chunks),
            score_threshold_enabled=True,
            score_threshold=float(rel_th) / 10.0,
        ),
        retriever_resource=RetrieverResourcesModel(enabled=True),
    )
