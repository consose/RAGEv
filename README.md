# RAGEv: Retrieval Augmented Generation Evaluation

This repository contains the experimental code and evaluation scripts for the paper:

> **"Optimizing Health Document Analysis with Retrieval Augmented Generation: A Systematic Evaluation Framework"**

RAGEv is a configurable reference framework for systematically evaluating RAG pipelines in health-related policy contexts. It implements a multifactorial experimental design to compare retrieval architectures, embedding models, reranking strategies, and generation backends across heterogeneous healthcare document collections.

## Repository Structure

```
ragev/
├── __init__.py
├── config.py              # Configuration and settings (uses .env file)
├── data_models.py         # Pydantic models for RAG pipeline configurations
├── doe.py                 # Design of Experiments: factorial design generation
├── utils.py               # API interaction utilities with rate limiting
├── data_collection/
│   ├── __init__.py
│   └── collect.py         # Main experiment runner
└── data_analysis/
    ├── __init__.py
    └── analysis.py        # Evaluation metrics computation and statistical analysis

data/                      # Input datasets (see RAGEv-Bench)
├── README.md              # Dataset download links and format description
└── (datasets downloaded from JRC Data Catalogue - see data/README.md)

configs/
└── experiment_levels.yaml # Full experimental design specification
```

## Experimental Design

The evaluation framework implements a full factorial design across the following dimensions:

| Factor | Levels |
|--------|--------|
| **Chunk Strategy** | VHT (amalia), AMR (phages) |
| **Pipeline** | Vector Search, Full-Text Search, Hybrid Search, ColBERTv2, UDAPDR, Individual Hybrid |
| **Embedding Model** | ADA2 (text-embedding-ada-002) |
| **Generation Model** | GPT-4 32k, LLaMA-3.3-70B-Instruct, Mistral-Small-3-24B |
| **Number of Retrieved Chunks** | 3, 10 |
| **Relevancy Score Threshold** | 0.0 (no filter), 0.5 |
| **Reranker** | None, Cohere (rerank-english-v3.0), Rewrite |

This yields 720 unique configurations evaluated across multiple document collections.

## Prompt Template

All experiments use the following prompt template:

```
{question} Answer providing an explanation.
```

Where `{question}` is substituted with each evaluation question from the dataset.

## Inference Parameters

- **Temperature**: 0
- **Response mode**: Blocking (synchronous)
- **Platform**: GPT@JRC (Dify-based RAG orchestration)

## Generation Models

| Model ID | Architecture | Provider |
|----------|-------------|----------|
| `gpt432k` | GPT-4 32k | Azure OpenAI |
| `llama-3.3-70b-instruct` | LLaMA-3.3-70B-Instruct | GPT@JRC |
| `mistral-small-3-24b` | Mistral-Small-3-24B (MoE) | GPT@JRC |

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy the environment template and configure your API credentials:

```bash
cp .env.template .env
```

Edit `.env` with your settings:
- `API_KEY`: Your RAG platform API key
- `URL`: API endpoint URL
- `DATADIR`: Path to input datasets
- `EXPDIR_BINS`: Path for experiment output (pickle files)
- `EXPDIR_LOGS`: Path for experiment logs

## Running Experiments

### Data Collection (Experiment Execution)

```bash
python -m ragev.data_collection.collect
```

This iterates over all factorial design configurations, queries the RAG API for each question in the dataset, and stores results as pickle files.

### Data Analysis (Metrics Computation)

```bash
python -m ragev.data_analysis.analysis --input <merged_results.pkl> --output <output_dir>
```

This computes ROUGE scores, BERTScore, classification metrics, SHAP-based waterfall analysis, and statistical tests.

## RAGEv-Bench Datasets

The RAGEv-Bench benchmark datasets are publicly available at:
- **JRC Data Catalogue**: https://data.jrc.ec.europa.eu/dataset/6e8b187d-7bf3-43ae-80a4-f94d5413cfaa
- **European Data Portal**: https://data.europa.eu/data/datasets/6e8b187d-7bf3-43ae-80a4-f94d5413cfaa

Direct download links:
| Dataset | File | Q&A Pairs |
|---------|------|-----------|
| Virtual Human Twin (VHT) | [ds-VHT.csv](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DIGLIFE/OPEN/RagEv-Bench/ds-VHT.csv) | 2,521 |
| Anti-Microbial Resistance (AMR) | [ds-AMR.csv](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DIGLIFE/OPEN/RagEv-Bench/ds-AMR.csv) | 3,656 |
| PubMedQA (100-document subset) | [ds-PubMedQA_sample100.csv](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DIGLIFE/OPEN/RagEv-Bench/ds-PubMedQA_sample100.csv) | 826 |

See `data/README.md` for detailed format description.

## Evaluation Metrics

### Generative evaluation (VHT, AMR collections)
- **ROUGE** (1, 2, L, Lsum): Lexical overlap with reference answers
- **BERTScore** (Precision, Recall, F1): Semantic similarity using contextual embeddings

### Classification evaluation (PubMedQA)
- **Accuracy, Precision, Recall, F1-score**: Standard classification metrics for binary (yes/no) answers

### Configuration analysis
- **SHAP Waterfall**: Marginal contribution of each configuration parameter
- **Independent t-tests**: Statistical significance of parameter effects (α = 0.001)

## Citation

If you use this code or the RAGEv-Bench datasets, please cite:

```bibtex
@article{consoli2025ragev,
  title={Optimizing Health Document Analysis with Retrieval Augmented Generation: A Systematic Evaluation Framework},
  author={Consoli, Sergio and Medinas, Riccardo and Bertolini, Lorenzo and Comte, Valentin and Spadaro, Nicholas and Mu{\~n}oz Pi{\~n}eiro, Amalia and Raffael, Barbara and Toussaint, Brigitte and Patak Dennstedt, Alexandre and Querci, Maddalena and Wiesenthal, Tobias and Reforgiato Recupero, Diego and Ceresa, Mario},
  journal={Frontiers in Artificial Intelligence},
  year={2025}
}
```

## License

This project is licensed under the EUPL-1.2. See [LICENSE](LICENSE) for details.
# RAGEv
