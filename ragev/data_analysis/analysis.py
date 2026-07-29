"""
RAGEv Data Analysis: Evaluation Metrics and Statistical Analysis.

Computes ROUGE scores, BERTScore, classification metrics, retrieval-specific
metrics (Precision@k, Recall@k, MRR, nDCG@k), SHAP-based waterfall analysis,
factorial ANOVA, and statistical significance tests for the multifactorial
RAG evaluation.

Usage:
    python -m ragev.data_analysis.analysis --input <merged_results.pkl> --output <output_dir>
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

import evaluate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import seaborn as sns
import matplotlib.pyplot as plt


# --- Metric Computation ---

def compute_rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE metrics between predictions and references.

    Args:
        predictions: List of generated answers.
        references: List of reference answers.

    Returns:
        Dictionary with rouge1, rouge2, rougeL, rougeLsum scores.
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore metrics.

    Args:
        predictions: List of generated answers.
        references: List of reference answers.

    Returns:
        Dictionary with precision, recall, f1 lists.
    """
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
    )
    return {
        "precision": np.mean(results["precision"]),
        "recall": np.mean(results["recall"]),
        "f1": np.mean(results["f1"]),
    }


def compute_classification_metrics(y_true: list, y_pred: list) -> dict:
    """Compute standard classification metrics for binary QA.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, precision, recall, f1.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", pos_label="yes"),
        "recall": recall_score(y_true, y_pred, average="binary", pos_label="yes"),
        "f1": f1_score(y_true, y_pred, average="binary", pos_label="yes"),
    }


# --- Retrieval-Specific Metrics ---

def compute_retrieval_metrics(retriever_resources: list[dict], ground_truth_doc: str, k: int = 10) -> dict:
    """Compute retrieval-specific metrics for a single query.

    Evaluates the quality of retrieved chunks against the known relevant document,
    independently of generation quality. Implements standard IR metrics as defined
    in Yilmaz et al. (2010).

    Args:
        retriever_resources: List of retrieved chunk metadata dicts (must contain 'document_name').
        ground_truth_doc: The identifier of the correct source document.
        k: Number of top results to consider.

    Returns:
        Dictionary with precision_k, recall_k, mrr, ndcg_k.
    """
    resources = retriever_resources[:k]
    if not resources:
        return {"precision_k": 0.0, "recall_k": 0.0, "mrr": 0.0, "ndcg_k": 0.0}

    gt_doc = str(ground_truth_doc).replace(".pdf", "")

    # Binary relevance: 1 if chunk belongs to the correct document
    relevances = []
    for r in resources:
        doc_name = str(r.get("document_name", "")).replace(".pdf", "")
        rel = 1 if (gt_doc in doc_name or doc_name in gt_doc) else 0
        relevances.append(rel)

    n_retrieved = len(relevances)
    n_relevant = sum(relevances)

    # Precision@k: fraction of retrieved chunks from correct document
    precision_k = n_relevant / n_retrieved

    # Recall@k (Hit_score@k): whether at least one relevant chunk is retrieved
    recall_k = 1.0 if n_relevant > 0 else 0.0

    # MRR: reciprocal rank of first relevant chunk
    mrr = 0.0
    for i, rel in enumerate(relevances):
        if rel == 1:
            mrr = 1.0 / (i + 1)
            break

    # nDCG@k: normalized discounted cumulative gain
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal_rels = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    ndcg_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision_k": precision_k,
        "recall_k": recall_k,
        "mrr": mrr,
        "ndcg_k": ndcg_k,
    }


def compute_retrieval_metrics_from_experiment(experiment_data: list, k: int = 10) -> pd.DataFrame:
    """Compute retrieval metrics for all queries in an experiment.

    Args:
        experiment_data: List of [pubid, question, api_response] entries.
        k: Number of top results to consider.

    Returns:
        DataFrame with per-query retrieval metrics.
    """
    results = []

    for entry in experiment_data:
        pubid = entry[0]
        res = entry[2]

        if not isinstance(res, dict) or "metadata" not in res:
            continue
        if "retriever_resources" not in res["metadata"]:
            continue

        metrics = compute_retrieval_metrics(
            res["metadata"]["retriever_resources"], pubid, k=k
        )
        metrics["pubid"] = pubid
        results.append(metrics)

    return pd.DataFrame(results)


def compute_latency_stats(experiment_data: list) -> dict:
    """Compute latency and token consumption statistics from experiment data.

    Extracts latency and prompt token counts from API response metadata.

    Args:
        experiment_data: List of [pubid, question, api_response] entries.

    Returns:
        Dictionary with median_latency, mean_latency, median_prompt_tokens, n_queries.
    """
    latencies = []
    prompt_tokens = []

    for entry in experiment_data:
        res = entry[2]
        if not isinstance(res, dict) or "metadata" not in res:
            continue
        if "usage" not in res["metadata"]:
            continue

        usage = res["metadata"]["usage"]
        lat = usage.get("latency", 0)
        pt = usage.get("prompt_tokens", 0)

        if lat > 0:
            latencies.append(lat)
        if pt > 0:
            prompt_tokens.append(pt)

    if not latencies:
        return {"median_latency": 0, "mean_latency": 0, "median_prompt_tokens": 0, "n_queries": 0}

    return {
        "median_latency": float(np.median(latencies)),
        "mean_latency": float(np.mean(latencies)),
        "median_prompt_tokens": int(np.median(prompt_tokens)) if prompt_tokens else 0,
        "n_queries": len(latencies),
    }


# --- Statistical Analysis ---

def run_ttest_analysis(df: pd.DataFrame, metric: str, factors: list[str], alpha: float = 0.001) -> pd.DataFrame:
    """Run independent t-tests for each factor's effect on a metric.

    Args:
        df: DataFrame with experimental results.
        metric: Name of the metric column to analyse.
        factors: List of factor column names.
        alpha: Significance threshold.

    Returns:
        DataFrame with t-test results per factor.
    """
    results = []

    for factor in factors:
        levels = df[factor].unique()
        if len(levels) < 2:
            continue

        # Compare min-performing vs max-performing level
        group_means = df.groupby(factor)[metric].mean()
        min_level = group_means.idxmin()
        max_level = group_means.idxmax()

        group_min = df[df[factor] == min_level][metric]
        group_max = df[df[factor] == max_level][metric]

        t_stat, p_value = stats.ttest_ind(group_min, group_max)

        results.append({
            "Parameter": factor,
            "p-value": p_value,
            "Min Avg": group_means.min(),
            "Max Avg": group_means.max(),
            "Avg Diff": group_means.max() - group_means.min(),
            "Significant": p_value < alpha,
        })

    return pd.DataFrame(results)


def run_factorial_anova(df: pd.DataFrame, metric: str, factors: list[str],
                        interactions: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Run factorial ANOVA with main effects and interaction terms.

    Uses Type II sum of squares for unbalanced designs. Tests whether
    each factor and interaction significantly affects the metric.

    Args:
        df: DataFrame with experimental results.
        metric: Name of the metric column (dependent variable).
        factors: List of factor column names (independent variables).
        interactions: Optional list of (factor1, factor2) tuples for interaction terms.
                     Defaults to Pipeline×GenModel and Pipeline×Reranker.

    Returns:
        ANOVA table as DataFrame with F-statistics and p-values.
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    # Build formula: main effects + interactions
    main_effects = " + ".join(f"C({f})" for f in factors)

    if interactions is None:
        interactions = [
            (factors[0], factors[1]),  # Pipeline × GenModel
            (factors[0], factors[-1]),  # Pipeline × Reranker
        ]

    interaction_terms = " + ".join(f"C({a}):C({b})" for a, b in interactions)
    formula = f"{metric} ~ {main_effects} + {interaction_terms}"

    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)

    return anova_table


# --- Conversion Tables ---

PIPELINE_NAMES = {
    "vctr": "Vector",
    "txt": "Full Text",
    "hbrd": "Hybrid",
    "clbrt": "ColBERTv2",
    "udapr": "UDAPDR",
}

GENERATION_MODEL_NAMES = {
    "gpt4": "ChatGPT",
    "llama3": "LLaMA3",
    "nous": "Nous-Hermes",
}

CORPUS_NAMES = {
    "amalia": "VHT",
    "phages": "AMR",
}


# --- Main Analysis Pipeline ---

def main():
    parser = argparse.ArgumentParser(description="RAGEv Analysis Pipeline")
    parser.add_argument("--input", required=True, help="Path to merged results pickle file")
    parser.add_argument("--output", default="./results", help="Output directory for figures and tables")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load merged experimental results
    print(f"Loading results from {args.input}")
    df = pd.read_pickle(args.input)

    # Apply human-readable names
    if "Pipeline" in df.columns:
        df["Pipeline"] = df["Pipeline"].map(PIPELINE_NAMES).fillna(df["Pipeline"])
    if "Generation Model" in df.columns:
        df["Generation Model"] = df["Generation Model"].map(GENERATION_MODEL_NAMES).fillna(df["Generation Model"])
    if "Chunk Length" in df.columns:
        df["Chunk Length"] = df["Chunk Length"].map(CORPUS_NAMES).fillna(df["Chunk Length"])

    # Define analysis factors
    factors = ["Pipeline", "Generation Model", "N Retrieved Chunks", "Score Threshold", "Reranker"]

    # Compute t-test analysis for each corpus and metric
    for corpus in df["Chunk Length"].unique():
        corpus_df = df[df["Chunk Length"] == corpus]
        print(f"\n=== Analysis for {corpus} ===")

        # ROUGE-L analysis
        if "rougeL" in corpus_df.columns:
            ttest_results = run_ttest_analysis(corpus_df, "rougeL", factors)
            print(f"\nT-test results (Rouge-L, {corpus}):")
            print(ttest_results.to_string(index=False))
            ttest_results.to_csv(os.path.join(args.output, f"ttest_rougeL_{corpus}.csv"), index=False)

        # BERTScore F1 analysis
        if "bertscore_f1" in corpus_df.columns:
            ttest_results = run_ttest_analysis(corpus_df, "bertscore_f1", factors)
            print(f"\nT-test results (BERTScore F1, {corpus}):")
            print(ttest_results.to_string(index=False))
            ttest_results.to_csv(os.path.join(args.output, f"ttest_bertf1_{corpus}.csv"), index=False)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
