"""
RAGEv Data Analysis: Evaluation Metrics and Statistical Analysis.

Computes ROUGE scores, BERTScore, classification metrics, SHAP-based
waterfall analysis, and statistical significance tests for the
multifactorial RAG evaluation.

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
