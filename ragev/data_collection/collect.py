"""
RAGEv Data Collection: Experiment Runner.

Executes the full factorial experimental design by querying the RAG platform
for each question in the dataset under each pipeline configuration.
Results are stored as pickle files for subsequent analysis.

Usage:
    python -m ragev.data_collection.collect
"""
import os
import pickle
import time

from loguru import logger as l
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing.dummy import Pool

from ragev.utils import get_vast_answer
from ragev.config import settings
from ragev.doe import create_fname, create_config, load_dataset, get_experiments


# Prompt template: appended to each dataset question
PROMPT_TEMPLATE = "{question} Answer providing an explanation."


def do_job(params: tuple) -> tuple[int, float]:
    """Execute a single experimental configuration on all questions.

    Args:
        params: Tuple of (job_id, config, output_path, log_path, questions, api_key, user).

    Returns:
        Tuple of (job_id, duration_seconds).
    """
    jid, exp_config, fname, job_log_fname, qsts, api_key, user = params

    # Load cached results if experiment was partially completed
    df_loaded = pd.DataFrame(columns=["pubid", "qst", "answer"])
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            loaded = pickle.load(f)
            df_loaded = pd.DataFrame(loaded, columns=["pubid", "qst", "answer"])

    results = []
    start = time.time()

    try:
        l.info(f"Process {jid} starting {os.path.basename(fname)}")

        for pubid, qst in tqdm(qsts):
            # Check if answer already cached
            if not df_loaded.empty and df_loaded["qst"].eq(qst).any():
                cached_row = df_loaded[df_loaded["qst"] == qst].head(1)
                res = cached_row.iloc[0].to_dict()["answer"]
            else:
                res = get_vast_answer(qst, api_key, user, exp_config)

            results.append([pubid, qst, res])

            # Save progress incrementally
            if "answer" in res:
                with open(job_log_fname, "a") as fp:
                    fp.write(f"-----\n {qst}\n {res['answer']} \n-----\n")
                with open(fname, "wb") as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        l.error(f"Error in job {jid}: {e}")

    duration = time.time() - start
    return jid, duration


def main():
    """Run the full experimental pipeline."""
    l.info("Loading dataset")
    # Select dataset: 'ds-VHT.csv' or 'ds-AMR.csv' (or 'ds-PubMedQA_sample100.csv')
    # Download from: https://data.jrc.ec.europa.eu/dataset/6e8b187d-7bf3-43ae-80a4-f94d5413cfaa
    ppr_lst, df = load_dataset("ds-VHT.csv")

    # Prepare questions with prompt template
    # Dataset columns: pubid, question, context, answer [, final_decision]
    qsts = [
        (row["pubid"], PROMPT_TEMPLATE.format(question=row["question"]))
        for _, row in df.iterrows()
    ]

    # Generate all experimental configurations
    exps = get_experiments()
    n_exps = len(exps)

    # Identify experiments that still need to be run
    jobs = []
    for i, exp in enumerate(exps):
        exp_conf = create_config(*exp)
        b_fname, l_fname = create_fname(exp)
        cached_fname = os.path.join(settings.EXPDIR_BINS, b_fname)
        log_fname = os.path.join(settings.EXPDIR_LOGS, l_fname)

        # Check if experiment is already complete
        is_complete = False
        if os.path.exists(cached_fname):
            with open(cached_fname, "rb") as f:
                loaded = pickle.load(f)
                is_complete = len(loaded) >= len(qsts)

        status = "DONE" if is_complete else "TODO"
        l.info(f"{i} [{status}] {exp}")

        if not is_complete:
            jobs.append(
                (i, exp_conf, cached_fname, log_fname, qsts, settings.API_KEY, settings.USER)
            )

    n_jobs = len(jobs)
    l.info(f"Total: {n_exps} experiments, {n_exps - n_jobs} done, {n_jobs} to run")

    # Execute experiments
    l.info(f"Launching {n_jobs} jobs on {settings.MAX_THREADS} thread(s)")
    p = Pool(settings.MAX_THREADS)
    for jid, duration in p.imap_unordered(do_job, jobs):
        l.info(f"Process {jid} finished after {duration:.2f} seconds")
    p.close()
    p.join()


if __name__ == "__main__":
    main()
