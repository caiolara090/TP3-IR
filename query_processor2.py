import os
import csv
import logging
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import datetime
from pyterrier_t5 import MonoT5ReRanker

pt.java.init()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

now_str = datetime.datetime.now().strftime("%m_%d_%H.%M")
SUBMISSION_OUTPUT_PATH = f"results/kaggle_submission_{now_str}.csv"

INDEX_DIR = os.path.abspath("indexes/terrier")
TRAIN_QUERIES_PATH = "data/train_queries.csv"
TRAIN_QRELS_PATH = "data/train_qrels.csv"
TEST_QUERIES_PATH = "data/test_queries.csv"
HITS_PER_QUERY = 1000


def concat_fields(df):
    df["doc_text"] = (
        "Title: " + df["title"].fillna("") + " "
        + "Body: " + df["text"].fillna("") + " "
        + "Keywords: " + df["keywords"].fillna("")
    )
    return df


def run_pipeline():
    index = pt.IndexFactory.of(INDEX_DIR)

    train_queries = pd.read_csv(TRAIN_QUERIES_PATH)
    train_queries["Query"] = train_queries["Query"]
    train_queries = train_queries.rename(columns={"QueryId": "qid", "Query": "query"})
    train_queries["qid"] = train_queries["qid"].astype(str)

    qrels = pd.read_csv(TRAIN_QRELS_PATH)
    qrels = qrels.rename(
        columns={"QueryId": "qid", "EntityId": "docno", "Relevance": "label"}
    )
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)

    bm25f = pt.terrier.Retriever(
        index,
        wmodel="BM25F",
        metadata=["docno", "title", "keywords", "text"],
        controls={"w.0": 3, "w.1": 1, "w.2": 1, "c.0": 0.5, "c.1": 0.5, "c.2": 0.5},
        verbose=True,
    )

    pipeline = (
        pt.rewrite.tokenise()
        >> bm25f % HITS_PER_QUERY
        >> pt.apply.generic(concat_fields)
        >> MonoT5ReRanker(text_field='doc_text')
    )

    logging.info("Avaliando desempenho no conjunto de treino...")
    results = pipeline.transform(train_queries)
    eval_metrics = pt.Utils.evaluate(
        results, qrels, metrics=["map", "ndcg", "recall", "P.10"]
    )
    for metric, value in eval_metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    test_queries = pd.read_csv(TEST_QUERIES_PATH)
    test_queries["Query"] = test_queries["Query"]
    test_queries = test_queries.rename(columns={"QueryId": "qid", "Query": "query"})
    test_queries["qid"] = test_queries["qid"].astype(str)

    logging.info("Executando pipeline no conjunto de teste...")
    test_results = pipeline.transform(test_queries)
    test_results = test_results.groupby("qid").head(100)

    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)
    with open(SUBMISSION_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["QueryId", "EntityId"])
        for _, row in tqdm(
            test_results.iterrows(), total=len(test_results), desc="Gerando submissão"
        ):
            writer.writerow([str(row["qid"]).zfill(3), str(row["docno"]).zfill(7)])

    logging.info(f"Submissão salva em {SUBMISSION_OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()
