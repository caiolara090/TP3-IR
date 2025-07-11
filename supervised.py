import os
import csv
import logging
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split

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


def run_ltr_pipeline():
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

    train_queries_train, train_queries_val = train_test_split(
        train_queries, test_size=0.2, random_state=42
    )
    train_qrels_train = qrels[qrels["qid"].isin(train_queries_train["qid"])]
    train_qrels_val = qrels[qrels["qid"].isin(train_queries_val["qid"])]

    bm25f = pt.terrier.Retriever(
        index,
        wmodel="BM25F",
        num_results=HITS_PER_QUERY,
        controls={"w.0": 3, "w.1": 1, "w.2": 1, "c.0": 0.5, "c.1": 0.5, "c.2": 0.5},
        verbose=True,
    )
    pl2 = pt.terrier.Retriever(index, wmodel="PL2", num_results=HITS_PER_QUERY)
    tfidf = pt.terrier.Retriever(index, wmodel="TF_IDF", num_results=HITS_PER_QUERY)

    rerank_feats = tfidf**bm25f**pl2
    ltr_feats_pipeline = pt.rewrite.tokenise() >> bm25f >> rerank_feats

    print("Pipeline de recursos LTR:")
    print(ltr_feats_pipeline)

    lmart_l = lgb.LGBMRanker(
        task="train",
        min_data_in_leaf=10,
        min_sum_hessian_in_leaf=1,
        max_bin=511,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        importance_type="gain",
        num_iterations=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=8,
    )

    lmart_l_pipe = ltr_feats_pipeline >> pt.ltr.apply_learned_model(lmart_l, form="ltr")

    logging.info("Treinando modelo LambdaMART com LightGBM...")
    lmart_l_pipe.fit(
        train_queries_train, train_qrels_train, train_queries_val, train_qrels_val
    )

    logging.info("Avaliando desempenho no conjunto de treino...")
    results = lmart_l_pipe.transform(train_queries)
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
    test_results = lmart_l_pipe.transform(test_queries)
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
    run_ltr_pipeline()
