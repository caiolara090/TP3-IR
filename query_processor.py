import os
import csv
import logging
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from numpy import arange

nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

pt.init()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [STEMMER.stem(t) for t in tokens if t.isalnum() and t not in STOPWORDS]
    return ' '.join(filtered)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INDEX_DIR = os.path.abspath('indexes/terrier_old')
TRAIN_QUERIES_PATH = 'data/train_queries.csv'
TRAIN_QRELS_PATH = 'data/train_qrels.csv'
TEST_QUERIES_PATH = 'data/test_queries.csv'
SUBMISSION_OUTPUT_PATH = 'results/kaggle_submission_ltr.csv'
HITS_PER_QUERY = 150

def run_ltr_pipeline():
    index = pt.IndexFactory.of(INDEX_DIR)
    
    # Pré-processa queries e qrels de treino
    train_queries = pd.read_csv(TRAIN_QUERIES_PATH)
    train_queries["Query"] = train_queries["Query"].apply(preprocess_text)
    train_queries = train_queries.rename(columns={"QueryId": "qid", "Query": "query"})
    train_queries["qid"] = train_queries["qid"].astype(str)

    qrels = pd.read_csv(TRAIN_QRELS_PATH)
    qrels = qrels.rename(columns={"QueryId": "qid", "EntityId": "docno", "Relevance": "label"})
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)

    # Divide treino/validação
    train_queries_train, train_queries_val = train_test_split(train_queries, test_size=0.2, random_state=42)
    train_qrels_train = qrels[qrels['qid'].isin(train_queries_train['qid'])]
    train_qrels_val = qrels[qrels['qid'].isin(train_queries_val['qid'])]

    #bm25_base = pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.b" : 0.75, "bm25.k_1": 0.75, "bm25.k_3": 0.75}, num_results=HITS_PER_QUERY)

    bm25_base = pt.terrier.Retriever(index, wmodel="BM25", num_results=HITS_PER_QUERY)

    # bm25_gs = pt.GridSearch(
    #     bm25_base,
    #     {bm25_base: {"bm25.b"  : [0.01, 0.025, 0.05, 0.1],
    #         "bm25.k_1": [1.2, 1.5],
    #         "bm25.k_3": [0.05, 0.1, 0.15, 0.25, 0.5, 0.75]
    # }},
    #     train_queries_val,
    #     train_qrels_val,
    #     'ndcg',
    #     jobs=4
    # )
    # try:
    #     print(bm25_base)
    #     logging.info(f"Melhores parâmetros BM25: {bm25_base}")
    # except Exception as e:
    #     print(f"Erro ao obter melhores parâmetros: {e}")

    bm25 = bm25_base
    
    # bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=HITS_PER_QUERY)
    tfidf = pt.terrier.Retriever(index, wmodel="TF_IDF", num_results=HITS_PER_QUERY)
    pl2 = pt.terrier.Retriever(index, wmodel="PL2", num_results=HITS_PER_QUERY)
    rerank_feats = tfidf ** pl2 ** bm25
    ltr_feats_pipeline = bm25 >> rerank_feats

    # Modelo

    lmart_l = lgb.LGBMRanker(
        task="train",
        min_data_in_leaf=10,
        min_sum_hessian_in_leaf=1,
        max_bin=511,
        num_leaves=31,
        max_depth=20,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        learning_rate=0.2,
        importance_type="gain",
        num_iterations=25000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    
    lmart_l_pipe = ltr_feats_pipeline >> pt.ltr.apply_learned_model(lmart_l, form="ltr")

    logging.info("Treinando modelo LambdaMART com LightGBM...")
    lmart_l_pipe.fit(train_queries_train, train_qrels_train, train_queries_val, train_qrels_val)

    # Avaliação no conjunto de treino
    logging.info("Avaliando desempenho no conjunto de treino...")
    results = lmart_l_pipe.transform(train_queries)
    eval_metrics = pt.Utils.evaluate(results, qrels, metrics=["map", "ndcg", "recall", "P.10"])
    for metric, value in eval_metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    # Carrega e pré-processa queries de teste
    test_queries = pd.read_csv(TEST_QUERIES_PATH)
    test_queries["Query"] = test_queries["Query"].apply(preprocess_text)
    test_queries = test_queries.rename(columns={"QueryId": "qid", "Query": "query"})
    test_queries["qid"] = test_queries["qid"].astype(str)

    # Geração da submissão
    logging.info("Executando pipeline no conjunto de teste...")
    test_results = lmart_l_pipe.transform(test_queries)
    test_results = test_results.groupby("qid").head(100)

    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)
    with open(SUBMISSION_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["QueryId", "EntityId"])
        for _, row in tqdm(test_results.iterrows(), total=len(test_results), desc="Gerando submissão"):
            writer.writerow([str(row["qid"]).zfill(3), str(row["docno"]).zfill(7)])

    logging.info(f"Submissão salva em {SUBMISSION_OUTPUT_PATH}")

    # Avaliação comparativa
    test_qrels = qrels[qrels["qid"].isin(test_queries["qid"])]
    pt.Experiment(
        [bm25, lmart_l_pipe],
        train_queries_val,
        train_qrels_val,
        ["map", "ndcg", "P.10"],
        names=["BM25", "LambdaMART-XGB", "LambdaMART-LightGBM"]
    ).to_markdown(index=False)

if __name__ == '__main__':
    run_ltr_pipeline()
