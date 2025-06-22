import json
import logging
import time
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_CORPUS_PATH = "old_data/corpus.jsonl"
INDEX_DIR = "./indexes/terrier"


def process_line(line):
    if not line.strip():
        return None
    try:
        data = json.loads(line)
        doc_id = data.get("id")
        title = data.get("title", "")
        text = data.get("text", "")
        keywords = data.get("keywords", [])
        return {
            "docno": doc_id,
            "title": title,
            "text": text,
            "keywords": " ".join(keywords),
        }
    except json.JSONDecodeError:
        return None


def preprocess_corpus_to_df():
    start = time.time()
    logging.info("Iniciando leitura e pré-processamento do corpus...")

    with open(INPUT_CORPUS_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with Pool(processes=cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_line, lines, chunksize=100),
                total=len(lines),
            )
        )

    docs = [doc for doc in results if doc is not None]
    df = pd.DataFrame(docs)
    logging.info(
        f"{len(df)} documentos pré-processados em {time.time() - start:.2f} segundos."
    )
    return df


def create_index(df):
    logging.info("Iniciando criação do índice...")
    start = time.time()
    indexer = pt.IterDictIndexer(
        INDEX_DIR,
        text_attrs=["title", "keywords", "text"],
        meta={"docno": 10, "title": 50, "keywords": 200, "text": 1000},
        fields=True,
        threads=8,
    )
    indexref = indexer.index(df.to_dict(orient="records"))
    logging.info(
        f"Index criado em {time.time() - start:.2f} segundos. Caminho: {INDEX_DIR}"
    )
    return indexref


if __name__ == "__main__":
    start_total = time.time()
    df = preprocess_corpus_to_df()
    create_index(df)
    logging.info(
        f"Execução total finalizada em {time.time() - start_total:.2f} segundos."
    )
