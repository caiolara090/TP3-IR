import csv
import logging
import os
from pyserini.search.lucene import LuceneSearcher

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuração de Caminhos ---
INDEX_DIR = 'indexes/corpus'
TEST_QUERIES_PATH = 'data/test_queries.csv'
SUBMISSION_OUTPUT_PATH = 'results/kaggle_submission.csv'

# --- Parâmetros de Busca ---
HITS_PER_QUERY = 100

def run_queries_for_kaggle():
    """
    Lê as queries, busca no índice Pyserini e salva os resultados no formato de submissão Kaggle.
    """
    if not os.path.exists(INDEX_DIR):
        logging.error(f"Diretório do índice '{INDEX_DIR}' não encontrado.")
        return

    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)

    try:
        searcher = LuceneSearcher(INDEX_DIR)
        logging.info(f"Buscador inicializado com sucesso a partir do índice em '{INDEX_DIR}'.")
    except Exception as e:
        logging.error(f"Falha ao inicializar o LuceneSearcher: {e}")
        return

    try:
        with open(TEST_QUERIES_PATH, 'r', encoding='utf-8') as f_queries, \
             open(SUBMISSION_OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f_submission:

            reader = csv.reader(f_queries)
            header = next(reader, None)  # Pula o cabeçalho

            writer = csv.writer(f_submission)
            writer.writerow(['QueryId', 'EntityId'])  # Cabeçalho Kaggle

            query_count = 0
            for row in reader:
                if not row:
                    continue
                query_id, query_text = row

                hits = searcher.search(query_text, k=HITS_PER_QUERY)

                for hit in hits:
                    writer.writerow([query_id, hit.docid])

                query_count += 1
                if query_count % 20 == 0:
                    logging.info(f"Processadas {query_count} queries...")

            logging.info(f"Busca concluída. {query_count} queries foram processadas.")
            logging.info(f"Arquivo de submissão salvo em: {SUBMISSION_OUTPUT_PATH}")

    except FileNotFoundError:
        logging.error(f"ERRO: O arquivo de queries '{TEST_QUERIES_PATH}' não foi encontrado.")
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado durante o processamento das queries: {e}")

if __name__ == '__main__':
    run_queries_for_kaggle()