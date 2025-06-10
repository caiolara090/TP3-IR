import json
import os
import subprocess
import logging

# Configura o logging para exibir informações úteis
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuração de Caminhos ---
# Certifique-se de que o corpus.jsonl está neste diretório
INPUT_CORPUS_PATH = 'old_data/corpus.jsonl' 

# Diretório para armazenar os arquivos pré-processados para o Pyserini
PREPROCESSED_DIR = 'data/preprocessed_corpus'

# Arquivo de saída para os dados pré-processados
PREPROCESSED_FILE_PATH = os.path.join(PREPROCESSED_DIR, 'preprocessed_corpus.jsonl')

# Diretório onde o índice do Pyserini será salvo
INDEX_DIR = 'indexes/corpus'

def preprocess_corpus():
    """
    Lê o arquivo corpus.jsonl original, transforma-o para o formato esperado pelo Pyserini
    e o salva em um novo arquivo .jsonl.

    O formato esperado pelo Pyserini é uma coleção de objetos JSON, cada um com uma chave 'id' e 'contents'.
    """
    logging.info(f"Iniciando pré-processamento do arquivo: {INPUT_CORPUS_PATH}")
    
    # Cria o diretório de saída se ele não existir
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    count = 0
    try:
        with open(INPUT_CORPUS_PATH, 'r', encoding='utf-8') as infile, \
             open(PREPROCESSED_FILE_PATH, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                
                try:
                    # Carrega o objeto JSON da linha
                    original_doc = json.loads(line)
                    data = {}

                    # Lógica corrigida para encontrar o dicionário com os dados
                    # Verifica se o 'id' está no nível principal, caso contrário, procura dentro de 'root'
                    if 'id' in original_doc:
                        data = original_doc
                    else:
                        data = original_doc.get('root', {})

                    doc_id = data.get('id')
                    title = data.get('title', '')
                    text = data.get('text', '')
                    keywords = data.get('keywords', [])
                    
                    if not doc_id:
                        logging.warning(f"Documento na linha {count+1} não possui 'id'. Pulando.")
                        continue

                    # Combina os campos de texto para formar o 'contents'.
                    # Dar mais peso ao título e às palavras-chave repetindo-os é uma boa estratégia.
                    contents = ' '.join([title] * 3) + ' ' + ' '.join(keywords) + ' ' + text
                    
                    # Cria o novo documento no formato Pyserini
                    pyserini_doc = {
                        "id": doc_id,
                        "contents": contents.strip()
                    }
                    
                    # Escreve o novo documento como uma linha no arquivo de saída
                    outfile.write(json.dumps(pyserini_doc) + '\n')
                    
                    count += 1
                    if count % 100000 == 0:
                        logging.info(f"Processados {count} documentos...")
                        
                except json.JSONDecodeError:
                    logging.error(f"Erro de decodificação JSON na linha: {count+1}. Pulando linha.")
                    continue

    except FileNotFoundError:
        logging.error(f"ERRO: O arquivo de entrada '{INPUT_CORPUS_PATH}' não foi encontrado.")
        logging.error("Por favor, certifique-se de que o arquivo 'corpus.jsonl' está no diretório 'old_data/'.")
        return False

    logging.info(f"Pré-processamento concluído. {count} documentos foram processados e salvos em {PREPROCESSED_FILE_PATH}")
    return True

def create_index():
    """
    Executa o comando de indexação do Pyserini usando o corpus pré-processado.
    """
    logging.info("Iniciando a criação do índice com Pyserini...")
    
    # Verifica se o diretório do índice já existe e não está vazio para evitar reindexação
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        logging.warning(f"O diretório do índice '{INDEX_DIR}' já existe e contém arquivos. Pulando a etapa de indexação.")
        logging.info("Se você deseja recriar o índice, exclua o diretório 'indexes/corpus' e execute o script novamente.")
        return

    # Comando Pyserini para indexar a coleção JSON pré-processada
    command = [
        'python3', '-m', 'pyserini.index',
        '--collection', 'JsonCollection',
        '--input', PREPROCESSED_DIR,
        '--index', INDEX_DIR,
        '--generator', 'DefaultLuceneDocumentGenerator',
        '--threads', '8',
        '--storePositions', '--storeDocvectors', '--storeRaw'
    ]

    try:
        # Executa o comando Pyserini
        subprocess.run(command, check=True)
        logging.info(f"Indexação concluída com sucesso. Índice salvo em '{INDEX_DIR}'")
    except subprocess.CalledProcessError as e:
        logging.error(f"Ocorreu um erro durante a indexação do Pyserini: {e}")
        logging.error("Verifique se o Pyserini está instalado corretamente ('pip install pyserini') e se o Java (JDK) está instalado e configurado.")
    except FileNotFoundError:
        logging.error("Erro: O comando 'python' não foi encontrado. Certifique-se de que o Python está no seu PATH.")


if __name__ == '__main__':
    # Etapa 1: Pré-processar o corpus
    #if preprocess_corpus():
        # Etapa 2: Criar o índice
        create_index()