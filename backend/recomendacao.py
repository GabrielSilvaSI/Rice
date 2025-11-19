import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import ndarray

# Caminho para o dataset de itens
ITENS_PATH = "../datasets/filmes.csv"

# --- Variáveis Globais ---
df_filmes = None
tfidf_matrix = None
vectorizer = None


def criar_content_soup(row):
    """
    Função auxiliar para concatenar atributos relevantes em uma única string.

    Remove espaços e converte para minúsculas para tratar 'Ação' e 'ação' como o mesmo termo.
    """
    # Lista de colunas a serem incluídas na vetorização
    atributos = ['Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']

    # 1. Limpeza e Concatenação dos atributos categóricos
    # Remove espaços, converte para minúsculas e junta com espaços
    soup_list = []
    for attr in atributos:
        # Verifica se o valor não é nulo e é uma string
        if pd.notna(row[attr]) and isinstance(row[attr], str):
            # Substitui espaços por underscores (ex: 'Christopher Nolan' vira 'Christopher_Nolan')
            # Isso garante que 'Christopher' e 'Nolan' sejam tratados como um único termo (token)
            clean_attr = row[attr].replace(" ", "_").lower()
            soup_list.append(clean_attr)

    # 2. Adiciona a descrição rica (Overview) ao final, sem modificações de token
    # O Overview usa o espaço normal para que o TF-IDF o analise como texto corrido
    overview = row['Overview'] if pd.notna(row['Overview']) else ''

    # Junta os atributos limpos e o Overview
    return " ".join(soup_list) + " " + overview


def carregar_dados_e_vetorizar(caminho_csv: str = ITENS_PATH) -> tuple[pd.DataFrame, ndarray]:
    """
    Carrega o dataset, cria o 'Content Soup' e aplica a vetorização TF-IDF.
    """
    global df_filmes, tfidf_matrix, vectorizer

    print("Iniciando a carga de dados e vetorização...")

    try:
        # 1. Carregar o Dataset
        df_filmes = pd.read_csv(caminho_csv)

        # 2. Preparar e criar o 'Content Soup'
        df_filmes['Content_Soup'] = df_filmes.apply(criar_content_soup, axis=1)

        # 3. Inicializar o Vetorizador TF-IDF
        # Usamos stop_words e ngram_range (1, 2) para capturar termos de 1 e 2 palavras
        # Isso pode melhorar o reconhecimento de frases-chave.
        vectorizer = TfidfVectorizer(stop_words='portuguese', ngram_range=(1, 2))

        # 4. Aplicar o TF-IDF no Content_Soup
        tfidf_matrix = vectorizer.fit_transform(df_filmes['Content_Soup'])

        print(f"Vetorização concluída. Matriz TF-IDF gerada com forma: {tfidf_matrix.shape}")

        return df_filmes, tfidf_matrix

    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {caminho_csv}. Verifique o caminho.")
        return None, None
    except Exception as e:
        print(f"Ocorreu um erro durante a vetorização: {e}")
        return None, None