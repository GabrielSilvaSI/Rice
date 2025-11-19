import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from numpy import ndarray, mean, array

# Caminhos para os datasets
ITENS_PATH = "../datasets/filmes.csv"
AVALIACOES_PATH = "../datasets/avaliacoes.csv"

# --- Variáveis Globais (Carregadas uma única vez) ---
df_filmes = None
tfidf_matrix = None
vectorizer = None


# ----------------------------------------------------------------------
# FUNÇÕES DE UTILIDADE (Para referências externas, não pedidas no prompt)
# Estas funções devem existir no seu arquivo para o código abaixo funcionar!
# ----------------------------------------------------------------------

def criar_content_soup(row):
    """Função auxiliar para concatenar atributos de conteúdo."""
    atributos = ['Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']
    soup_list = []
    for attr in atributos:
        if pd.notna(row[attr]) and isinstance(row[attr], str):
            clean_attr = row[attr].replace(" ", "_").lower()
            soup_list.append(clean_attr)
    overview = row['Overview'] if pd.notna(row['Overview']) else ''
    return " ".join(soup_list) + " " + overview


def carregar_dados_e_vetorizar(caminho_csv: str = ITENS_PATH) -> tuple[pd.DataFrame, ndarray]:
    """Carrega o catálogo, cria o Content Soup e aplica a vetorização TF-IDF."""
    global df_filmes, tfidf_matrix, vectorizer
    df_filmes = pd.read_csv(caminho_csv)
    df_filmes['Content_Soup'] = df_filmes.apply(criar_content_soup, axis=1)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df_filmes['Content_Soup'])
    print(f"Vetorização concluída. Matriz TF-IDF gerada com forma: {tfidf_matrix.shape}")
    return df_filmes, tfidf_matrix


def construir_perfil_usuario(usuario_id: int, df_itens: pd.DataFrame, tfidf_matriz: ndarray) -> ndarray | None:
    """Constrói o perfil do usuário como a média dos vetores dos itens preferidos (avaliação = 1)."""
    try:
        df_avaliacoes = pd.read_csv(AVALIACOES_PATH)
        avaliacoes_positivas = df_avaliacoes[
            (df_avaliacoes['usuario_id'] == usuario_id) &
            (df_avaliacoes['avaliacao'] == 1)
            ]
        if avaliacoes_positivas.empty: return None

        indices_preferidos = avaliacoes_positivas['filme_id'].tolist()
        indices_validos = [idx for idx in indices_preferidos if 0 <= idx < len(df_itens)]
        if not indices_validos: return None

        vetores_preferidos = tfidf_matriz[indices_validos]
        perfil_usuario = mean(vetores_preferidos, axis=0)

        # GARANTIA DE NDARRAY (SOLUÇÃO DO TYPERROR)
        # O perfil precisa ser um ndarray para o linear_kernel funcionar corretamente com a matriz esparsa.
        perfil_usuario_array = perfil_usuario.getA() if hasattr(perfil_usuario, 'getA') else perfil_usuario

        print(f"Perfil simples construído para o Usuário {usuario_id}. Vetor de forma: {perfil_usuario_array.shape}")

        return perfil_usuario_array.reshape(1, -1)  # Forma (1, N)
    except Exception as e:
        print(f"Ocorreu um erro na construção do perfil: {e}")
        return None


# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL: GERAR RECOMENDAÇÕES (PASSO 3)
# ----------------------------------------------------------------------

def gerar_recomendacoes(
        perfil_usuario: ndarray,
        df_itens: pd.DataFrame,
        tfidf_matriz: ndarray,
        num_recomendacoes: int = 10
) -> list[tuple[str, float]]:
    """
    Calcula a Similaridade do Cosseno entre o Perfil do Usuário e todos os itens
    no catálogo e retorna os mais similares.
    """

    if perfil_usuario is None:
        return []

    print("Calculando Similaridade do Cosseno...")

    # 1. Cálculo da Similaridade do Cosseno
    # linear_kernel calcula a similaridade entre o vetor de perfil (1, N) e a matriz de itens (M, N).
    # O resultado é um array (1, M) de pontuações de similaridade.

    # PERFIL_USUARIO já é garantido como ndarray no Passo 2
    similaridade_scores = linear_kernel(perfil_usuario, tfidf_matriz)

    # 2. Obter a Lista de Scores
    scores = similaridade_scores.flatten()

    # 3. Mapear os Scores para os Filmes
    # Obtém os índices dos filmes que têm as maiores pontuações. [::-1] inverte (maior primeiro).
    indices_ordenados = scores.argsort()[::-1]

    # 4. Gerar a Lista Final de Recomendações
    recomendacoes = []

    # Loop pelos índices ordenados
    for i in indices_ordenados:

        # O filme mais similar (índice 0) é frequentemente um dos filmes que o usuário já assistiu.
        # Filtros mais robustos seriam aplicados aqui para excluir itens já vistos.

        # Se atingiu o número desejado de recomendações, pare.
        if len(recomendacoes) >= num_recomendacoes:
            break

        # Pega o título do filme e o score
        titulo = df_itens.iloc[i]['Series_Title']
        score = scores[i]

        # Adiciona à lista (Título do Filme, Pontuação de Similaridade)
        recomendacoes.append((titulo, score))

    print(f"Geração de recomendações concluída. {len(recomendacoes)} filmes encontrados.")
    return recomendacoes


if __name__ == '__main__':
    # --- Bloco de Teste Unificado ---
    print("\nIniciando a carga de dados e vetorização...")

    # 1. Vetorização (Passo 1)
    df_filmes_global, tfidf_matrix_global = carregar_dados_e_vetorizar()

    if tfidf_matrix_global is not None:
        TEST_USER_ID = 1

        # 2. Construção do Perfil (Passo 2)
        perfil = construir_perfil_usuario(
            usuario_id=TEST_USER_ID,
            df_itens=df_filmes_global,
            tfidf_matriz=tfidf_matrix_global
        )

        if perfil is not None:
            # 3. Geração de Recomendações (Passo 3)
            lista_recomendada = gerar_recomendacoes(
                perfil_usuario=perfil,
                df_itens=df_filmes_global,
                tfidf_matriz=tfidf_matrix_global,
                num_recomendacoes=5
            )

            print(f"\n✅ RECOMENDAÇÕES PARA O USUÁRIO {TEST_USER_ID} (TOP 5):")
            for rank, (titulo, score) in enumerate(lista_recomendada):
                score_formatado = f"{score:.4f}"
                print(f"{rank + 1}: {titulo} | Similaridade: {score_formatado}")