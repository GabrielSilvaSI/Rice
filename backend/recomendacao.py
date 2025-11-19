import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import ndarray, mean, array
from sklearn.metrics.pairwise import linear_kernel

# Caminho para o dataset de itens
ITENS_PATH = "../datasets/filmes.csv"
AVALIACOES_PATH = "../datasets/avaliacoes.csv"

# --- Variáveis Globais ---
df_filmes = None
tfidf_matrix = None
vectorizer = None

# FATOR DE PONDERAÇÃO: Define o quanto a aversão penaliza o perfil.
# Lambda (λ) = 0.5 significa que o 'dislike' tem metade do peso do 'gosto'.
FATOR_PONDERACAO = 0.5

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
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

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


def _obter_vetores_por_avaliacao(
        usuario_id: int,
        df_avaliacoes: pd.DataFrame,
        df_itens: pd.DataFrame,
        tfidf_matriz: ndarray,
        avaliacao_alvo: int
) -> ndarray | None:
    """
    Função auxiliar para extrair vetores com uma avaliação específica (0 ou 1).

    Atenção: Assume que 'filme_id' no avaliacoes.csv CORRESPONDE ao índice do item no filmes.csv.
    """

    # Filtrar avaliações pelo usuário e pelo valor de avaliação (0 ou 1)
    avaliacoes_alvo = df_avaliacoes[
        (df_avaliacoes['usuario_id'] == usuario_id) &
        (df_avaliacoes['avaliacao'] == avaliacao_alvo)
        ]

    if avaliacoes_alvo.empty:
        return None

    # Mapeamento e Extração de Índices

    # 1. Obtém os IDs de filme (que são os índices)
    indices_preferidos = avaliacoes_alvo['filme_id'].tolist()

    # 2. Verifica se os índices estão dentro do limite da matriz
    # (O número máximo de índice é len(df_itens) - 1, que é o número de linhas - 1)
    indices_validos = [
        idx
        for idx in indices_preferidos
        if 0 <= idx < len(df_itens)
    ]

    if not indices_validos:
        # Isso ocorre se os IDs no avaliacoes.csv forem muito altos ou inválidos
        print(
            f"Nenhum índice de filme válido encontrado no catálogo para o Usuário {usuario_id} e Avaliação {avaliacao_alvo}.")
        return None

    # Extração e Média: Retorna a média dos vetores
    # tfidf_matriz[indices_validos] extrai as linhas diretamente pelos índices.
    vetores_alvo = tfidf_matriz[indices_validos]

    return mean(vetores_alvo, axis=0)

def construir_perfil_usuario(usuario_id: int, df_itens: pd.DataFrame, tfidf_matriz: ndarray) -> ndarray | None:
    """
    Constrói o Perfil do Usuário como Perfil Positivo - (λ * Perfil Negativo).
    """

    try:
        df_avaliacoes = pd.read_csv(AVALIACOES_PATH)

        # 1. Obter o Perfil Positivo (Gostou = 1)
        perfil_positivo = _obter_vetores_por_avaliacao(
            usuario_id, df_avaliacoes, df_itens, tfidf_matriz, avaliacao_alvo=1
        )

        if perfil_positivo is None:
            print(f"Usuário {usuario_id} não tem avaliações positivas suficientes. Não é possível gerar o perfil.")
            return None

        # 2. Obter o Perfil Negativo (Não Gostou = 0)
        perfil_negativo = _obter_vetores_por_avaliacao(
            usuario_id, df_avaliacoes, df_itens, tfidf_matriz, avaliacao_alvo=0
        )

        # 3. Cálculo do Perfil Final Ponderado
        if perfil_negativo is not None:
            # Ponderação: Subtrai a média dos vetores negativos, aplicando o fator λ
            # O array vazio (sparse matrix) precisa ser convertido para denso
            perfil_final = array(perfil_positivo) - (FATOR_PONDERACAO * array(perfil_negativo))
            print(f"Perfil refinado para o Usuário {usuario_id} usando avaliações negativas (λ={FATOR_PONDERACAO}).")
        else:
            # Se não houver avaliações negativas, usa-se apenas o perfil positivo
            perfil_final = perfil_positivo
            print(f"Perfil simples construído para o Usuário {usuario_id} (sem avaliações negativas).")

        return perfil_final.reshape(1,
                                    -1)  # Garante que o vetor de perfil tenha a forma (1, N) para o próximo passo (Similaridade)

    except FileNotFoundError:
        print(f"ERRO: Arquivo de avaliações não encontrado em {AVALIACOES_PATH}.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro na construção do perfil: {e}")
        return None


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
    # linear_kernel(A, B) calcula o produto escalar de todas as linhas em A com todas as linhas em B.
    # Como os vetores TF-IDF são implicitamente normalizados, o resultado é a Similaridade do Cosseno.
    # O perfil_usuario deve ser de forma (1, N) e a tfidf_matriz de forma (M, N).

    # O resultado é uma matriz (1, M) onde M é o número de filmes.
    similaridade_scores = linear_kernel(perfil_usuario, tfidf_matriz)

    # 2. Obter a Lista de Scores
    # Achata a matriz de (1, M) para um array simples (M)
    scores = similaridade_scores.flatten()

    # 3. Mapear os Scores para os Filmes

    # Obtém os índices dos filmes que têm as maiores pontuações de similaridade.
    # [::-1] inverte a ordem para que o maior score venha primeiro.
    indices_ordenados = scores.argsort()[::-1]

    # 4. Excluir Filmes já Vistos/Preferidos (Opcional, mas Recomendado)
    # Por enquanto, focaremos na similaridade. Em uma versão final, você faria um filtro
    # para não recomendar filmes que o usuário já assistiu e gostou.

    # 5. Gerar a Lista Final de Recomendações
    recomendacoes = []

    # Pega os N filmes com maior score (excluindo o índice 0, que geralmente é o próprio perfil se não for filtrado)
    for i in indices_ordenados:
        if len(recomendacoes) >= num_recomendacoes:
            break

        # Obtém o título do filme e o score
        titulo = df_itens.iloc[i]['Series_Title']
        score = scores[i]

        # Adiciona à lista (Título do Filme, Pontuação de Similaridade)
        recomendacoes.append((titulo, score))

    print(f"Geração de recomendações concluída. {len(recomendacoes)} filmes encontrados.")
    return recomendacoes


if __name__ == '__main__':
    # --- Bloco de Teste Unificado ---
    # 1. Vetorização (Passo 1)
    df_filmes_global, tfidf_matrix_global = carregar_dados_e_vetorizar()

    if tfidf_matrix_global is not None:
        TEST_USER_ID = 1

        # 2. Construção do Perfil (Passo 2 - Lógica Ponderada)
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
                # Formata a pontuação para 4 casas decimais
                score_formatado = f"{score:.4f}"
                print(f"{rank + 1}: {titulo} | Similaridade: {score_formatado}")