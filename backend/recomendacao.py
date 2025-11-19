import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from numpy import ndarray, mean
from collections.abc import Collection

# Caminhos para os datasets
ITENS_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'filmes.csv')
AVALIACOES_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'avaliacoes.csv')
USUARIOS_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'usuarios.csv')

# --- Variáveis Globais (Carregadas uma única vez) ---
df_filmes = None
tfidf_matrix = None
vectorizer = None


# ----------------------------------------------------------------------
# FUNÇÕES DE UTILIDADE
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
        if not os.path.exists(AVALIACOES_PATH): return None
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
        perfil_usuario_array = perfil_usuario.getA() if hasattr(perfil_usuario, 'getA') else perfil_usuario

        return perfil_usuario_array.reshape(1, -1)
    except Exception as e:
        print(f"Ocorreu um erro na construção do perfil: {e}")
        return None


def gerar_recomendacoes(
        perfil_usuario: ndarray,
        df_itens: pd.DataFrame,
        tfidf_matriz: ndarray,
        num_recomendacoes: int = 10
) -> list[tuple[str, float]]:
    """Calcula a Similaridade do Cosseno e retorna os filmes mais similares."""
    if perfil_usuario is None: return []

    similaridade_scores = linear_kernel(perfil_usuario, tfidf_matriz).flatten()
    indices_ordenados = similaridade_scores.argsort()[::-1]

    recomendacoes = []
    for i in indices_ordenados:
        if len(recomendacoes) >= num_recomendacoes: break
        titulo = df_itens.iloc[i]['Series_Title']
        score = similaridade_scores[i]
        recomendacoes.append((titulo, score))

    return recomendacoes


def salvar_avaliacao(usuario_id: int, filme_id: int, avaliacao: int) -> bool:
    """Salva uma nova avaliação no arquivo avaliacoes.csv."""
    novo_registro = pd.DataFrame([{'usuario_id': usuario_id, 'filme_id': filme_id, 'avaliacao': avaliacao}])
    try:
        file_exists = os.path.isfile(AVALIACOES_PATH)
        novo_registro.to_csv(AVALIACOES_PATH, mode='a', index=False, header=not file_exists)
        return True
    except Exception as e:
        print(f"ERRO ao salvar avaliação: {e}")
        return False


def carregar_e_listar_usuarios():
    """Carrega nomes de usuarios.csv e complementa com IDs de avaliacoes.csv."""
    users = {}
    if os.path.exists(USUARIOS_PATH):
        try:
            df_nomes = pd.read_csv(USUARIOS_PATH).drop_duplicates(subset=['usuario_id'], keep='last')
            for _, row in df_nomes.iterrows():
                users[int(row['usuario_id'])] = row['nome']
        except Exception: pass

    if os.path.exists(AVALIACOES_PATH):
        try:
            df_avaliacoes = pd.read_csv(AVALIACOES_PATH)
            for user_id in df_avaliacoes['usuario_id'].unique():
                if int(user_id) not in users:
                    users[int(user_id)] = f"Usuário {user_id}"
        except Exception: pass

    if not users: return []
    usuarios_list = sorted([{"usuario_id": k, "nome": v} for k, v in users.items()], key=lambda x: x['usuario_id'])
    return usuarios_list


def calcular_metricas_usuario(
    usuario_id: int,
    recomendacoes_do_sistema: Collection[str],
    df_avaliacoes_global: pd.DataFrame,
    df_filmes: pd.DataFrame
) -> dict:
    """
    Calcula Precision, Recall e F1-score comparando as recomendações do sistema
    com o que o usuário gostou (gabarito).
    """
    gabarito_positivos_ids = df_avaliacoes_global[
        (df_avaliacoes_global['usuario_id'] == usuario_id) &
        (df_avaliacoes_global['avaliacao'] == 1)
    ]['filme_id'].unique()

    if len(gabarito_positivos_ids) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "tp_count": 0, "gabarito_count": 0, "recomendados_count": len(recomendacoes_do_sistema), "mensagem": "Usuário sem avaliações positivas no gabarito."}

    gabarito_titulos = df_filmes.iloc[gabarito_positivos_ids]['Series_Title'].tolist()

    tp_titulos = set(recomendacoes_do_sistema) & set(gabarito_titulos)
    TP = len(tp_titulos)

    total_recomendacoes = len(recomendacoes_do_sistema)
    precision = TP / total_recomendacoes if total_recomendacoes > 0 else 0.0

    total_gabarito = len(gabarito_titulos)
    recall = TP / total_gabarito if total_gabarito > 0 else 0.0

    f1_score = 0.0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp_count": TP,
        "gabarito_count": total_gabarito,
        "recomendados_count": total_recomendacoes
    }