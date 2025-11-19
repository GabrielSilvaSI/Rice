from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from numpy import ndarray
import sys
import os

# Ajusta o caminho para importar recomendacao.py (lógica de ML)
sys.path.append(os.path.dirname(__file__))
from recomendacao import (
    carregar_dados_e_vetorizar,
    construir_perfil_usuario,
    gerar_recomendacoes,
    salvar_avaliacao,
    carregar_e_listar_usuarios,
    calcular_metricas_usuario
)

# --- Caminhos ---
BASE_DIR = os.path.dirname(__file__)
USUARIOS_PATH = os.path.join(BASE_DIR, "..", "datasets", "usuarios.csv")
AVALIACOES_PATH = os.path.join(BASE_DIR, "..", "datasets", "avaliacoes.csv")

# -----------------------------------------------------------
# APP e VARIÁVEIS GLOBAIS
# -----------------------------------------------------------

app = FastAPI(
    title="RICE API: Sistema de Recomendação de Filmes",
    description="Backend para Filtragem Baseada em Conteúdo.",
    version="1.0.0"
)

CATALOGO_FILMES: pd.DataFrame = None
MATRIZ_VETORES: ndarray = None
DF_AVALIACOES_GLOBAL: pd.DataFrame = None

# -----------------------------------------------------------
# STARTUP
# -----------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """
    Função executada na inicialização. Carrega os dados e treina o vetorizador.
    """
    global CATALOGO_FILMES, MATRIZ_VETORES, DF_AVALIACOES_GLOBAL
    print("Iniciando carga de dados e vetorização...")
    try:
        df, matriz = carregar_dados_e_vetorizar()
        CATALOGO_FILMES = df
        MATRIZ_VETORES = matriz
        print("✅ Modelo de recomendação pronto.")
        
        if os.path.exists(AVALIACOES_PATH):
            DF_AVALIACOES_GLOBAL = pd.read_csv(AVALIACOES_PATH)
            print("✅ Dataset de avaliações carregado.")

    except Exception as e:
        print(f"ERRO CRÍTICO na inicialização: {e}")
        pass

# -----------------------------------------------------------
# MODELOS DE DADOS (Pydantic)
# -----------------------------------------------------------

class RecomendacaoRequest(BaseModel):
    usuario_id: int
    num_recomendacoes: int = 10

class UsuarioNovo(BaseModel):
    usuario_id: int
    nome: str

class AvaliacaoRequest(BaseModel):
    usuario_id: int
    filme_id: int
    avaliacao: int

# -----------------------------------------------------------
# ENDPOINTS DA API
# -----------------------------------------------------------

@app.get("/itens")
def get_itens():
    """Retorna o catálogo de filmes para o frontend."""
    if CATALOGO_FILMES is None:
        raise HTTPException(status_code=503, detail="Catálogo de filmes não carregado.")
    return CATALOGO_FILMES.reset_index().rename(columns={'index': 'filme_id'}).to_dict('records')

@app.get("/usuarios")
def get_usuarios():
    """Retorna a lista de usuários do sistema."""
    try:
        return {"usuarios": carregar_e_listar_usuarios()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar usuários: {e}")

@app.post("/usuarios", status_code=201)
def add_usuario(novo_usuario: UsuarioNovo):
    """Adiciona um novo usuário ao arquivo usuarios.csv."""
    novo_registro = pd.DataFrame([novo_usuario.dict()])
    try:
        header = not os.path.exists(USUARIOS_PATH)
        novo_registro.to_csv(USUARIOS_PATH, mode='a', index=False, header=header)
        return {"message": "Usuário salvo com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar usuário: {e}")

@app.post("/avaliacoes", status_code=201)
def add_avaliacao(request: AvaliacaoRequest):
    """Salva uma nova avaliação de filme para um usuário."""
    if not salvar_avaliacao(request.usuario_id, request.filme_id, request.avaliacao):
        raise HTTPException(status_code=500, detail="Falha ao persistir a avaliação.")
    return {"message": "Avaliação salva com sucesso."}

@app.get("/avaliacoes/{usuario_id}")
def get_avaliacoes_usuario(usuario_id: int):
    """Retorna o histórico de avaliações de um usuário."""
    if DF_AVALIACOES_GLOBAL is None: return []
    user_ratings = DF_AVALIACOES_GLOBAL[DF_AVALIACOES_GLOBAL['usuario_id'] == usuario_id]
    return user_ratings.to_dict('records')

@app.post("/recomendar")
def recomendar_filmes(request: RecomendacaoRequest):
    """Gera recomendações de filmes para um usuário."""
    if CATALOGO_FILMES is None or MATRIZ_VETORES is None:
        raise HTTPException(status_code=503, detail="Modelo de recomendação não carregado.")

    perfil = construir_perfil_usuario(request.usuario_id, CATALOGO_FILMES, MATRIZ_VETORES)
    if perfil is None:
        raise HTTPException(status_code=404, detail="Usuário sem perfil para gerar recomendações.")

    recomendacoes = gerar_recomendacoes(perfil, CATALOGO_FILMES, MATRIZ_VETORES, request.num_recomendacoes)
    
    lista_final = []
    for titulo, score in recomendacoes:
        filme_info = CATALOGO_FILMES[CATALOGO_FILMES['Series_Title'] == titulo].iloc[0]
        lista_final.append({
            "titulo": titulo,
            "similaridade": f"{score:.4f}",
            "poster_link": filme_info.get('Poster_Link')
        })
    return {"recomendacoes": lista_final}

@app.get("/metricas/{usuario_id}")
def get_metricas(usuario_id: int, num_recomendacoes: int = 10):
    """Calcula e retorna Precision, Recall e F1-score para o usuário."""
    if CATALOGO_FILMES is None or MATRIZ_VETORES is None or DF_AVALIACOES_GLOBAL is None:
        raise HTTPException(status_code=503, detail="Modelo ou dados de avaliação não carregados.")

    perfil_usuario = construir_perfil_usuario(usuario_id, CATALOGO_FILMES, MATRIZ_VETORES)
    if perfil_usuario is None:
        raise HTTPException(status_code=404, detail="Usuário sem avaliações positivas para calcular o perfil.")

    recomendacoes = gerar_recomendacoes(perfil_usuario, CATALOGO_FILMES, MATRIZ_VETORES, num_recomendacoes)
    titulos_recomendados = [titulo for titulo, score in recomendacoes]

    metricas = calcular_metricas_usuario(
        usuario_id=usuario_id,
        recomendacoes_do_sistema=titulos_recomendados,
        df_avaliacoes_global=DF_AVALIACOES_GLOBAL,
        df_filmes=CATALOGO_FILMES
    )
    
    return {
        "usuario_id": usuario_id,
        "num_recomendacoes": num_recomendacoes,
        "precision": f"{metricas['precision']:.4f}",
        "recall": f"{metricas['recall']:.4f}",
        "f1_score": f"{metricas['f1_score']:.4f}",
        "detalhes": f"TP={metricas['tp_count']}, Gabarito={metricas['gabarito_count']}, Recomendados={metricas['recomendados_count']}"
    }