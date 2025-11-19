from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from numpy import ndarray
import sys
import os

USUARIOS_PATH = "../datasets/usuarios.csv"  # Defina o caminho
AVALIACOES_PATH = "../datasets/avaliacoes.csv"

# Ajusta o caminho para importar recomendacao.py (lógica de ML)
sys.path.append(os.path.dirname(__file__))
from recomendacao import (
    carregar_dados_e_vetorizar,
    construir_perfil_usuario,
    gerar_recomendacoes,
    salvar_avaliacao,
    carregar_e_listar_usuarios
)

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

# -----------------------------------------------------------
# STARTUP
# -----------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """
    Função executada na inicialização. Carrega os dados e treina o vetorizador.
    """
    global CATALOGO_FILMES, MATRIZ_VETORES
    print("Iniciando carga de dados e vetorização...")
    try:
        df, matriz = carregar_dados_e_vetorizar()
        CATALOGO_FILMES = df
        MATRIZ_VETORES = matriz
        print("✅ Backend RICE pronto.")
    except Exception as e:
        print(f"ERRO CRÍTICO na inicialização do modelo: {e}")
        # Permite que a API continue rodando para servir outros endpoints (ex: /usuarios)
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
        raise HTTPException(status_code=503, detail="Catálogo de filmes não carregado. Verifique o log do servidor.")
    return CATALOGO_FILMES[['Series_Title', 'Genre', 'Overview', 'Poster_Link']].reset_index().rename(
        columns={'index': 'filme_id'}).to_dict('records')

@app.get("/usuarios")
def get_usuarios():
    """Retorna a lista de usuários (ID e Nome) do sistema."""
    try:
        usuarios_list = carregar_e_listar_usuarios()
        return {"usuarios": usuarios_list}
    except FileNotFoundError:
        return {"usuarios": []}  # Retorna lista vazia se os arquivos não existirem
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar lista de usuários: {e}")

@app.post("/usuarios", status_code=201)
def add_usuario(novo_usuario: UsuarioNovo):
    """Adiciona um novo usuário ao arquivo usuarios.csv."""
    novo_registro = pd.DataFrame([{'usuario_id': novo_usuario.usuario_id, 'nome': novo_usuario.nome}])
    try:
        novo_registro.to_csv(
            USUARIOS_PATH,
            mode='a',
            index=False,
            header=not os.path.exists(USUARIOS_PATH)
        )
        return {"message": "Usuário salvo com sucesso", "usuario_id": novo_usuario.usuario_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar usuário: {e}")

@app.post("/avaliacoes", status_code=201)
def add_avaliacao(request: AvaliacaoRequest):
    """Salva uma nova avaliação de filme para um usuário."""
    sucesso = salvar_avaliacao(
        usuario_id=request.usuario_id,
        filme_id=request.filme_id,
        avaliacao=request.avaliacao
    )
    if not sucesso:
        raise HTTPException(status_code=500, detail="Falha interna ao persistir a avaliação.")
    
    # A função construir_perfil_usuario não é cacheada, então a nova avaliação
    # será automaticamente considerada na próxima chamada a /recomendar.
    return {"message": "Avaliação salva com sucesso."}

@app.get("/avaliacoes/{usuario_id}")
def get_avaliacoes_usuario(usuario_id: int):
    """Retorna o histórico de avaliações de um usuário específico."""
    if not os.path.exists(AVALIACOES_PATH):
        return []
    try:
        df_avaliacoes = pd.read_csv(AVALIACOES_PATH)
        user_ratings = df_avaliacoes[df_avaliacoes['usuario_id'] == usuario_id]
        return user_ratings.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler o arquivo de avaliações: {e}")

@app.post("/recomendar")
def recomendar_filmes(request: RecomendacaoRequest):
    """Gera recomendações de filmes para um usuário."""
    if CATALOGO_FILMES is None or MATRIZ_VETORES is None:
        raise HTTPException(status_code=503, detail="Modelo de recomendação não carregado. Verifique o log do servidor.")

    perfil_usuario = construir_perfil_usuario(
        usuario_id=request.usuario_id,
        df_itens=CATALOGO_FILMES,
        tfidf_matriz=MATRIZ_VETORES
    )
    if perfil_usuario is None:
        raise HTTPException(status_code=404, detail=f"Usuário {request.usuario_id} não encontrado ou sem avaliações positivas.")

    recomendacoes = gerar_recomendacoes(
        perfil_usuario=perfil_usuario,
        df_itens=CATALOGO_FILMES,
        tfidf_matriz=MATRIZ_VETORES,
        num_recomendacoes=request.num_recomendacoes
    )

    lista_final = []
    for titulo, score in recomendacoes:
        filme = CATALOGO_FILMES[CATALOGO_FILMES['Series_Title'] == titulo].iloc[0]
        lista_final.append({
            "titulo": titulo,
            "similaridade": f"{score:.4f}",
            "poster_link": filme['Poster_Link']
        })

    return {
        "usuario_id": request.usuario_id,
        "recomendacoes": lista_final
    }
