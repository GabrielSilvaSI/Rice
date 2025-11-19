from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from numpy import ndarray
import sys
import os

# Ajusta o caminho para importar recomendacao.py
# (Presume que recomendacao.py está no mesmo diretório)
sys.path.append(os.path.dirname(__file__))
from recomendacao import (
    carregar_dados_e_vetorizar,
    construir_perfil_usuario,
    gerar_recomendacoes,
    # As variáveis globais serão carregadas aqui
)

# -----------------------------------------------------------
# 1. INICIALIZAÇÃO DA API E CARGA DE DADOS
# -----------------------------------------------------------

app = FastAPI(
    title="RICE API: Sistema de Recomendação de Filmes",
    description="Backend para Filtragem Baseada em Conteúdo (RICE).",
    version="1.0.0"
)

# Variáveis para armazenar o catálogo e a matriz TF-IDF
CATALOGO_FILMES: pd.DataFrame = None
MATRIZ_VETORES: ndarray = None


@app.on_event("startup")
def startup_event():
    """
    Função executada quando a API é iniciada (carga única e otimizada).
    """
    global CATALOGO_FILMES, MATRIZ_VETORES

    try:
        # Executa o Passo 1: Vetorização
        df, matriz = carregar_dados_e_vetorizar()
        if df is None or matriz is None:
            raise Exception("Falha na carga dos datasets ou vetorização.")

        CATALOGO_FILMES = df
        MATRIZ_VETORES = matriz
        print("✅ Backend RICE pronto. Dados e vetores carregados na memória.")

    except Exception as e:
        print(f"ERRO CRÍTICO na inicialização: {e}")
        sys.exit(1)


# -----------------------------------------------------------
# 2. DEFINIÇÃO DO MODELO DE DADOS PARA A REQUISIÇÃO
# -----------------------------------------------------------

class RecomendacaoRequest(BaseModel):
    """Modelo de dados para o corpo da requisição POST /recomendar."""
    usuario_id: int
    num_recomendacoes: int = 10


# -----------------------------------------------------------
# 3. ENDPOINTS OBRIGATÓRIOS (ATUALIZADOS)
# -----------------------------------------------------------

@app.get("/itens")
def get_itens():
    """Endpoint para o Streamlit listar o catálogo de filmes, incluindo o Poster_Link."""
    if CATALOGO_FILMES is None:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    # Retorna o ID, Título, Gênero, Overview E AGORA O Poster_Link de todos os filmes.
    return CATALOGO_FILMES[['Series_Title', 'Genre', 'Overview', 'Poster_Link']].reset_index().rename(
        columns={'index': 'filme_id'}).to_dict('records')


@app.post("/recomendar")
def recomendar_filmes(request: RecomendacaoRequest):
    """
    Endpoint principal para gerar recomendações, retornando o Poster_Link.
    """
    if CATALOGO_FILMES is None:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    usuario_id = request.usuario_id
    num_rec = request.num_recomendacoes

    # 1. Construção do Perfil (Passo 2)
    perfil_usuario = construir_perfil_usuario(
        usuario_id=usuario_id,
        df_itens=CATALOGO_FILMES,
        tfidf_matriz=MATRIZ_VETORES
    )

    if perfil_usuario is None:
        raise HTTPException(status_code=404, detail=f"Usuário {usuario_id} não encontrado ou sem avaliações positivas.")

    # 2. Geração da Lista (Passo 3: Similaridade)
    recomendacoes = gerar_recomendacoes(
        perfil_usuario=perfil_usuario,
        df_itens=CATALOGO_FILMES,
        tfidf_matriz=MATRIZ_VETORES,
        num_recomendacoes=num_rec
    )

    # 3. Formata a resposta com o Poster_Link
    lista_final = []
    for titulo, score in recomendacoes:
        # Faz um lookup no DataFrame original para encontrar o link do poster
        # Usamos .iloc[0] para pegar a primeira (e presumivelmente única) ocorrência
        filme = CATALOGO_FILMES[CATALOGO_FILMES['Series_Title'] == titulo].iloc[0]

        lista_final.append({
            "titulo": titulo,
            "similaridade": f"{score:.4f}",
            "poster_link": filme['Poster_Link']  # <--- ADICIONADO
        })

    return {
        "usuario_id": usuario_id,
        "recomendacoes": lista_final
    }