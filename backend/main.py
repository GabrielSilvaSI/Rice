from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from numpy import ndarray
import sys
import os

USUARIOS_PATH = "../datasets/usuarios.csv"  # Defina o caminho

# Ajusta o caminho para importar recomendacao.py (lógica de ML)
# Assume que recomendacao.py está no mesmo diretório
sys.path.append(os.path.dirname(__file__))
from recomendacao import (
    carregar_dados_e_vetorizar,
    construir_perfil_usuario,
    gerar_recomendacoes,
    salvar_avaliacao,
    carregar_e_listar_usuarios
)

# -----------------------------------------------------------
# VARIÁVEIS GLOBAIS
# -----------------------------------------------------------

app = FastAPI(
    title="RICE API: Sistema de Recomendação de Filmes",
    description="Backend para Filtragem Baseada em Conteúdo.",
    version="1.0.0"
)

# Variáveis para armazenar o catálogo, a matriz TF-IDF e os IDs dos usuários
CATALOGO_FILMES: pd.DataFrame = None
MATRIZ_VETORES: ndarray = None
LISTA_USUARIOS: list[int] = []


@app.on_event("startup")
def startup_event():
    """
    Função executada quando a API é iniciada. Carrega os datasets e processa a vetorização.
    """
    global CATALOGO_FILMES, MATRIZ_VETORES, LISTA_USUARIOS

    print("Iniciando carga de dados e vetorização...")

    try:
        # Carregar e Vetorizar (Passo 1)
        df, matriz = carregar_dados_e_vetorizar()
        if df is None or matriz is None:
            raise Exception("Falha na vetorização.")

        CATALOGO_FILMES = df
        MATRIZ_VETORES = matriz

        # Carregar Usuários (Necessário para GET /usuarios)
        # Lendo o arquivo avaliacoes.csv para obter a lista de IDs únicos.
        # Assumindo que o avaliacoes.csv está em ../datasets/avaliacoes.csv
        df_avaliacoes = pd.read_csv("../datasets/avaliacoes.csv")
        LISTA_USUARIOS = sorted(df_avaliacoes['usuario_id'].unique().tolist())

        print(f"✅ Backend RICE pronto. {len(LISTA_USUARIOS)} usuários carregados.")

    except Exception as e:
        print(f"ERRO CRÍTICO na inicialização: {e}")
        sys.exit(1)

# -----------------------------------------------------------
# ENDPOINT: MODELO DE DADOS
# -----------------------------------------------------------

class RecomendacaoRequest(BaseModel):
    """Modelo de dados para o corpo da requisição POST /recomendar."""
    usuario_id: int
    num_recomendacoes: int = 10

class UsuarioNovo(BaseModel):
    """Modelo para receber dados de novo usuário."""
    usuario_id: int
    nome: str

class AvaliacaoRequest(BaseModel):
    """Modelo de dados para receber uma nova avaliação."""
    usuario_id: int
    filme_id: int
    avaliacao: int # 0 ou 1


# -----------------------------------------------------------
# ENDPOINTS OBRIGATÓRIOS DO GUIA
# -----------------------------------------------------------
@app.get("/itens")
def get_itens():
    """GET /itens: Retorna o catálogo de filmes, incluindo dados visuais."""
    if CATALOGO_FILMES is None:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    # Retorna as colunas essenciais para exibição no frontend.
    return CATALOGO_FILMES[['Series_Title', 'Genre', 'Overview', 'Poster_Link']].reset_index().rename(
        columns={'index': 'filme_id'}).to_dict('records')

@app.get("/usuarios")
def get_usuarios():
    """GET /usuarios: Retorna a lista de IDs de usuários disponíveis para simulação."""
    if not LISTA_USUARIOS:
        raise HTTPException(status_code=503, detail="Lista de usuários não carregada.")

    # Retorna uma lista simples de todos os IDs de usuários únicos.
    return {"usuarios": LISTA_USUARIOS}

@app.post("/recomendar")
def recomendar_filmes(request: RecomendacaoRequest):
    """
    POST /recomendar: Gera recomendações usando o Perfil do Usuário e Similaridade.
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
        filme = CATALOGO_FILMES[CATALOGO_FILMES['Series_Title'] == titulo].iloc[0]

        lista_final.append({
            "titulo": titulo,
            "similaridade": f"{score:.4f}",
            "poster_link": filme['Poster_Link']
        })

    return {
        "usuario_id": usuario_id,
        "recomendacoes": lista_final
    }

@app.post("/usuarios", status_code=201)
def add_usuario(novo_usuario: UsuarioNovo):
    """
    POST /usuarios: Adiciona um novo usuário ao CSV de forma persistente.
    """

    # Cria um novo DataFrame com os dados do novo usuário
    novo_registro = pd.DataFrame([{'usuario_id': novo_usuario.usuario_id, 'nome': novo_usuario.nome}])

    try:
        # Modo 'a' (append) para adicionar a linha no final.
        # header=False para evitar reescrever o cabeçalho.
        novo_registro.to_csv(USUARIOS_PATH, mode='a', index=False, header=False)

        # Atualiza a lista global de IDs de usuário carregada na inicialização
        global LISTA_USUARIOS
        if novo_usuario.usuario_id not in LISTA_USUARIOS:
            LISTA_USUARIOS.append(novo_usuario.usuario_id)
            LISTA_USUARIOS.sort()

        return {"message": "Usuário salvo com sucesso", "usuario_id": novo_usuario.usuario_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar usuário: {e}")

@app.post("/avaliacoes", status_code=201)
def add_avaliacao(request: AvaliacaoRequest):
    """
    POST /avaliacoes: Salva a avaliação de um usuário para um filme específico.
    """

    # 1. Tenta salvar a avaliação no CSV
    sucesso = salvar_avaliacao(
        usuario_id=request.usuario_id,
        filme_id=request.filme_id,
        avaliacao=request.avaliacao
    )

    if sucesso:
        # Invalida o cache para que, na próxima recomendação, o perfil seja recalculado.
        # Isto é crucial, pois o perfil depende das avaliações.
        construir_perfil_usuario.clear()

        return {"message": "Avaliação salva com sucesso e cache de perfil limpo.",
                "usuario_id": request.usuario_id,
                "filme_id": request.filme_id}
    else:
        raise HTTPException(status_code=500, detail="Falha interna ao persistir a avaliação.")

@app.get("/usuarios")
def get_usuarios():
    """
    GET /usuarios: Retorna a lista de todos os usuários (IDs e Nomes)
    disponíveis no sistema, combinando dados de avaliacoes.csv e usuarios.csv.
    """
    try:
        # Chama a função que carrega e formata a lista de usuários
        usuarios_list = carregar_e_listar_usuarios()

        # Formato esperado pelo frontend: {"usuarios": [...]}
        return {"usuarios": usuarios_list}
    except Exception as e:
        # Caso haja erro na leitura ou processamento (por exemplo, caminho errado)
        raise HTTPException(status_code=500, detail=f"Erro ao carregar lista de usuários: {e}")