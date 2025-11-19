import streamlit as st
import requests
import pandas as pd

# URL base do seu backend FastAPI
BASE_URL = "http://127.0.0.1:8000"


# -------------------------------------------------------------
# 1. FUNÇÕES DE COMUNICAÇÃO E UTILIDADE
# -------------------------------------------------------------

@st.cache_data
def get_catalogo():
    """Busca o catálogo de filmes via GET /itens."""
    try:
        response = requests.get(f"{BASE_URL}/itens")
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Erro ao carregar catálogo. Backend está OK? {e}")
        st.stop()


@st.cache_data
def get_usuarios_validos():
    """
    Busca a lista de usuários válidos (ID e Nome) do backend.

    EXPECTATIVA: O backend deve retornar uma lista de dicionários como:
    [{"usuario_id": 1, "nome": "Alice"}, {"usuario_id": 2, "nome": "Usuário 2"}]
    """
    try:
        response = requests.get(f"{BASE_URL}/usuarios")
        response.raise_for_status()

        # O backend deve retornar uma lista de objetos usuário
        usuarios_data = response.json().get('usuarios', [])

        # Mapeamento ID -> Objeto completo, e lista formatada para o selectbox
        user_map = {user['usuario_id']: user['nome'] for user in usuarios_data}
        selectbox_options = [f"{user['nome']} (ID: {user['usuario_id']})" for user in usuarios_data]

        return user_map, ["--- Selecione ou Crie um Usuário ---"] + selectbox_options
    except Exception as e:
        st.error("Erro ao carregar lista de usuários. Backend está rodando?")
        return {}, ["--- Erro ao carregar usuários ---"]


def get_next_user_id(user_map: dict) -> int:
    """Gera o próximo ID sequencial."""
    if not user_map:
        return 1
    return max(user_map.keys()) + 1


# -------------------------------------------------------------
# 2. INTERFACES E CONTROLE DE USUÁRIO ATIVO
# -------------------------------------------------------------

# Variável de estado para o usuário ativo (valor do selectbox)
active_user_id = None


def user_management_and_selection(user_map, selectbox_options):
    """
    Interface para Adicionar Novo Usuário e Selecionar Usuário Ativo.
    Retorna o ID do usuário selecionado (ou None).
    """
    global active_user_id

    st.subheader("1. Seleção de Usuário Ativo")

    # --- Seleção de Usuário Existente ---
    selected_option = st.selectbox(
        "Selecione um Usuário para Atividade:",
        options=selectbox_options,
        index=0,
        key="user_selector"
    )

    # Extrai o ID da opção selecionada (Ex: "Nome (ID: 10)" -> 10)
    if "ID:" in selected_option:
        # Regex simples para pegar o número após 'ID: '
        user_id_str = selected_option.split("ID: ")[-1].replace(")", "")
        active_user_id = int(user_id_str)
        st.markdown(f"**Usuário Ativo:** **{user_map[active_user_id]}** (ID: `{active_user_id}`)")
    else:
        active_user_id = None
        st.markdown(f"**Usuário Ativo:** ❌ Nenhum selecionado.")

    st.markdown("---")

    # --- Adicionar Novo Usuário ---
    st.subheader("2. Adicionar Novo Usuário")

    next_id = get_next_user_id(user_map)

    with st.form("new_user_form"):
        st.info(f"O ID do novo usuário será: **{next_id}**")
        new_user_name = st.text_input("Nome:", key="new_user_name_input")
        submitted = st.form_submit_button("Criar e Ativar")

        if submitted:
            if not new_user_name.strip():
                st.error("O nome do usuário não pode ser vazio.")
            else:
                # Envia para o Backend (POST /usuarios)
                response = requests.post(
                    f"{BASE_URL}/usuarios",
                    json={"usuario_id": next_id, "nome": new_user_name.strip()}
                )

                if response.status_code == 201:
                    get_usuarios_validos.clear()  # Recarrega a lista
                    st.toast(f"Usuário {next_id} criado. Recarregando...")
                    st.rerun()  # Atualiza a combobox
                else:
                    st.error(f"Falha ao persistir usuário: {response.json().get('detail', 'Erro desconhecido')}")

    return active_user_id


def add_evaluation_page(user_id, catalogo_df, user_map):
    """Tela para adicionar uma nova avaliação para o usuário ativo."""
    st.title("➕ Adicionar Nova Avaliação")

    if user_id is None:
        st.warning("Selecione um Usuário na aba 'Gerenciar Usuário' para adicionar avaliações.")
        return

    st.subheader(f"Avaliar Filmes para: {user_map.get(user_id, f'ID {user_id}')}")  # Usa o nome

    # ... (Restante da lógica de seleção de filme e submissão) ...
    # Lista de filmes no catálogo (Título + ID)
    catalogo_df['display_name'] = catalogo_df.apply(lambda row: f"{row['Series_Title']} ({row['filme_id']})", axis=1)

    # Combobox para seleção do filme
    filme_selecionado = st.selectbox(
        "Selecione o Filme:",
        options=catalogo_df['display_name'].tolist(),
        index=0
    )

    # Extrai o ID do filme
    filme_id = catalogo_df[catalogo_df['display_name'] == filme_selecionado]['filme_id'].iloc[0]

    # Seleção da avaliação (0 ou 1)
    avaliacao = st.radio(
        "Você gostou deste filme?",
        options=[1, 0],
        format_func=lambda x: "👍 Sim (1)" if x == 1 else "👎 Não (0)"
    )

    if st.button("Submeter Avaliação"):
        payload = {
            "usuario_id": user_id,
            "filme_id": int(filme_id),
            "avaliacao": int(avaliacao)
        }

        try:
            response = requests.post(f"{BASE_URL}/avaliacoes", json=payload)
            response.raise_for_status()

            st.success(f"Avaliação (Nota {avaliacao}) submetida para '{filme_selecionado}' pelo Usuário {user_id}.")
            st.toast("Avaliação salva! 🎉")

            # Limpa o cache de recomendações (Passo crucial para o backend)
            # Como não temos o perfil salvo na sessão, só o backend limpa o cache.

        except requests.HTTPError as e:
            st.error(f"Erro ao salvar avaliação (Backend): {e}. Verifique o console do FastAPI.")
        except Exception as e:
            st.error(f"Erro de conexão: {e}")


def recommendation_page(user_id, catalogo_df, user_map):
    """Tela para gerar recomendações com layout de Cards."""
    st.title("⭐ Recomendações RICE")

    if user_id is None:
        st.warning("Selecione um Usuário na aba 'Gerenciar Usuário' para gerar recomendações.")
        return

    st.subheader(f"Gerar Recomendações para: {user_map.get(user_id, f'ID {user_id}')}")

    num_rec = st.slider("Número de Recomendações:", 5, 20, 10)

    if st.button("Gerar Recomendações", type="primary"):
        with st.spinner('Calculando perfil e similaridade...'):
            payload = {
                "usuario_id": user_id,
                "num_recomendacoes": num_rec
            }

            try:
                response = requests.post(f"{BASE_URL}/recomendar", json=payload)
                response.raise_for_status()
                data = response.json()

                st.success(f"✅ Top {len(data['recomendacoes'])} Recomendações Recebidas")

                # --- NOVO PAINEL DE CARDS ---

                # Cria 5 colunas para o layout de cards/grid
                cols = st.columns(5)

                for rank, rec in enumerate(data['recomendacoes']):
                    # Seleciona a coluna atual (rank % 5)
                    col = cols[rank % 5]

                    with col:
                        # Exibe a imagem/poster
                        if rec.get('poster_link'):
                            st.image(rec['poster_link'], caption=f"Rank {rank + 1}: {rec['titulo']}",
                                     use_column_width=True)

                        # Adiciona detalhes do score
                        st.markdown(f"**Score:** `{rec['similaridade']}`")

                        # Adiciona a sinopse/overview (opcional, para enriquecer o card)
                        # Nota: É necessário buscar o overview localmente no catalogo_df pelo título
                        overview = catalogo_df[catalogo_df['Series_Title'] == rec['titulo']]['Overview'].iloc[0]
                        with st.expander("Sinopse"):
                            st.caption(overview)

                # --- FIM DO PAINEL DE CARDS ---

            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    st.error(
                        f"Erro 404: Usuário {user_id} sem avaliações positivas no avaliacoes.csv para gerar perfil.")
                else:
                    st.error(f"Erro no Backend: {e}")
            except Exception as e:
                st.error(f"Erro de Conexão: {e}")


# -------------------------------------------------------------
# 4. CONTROLE DO FLUXO PRINCIPAL
# -------------------------------------------------------------

def app():
    st.set_page_config(layout="wide", page_title="RICE - Recomendações de Filmes")
    st.title("🎬 RICE: Sistema de Recomendação de Filmes")

    # Carrega dados essenciais
    catalogo_df = get_catalogo()
    user_map, selectbox_options = get_usuarios_validos()  # Retorna mapa ID:Nome e lista formatada

    # Variável de controle do ID Ativo (Será atualizada pelo widget na aba 1)
    active_user_id = None

    # Abas
    tab1, tab2, tab3 = st.tabs(["Gerenciar Usuário", "Adicionar Avaliação", "Recomendações"])

    with tab1:
        # A função user_management_and_selection lê o valor do selectbox
        # e o selectbox é um widget persistente.
        active_user_id = user_management_and_selection(user_map, selectbox_options)

    # Usamos o ID ativo para as outras abas
    with tab2:
        add_evaluation_page(active_user_id, catalogo_df, user_map)

    with tab3:
        recommendation_page(active_user_id, catalogo_df, user_map)


if __name__ == "__main__":
    app()