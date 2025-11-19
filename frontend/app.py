import streamlit as st
import requests
import pandas as pd
import plotly.express as px

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
    """Busca a lista de usuários válidos do backend."""
    try:
        response = requests.get(f"{BASE_URL}/usuarios")
        response.raise_for_status()
        usuarios_data = response.json().get('usuarios', [])
        user_map = {user['usuario_id']: user['nome'] for user in usuarios_data}
        selectbox_options = [f"{user['nome']} (ID: {user['usuario_id']})" for user in usuarios_data]
        return user_map, ["--- Selecione ou Crie um Usuário ---"] + selectbox_options
    except Exception as e:
        st.error("Erro ao carregar lista de usuários. Backend está rodando?")
        return {}, ["--- Erro ao carregar usuários ---"]


def get_next_user_id(user_map: dict) -> int:
    """Gera o próximo ID sequencial."""
    if not user_map: return 1
    return max(user_map.keys()) + 1


# -------------------------------------------------------------
# 2. INTERFACES E CONTROLE DE USUÁRIO ATIVO
# -------------------------------------------------------------

def user_management_and_selection(user_map, selectbox_options):
    """Interface para Adicionar e Selecionar Usuário Ativo."""
    st.subheader("1. Seleção de Usuário Ativo")
    selected_option = st.selectbox("Selecione um Usuário:", options=selectbox_options, key="user_selector")

    active_user_id = None
    if "ID:" in selected_option:
        user_id_str = selected_option.split("ID: ")[-1].replace(")", "")
        active_user_id = int(user_id_str)
        st.markdown(f"**Usuário Ativo:** **{user_map.get(active_user_id, 'N/A')}** (ID: `{active_user_id}`)")
    else:
        st.markdown("**Usuário Ativo:** ❌ Nenhum selecionado.")

    st.markdown("---")
    st.subheader("2. Adicionar Novo Usuário")
    next_id = get_next_user_id(user_map)

    with st.form("new_user_form"):
        st.info(f"O ID do novo usuário será: **{next_id}**")
        new_user_name = st.text_input("Nome:")
        if st.form_submit_button("Criar e Ativar"):
            if new_user_name.strip():
                response = requests.post(f"{BASE_URL}/usuarios", json={"usuario_id": next_id, "nome": new_user_name.strip()})
                if response.status_code == 201:
                    get_usuarios_validos.clear()
                    st.success(f"Usuário '{new_user_name}' criado com ID {next_id}. Recarregando...")
                    st.rerun()
                else:
                    st.error(f"Falha ao criar usuário: {response.text}")
            else:
                st.error("O nome não pode ser vazio.")
    return active_user_id


def add_evaluation_page(user_id, catalogo_df, user_map):
    """Tela para adicionar uma nova avaliação."""
    st.title("➕ Adicionar Nova Avaliação")
    if user_id is None:
        st.warning("Selecione um Usuário na aba 'Gerenciar Usuário'.")
        return

    st.subheader(f"Avaliar Filmes para: {user_map.get(user_id, f'ID {user_id}')}")
    catalogo_df['display_name'] = catalogo_df.apply(lambda row: f"{row['Series_Title']} ({row['filme_id']})", axis=1)
    filme_selecionado = st.selectbox("Selecione o Filme:", options=catalogo_df['display_name'].tolist())
    filme_id = catalogo_df[catalogo_df['display_name'] == filme_selecionado]['filme_id'].iloc[0]
    avaliacao = st.radio("Gostou do filme?", options=[1, 0], format_func=lambda x: "👍 Sim" if x == 1 else "👎 Não")

    if st.button("Submeter Avaliação"):
        payload = {"usuario_id": user_id, "filme_id": int(filme_id), "avaliacao": int(avaliacao)}
        try:
            response = requests.post(f"{BASE_URL}/avaliacoes", json=payload)
            response.raise_for_status()
            st.success("Avaliação submetida com sucesso!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao submeter avaliação: {e}")

    st.subheader("Histórico de Avaliações")
    try:
        response = requests.get(f"{BASE_URL}/avaliacoes/{user_id}")
        response.raise_for_status()
        avaliacoes = response.json()
        if avaliacoes:
            st.dataframe(pd.DataFrame(avaliacoes))
        else:
            st.info("Usuário sem avaliações.")
    except Exception as e:
        st.error(f"Não foi possível carregar o histórico: {e}")


def recommendation_page(user_id, catalogo_df, user_map):
    """Tela para gerar recomendações."""
    st.title("⭐ Recomendações RICE")
    if user_id is None:
        st.warning("Selecione um Usuário na aba 'Gerenciar Usuário'.")
        return

    st.subheader(f"Gerar Recomendações para: {user_map.get(user_id, f'ID {user_id}')}")
    num_rec = st.slider("Número de Recomendações:", 5, 20, 10, key="num_rec_slider_rec")

    if st.button("Gerar Recomendações", type="primary"):
        with st.spinner('Calculando recomendações...'):
            try:
                payload = {"usuario_id": user_id, "num_recomendacoes": num_rec}
                response = requests.post(f"{BASE_URL}/recomendar", json=payload)
                response.raise_for_status()
                data = response.json()
                st.success(f"✅ Top {len(data['recomendacoes'])} Recomendações:")

                cols = st.columns(5)
                for i, rec in enumerate(data['recomendacoes']):
                    with cols[i % 5]:
                        st.image(
                            rec['poster_link'].replace("UX67_CR0,0,67,98", "UX260_CR0,0,260,380"),
                            caption=f"Rank {i + 1}: {rec['titulo']}",
                            use_container_width=True
                        )
                        st.markdown(f"**Score:** `{rec['similaridade']}`")
            except Exception as e:
                st.error(f"Erro ao gerar recomendações: {e}")

def evaluation_tab(user_id, user_map):
    """Aba para exibir métricas de avaliação do sistema e matriz de confusão."""
    st.title("📊 Avaliação do Sistema")
    if user_id is None:
        st.warning("Selecione um Usuário na aba 'Gerenciar Usuário' para calcular as métricas.")
        return

    st.subheader(f"Métricas para: {user_map.get(user_id, f'ID {user_id}')}")
    num_rec_eval = st.slider("Número de Recomendações para Avaliação:", 5, 20, 10, key="num_rec_slider_eval")

    try:
        response = requests.get(f"{BASE_URL}/metricas/{user_id}", params={"num_recomendacoes": num_rec_eval})
        response.raise_for_status()
        metricas = response.json()

        st.markdown("### Métricas de Classificação")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{float(metricas['precision']):.2%}")
        col2.metric("Recall", f"{float(metricas['recall']):.2%}")
        col3.metric("F1-Score", f"{float(metricas['f1_score']):.2f}")
        st.caption(f"Detalhes do cálculo: {metricas['detalhes']}")

        st.markdown("### Matriz de Confusão (Comparativo)")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Verdadeiros Positivos (TP): {len(metricas['tp_titulos'])}")
            st.expander("Ver Filmes").json(metricas['tp_titulos'])
            
            st.error(f"Falsos Negativos (FN): {len(metricas['fn_titulos'])}")
            st.expander("Ver Filmes").json(metricas['fn_titulos'])
        with col2:
            st.warning(f"Falsos Positivos (FP): {len(metricas['fp_titulos'])}")
            st.expander("Ver Filmes").json(metricas['fp_titulos'])

            st.info(f"Verdadeiros Negativos (TN): {len(metricas['tn_titulos'])}")
            st.expander("Ver Filmes").json(metricas['tn_titulos'])

    except requests.HTTPError as e:
        st.error(f"Erro no cálculo (Backend): {e.response.json().get('detail', 'Erro')}")
    except Exception as e:
        st.error(f"Erro de conexão: {e}")


# -------------------------------------------------------------
# 4. CONTROLE DO FLUXO PRINCIPAL
# -------------------------------------------------------------

def app():
    st.set_page_config(layout="wide", page_title="RICE - Recomendações")
    st.title("🎬 RICE: Sistema de Recomendação de Filmes")

    catalogo_df = get_catalogo()
    user_map, selectbox_options = get_usuarios_validos()
    
    active_user_id = None
    
    tab1, tab2, tab3, tab4 = st.tabs(["Gerenciar Usuário", "Adicionar Avaliação", "Recomendações", "Avaliação do Sistema"])

    with tab1:
        active_user_id = user_management_and_selection(user_map, selectbox_options)

    with tab2:
        add_evaluation_page(active_user_id, catalogo_df, user_map)

    with tab3:
        recommendation_page(active_user_id, catalogo_df, user_map)
        
    with tab4:
        evaluation_tab(active_user_id, user_map)


if __name__ == "__main__":
    app()