# RICE 🍚
## Recomendações Inteligentes para Cinema e Entretenimento
---

## Como Iniciar o Projeto

Este projeto é dividido em duas partes: Backend (FastAPI) e Frontend (Streamlit). Você precisará de dois terminais para rodar a aplicação.

### 1. Backend (API)

Abra um terminal, navegue até a pasta raiz do projeto e execute:

```bash
# Instalar dependências
pip install -r backend/requirements.txt

# Iniciar o servidor
cd backend
uvicorn main:app --reload
```

O backend estará rodando em: `http://127.0.0.1:8000`

### 2. Frontend (Interface)

Abra um **segundo terminal**, navegue até a pasta raiz do projeto e execute:

```bash
# Instalar dependências
pip install -r frontend/requirements.txt

# Iniciar a interface
cd frontend
streamlit run app.py
```

O frontend abrirá automaticamente no seu navegador (geralmente em `http://localhost:8501`).
