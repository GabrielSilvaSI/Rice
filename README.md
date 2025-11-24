# 🍚 RICE: Recomendações Inteligentes para Cinema e Entretenimento

Este projeto foi desenvolvido como parte de uma avaliação acadêmica, seguindo as diretrizes de criar um sistema de recomendação completo e funcional. O RICE (Recomendações Inteligentes para Cinema e Entretenimento) é uma aplicação que utiliza técnicas de **filtragem baseada em conteúdo** para sugerir filmes aos usuários de forma personalizada e interativa.

## 🎯 Objetivo do Sistema

O objetivo principal do RICE é aplicar os conceitos de sistemas de recomendação para criar uma aplicação robusta, com as seguintes características:
* **Backend Robusto (FastAPI)**: Processa os dados, constrói perfis de usuário e serve as recomendações através de uma API RESTful.
* **Frontend Interativo (Streamlit)**: Permite que os usuários gerenciem perfis, obtenham recomendações, adicionem novas avaliações e visualizem a performance do modelo em tempo real.
* **Modelo Dinâmico**: O sistema constrói um perfil de usuário dinâmico com base em suas avaliações, permitindo que as recomendações se adaptem aos seus gostos.
* **Avaliação de Performance**: Inclui uma aba dedicada para calcular e visualizar métricas de performance do sistema (Precision, Recall, F1-Score) e analisar a matriz de confusão.

---

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e executar o RICE em sua máquina local.

### Pré-requisitos
* Python 3.8+
* Pip (Gerenciador de pacotes do Python)

### 1. Estrutura de Pastas
Certifique-se de que o projeto mantém a seguinte estrutura:

```
/Rice
├── /backend
│   ├── main.py
│   ├── recomendacao.py
│   └── requirements.txt
├── /frontend
│   ├── app.py
│   └── requirements.txt
├── /datasets
│   ├── filmes.csv
│   ├── avaliacoes.csv
│   └── usuarios.csv
└── README.md
```

### 2. Configuração e Execução do Backend
O backend é o cérebro do sistema e precisa ser iniciado primeiro.

```bash
# 1. Navegue até a pasta do backend
cd backend

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Inicie o servidor da API
uvicorn main:app --reload
```

O servidor estará rodando em `http://127.0.0.1:8000`. Deixe este terminal aberto.

### 3. Configuração e Execução do Frontend

O frontend é a interface com o usuário e deve ser executado em um novo terminal.

```bash
# 1. Navegue até a pasta do frontend (a partir da raiz do projeto)
cd frontend

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Inicie a aplicação web
streamlit run app.py
```

A aplicação estará disponível no seu navegador em `http://localhost:8501`.

## 🧠 Como o Sistema Funciona: A Lógica de Recomendação

O RICE utiliza **Filtragem Baseada em Conteúdo (Content-Based Filtering)**. A ideia central é que, se um usuário gostou de um determinado filme, ele provavelmente gostará de outros filmes com características *similares*.

O processo ocorre em quatro etapas principais:

1.  **Representação do Conteúdo**: Para cada filme no dataset, o sistema cria um "documento" de texto, chamado **Content Soup**, que combina seus atributos mais importantes: gênero, diretor, atores principais e sinopse (overview).

2.  **Vetorização com TF-IDF**: O sistema utiliza a técnica **TF-IDF (Term Frequency-Inverse Document Frequency)** para converter o "Content Soup" de cada filme em um vetor numérico. Isso cria uma matriz onde cada linha representa um filme e cada coluna representa a importância de uma palavra para aquele filme.

3.  **Construção do Perfil do Usuário**: Quando um usuário solicita recomendações, o sistema analisa todos os filmes que ele avaliou positivamente (nota "Gostei"). Em seguida, calcula a **média dos vetores TF-IDF** desses filmes. O vetor resultante é o **perfil do usuário**, representando numericamente seus gostos.

4.  **Geração de Recomendações**: O sistema calcula a **Similaridade de Cosseno** entre o vetor de perfil do usuário e os vetores de todos os outros filmes no catálogo. Os filmes com a maior pontuação de similaridade são retornados como recomendação.

### Métrica de Similaridade: Por que a Similaridade de Cosseno?

A Similaridade de Cosseno é ideal para este cenário, pois mede o ângulo entre dois vetores. Isso permite que o sistema identifique a semelhança de "direção" ou "gosto" entre o perfil do usuário e os filmes, independentemente da magnitude dos valores nos vetores.

## 📈 Avaliação de Performance: Medindo a Eficácia

Na aba **"Avaliação do Sistema"**, o RICE calcula métricas para medir a qualidade das recomendações para um usuário selecionado.

1.  **Geração das Recomendações**: O sistema gera uma lista de N recomendações para o usuário ativo.

2.  **Definição do Gabarito**: O "gabarito" é o conjunto de todos os filmes que o usuário avaliou como "Gostei" em seu histórico.

3.  **Cálculo da Matriz de Confusão**: O sistema compara as recomendações com o gabarito para classificar cada filme:
    *   **Verdadeiro Positivo (TP)**: Filme recomendado que o usuário realmente gostou.
    *   **Falso Positivo (FP)**: Filme recomendado que o usuário não gostou (ou não avaliou).
    *   **Falso Negativo (FN)**: Filme que o usuário gostou, mas que o sistema **não** recomendou.
    *   **Verdadeiro Negativo (TN)**: Filme que o usuário não gostou e que o sistema corretamente não recomendou.

4.  **Cálculo das Métricas**:
    *   **Precision (Precisão)**: Dos filmes recomendados, quantos foram acertos? `TP / (TP + FP)`
    *   **Recall (Revocação)**: De todos os filmes que o usuário gostou, quantos o sistema conseguiu recomendar? `TP / (TP + FN)`
    *   **F1-Score**: A média harmônica entre Precision e Recall, fornecendo uma métrica de performance balanceada.

## ⚙️ Funcionamento da API e Endpoints

O backend, construído com FastAPI, expõe uma API RESTful para interagir com o sistema.

*   `GET /itens`: Retorna o catálogo completo de filmes.
*   `GET /usuarios`: Retorna a lista de todos os usuários cadastrados.
*   `POST /usuarios`: Adiciona um novo usuário ao sistema.
*   `POST /avaliacoes`: Adiciona uma nova avaliação (gostei/não gostei) de um filme para um usuário.
*   `GET /avaliacoes/{usuario_id}`: Retorna o histórico de avaliações de um usuário específico.
*   `POST /recomendar`: Gera e retorna uma lista de filmes recomendados para um usuário.
*   `GET /metricas/{usuario_id}`: Calcula e retorna as métricas de performance (Precision, Recall, F1) e a matriz de confusão detalhada para um usuário.

## ✨ Funcionalidades do Frontend

A interface do RICE, desenvolvida com Streamlit, é organizada em abas para uma experiência de usuário clara e funcional:

*   **Gerenciar Usuário**: Permite selecionar um usuário ativo ou criar um novo. O usuário selecionado aqui é usado como contexto para todas as outras abas.
*   **Adicionar Avaliação**: Permite que o usuário ativo avalie filmes do catálogo, influenciando diretamente seu perfil de recomendação.
*   **Recomendações**: Gera e exibe uma lista de filmes recomendados em formato de cards, com pôster e pontuação de similaridade.
*   **Avaliação do Sistema**: Exibe as métricas de performance (Precision, Recall, F1-Score) em tempo real para o usuário ativo, além de gráficos e listas detalhadas da matriz de confusão (TP, FP, FN, TN).

---
