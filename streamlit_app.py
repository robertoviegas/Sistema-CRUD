from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, inspect

# Configurações
API_URL = "http://localhost:8000"
MLFLOW_URL = "http://localhost:5000"
TRAIN_FILE_PATH = Path(
    r"C:\Users\rvdutra\Documents\Sistema-CRUD\sistema-crud\data\05_model_input\train.csv"
)

st.set_page_config(
    page_title="Sistema CRUD - Treinamento", page_icon="🤖", layout="wide"
)

st.title("🤖 Sistema CRUD - Treinamento de Modelo")
st.markdown("---")

# Seção de upload de arquivo
st.header("📤 Upload do Arquivo de Treino")
st.markdown("Faça upload do arquivo CSV para treinar o modelo.")

uploaded_file = st.file_uploader(
    "Selecione o arquivo CSV de treino",
    type=["csv"],
    help="O arquivo será salvo em: sistema-crud/data/05_model_input/train.csv",
)

if uploaded_file is not None:
    # Mostrar preview do arquivo
    st.success(f"✅ Arquivo carregado: {uploaded_file.name}")

    # Mostrar preview dos dados
    import pandas as pd

    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Preview do Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"📈 Total de linhas: {len(df)} | Total de colunas: {len(df.columns)}")

        # Botão para salvar arquivo
        if st.button("💾 Salvar Arquivo", type="primary"):
            try:
                # Criar diretório se não existir
                TRAIN_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

                # Salvar arquivo
                with open(TRAIN_FILE_PATH, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"✅ Arquivo salvo com sucesso em: {TRAIN_FILE_PATH}")
            except Exception as e:
                st.error(f"❌ Erro ao salvar arquivo: {str(e)}")
    except Exception as e:
        st.error(f"❌ Erro ao ler arquivo CSV: {str(e)}")

st.markdown("---")

# Seção de treinamento
st.header("🚀 Treinar Modelo")
st.markdown("Execute o treinamento do modelo usando o arquivo salvo.")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("▶️ Iniciar Treinamento", type="primary", use_container_width=True):
        if not TRAIN_FILE_PATH.exists():
            st.error(
                "❌ Arquivo de treino não encontrado! Por favor, faça upload do arquivo primeiro."
            )
        else:
            with st.spinner("🔄 Treinando modelo... Isso pode levar alguns minutos."):
                try:
                    # Fazer requisição para o endpoint de treino
                    response = requests.post(f"{API_URL}/train", timeout=300)

                    if response.status_code == 200:
                        result = response.json()

                        st.success("✅ Treinamento concluído com sucesso!")

                        # Mostrar resultados
                        st.subheader("📊 Resultados do Treinamento")

                        # Métricas
                        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = (
                            st.columns(4)
                        )

                        with col_metrics1:
                            st.metric("MSE", f"{result.get('mse', 0):.4f}")

                        with col_metrics2:
                            st.metric("R²", f"{result.get('r2', 0):.4f}")

                        with col_metrics3:
                            st.metric("MAPE", f"{result.get('mape', 0):.2f}%")

                        with col_metrics4:
                            st.metric("MEAPE", f"{result.get('meape', 0):.2f}%")

                        # Informações do modelo
                        st.subheader("ℹ️ Informações do Modelo")
                        info_col1, info_col2, info_col3 = st.columns(3)

                        with info_col1:
                            st.write(f"**Model ID:** {result.get('model_id')}")
                            st.write(f"**Flavor:** {result.get('flavor')}")

                        with info_col2:
                            st.write(f"**Version:** {result.get('version')}")
                            st.write(
                                f"**Retraining ID:** {result.get('retraining_id')}"
                            )

                        with info_col3:
                            mlflow_run_id = result.get("mlflow_run_id")
                            if mlflow_run_id:
                                st.write(f"**MLflow Run ID:** {mlflow_run_id}")
                            else:
                                st.warning("⚠️ MLflow Run ID não disponível")

                        # Resposta completa (expansível)
                        with st.expander("📋 Resposta Completa da API"):
                            st.json(result)

                        # Botão para abrir MLflow
                        if mlflow_run_id:
                            st.markdown("---")
                            st.subheader("🔗 Acessar MLflow")
                            mlflow_link = (
                                f"{MLFLOW_URL}/#/experiments/0/runs/{mlflow_run_id}"
                            )
                            st.markdown(f"[🔗 Abrir MLflow em nova aba]({mlflow_link})")

                            # Auto-redirecionar se o usuário quiser
                            if st.button(
                                "🌐 Abrir MLflow Automaticamente",
                                use_container_width=True,
                            ):
                                st.markdown(
                                    f'<meta http-equiv="refresh" content="0; url={mlflow_link}">',
                                    unsafe_allow_html=True,
                                )
                                st.info("Redirecionando para MLflow...")
                    else:
                        st.error(f"❌ Erro no treinamento: {response.status_code}")
                        try:
                            error_data = response.json()
                            st.json(error_data)
                        except Exception:
                            st.text(response.text)

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Erro de conexão! Verifique se o servidor Flask está rodando na porta 8000."
                    )
                except requests.exceptions.Timeout:
                    st.error(
                        "❌ Timeout! O treinamento está demorando muito. Tente novamente."
                    )
                except Exception as e:
                    st.error(f"❌ Erro inesperado: {str(e)}")

with col2:
    # Verificar status do servidor
    st.subheader("🔍 Status do Servidor")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ Servidor Flask está online")
            health_data = health_response.json()
            st.json(health_data)
        else:
            st.warning("⚠️ Servidor respondeu com erro")
    except requests.exceptions.ConnectionError:
        st.error("❌ Servidor Flask não está acessível")
        st.info("💡 Execute: `python manage.py run` para iniciar o servidor")
    except Exception as e:
        st.error(f"❌ Erro ao verificar servidor: {str(e)}")

st.markdown("---")

# Seção de visualização do banco de dados
st.header("🗄️ Visualizar Banco de Dados")
st.markdown("Explore os dados armazenados no banco de dados SQLite.")

DB_PATH = Path(r"C:\Users\rvdutra\Documents\Sistema-CRUD\crud.db")
DB_URL = f"sqlite:///{DB_PATH}"

if DB_PATH.exists():
    try:
        engine = create_engine(DB_URL)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if tables:
            # Selecionar tabela para visualizar
            selected_table = st.selectbox(
                "Selecione a tabela para visualizar:",
                tables,
                help="Escolha qual tabela do banco de dados você deseja visualizar",
            )

            if selected_table:
                # Carregar dados da tabela
                # Tentar ordenar por id ou created_at, se existir
                try:
                    # Verificar colunas disponíveis
                    columns = inspector.get_columns(selected_table)
                    col_names = [col["name"] for col in columns]

                    # Determinar coluna de ordenação
                    order_by = None
                    if "id" in col_names:
                        order_by = "id DESC"
                    elif "created_at" in col_names:
                        order_by = "created_at DESC"

                    if order_by:
                        query = f"SELECT * FROM {selected_table} ORDER BY {order_by} LIMIT 100"
                    else:
                        query = f"SELECT * FROM {selected_table} LIMIT 100"

                    df = pd.read_sql_query(query, engine)
                except Exception:
                    # Se falhar, tentar query simples
                    query = f"SELECT * FROM {selected_table} LIMIT 100"
                    df = pd.read_sql_query(query, engine)

                if not df.empty:
                    st.subheader(f"📋 Dados da tabela: `{selected_table}`")
                    st.dataframe(df, use_container_width=True)
                    st.info(
                        f"📊 Mostrando {len(df)} registros (máximo 100 mais recentes)"
                    )

                    # Estatísticas básicas
                    with st.expander("📈 Estatísticas"):
                        st.write(f"**Total de registros na tabela:** {len(df)}")
                        st.write(f"**Colunas:** {', '.join(df.columns.tolist())}")
                        if "created_at" in df.columns:
                            st.write(
                                f"**Último registro:** {df['created_at'].max() if not df['created_at'].isna().all() else 'N/A'}"
                            )
                else:
                    st.warning(f"⚠️ A tabela `{selected_table}` está vazia.")

            # Botão para atualizar dados
            if st.button("🔄 Atualizar Dados", use_container_width=True):
                st.rerun()
        else:
            st.warning("⚠️ Nenhuma tabela encontrada no banco de dados.")
    except Exception as e:
        st.error(f"❌ Erro ao acessar banco de dados: {str(e)}")
        st.info("💡 Certifique-se de que o banco de dados existe e está acessível.")
else:
    st.error(f"❌ Banco de dados não encontrado em: {DB_PATH}")
    st.info("💡 Execute: `python manage.py init-db` para criar o banco de dados.")

st.markdown("---")

# Informações adicionais
st.sidebar.header("ℹ️ Informações")
st.sidebar.markdown(
    """
    ### Como usar:
    1. Faça upload do arquivo CSV de treino
    2. Clique em "Salvar Arquivo"
    3. Clique em "Iniciar Treinamento"
    4. Aguarde o resultado
    5. Acesse o MLflow para ver mais detalhes
    
    ### Endpoints:
    - **API:** http://localhost:8000
    - **MLflow:** http://localhost:5000
    
    ### Arquivo de treino:
    O arquivo será salvo em:
    `sistema-crud/data/05_model_input/train.csv`
    
    ### Visualizar Banco de Dados:
    - Use a seção "Visualizar Banco de Dados" para explorar os dados
    - Ou use DB Browser for SQLite:
      https://sqlitebrowser.org/
    """
)
