import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, inspect

# Configurações - usar variáveis de ambiente ou valores padrão
API_URL = os.getenv("API_URL", "http://localhost:8000")
TRAIN_FILE_PATH = Path(
    os.getenv("TRAIN_FILE_PATH", "/app/data/sistema-crud/data/05_model_input/train.csv")
)
DB_PATH = Path(os.getenv("DB_PATH", "/app/data/crud.db"))

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
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Preview do Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"📈 Total de linhas: {len(df)} | Total de colunas: {len(df.columns)}")

        # Mostrar colunas disponíveis para ajudar o usuário
        with st.expander("📋 Colunas Disponíveis no Dataset"):
            st.write("**Colunas:**", ", ".join(df.columns.tolist()))

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

# Campo para variável resposta
# Tentar carregar o dataset para mostrar as colunas disponíveis
available_columns = []
df_full = None
if TRAIN_FILE_PATH.exists():
    try:
        df_full = pd.read_csv(TRAIN_FILE_PATH)
        available_columns = df_full.columns.tolist()
    except Exception:
        pass

if available_columns:
    target_column = st.selectbox(
        "📊 Variável Resposta (Target Column)",
        options=available_columns,
        index=(
            available_columns.index("SalePrice")
            if "SalePrice" in available_columns
            else 0
        ),
        help="Selecione a coluna que será usada como variável resposta (target) no treinamento",
    )
else:
    target_column = st.text_input(
        "📊 Variável Resposta (Target Column)",
        value="SalePrice",
        help="Nome da coluna que será usada como variável resposta (target) no treinamento",
        placeholder="Ex: SalePrice, price, target, etc.",
    )

# Seção informativa sobre processamento dos dados
if df_full is not None and target_column:
    st.markdown("---")
    st.subheader("ℹ️ Processamento dos Dados (Pipeline Kedro)")

    # Analisar quais colunas serão usadas
    # Verificar se existe coluna "Id" (case-insensitive)
    id_col = None
    for col in df_full.columns:
        if col.lower() == "id":
            id_col = col
            break

    cols_to_remove = [target_column]
    if id_col:
        cols_to_remove.append(id_col)

    remaining_cols = [col for col in df_full.columns if col not in cols_to_remove]

    # Separar colunas numéricas e categóricas
    numeric_cols = (
        df_full[remaining_cols].select_dtypes(include=[np.number]).columns.tolist()
    )
    categorical_cols = (
        df_full[remaining_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    )

    # Informações sobre remoção
    with st.expander("📋 Detalhes do Processamento", expanded=True):
        st.markdown("### 🔄 Transformações Aplicadas:")

        # Remoção de colunas
        st.markdown("#### 1️⃣ Colunas Removidas:")
        removal_info = []
        if id_col:
            removal_info.append(
                f"**Coluna '{id_col}'**: Removida automaticamente (identificador, não é feature)"
            )
        if target_column in df_full.columns:
            removal_info.append(
                f"**Coluna '{target_column}'**: Removida (variável resposta/target)"
            )

        if removal_info:
            for info in removal_info:
                st.markdown(f"- {info}")
        else:
            st.info("Nenhuma coluna será removida.")

        # Variáveis numéricas
        st.markdown("#### 2️⃣ Variáveis Numéricas (Serão Usadas):")
        if numeric_cols:
            st.success(
                f"✅ **{len(numeric_cols)} colunas numéricas** serão usadas no treinamento:"
            )
            st.code(
                ", ".join(
                    numeric_cols[:10] + (["..."] if len(numeric_cols) > 10 else [])
                ),
                language=None,
            )
            if len(numeric_cols) > 10:
                st.caption(f"Total: {len(numeric_cols)} colunas numéricas")
            st.info(
                "💡 **Processamento**: Valores NaN serão preenchidos com 0 antes do treinamento."
            )
        else:
            st.warning(
                "⚠️ Nenhuma coluna numérica encontrada (além da variável resposta)."
            )

        # Variáveis categóricas
        st.markdown("#### 3️⃣ Variáveis Categóricas (Serão Ignoradas):")
        if categorical_cols:
            st.warning(
                f"⚠️ **{len(categorical_cols)} colunas categóricas** serão **ignoradas** no treinamento:"
            )
            st.code(
                ", ".join(
                    categorical_cols[:10]
                    + (["..."] if len(categorical_cols) > 10 else [])
                ),
                language=None,
            )
            if len(categorical_cols) > 10:
                st.caption(f"Total: {len(categorical_cols)} colunas categóricas")
            st.info(
                "💡 **Nota**: O pipeline atual processa apenas variáveis numéricas. Para usar variáveis categóricas, é necessário aplicar encoding (ex: One-Hot Encoding, Label Encoding) antes do treinamento."
            )
        else:
            st.success(
                "✅ Nenhuma coluna categórica encontrada. Todas as features são numéricas."
            )

        # Resumo
        st.markdown("---")
        st.markdown("### 📊 Resumo:")
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        with col_summary1:
            st.metric("Colunas Totais", len(df_full.columns))
        with col_summary2:
            st.metric("Features Numéricas", len(numeric_cols))
        with col_summary3:
            st.metric("Features Categóricas", len(categorical_cols))

        if len(numeric_cols) == 0:
            st.error(
                "❌ **Atenção**: Não há colunas numéricas disponíveis para treinamento (além da variável resposta). Verifique seu dataset."
            )

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("▶️ Iniciar Treinamento", type="primary", use_container_width=True):
        if not TRAIN_FILE_PATH.exists():
            st.error(
                "❌ Arquivo de treino não encontrado! Por favor, faça upload do arquivo primeiro."
            )
        elif not target_column or target_column.strip() == "":
            st.error(
                "❌ Por favor, informe o nome da variável resposta (target column)."
            )
        else:
            with st.spinner("🔄 Treinando modelo... Isso pode levar alguns minutos."):
                try:
                    # Fazer requisição para o endpoint de treino com target_column
                    payload = {"target_column": target_column.strip()}
                    response = requests.post(
                        f"{API_URL}/train", json=payload, timeout=300
                    )

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
                            model_path = result.get("model_path")
                            if model_path:
                                st.write(f"**Model Path:** {model_path}")
                            else:
                                st.warning("⚠️ Caminho do modelo não disponível")

                        # Resposta completa (expansível)
                        with st.expander("📋 Resposta Completa da API"):
                            st.json(result)
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
st.markdown("Explore os dados armazenados no banco de dados.")

# Usar PostgreSQL se DB_URL estiver configurado, caso contrário usar SQLite
DB_URL_ENV = os.getenv("DB_URL")
if DB_URL_ENV and DB_URL_ENV.startswith("postgresql"):
    # Usar PostgreSQL
    DB_URL = DB_URL_ENV
    db_type = "PostgreSQL"
else:
    # Usar SQLite como fallback
    DB_URL = f"sqlite:///{DB_PATH}"
    db_type = "SQLite"

# Verificar se o banco existe (apenas para SQLite)
if db_type == "SQLite" and not DB_PATH.exists():
    st.error(f"❌ Banco de dados não encontrado em: {DB_PATH}")
    st.info("💡 Execute: `python manage.py init-db` para criar o banco de dados.")
else:
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
        st.error(f"❌ Erro ao acessar banco de dados ({db_type}): {str(e)}")
        st.info("💡 Certifique-se de que o banco de dados existe e está acessível.")

st.markdown("---")

# Seção de documentação Swagger
st.header("📚 Documentação da API (Swagger)")
st.markdown("Explore e teste todos os endpoints da API usando a interface Swagger.")

# Verificar se a API está acessível antes de mostrar o Swagger
try:
    health_check = requests.get(f"{API_URL}/health", timeout=5)
    if health_check.status_code == 200:
        # Converter API_URL para localhost se estiver usando api:8000
        if "api:8000" in API_URL:
            swagger_url = "http://localhost:8000/swagger"
            api_base_url = "http://localhost:8000"
        else:
            swagger_url = f"{API_URL}/swagger"
            api_base_url = API_URL

        # Status e links
        col_status, col_link = st.columns([2, 1])
        with col_status:
            st.success("✅ API está online")
        with col_link:
            st.markdown(f"[🔗 Abrir Swagger em nova aba]({swagger_url})")

        # Opção para mostrar/ocultar iframe
        show_iframe = st.checkbox(
            "📺 Mostrar Swagger UI embutido",
            value=False,
            help="Marque para exibir o Swagger UI diretamente nesta página",
        )

        if show_iframe:
            # Usar expander para melhor controle
            with st.expander(
                "📚 Swagger UI - Interface de Documentação", expanded=True
            ):
                # Informação sobre o iframe
                st.info(
                    "💡 **Dica:** Se o iframe não carregar corretamente, use o link acima para abrir em nova aba."
                )

                # Iframe com altura ajustada e melhor configuração
                try:
                    st.components.v1.iframe(src=swagger_url, height=700, scrolling=True)
                except Exception as iframe_error:
                    st.warning(f"⚠️ Erro ao carregar iframe: {str(iframe_error)}")
                    st.markdown(
                        f"**Por favor, acesse diretamente:** [{swagger_url}]({swagger_url})"
                    )

        # Informações adicionais em colunas
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔗 Links úteis:**")
            st.markdown(f"- [Swagger UI]({swagger_url})")
            st.markdown(f"- [Health Check]({api_base_url}/health)")
            st.markdown(f"- [OpenAPI Spec]({api_base_url}/openapi.json)")

        with col2:
            st.markdown("**📖 Endpoints principais:**")
            st.markdown("- `/predict` - Fazer predições")
            st.markdown("- `/train` - Treinar modelo")
            st.markdown("- `/models` - Listar modelos")
            st.markdown("- `/predictions` - Ver predições")
    else:
        st.warning("⚠️ API respondeu com erro. Verifique os logs.")
except requests.exceptions.ConnectionError:
    st.error("❌ API não está acessível. Verifique se o servidor Flask está rodando.")
    st.info(f"💡 A URL configurada é: {API_URL}")
except Exception as e:
    st.error(f"❌ Erro ao verificar API: {str(e)}")

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
    5. Visualize as métricas e o banco de dados
    
    ### Endpoints:
    - **API:** {API_URL}
    - **Swagger UI:** {API_URL}/swagger
    - **Streamlit:** http://localhost:8501
    
    ### Arquivo de treino:
    O arquivo será salvo em:
    `{TRAIN_FILE_PATH}`
    
    ### Banco de Dados:
    - Tipo: {db_type}
    - Use a seção "Visualizar Banco de Dados" para explorar os dados
    """.format(
        API_URL=API_URL, TRAIN_FILE_PATH=TRAIN_FILE_PATH, db_type=db_type
    )
)
