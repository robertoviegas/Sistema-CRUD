## Sistema-CRUD com Flask e Kedro

API de predição com persistência, métricas por predição, retreino e troca de tipo de modelo (scikit-learn). Usa Kedro para pipelines de treinamento e salva modelos localmente.

### Como rodar

#### Opção 1: Usando Docker (Recomendado)

O projeto está configurado para rodar em 3 containers:
- **PostgreSQL**: Banco de dados
- **API Flask**: Servidor REST
- **Streamlit**: Interface web

```bash
# 1. Construir e iniciar todos os containers
docker-compose up --build

# 2. Inicializar o banco de dados (em outro terminal)
docker-compose exec api python manage.py init-db

# 3. Executar migração se necessário
docker-compose exec api python manage.py migrate-db

# Acessar:
# - API: http://localhost:8000
# - Streamlit: http://localhost:8501
# - PostgreSQL: localhost:5432
```

**Comandos úteis:**
```bash
# Parar containers
docker-compose down

# Ver logs
docker-compose logs -f

# Reconstruir containers
docker-compose up --build

# Executar comandos no container da API
docker-compose exec api python manage.py <comando>
```

#### Opção 2: Desenvolvimento Local

1. Ative seu ambiente conda: `conda activate <seu_ambiente>`
2. pip install -r requirements.txt
3. Configure o arquivo `.env` (já criado com valores padrão)
4. python manage.py init-db
5. python manage.py run

### Variáveis (.env)
APP_ENV=dev
DB_URL=sqlite+pysqlite:///./crud.db
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=changeme
AWS_SECRET_ACCESS_KEY=changeme
MODEL_FLAVOR=sklearn

### Endpoints principais
- POST /predict - Predição com features (aceita y_true opcional)
- GET /predictions - Lista predições (com paginação e filtros)
- GET /metrics - Lista métricas por predição
- GET /models - Lista modelos registrados
- GET /retrainings - Lista retreinamentos
- POST /train - Treina modelo via pipeline Kedro
- POST /switch-model - Troca tipo de modelo (sklearn)
- DELETE /records/{table}/{id} - Deleta registro

### Comandos CLI
- python manage.py init-db - Inicializa banco de dados
- python manage.py migrate-db - Adiciona coluna model_path ao banco de dados existente (execute após remover MLflow)
- python manage.py run - Roda servidor Flask
- python manage.py train-kedro - Executa treino via Kedro
- python manage.py predict-csv train.csv --feature-cols "col1,col2,col3" --y-col "target" --limit 10 - Testa predições com CSV

### Interface Streamlit

#### Com Docker:
A interface Streamlit já está rodando automaticamente no container. Acesse: http://localhost:8501

#### Desenvolvimento Local:
1. Certifique-se de que o servidor Flask está rodando: `python manage.py run`
2. Execute: `streamlit run streamlit_app.py`
3. A interface abrirá no navegador onde você pode:
   - Fazer upload do arquivo CSV de treino
   - Salvar o arquivo no diretório correto
   - Executar o treinamento
   - Ver os resultados e métricas
   - Visualizar o banco de dados com todas as execuções

### Testando com train.csv
```bash
# Iniciar servidor
python manage.py init-db
python manage.py run

# Em outro terminal, testar predições
python manage.py predict-csv train.csv --feature-cols "MSSubClass,LotFrontage,LotArea,OverallQual,OverallCond,YearBuilt,YearRemodAdd,1stFlrSF,2ndFlrSF,GrLivArea,BsmtFullBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,GarageCars,GarageArea,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,MoSold,YrSold" --y-col "SalePrice" --limit 10
```

### Estrutura do projeto
- `app/` - API Flask, modelos DB, camada ML
- `sistema-crud/` - Projeto Kedro completo (pipelines, conf, data)
- `streamlit_app.py` - Interface web Streamlit
- `Dockerfile` - Container da API Flask
- `Dockerfile.streamlit` - Container do Streamlit
- `docker-compose.yml` - Orquestração dos 3 containers
- `train.csv` - Dataset de exemplo para testes

### Funcionalidades
- Predição com ID único e persistência
- Métricas por predição (incluindo erro quando y_true fornecido)
- Retreino via pipeline Kedro com salvamento local de modelos
- Troca de tipo de modelo (sklearn)
- Consultas paginadas e filtradas
- Carregamento de modelos salvos localmente (usando joblib)
- Visualização de métricas e histórico no Streamlit

