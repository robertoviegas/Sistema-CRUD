## Sistema-CRUD com Flask, Kedro e MLflow

API de predição com persistência, métricas por predição, drift/robustez, retreino e troca de tipo de modelo (scikit-learn). Usa MLflow e Kedro.

### Como rodar
1. Ative seu ambiente conda: `conda activate <seu_ambiente>`
2. pip install -r requirements.txt
3. Configure o arquivo `.env` (já criado com valores padrão)
4. python manage.py init-db
5. python manage.py run

### Variáveis (.env)
APP_ENV=dev
DB_URL=sqlite+pysqlite:///./crud.db
MLFLOW_TRACKING_URI=http://localhost:5001
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=changeme
AWS_SECRET_ACCESS_KEY=changeme
MODEL_FLAVOR=sklearn
# Evidently (opcional)
EVIDENTLY_BASELINE_CSV=
EVIDENTLY_WINDOW_SIZE=200
EVIDENTLY_MIN_SAMPLES=50
EVIDENTLY_FEATURE_KEYS=

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
- python manage.py run - Roda servidor Flask
- python manage.py train-kedro - Executa treino via Kedro
- python manage.py predict-csv train.csv --feature-cols "col1,col2,col3" --y-col "target" --limit 10 - Testa predições com CSV

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
- `train.csv` - Dataset de exemplo para testes

### Funcionalidades
- Predição com ID único e persistência
- Métricas por predição (incluindo erro quando y_true fornecido)
- Detecção de drift com Evidently (baseline configurável)
- Retreino via pipeline Kedro com registro no MLflow
- Troca de tipo de modelo (sklearn)
- Consultas paginadas e filtradas
- Carregamento de modelos do MLflow

