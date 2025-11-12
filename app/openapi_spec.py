"""
Especificação OpenAPI para a API do Sistema CRUD
"""
import os

def get_openapi_spec():
    """Retorna a especificação OpenAPI com a URL da API configurada dinamicamente"""
    # Obter a URL da API do ambiente ou usar padrão
    # Em Docker, usar o nome do serviço; localmente usar localhost
    api_url = os.getenv("API_URL", "http://localhost:8000")
    # Se não estiver definido e estiver em Docker, usar o nome do serviço
    if api_url == "http://localhost:8000" and os.getenv("APP_ENV") == "dev":
        # Tentar detectar se estamos em Docker
        api_url = "http://api:8000" if os.path.exists("/.dockerenv") else "http://localhost:8000"
    
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Sistema CRUD API",
            "description": "API de predição com persistência, métricas por predição, retreino e troca de tipo de modelo (scikit-learn)",
            "version": "1.0.0",
        },
        "servers": [
            {
                "url": api_url,
                "description": "Servidor de produção"
            }
        ],
    "paths": {
        "/health": {
            "get": {
                "summary": "Verificar saúde da API",
                "description": "Retorna o status da API",
                "tags": ["Health"],
                "responses": {
                    "200": {
                        "description": "API está funcionando",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "ok"},
                                        "env": {"type": "string", "example": "dev"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/predict": {
            "post": {
                "summary": "Fazer predição",
                "description": "Realiza uma predição usando o modelo ativo e salva no banco de dados",
                "tags": ["Prediction"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["features"],
                                "properties": {
                                    "features": {
                                        "type": "object",
                                        "description": "Mapa de features para predição",
                                        "additionalProperties": True,
                                        "example": {
                                            "MSSubClass": 60,
                                            "LotFrontage": 65.0,
                                            "LotArea": 8450,
                                            "OverallQual": 7,
                                            "OverallCond": 5
                                        }
                                    },
                                    "y_true": {
                                        "type": "number",
                                        "description": "Valor real (opcional, usado para calcular métricas)",
                                        "example": 208500.0
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Predição realizada com sucesso",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "prediction_id": {"type": "string"},
                                        "prediction": {"type": "number"},
                                        "model_id": {"type": "integer"},
                                        "metrics": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "value": {"type": "number"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/predictions": {
            "get": {
                "summary": "Listar predições",
                "description": "Retorna uma lista paginada de predições",
                "tags": ["Prediction"],
                "parameters": [
                    {
                        "name": "page",
                        "in": "query",
                        "schema": {"type": "integer", "default": 1},
                        "description": "Número da página"
                    },
                    {
                        "name": "size",
                        "in": "query",
                        "schema": {"type": "integer", "default": 50, "maximum": 200},
                        "description": "Tamanho da página"
                    },
                    {
                        "name": "model_id",
                        "in": "query",
                        "schema": {"type": "integer"},
                        "description": "Filtrar por ID do modelo"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Lista de predições",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "model_id": {"type": "integer"},
                                            "prediction": {"type": "number"},
                                            "features": {"type": "object"},
                                            "created_at": {"type": "string", "format": "date-time"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/metrics": {
            "get": {
                "summary": "Listar métricas",
                "description": "Retorna métricas de predições",
                "tags": ["Metrics"],
                "parameters": [
                    {
                        "name": "prediction_id",
                        "in": "query",
                        "schema": {"type": "string"},
                        "description": "Filtrar por ID da predição"
                    },
                    {
                        "name": "name",
                        "in": "query",
                        "schema": {"type": "string"},
                        "description": "Filtrar por nome da métrica"
                    },
                    {
                        "name": "page",
                        "in": "query",
                        "schema": {"type": "integer", "default": 1},
                        "description": "Número da página"
                    },
                    {
                        "name": "size",
                        "in": "query",
                        "schema": {"type": "integer", "default": 100, "maximum": 500},
                        "description": "Tamanho da página"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Lista de métricas",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "prediction_id": {"type": "string"},
                                            "name": {"type": "string"},
                                            "value": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/models": {
            "get": {
                "summary": "Listar modelos",
                "description": "Retorna modelos registrados",
                "tags": ["Models"],
                "parameters": [
                    {
                        "name": "page",
                        "in": "query",
                        "schema": {"type": "integer", "default": 1},
                        "description": "Número da página"
                    },
                    {
                        "name": "size",
                        "in": "query",
                        "schema": {"type": "integer", "default": 50, "maximum": 200},
                        "description": "Tamanho da página"
                    },
                    {
                        "name": "flavor",
                        "in": "query",
                        "schema": {"type": "string"},
                        "description": "Filtrar por tipo de modelo (ex: sklearn)"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Lista de modelos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "flavor": {"type": "string"},
                                            "version": {"type": "string"},
                                            "model_path": {"type": "string", "nullable": True},
                                            "created_at": {"type": "string", "format": "date-time"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/retrainings": {
            "get": {
                "summary": "Listar retreinamentos",
                "description": "Retorna histórico de retreinamentos",
                "tags": ["Training"],
                "parameters": [
                    {
                        "name": "page",
                        "in": "query",
                        "schema": {"type": "integer", "default": 1},
                        "description": "Número da página"
                    },
                    {
                        "name": "size",
                        "in": "query",
                        "schema": {"type": "integer", "default": 50, "maximum": 200},
                        "description": "Tamanho da página"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Lista de retreinamentos",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "model_id": {"type": "integer"},
                                            "triggered_by": {"type": "string"},
                                            "notes": {"type": "string"},
                                            "created_at": {"type": "string", "format": "date-time"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/train": {
            "post": {
                "summary": "Treinar modelo",
                "description": "Executa o treinamento do modelo via pipeline Kedro",
                "tags": ["Training"],
                "responses": {
                    "200": {
                        "description": "Treinamento concluído",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model_id": {"type": "integer"},
                                        "flavor": {"type": "string"},
                                        "version": {"type": "string"},
                                        "model_path": {"type": "string", "nullable": True},
                                        "mse": {"type": "number"},
                                        "r2": {"type": "number"},
                                        "mape": {"type": "number"},
                                        "meape": {"type": "number"},
                                        "retraining_id": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/switch-model": {
            "post": {
                "summary": "Trocar tipo de modelo",
                "description": "Cria um novo modelo com o flavor especificado",
                "tags": ["Models"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["flavor"],
                                "properties": {
                                    "flavor": {
                                        "type": "string",
                                        "enum": ["sklearn"],
                                        "example": "sklearn"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Modelo trocado com sucesso",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model_id": {"type": "integer"},
                                        "flavor": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Flavor inválido",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "error": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/records/{table}/{item_id}": {
            "delete": {
                "summary": "Deletar registro",
                "description": "Deleta um registro de uma tabela específica",
                "tags": ["Records"],
                "parameters": [
                    {
                        "name": "table",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string",
                            "enum": ["predictions", "prediction_metrics", "models", "retrainings"]
                        },
                        "description": "Nome da tabela"
                    },
                    {
                        "name": "item_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "ID do registro"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Registro deletado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "deleted": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Erro na requisição",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "error": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    }

# Manter compatibilidade com código existente
OPENAPI_SPEC = get_openapi_spec()

