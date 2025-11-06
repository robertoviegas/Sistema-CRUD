from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = Field(default="dev", validation_alias="APP_ENV")
    db_url: str = Field(
        default="sqlite+pysqlite:///./crud.db", validation_alias="DB_URL"
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI"
    )
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    aws_access_key_id: str = Field(default="", validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(
        default="", validation_alias="AWS_SECRET_ACCESS_KEY"
    )
    model_flavor: str = Field(default="sklearn", validation_alias="MODEL_FLAVOR")

    class Config:
        env_file = ".env"
        extra = "ignore"
