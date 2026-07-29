"""
RAGEv Configuration Module.

Settings are loaded from a .env file. See .env.template for required variables.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAGEv experiment configuration."""

    # API credentials
    USER: str = "user"
    API_KEY: str = ""
    URL: str = "https://your-rag-api-endpoint/v1/chat-messages"

    # Processing
    MAX_THREADS: int = 1

    # Data paths
    DATADIR: str = "./data"
    EXPDIR_BINS: str = "./experiments/bins"
    EXPDIR_LOGS: str = "./experiments/logs"
    PAPERSDIR: str = "./data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
