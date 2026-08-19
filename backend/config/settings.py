import json
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings managed via pydantic-settings.
    Loads configurations from environment variables and an optional .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Answer-generation provider configuration.
    # Supported values are "openrouter" and "groq".
    LLM_PROVIDER: str = Field("openrouter", alias="LLM_PROVIDER")

    # OpenRouter remains available for answer generation and image ingestion.
    OPENROUTER_API_KEY: str = Field("", alias="OPENROUTER_API_KEY")
    OPENROUTER_API_KEY_2: str = Field("", alias="OPENROUTER_API_KEY_2")  # optional 2nd key,
                                                                            # used by image_parser.py
                                                                            # to round-robin free-tier
                                                                            # rate limits during batch
                                                                            # image ingestion
    OPENROUTER_MODEL: str = Field("meta-llama/llama-3-8b-instruct:free", alias="OPENROUTER_MODEL")
    OPENROUTER_BASE_URL: str = Field("https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    # Groq OpenAI-compatible API configuration for low-latency answer generation.
    GROQ_API_KEY: str = Field("", alias="GROQ_API_KEY")
    GROQ_MODEL: str = Field("openai/gpt-oss-20b", alias="GROQ_MODEL")
    GROQ_BASE_URL: str = Field("https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")

    # Pinecone Configuration
    PINECONE_API_KEY: str = Field(..., alias="PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = Field("lbrce-index", alias="PINECONE_INDEX_NAME")
    PINECONE_NAMESPACE: str = Field("", alias="PINECONE_NAMESPACE")

    # Query/document embedding configuration. Production defaults preserve the
    # legacy Pinecone inference path; the migration copy sets EMBEDDING_PROVIDER=local.
    EMBEDDING_PROVIDER: str = Field("pinecone", alias="EMBEDDING_PROVIDER")
    EMBEDDING_MODEL: str = Field("BAAI/bge-large-en-v1.5", alias="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(1024, alias="EMBEDDING_DIMENSION")

    # Tavily Configuration
    TAVILY_API_KEY: str = Field(..., alias="TAVILY_API_KEY")

    # College Configuration
    LBRCE_BASE_URL: str = Field("https://www.lbrce.ac.in", alias="LBRCE_BASE_URL")

    # RAG evidence evaluation
    RAG_RELEVANCE_THRESHOLD: float = Field(0.65, alias="RAG_RELEVANCE_THRESHOLD")

    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(default=["*"], alias="CORS_ORIGINS")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            try:
                # Attempt to parse as JSON list: ["http://localhost:3000"]
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                # Fallback to comma-separated list
                return [item.strip() for item in v.split(",") if item.strip()]
        return v


# Global settings instance
try:
    settings = Settings()
except Exception as e:
    # During testing or early local setup, we might not have a .env file yet.
    # We allow importing Settings class, but warn if instantiation fails.
    import sys
    # Only raise exception if running the app directly (not during testing with mocks or imports)
    if "pytest" not in sys.modules:
        print(f"Error loading configuration. Ensure your .env file is set up: {e}", file=sys.stderr)
        raise e
    settings = None  # Fallback for unit testing imports