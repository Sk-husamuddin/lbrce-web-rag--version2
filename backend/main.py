import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config.settings import settings
from backend.api.routes import health, chat

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lbrce-assistant")

def create_app() -> FastAPI:
    """
    Application factory to create and configure the FastAPI app instance.
    """
    app = FastAPI(
        title="LBRCE AI Assistant",
        description="Stateless Agentic RAG application with live web retrieval for LBRCE college.",
        version="1.0.0"
    )

    # Configure CORS
    cors_origins = settings.CORS_ORIGINS if settings else ["*"]

    wildcard_cors = "*" in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        # Wildcard origins cannot be combined with credentialed CORS. Use
        # explicit CORS_ORIGINS in .env when cookie credentials are required.
        allow_credentials=not wildcard_cors,

        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve frontend static files when the directory is present. StaticFiles
    # raises at construction time if a backend-only deployment omits it.
    if os.path.isdir("frontend"):
        app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
    else:
        logger.warning("frontend/ directory not found; skipping static file mount.")

    # Register Routers
    app.include_router(health.router)
    app.include_router(chat.router)

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting LBRCE AI Assistant API...")
        # Validate configuration settings
        if settings is None:
            logger.warning("Settings could not be initialized. Running in degraded/mock mode.")
        else:
            provider = str(getattr(settings, "LLM_PROVIDER", "openrouter")).strip().lower()
            active_model = (
                getattr(settings, "GROQ_MODEL", "openai/gpt-oss-20b")
                if provider == "groq"
                else getattr(settings, "OPENROUTER_MODEL", "")
            )
            logger.info(
                "Configuration loaded. Index target: %s, LLM provider: %s, model: %s",
                settings.PINECONE_INDEX_NAME,
                provider,
                active_model,
            )

    return app


app = create_app()
