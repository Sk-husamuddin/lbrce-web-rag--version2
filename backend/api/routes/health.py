from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["system"])
async def health_check():
    """
    Health check endpoint to verify that the API is up and running.
    """
    return {"status": "ok"}
