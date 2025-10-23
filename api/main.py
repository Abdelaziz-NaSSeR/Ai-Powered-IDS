from fastapi import FastAPI
import logging
import yaml
from pathlib import Path
from api.services.model_service import ModelService

logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI-Powered IDS API",
    description="Real-time intrusion detection system API",
    version="1.0.0"
)

# include routers
from api.routes import inference  # safe: api is a package because of api/__init__.py
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])


@app.get("/")
async def root():
    return {"message": "AI-Powered IDS API is running"}


@app.get("/health")
async def health_check():
    model_service = getattr(app.state, "model_service", None)
    model_loaded = bool(model_service and model_service.is_loaded)
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


@app.on_event("startup")
async def startup_event():
    """
    Create and attach ModelService on startup so routes can access it via request.app.state.model_service
    """
    try:
        app_root = Path(__file__).resolve().parent
        cfg_path = app_root / "config" / "config.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logger.exception("Failed to read config/config.yaml, using defaults.")
        cfg = {}

    # Instantiate the ModelService and attach to app state
    app.state.model_service = ModelService(config=cfg)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
