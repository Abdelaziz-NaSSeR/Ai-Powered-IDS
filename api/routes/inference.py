# api/routes/inference.py
from fastapi import APIRouter, Request, HTTPException
import pandas as pd
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict")
async def predict(request: Request):
    ms = getattr(request.app.state, "model_service", None)
    if ms is None:
        raise HTTPException(500, "Model service not initialized")
    try:
        res = ms.predict_live_traffic()
        return {"status": "success", "predictions": res, "timestamp": pd.Timestamp.utcnow().isoformat()}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, str(e))

@router.get("/predictions/latest")
async def latest(request: Request, limit: int = 100):
    ms = getattr(request.app.state, "model_service", None)
    if ms is None:
        raise HTTPException(500, "Model service not initialized")
    return {"status": "success", "predictions": ms.get_latest_predictions(limit)}
