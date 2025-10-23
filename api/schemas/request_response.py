# api/schemas/request_response.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class PredictionItem(BaseModel):
    flow_id: int
    prediction: str
    confidence: Optional[float]
    timestamp: datetime
    features: Optional[Dict[str, Any]]

class PredictionResponse(BaseModel):
    timestamp: datetime
    predictions: List[PredictionItem]
