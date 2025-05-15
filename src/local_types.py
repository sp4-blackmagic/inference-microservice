from pydantic import BaseModel
from typing import Literal, Optional, List, Dict


class InferenceInfo(BaseModel):
    file_uid: str
    models: list[str]


class PredictionResult(BaseModel):
    status: Literal["success", "error"]
    prediction_encoded: Optional[List[float]] = None
    prediction_proba: Optional[List[float]] = None
    prediction_readable: Optional[str] = None
    message: Optional[str] = None


class ModelResult(BaseModel):
    ripeness: PredictionResult
    firmness: PredictionResult


class EvaluationResult(BaseModel):
    file_uid: str
    requested_models: List[str]
    results: Dict[str, ModelResult]
