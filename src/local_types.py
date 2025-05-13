from pydantic import BaseModel
from typing import Literal, Optional, List, Dict


class InferenceInfo(BaseModel):
    file_uid: str
    models: list[str]
# Model for the result of a single inference run (for one model)


class ModelResult(BaseModel):
    # Status is expected to be either "success" or "error"
    status: Literal["success", "error"]

    # Prediction is a list of floats if status is "success", otherwise it's None
    # We use Optional[List[float]] to indicate it can be a list of floats or None
    prediction: Optional[List[float]] = None

    # Message is a string only if status is "error", otherwise it's None
    # We use Optional[str] to indicate it can be a string or None
    message: Optional[str] = None

# Model for the overall response body


class EvaluationResult(BaseModel):
    file_uid: str
    requested_models: List[str]

    # The 'results' field is a dictionary
    # The keys are strings (the model names)
    # The values are instances of the ModelResult model defined above
    results: Dict[str, ModelResult]
