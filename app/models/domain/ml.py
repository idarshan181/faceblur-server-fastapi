from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ml.model import ml_model

router = APIRouter()

class ModelStatus(BaseModel):
    is_loaded: bool

@router.get("/status", response_model=ModelStatus)
async def get_model_status():
    try:
        # Attempt to load the model if it's not already loaded
        if ml_model.model is None:
            ml_model.load_model()
        # If we reach this point, the model is loaded successfully
        return ModelStatus(is_loaded=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")