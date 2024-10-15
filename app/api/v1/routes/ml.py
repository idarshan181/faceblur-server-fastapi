from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ml.model import ml_model
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class ModelStatus(BaseModel):
    is_loaded: bool

@router.get("/status", response_model=ModelStatus)
async def get_model_status():
    try:
        if ml_model.model is None:
            logger.info("Model not loaded. Attempting to load...")
            ml_model.load_model()
        logger.info("Model loaded successfully")
        return ModelStatus(is_loaded=True)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# We'll keep the POST route commented out for now
# 
# class PredictionInput(BaseModel):
#     data: list
# 
# class PredictionOutput(BaseModel):
#     result: list
# 
# @router.post("/predict", response_model=PredictionOutput)
# async def predict(input_data: PredictionInput):
#     try:
#         validate_input(input_data.data)
#         processed_input = preprocess_input(input_data.data)
#         prediction = ml_model.predict(processed_input)
#         processed_output = postprocess_output(prediction)
#         return PredictionOutput(result=processed_output)
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Prediction failed")