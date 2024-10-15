from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.ml.model import ml_model
import logging
import cv2
import os
from io import BytesIO
import numpy as np
from app.ml.process import process_image_bytes, process_video_bytes
from app.ml.utils import save_video


logger = logging.getLogger(__name__)

router = APIRouter()

class ModelStatus(BaseModel):
    is_loaded: bool


class PredictionResponse(BaseModel):
    status: str
    detail: str
    media_type: str
    media_size: tuple

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


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), should_blur: bool = Query(False, description="Set to True to blur faces, or False to only draw bounding boxes.")):
    try:
        contents = await file.read()
        filename = file.filename
        ext = filename.split('.')[-1].lower()

        if ext not in ["jpg", "jpeg", "png", "mp4", "avi", "mov"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # For images
        if ext in ["jpg", "jpeg", "png"]:
            processed_image = process_image_bytes(contents, ml_model, should_blur=should_blur)
            _, buffer = cv2.imencode(".jpg", processed_image)
            result_bytes = BytesIO(buffer)
            return StreamingResponse(result_bytes, media_type="image/jpeg")


        # For videos
        elif ext in ["mp4", "avi", "mov"]:
            frames, fps, frame_size = process_video_bytes(contents, ml_model, should_blur=should_blur)
            output_path = "/tmp/processed_video.mp4"
            save_video(frames, output_path, fps, frame_size)

            with open(output_path, "rb") as f:
                video_bytes = BytesIO(f.read())
            return StreamingResponse(video_bytes, media_type="video/mp4")

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")



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