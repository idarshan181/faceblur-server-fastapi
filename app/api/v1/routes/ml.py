from typing import Dict
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from app.gesturedetection.video_queue_processor import VideoQueueProcessor
from app.ml.model import ml_model
import logging
import cv2
import os
from io import BytesIO
import numpy as np
from app.ml.process import process_image_bytes, process_video_bytes
from app.ml.utils import save_video
from tempfile import NamedTemporaryFile


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
            
            with NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_video_file:
                temp_video_file.write(contents)
                temp_video_file.flush()
                temp_video_path = temp_video_file.name
                
            frames, fps, frame_size = process_video_bytes(temp_video_path, ml_model, should_blur=should_blur)
            output_path = "/tmp/processed_video.mp4"
            save_video(frames, output_path, fps, frame_size)

            return FileResponse(path=output_path, filename="processed_video.mp4", media_type="video/mp4")


    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "OK"}

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



# @router.get("/play")
# async def play_large_file():
#     some_file_path = "app/assets/2.mp4"

#     def iterfile():  # 
#         with open(some_file_path, mode="rb") as file_like:  # 
#             yield from file_like  # 
    
#     return StreamingResponse(iterfile(), media_type="video/mp4")
@router.get("/video/{video_id}")
async def get_video_status(
    video_id: int,
    queue_processor: VideoQueueProcessor = Depends(VideoQueueProcessor)
):
    """Get the status of a processed video"""
    try:
        # Check Redis for processing status
        status_key = f"video_processing_status:{video_id}"
        status = queue_processor.redis_client.hgetall(status_key)
        
        # Check database for completed video
        video = Videos.get_by_id(video_id)
        
        if video and video.processed_video_url:
            return {
                "id": video.id,
                "status": "completed",
                "progress": 100,
                "processed_url": video.processed_video_url
            }
        elif status:
            return {
                "id": video_id,
                "status": status.get("status", "unknown"),
                "progress": int(status.get("progress", 0)),
                "processed_url": None
            }
        else:
            raise HTTPException(status_code=404, detail="Video not found")
            
    except Exception as e:
        logger.error(f"Error getting video status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-video")
async def process_video(
    video_data: Dict,
    background_tasks: BackgroundTasks,
    queue_processor: VideoQueueProcessor = Depends(VideoQueueProcessor)
):
    """Endpoint to trigger video processing"""
    try:
        background_tasks.add_task(queue_processor.process_queue_message, video_data)
        return {"message": "Video processing started", "video_id": video_data['video_id']}
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))