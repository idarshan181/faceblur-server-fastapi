import numpy as np
import cv2
from io import BytesIO
from app.ml.utils import detect_faces_in_frame, save_result
from ultralytics import YOLO

def process_image_bytes(image_bytes, model, should_blur=True):
    """Process an uploaded image file for face detection and blurring."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    processed_image = detect_faces_in_frame(image, model, should_blur=should_blur)
    return processed_image

def process_video_bytes(video_bytes, model, should_blur=True):
    """Process an uploaded video file for face detection and blurring."""
    video = cv2.VideoCapture(BytesIO(video_bytes))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        processed_frame = detect_faces_in_frame(frame, model, should_blur=should_blur)
        frames.append(processed_frame)
    
    video.release()
    return frames, fps, frame_size
