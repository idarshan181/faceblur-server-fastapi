import numpy as np
import cv2
from io import BytesIO
from app.ml.utils import detect_faces_in_frame, save_result
from ultralytics import YOLO
import imageio
import os
from tempfile import NamedTemporaryFile


def process_image_bytes(image_bytes, model, should_blur=True):
    """Process an uploaded image file for face detection and blurring."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    processed_image = detect_faces_in_frame(image, model, should_blur=should_blur)
    return processed_image

# def process_video_bytes(video_bytes, model, should_blur=True):
#     """Process an uploaded video file for face detection and blurring."""
#     video = cv2.VideoCapture(BytesIO(video_bytes))
#     fps = video.get(cv2.CAP_PROP_FPS)
#     frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
#     frames = []
#     while video.isOpened():
#         ret, frame = video.read()
#         if not ret:
#             break
#         processed_frame = detect_faces_in_frame(frame, model, should_blur=should_blur)
#         frames.append(processed_frame)
    
#     video.release()
#     return frames, fps, frame_size

# def process_video_bytes(video_bytes, model, video_format: str = "mp4", should_blur: bool = True):
#     """
#     Process an uploaded video file for face detection and blurring.
    
#     Parameters:
#     - video_bytes: Byte stream of the video
#     - model: YOLO model for face detection
#     - video_format: Format of the video (e.g., "mp4", "avi")
#     - should_blur: Whether to blur detected faces (True/False)
    
#     Returns:
#     - frames: List of processed frames
#     - fps: Frames per second of the video
#     - frame_size: Size of the video frames (width, height)
#     """
#     # Read video from the in-memory byte stream with the specified format
#     reader = imageio.get_reader(BytesIO(video_bytes), format=video_format)
    
#     # Extract fps and frame size from video metadata
#     fps = reader.get_meta_data()["fps"]
#     frame_size = reader.get_meta_data()["size"]
    
#     frames = []
    
#     # Iterate through frames
#     for frame in reader:
#         # Convert frame (which is in RGB format) to BGR for OpenCV
#         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
#         # Detect and blur faces in the frame if requested
#         processed_frame = detect_faces_in_frame(frame_bgr, model, should_blur=should_blur)
        
#         # Append processed frame
#         frames.append(processed_frame)
    
#     reader.close()

#     return frames, fps, frame_size

def process_video_bytes(video_path, model, should_blur=True):
    """
    Process a video file for face detection and blurring.

    Parameters:
    - video_path: Path to the video file.
    - model: YOLO model for face detection.
    - should_blur: Whether to blur detected faces (True/False).

    Returns:
    - frames: List of processed frames.
    - fps: Frames per second of the video.
    - frame_size: Size of the video frames (width, height).
    """
    
    # Open the video file using OpenCV's VideoCapture
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video file was successfully opened
    if not video_capture.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get FPS and frame size from the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    frames = []
    
    # Process each frame in the video
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        # Detect and blur faces in the frame if requested
        processed_frame = detect_faces_in_frame(frame, model, should_blur=should_blur)
        frames.append(processed_frame)
    
    # Release the video capture object
    video_capture.release()
    
    return frames, fps, frame_size
