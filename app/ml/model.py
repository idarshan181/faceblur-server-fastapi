import mediapipe as mp
import numpy as np
from pathlib import Path
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self):
        self.model = None

    def load_model(self):
        model_path = Path(__file__).parent / "models" / "yolov11n-face.pt"
        logger.info(f"Attempting to load model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_content = f.read()
            logger.info(f"Successfully read {len(model_content)} bytes from model file")
            
            # self.model = mp.tasks.vision.FaceLandmarker.create_from_model_content(model_content)
            
            self.model = YOLO(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, input_data):
        if self.model is None:
            self.load_model()
        # YOLO model prediction
        return self.model(input_data)

        # Parse YOLO output for bounding boxes
        # face_boxes = []
        # if detection_result:
        #     for result in detection_result[0].boxes.data.tolist():
        #         x1, y1, x2, y2, score, class_id = result
        #         if score > 0.3:  # Adjust the confidence threshold as needed
        #             face_boxes.append((x1, y1, x2, y2, score))
        
        # return face_boxes

ml_model = MLModel()