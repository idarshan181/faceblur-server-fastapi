# video_processor.py
import boto3
import cv2
from app.core.settings.base import BaseAppSettings
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np
import logging
import tempfile
from datetime import datetime
from typing import Tuple, Dict, Optional
import os
from pathlib import Path
import json
import joblib
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
# aws_access_key_id = os.getenv('AWS_ACCESS_KEY')
# aws_secret_access_key = os.getenv('AWS_SECRET_KEY')
# aws_region = os.getenv('AWS_REGION')
# aws_s3_video_bucket = os.getenv('AWS_S3_VIDEO_BUCKET')
aws_access_key_id = "AKIA4MTWJX7JBINQ4VVD"
aws_secret_access_key = "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"
aws_region = "us-east-1"
aws_s3_video_bucket = "gesture-video-uploads-bucket"

settings = BaseAppSettings()

# Initialize the S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)

class VideoFormatError(Exception):
    """Custom exception for video format validation"""
    pass

def load_recognition_components(model_path: str, scaler_path: str, mappings_path: str):
    """Load all recognition components"""
    
    # Load Keras model
    model = load_model(model_path)

    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Load label mappings
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
        label_map = mappings['label_map']
        reverse_label_map = mappings['reverse_label_map']
    
    return model, scaler, label_map, reverse_label_map

class VideoProcessor:
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov']
    PREDICTION_INTERVAL = 2  # Now predicting every 2 frames
    
    def __init__(self, model_path: str, scaler_path: str, mappings_path: str):
        # Load all components
        self.model, self.scaler, self.label_map, self.reverse_label_map = \
            load_recognition_components(model_path, scaler_path, mappings_path)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        # Progress tracking
        self.progress = 0
        self.status_message = ""
    
    def preprocess_landmarks(self, landmarks) -> np.ndarray:
        """Preprocess landmarks for model input"""
        # Convert landmarks to flat array
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        landmarks_flat = landmarks_array.flatten()
        
        # Scale the features
        landmarks_scaled = self.scaler.transform(landmarks_flat.reshape(1, -1))

        # Reshape for the Keras model (batch_size, 21, 3)
        return landmarks_scaled.reshape(1, 21, 3)
    
    def predict_gesture(self, input_data: np.ndarray) -> Tuple[str, float]:
        """Make prediction using Keras model"""
        # Run prediction
        output_data = self.model.predict(input_data, verbose=0)
        
        # Get prediction and confidence
        prediction_idx = np.argmax(output_data[0])
        confidence = output_data[0][prediction_idx]
        
        # Convert to label
        predicted_label = self.reverse_label_map[str(prediction_idx)]
        
        return predicted_label, confidence
    
    def validate_video(self, file_path: str) -> None:
        """Validate video format and readability"""
        try:
            extension = Path(file_path).suffix.lower()
            if extension not in self.SUPPORTED_FORMATS:
                raise VideoFormatError(f"Unsupported video format. Supported formats: {self.SUPPORTED_FORMATS}")
            
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise VideoFormatError("Unable to open video file")
            
            # Check video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if width == 0 or height == 0 or fps == 0:
                raise VideoFormatError("Invalid video properties")
                
            cap.release()
            
        except Exception as e:
            logger.error(f"Video validation failed: {str(e)}")
            raise
    
    def create_3d_landmark_visualization(self, landmarks, frame_size: Tuple[int, int]) -> np.ndarray:
        """Create 3D landmark visualization frame"""
        frame_w, frame_h = frame_size
        landmark_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        
        # Draw connections
        for connection in self.mp_hands.HAND_CONNECTIONS:
            start_pos = landmarks.landmark[connection[0]]
            end_pos = landmarks.landmark[connection[1]]
            
            start_x = int(start_pos.x * frame_w)
            start_y = int(start_pos.y * frame_h)
            end_x = int(end_pos.x * frame_w)
            end_y = int(end_pos.y * frame_h)
            
            avg_z = ((start_pos.z + 1) / 2 + (end_pos.z + 1) / 2) / 2
            color_intensity = int(avg_z * 255)
            
            cv2.line(landmark_frame, (start_x, start_y), (end_x, end_y),
                    (0, color_intensity, 255-color_intensity), 2)
        
        # Draw landmarks
        for idx, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            
            z_normalized = (landmark.z + 1) / 2
            radius = int((1 - z_normalized) * 8) + 2
            color_intensity = int(z_normalized * 255)
            
            cv2.circle(landmark_frame, (x, y), radius, 
                     (0, color_intensity, 255-color_intensity), -1)
            
            # Add landmark ID
            cv2.putText(landmark_frame, str(idx), (x+5, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return landmark_frame
    
    def process_video(self, video_path: str) -> str:
        """Process video using Keras model and return path to processed video"""
        try:
            self.status_message = "Processing video..."
            self.progress = 30
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            output_path = tempfile.mktemp(suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
            
            frame_count = 0
            last_prediction = None
            last_confidence = 0.0
            frame_since_last_prediction = 0
            prediction_stats = {
                'total_predictions': 0,
                'successful_detections': 0
            }
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)
                
                # Create visualization frame
                landmark_frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    prediction_stats['successful_detections'] += 1
                    
                    # Draw landmarks on original frame
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Create 3D visualization
                    landmark_frame = self.create_3d_landmark_visualization(
                        hand_landmarks, (width, height)
                    )
                    
                    # Update prediction every 2 frames
                    if frame_since_last_prediction >= self.PREDICTION_INTERVAL:
                        # Preprocess landmarks and make prediction
                        input_data = self.preprocess_landmarks(hand_landmarks)
                        last_prediction, last_confidence = self.predict_gesture(input_data)
                        frame_since_last_prediction = 0
                        prediction_stats['total_predictions'] += 1
                    else:
                        frame_since_last_prediction += 1
                else:
                    frame_since_last_prediction += 1
                
                # Add prediction text
                if last_prediction is not None:
                    text = f"Gesture: {last_prediction} ({last_confidence:.2f})"
                    cv2.putText(frame, text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add title and stats to landmark frame
                cv2.putText(landmark_frame, "3D Hand Landmarks", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(landmark_frame, f"Frame: {frame_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Combine frames
                combined_frame = np.hstack((frame, landmark_frame))
                out.write(combined_frame)
                
                # Update progress
                frame_count += 1
                self.progress = 30 + int((frame_count / total_frames) * 60)
            
            # Log processing statistics
            logger.info(f"Video processing completed. Statistics: "
                       f"Total frames: {frame_count}, "
                       f"Total predictions: {prediction_stats['total_predictions']}, "
                       f"Successful detections: {prediction_stats['successful_detections']}")
            
            cap.release()
            out.release()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    def download_video(self, signed_url: str, s3_key: str) -> str:
        """Download video from S3 using signed URL"""
        try:
            self.status_message = "Downloading video..."
            self.progress = 10
            filename=str(((s3_key).split("/"))[-1])
            
            temp_path = f"app/gesturedetection/models/{filename}"

            s3_client.download_file(aws_s3_video_bucket, s3_key, temp_path)
            
            self.validate_video(temp_path)
            logger.info(f"Video downloaded successfully to {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise
    
    def upload_to_s3(self, file_path: str, message: Dict) -> str:
        """Upload file to S3 and return signed URL"""
        try:
            self.status_message = "Uploading processed video..."
            self.progress = 90
            filename="processed_" + str(((message['s3_key']).split("/"))[-1])
            s3_key = f"{message['user_id']}/{filename}"
            
            s3_client.upload_file(file_path, aws_s3_video_bucket, s3_key,
                                  ExtraArgs={'ContentType': 
                                             self.get_s3_file_content_type(
                                                aws_s3_video_bucket,message['s3_key'])})
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': aws_s3_video_bucket, 'Key': s3_key},
                ExpiresIn=3600
            )
            
            self.progress = 100
            self.status_message = "Processing completed"
            
            logger.info(f"Video uploaded successfully to S3: {s3_key}")
            return url
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise
    
    def get_progress(self) -> Dict[str, any]:
        """Get current progress and status"""
        return {
            "progress": self.progress,
            "status": self.status_message
        }
    
    def get_s3_file_content_type(self, bucket_name, s3_key):
        try:
            # Get the metadata for the file in S3
            response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            
            # Extract the Content-Type from the metadata
            content_type = response.get('ContentType', 'unknown')
            
            return content_type
        except Exception as e:
            logger.error(f"Error retrieving content type: {str(e)}")
            return None
