# video_queue_processor.py
from datetime import datetime
from redis import Redis
from typing import Dict
import json
import os
import logging
from urllib.parse import urlparse

from app.core import settings
from app.core.settings.base import BaseAppSettings
from app.db.queries.tables import Videos
from .video_processor import VideoProcessor, VideoFormatError

redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_db = os.getenv('REDIS_DB')
gesture_model_path = os.getenv('GESTURE_MODEL_PATH')
scaler_path = os.getenv('SCALER_PATH')
mappings_path = os.getenv('MAPPINGS_PATH')
# aws_access_key_id = os.getenv('AWS_ACCESS_KEY')
# aws_secret_access_key = os.getenv('AWS_SECRET_KEY')
# aws_region = os.getenv('AWS_REGION')
# aws_s3_video_bucket = os.getenv('AWS_S3_VIDEO_BUCKET')

aws_access_key_id = "AKIA4MTWJX7JBINQ4VVD"
aws_secret_access_key = "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"
aws_region = "us-east-1"
aws_s3_video_bucket = "gesture-video-uploads-bucket"

logger = logging.getLogger(__name__)
settings = BaseAppSettings()

class VideoQueueProcessor:
    def __init__(self):
        self.redis_client = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.video_processor = VideoProcessor(
    model_path=gesture_model_path, scaler_path=scaler_path, mappings_path=mappings_path
)
    
    def process_queue_message(self, message: Dict) -> None:
        """Process a message from the Redis queue"""
        try:
            video_id = message['video_id']
            s3_url = message['s3_url']
            user_id = message['user_id']
            s3_key = message['s3_key']
            
            logger.info(f"Starting processing for video ID: {video_id}")
            
            # Update status in Redis
            self.update_processing_status(video_id, "downloading", 0)
            
            # Download and process video
            try:
                local_video_path = self.video_processor.download_video(s3_url, s3_key)
                self.update_processing_status(video_id, "processing", 30)
                
                processed_video_path = self.video_processor.process_video(local_video_path)
                self.update_processing_status(video_id, "uploading", 80)
                
                # Upload processed video
                processed_url = self.video_processor.upload_to_s3(processed_video_path,message)
                
                # Update database here
                # video = Videos(processed_video_url=processed_url)
                # video.save()
                
                self.update_processing_status(video_id, "completed", 100)
                logger.info(f"Video processing completed for ID: {video_id}")
                
            finally:
                # Cleanup temporary files
                for path in [local_video_path, processed_video_path]:
                    if path and os.path.exists(path):
                        pass
                        os.remove(path)
                        
        except VideoFormatError as e:
            logger.error(f"Video format error for ID {video_id}: {str(e)}")
            self.update_processing_status(video_id, f"failed: {str(e)}", -1)
            
        except Exception as e:
            logger.error(f"Error processing video ID {video_id}: {str(e)}")
            self.update_processing_status(video_id, f"failed: {str(e)}", -1)
    
    def update_processing_status(self, video_id: str, status: str, progress: int) -> None:
        """Update processing status in Redis"""
        status_key = f"video_processing_status:{video_id}"
        mappings = json.dumps({
            "status": status,
            "progress": progress,
            "updated_at": datetime.utcnow().isoformat()
        })

        self.redis_client.set(status_key, value=mappings)

        self.redis_client.expire(status_key, 86400)  # Expire after 24 hours
