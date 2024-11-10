from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from app.core.settings.base import BaseAppSettings
import os
from datetime import datetime


aws_access_key_id = os.getenv('AWS_ACCESS_KEY')
aws_secret_access_key = os.getenv('AWS_SECRET_KEY')
aws_region = os.getenv('AWS_REGION')
aws_s3_video_bucket = os.getenv('AWS_S3_VIDEO_BUCKET')

logger = logging.getLogger(__name__)
router = APIRouter()

settings = BaseAppSettings()

# Initialize the S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)

class PresignedURLRequest(BaseModel):
    user_id: str
    content_type: str = "video/mp4"

class PresignedURLResponse(BaseModel):
    url: str
    key: str

@router.post("/generate-presigned-url", response_model=PresignedURLResponse)
async def generate_presigned_url(request: PresignedURLRequest):
    """
    Generate a pre-signed URL for uploading a file to S3.
    """
    try:
        user_id = request.user_id
        content_type = request.content_type

        # Generate a timestamp-based filename
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}.mp4"
        s3_key = f"{user_id}/{filename}"

        # Generate a pre-signed URL for uploading the file
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": aws_s3_video_bucket,
                "Key": s3_key,
                "ContentType": content_type,
                "ACL": "public-read"
            },
            ExpiresIn=3600  # URL expiration time in seconds (1 hour)
        )

        logger.info(f"Generated presigned URL for file: {s3_key}")
        return PresignedURLResponse(url=presigned_url, key=s3_key)

    except NoCredentialsError:
        logger.error("AWS credentials not found", exc_info=True)
        raise HTTPException(status_code=403, detail="AWS credentials not found")
    except ClientError as e:
        logger.error(f"ClientError: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating presigned URL")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
