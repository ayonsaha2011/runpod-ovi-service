"""
Cloudflare R2 Storage Handler for video uploads
"""
import os
import boto3
import logging
import hashlib
from datetime import datetime
from botocore.config import Config
from typing import Optional

logger = logging.getLogger(__name__)


class R2StorageHandler:
    """
    Handles video uploads to Cloudflare R2 storage
    
    Environment Variables Required:
        R2_ACCOUNT_ID: Cloudflare account ID
        R2_ACCESS_KEY_ID: R2 access key
        R2_SECRET_ACCESS_KEY: R2 secret key
        R2_BUCKET_NAME: R2 bucket name
        R2_PUBLIC_URL: Public URL base for the bucket (optional, for custom domains)
    """
    
    def __init__(self):
        self.account_id = os.environ.get("R2_ACCOUNT_ID")
        self.access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
        self.secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.environ.get("R2_BUCKET_NAME", "ovi-videos")
        self.public_url_base = os.environ.get("R2_PUBLIC_URL")
        
        if not all([self.account_id, self.access_key_id, self.secret_access_key]):
            raise ValueError(
                "Missing required R2 environment variables: "
                "R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
            )
        
        # Initialize S3 client for R2
        self.endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
        
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"}
            ),
            region_name="auto"
        )
        
        logger.info(f"R2 Storage Handler initialized for bucket: {self.bucket_name}")
    
    def _generate_object_key(self, job_id: str, model: str) -> str:
        """Generate unique object key for video"""
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        hash_suffix = hashlib.md5(f"{job_id}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8]
        return f"videos/{timestamp}/{model}/{job_id}_{hash_suffix}.mp4"
    
    def upload_video(
        self,
        video_data: bytes,
        job_id: str,
        model: str,
        content_type: str = "video/mp4",
        metadata: Optional[dict] = None
    ) -> str:
        """
        Upload video to R2 and return public URL
        
        Args:
            video_data: Raw video bytes
            job_id: Unique job identifier
            model: Model variant used
            content_type: MIME type
            metadata: Optional metadata dict
            
        Returns:
            Public URL to the uploaded video
        """
        object_key = self._generate_object_key(job_id, model)
        
        extra_args = {
            "ContentType": content_type,
            "CacheControl": "public, max-age=31536000",  # 1 year cache
        }
        
        if metadata:
            extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}
        
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=video_data,
                **extra_args
            )
            
            # Generate public URL
            if self.public_url_base:
                url = f"{self.public_url_base.rstrip('/')}/{object_key}"
            else:
                # Use default R2 public URL format
                url = f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{object_key}"
            
            logger.info(f"Video uploaded successfully: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to upload video to R2: {e}")
            raise RuntimeError(f"Video upload failed: {str(e)}")
    
    def upload_video_file(
        self,
        file_path: str,
        job_id: str,
        model: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Upload video file from disk to R2
        
        Args:
            file_path: Path to video file
            job_id: Unique job identifier
            model: Model variant used
            metadata: Optional metadata dict
            
        Returns:
            Public URL to the uploaded video
        """
        with open(file_path, "rb") as f:
            video_data = f.read()
        
        return self.upload_video(
            video_data=video_data,
            job_id=job_id,
            model=model,
            metadata=metadata
        )
    
    def delete_video(self, url: str) -> bool:
        """
        Delete video from R2 by URL
        
        Args:
            url: Video URL
            
        Returns:
            True if deleted successfully
        """
        try:
            # Extract object key from URL
            if self.public_url_base and url.startswith(self.public_url_base):
                object_key = url[len(self.public_url_base):].lstrip("/")
            else:
                # Parse from default R2 URL format
                parts = url.split(f"{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/")
                if len(parts) > 1:
                    object_key = parts[1]
                else:
                    raise ValueError(f"Cannot parse object key from URL: {url}")
            
            self.client.delete_object(Bucket=self.bucket_name, Key=object_key)
            logger.info(f"Video deleted: {object_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete video: {e}")
            return False
    
    def get_presigned_url(self, object_key: str, expires_in: int = 3600) -> str:
        """
        Generate presigned URL for private bucket access
        
        Args:
            object_key: Object key in bucket
            expires_in: URL expiration in seconds
            
        Returns:
            Presigned URL
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": object_key},
            ExpiresIn=expires_in
        )


# Singleton instance
_storage_handler: Optional[R2StorageHandler] = None


def get_storage_handler() -> R2StorageHandler:
    """Get or create R2 storage handler singleton"""
    global _storage_handler
    if _storage_handler is None:
        _storage_handler = R2StorageHandler()
    return _storage_handler
