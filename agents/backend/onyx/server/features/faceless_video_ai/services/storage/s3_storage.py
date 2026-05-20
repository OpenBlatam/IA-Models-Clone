"""
AWS S3 Storage Service
"""

import os
from typing import Optional, BinaryIO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class S3Storage:
    """AWS S3 storage integration"""
    
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region: str = "us-east-1"
    ):
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.client = None
        
        if self.bucket_name and self.aws_access_key_id and self.aws_secret_access_key:
            try:
                import boto3
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region
                )
                logger.info("S3 storage initialized")
            except ImportError:
                logger.warning("boto3 not installed, S3 storage disabled")
            except Exception as e:
                logger.warning(f"S3 initialization failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if S3 is available"""
        return self.client is not None
    
    def upload_file(
        self,
        file_path: Path,
        s3_key: str,
        content_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload file to S3
        
        Args:
            file_path: Local file path
            s3_key: S3 object key
            content_type: Content type (optional)
            
        Returns:
            S3 URL if successful, None otherwise
        """
        if not self.is_available():
            return None
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.client.upload_file(
                str(file_path),
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"Uploaded file to S3: {s3_key}")
            return url
        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
            return None
    
    def upload_bytes(
        self,
        data: bytes,
        s3_key: str,
        content_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload bytes to S3
        
        Args:
            data: File data as bytes
            s3_key: S3 object key
            content_type: Content type (optional)
            
        Returns:
            S3 URL if successful, None otherwise
        """
        if not self.is_available():
            return None
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data,
                **extra_args
            )
            
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"Uploaded bytes to S3: {s3_key}")
            return url
        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
            return None
    
    def download_file(self, s3_key: str, local_path: Path) -> bool:
        """
        Download file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local file path
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self.client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            logger.info(f"Downloaded file from S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"S3 download failed: {str(e)}")
            return False
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"Deleted file from S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"S3 delete failed: {str(e)}")
            return False
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Get presigned URL for temporary access
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds
            
        Returns:
            Presigned URL if successful, None otherwise
        """
        if not self.is_available():
            return None
        
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}")
            return None


_s3_storage: Optional[S3Storage] = None


def get_s3_storage() -> S3Storage:
    """Get S3 storage instance (singleton)"""
    global _s3_storage
    if _s3_storage is None:
        _s3_storage = S3Storage()
    return _s3_storage

