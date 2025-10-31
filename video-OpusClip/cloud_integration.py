"""
Cloud Integration for Ultimate Opus Clip

Advanced cloud integration for storage, processing, and AI services
across multiple cloud providers.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import time
import json
import base64
from dataclasses import dataclass
from enum import Enum
import structlog
from pathlib import Path
import aiohttp
import aiofiles
import boto3
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient
import requests
from concurrent.futures import ThreadPoolExecutor

logger = structlog.get_logger("cloud_integration")

class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GOOGLE = "google"
    AZURE = "azure"
    CLOUDFLARE = "cloudflare"
    DIGITAL_OCEAN = "digital_ocean"

class ServiceType(Enum):
    """Types of cloud services."""
    STORAGE = "storage"
    COMPUTE = "compute"
    AI = "ai"
    CDN = "cdn"
    DATABASE = "database"

@dataclass
class CloudConfig:
    """Configuration for cloud services."""
    provider: CloudProvider
    service_type: ServiceType
    region: str
    credentials: Dict[str, Any]
    bucket_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class CloudFile:
    """Represents a file in cloud storage."""
    key: str
    url: str
    size: int
    last_modified: float
    content_type: str
    metadata: Dict[str, Any] = None

class AWSIntegration:
    """AWS cloud integration."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.s3_client = None
        self.lambda_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS clients."""
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.config.region,
                aws_access_key_id=self.config.credentials.get('access_key'),
                aws_secret_access_key=self.config.credentials.get('secret_key')
            )
            
            self.lambda_client = boto3.client(
                'lambda',
                region_name=self.config.region,
                aws_access_key_id=self.config.credentials.get('access_key'),
                aws_secret_access_key=self.config.credentials.get('secret_key')
            )
            
            logger.info("AWS clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise
    
    async def upload_file(self, file_path: str, key: str) -> CloudFile:
        """Upload file to S3."""
        try:
            with open(file_path, 'rb') as file:
                self.s3_client.upload_fileobj(
                    file,
                    self.config.bucket_name,
                    key
                )
            
            # Get file info
            response = self.s3_client.head_object(
                Bucket=self.config.bucket_name,
                Key=key
            )
            
            return CloudFile(
                key=key,
                url=f"https://{self.config.bucket_name}.s3.{self.config.region}.amazonaws.com/{key}",
                size=response['ContentLength'],
                last_modified=response['LastModified'].timestamp(),
                content_type=response['ContentType'],
                metadata=response.get('Metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise
    
    async def download_file(self, key: str, local_path: str) -> bool:
        """Download file from S3."""
        try:
            self.s3_client.download_file(
                self.config.bucket_name,
                key,
                local_path
            )
            return True
        except Exception as e:
            logger.error(f"Error downloading file from S3: {e}")
            return False
    
    async def list_files(self, prefix: str = "") -> List[CloudFile]:
        """List files in S3 bucket."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append(CloudFile(
                    key=obj['Key'],
                    url=f"https://{self.config.bucket_name}.s3.{self.config.region}.amazonaws.com/{obj['Key']}",
                    size=obj['Size'],
                    last_modified=obj['LastModified'].timestamp(),
                    content_type="application/octet-stream"
                ))
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files from S3: {e}")
            return []
    
    async def delete_file(self, key: str) -> bool:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(
                Bucket=self.config.bucket_name,
                Key=key
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting file from S3: {e}")
            return False
    
    async def invoke_lambda(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke AWS Lambda function."""
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read())
            return result
            
        except Exception as e:
            logger.error(f"Error invoking Lambda function: {e}")
            return {"error": str(e)}

class GoogleCloudIntegration:
    """Google Cloud integration."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.storage_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Cloud client."""
        try:
            self.storage_client = gcs.Client.from_service_account_info(
                self.config.credentials
            )
            logger.info("Google Cloud client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud client: {e}")
            raise
    
    async def upload_file(self, file_path: str, key: str) -> CloudFile:
        """Upload file to Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)
            blob = bucket.blob(key)
            
            blob.upload_from_filename(file_path)
            
            # Get file info
            blob.reload()
            
            return CloudFile(
                key=key,
                url=blob.public_url,
                size=blob.size,
                last_modified=blob.time_created.timestamp(),
                content_type=blob.content_type,
                metadata=blob.metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Error uploading file to GCS: {e}")
            raise
    
    async def download_file(self, key: str, local_path: str) -> bool:
        """Download file from Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)
            blob = bucket.blob(key)
            
            blob.download_to_filename(local_path)
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file from GCS: {e}")
            return False
    
    async def list_files(self, prefix: str = "") -> List[CloudFile]:
        """List files in Google Cloud Storage bucket."""
        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            files = []
            for blob in blobs:
                files.append(CloudFile(
                    key=blob.name,
                    url=blob.public_url,
                    size=blob.size,
                    last_modified=blob.time_created.timestamp(),
                    content_type=blob.content_type,
                    metadata=blob.metadata or {}
                ))
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files from GCS: {e}")
            return []
    
    async def delete_file(self, key: str) -> bool:
        """Delete file from Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)
            blob = bucket.blob(key)
            
            blob.delete()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file from GCS: {e}")
            return False

class AzureIntegration:
    """Azure cloud integration."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.blob_service_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure client."""
        try:
            connection_string = self.config.credentials.get('connection_string')
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            logger.info("Azure client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure client: {e}")
            raise
    
    async def upload_file(self, file_path: str, key: str) -> CloudFile:
        """Upload file to Azure Blob Storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.config.bucket_name,
                blob=key
            )
            
            with open(file_path, 'rb') as file:
                blob_client.upload_blob(file, overwrite=True)
            
            # Get blob properties
            properties = blob_client.get_blob_properties()
            
            return CloudFile(
                key=key,
                url=blob_client.url,
                size=properties.size,
                last_modified=properties.last_modified.timestamp(),
                content_type=properties.content_settings.content_type,
                metadata=properties.metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Error uploading file to Azure: {e}")
            raise
    
    async def download_file(self, key: str, local_path: str) -> bool:
        """Download file from Azure Blob Storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.config.bucket_name,
                blob=key
            )
            
            with open(local_path, 'wb') as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file from Azure: {e}")
            return False
    
    async def list_files(self, prefix: str = "") -> List[CloudFile]:
        """List files in Azure Blob Storage."""
        try:
            container_client = self.blob_service_client.get_container_client(self.config.bucket_name)
            blobs = container_client.list_blobs(name_starts_with=prefix)
            
            files = []
            for blob in blobs:
                files.append(CloudFile(
                    key=blob.name,
                    url=f"https://{self.config.credentials.get('account_name')}.blob.core.windows.net/{self.config.bucket_name}/{blob.name}",
                    size=blob.size,
                    last_modified=blob.last_modified.timestamp(),
                    content_type=blob.content_settings.content_type if blob.content_settings else "application/octet-stream",
                    metadata=blob.metadata or {}
                ))
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files from Azure: {e}")
            return []
    
    async def delete_file(self, key: str) -> bool:
        """Delete file from Azure Blob Storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.config.bucket_name,
                blob=key
            )
            
            blob_client.delete_blob()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file from Azure: {e}")
            return False

class CloudIntegrationManager:
    """Manages multiple cloud integrations."""
    
    def __init__(self):
        self.integrations: Dict[CloudProvider, Any] = {}
        self.default_provider: Optional[CloudProvider] = None
        
        logger.info("Cloud integration manager initialized")
    
    def add_integration(self, provider: CloudProvider, config: CloudConfig):
        """Add a cloud integration."""
        try:
            if provider == CloudProvider.AWS:
                integration = AWSIntegration(config)
            elif provider == CloudProvider.GOOGLE:
                integration = GoogleCloudIntegration(config)
            elif provider == CloudProvider.AZURE:
                integration = AzureIntegration(config)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            self.integrations[provider] = integration
            
            if self.default_provider is None:
                self.default_provider = provider
            
            logger.info(f"Added {provider.value} integration")
            
        except Exception as e:
            logger.error(f"Failed to add {provider.value} integration: {e}")
            raise
    
    def get_integration(self, provider: Optional[CloudProvider] = None) -> Any:
        """Get cloud integration."""
        provider = provider or self.default_provider
        
        if provider not in self.integrations:
            raise ValueError(f"No integration found for provider: {provider}")
        
        return self.integrations[provider]
    
    async def upload_file(self, file_path: str, key: str, provider: Optional[CloudProvider] = None) -> CloudFile:
        """Upload file to cloud storage."""
        integration = self.get_integration(provider)
        return await integration.upload_file(file_path, key)
    
    async def download_file(self, key: str, local_path: str, provider: Optional[CloudProvider] = None) -> bool:
        """Download file from cloud storage."""
        integration = self.get_integration(provider)
        return await integration.download_file(key, local_path)
    
    async def list_files(self, prefix: str = "", provider: Optional[CloudProvider] = None) -> List[CloudFile]:
        """List files in cloud storage."""
        integration = self.get_integration(provider)
        return await integration.list_files(prefix)
    
    async def delete_file(self, key: str, provider: Optional[CloudProvider] = None) -> bool:
        """Delete file from cloud storage."""
        integration = self.get_integration(provider)
        return await integration.delete_file(key)
    
    async def sync_to_cloud(self, local_dir: str, cloud_prefix: str = "", provider: Optional[CloudProvider] = None) -> List[CloudFile]:
        """Sync local directory to cloud storage."""
        uploaded_files = []
        local_path = Path(local_dir)
        
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                cloud_key = f"{cloud_prefix}/{relative_path}".replace("\\", "/")
                
                try:
                    cloud_file = await self.upload_file(str(file_path), cloud_key, provider)
                    uploaded_files.append(cloud_file)
                    logger.info(f"Uploaded {file_path} to {cloud_key}")
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
        
        return uploaded_files
    
    async def sync_from_cloud(self, cloud_prefix: str, local_dir: str, provider: Optional[CloudProvider] = None) -> List[str]:
        """Sync cloud storage to local directory."""
        downloaded_files = []
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        cloud_files = await self.list_files(cloud_prefix, provider)
        
        for cloud_file in cloud_files:
            local_file_path = local_path / cloud_file.key.replace(cloud_prefix + "/", "")
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                success = await self.download_file(cloud_file.key, str(local_file_path), provider)
                if success:
                    downloaded_files.append(str(local_file_path))
                    logger.info(f"Downloaded {cloud_file.key} to {local_file_path}")
            except Exception as e:
                logger.error(f"Failed to download {cloud_file.key}: {e}")
        
        return downloaded_files

# Global cloud integration manager
_global_cloud_manager: Optional[CloudIntegrationManager] = None

def get_cloud_manager() -> CloudIntegrationManager:
    """Get the global cloud integration manager."""
    global _global_cloud_manager
    if _global_cloud_manager is None:
        _global_cloud_manager = CloudIntegrationManager()
    return _global_cloud_manager

def setup_aws_integration(access_key: str, secret_key: str, region: str, bucket_name: str):
    """Setup AWS integration."""
    config = CloudConfig(
        provider=CloudProvider.AWS,
        service_type=ServiceType.STORAGE,
        region=region,
        credentials={
            'access_key': access_key,
            'secret_key': secret_key
        },
        bucket_name=bucket_name
    )
    
    manager = get_cloud_manager()
    manager.add_integration(CloudProvider.AWS, config)

def setup_google_cloud_integration(credentials: Dict[str, Any], bucket_name: str, region: str = "us-central1"):
    """Setup Google Cloud integration."""
    config = CloudConfig(
        provider=CloudProvider.GOOGLE,
        service_type=ServiceType.STORAGE,
        region=region,
        credentials=credentials,
        bucket_name=bucket_name
    )
    
    manager = get_cloud_manager()
    manager.add_integration(CloudProvider.GOOGLE, config)

def setup_azure_integration(connection_string: str, container_name: str, account_name: str):
    """Setup Azure integration."""
    config = CloudConfig(
        provider=CloudProvider.AZURE,
        service_type=ServiceType.STORAGE,
        region="global",
        credentials={
            'connection_string': connection_string,
            'account_name': account_name
        },
        bucket_name=container_name
    )
    
    manager = get_cloud_manager()
    manager.add_integration(CloudProvider.AZURE, config)


