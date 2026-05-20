"""
BUL Cloud Integration Manager
============================

Advanced cloud integration management for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import boto3
import google.cloud.storage as gcs
import azure.storage.blob as azure_blob
from dataclasses import dataclass
from enum import Enum
import yaml

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"

@dataclass
class CloudConfig:
    """Cloud configuration."""
    provider: CloudProvider
    region: str
    credentials: Dict[str, Any]
    bucket_name: str
    endpoint_url: Optional[str] = None

class CloudIntegrationManager:
    """Advanced cloud integration management."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.cloud_configs = {}
        self.active_connections = {}
        self.sync_status = {}
        self.init_cloud_providers()
    
    def init_cloud_providers(self):
        """Initialize cloud providers."""
        print("☁️ Initializing cloud providers...")
        
        # AWS S3
        if hasattr(self.config, 'aws_access_key_id') and self.config.aws_access_key_id:
            self.cloud_configs[CloudProvider.AWS] = CloudConfig(
                provider=CloudProvider.AWS,
                region=getattr(self.config, 'aws_region', 'us-east-1'),
                credentials={
                    'aws_access_key_id': self.config.aws_access_key_id,
                    'aws_secret_access_key': getattr(self.config, 'aws_secret_access_key', ''),
                    'region_name': getattr(self.config, 'aws_region', 'us-east-1')
                },
                bucket_name=getattr(self.config, 'aws_bucket_name', 'bul-documents')
            )
            print("✅ AWS S3 provider initialized")
        
        # Google Cloud Storage
        if hasattr(self.config, 'gcp_project_id') and self.config.gcp_project_id:
            self.cloud_configs[CloudProvider.GCP] = CloudConfig(
                provider=CloudProvider.GCP,
                region=getattr(self.config, 'gcp_region', 'us-central1'),
                credentials={
                    'project_id': self.config.gcp_project_id,
                    'credentials_path': getattr(self.config, 'gcp_credentials_path', ''),
                    'region': getattr(self.config, 'gcp_region', 'us-central1')
                },
                bucket_name=getattr(self.config, 'gcp_bucket_name', 'bul-documents')
            )
            print("✅ Google Cloud Storage provider initialized")
        
        # Azure Blob Storage
        if hasattr(self.config, 'azure_account_name') and self.config.azure_account_name:
            self.cloud_configs[CloudProvider.AZURE] = CloudConfig(
                provider=CloudProvider.AZURE,
                region=getattr(self.config, 'azure_region', 'eastus'),
                credentials={
                    'account_name': self.config.azure_account_name,
                    'account_key': getattr(self.config, 'azure_account_key', ''),
                    'region': getattr(self.config, 'azure_region', 'eastus')
                },
                bucket_name=getattr(self.config, 'azure_container_name', 'bul-documents')
            )
            print("✅ Azure Blob Storage provider initialized")
    
    def get_available_providers(self) -> List[CloudProvider]:
        """Get list of available cloud providers."""
        return list(self.cloud_configs.keys())
    
    async def upload_file(self, provider: CloudProvider, local_path: str, 
                         cloud_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload file to cloud storage."""
        if provider not in self.cloud_configs:
            raise ValueError(f"Provider {provider.value} not configured")
        
        cloud_config = self.cloud_configs[provider]
        local_file = Path(local_path)
        
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        print(f"☁️ Uploading {local_path} to {provider.value}...")
        
        try:
            if provider == CloudProvider.AWS:
                result = await self._upload_to_s3(cloud_config, local_path, cloud_path, metadata)
            elif provider == CloudProvider.GCP:
                result = await self._upload_to_gcs(cloud_config, local_path, cloud_path, metadata)
            elif provider == CloudProvider.AZURE:
                result = await self._upload_to_azure(cloud_config, local_path, cloud_path, metadata)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Update sync status
            self.sync_status[cloud_path] = {
                'provider': provider.value,
                'local_path': local_path,
                'cloud_path': cloud_path,
                'uploaded_at': datetime.now(),
                'size': local_file.stat().st_size,
                'metadata': metadata or {}
            }
            
            print(f"✅ File uploaded successfully: {cloud_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error uploading file to {provider.value}: {e}")
            raise
    
    async def download_file(self, provider: CloudProvider, cloud_path: str, 
                           local_path: str) -> Dict[str, Any]:
        """Download file from cloud storage."""
        if provider not in self.cloud_configs:
            raise ValueError(f"Provider {provider.value} not configured")
        
        cloud_config = self.cloud_configs[provider]
        
        print(f"☁️ Downloading {cloud_path} from {provider.value}...")
        
        try:
            if provider == CloudProvider.AWS:
                result = await self._download_from_s3(cloud_config, cloud_path, local_path)
            elif provider == CloudProvider.GCP:
                result = await self._download_from_gcs(cloud_config, cloud_path, local_path)
            elif provider == CloudProvider.AZURE:
                result = await self._download_from_azure(cloud_config, cloud_path, local_path)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            print(f"✅ File downloaded successfully: {local_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error downloading file from {provider.value}: {e}")
            raise
    
    async def list_files(self, provider: CloudProvider, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in cloud storage."""
        if provider not in self.cloud_configs:
            raise ValueError(f"Provider {provider.value} not configured")
        
        cloud_config = self.cloud_configs[provider]
        
        try:
            if provider == CloudProvider.AWS:
                files = await self._list_s3_files(cloud_config, prefix)
            elif provider == CloudProvider.GCP:
                files = await self._list_gcs_files(cloud_config, prefix)
            elif provider == CloudProvider.AZURE:
                files = await self._list_azure_files(cloud_config, prefix)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files from {provider.value}: {e}")
            raise
    
    async def delete_file(self, provider: CloudProvider, cloud_path: str) -> Dict[str, Any]:
        """Delete file from cloud storage."""
        if provider not in self.cloud_configs:
            raise ValueError(f"Provider {provider.value} not configured")
        
        cloud_config = self.cloud_configs[provider]
        
        print(f"☁️ Deleting {cloud_path} from {provider.value}...")
        
        try:
            if provider == CloudProvider.AWS:
                result = await self._delete_from_s3(cloud_config, cloud_path)
            elif provider == CloudProvider.GCP:
                result = await self._delete_from_gcs(cloud_config, cloud_path)
            elif provider == CloudProvider.AZURE:
                result = await self._delete_from_azure(cloud_config, cloud_path)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Update sync status
            if cloud_path in self.sync_status:
                del self.sync_status[cloud_path]
            
            print(f"✅ File deleted successfully: {cloud_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error deleting file from {provider.value}: {e}")
            raise
    
    async def sync_directory(self, provider: CloudProvider, local_dir: str, 
                           cloud_prefix: str, direction: str = "upload") -> Dict[str, Any]:
        """Sync directory with cloud storage."""
        if provider not in self.cloud_configs:
            raise ValueError(f"Provider {provider.value} not configured")
        
        local_path = Path(local_dir)
        if not local_path.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")
        
        print(f"☁️ Syncing directory {local_dir} with {provider.value}...")
        
        synced_files = []
        errors = []
        
        try:
            if direction == "upload":
                # Upload all files in directory
                for file_path in local_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        cloud_path = f"{cloud_prefix}/{relative_path}".replace("\\", "/")
                        
                        try:
                            result = await self.upload_file(provider, str(file_path), cloud_path)
                            synced_files.append({
                                'local_path': str(file_path),
                                'cloud_path': cloud_path,
                                'status': 'uploaded',
                                'size': file_path.stat().st_size
                            })
                        except Exception as e:
                            errors.append({
                                'file': str(file_path),
                                'error': str(e)
                            })
            
            elif direction == "download":
                # Download all files from cloud
                cloud_files = await self.list_files(provider, cloud_prefix)
                
                for file_info in cloud_files:
                    cloud_path = file_info['name']
                    relative_path = cloud_path.replace(cloud_prefix + "/", "")
                    local_file_path = local_path / relative_path
                    
                    # Create directory if needed
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        result = await self.download_file(provider, cloud_path, str(local_file_path))
                        synced_files.append({
                            'local_path': str(local_file_path),
                            'cloud_path': cloud_path,
                            'status': 'downloaded',
                            'size': file_info.get('size', 0)
                        })
                    except Exception as e:
                        errors.append({
                            'file': cloud_path,
                            'error': str(e)
                        })
            
            else:
                raise ValueError(f"Invalid sync direction: {direction}")
            
            return {
                'direction': direction,
                'provider': provider.value,
                'local_dir': local_dir,
                'cloud_prefix': cloud_prefix,
                'synced_files': len(synced_files),
                'errors': len(errors),
                'files': synced_files,
                'error_details': errors
            }
            
        except Exception as e:
            logger.error(f"Error syncing directory: {e}")
            raise
    
    async def _upload_to_s3(self, config: CloudConfig, local_path: str, 
                           cloud_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload file to AWS S3."""
        s3_client = boto3.client(
            's3',
            aws_access_key_id=config.credentials['aws_access_key_id'],
            aws_secret_access_key=config.credentials['aws_secret_access_key'],
            region_name=config.credentials['region_name']
        )
        
        extra_args = {}
        if metadata:
            extra_args['Metadata'] = metadata
        
        s3_client.upload_file(local_path, config.bucket_name, cloud_path, ExtraArgs=extra_args)
        
        return {
            'provider': 'aws',
            'bucket': config.bucket_name,
            'key': cloud_path,
            'local_path': local_path,
            'uploaded_at': datetime.now().isoformat()
        }
    
    async def _download_from_s3(self, config: CloudConfig, cloud_path: str, 
                               local_path: str) -> Dict[str, Any]:
        """Download file from AWS S3."""
        s3_client = boto3.client(
            's3',
            aws_access_key_id=config.credentials['aws_access_key_id'],
            aws_secret_access_key=config.credentials['aws_secret_access_key'],
            region_name=config.credentials['region_name']
        )
        
        s3_client.download_file(config.bucket_name, cloud_path, local_path)
        
        return {
            'provider': 'aws',
            'bucket': config.bucket_name,
            'key': cloud_path,
            'local_path': local_path,
            'downloaded_at': datetime.now().isoformat()
        }
    
    async def _list_s3_files(self, config: CloudConfig, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in AWS S3."""
        s3_client = boto3.client(
            's3',
            aws_access_key_id=config.credentials['aws_access_key_id'],
            aws_secret_access_key=config.credentials['aws_secret_access_key'],
            region_name=config.credentials['region_name']
        )
        
        response = s3_client.list_objects_v2(Bucket=config.bucket_name, Prefix=prefix)
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append({
                    'name': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag']
                })
        
        return files
    
    async def _delete_from_s3(self, config: CloudConfig, cloud_path: str) -> Dict[str, Any]:
        """Delete file from AWS S3."""
        s3_client = boto3.client(
            's3',
            aws_access_key_id=config.credentials['aws_access_key_id'],
            aws_secret_access_key=config.credentials['aws_secret_access_key'],
            region_name=config.credentials['region_name']
        )
        
        s3_client.delete_object(Bucket=config.bucket_name, Key=cloud_path)
        
        return {
            'provider': 'aws',
            'bucket': config.bucket_name,
            'key': cloud_path,
            'deleted_at': datetime.now().isoformat()
        }
    
    async def _upload_to_gcs(self, config: CloudConfig, local_path: str, 
                            cloud_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload file to Google Cloud Storage."""
        client = gcs.Client(project=config.credentials['project_id'])
        bucket = client.bucket(config.bucket_name)
        blob = bucket.blob(cloud_path)
        
        if metadata:
            blob.metadata = metadata
        
        blob.upload_from_filename(local_path)
        
        return {
            'provider': 'gcp',
            'bucket': config.bucket_name,
            'name': cloud_path,
            'local_path': local_path,
            'uploaded_at': datetime.now().isoformat()
        }
    
    async def _download_from_gcs(self, config: CloudConfig, cloud_path: str, 
                                local_path: str) -> Dict[str, Any]:
        """Download file from Google Cloud Storage."""
        client = gcs.Client(project=config.credentials['project_id'])
        bucket = client.bucket(config.bucket_name)
        blob = bucket.blob(cloud_path)
        
        blob.download_to_filename(local_path)
        
        return {
            'provider': 'gcp',
            'bucket': config.bucket_name,
            'name': cloud_path,
            'local_path': local_path,
            'downloaded_at': datetime.now().isoformat()
        }
    
    async def _list_gcs_files(self, config: CloudConfig, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in Google Cloud Storage."""
        client = gcs.Client(project=config.credentials['project_id'])
        bucket = client.bucket(config.bucket_name)
        
        blobs = bucket.list_blobs(prefix=prefix)
        
        files = []
        for blob in blobs:
            files.append({
                'name': blob.name,
                'size': blob.size,
                'last_modified': blob.updated.isoformat(),
                'etag': blob.etag
            })
        
        return files
    
    async def _delete_from_gcs(self, config: CloudConfig, cloud_path: str) -> Dict[str, Any]:
        """Delete file from Google Cloud Storage."""
        client = gcs.Client(project=config.credentials['project_id'])
        bucket = client.bucket(config.bucket_name)
        blob = bucket.blob(cloud_path)
        
        blob.delete()
        
        return {
            'provider': 'gcp',
            'bucket': config.bucket_name,
            'name': cloud_path,
            'deleted_at': datetime.now().isoformat()
        }
    
    async def _upload_to_azure(self, config: CloudConfig, local_path: str, 
                              cloud_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload file to Azure Blob Storage."""
        blob_service_client = azure_blob.BlobServiceClient(
            account_url=f"https://{config.credentials['account_name']}.blob.core.windows.net",
            credential=config.credentials['account_key']
        )
        
        container_client = blob_service_client.get_container_client(config.bucket_name)
        blob_client = container_client.get_blob_client(cloud_path)
        
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True, metadata=metadata)
        
        return {
            'provider': 'azure',
            'container': config.bucket_name,
            'name': cloud_path,
            'local_path': local_path,
            'uploaded_at': datetime.now().isoformat()
        }
    
    async def _download_from_azure(self, config: CloudConfig, cloud_path: str, 
                                  local_path: str) -> Dict[str, Any]:
        """Download file from Azure Blob Storage."""
        blob_service_client = azure_blob.BlobServiceClient(
            account_url=f"https://{config.credentials['account_name']}.blob.core.windows.net",
            credential=config.credentials['account_key']
        )
        
        container_client = blob_service_client.get_container_client(config.bucket_name)
        blob_client = container_client.get_blob_client(cloud_path)
        
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        return {
            'provider': 'azure',
            'container': config.bucket_name,
            'name': cloud_path,
            'local_path': local_path,
            'downloaded_at': datetime.now().isoformat()
        }
    
    async def _list_azure_files(self, config: CloudConfig, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in Azure Blob Storage."""
        blob_service_client = azure_blob.BlobServiceClient(
            account_url=f"https://{config.credentials['account_name']}.blob.core.windows.net",
            credential=config.credentials['account_key']
        )
        
        container_client = blob_service_client.get_container_client(config.bucket_name)
        
        blobs = container_client.list_blobs(name_starts_with=prefix)
        
        files = []
        for blob in blobs:
            files.append({
                'name': blob.name,
                'size': blob.size,
                'last_modified': blob.last_modified.isoformat(),
                'etag': blob.etag
            })
        
        return files
    
    async def _delete_from_azure(self, config: CloudConfig, cloud_path: str) -> Dict[str, Any]:
        """Delete file from Azure Blob Storage."""
        blob_service_client = azure_blob.BlobServiceClient(
            account_url=f"https://{config.credentials['account_name']}.blob.core.windows.net",
            credential=config.credentials['account_key']
        )
        
        container_client = blob_service_client.get_container_client(config.bucket_name)
        blob_client = container_client.get_blob_client(cloud_path)
        
        blob_client.delete_blob()
        
        return {
            'provider': 'azure',
            'container': config.bucket_name,
            'name': cloud_path,
            'deleted_at': datetime.now().isoformat()
        }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        return {
            'total_files': len(self.sync_status),
            'providers': list(set(file_info['provider'] for file_info in self.sync_status.values())),
            'files': self.sync_status
        }
    
    def generate_cloud_report(self) -> str:
        """Generate cloud integration report."""
        available_providers = self.get_available_providers()
        sync_status = self.get_sync_status()
        
        report = f"""
BUL Cloud Integration Report
===========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

AVAILABLE PROVIDERS
-------------------
Total Providers: {len(available_providers)}
"""
        
        for provider in available_providers:
            config = self.cloud_configs[provider]
            report += f"""
{provider.value.upper()}:
  Region: {config.region}
  Bucket: {config.bucket_name}
  Status: Configured
"""
        
        report += f"""
SYNC STATUS
-----------
Total Files: {sync_status['total_files']}
Providers Used: {', '.join(sync_status['providers'])}

RECENT ACTIVITY
---------------
"""
        
        # Show recent sync activity
        recent_files = sorted(
            self.sync_status.items(),
            key=lambda x: x[1]['uploaded_at'],
            reverse=True
        )[:10]
        
        for cloud_path, file_info in recent_files:
            report += f"""
{cloud_path}:
  Provider: {file_info['provider']}
  Local Path: {file_info['local_path']}
  Size: {file_info['size']} bytes
  Uploaded: {file_info['uploaded_at']}
"""
        
        return report

def main():
    """Main cloud integration manager function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Cloud Integration Manager")
    parser.add_argument("--list-providers", action="store_true", help="List available cloud providers")
    parser.add_argument("--upload", help="Upload file to cloud")
    parser.add_argument("--download", help="Download file from cloud")
    parser.add_argument("--list-files", help="List files in cloud storage")
    parser.add_argument("--delete", help="Delete file from cloud storage")
    parser.add_argument("--sync", help="Sync directory with cloud storage")
    parser.add_argument("--provider", choices=['aws', 'gcp', 'azure'], required=True, help="Cloud provider")
    parser.add_argument("--local-path", help="Local file/directory path")
    parser.add_argument("--cloud-path", help="Cloud file path")
    parser.add_argument("--direction", choices=['upload', 'download'], default='upload', help="Sync direction")
    parser.add_argument("--report", action="store_true", help="Generate cloud integration report")
    
    args = parser.parse_args()
    
    manager = CloudIntegrationManager()
    
    print("☁️ BUL Cloud Integration Manager")
    print("=" * 40)
    
    if args.list_providers:
        providers = manager.get_available_providers()
        if providers:
            print(f"\n☁️ Available Cloud Providers ({len(providers)}):")
            print("-" * 50)
            for provider in providers:
                config = manager.cloud_configs[provider]
                print(f"{provider.value.upper()}:")
                print(f"  Region: {config.region}")
                print(f"  Bucket: {config.bucket_name}")
                print()
        else:
            print("No cloud providers configured.")
    
    elif args.upload:
        async def upload_file():
            try:
                result = await manager.upload_file(
                    CloudProvider(args.provider),
                    args.local_path,
                    args.cloud_path
                )
                print(f"✅ File uploaded successfully")
                print(f"   Provider: {result['provider']}")
                print(f"   Cloud Path: {result.get('key', result.get('name', 'N/A'))}")
            except Exception as e:
                print(f"❌ Upload failed: {e}")
        
        asyncio.run(upload_file())
    
    elif args.download:
        async def download_file():
            try:
                result = await manager.download_file(
                    CloudProvider(args.provider),
                    args.cloud_path,
                    args.local_path
                )
                print(f"✅ File downloaded successfully")
                print(f"   Provider: {result['provider']}")
                print(f"   Local Path: {result['local_path']}")
            except Exception as e:
                print(f"❌ Download failed: {e}")
        
        asyncio.run(download_file())
    
    elif args.list_files:
        async def list_files():
            try:
                files = await manager.list_files(
                    CloudProvider(args.provider),
                    args.list_files
                )
                print(f"📁 Files in {args.provider} storage:")
                print("-" * 50)
                for file_info in files:
                    print(f"{file_info['name']}")
                    print(f"  Size: {file_info['size']} bytes")
                    print(f"  Modified: {file_info['last_modified']}")
                    print()
            except Exception as e:
                print(f"❌ List files failed: {e}")
        
        asyncio.run(list_files())
    
    elif args.delete:
        async def delete_file():
            try:
                result = await manager.delete_file(
                    CloudProvider(args.provider),
                    args.delete
                )
                print(f"✅ File deleted successfully")
                print(f"   Provider: {result['provider']}")
                print(f"   Cloud Path: {result.get('key', result.get('name', 'N/A'))}")
            except Exception as e:
                print(f"❌ Delete failed: {e}")
        
        asyncio.run(delete_file())
    
    elif args.sync:
        async def sync_directory():
            try:
                result = await manager.sync_directory(
                    CloudProvider(args.provider),
                    args.local_path,
                    args.sync,
                    args.direction
                )
                print(f"✅ Directory sync completed")
                print(f"   Direction: {result['direction']}")
                print(f"   Provider: {result['provider']}")
                print(f"   Files Synced: {result['synced_files']}")
                print(f"   Errors: {result['errors']}")
            except Exception as e:
                print(f"❌ Sync failed: {e}")
        
        asyncio.run(sync_directory())
    
    elif args.report:
        report = manager.generate_cloud_report()
        print(report)
        
        # Save report
        report_file = f"cloud_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        providers = manager.get_available_providers()
        sync_status = manager.get_sync_status()
        
        print(f"☁️ Available Providers: {len(providers)}")
        print(f"📁 Synced Files: {sync_status['total_files']}")
        print(f"\n💡 Use --list-providers to see all providers")
        print(f"💡 Use --upload to upload files")
        print(f"💡 Use --sync to sync directories")
        print(f"💡 Use --report to generate integration report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
