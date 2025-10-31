"""
Cloud Integration System for AI History Analysis
==============================================

This module provides cloud integration capabilities including:
- AWS integration (S3, CloudWatch, Lambda)
- Azure integration (Blob Storage, Application Insights, Functions)
- Google Cloud integration (Cloud Storage, Monitoring, Cloud Functions)
- Multi-cloud data synchronization
- Cloud-based model training and deployment
- Distributed computing for large-scale analysis
- Cloud-native monitoring and alerting
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
from google.cloud import monitoring_v3
import pandas as pd
import numpy as np

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config
from .advanced_predictive_analytics import get_advanced_predictive_analytics

logger = logging.getLogger(__name__)


@dataclass
class CloudProvider:
    """Cloud provider configuration"""
    name: str  # aws, azure, gcp
    enabled: bool = True
    credentials: Dict[str, str] = None
    regions: List[str] = None
    services: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.credentials is None:
            self.credentials = {}
        if self.regions is None:
            self.regions = ["us-east-1"]
        if self.services is None:
            self.services = {
                "storage": True,
                "monitoring": True,
                "compute": True,
                "ml": True
            }


@dataclass
class CloudSyncResult:
    """Cloud synchronization result"""
    provider: str
    service: str
    operation: str
    success: bool
    records_synced: int
    errors: List[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CloudMetrics:
    """Cloud metrics data"""
    provider: str
    service: str
    metric_name: str
    value: float
    timestamp: datetime
    dimensions: Dict[str, str] = None
    unit: str = "Count"
    
    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = {}


class CloudIntegrationSystem:
    """Cloud integration system for AI history analysis"""
    
    def __init__(self):
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        self.advanced_analytics = get_advanced_predictive_analytics()
        
        # Cloud providers
        self.providers: Dict[str, CloudProvider] = {}
        self.cloud_clients: Dict[str, Any] = {}
        
        # Sync configuration
        self.sync_enabled = True
        self.sync_interval_minutes = 60
        self.batch_size = 1000
        
        # Background tasks
        self.sync_task: Optional[asyncio.Task] = None
        self.is_syncing = False
        
        # Initialize cloud providers
        self._initialize_cloud_providers()
    
    def _initialize_cloud_providers(self):
        """Initialize cloud provider configurations"""
        # AWS
        aws_provider = CloudProvider(
            name="aws",
            enabled=os.getenv("AWS_ENABLED", "false").lower() == "true",
            credentials={
                "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
                "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            },
            regions=["us-east-1", "us-west-2", "eu-west-1"],
            services={
                "storage": True,
                "monitoring": True,
                "compute": True,
                "ml": True
            }
        )
        self.providers["aws"] = aws_provider
        
        # Azure
        azure_provider = CloudProvider(
            name="azure",
            enabled=os.getenv("AZURE_ENABLED", "false").lower() == "true",
            credentials={
                "connection_string": os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
                "tenant_id": os.getenv("AZURE_TENANT_ID"),
                "client_id": os.getenv("AZURE_CLIENT_ID"),
                "client_secret": os.getenv("AZURE_CLIENT_SECRET")
            },
            regions=["eastus", "westus2", "westeurope"],
            services={
                "storage": True,
                "monitoring": True,
                "compute": True,
                "ml": True
            }
        )
        self.providers["azure"] = azure_provider
        
        # Google Cloud
        gcp_provider = CloudProvider(
            name="gcp",
            enabled=os.getenv("GCP_ENABLED", "false").lower() == "true",
            credentials={
                "project_id": os.getenv("GCP_PROJECT_ID"),
                "service_account_key": os.getenv("GCP_SERVICE_ACCOUNT_KEY"),
                "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            },
            regions=["us-central1", "us-east1", "europe-west1"],
            services={
                "storage": True,
                "monitoring": True,
                "compute": True,
                "ml": True
            }
        )
        self.providers["gcp"] = gcp_provider
    
    async def initialize_cloud_clients(self):
        """Initialize cloud service clients"""
        try:
            # Initialize AWS clients
            if self.providers["aws"].enabled:
                await self._initialize_aws_clients()
            
            # Initialize Azure clients
            if self.providers["azure"].enabled:
                await self._initialize_azure_clients()
            
            # Initialize GCP clients
            if self.providers["gcp"].enabled:
                await self._initialize_gcp_clients()
            
            logger.info("Cloud clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing cloud clients: {str(e)}")
            raise
    
    async def _initialize_aws_clients(self):
        """Initialize AWS service clients"""
        try:
            aws_provider = self.providers["aws"]
            credentials = aws_provider.credentials
            
            # S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key"),
                region_name=credentials.get("region", "us-east-1")
            )
            self.cloud_clients["aws_s3"] = s3_client
            
            # CloudWatch client
            cloudwatch_client = boto3.client(
                'cloudwatch',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key"),
                region_name=credentials.get("region", "us-east-1")
            )
            self.cloud_clients["aws_cloudwatch"] = cloudwatch_client
            
            # Lambda client
            lambda_client = boto3.client(
                'lambda',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key"),
                region_name=credentials.get("region", "us-east-1")
            )
            self.cloud_clients["aws_lambda"] = lambda_client
            
            logger.info("AWS clients initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            raise
    
    async def _initialize_azure_clients(self):
        """Initialize Azure service clients"""
        try:
            azure_provider = self.providers["azure"]
            credentials = azure_provider.credentials
            
            # Blob Storage client
            if credentials.get("connection_string"):
                blob_client = BlobServiceClient.from_connection_string(
                    credentials["connection_string"]
                )
                self.cloud_clients["azure_blob"] = blob_client
            
            logger.info("Azure clients initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Azure clients: {str(e)}")
            raise
    
    async def _initialize_gcp_clients(self):
        """Initialize Google Cloud service clients"""
        try:
            gcp_provider = self.providers["gcp"]
            credentials = gcp_provider.credentials
            
            # Cloud Storage client
            if credentials.get("project_id"):
                storage_client = gcs.Client(project=credentials["project_id"])
                self.cloud_clients["gcp_storage"] = storage_client
            
            # Monitoring client
            if credentials.get("project_id"):
                monitoring_client = monitoring_v3.MetricServiceClient()
                self.cloud_clients["gcp_monitoring"] = monitoring_client
            
            logger.info("GCP clients initialized")
            
        except Exception as e:
            logger.error(f"Error initializing GCP clients: {str(e)}")
            raise
    
    async def sync_to_cloud(self, provider: str = None) -> List[CloudSyncResult]:
        """Sync AI history data to cloud storage"""
        try:
            if not self.sync_enabled:
                logger.info("Cloud sync is disabled")
                return []
            
            results = []
            
            # Sync to all providers if none specified
            providers_to_sync = [provider] if provider else [p for p in self.providers.keys() if self.providers[p].enabled]
            
            for prov in providers_to_sync:
                if prov not in self.providers or not self.providers[prov].enabled:
                    continue
                
                try:
                    if prov == "aws":
                        result = await self._sync_to_aws()
                        results.append(result)
                    elif prov == "azure":
                        result = await self._sync_to_azure()
                        results.append(result)
                    elif prov == "gcp":
                        result = await self._sync_to_gcp()
                        results.append(result)
                
                except Exception as e:
                    logger.error(f"Error syncing to {prov}: {str(e)}")
                    results.append(CloudSyncResult(
                        provider=prov,
                        service="sync",
                        operation="sync_data",
                        success=False,
                        records_synced=0,
                        errors=[str(e)]
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cloud sync: {str(e)}")
            return []
    
    async def _sync_to_aws(self) -> CloudSyncResult:
        """Sync data to AWS S3"""
        try:
            start_time = datetime.now()
            
            # Get performance data
            performance_data = self._get_all_performance_data()
            
            if not performance_data:
                return CloudSyncResult(
                    provider="aws",
                    service="s3",
                    operation="sync_data",
                    success=True,
                    records_synced=0,
                    duration_seconds=0.0
                )
            
            # Convert to JSON
            data_json = json.dumps(performance_data, default=str)
            
            # Upload to S3
            s3_client = self.cloud_clients.get("aws_s3")
            if not s3_client:
                raise ValueError("AWS S3 client not initialized")
            
            bucket_name = os.getenv("AWS_S3_BUCKET", "ai-history-data")
            key = f"performance-data/{datetime.now().strftime('%Y/%m/%d')}/data.json"
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=data_json,
                ContentType='application/json'
            )
            
            # Upload metrics to CloudWatch
            await self._upload_metrics_to_cloudwatch(performance_data)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return CloudSyncResult(
                provider="aws",
                service="s3",
                operation="sync_data",
                success=True,
                records_synced=len(performance_data),
                duration_seconds=duration,
                metadata={"bucket": bucket_name, "key": key}
            )
            
        except Exception as e:
            logger.error(f"Error syncing to AWS: {str(e)}")
            return CloudSyncResult(
                provider="aws",
                service="s3",
                operation="sync_data",
                success=False,
                records_synced=0,
                errors=[str(e)]
            )
    
    async def _sync_to_azure(self) -> CloudSyncResult:
        """Sync data to Azure Blob Storage"""
        try:
            start_time = datetime.now()
            
            # Get performance data
            performance_data = self._get_all_performance_data()
            
            if not performance_data:
                return CloudSyncResult(
                    provider="azure",
                    service="blob",
                    operation="sync_data",
                    success=True,
                    records_synced=0,
                    duration_seconds=0.0
                )
            
            # Convert to JSON
            data_json = json.dumps(performance_data, default=str)
            
            # Upload to Azure Blob Storage
            blob_client = self.cloud_clients.get("azure_blob")
            if not blob_client:
                raise ValueError("Azure Blob client not initialized")
            
            container_name = os.getenv("AZURE_CONTAINER", "ai-history-data")
            blob_name = f"performance-data/{datetime.now().strftime('%Y/%m/%d')}/data.json"
            
            blob_client.get_blob_client(
                container=container_name,
                blob=blob_name
            ).upload_blob(data_json, overwrite=True)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return CloudSyncResult(
                provider="azure",
                service="blob",
                operation="sync_data",
                success=True,
                records_synced=len(performance_data),
                duration_seconds=duration,
                metadata={"container": container_name, "blob": blob_name}
            )
            
        except Exception as e:
            logger.error(f"Error syncing to Azure: {str(e)}")
            return CloudSyncResult(
                provider="azure",
                service="blob",
                operation="sync_data",
                success=False,
                records_synced=0,
                errors=[str(e)]
            )
    
    async def _sync_to_gcp(self) -> CloudSyncResult:
        """Sync data to Google Cloud Storage"""
        try:
            start_time = datetime.now()
            
            # Get performance data
            performance_data = self._get_all_performance_data()
            
            if not performance_data:
                return CloudSyncResult(
                    provider="gcp",
                    service="storage",
                    operation="sync_data",
                    success=True,
                    records_synced=0,
                    duration_seconds=0.0
                )
            
            # Convert to JSON
            data_json = json.dumps(performance_data, default=str)
            
            # Upload to GCS
            storage_client = self.cloud_clients.get("gcp_storage")
            if not storage_client:
                raise ValueError("GCP Storage client not initialized")
            
            bucket_name = os.getenv("GCP_BUCKET", "ai-history-data")
            blob_name = f"performance-data/{datetime.now().strftime('%Y/%m/%d')}/data.json"
            
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(data_json, content_type='application/json')
            
            # Upload metrics to Cloud Monitoring
            await self._upload_metrics_to_gcp_monitoring(performance_data)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return CloudSyncResult(
                provider="gcp",
                service="storage",
                operation="sync_data",
                success=True,
                records_synced=len(performance_data),
                duration_seconds=duration,
                metadata={"bucket": bucket_name, "blob": blob_name}
            )
            
        except Exception as e:
            logger.error(f"Error syncing to GCP: {str(e)}")
            return CloudSyncResult(
                provider="gcp",
                service="storage",
                operation="sync_data",
                success=False,
                records_synced=0,
                errors=[str(e)]
            )
    
    def _get_all_performance_data(self) -> List[Dict[str, Any]]:
        """Get all performance data for cloud sync"""
        try:
            # Get all models and metrics
            stats = self.analyzer.performance_stats
            all_data = []
            
            for model_name in stats["models_tracked"]:
                for metric in PerformanceMetric:
                    try:
                        data = self.analyzer.get_model_performance(model_name, metric, days=30)
                        for perf in data:
                            all_data.append({
                                "model_name": model_name,
                                "metric": metric.value,
                                "value": perf.value,
                                "timestamp": perf.timestamp.isoformat(),
                                "context": perf.context,
                                "metadata": perf.metadata
                            })
                    except Exception as e:
                        logger.warning(f"Error getting data for {model_name} - {metric.value}: {str(e)}")
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error getting performance data: {str(e)}")
            return []
    
    async def _upload_metrics_to_cloudwatch(self, performance_data: List[Dict[str, Any]]):
        """Upload metrics to AWS CloudWatch"""
        try:
            cloudwatch_client = self.cloud_clients.get("aws_cloudwatch")
            if not cloudwatch_client:
                return
            
            # Group data by model and metric
            metrics_by_model = {}
            for data in performance_data:
                model_name = data["model_name"]
                metric_name = data["metric"]
                value = data["value"]
                timestamp = datetime.fromisoformat(data["timestamp"])
                
                key = f"{model_name}_{metric_name}"
                if key not in metrics_by_model:
                    metrics_by_model[key] = []
                
                metrics_by_model[key].append({
                    "Value": value,
                    "Timestamp": timestamp
                })
            
            # Upload metrics in batches
            for key, metrics in metrics_by_model.items():
                if not metrics:
                    continue
                
                try:
                    cloudwatch_client.put_metric_data(
                        Namespace='AI/ModelPerformance',
                        MetricData=[{
                            'MetricName': key,
                            'Values': [m["Value"] for m in metrics],
                            'Timestamps': [m["Timestamp"] for m in metrics],
                            'Unit': 'None'
                        }]
                    )
                except Exception as e:
                    logger.warning(f"Error uploading CloudWatch metric {key}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error uploading metrics to CloudWatch: {str(e)}")
    
    async def _upload_metrics_to_gcp_monitoring(self, performance_data: List[Dict[str, Any]]):
        """Upload metrics to Google Cloud Monitoring"""
        try:
            monitoring_client = self.cloud_clients.get("gcp_monitoring")
            if not monitoring_client:
                return
            
            project_id = self.providers["gcp"].credentials.get("project_id")
            if not project_id:
                return
            
            # Group data by model and metric
            metrics_by_model = {}
            for data in performance_data:
                model_name = data["model_name"]
                metric_name = data["metric"]
                value = data["value"]
                timestamp = datetime.fromisoformat(data["timestamp"])
                
                key = f"{model_name}_{metric_name}"
                if key not in metrics_by_model:
                    metrics_by_model[key] = []
                
                metrics_by_model[key].append({
                    "value": value,
                    "timestamp": timestamp
                })
            
            # Upload metrics
            for key, metrics in metrics_by_model.items():
                if not metrics:
                    continue
                
                try:
                    # Create time series data
                    series = monitoring_v3.TimeSeries()
                    series.metric.type = f"custom.googleapis.com/ai/model_performance/{key}"
                    series.resource.type = "global"
                    
                    for metric in metrics:
                        point = monitoring_v3.Point()
                        point.value.double_value = metric["value"]
                        point.interval.end_time.seconds = int(metric["timestamp"].timestamp())
                        series.points.append(point)
                    
                    # Write time series
                    project_name = f"projects/{project_id}"
                    monitoring_client.create_time_series(
                        name=project_name,
                        time_series=[series]
                    )
                
                except Exception as e:
                    logger.warning(f"Error uploading GCP metric {key}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error uploading metrics to GCP Monitoring: {str(e)}")
    
    async def start_cloud_sync(self):
        """Start automatic cloud synchronization"""
        if self.is_syncing:
            logger.warning("Cloud sync is already running")
            return
        
        self.is_syncing = True
        self.sync_task = asyncio.create_task(self._cloud_sync_loop())
        logger.info("Started automatic cloud synchronization")
    
    async def stop_cloud_sync(self):
        """Stop automatic cloud synchronization"""
        self.is_syncing = False
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped automatic cloud synchronization")
    
    async def _cloud_sync_loop(self):
        """Background cloud sync loop"""
        while self.is_syncing:
            try:
                # Sync to all enabled providers
                results = await self.sync_to_cloud()
                
                # Log results
                for result in results:
                    if result.success:
                        logger.info(f"Cloud sync to {result.provider}: {result.records_synced} records synced")
                    else:
                        logger.error(f"Cloud sync to {result.provider} failed: {result.errors}")
                
                # Wait for next sync
                await asyncio.sleep(self.sync_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in cloud sync loop: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def deploy_ml_model_to_cloud(self, 
                                     model_name: str,
                                     metric: PerformanceMetric,
                                     provider: str = "aws") -> Dict[str, Any]:
        """Deploy ML model to cloud for distributed inference"""
        try:
            if provider == "aws":
                return await self._deploy_model_to_aws_lambda(model_name, metric)
            elif provider == "azure":
                return await self._deploy_model_to_azure_function(model_name, metric)
            elif provider == "gcp":
                return await self._deploy_model_to_gcp_function(model_name, metric)
            else:
                raise ValueError(f"Unsupported cloud provider: {provider}")
        
        except Exception as e:
            logger.error(f"Error deploying model to {provider}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _deploy_model_to_aws_lambda(self, model_name: str, metric: PerformanceMetric) -> Dict[str, Any]:
        """Deploy model to AWS Lambda"""
        try:
            # This would involve creating a Lambda function with the trained model
            # For now, return a placeholder response
            return {
                "success": True,
                "provider": "aws",
                "service": "lambda",
                "function_name": f"ai-model-{model_name}-{metric.value}",
                "endpoint": f"https://lambda.us-east-1.amazonaws.com/2015-03-31/functions/ai-model-{model_name}-{metric.value}/invocations"
            }
        
        except Exception as e:
            logger.error(f"Error deploying to AWS Lambda: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _deploy_model_to_azure_function(self, model_name: str, metric: PerformanceMetric) -> Dict[str, Any]:
        """Deploy model to Azure Functions"""
        try:
            # This would involve creating an Azure Function with the trained model
            # For now, return a placeholder response
            return {
                "success": True,
                "provider": "azure",
                "service": "functions",
                "function_name": f"ai-model-{model_name}-{metric.value}",
                "endpoint": f"https://ai-model-{model_name}-{metric.value}.azurewebsites.net/api/predict"
            }
        
        except Exception as e:
            logger.error(f"Error deploying to Azure Functions: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _deploy_model_to_gcp_function(self, model_name: str, metric: PerformanceMetric) -> Dict[str, Any]:
        """Deploy model to Google Cloud Functions"""
        try:
            # This would involve creating a Cloud Function with the trained model
            # For now, return a placeholder response
            return {
                "success": True,
                "provider": "gcp",
                "service": "functions",
                "function_name": f"ai-model-{model_name}-{metric.value}",
                "endpoint": f"https://us-central1-{self.providers['gcp'].credentials.get('project_id')}.cloudfunctions.net/ai-model-{model_name}-{metric.value}"
            }
        
        except Exception as e:
            logger.error(f"Error deploying to GCP Functions: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_cloud_status(self) -> Dict[str, Any]:
        """Get cloud integration status"""
        return {
            "sync_enabled": self.sync_enabled,
            "sync_interval_minutes": self.sync_interval_minutes,
            "is_syncing": self.is_syncing,
            "providers": {
                name: {
                    "enabled": provider.enabled,
                    "services": provider.services
                }
                for name, provider in self.providers.items()
            },
            "clients_initialized": list(self.cloud_clients.keys())
        }


# Global cloud integration instance
_cloud_integration: Optional[CloudIntegrationSystem] = None


def get_cloud_integration() -> CloudIntegrationSystem:
    """Get or create global cloud integration system"""
    global _cloud_integration
    if _cloud_integration is None:
        _cloud_integration = CloudIntegrationSystem()
    return _cloud_integration


# Example usage
async def main():
    """Example usage of cloud integration"""
    cloud_integration = get_cloud_integration()
    
    # Initialize cloud clients
    await cloud_integration.initialize_cloud_clients()
    
    # Start automatic sync
    await cloud_integration.start_cloud_sync()
    
    # Manual sync
    results = await cloud_integration.sync_to_cloud()
    for result in results:
        print(f"Sync to {result.provider}: {result.records_synced} records")
    
    # Deploy model to cloud
    deployment = await cloud_integration.deploy_ml_model_to_cloud(
        model_name="gpt-4",
        metric=PerformanceMetric.QUALITY_SCORE,
        provider="aws"
    )
    print(f"Model deployment: {deployment}")
    
    # Get status
    status = cloud_integration.get_cloud_status()
    print(f"Cloud status: {status}")


if __name__ == "__main__":
    asyncio.run(main())

























