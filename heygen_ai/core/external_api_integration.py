"""
External API Integration for HeyGen AI
=====================================

Provides comprehensive integration with external APIs including TTS,
voice cloning, video processing, and social media platforms
with enterprise-grade performance and reliability.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

# Core imports
from .base_service import BaseService, ServiceType, HealthCheckResult, ServiceStatus
from .error_handler import ErrorHandler, with_error_handling, with_retry
from .config_manager import ConfigurationManager
from .logging_service import LoggingService

# HTTP client imports
try:
    import aiohttp
    import httpx
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

# Cloud storage imports
try:
    import boto3
    from google.cloud import storage
    from azure.storage.blob import BlobServiceClient
    CLOUD_STORAGE_AVAILABLE = True
except ImportError:
    CLOUD_STORAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """API endpoint configuration."""
    
    name: str
    base_url: str
    api_key: str
    rate_limit: int = 100  # requests per minute
    timeout: int = 30
    retry_attempts: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    authentication_type: str = "api_key"  # api_key, oauth, bearer


@dataclass
class APIRequest:
    """Request for external API call."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    endpoint: str = ""
    method: str = "GET"
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None
    files: Optional[Dict[str, str]] = None
    timeout: int = 30
    retry_attempts: int = 3
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class APIResponse:
    """Response from external API call."""
    
    request_id: str
    status_code: int
    headers: Dict[str, str]
    data: Any
    response_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ServiceConfig:
    """Service configuration for external API integration."""
    
    name: str
    base_url: str
    api_key: str
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit: int = 100
    enabled: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    authentication_type: str = "api_key"


@dataclass
class CloudStorageConfig:
    """Cloud storage configuration."""
    
    provider: str  # aws, gcp, azure
    bucket_name: str
    region: str = "us-east-1"
    credentials: Dict[str, str] = field(default_factory=dict)
    endpoint_url: Optional[str] = None


class ExternalAPIManager(BaseService):
    """Manager for external API integrations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the external API manager."""
        super().__init__("ExternalAPIManager", ServiceType.PHASE3, config)
        
        # API endpoints
        self.api_endpoints: Dict[str, APIEndpoint] = {}
        
        # HTTP clients
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.httpx_client: Optional[httpx.AsyncClient] = None
        
        # Cloud storage clients
        self.aws_s3_client = None
        self.gcp_storage_client = None
        self.azure_blob_client = None
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Configuration manager
        self.config_manager = ConfigurationManager()
        
        # Logging service
        self.logging_service = LoggingService()
        
        # Performance tracking
        self.api_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0
        }
        
        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}

    async def _initialize_service_impl(self) -> None:
        """Initialize external API services."""
        try:
            logger.info("Initializing external API manager...")
            
            # Check dependencies
            await self._check_dependencies()
            
            # Initialize HTTP clients
            await self._initialize_http_clients()
            
            # Load API endpoints
            await self._load_api_endpoints()
            
            # Initialize cloud storage
            await self._initialize_cloud_storage()
            
            # Validate configuration
            await self._validate_configuration()
            
            logger.info("External API manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize external API manager: {e}")
            raise

    async def _check_dependencies(self) -> None:
        """Check required dependencies."""
        missing_deps = []
        
        if not HTTP_AVAILABLE:
            missing_deps.append("aiohttp/httpx")
        
        if not CLOUD_STORAGE_AVAILABLE:
            missing_deps.append("cloud storage libraries")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {missing_deps}")
            logger.warning("Some API features may not be available")

    async def _initialize_http_clients(self) -> None:
        """Initialize HTTP clients."""
        try:
            if HTTP_AVAILABLE:
                # Initialize aiohttp session
                self.http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers={"User-Agent": "HeyGenAI/1.0"}
                )
                
                # Initialize httpx client
                self.httpx_client = httpx.AsyncClient(
                    timeout=30.0,
                    headers={"User-Agent": "HeyGenAI/1.0"}
                )
                
                logger.info("HTTP clients initialized successfully")
            else:
                logger.warning("HTTP libraries not available")
                
        except Exception as e:
            logger.warning(f"HTTP client initialization had issues: {e}")

    async def _load_api_endpoints(self) -> None:
        """Load API endpoint configurations."""
        try:
            # ElevenLabs TTS API
            self.api_endpoints["elevenlabs"] = APIEndpoint(
                name="ElevenLabs",
                base_url="https://api.elevenlabs.io/v1",
                api_key="your_elevenlabs_api_key",
                rate_limit=100,
                timeout=30,
                headers={"xi-api-key": "your_elevenlabs_api_key"}
            )
            
            # OpenAI API
            self.api_endpoints["openai"] = APIEndpoint(
                name="OpenAI",
                base_url="https://api.openai.com/v1",
                api_key="your_openai_api_key",
                rate_limit=3000,
                timeout=60,
                headers={"Authorization": "Bearer your_openai_api_key"}
            )
            
            # Anthropic API
            self.api_endpoints["anthropic"] = APIEndpoint(
                name="Anthropic",
                base_url="https://api.anthropic.com/v1",
                api_key="your_anthropic_api_key",
                rate_limit=100,
                timeout=60,
                headers={"x-api-key": "your_anthropic_api_key"}
            )
            
            # YouTube API
            self.api_endpoints["youtube"] = APIEndpoint(
                name="YouTube",
                base_url="https://www.googleapis.com/youtube/v3",
                api_key="your_youtube_api_key",
                rate_limit=10000,
                timeout=30
            )
            
            # Instagram API
            self.api_endpoints["instagram"] = APIEndpoint(
                name="Instagram",
                base_url="https://graph.instagram.com/v12.0",
                api_key="your_instagram_access_token",
                rate_limit=200,
                timeout=30
            )
            
            # TikTok API
            self.api_endpoints["tiktok"] = APIEndpoint(
                name="TikTok",
                base_url="https://open.tiktokapis.com/v2",
                api_key="your_tiktok_access_token",
                rate_limit=100,
                timeout=30
            )
            
            # LinkedIn API
            self.api_endpoints["linkedin"] = APIEndpoint(
                name="LinkedIn",
                base_url="https://api.linkedin.com/v2",
                api_key="your_linkedin_access_token",
                rate_limit=100,
                timeout=30
            )
            
            # Facebook API
            self.api_endpoints["facebook"] = APIEndpoint(
                name="Facebook",
                base_url="https://graph.facebook.com/v18.0",
                api_key="your_facebook_access_token",
                rate_limit=200,
                timeout=30
            )
            
            # Twitter API
            self.api_endpoints["twitter"] = APIEndpoint(
                name="Twitter",
                base_url="https://api.twitter.com/2",
                api_key="your_twitter_bearer_token",
                rate_limit=300,
                timeout=30,
                authentication_type="bearer",
                headers={"Authorization": "Bearer your_twitter_bearer_token"}
            )
            
            # Twitch API
            self.api_endpoints["twitch"] = APIEndpoint(
                name="Twitch",
                base_url="https://api.twitch.tv/helix",
                api_key="your_twitch_client_id",
                rate_limit=800,
                timeout=30,
                headers={"Client-ID": "your_twitch_client_id"}
            )
            
            logger.info(f"Loaded {len(self.api_endpoints)} API endpoints")
            
        except Exception as e:
            logger.warning(f"Failed to load some API endpoints: {e}")

    async def _initialize_cloud_storage(self) -> None:
        """Initialize cloud storage clients."""
        try:
            if CLOUD_STORAGE_AVAILABLE:
                # AWS S3
                try:
                    self.aws_s3_client = boto3.client(
                        's3',
                        region_name='us-east-1',
                        aws_access_key_id='your_access_key',
                        aws_secret_access_key='your_secret_key'
                    )
                    logger.info("AWS S3 client initialized")
                except Exception as e:
                    logger.warning(f"AWS S3 initialization failed: {e}")
                
                # Google Cloud Storage
                try:
                    self.gcp_storage_client = storage.Client()
                    logger.info("Google Cloud Storage client initialized")
                except Exception as e:
                    logger.warning(f"Google Cloud Storage initialization failed: {e}")
                
                # Azure Blob Storage
                try:
                    connection_string = "your_azure_connection_string"
                    self.azure_blob_client = BlobServiceClient.from_connection_string(connection_string)
                    logger.info("Azure Blob Storage client initialized")
                except Exception as e:
                    logger.warning(f"Azure Blob Storage initialization failed: {e}")
                
            else:
                logger.warning("Cloud storage libraries not available")
                
        except Exception as e:
            logger.warning(f"Cloud storage initialization had issues: {e}")

    async def _validate_configuration(self) -> None:
        """Validate API manager configuration."""
        if not self.api_endpoints:
            raise RuntimeError("No API endpoints configured")
        
        if not HTTP_AVAILABLE:
            raise RuntimeError("HTTP libraries not available")

    @with_error_handling
    @with_retry(max_attempts=3)
    async def make_api_request(self, request: APIRequest) -> APIResponse:
        """Make a request to an external API."""
        start_time = time.time()
        
        try:
            logger.info(f"Making API request {request.request_id} to {request.endpoint}")
            
            # Validate request
            if not request.endpoint or request.endpoint not in self.api_endpoints:
                raise ValueError(f"Invalid endpoint: {request.endpoint}")
            
            # Check rate limiting
            await self._check_rate_limit(request.endpoint)
            
            # Make HTTP request
            response = await self._execute_http_request(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update statistics
            self._update_api_stats(response_time, response.success)
            
            logger.info(f"API request completed in {response_time:.2f}s")
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_api_stats(response_time, False)
            logger.error(f"API request failed: {e}")
            raise

    async def _check_rate_limit(self, endpoint: str) -> None:
        """Check rate limiting for an endpoint."""
        try:
            if endpoint not in self.rate_limiters:
                self.rate_limiters[endpoint] = {
                    "requests": [],
                    "limit": self.api_endpoints[endpoint].rate_limit
                }
            
            limiter = self.rate_limiters[endpoint]
            current_time = time.time()
            
            # Remove old requests (older than 1 minute)
            limiter["requests"] = [req_time for req_time in limiter["requests"] 
                                 if current_time - req_time < 60]
            
            # Check if we're at the limit
            if len(limiter["requests"]) >= limiter["limit"]:
                wait_time = 60 - (current_time - limiter["requests"][0])
                if wait_time > 0:
                    logger.warning(f"Rate limit reached for {endpoint}, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
            
            # Add current request
            limiter["requests"].append(current_time)
            
        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")

    async def _execute_http_request(self, request: APIRequest) -> APIResponse:
        """Execute the HTTP request."""
        try:
            endpoint_config = self.api_endpoints[request.endpoint]
            
            # Build full URL
            if request.url.startswith("http"):
                full_url = request.url
            else:
                full_url = f"{endpoint_config.base_url}{request.url}"
            
            # Merge headers
            headers = {**endpoint_config.headers, **request.headers}
            
            # Use httpx for the request
            if self.httpx_client:
                async with self.httpx_client.stream(
                    method=request.method,
                    url=full_url,
                    headers=headers,
                    params=request.params,
                    json=request.data,
                    files=request.files,
                    timeout=request.timeout
                ) as response:
                    response_data = await response.aread()
                    
                    # Parse response data
                    try:
                        parsed_data = response.json()
                    except:
                        parsed_data = response_data.decode('utf-8')
                    
                    return APIResponse(
                        request_id=request.request_id,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        data=parsed_data,
                        response_time=0.0,  # Will be set by caller
                        success=response.status_code < 400,
                        error_message=None if response.status_code < 400 else f"HTTP {response.status_code}"
                    )
            else:
                raise RuntimeError("HTTP client not available")
                
        except Exception as e:
            return APIResponse(
                request_id=request.request_id,
                status_code=0,
                headers={},
                data=None,
                response_time=0.0,
                success=False,
                error_message=str(e)
            )

    def _update_api_stats(self, response_time: float, success: bool):
        """Update API statistics."""
        self.api_stats["total_requests"] += 1
        
        if success:
            self.api_stats["successful_requests"] += 1
            self.api_stats["total_response_time"] += response_time
        else:
            self.api_stats["failed_requests"] += 1
        
        # Update average response time
        current_avg = self.api_stats["average_response_time"]
        total_successful = self.api_stats["successful_requests"]
        
        if total_successful > 0:
            self.api_stats["average_response_time"] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )

    async def health_check_all(self) -> Dict[str, HealthCheckResult]:
        """Check health of all external APIs."""
        try:
            health_results = {}
            
            for endpoint_name, endpoint_config in self.api_endpoints.items():
                try:
                    # Make a simple health check request
                    health_request = APIRequest(
                        endpoint=endpoint_name,
                        method="GET",
                        url="/health" if endpoint_config.base_url.endswith("/v1") else "",
                        timeout=10
                    )
                    
                    response = await self.make_api_request(health_request)
                    
                    if response.success:
                        health_results[endpoint_name] = HealthCheckResult(
                            status=ServiceStatus.HEALTHY,
                            details={
                                "endpoint": endpoint_name,
                                "base_url": endpoint_config.base_url,
                                "rate_limit": endpoint_config.rate_limit,
                                "last_response_time": response.response_time
                            }
                        )
                    else:
                        health_results[endpoint_name] = HealthCheckResult(
                            status=ServiceStatus.DEGRADED,
                            details={
                                "endpoint": endpoint_name,
                                "error": response.error_message
                            }
                        )
                        
                except Exception as e:
                    health_results[endpoint_name] = HealthCheckResult(
                        status=ServiceStatus.UNHEALTHY,
                        error_message=str(e)
                    )
            
            return health_results
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"error": HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                error_message=str(e)
            )}

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the external API manager."""
        try:
            # Check base service health
            base_health = await super().health_check()
            
            # Check dependencies
            dependencies = {
                "http_libraries": HTTP_AVAILABLE,
                "cloud_storage": CLOUD_STORAGE_AVAILABLE
            }
            
            # Check HTTP clients
            http_clients = {
                "aiohttp_session": self.http_session is not None,
                "httpx_client": self.httpx_client is not None
            }
            
            # Check cloud storage clients
            cloud_clients = {
                "aws_s3": self.aws_s3_client is not None,
                "gcp_storage": self.gcp_storage_client is not None,
                "azure_blob": self.azure_blob_client is not None
            }
            
            # Update base health
            base_health.details.update({
                "dependencies": dependencies,
                "http_clients": http_clients,
                "cloud_clients": cloud_clients,
                "api_endpoints": len(self.api_endpoints),
                "api_stats": self.api_stats
            })
            
            return base_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                error_message=str(e)
            )

    async def get_api_endpoint(self, name: str) -> Optional[APIEndpoint]:
        """Get a specific API endpoint configuration."""
        return self.api_endpoints.get(name)

    async def add_api_endpoint(self, endpoint: APIEndpoint) -> bool:
        """Add a new API endpoint."""
        try:
            self.api_endpoints[endpoint.name] = endpoint
            logger.info(f"Added API endpoint: {endpoint.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add API endpoint: {e}")
            return False

    async def remove_api_endpoint(self, name: str) -> bool:
        """Remove an API endpoint."""
        try:
            if name in self.api_endpoints:
                del self.api_endpoints[name]
                logger.info(f"Removed API endpoint: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove API endpoint: {e}")
            return False

    async def upload_to_cloud_storage(self, file_path: str, config: CloudStorageConfig) -> str:
        """Upload a file to cloud storage."""
        try:
            if not Path(file_path).exists():
                raise ValueError(f"File not found: {file_path}")
            
            if config.provider == "aws":
                if not self.aws_s3_client:
                    raise RuntimeError("AWS S3 client not initialized")
                
                # Upload to S3
                file_name = Path(file_path).name
                self.aws_s3_client.upload_file(
                    file_path, config.bucket_name, file_name
                )
                
                return f"s3://{config.bucket_name}/{file_name}"
                
            elif config.provider == "gcp":
                if not self.gcp_storage_client:
                    raise RuntimeError("Google Cloud Storage client not initialized")
                
                # Upload to GCP
                bucket = self.gcp_storage_client.bucket(config.bucket_name)
                blob = bucket.blob(Path(file_path).name)
                blob.upload_from_filename(file_path)
                
                return f"gs://{config.bucket_name}/{Path(file_path).name}"
                
            elif config.provider == "azure":
                if not self.azure_blob_client:
                    raise RuntimeError("Azure Blob Storage client not initialized")
                
                # Upload to Azure
                container_client = self.azure_blob_client.get_container_client(config.bucket_name)
                blob_client = container_client.get_blob_client(Path(file_path).name)
                
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                
                return f"https://{config.bucket_name}.blob.core.windows.net/{Path(file_path).name}"
                
            else:
                raise ValueError(f"Unsupported cloud provider: {config.provider}")
                
        except Exception as e:
            logger.error(f"Cloud storage upload failed: {e}")
            raise

    async def cleanup_temp_files(self) -> None:
        """Clean up temporary API files."""
        try:
            temp_dir = Path("./temp")
            if temp_dir.exists():
                for api_file in temp_dir.glob("api_*"):
                    api_file.unlink()
                    logger.debug(f"Cleaned up temp file: {api_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    async def shutdown(self) -> None:
        """Shutdown the external API manager."""
        try:
            if self.http_session:
                await self.http_session.close()
            
            if self.httpx_client:
                await self.httpx_client.aclose()
            
            logger.info("External API manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
