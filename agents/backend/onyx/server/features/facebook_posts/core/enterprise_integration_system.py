"""
Enterprise Integration System
Ultra-modular Facebook Posts System v8.0

Enterprise integration capabilities:
- API management and governance
- Enterprise authentication and authorization
- Multi-tenant architecture
- Service mesh integration
- Enterprise monitoring and logging
- Compliance and audit trails
- Enterprise security
- Business intelligence integration
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from collections import defaultdict
import structlog
from pathlib import Path
import yaml
import jwt
from cryptography.fernet import Fernet
import hashlib
import hmac
import base64

logger = structlog.get_logger(__name__)

class IntegrationType(Enum):
    """Integration types"""
    API_GATEWAY = "api_gateway"
    SERVICE_MESH = "service_mesh"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    CACHE = "cache"
    MONITORING = "monitoring"
    LOGGING = "logging"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    AUDIT = "audit"

class TenantTier(Enum):
    """Tenant tiers"""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    PREMIUM = "premium"
    CUSTOM = "custom"

class ServiceStatus(Enum):
    """Service status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

class AuditEventType(Enum):
    """Audit event types"""
    API_CALL = "api_call"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"
    BUSINESS_EVENT = "business_event"

@dataclass
class Tenant:
    """Tenant information"""
    tenant_id: str
    name: str
    tier: TenantTier
    status: str
    created_at: datetime
    updated_at: datetime
    settings: Dict[str, Any]
    limits: Dict[str, Any]
    features: List[str]
    api_keys: List[str]
    webhooks: List[str]
    metadata: Dict[str, Any]

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    service_id: str
    name: str
    url: str
    method: str
    integration_type: IntegrationType
    status: ServiceStatus
    health_check_url: Optional[str] = None
    timeout: float = 30.0
    retry_count: int = 3
    circuit_breaker: bool = True
    rate_limit: Optional[int] = None
    authentication_required: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class APIKey:
    """API key information"""
    key_id: str
    tenant_id: str
    name: str
    key_hash: str
    permissions: List[str]
    rate_limit: int
    expires_at: Optional[datetime] = None
    created_at: datetime = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class AuditEvent:
    """Audit event"""
    event_id: str
    tenant_id: str
    user_id: Optional[str]
    event_type: AuditEventType
    action: str
    resource: str
    result: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ServiceHealth:
    """Service health information"""
    service_id: str
    status: ServiceStatus
    response_time: float
    error_rate: float
    uptime: float
    last_check: datetime
    details: Dict[str, Any]

class EnterpriseIntegrationSystem:
    """Enterprise integration and API management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_running = False
        self.is_initialized = False
        
        # Enterprise configuration
        self.multi_tenant_enabled = self.config.get("multi_tenant_enabled", True)
        self.api_gateway_enabled = self.config.get("api_gateway_enabled", True)
        self.service_mesh_enabled = self.config.get("service_mesh_enabled", True)
        self.audit_enabled = self.config.get("audit_enabled", True)
        self.rate_limiting_enabled = self.config.get("rate_limiting_enabled", True)
        
        # Data storage
        self.tenants = {}
        self.service_endpoints = {}
        self.api_keys = {}
        self.audit_events = deque(maxlen=1000000)  # 1M audit events
        self.service_health = {}
        
        # Rate limiting
        self.rate_limits = defaultdict(lambda: defaultdict(int))
        self.rate_limit_windows = defaultdict(lambda: defaultdict(float))
        
        # Circuit breakers
        self.circuit_breakers = {}
        
        # Performance metrics
        self.performance_metrics = {
            "api_calls_total": 0,
            "api_calls_successful": 0,
            "api_calls_failed": 0,
            "rate_limited_requests": 0,
            "circuit_breaker_trips": 0,
            "audit_events_logged": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0
        }
        
        # HTTP client for external calls
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Encryption key for sensitive data
        self.encryption_key = self.config.get("encryption_key", Fernet.generate_key())
        self.cipher = Fernet(self.encryption_key)
        
        # JWT secret for token generation
        self.jwt_secret = self.config.get("jwt_secret", "enterprise-secret-key")
        
    async def initialize(self) -> bool:
        """Initialize enterprise integration system"""
        try:
            logger.info("Initializing Enterprise Integration System...")
            
            # Initialize tenant management
            await self._initialize_tenant_management()
            
            # Initialize service discovery
            await self._initialize_service_discovery()
            
            # Initialize API gateway
            await self._initialize_api_gateway()
            
            # Initialize authentication and authorization
            await self._initialize_auth_system()
            
            # Initialize audit system
            await self._initialize_audit_system()
            
            # Initialize monitoring and health checks
            await self._initialize_monitoring()
            
            # Load existing configuration
            await self._load_configuration()
            
            self.is_initialized = True
            logger.info("✓ Enterprise Integration System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enterprise Integration System: {e}")
            return False
    
    async def start(self) -> bool:
        """Start enterprise integration system"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info("Starting Enterprise Integration System...")
            
            # Start health monitoring
            self.health_monitoring_task = asyncio.create_task(self._monitor_service_health())
            
            # Start rate limit cleanup
            self.rate_limit_cleanup_task = asyncio.create_task(self._cleanup_rate_limits())
            
            # Start audit processing
            self.audit_processing_task = asyncio.create_task(self._process_audit_events())
            
            # Start circuit breaker monitoring
            self.circuit_breaker_task = asyncio.create_task(self._monitor_circuit_breakers())
            
            self.is_running = True
            logger.info("✓ Enterprise Integration System started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Enterprise Integration System: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop enterprise integration system"""
        try:
            logger.info("Stopping Enterprise Integration System...")
            
            self.is_running = False
            
            # Cancel background tasks
            if hasattr(self, 'health_monitoring_task'):
                self.health_monitoring_task.cancel()
            
            if hasattr(self, 'rate_limit_cleanup_task'):
                self.rate_limit_cleanup_task.cancel()
            
            if hasattr(self, 'audit_processing_task'):
                self.audit_processing_task.cancel()
            
            if hasattr(self, 'circuit_breaker_task'):
                self.circuit_breaker_task.cancel()
            
            # Close HTTP client
            await self.http_client.aclose()
            
            # Save configuration
            await self._save_configuration()
            
            logger.info("✓ Enterprise Integration System stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Enterprise Integration System: {e}")
            return False
    
    async def _initialize_tenant_management(self) -> None:
        """Initialize tenant management system"""
        logger.info("Initializing tenant management...")
        
        # Initialize default tenant
        default_tenant = Tenant(
            tenant_id="default",
            name="Default Tenant",
            tier=TenantTier.ENTERPRISE,
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            settings={
                "max_api_calls_per_minute": 10000,
                "max_data_storage": "1TB",
                "features_enabled": ["all"]
            },
            limits={
                "api_calls_per_minute": 10000,
                "api_calls_per_day": 1000000,
                "data_storage_mb": 1048576,
                "concurrent_requests": 1000
            },
            features=["all"],
            api_keys=[],
            webhooks=[],
            metadata={}
        )
        
        self.tenants["default"] = default_tenant
        
        logger.info("✓ Tenant management initialized")
    
    async def _initialize_service_discovery(self) -> None:
        """Initialize service discovery"""
        logger.info("Initializing service discovery...")
        
        # Initialize default services
        default_services = [
            ServiceEndpoint(
                service_id="facebook-posts-api",
                name="Facebook Posts API",
                url="http://localhost:8000",
                method="GET",
                integration_type=IntegrationType.API_GATEWAY,
                status=ServiceStatus.HEALTHY,
                health_check_url="http://localhost:8000/health",
                timeout=30.0,
                retry_count=3,
                circuit_breaker=True,
                rate_limit=1000,
                authentication_required=True,
                metadata={"version": "8.0.0"}
            ),
            ServiceEndpoint(
                service_id="analytics-service",
                name="Analytics Service",
                url="http://localhost:8001",
                method="GET",
                integration_type=IntegrationType.API_GATEWAY,
                status=ServiceStatus.HEALTHY,
                health_check_url="http://localhost:8001/health",
                timeout=30.0,
                retry_count=3,
                circuit_breaker=True,
                rate_limit=500,
                authentication_required=True,
                metadata={"version": "8.0.0"}
            ),
            ServiceEndpoint(
                service_id="neural-interface-service",
                name="Neural Interface Service",
                url="http://localhost:8002",
                method="GET",
                integration_type=IntegrationType.API_GATEWAY,
                status=ServiceStatus.HEALTHY,
                health_check_url="http://localhost:8002/health",
                timeout=30.0,
                retry_count=3,
                circuit_breaker=True,
                rate_limit=100,
                authentication_required=True,
                metadata={"version": "8.0.0"}
            )
        ]
        
        for service in default_services:
            self.service_endpoints[service.service_id] = service
            self.service_health[service.service_id] = ServiceHealth(
                service_id=service.service_id,
                status=ServiceStatus.UNKNOWN,
                response_time=0.0,
                error_rate=0.0,
                uptime=0.0,
                last_check=datetime.now(),
                details={}
            )
        
        logger.info("✓ Service discovery initialized")
    
    async def _initialize_api_gateway(self) -> None:
        """Initialize API gateway"""
        logger.info("Initializing API gateway...")
        
        # Initialize routing rules
        self.routing_rules = {
            "/api/v1/": "facebook-posts-api",
            "/api/v2/": "facebook-posts-api",
            "/api/v3/": "facebook-posts-api",
            "/api/v5/": "facebook-posts-api",
            "/analytics/": "analytics-service",
            "/neural/": "neural-interface-service",
            "/holographic/": "holographic-interface-service",
            "/quantum/": "quantum-ai-service"
        }
        
        # Initialize middleware
        self.middleware = [
            self._rate_limit_middleware,
            self._authentication_middleware,
            self._authorization_middleware,
            self._audit_middleware,
            self._circuit_breaker_middleware
        ]
        
        logger.info("✓ API gateway initialized")
    
    async def _initialize_auth_system(self) -> None:
        """Initialize authentication and authorization system"""
        logger.info("Initializing authentication and authorization...")
        
        # Initialize JWT configuration
        self.jwt_config = {
            "algorithm": "HS256",
            "expiration": 3600,  # 1 hour
            "refresh_expiration": 86400  # 24 hours
        }
        
        # Initialize permission system
        self.permissions = {
            "read": ["GET"],
            "write": ["POST", "PUT", "PATCH"],
            "delete": ["DELETE"],
            "admin": ["*"]
        }
        
        # Initialize role-based access control
        self.roles = {
            "admin": ["*"],
            "user": ["read", "write"],
            "viewer": ["read"],
            "api": ["read", "write"]
        }
        
        logger.info("✓ Authentication and authorization initialized")
    
    async def _initialize_audit_system(self) -> None:
        """Initialize audit system"""
        logger.info("Initializing audit system...")
        
        # Initialize audit configuration
        self.audit_config = {
            "retention_days": 365,
            "batch_size": 1000,
            "flush_interval": 60,  # seconds
            "encryption_enabled": True
        }
        
        # Initialize audit storage
        self.audit_storage = deque(maxlen=1000000)
        
        logger.info("✓ Audit system initialized")
    
    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring and health checks"""
        logger.info("Initializing monitoring...")
        
        # Initialize health check configuration
        self.health_check_config = {
            "interval": 30,  # seconds
            "timeout": 10,  # seconds
            "retry_count": 3,
            "failure_threshold": 3
        }
        
        # Initialize metrics collection
        self.metrics_collection = {
            "response_times": deque(maxlen=10000),
            "error_rates": deque(maxlen=10000),
            "throughput": deque(maxlen=10000)
        }
        
        logger.info("✓ Monitoring initialized")
    
    async def _load_configuration(self) -> None:
        """Load configuration from storage"""
        try:
            config_path = Path("config/enterprise_integration.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Load tenants
                if "tenants" in config:
                    for tenant_data in config["tenants"]:
                        tenant = Tenant(**tenant_data)
                        self.tenants[tenant.tenant_id] = tenant
                
                # Load service endpoints
                if "services" in config:
                    for service_data in config["services"]:
                        service = ServiceEndpoint(**service_data)
                        self.service_endpoints[service.service_id] = service
                
                logger.info("Configuration loaded successfully")
        
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
    
    async def _save_configuration(self) -> None:
        """Save configuration to storage"""
        try:
            config_path = Path("config/enterprise_integration.yaml")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config = {
                "tenants": [asdict(tenant) for tenant in self.tenants.values()],
                "services": [asdict(service) for service in self.service_endpoints.values()],
                "api_keys": [asdict(key) for key in self.api_keys.values()],
                "routing_rules": self.routing_rules,
                "audit_config": self.audit_config
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info("Configuration saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    async def _monitor_service_health(self) -> None:
        """Monitor service health"""
        while self.is_running:
            try:
                for service_id, service in self.service_endpoints.items():
                    await self._check_service_health(service)
                
                await asyncio.sleep(self.health_check_config["interval"])
                
            except Exception as e:
                logger.error(f"Service health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_service_health(self, service: ServiceEndpoint) -> None:
        """Check health of a specific service"""
        try:
            start_time = time.time()
            
            if service.health_check_url:
                response = await self.http_client.get(
                    service.health_check_url,
                    timeout=self.health_check_config["timeout"]
                )
                
                response_time = time.time() - start_time
                is_healthy = response.status_code == 200
                
                # Update service health
                health = self.service_health[service.service_id]
                health.response_time = response_time
                health.last_check = datetime.now()
                
                if is_healthy:
                    health.status = ServiceStatus.HEALTHY
                    health.error_rate = max(0.0, health.error_rate - 0.1)
                else:
                    health.status = ServiceStatus.UNHEALTHY
                    health.error_rate = min(1.0, health.error_rate + 0.1)
                
                # Update service status
                service.status = health.status
                
                # Update metrics
                self.metrics_collection["response_times"].append(response_time)
                
            else:
                # No health check URL, assume healthy
                health = self.service_health[service.service_id]
                health.status = ServiceStatus.HEALTHY
                health.last_check = datetime.now()
                service.status = ServiceStatus.HEALTHY
        
        except Exception as e:
            logger.error(f"Health check failed for {service.service_id}: {e}")
            
            # Update service health
            health = self.service_health[service.service_id]
            health.status = ServiceStatus.UNHEALTHY
            health.error_rate = min(1.0, health.error_rate + 0.1)
            health.last_check = datetime.now()
            service.status = ServiceStatus.UNHEALTHY
    
    async def _cleanup_rate_limits(self) -> None:
        """Clean up expired rate limits"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Clean up expired rate limit windows
                for tenant_id in list(self.rate_limit_windows.keys()):
                    for endpoint in list(self.rate_limit_windows[tenant_id].keys()):
                        if current_time - self.rate_limit_windows[tenant_id][endpoint] > 60:  # 1 minute window
                            del self.rate_limit_windows[tenant_id][endpoint]
                            if endpoint in self.rate_limits[tenant_id]:
                                del self.rate_limits[tenant_id][endpoint]
                
                await asyncio.sleep(60)  # Clean up every minute
                
            except Exception as e:
                logger.error(f"Rate limit cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _process_audit_events(self) -> None:
        """Process audit events"""
        while self.is_running:
            try:
                # Process audit events in batches
                if len(self.audit_events) > 0:
                    batch_size = min(self.audit_config["batch_size"], len(self.audit_events))
                    batch = [self.audit_events.popleft() for _ in range(batch_size)]
                    
                    # Process batch
                    await self._process_audit_batch(batch)
                
                await asyncio.sleep(self.audit_config["flush_interval"])
                
            except Exception as e:
                logger.error(f"Audit processing error: {e}")
                await asyncio.sleep(60)
    
    async def _process_audit_batch(self, batch: List[AuditEvent]) -> None:
        """Process a batch of audit events"""
        try:
            # Log audit events
            for event in batch:
                logger.info(
                    "Audit event",
                    event_id=event.event_id,
                    tenant_id=event.tenant_id,
                    user_id=event.user_id,
                    event_type=event.event_type.value,
                    action=event.action,
                    resource=event.resource,
                    result=event.result,
                    timestamp=event.timestamp.isoformat(),
                    ip_address=event.ip_address,
                    user_agent=event.user_agent
                )
            
            # Update metrics
            self.performance_metrics["audit_events_logged"] += len(batch)
            
        except Exception as e:
            logger.error(f"Audit batch processing error: {e}")
    
    async def _monitor_circuit_breakers(self) -> None:
        """Monitor circuit breakers"""
        while self.is_running:
            try:
                for service_id, service in self.service_endpoints.items():
                    if service.circuit_breaker:
                        await self._check_circuit_breaker(service_id, service)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Circuit breaker monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _check_circuit_breaker(self, service_id: str, service: ServiceEndpoint) -> None:
        """Check circuit breaker for a service"""
        try:
            health = self.service_health[service_id]
            
            # Initialize circuit breaker if not exists
            if service_id not in self.circuit_breakers:
                self.circuit_breakers[service_id] = {
                    "state": "closed",  # closed, open, half-open
                    "failure_count": 0,
                    "last_failure": None,
                    "next_attempt": None
                }
            
            cb = self.circuit_breakers[service_id]
            current_time = datetime.now()
            
            # Check if circuit breaker should be opened
            if health.error_rate > 0.5 and cb["state"] == "closed":
                cb["state"] = "open"
                cb["last_failure"] = current_time
                cb["next_attempt"] = current_time + timedelta(minutes=5)
                self.performance_metrics["circuit_breaker_trips"] += 1
                logger.warning(f"Circuit breaker opened for {service_id}")
            
            # Check if circuit breaker should be half-opened
            elif cb["state"] == "open" and cb["next_attempt"] and current_time >= cb["next_attempt"]:
                cb["state"] = "half-open"
                logger.info(f"Circuit breaker half-opened for {service_id}")
            
            # Check if circuit breaker should be closed
            elif cb["state"] == "half-open" and health.error_rate < 0.1:
                cb["state"] = "closed"
                cb["failure_count"] = 0
                logger.info(f"Circuit breaker closed for {service_id}")
        
        except Exception as e:
            logger.error(f"Circuit breaker check error for {service_id}: {e}")
    
    # Middleware functions
    
    async def _rate_limit_middleware(self, request: Dict[str, Any]) -> bool:
        """Rate limiting middleware"""
        try:
            if not self.rate_limiting_enabled:
                return True
            
            tenant_id = request.get("tenant_id", "default")
            endpoint = request.get("endpoint", "/")
            current_time = time.time()
            
            # Get tenant limits
            tenant = self.tenants.get(tenant_id)
            if not tenant:
                return False
            
            rate_limit = tenant.limits.get("api_calls_per_minute", 1000)
            
            # Check rate limit
            window_key = f"{tenant_id}:{endpoint}"
            if current_time - self.rate_limit_windows[tenant_id][endpoint] > 60:
                self.rate_limit_windows[tenant_id][endpoint] = current_time
                self.rate_limits[tenant_id][endpoint] = 0
            
            if self.rate_limits[tenant_id][endpoint] >= rate_limit:
                self.performance_metrics["rate_limited_requests"] += 1
                return False
            
            self.rate_limits[tenant_id][endpoint] += 1
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            return False
    
    async def _authentication_middleware(self, request: Dict[str, Any]) -> bool:
        """Authentication middleware"""
        try:
            api_key = request.get("api_key")
            if not api_key:
                return False
            
            # Find API key
            api_key_obj = None
            for key in self.api_keys.values():
                if self._verify_api_key(api_key, key.key_hash):
                    api_key_obj = key
                    break
            
            if not api_key_obj or not api_key_obj.is_active:
                return False
            
            # Check expiration
            if api_key_obj.expires_at and datetime.now() > api_key_obj.expires_at:
                return False
            
            # Update last used
            api_key_obj.last_used = datetime.now()
            
            # Add tenant info to request
            request["tenant_id"] = api_key_obj.tenant_id
            request["api_key_id"] = api_key_obj.key_id
            
            return True
            
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            return False
    
    async def _authorization_middleware(self, request: Dict[str, Any]) -> bool:
        """Authorization middleware"""
        try:
            api_key_id = request.get("api_key_id")
            if not api_key_id:
                return False
            
            api_key = self.api_keys.get(api_key_id)
            if not api_key:
                return False
            
            # Check permissions
            required_permission = self._get_required_permission(request.get("method", "GET"))
            if required_permission not in api_key.permissions and "admin" not in api_key.permissions:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Authorization middleware error: {e}")
            return False
    
    async def _audit_middleware(self, request: Dict[str, Any]) -> bool:
        """Audit middleware"""
        try:
            if not self.audit_enabled:
                return True
            
            # Create audit event
            audit_event = AuditEvent(
                event_id=str(uuid.uuid4()),
                tenant_id=request.get("tenant_id", "unknown"),
                user_id=request.get("user_id"),
                event_type=AuditEventType.API_CALL,
                action=request.get("method", "GET"),
                resource=request.get("endpoint", "/"),
                result="success",
                timestamp=datetime.now(),
                ip_address=request.get("ip_address"),
                user_agent=request.get("user_agent"),
                metadata=request.get("metadata", {})
            )
            
            # Add to audit queue
            self.audit_events.append(audit_event)
            
            return True
            
        except Exception as e:
            logger.error(f"Audit middleware error: {e}")
            return True  # Don't fail the request for audit errors
    
    async def _circuit_breaker_middleware(self, request: Dict[str, Any]) -> bool:
        """Circuit breaker middleware"""
        try:
            service_id = request.get("service_id")
            if not service_id:
                return True
            
            # Check circuit breaker state
            if service_id in self.circuit_breakers:
                cb = self.circuit_breakers[service_id]
                if cb["state"] == "open":
                    return False
                elif cb["state"] == "half-open":
                    # Allow one request to test
                    pass
            
            return True
            
        except Exception as e:
            logger.error(f"Circuit breaker middleware error: {e}")
            return True
    
    def _verify_api_key(self, api_key: str, key_hash: str) -> bool:
        """Verify API key"""
        try:
            # Hash the provided key
            provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
            return hmac.compare_digest(provided_hash, key_hash)
        except Exception:
            return False
    
    def _get_required_permission(self, method: str) -> str:
        """Get required permission for HTTP method"""
        if method in ["GET", "HEAD", "OPTIONS"]:
            return "read"
        elif method in ["POST", "PUT", "PATCH"]:
            return "write"
        elif method == "DELETE":
            return "delete"
        else:
            return "read"
    
    # Public API methods
    
    async def create_tenant(self, tenant_data: Dict[str, Any]) -> str:
        """Create a new tenant"""
        try:
            tenant_id = str(uuid.uuid4())
            
            tenant = Tenant(
                tenant_id=tenant_id,
                name=tenant_data["name"],
                tier=TenantTier(tenant_data.get("tier", "basic")),
                status="active",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                settings=tenant_data.get("settings", {}),
                limits=tenant_data.get("limits", {}),
                features=tenant_data.get("features", []),
                api_keys=[],
                webhooks=[],
                metadata=tenant_data.get("metadata", {})
            )
            
            self.tenants[tenant_id] = tenant
            
            # Create default API key
            api_key = await self.create_api_key(tenant_id, "Default API Key")
            
            logger.info(f"Created tenant: {tenant_id}")
            return tenant_id
            
        except Exception as e:
            logger.error(f"Failed to create tenant: {e}")
            raise
    
    async def create_api_key(self, tenant_id: str, name: str, 
                           permissions: List[str] = None, 
                           rate_limit: int = None) -> str:
        """Create API key for tenant"""
        try:
            # Generate API key
            api_key = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip('=')
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Get tenant
            tenant = self.tenants.get(tenant_id)
            if not tenant:
                raise ValueError(f"Tenant {tenant_id} not found")
            
            # Create API key object
            key_id = str(uuid.uuid4())
            api_key_obj = APIKey(
                key_id=key_id,
                tenant_id=tenant_id,
                name=name,
                key_hash=key_hash,
                permissions=permissions or ["read", "write"],
                rate_limit=rate_limit or tenant.limits.get("api_calls_per_minute", 1000),
                created_at=datetime.now(),
                is_active=True,
                metadata={}
            )
            
            self.api_keys[key_id] = api_key_obj
            tenant.api_keys.append(key_id)
            
            logger.info(f"Created API key: {key_id} for tenant: {tenant_id}")
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise
    
    async def add_service_endpoint(self, service_data: Dict[str, Any]) -> str:
        """Add service endpoint"""
        try:
            service_id = str(uuid.uuid4())
            
            service = ServiceEndpoint(
                service_id=service_id,
                name=service_data["name"],
                url=service_data["url"],
                method=service_data.get("method", "GET"),
                integration_type=IntegrationType(service_data.get("integration_type", "api_gateway")),
                status=ServiceStatus.HEALTHY,
                health_check_url=service_data.get("health_check_url"),
                timeout=service_data.get("timeout", 30.0),
                retry_count=service_data.get("retry_count", 3),
                circuit_breaker=service_data.get("circuit_breaker", True),
                rate_limit=service_data.get("rate_limit"),
                authentication_required=service_data.get("authentication_required", True),
                metadata=service_data.get("metadata", {})
            )
            
            self.service_endpoints[service_id] = service
            self.service_health[service_id] = ServiceHealth(
                service_id=service_id,
                status=ServiceStatus.UNKNOWN,
                response_time=0.0,
                error_rate=0.0,
                uptime=0.0,
                last_check=datetime.now(),
                details={}
            )
            
            logger.info(f"Added service endpoint: {service_id}")
            return service_id
            
        except Exception as e:
            logger.error(f"Failed to add service endpoint: {e}")
            raise
    
    async def route_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request through API gateway"""
        try:
            # Apply middleware
            for middleware in self.middleware:
                if not await middleware(request):
                    return {
                        "success": False,
                        "error": "Request blocked by middleware",
                        "status_code": 403
                    }
            
            # Find target service
            endpoint = request.get("endpoint", "/")
            target_service = None
            
            for pattern, service_id in self.routing_rules.items():
                if endpoint.startswith(pattern):
                    target_service = self.service_endpoints.get(service_id)
                    break
            
            if not target_service:
                return {
                    "success": False,
                    "error": "No service found for endpoint",
                    "status_code": 404
                }
            
            # Check circuit breaker
            if target_service.circuit_breaker:
                cb = self.circuit_breakers.get(target_service.service_id)
                if cb and cb["state"] == "open":
                    return {
                        "success": False,
                        "error": "Service temporarily unavailable",
                        "status_code": 503
                    }
            
            # Make request to target service
            start_time = time.time()
            
            try:
                response = await self.http_client.request(
                    method=request.get("method", "GET"),
                    url=f"{target_service.url}{endpoint}",
                    headers=request.get("headers", {}),
                    json=request.get("data"),
                    timeout=target_service.timeout
                )
                
                response_time = time.time() - start_time
                
                # Update metrics
                self.performance_metrics["api_calls_total"] += 1
                self.performance_metrics["api_calls_successful"] += 1
                self.performance_metrics["total_response_time"] += response_time
                self.performance_metrics["avg_response_time"] = (
                    self.performance_metrics["total_response_time"] / 
                    self.performance_metrics["api_calls_total"]
                )
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    "response_time": response_time,
                    "service_id": target_service.service_id
                }
                
            except Exception as e:
                response_time = time.time() - start_time
                
                # Update metrics
                self.performance_metrics["api_calls_total"] += 1
                self.performance_metrics["api_calls_failed"] += 1
                
                # Update circuit breaker
                if target_service.circuit_breaker:
                    cb = self.circuit_breakers.get(target_service.service_id, {})
                    cb["failure_count"] = cb.get("failure_count", 0) + 1
                    cb["last_failure"] = datetime.now()
                
                return {
                    "success": False,
                    "error": str(e),
                    "status_code": 500,
                    "response_time": response_time,
                    "service_id": target_service.service_id
                }
            
        except Exception as e:
            logger.error(f"Request routing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 500
            }
    
    async def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant information"""
        tenant = self.tenants.get(tenant_id)
        return asdict(tenant) if tenant else None
    
    async def get_service_endpoints(self) -> List[Dict[str, Any]]:
        """Get all service endpoints"""
        return [asdict(service) for service in self.service_endpoints.values()]
    
    async def get_service_health(self, service_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get service health information"""
        if service_id:
            health = self.service_health.get(service_id)
            return asdict(health) if health else None
        else:
            return [asdict(health) for health in self.service_health.values()]
    
    async def get_audit_events(self, tenant_id: Optional[str] = None, 
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit events"""
        events = list(self.audit_events)
        
        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        
        return [asdict(e) for e in events[-limit:]]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get enterprise integration system health status"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "running": self.is_running,
            "tenants": len(self.tenants),
            "services": len(self.service_endpoints),
            "api_keys": len(self.api_keys),
            "audit_events": len(self.audit_events),
            "circuit_breakers": len(self.circuit_breakers)
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "enterprise_integration_system": {
                "status": "running" if self.is_running else "stopped",
                "tenants": len(self.tenants),
                "services": len(self.service_endpoints),
                "api_keys": len(self.api_keys),
                "audit_events": len(self.audit_events),
                "circuit_breakers": len(self.circuit_breakers),
                "performance": self.performance_metrics
            },
            "timestamp": datetime.now().isoformat()
        }
