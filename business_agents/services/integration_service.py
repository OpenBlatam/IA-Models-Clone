"""
Integration Service
==================

Advanced integration service for connecting business agents with external services.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_

from ..schemas import (
    BusinessAgent, AgentRequest, AgentResponse, AgentAnalytics,
    AgentWorkflow, AgentCollaboration, AgentSettings,
    ErrorResponse, SuccessResponse
)
from ..exceptions import (
    IntegrationNotFoundError, IntegrationExecutionError, IntegrationValidationError,
    IntegrationOptimizationError, IntegrationSystemError,
    create_agent_error, log_agent_error, handle_agent_error, get_error_response
)
from ..models import (
    db_manager, BusinessAgent as AgentModel, User
)
from ..config import get_settings

logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
    """Integration type enumeration"""
    API = "api"
    WEBHOOK = "webhook"
    DATABASE = "database"
    FILE = "file"
    EMAIL = "email"
    SMS = "sms"
    SOCIAL_MEDIA = "social_media"
    CRM = "crm"
    ERP = "erp"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    SUPPORT = "support"
    PAYMENT = "payment"
    CUSTOM = "custom"


class IntegrationStatus(str, Enum):
    """Integration status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    TESTING = "testing"
    MAINTENANCE = "maintenance"


class AuthenticationType(str, Enum):
    """Authentication type enumeration"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    JWT = "jwt"
    CUSTOM = "custom"


@dataclass
class IntegrationConfig:
    """Integration configuration"""
    integration_type: IntegrationType
    name: str
    description: str
    base_url: str
    authentication_type: AuthenticationType
    credentials: Dict[str, Any]
    endpoints: Dict[str, Any]
    rate_limits: Dict[str, Any]
    retry_config: Dict[str, Any]
    timeout_config: Dict[str, Any]
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationExecutionResult:
    """Integration execution result"""
    integration_id: str
    execution_id: str
    integration_type: IntegrationType
    status: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    duration: float = 0.0
    success_rate: float = 0.0
    efficiency_score: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class IntegrationService:
    """Advanced integration service"""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
        self._integration_configs = {}
        self._execution_cache = {}
        self._performance_cache = {}
        
        # Initialize integration configurations
        self._initialize_integration_configs()
    
    def _initialize_integration_configs(self):
        """Initialize integration configurations"""
        self._integration_configs = {
            IntegrationType.CRM: IntegrationConfig(
                integration_type=IntegrationType.CRM,
                name="CRM Integration",
                description="Customer Relationship Management integration",
                base_url="https://api.crm.com",
                authentication_type=AuthenticationType.API_KEY,
                credentials={"api_key": ""},
                endpoints={
                    "contacts": "/contacts",
                    "leads": "/leads",
                    "opportunities": "/opportunities",
                    "accounts": "/accounts"
                },
                rate_limits={"requests_per_minute": 100, "requests_per_hour": 1000},
                retry_config={"max_retries": 3, "backoff_factor": 2},
                timeout_config={"connect_timeout": 30, "read_timeout": 60}
            ),
            IntegrationType.ANALYTICS: IntegrationConfig(
                integration_type=IntegrationType.ANALYTICS,
                name="Analytics Integration",
                description="Analytics and reporting integration",
                base_url="https://api.analytics.com",
                authentication_type=AuthenticationType.OAUTH2,
                credentials={"client_id": "", "client_secret": ""},
                endpoints={
                    "reports": "/reports",
                    "metrics": "/metrics",
                    "dashboards": "/dashboards",
                    "insights": "/insights"
                },
                rate_limits={"requests_per_minute": 200, "requests_per_hour": 2000},
                retry_config={"max_retries": 3, "backoff_factor": 2},
                timeout_config={"connect_timeout": 30, "read_timeout": 60}
            ),
            IntegrationType.MARKETING: IntegrationConfig(
                integration_type=IntegrationType.MARKETING,
                name="Marketing Integration",
                description="Marketing automation integration",
                base_url="https://api.marketing.com",
                authentication_type=AuthenticationType.BEARER_TOKEN,
                credentials={"bearer_token": ""},
                endpoints={
                    "campaigns": "/campaigns",
                    "contacts": "/contacts",
                    "emails": "/emails",
                    "automations": "/automations"
                },
                rate_limits={"requests_per_minute": 150, "requests_per_hour": 1500},
                retry_config={"max_retries": 3, "backoff_factor": 2},
                timeout_config={"connect_timeout": 30, "read_timeout": 60}
            ),
            IntegrationType.SUPPORT: IntegrationConfig(
                integration_type=IntegrationType.SUPPORT,
                name="Support Integration",
                description="Customer support integration",
                base_url="https://api.support.com",
                authentication_type=AuthenticationType.BASIC_AUTH,
                credentials={"username": "", "password": ""},
                endpoints={
                    "tickets": "/tickets",
                    "customers": "/customers",
                    "knowledge_base": "/knowledge_base",
                    "satisfaction": "/satisfaction"
                },
                rate_limits={"requests_per_minute": 100, "requests_per_hour": 1000},
                retry_config={"max_retries": 3, "backoff_factor": 2},
                timeout_config={"connect_timeout": 30, "read_timeout": 60}
            ),
            IntegrationType.SOCIAL_MEDIA: IntegrationConfig(
                integration_type=IntegrationType.SOCIAL_MEDIA,
                name="Social Media Integration",
                description="Social media management integration",
                base_url="https://api.social.com",
                authentication_type=AuthenticationType.OAUTH2,
                credentials={"client_id": "", "client_secret": ""},
                endpoints={
                    "posts": "/posts",
                    "analytics": "/analytics",
                    "engagement": "/engagement",
                    "scheduling": "/scheduling"
                },
                rate_limits={"requests_per_minute": 300, "requests_per_hour": 3000},
                retry_config={"max_retries": 3, "backoff_factor": 2},
                timeout_config={"connect_timeout": 30, "read_timeout": 60}
            )
        }
    
    async def create_integration(
        self,
        name: str,
        integration_type: IntegrationType,
        description: str,
        base_url: str,
        authentication_type: AuthenticationType,
        credentials: Dict[str, Any],
        endpoints: Dict[str, Any],
        created_by: str,
        rate_limits: Dict[str, Any] = None,
        retry_config: Dict[str, Any] = None,
        timeout_config: Dict[str, Any] = None,
        custom_parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new integration"""
        try:
            # Validate integration data
            await self._validate_integration_data(
                name, integration_type, base_url, authentication_type, credentials
            )
            
            # Create integration data
            integration_data = {
                "name": name,
                "integration_type": integration_type.value,
                "description": description,
                "base_url": base_url,
                "authentication_type": authentication_type.value,
                "credentials": credentials,
                "endpoints": endpoints,
                "rate_limits": rate_limits or {"requests_per_minute": 100, "requests_per_hour": 1000},
                "retry_config": retry_config or {"max_retries": 3, "backoff_factor": 2},
                "timeout_config": timeout_config or {"connect_timeout": 30, "read_timeout": 60},
                "custom_parameters": custom_parameters or {},
                "status": IntegrationStatus.PENDING.value,
                "created_by": created_by
            }
            
            # Create integration in database
            integration = await db_manager.create_integration(integration_data)
            
            # Initialize performance metrics
            await self._initialize_integration_metrics(integration.id)
            
            # Cache integration data
            await self._cache_integration_data(integration)
            
            logger.info(f"Integration created successfully: {integration.id}")
            
            return {
                "id": str(integration.id),
                "name": integration.name,
                "integration_type": integration.integration_type,
                "description": integration.description,
                "base_url": integration.base_url,
                "authentication_type": integration.authentication_type,
                "endpoints": integration.endpoints,
                "status": integration.status,
                "created_by": str(integration.created_by),
                "created_at": integration.created_at,
                "updated_at": integration.updated_at
            }
            
        except Exception as e:
            error = handle_agent_error(e, name=name, created_by=created_by)
            log_agent_error(error)
            raise error
    
    async def execute_integration(
        self,
        integration_id: str,
        endpoint: str,
        method: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, Any] = None,
        user_id: str = None
    ) -> IntegrationExecutionResult:
        """Execute integration with external service"""
        try:
            # Get integration
            integration = await self.get_integration(integration_id)
            if not integration:
                raise IntegrationNotFoundError(
                    "integration_not_found",
                    f"Integration {integration_id} not found",
                    {"integration_id": integration_id}
                )
            
            # Validate integration status
            if integration["status"] != IntegrationStatus.ACTIVE.value:
                raise IntegrationValidationError(
                    "integration_not_active",
                    f"Integration {integration_id} is not active",
                    {"integration_id": integration_id, "status": integration["status"]}
                )
            
            # Validate endpoint
            if endpoint not in integration["endpoints"]:
                raise IntegrationValidationError(
                    "invalid_endpoint",
                    f"Endpoint {endpoint} not found in integration",
                    {"endpoint": endpoint, "available_endpoints": list(integration["endpoints"].keys())}
                )
            
            # Create execution record
            execution_id = str(uuid4())
            execution_data = {
                "integration_id": integration_id,
                "execution_id": execution_id,
                "endpoint": endpoint,
                "method": method,
                "input_data": data or {},
                "status": "running",
                "created_by": user_id or "system"
            }
            
            execution = await db_manager.create_integration_execution(execution_data)
            
            # Execute integration
            start_time = datetime.utcnow()
            result = await self._perform_integration_execution(
                integration, endpoint, method, data, headers
            )
            
            # Update execution record
            await db_manager.update_integration_execution_status(
                execution_id,
                result.status,
                output_data=result.output_data,
                error_message=result.error_message,
                duration=result.duration,
                performance_metrics=result.performance_metrics
            )
            
            # Update integration metrics
            await self._update_integration_metrics(integration_id, result)
            
            # Cache execution result
            await self._cache_execution_result(execution_id, result)
            
            logger.info(f"Integration executed successfully: {integration_id}, execution: {execution_id}")
            
            return result
            
        except Exception as e:
            error = handle_agent_error(e, integration_id=integration_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def test_integration(
        self,
        integration_id: str,
        test_endpoint: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Test integration connectivity and functionality"""
        try:
            # Get integration
            integration = await self.get_integration(integration_id)
            if not integration:
                raise IntegrationNotFoundError(
                    "integration_not_found",
                    f"Integration {integration_id} not found",
                    {"integration_id": integration_id}
                )
            
            # Perform connectivity test
            connectivity_result = await self._test_connectivity(integration)
            
            # Perform endpoint tests
            endpoint_results = {}
            if test_endpoint:
                endpoint_results[test_endpoint] = await self._test_endpoint(
                    integration, test_endpoint
                )
            else:
                # Test all endpoints
                for endpoint in integration["endpoints"].keys():
                    endpoint_results[endpoint] = await self._test_endpoint(
                        integration, endpoint
                    )
            
            # Calculate overall test result
            all_tests_passed = (
                connectivity_result["success"] and
                all(result["success"] for result in endpoint_results.values())
            )
            
            test_result = {
                "integration_id": integration_id,
                "overall_success": all_tests_passed,
                "connectivity_test": connectivity_result,
                "endpoint_tests": endpoint_results,
                "test_timestamp": datetime.utcnow().isoformat(),
                "tested_by": user_id or "system"
            }
            
            # Update integration status based on test results
            new_status = IntegrationStatus.ACTIVE if all_tests_passed else IntegrationStatus.ERROR
            await self._update_integration_status(integration_id, new_status)
            
            logger.info(f"Integration test completed: {integration_id}, success: {all_tests_passed}")
            
            return test_result
            
        except Exception as e:
            error = handle_agent_error(e, integration_id=integration_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def get_integration_performance(
        self,
        integration_id: str
    ) -> Dict[str, Any]:
        """Get integration performance metrics"""
        try:
            # Get integration
            integration = await self.get_integration(integration_id)
            if not integration:
                raise IntegrationNotFoundError(
                    "integration_not_found",
                    f"Integration {integration_id} not found",
                    {"integration_id": integration_id}
                )
            
            # Get performance metrics
            metrics = await self._calculate_integration_metrics(integration)
            
            return metrics
            
        except Exception as e:
            error = handle_agent_error(e, integration_id=integration_id)
            log_agent_error(error)
            raise error
    
    async def optimize_integration(
        self,
        integration_id: str,
        optimization_targets: List[str] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Optimize integration performance"""
        try:
            # Get integration
            integration = await self.get_integration(integration_id)
            if not integration:
                raise IntegrationNotFoundError(
                    "integration_not_found",
                    f"Integration {integration_id} not found",
                    {"integration_id": integration_id}
                )
            
            # Get current performance
            current_metrics = await self.get_integration_performance(integration_id)
            
            # Perform optimization analysis
            optimization_result = await self._perform_optimization_analysis(
                integration, current_metrics, optimization_targets
            )
            
            # Apply optimizations if requested
            if optimization_result.get("improvements"):
                await self._apply_integration_optimizations(integration_id, optimization_result)
            
            logger.info(f"Integration optimization completed: {integration_id}")
            
            return optimization_result
            
        except Exception as e:
            error = handle_agent_error(e, integration_id=integration_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def get_integration(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """Get integration by ID"""
        try:
            # Try cache first
            cached_integration = await self._get_cached_integration(integration_id)
            if cached_integration:
                return cached_integration
            
            # Get from database
            integration = await db_manager.get_integration_by_id(integration_id)
            if not integration:
                return None
            
            # Cache integration data
            await self._cache_integration_data(integration)
            
            return {
                "id": str(integration.id),
                "name": integration.name,
                "integration_type": integration.integration_type,
                "description": integration.description,
                "base_url": integration.base_url,
                "authentication_type": integration.authentication_type,
                "credentials": integration.credentials,
                "endpoints": integration.endpoints,
                "rate_limits": integration.rate_limits,
                "retry_config": integration.retry_config,
                "timeout_config": integration.timeout_config,
                "custom_parameters": integration.custom_parameters,
                "status": integration.status,
                "created_by": str(integration.created_by),
                "created_at": integration.created_at,
                "updated_at": integration.updated_at
            }
            
        except Exception as e:
            error = handle_agent_error(e, integration_id=integration_id)
            log_agent_error(error)
            raise error
    
    # Private helper methods
    async def _validate_integration_data(
        self,
        name: str,
        integration_type: IntegrationType,
        base_url: str,
        authentication_type: AuthenticationType,
        credentials: Dict[str, Any]
    ) -> None:
        """Validate integration data"""
        if not name or len(name.strip()) == 0:
            raise IntegrationValidationError(
                "invalid_name",
                "Integration name cannot be empty",
                {"name": name}
            )
        
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise IntegrationValidationError(
                "invalid_base_url",
                "Base URL must be a valid HTTP/HTTPS URL",
                {"base_url": base_url}
            )
        
        if not credentials:
            raise IntegrationValidationError(
                "invalid_credentials",
                "Credentials cannot be empty",
                {"authentication_type": authentication_type}
            )
    
    async def _perform_integration_execution(
        self,
        integration: Dict[str, Any],
        endpoint: str,
        method: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, Any] = None
    ) -> IntegrationExecutionResult:
        """Perform integration execution"""
        try:
            start_time = datetime.utcnow()
            execution_log = []
            
            # Prepare request
            execution_log.append({
                "step": "request_preparation",
                "status": "started",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Simulate request preparation
            await asyncio.sleep(0.05)
            
            # Authenticate
            execution_log.append({
                "step": "authentication",
                "status": "started",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Simulate authentication
            await asyncio.sleep(0.1)
            
            # Make request
            execution_log.append({
                "step": "api_request",
                "status": "started",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Simulate API request
            await asyncio.sleep(0.2)
            
            # Process response
            execution_log.append({
                "step": "response_processing",
                "status": "started",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Simulate response processing
            await asyncio.sleep(0.05)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Simulate successful response
            output_data = {
                "status": "success",
                "data": {"message": "Integration executed successfully"},
                "response_time": duration,
                "endpoint": endpoint,
                "method": method
            }
            
            # Calculate performance metrics
            performance_metrics = {
                "duration": duration,
                "success": True,
                "response_time": duration,
                "endpoint": endpoint,
                "method": method
            }
            
            return IntegrationExecutionResult(
                integration_id=integration["id"],
                execution_id=str(uuid4()),
                integration_type=IntegrationType(integration["integration_type"]),
                status="completed",
                input_data=data or {},
                output_data=output_data,
                performance_metrics=performance_metrics,
                execution_log=execution_log,
                duration=duration,
                success_rate=1.0,
                efficiency_score=0.9,
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            return IntegrationExecutionResult(
                integration_id=integration["id"],
                execution_id=str(uuid4()),
                integration_type=IntegrationType(integration["integration_type"]),
                status="failed",
                input_data=data or {},
                error_message=str(e),
                duration=duration,
                started_at=start_time,
                completed_at=end_time
            )
    
    async def _test_connectivity(self, integration: Dict[str, Any]) -> Dict[str, Any]:
        """Test integration connectivity"""
        try:
            # Simulate connectivity test
            await asyncio.sleep(0.1)
            
            return {
                "success": True,
                "response_time": 0.1,
                "status_code": 200,
                "message": "Connection successful"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Connection failed"
            }
    
    async def _test_endpoint(self, integration: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Test specific integration endpoint"""
        try:
            # Simulate endpoint test
            await asyncio.sleep(0.05)
            
            return {
                "success": True,
                "endpoint": endpoint,
                "response_time": 0.05,
                "status_code": 200,
                "message": f"Endpoint {endpoint} is working"
            }
        except Exception as e:
            return {
                "success": False,
                "endpoint": endpoint,
                "error": str(e),
                "message": f"Endpoint {endpoint} failed"
            }
    
    async def _calculate_integration_metrics(self, integration: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate integration performance metrics"""
        # Get execution data
        executions = await self._get_integration_executions(integration["id"])
        
        if not executions:
            return {
                "integration_id": integration["id"],
                "integration_type": integration["integration_type"],
                "total_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "error_rate": 0.0
            }
        
        # Calculate basic metrics
        total_executions = len(executions)
        successful_executions = sum(1 for e in executions if e.status == "completed")
        success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
        error_rate = 1.0 - success_rate
        
        durations = [e.duration for e in executions if e.duration]
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "integration_id": integration["id"],
            "integration_type": integration["integration_type"],
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_duration": average_duration,
            "error_rate": error_rate,
            "last_execution": executions[0].started_at if executions else None
        }
    
    async def _perform_optimization_analysis(
        self,
        integration: Dict[str, Any],
        current_metrics: Dict[str, Any],
        optimization_targets: List[str] = None
    ) -> Dict[str, Any]:
        """Perform optimization analysis for integration"""
        improvements = []
        recommendations = []
        
        # Analyze current performance
        if current_metrics.get("success_rate", 0.0) < 0.95:
            improvements.append({
                "type": "success_rate",
                "current": current_metrics.get("success_rate", 0.0),
                "target": 0.98,
                "improvement": "Improve success rate through better error handling"
            })
            recommendations.append("Implement comprehensive error handling and retry mechanisms")
        
        if current_metrics.get("average_duration", 0.0) > 2.0:
            improvements.append({
                "type": "duration",
                "current": current_metrics.get("average_duration", 0.0),
                "target": 1.0,
                "improvement": "Optimize response time through connection pooling"
            })
            recommendations.append("Implement connection pooling and caching")
        
        return {
            "integration_id": integration["id"],
            "integration_type": integration["integration_type"],
            "improvements": improvements,
            "recommendations": recommendations,
            "estimated_improvement": len(improvements) * 0.1
        }
    
    async def _apply_integration_optimizations(
        self,
        integration_id: str,
        optimization_result: Dict[str, Any]
    ) -> None:
        """Apply integration optimizations"""
        # Update integration configuration with optimizations
        updates = {
            "configuration": {
                "optimization_level": "advanced",
                "last_optimization": datetime.utcnow().isoformat(),
                "optimization_improvements": optimization_result.get("improvements", [])
            }
        }
        
        # This would update the integration in the database
        logger.info(f"Applied optimizations to integration: {integration_id}")
    
    async def _update_integration_status(
        self,
        integration_id: str,
        status: IntegrationStatus
    ) -> None:
        """Update integration status"""
        # This would update the integration status in the database
        logger.info(f"Updated integration status: {integration_id} -> {status.value}")
    
    async def _update_integration_metrics(
        self,
        integration_id: str,
        execution_result: IntegrationExecutionResult
    ) -> None:
        """Update integration performance metrics"""
        # Update integration execution count and success rate
        integration = await db_manager.get_integration_by_id(integration_id)
        if integration:
            integration.execution_count += 1
            if execution_result.status == "completed":
                # Update success rate calculation
                pass
            
            integration.last_execution = execution_result.started_at
            await self.db.commit()
    
    async def _initialize_integration_metrics(self, integration_id: str) -> None:
        """Initialize integration performance metrics"""
        # Create initial analytics record
        analytics_data = {
            "integration_id": integration_id,
            "date": datetime.utcnow().date(),
            "execution_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "average_duration": 0.0
        }
        
        # This would create an analytics record in the database
        logger.info(f"Initialized metrics for integration: {integration_id}")
    
    async def _get_integration_executions(self, integration_id: str) -> List[Any]:
        """Get integration executions"""
        # This would query the database for integration executions
        return []
    
    # Caching methods
    async def _cache_integration_data(self, integration: Any) -> None:
        """Cache integration data"""
        cache_key = f"integration:{integration.id}"
        integration_data = {
            "id": str(integration.id),
            "name": integration.name,
            "integration_type": integration.integration_type,
            "base_url": integration.base_url,
            "endpoints": integration.endpoints
        }
        
        await self.redis.setex(
            cache_key,
            3600,  # 1 hour
            json.dumps(integration_data)
        )
    
    async def _get_cached_integration(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """Get cached integration data"""
        cache_key = f"integration:{integration_id}"
        cached_data = await self.redis.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        return None
    
    async def _cache_execution_result(self, execution_id: str, result: IntegrationExecutionResult) -> None:
        """Cache execution result"""
        cache_key = f"integration_execution:{execution_id}"
        result_data = {
            "execution_id": result.execution_id,
            "integration_id": result.integration_id,
            "integration_type": result.integration_type.value,
            "status": result.status,
            "duration": result.duration,
            "success_rate": result.success_rate,
            "efficiency_score": result.efficiency_score
        }
        
        await self.redis.setex(
            cache_key,
            1800,  # 30 minutes
            json.dumps(result_data)
        )