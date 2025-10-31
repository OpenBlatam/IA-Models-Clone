"""
Advanced Business Agents Schemas
==============================

Pydantic v2 models for business agents system with comprehensive validation.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator, root_validator
import json


class AgentType(str, Enum):
    """Agent types"""
    SALES = "sales"
    MARKETING = "marketing"
    SUPPORT = "support"
    ANALYTICS = "analytics"
    CONTENT = "content"
    SOCIAL_MEDIA = "social_media"
    CUSTOMER_SUCCESS = "customer_success"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    AUTOMATION = "automation"
    INTEGRATION = "integration"


class AgentStatus(str, Enum):
    """Agent status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CommunicationChannel(str, Enum):
    """Communication channels"""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    API = "api"
    SMS = "sms"
    PHONE = "phone"


class AgentCapability(str, Enum):
    """Agent capabilities"""
    CONTENT_ANALYSIS = "content_analysis"
    CUSTOMER_INTERACTION = "customer_interaction"
    DATA_ANALYSIS = "data_analysis"
    AUTOMATION = "automation"
    REPORTING = "reporting"
    INTEGRATION = "integration"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    ALERTING = "alerting"


class AgentConfiguration(BaseModel):
    """Agent configuration"""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    agent_type: AgentType = Field(..., description="Type of agent")
    capabilities: List[AgentCapability] = Field(..., description="Agent capabilities")
    ai_model: str = Field(default="gpt-4", description="AI model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="AI temperature")
    max_tokens: int = Field(default=2000, ge=1, le=8000, description="Max tokens")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Retry attempts")
    rate_limit: int = Field(default=100, ge=1, le=10000, description="Requests per hour")
    enabled_features: List[str] = Field(default_factory=list, description="Enabled features")
    custom_prompts: Dict[str, str] = Field(default_factory=dict, description="Custom prompts")
    integration_configs: Dict[str, Any] = Field(default_factory=dict, description="Integration configs")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v < 1 or v > 8000:
            raise ValueError('Max tokens must be between 1 and 8000')
        return v


class AgentMetrics(BaseModel):
    """Agent performance metrics"""
    agent_id: str = Field(..., description="Agent ID")
    total_tasks: int = Field(default=0, description="Total tasks processed")
    completed_tasks: int = Field(default=0, description="Completed tasks")
    failed_tasks: int = Field(default=0, description="Failed tasks")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    average_response_time: float = Field(default=0.0, ge=0.0, description="Average response time")
    total_uptime: float = Field(default=0.0, ge=0.0, description="Total uptime in hours")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity timestamp")
    error_count: int = Field(default=0, description="Error count")
    performance_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Performance score")
    
    @validator('success_rate')
    def validate_success_rate(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Success rate must be between 0.0 and 1.0')
        return v


class TaskDefinition(BaseModel):
    """Task definition"""
    task_id: str = Field(default_factory=lambda: str(uuid4()), description="Task ID")
    agent_id: str = Field(..., description="Agent ID")
    task_type: str = Field(..., description="Task type")
    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    expected_output: Optional[Dict[str, Any]] = Field(default=None, description="Expected output")
    deadline: Optional[datetime] = Field(default=None, description="Task deadline")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message")
    retry_count: int = Field(default=0, ge=0, le=10, description="Retry count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Task metadata")
    
    @validator('deadline')
    def validate_deadline(cls, v, values):
        if v and 'created_at' in values and v <= values['created_at']:
            raise ValueError('Deadline must be after creation time')
        return v


class AgentMessage(BaseModel):
    """Agent communication message"""
    message_id: str = Field(default_factory=lambda: str(uuid4()), description="Message ID")
    sender_id: str = Field(..., description="Sender agent ID")
    receiver_id: str = Field(..., description="Receiver agent ID")
    channel: CommunicationChannel = Field(..., description="Communication channel")
    message_type: str = Field(..., description="Message type")
    content: Dict[str, Any] = Field(..., description="Message content")
    priority: Priority = Field(default=Priority.MEDIUM, description="Message priority")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    status: str = Field(default="sent", description="Message status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")


class WorkflowDefinition(BaseModel):
    """Workflow definition"""
    workflow_id: str = Field(default_factory=lambda: str(uuid4()), description="Workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    trigger_conditions: List[Dict[str, Any]] = Field(..., description="Trigger conditions")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    agents_involved: List[str] = Field(..., description="Agents involved")
    priority: Priority = Field(default=Priority.MEDIUM, description="Workflow priority")
    timeout: int = Field(default=3600, ge=1, le=86400, description="Workflow timeout in seconds")
    retry_policy: Dict[str, Any] = Field(default_factory=dict, description="Retry policy")
    success_criteria: List[Dict[str, Any]] = Field(..., description="Success criteria")
    failure_handling: Dict[str, Any] = Field(default_factory=dict, description="Failure handling")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
    is_active: bool = Field(default=True, description="Is workflow active")


class AgentRequest(BaseModel):
    """Agent request"""
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Request ID")
    agent_type: AgentType = Field(..., description="Requested agent type")
    task_description: str = Field(..., description="Task description")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    priority: Priority = Field(default=Priority.MEDIUM, description="Request priority")
    deadline: Optional[datetime] = Field(default=None, description="Request deadline")
    expected_output_format: Optional[str] = Field(default=None, description="Expected output format")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class AgentResponse(BaseModel):
    """Agent response"""
    response_id: str = Field(default_factory=lambda: str(uuid4()), description="Response ID")
    request_id: str = Field(..., description="Original request ID")
    agent_id: str = Field(..., description="Agent ID")
    status: TaskStatus = Field(..., description="Response status")
    output_data: Dict[str, Any] = Field(..., description="Output data")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    processing_time: float = Field(default=0.0, ge=0.0, description="Processing time in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v


class AgentHealthCheck(BaseModel):
    """Agent health check"""
    agent_id: str = Field(..., description="Agent ID")
    status: AgentStatus = Field(..., description="Agent status")
    health_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Health score")
    last_heartbeat: datetime = Field(..., description="Last heartbeat")
    response_time: float = Field(default=0.0, ge=0.0, description="Response time")
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Error rate")
    memory_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="Memory usage percentage")
    cpu_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="CPU usage percentage")
    active_tasks: int = Field(default=0, ge=0, description="Active tasks count")
    queue_size: int = Field(default=0, ge=0, description="Queue size")
    issues: List[str] = Field(default_factory=list, description="Current issues")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class AgentAnalytics(BaseModel):
    """Agent analytics"""
    agent_id: str = Field(..., description="Agent ID")
    time_period: str = Field(..., description="Time period")
    total_requests: int = Field(default=0, description="Total requests")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    average_response_time: float = Field(default=0.0, description="Average response time")
    peak_usage_time: Optional[datetime] = Field(default=None, description="Peak usage time")
    most_common_tasks: List[Dict[str, Any]] = Field(default_factory=list, description="Most common tasks")
    performance_trends: List[Dict[str, Any]] = Field(default_factory=list, description="Performance trends")
    user_satisfaction: float = Field(default=0.0, ge=0.0, le=5.0, description="User satisfaction score")
    cost_analysis: Dict[str, Any] = Field(default_factory=dict, description="Cost analysis")
    recommendations: List[str] = Field(default_factory=list, description="Analytics recommendations")


class AgentIntegration(BaseModel):
    """Agent integration configuration"""
    integration_id: str = Field(default_factory=lambda: str(uuid4()), description="Integration ID")
    agent_id: str = Field(..., description="Agent ID")
    service_name: str = Field(..., description="Service name")
    integration_type: str = Field(..., description="Integration type")
    configuration: Dict[str, Any] = Field(..., description="Integration configuration")
    authentication: Dict[str, Any] = Field(..., description="Authentication details")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL")
    api_endpoints: List[str] = Field(default_factory=list, description="API endpoints")
    rate_limits: Dict[str, int] = Field(default_factory=dict, description="Rate limits")
    is_active: bool = Field(default=True, description="Is integration active")
    last_sync: Optional[datetime] = Field(default=None, description="Last sync timestamp")
    sync_frequency: int = Field(default=3600, ge=60, le=86400, description="Sync frequency in seconds")
    error_count: int = Field(default=0, ge=0, description="Error count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Integration metadata")


class AgentTeam(BaseModel):
    """Agent team configuration"""
    team_id: str = Field(default_factory=lambda: str(uuid4()), description="Team ID")
    name: str = Field(..., description="Team name")
    description: str = Field(..., description="Team description")
    agent_ids: List[str] = Field(..., description="Agent IDs in team")
    team_lead: Optional[str] = Field(default=None, description="Team lead agent ID")
    collaboration_rules: Dict[str, Any] = Field(default_factory=dict, description="Collaboration rules")
    shared_resources: List[str] = Field(default_factory=list, description="Shared resources")
    communication_channels: List[CommunicationChannel] = Field(default_factory=list, description="Communication channels")
    performance_targets: Dict[str, float] = Field(default_factory=dict, description="Performance targets")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
    is_active: bool = Field(default=True, description="Is team active")


class AgentLearning(BaseModel):
    """Agent learning configuration"""
    learning_id: str = Field(default_factory=lambda: str(uuid4()), description="Learning ID")
    agent_id: str = Field(..., description="Agent ID")
    learning_type: str = Field(..., description="Learning type")
    training_data: List[Dict[str, Any]] = Field(..., description="Training data")
    model_version: str = Field(..., description="Model version")
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Accuracy score")
    training_status: str = Field(default="pending", description="Training status")
    training_started_at: Optional[datetime] = Field(default=None, description="Training start time")
    training_completed_at: Optional[datetime] = Field(default=None, description="Training completion time")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results")
    deployment_status: str = Field(default="pending", description="Deployment status")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    feedback_data: List[Dict[str, Any]] = Field(default_factory=list, description="Feedback data")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class AgentAuditLog(BaseModel):
    """Agent audit log entry"""
    log_id: str = Field(default_factory=lambda: str(uuid4()), description="Log ID")
    agent_id: str = Field(..., description="Agent ID")
    action: str = Field(..., description="Action performed")
    details: Dict[str, Any] = Field(..., description="Action details")
    user_id: Optional[str] = Field(default=None, description="User ID")
    ip_address: Optional[str] = Field(default=None, description="IP address")
    user_agent: Optional[str] = Field(default=None, description="User agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    severity: str = Field(default="info", description="Log severity")
    category: str = Field(default="general", description="Log category")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentSystemStatus(BaseModel):
    """Overall agent system status"""
    total_agents: int = Field(default=0, description="Total agents")
    active_agents: int = Field(default=0, description="Active agents")
    inactive_agents: int = Field(default=0, description="Inactive agents")
    agents_in_error: int = Field(default=0, description="Agents in error state")
    total_tasks: int = Field(default=0, description="Total tasks")
    pending_tasks: int = Field(default=0, description="Pending tasks")
    completed_tasks: int = Field(default=0, description="Completed tasks")
    failed_tasks: int = Field(default=0, description="Failed tasks")
    system_health_score: float = Field(default=0.0, ge=0.0, le=100.0, description="System health score")
    average_response_time: float = Field(default=0.0, ge=0.0, description="Average response time")
    total_uptime: float = Field(default=0.0, ge=0.0, description="Total uptime in hours")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    alerts: List[str] = Field(default_factory=list, description="Active alerts")
    recommendations: List[str] = Field(default_factory=list, description="System recommendations")


# Request/Response models for API endpoints
class CreateAgentRequest(BaseModel):
    """Create agent request"""
    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    description: str = Field(..., min_length=1, max_length=500, description="Agent description")
    agent_type: AgentType = Field(..., description="Agent type")
    capabilities: List[AgentCapability] = Field(..., min_items=1, description="Agent capabilities")
    configuration: Optional[AgentConfiguration] = Field(default=None, description="Agent configuration")
    team_id: Optional[str] = Field(default=None, description="Team ID")


class UpdateAgentRequest(BaseModel):
    """Update agent request"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100, description="Agent name")
    description: Optional[str] = Field(default=None, min_length=1, max_length=500, description="Agent description")
    status: Optional[AgentStatus] = Field(default=None, description="Agent status")
    configuration: Optional[AgentConfiguration] = Field(default=None, description="Agent configuration")
    capabilities: Optional[List[AgentCapability]] = Field(default=None, min_items=1, description="Agent capabilities")


class AgentTaskRequest(BaseModel):
    """Agent task request"""
    agent_id: str = Field(..., description="Agent ID")
    task_type: str = Field(..., description="Task type")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority")
    deadline: Optional[datetime] = Field(default=None, description="Task deadline")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")


class AgentResponseModel(BaseModel):
    """Standard agent response"""
    success: bool = Field(..., description="Success status")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    message: str = Field(..., description="Response message")
    agent_id: str = Field(..., description="Agent ID")
    processing_time: float = Field(default=0.0, description="Processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    agents: Dict[str, AgentHealthCheck] = Field(..., description="Agent health status")
    system: AgentSystemStatus = Field(..., description="System status")
    uptime: float = Field(..., description="System uptime")


class AnalyticsResponse(BaseModel):
    """Analytics response"""
    agent_id: str = Field(..., description="Agent ID")
    time_period: str = Field(..., description="Time period")
    analytics: AgentAnalytics = Field(..., description="Analytics data")
    insights: List[str] = Field(default_factory=list, description="Insights")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")


# Error response models
class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID")


# Configuration models
class AgentSystemConfig(BaseModel):
    """Agent system configuration"""
    max_concurrent_agents: int = Field(default=100, ge=1, le=1000, description="Max concurrent agents")
    default_timeout: int = Field(default=30, ge=1, le=300, description="Default timeout")
    max_retry_attempts: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    health_check_interval: int = Field(default=60, ge=10, le=3600, description="Health check interval")
    metrics_retention_days: int = Field(default=30, ge=1, le=365, description="Metrics retention days")
    auto_scaling_enabled: bool = Field(default=True, description="Auto scaling enabled")
    monitoring_enabled: bool = Field(default=True, description="Monitoring enabled")
    alerting_enabled: bool = Field(default=True, description="Alerting enabled")
    backup_enabled: bool = Field(default=True, description="Backup enabled")
    security_enabled: bool = Field(default=True, description="Security enabled")


def get_settings() -> AgentSystemConfig:
    """Get agent system settings"""
    return AgentSystemConfig()





























