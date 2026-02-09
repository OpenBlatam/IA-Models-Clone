"""
Database models for advanced features
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()


class Integration(Base):
    """Integration configuration model."""
    __tablename__ = "integrations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    service_type = Column(String(50), nullable=False)
    api_key = Column(Text, nullable=True)
    webhook_url = Column(Text, nullable=True)
    webhook_secret = Column(Text, nullable=True)
    events = Column(JSON, nullable=True)
    configuration = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WebhookEvent(Base):
    """Webhook event model."""
    __tablename__ = "webhook_events"
    
    id = Column(Integer, primary_key=True, index=True)
    integration_name = Column(String(100), nullable=False, index=True)
    event_type = Column(String(100), nullable=False)
    payload = Column(JSON, nullable=False)
    signature = Column(Text, nullable=True)
    processed = Column(Boolean, default=False)
    received_at = Column(DateTime, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime, nullable=True)


class APICall(Base):
    """API call log model."""
    __tablename__ = "api_calls"
    
    id = Column(Integer, primary_key=True, index=True)
    integration_name = Column(String(100), nullable=False, index=True)
    endpoint = Column(Text, nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_data = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class CacheEntry(Base):
    """Cache entry model."""
    __tablename__ = "cache_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CacheStats(Base):
    """Cache statistics model."""
    __tablename__ = "cache_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_type = Column(String(50), nullable=False)
    hits = Column(Integer, default=0)
    misses = Column(Integer, default=0)
    sets = Column(Integer, default=0)
    deletes = Column(Integer, default=0)
    expires = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class SecurityEvent(Base):
    """Security event model."""
    __tablename__ = "security_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    details = Column(JSON, nullable=True)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class LoginAttempt(Base):
    """Login attempt model."""
    __tablename__ = "login_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), nullable=False, index=True)
    success = Column(Boolean, nullable=False)
    reason = Column(String(100), nullable=True)
    user_agent = Column(Text, nullable=True)
    attempted_at = Column(DateTime, default=datetime.utcnow, index=True)


class SystemMetric(Base):
    """System metric model."""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)
    tags = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class Alert(Base):
    """Alert model."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    threshold_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class MetricThreshold(Base):
    """Metric threshold model."""
    __tablename__ = "metric_thresholds"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), unique=True, index=True, nullable=False)
    warning_threshold = Column(Float, nullable=False)
    critical_threshold = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WorkflowExecution(Base):
    """Workflow execution model."""
    __tablename__ = "workflow_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    workflow_name = Column(String(100), nullable=False, index=True)
    execution_id = Column(String(100), unique=True, index=True, nullable=False)
    status = Column(String(20), nullable=False, index=True)
    context = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)


class WorkflowStep(Base):
    """Workflow step model."""
    __tablename__ = "workflow_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(String(100), nullable=False, index=True)
    step_name = Column(String(100), nullable=False)
    step_order = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)


class MLModel(Base):
    """Machine learning model model."""
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), unique=True, index=True, nullable=False)
    model_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, index=True)
    accuracy = Column(Float, nullable=True)
    training_data_size = Column(Integer, nullable=True)
    last_trained = Column(DateTime, nullable=True)
    model_path = Column(Text, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MLTrainingJob(Base):
    """ML training job model."""
    __tablename__ = "ml_training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    job_id = Column(String(100), unique=True, index=True, nullable=False)
    status = Column(String(20), nullable=False, index=True)
    training_data_size = Column(Integer, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)


class AnalyticsEvent(Base):
    """Analytics event model."""
    __tablename__ = "analytics_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    properties = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class ContentPerformance(Base):
    """Content performance model."""
    __tablename__ = "content_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String(100), nullable=False, index=True)
    content_type = Column(String(50), nullable=False)
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)
    avg_time_on_page = Column(Float, default=0.0)
    bounce_rate = Column(Float, default=0.0)
    date = Column(DateTime, default=datetime.utcnow, index=True)


class UserBehavior(Base):
    """User behavior model."""
    __tablename__ = "user_behavior"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)
    content_id = Column(String(100), nullable=True, index=True)
    properties = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class BlockchainTransaction(Base):
    """Blockchain transaction model."""
    __tablename__ = "blockchain_transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_hash = Column(String(66), unique=True, index=True, nullable=False)
    content_id = Column(String(100), nullable=True, index=True)
    transaction_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, index=True)
    gas_used = Column(Integer, nullable=True)
    gas_price = Column(Integer, nullable=True)
    block_number = Column(Integer, nullable=True)
    from_address = Column(String(42), nullable=True)
    to_address = Column(String(42), nullable=True)
    value = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class NFT(Base):
    """NFT model."""
    __tablename__ = "nfts"
    
    id = Column(Integer, primary_key=True, index=True)
    token_id = Column(String(100), unique=True, index=True, nullable=False)
    content_id = Column(String(100), nullable=False, index=True)
    owner_address = Column(String(42), nullable=False, index=True)
    contract_address = Column(String(42), nullable=False)
    metadata_uri = Column(Text, nullable=True)
    price = Column(String(20), nullable=True)
    is_listed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class IPFSContent(Base):
    """IPFS content model."""
    __tablename__ = "ipfs_content"
    
    id = Column(Integer, primary_key=True, index=True)
    ipfs_hash = Column(String(100), unique=True, index=True, nullable=False)
    content_id = Column(String(100), nullable=False, index=True)
    content_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=True)
    pin_status = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class QuantumOptimization(Base):
    """Quantum optimization model."""
    __tablename__ = "quantum_optimizations"
    
    id = Column(Integer, primary_key=True, index=True)
    optimization_id = Column(String(100), unique=True, index=True, nullable=False)
    optimization_type = Column(String(50), nullable=False)
    algorithm = Column(String(50), nullable=False)
    input_data = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    execution_time = Column(Float, nullable=True)
    qubits_used = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class QuantumCircuit(Base):
    """Quantum circuit model."""
    __tablename__ = "quantum_circuits"
    
    id = Column(Integer, primary_key=True, index=True)
    circuit_id = Column(String(100), unique=True, index=True, nullable=False)
    optimization_id = Column(String(100), nullable=False, index=True)
    circuit_data = Column(JSON, nullable=False)
    depth = Column(Integer, nullable=True)
    gate_count = Column(Integer, nullable=True)
    qubit_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# Create indexes for better performance
Index('idx_integrations_service_type', Integration.service_type)
Index('idx_webhook_events_processed', WebhookEvent.processed)
Index('idx_api_calls_integration_timestamp', APICall.integration_name, APICall.timestamp)
Index('idx_cache_entries_expires', CacheEntry.expires_at)
Index('idx_security_events_severity_created', SecurityEvent.severity, SecurityEvent.created_at)
Index('idx_login_attempts_ip_attempted', LoginAttempt.ip_address, LoginAttempt.attempted_at)
Index('idx_system_metrics_name_timestamp', SystemMetric.metric_name, SystemMetric.timestamp)
Index('idx_alerts_severity_resolved', Alert.severity, Alert.resolved)
Index('idx_workflow_executions_status_started', WorkflowExecution.status, WorkflowExecution.started_at)
Index('idx_workflow_steps_execution_order', WorkflowStep.execution_id, WorkflowStep.step_order)
Index('idx_ml_models_type_status', MLModel.model_type, MLModel.status)
Index('idx_ml_training_jobs_model_status', MLTrainingJob.model_name, MLTrainingJob.status)
Index('idx_analytics_events_type_timestamp', AnalyticsEvent.event_type, AnalyticsEvent.timestamp)
Index('idx_content_performance_content_date', ContentPerformance.content_id, ContentPerformance.date)
Index('idx_user_behavior_user_action', UserBehavior.user_id, UserBehavior.action)
Index('idx_blockchain_transactions_type_status', BlockchainTransaction.transaction_type, BlockchainTransaction.status)
Index('idx_nfts_content_owner', NFT.content_id, NFT.owner_address)
Index('idx_ipfs_content_content_type', IPFSContent.content_id, IPFSContent.content_type)
Index('idx_quantum_optimizations_type_status', QuantumOptimization.optimization_type, QuantumOptimization.status)
Index('idx_quantum_circuits_optimization', QuantumCircuit.optimization_id)