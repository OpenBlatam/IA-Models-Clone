"""
Database Models for BUL System
Handles user management, document history, analytics, and system data
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import create_engine
from datetime import datetime
import uuid
import json

Base = declarative_base()

class User(Base):
    """User model for authentication and management"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_premium = Column(Boolean, default=False, nullable=False)
    subscription_tier = Column(String(50), default="free", nullable=False)
    api_key = Column(String(255), unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    preferences = Column(JSON, default=dict, nullable=False)
    usage_stats = Column(JSON, default=dict, nullable=False)
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    api_usage = relationship("APIUsage", back_populates="user", cascade="all, delete-orphan")
    workflows = relationship("WorkflowExecution", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, username={self.username})>"


class Document(Base):
    """Document model for storing generated documents"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    document_type = Column(String(100), nullable=False, index=True)
    template_id = Column(String(100), nullable=True, index=True)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, default=dict, nullable=False)
    status = Column(String(50), default="draft", nullable=False, index=True)
    version = Column(Integer, default=1, nullable=False)
    parent_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    published_at = Column(DateTime, nullable=True)
    tags = Column(JSON, default=list, nullable=False)
    file_size = Column(Integer, nullable=True)
    generation_time = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    model_used = Column(String(100), nullable=True)
    workflow_execution_id = Column(String(100), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="documents")
    parent_document = relationship("Document", remote_side=[id])
    versions = relationship("Document", back_populates="parent_document")
    shares = relationship("DocumentShare", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_user_type', 'user_id', 'document_type'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_status', 'status'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title}, type={self.document_type})>"


class DocumentShare(Base):
    """Document sharing model"""
    __tablename__ = "document_shares"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    shared_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    shared_with_email = Column(String(255), nullable=True)
    shared_with_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    permission = Column(String(50), default="read", nullable=False)  # read, write, admin
    share_token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    accessed_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="shares")
    shared_by_user = relationship("User", foreign_keys=[shared_by_user_id])
    shared_with_user = relationship("User", foreign_keys=[shared_with_user_id])
    
    def __repr__(self):
        return f"<DocumentShare(id={self.id}, document_id={self.document_id}, permission={self.permission})>"


class APIUsage(Base):
    """API usage tracking model"""
    __tablename__ = "api_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    endpoint = Column(String(255), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False, index=True)
    response_time = Column(Float, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    model_used = Column(String(100), nullable=True)
    request_size = Column(Integer, nullable=True)
    response_size = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="api_usage")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_usage_user_date', 'user_id', 'created_at'),
        Index('idx_api_usage_endpoint_date', 'endpoint', 'created_at'),
        Index('idx_api_usage_status_date', 'status_code', 'created_at'),
    )
    
    def __repr__(self):
        return f"<APIUsage(id={self.id}, user_id={self.user_id}, endpoint={self.endpoint})>"


class WorkflowExecution(Base):
    """Workflow execution tracking model"""
    __tablename__ = "workflow_executions"
    
    id = Column(String(100), primary_key=True)  # Using string ID from workflow engine
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    workflow_id = Column(String(100), nullable=False, index=True)
    workflow_name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, index=True)
    progress = Column(Float, default=0.0, nullable=False)
    context = Column(JSON, default=dict, nullable=False)
    results = Column(JSON, default=dict, nullable=False)
    errors = Column(JSON, default=list, nullable=False)
    started_at = Column(DateTime, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    current_step = Column(String(100), nullable=True)
    step_results = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="workflows")
    
    # Indexes
    __table_args__ = (
        Index('idx_workflow_executions_user_date', 'user_id', 'started_at'),
        Index('idx_workflow_executions_status', 'status'),
        Index('idx_workflow_executions_workflow', 'workflow_id'),
    )
    
    def __repr__(self):
        return f"<WorkflowExecution(id={self.id}, workflow_id={self.workflow_id}, status={self.status})>"


class TemplateUsage(Base):
    """Template usage tracking model"""
    __tablename__ = "template_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    template_id = Column(String(100), nullable=False, index=True)
    template_name = Column(String(255), nullable=False)
    document_type = Column(String(100), nullable=False, index=True)
    industry = Column(String(100), nullable=True, index=True)
    complexity = Column(String(50), nullable=False, index=True)
    fields_used = Column(JSON, default=dict, nullable=False)
    generation_time = Column(Float, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    model_used = Column(String(100), nullable=True)
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_template_usage_user_date', 'user_id', 'created_at'),
        Index('idx_template_usage_template_date', 'template_id', 'created_at'),
        Index('idx_template_usage_type_date', 'document_type', 'created_at'),
    )
    
    def __repr__(self):
        return f"<TemplateUsage(id={self.id}, template_id={self.template_id}, user_id={self.user_id})>"


class ModelUsage(Base):
    """Model usage tracking model"""
    __tablename__ = "model_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    model_id = Column(String(100), nullable=False, index=True)
    model_name = Column(String(255), nullable=False)
    provider = Column(String(100), nullable=False, index=True)
    request_type = Column(String(100), nullable=False, index=True)
    tokens_used = Column(Integer, nullable=False)
    cost = Column(Float, nullable=False)
    response_time = Column(Float, nullable=False)
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_model_usage_user_date', 'user_id', 'created_at'),
        Index('idx_model_usage_model_date', 'model_id', 'created_at'),
        Index('idx_model_usage_provider_date', 'provider', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ModelUsage(id={self.id}, model_id={self.model_id}, user_id={self.user_id})>"


class SystemMetrics(Base):
    """System metrics and analytics model"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # counter, gauge, histogram
    value = Column(Float, nullable=False)
    labels = Column(JSON, default=dict, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metadata = Column(JSON, default=dict, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_system_metrics_name_time', 'metric_name', 'timestamp'),
        Index('idx_system_metrics_type_time', 'metric_type', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, metric_name={self.metric_name}, value={self.value})>"


class AuditLog(Base):
    """Audit logging model for security and compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(100), nullable=True, index=True)
    details = Column(JSON, default=dict, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    success = Column(Boolean, default=True, nullable=False, index=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_logs_user_date', 'user_id', 'created_at'),
        Index('idx_audit_logs_action_date', 'action', 'created_at'),
        Index('idx_audit_logs_resource_date', 'resource_type', 'created_at'),
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, user_id={self.user_id})>"


class Notification(Base):
    """Notification model for user communications"""
    __tablename__ = "notifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    type = Column(String(50), nullable=False, index=True)  # info, warning, error, success
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSON, default=dict, nullable=False)
    is_read = Column(Boolean, default=False, nullable=False, index=True)
    is_important = Column(Boolean, default=False, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    read_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_notifications_user_read', 'user_id', 'is_read'),
        Index('idx_notifications_user_date', 'user_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Notification(id={self.id}, type={self.type}, user_id={self.user_id})>"


class Subscription(Base):
    """Subscription and billing model"""
    __tablename__ = "subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    plan_name = Column(String(100), nullable=False, index=True)
    plan_tier = Column(String(50), nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)  # active, cancelled, expired, trial
    billing_cycle = Column(String(20), nullable=False)  # monthly, yearly
    price = Column(Float, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    started_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    trial_ends_at = Column(DateTime, nullable=True)
    features = Column(JSON, default=dict, nullable=False)
    usage_limits = Column(JSON, default=dict, nullable=False)
    current_usage = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_subscriptions_user_status', 'user_id', 'status'),
        Index('idx_subscriptions_expires_at', 'expires_at'),
    )
    
    def __repr__(self):
        return f"<Subscription(id={self.id}, user_id={self.user_id}, plan={self.plan_name})>"


# Database configuration and session management
class DatabaseManager:
    """Database manager for BUL system"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()


# Example usage and initialization
def init_database(database_url: str = "sqlite:///./bul_database.db"):
    """Initialize database with tables"""
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    return db_manager


# Database session dependency for FastAPI
def get_db_session():
    """Get database session for dependency injection"""
    db = DatabaseManager("sqlite:///./bul_database.db")
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()


if __name__ == "__main__":
    # Initialize database for testing
    db_manager = init_database()
    print("Database initialized successfully!")
    print("Tables created:")
    for table in Base.metadata.tables:
        print(f"  - {table}")