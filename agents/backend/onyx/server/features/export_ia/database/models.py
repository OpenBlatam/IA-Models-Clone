"""
Database models for Export IA.
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any

Base = declarative_base()


class ExportTask(Base):
    """Database model for export tasks."""
    __tablename__ = "export_tasks"
    
    id = Column(String, primary_key=True)
    content = Column(JSON, nullable=False)
    format = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    quality_level = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    file_path = Column(String)
    file_size = Column(Integer)
    progress = Column(Float, default=0.0)
    quality_score = Column(Float)
    error_message = Column(Text)
    metadata = Column(JSON)
    
    # Relationships
    results = relationship("ExportResult", back_populates="task")


class ExportResult(Base):
    """Database model for export results."""
    __tablename__ = "export_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("export_tasks.id"), nullable=False)
    success = Column(Boolean, nullable=False)
    file_path = Column(String)
    file_size = Column(Integer)
    format = Column(String)
    quality_score = Column(Float)
    processing_time = Column(Float)
    error_message = Column(Text)
    warnings = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    task = relationship("ExportTask", back_populates="results")


class ServiceInstance(Base):
    """Database model for service instances."""
    __tablename__ = "service_instances"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    host = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default="stopped")
    health_url = Column(String)
    api_url = Column(String)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    dependencies = Column(JSON)


class SystemMetrics(Base):
    """Database model for system metrics."""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String)
    tags = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    service_name = Column(String)


class UserSession(Base):
    """Database model for user sessions."""
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    session_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class AuditLog(Base):
    """Database model for audit logs."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String)
    details = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)




