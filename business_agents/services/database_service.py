"""
Database Service
================

Advanced database service for managing data operations, caching,
and performance optimization.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Type, TypeVar
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, text, func, and_, or_
from sqlalchemy.exc import SQLAlchemyError
from contextlib import asynccontextmanager
import redis
import pickle
from functools import wraps
import hashlib

from ..models import (
    Base, User, Role, BusinessAgent, Workflow, Template, Document,
    WorkflowExecution, AgentExecution, Notification, Metric, Alert,
    Integration, MLPipeline, EnhancementRequest, db_manager
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Base)

class DatabaseService:
    """
    Advanced database service with caching and performance optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = db_manager.engine
        self.SessionLocal = db_manager.SessionLocal
        
        # Initialize Redis for caching
        self.redis_client = None
        if config.get("redis_url"):
            try:
                self.redis_client = redis.from_url(config["redis_url"])
                self.redis_client.ping()  # Test connection
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}")
                self.redis_client = None
        
        # Cache configuration
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour default
        self.cache_enabled = config.get("cache_enabled", True)
        
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
        
    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
            
    def cache_key(self, prefix: str, *args) -> str:
        """Generate cache key."""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        if not self.cache_enabled or not self.redis_client:
            return None
            
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            
        return None
        
    def set_cache(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Set data in cache."""
        if not self.cache_enabled or not self.redis_client:
            return False
            
        try:
            ttl = ttl or self.cache_ttl
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False
            
    def delete_cache(self, key: str) -> bool:
        """Delete data from cache."""
        if not self.cache_enabled or not self.redis_client:
            return False
            
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False
            
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern."""
        if not self.cache_enabled or not self.redis_client:
            return 0
            
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache pattern invalidation error: {str(e)}")
            
        return 0
        
    def cached_query(self, cache_key: str, ttl: Optional[int] = None):
        """Decorator for caching query results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Try to get from cache first
                cached_result = self.get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
                    
                # Execute query
                result = await func(*args, **kwargs)
                
                # Cache result
                self.set_cache(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
        
    # User operations
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user."""
        session = self.get_session()
        try:
            user = User(**user_data)
            session.add(user)
            session.commit()
            session.refresh(user)
            
            # Invalidate user cache
            self.invalidate_pattern("users:*")
            
            return user
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise
        finally:
            session.close()
            
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID with caching."""
        cache_key = self.cache_key("user", user_id)
        cached_user = self.get_from_cache(cache_key)
        if cached_user:
            return cached_user
            
        session = self.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                self.set_cache(cache_key, user)
            return user
        finally:
            session.close()
            
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username with caching."""
        cache_key = self.cache_key("user_username", username)
        cached_user = self.get_from_cache(cache_key)
        if cached_user:
            return cached_user
            
        session = self.get_session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if user:
                self.set_cache(cache_key, user)
            return user
        finally:
            session.close()
            
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user."""
        session = self.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return None
                
            for key, value in update_data.items():
                if hasattr(user, key):
                    setattr(user, key, value)
                    
            user.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(user)
            
            # Invalidate user cache
            self.invalidate_pattern(f"user:{user_id}")
            self.invalidate_pattern(f"user_username:{user.username}")
            
            return user
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating user: {str(e)}")
            raise
        finally:
            session.close()
            
    # Business Agent operations
    async def create_business_agent(self, agent_data: Dict[str, Any]) -> BusinessAgent:
        """Create a new business agent."""
        session = self.get_session()
        try:
            agent = BusinessAgent(**agent_data)
            session.add(agent)
            session.commit()
            session.refresh(agent)
            
            # Invalidate agent cache
            self.invalidate_pattern("agents:*")
            
            return agent
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating business agent: {str(e)}")
            raise
        finally:
            session.close()
            
    async def get_business_agents(self, business_area: Optional[str] = None) -> List[BusinessAgent]:
        """Get business agents with optional filtering."""
        cache_key = self.cache_key("agents", business_area or "all")
        cached_agents = self.get_from_cache(cache_key)
        if cached_agents:
            return cached_agents
            
        session = self.get_session()
        try:
            query = session.query(BusinessAgent).filter(BusinessAgent.is_active == True)
            if business_area:
                query = query.filter(BusinessAgent.business_area == business_area)
                
            agents = query.all()
            self.set_cache(cache_key, agents)
            return agents
        finally:
            session.close()
            
    async def get_business_agent_by_id(self, agent_id: str) -> Optional[BusinessAgent]:
        """Get business agent by ID."""
        cache_key = self.cache_key("agent", agent_id)
        cached_agent = self.get_from_cache(cache_key)
        if cached_agent:
            return cached_agent
            
        session = self.get_session()
        try:
            agent = session.query(BusinessAgent).filter(BusinessAgent.id == agent_id).first()
            if agent:
                self.set_cache(cache_key, agent)
            return agent
        finally:
            session.close()
            
    # Workflow operations
    async def create_workflow(self, workflow_data: Dict[str, Any]) -> Workflow:
        """Create a new workflow."""
        session = self.get_session()
        try:
            workflow = Workflow(**workflow_data)
            session.add(workflow)
            session.commit()
            session.refresh(workflow)
            
            # Invalidate workflow cache
            self.invalidate_pattern("workflows:*")
            
            return workflow
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating workflow: {str(e)}")
            raise
        finally:
            session.close()
            
    async def get_workflows(self, business_area: Optional[str] = None, status: Optional[str] = None) -> List[Workflow]:
        """Get workflows with optional filtering."""
        cache_key = self.cache_key("workflows", business_area or "all", status or "all")
        cached_workflows = self.get_from_cache(cache_key)
        if cached_workflows:
            return cached_workflows
            
        session = self.get_session()
        try:
            query = session.query(Workflow).filter(Workflow.is_active == True)
            if business_area:
                query = query.filter(Workflow.business_area == business_area)
            if status:
                query = query.filter(Workflow.status == status)
                
            workflows = query.all()
            self.set_cache(cache_key, workflows)
            return workflows
        finally:
            session.close()
            
    async def get_workflow_by_id(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        cache_key = self.cache_key("workflow", workflow_id)
        cached_workflow = self.get_from_cache(cache_key)
        if cached_workflow:
            return cached_workflow
            
        session = self.get_session()
        try:
            workflow = session.query(Workflow).filter(Workflow.id == workflow_id).first()
            if workflow:
                self.set_cache(cache_key, workflow)
            return workflow
        finally:
            session.close()
            
    async def update_workflow(self, workflow_id: str, update_data: Dict[str, Any]) -> Optional[Workflow]:
        """Update workflow."""
        session = self.get_session()
        try:
            workflow = session.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                return None
                
            for key, value in update_data.items():
                if hasattr(workflow, key):
                    setattr(workflow, key, value)
                    
            workflow.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(workflow)
            
            # Invalidate workflow cache
            self.invalidate_pattern(f"workflow:{workflow_id}")
            self.invalidate_pattern("workflows:*")
            
            return workflow
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating workflow: {str(e)}")
            raise
        finally:
            session.close()
            
    # Document operations
    async def create_document(self, document_data: Dict[str, Any]) -> Document:
        """Create a new document."""
        session = self.get_session()
        try:
            document = Document(**document_data)
            session.add(document)
            session.commit()
            session.refresh(document)
            
            # Invalidate document cache
            self.invalidate_pattern("documents:*")
            
            return document
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating document: {str(e)}")
            raise
        finally:
            session.close()
            
    async def get_documents(self, document_type: Optional[str] = None, business_area: Optional[str] = None) -> List[Document]:
        """Get documents with optional filtering."""
        cache_key = self.cache_key("documents", document_type or "all", business_area or "all")
        cached_documents = self.get_from_cache(cache_key)
        if cached_documents:
            return cached_documents
            
        session = self.get_session()
        try:
            query = session.query(Document)
            if document_type:
                query = query.filter(Document.document_type == document_type)
            if business_area:
                query = query.filter(Document.business_area == business_area)
                
            documents = query.order_by(Document.created_at.desc()).all()
            self.set_cache(cache_key, documents)
            return documents
        finally:
            session.close()
            
    # Workflow Execution operations
    async def create_workflow_execution(self, execution_data: Dict[str, Any]) -> WorkflowExecution:
        """Create a new workflow execution."""
        session = self.get_session()
        try:
            execution = WorkflowExecution(**execution_data)
            session.add(execution)
            session.commit()
            session.refresh(execution)
            
            # Invalidate execution cache
            self.invalidate_pattern("executions:*")
            
            return execution
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating workflow execution: {str(e)}")
            raise
        finally:
            session.close()
            
    async def update_workflow_execution(self, execution_id: str, update_data: Dict[str, Any]) -> Optional[WorkflowExecution]:
        """Update workflow execution."""
        session = self.get_session()
        try:
            execution = session.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
            if not execution:
                return None
                
            for key, value in update_data.items():
                if hasattr(execution, key):
                    setattr(execution, key, value)
                    
            session.commit()
            session.refresh(execution)
            
            # Invalidate execution cache
            self.invalidate_pattern(f"execution:{execution_id}")
            self.invalidate_pattern("executions:*")
            
            return execution
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating workflow execution: {str(e)}")
            raise
        finally:
            session.close()
            
    # Metric operations
    async def record_metric(self, metric_data: Dict[str, Any]) -> Metric:
        """Record a metric."""
        session = self.get_session()
        try:
            metric = Metric(**metric_data)
            session.add(metric)
            session.commit()
            session.refresh(metric)
            
            # Invalidate metric cache
            self.invalidate_pattern("metrics:*")
            
            return metric
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error recording metric: {str(e)}")
            raise
        finally:
            session.close()
            
    async def get_metrics(self, name: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Metric]:
        """Get metrics with optional filtering."""
        cache_key = self.cache_key("metrics", name or "all", str(start_time) if start_time else "all", str(end_time) if end_time else "all")
        cached_metrics = self.get_from_cache(cache_key)
        if cached_metrics:
            return cached_metrics
            
        session = self.get_session()
        try:
            query = session.query(Metric)
            if name:
                query = query.filter(Metric.name == name)
            if start_time:
                query = query.filter(Metric.timestamp >= start_time)
            if end_time:
                query = query.filter(Metric.timestamp <= end_time)
                
            metrics = query.order_by(Metric.timestamp.desc()).all()
            self.set_cache(cache_key, metrics, ttl=300)  # 5 minutes cache for metrics
            return metrics
        finally:
            session.close()
            
    # Analytics operations
    async def get_workflow_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get workflow analytics for the last N days."""
        cache_key = self.cache_key("analytics", "workflows", str(days))
        cached_analytics = self.get_from_cache(cache_key)
        if cached_analytics:
            return cached_analytics
            
        session = self.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Total workflows
            total_workflows = session.query(Workflow).filter(Workflow.created_at >= start_date).count()
            
            # Active workflows
            active_workflows = session.query(Workflow).filter(
                and_(Workflow.created_at >= start_date, Workflow.is_active == True)
            ).count()
            
            # Workflow executions
            total_executions = session.query(WorkflowExecution).filter(WorkflowExecution.started_at >= start_date).count()
            
            # Successful executions
            successful_executions = session.query(WorkflowExecution).filter(
                and_(WorkflowExecution.started_at >= start_date, WorkflowExecution.status == "completed")
            ).count()
            
            # Average execution time
            avg_duration = session.query(func.avg(WorkflowExecution.duration)).filter(
                and_(WorkflowExecution.started_at >= start_date, WorkflowExecution.duration.isnot(None))
            ).scalar() or 0
            
            # Business area distribution
            area_distribution = session.query(
                Workflow.business_area,
                func.count(Workflow.id)
            ).filter(Workflow.created_at >= start_date).group_by(Workflow.business_area).all()
            
            analytics = {
                "total_workflows": total_workflows,
                "active_workflows": active_workflows,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
                "average_execution_time": float(avg_duration),
                "business_area_distribution": dict(area_distribution),
                "period_days": days
            }
            
            self.set_cache(cache_key, analytics, ttl=600)  # 10 minutes cache
            return analytics
        finally:
            session.close()
            
    async def get_agent_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get agent analytics for the last N days."""
        cache_key = self.cache_key("analytics", "agents", str(days))
        cached_analytics = self.get_from_cache(cache_key)
        if cached_analytics:
            return cached_analytics
            
        session = self.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Total agents
            total_agents = session.query(BusinessAgent).filter(BusinessAgent.created_at >= start_date).count()
            
            # Active agents
            active_agents = session.query(BusinessAgent).filter(
                and_(BusinessAgent.created_at >= start_date, BusinessAgent.is_active == True)
            ).count()
            
            # Agent executions
            total_executions = session.query(AgentExecution).filter(AgentExecution.started_at >= start_date).count()
            
            # Successful executions
            successful_executions = session.query(AgentExecution).filter(
                and_(AgentExecution.started_at >= start_date, AgentExecution.status == "completed")
            ).count()
            
            # Average execution time
            avg_duration = session.query(func.avg(AgentExecution.duration)).filter(
                and_(AgentExecution.started_at >= start_date, AgentExecution.duration.isnot(None))
            ).scalar() or 0
            
            # Business area distribution
            area_distribution = session.query(
                BusinessAgent.business_area,
                func.count(BusinessAgent.id)
            ).filter(BusinessAgent.created_at >= start_date).group_by(BusinessAgent.business_area).all()
            
            analytics = {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
                "average_execution_time": float(avg_duration),
                "business_area_distribution": dict(area_distribution),
                "period_days": days
            }
            
            self.set_cache(cache_key, analytics, ttl=600)  # 10 minutes cache
            return analytics
        finally:
            session.close()
            
    # Search operations
    async def search_workflows(self, query: str, business_area: Optional[str] = None) -> List[Workflow]:
        """Search workflows by name or description."""
        cache_key = self.cache_key("search", "workflows", query, business_area or "all")
        cached_results = self.get_from_cache(cache_key)
        if cached_results:
            return cached_results
            
        session = self.get_session()
        try:
            search_query = session.query(Workflow).filter(
                and_(
                    Workflow.is_active == True,
                    or_(
                        Workflow.name.ilike(f"%{query}%"),
                        Workflow.description.ilike(f"%{query}%")
                    )
                )
            )
            
            if business_area:
                search_query = search_query.filter(Workflow.business_area == business_area)
                
            results = search_query.all()
            self.set_cache(cache_key, results, ttl=300)  # 5 minutes cache
            return results
        finally:
            session.close()
            
    async def search_documents(self, query: str, document_type: Optional[str] = None) -> List[Document]:
        """Search documents by title or content."""
        cache_key = self.cache_key("search", "documents", query, document_type or "all")
        cached_results = self.get_from_cache(cache_key)
        if cached_results:
            return cached_results
            
        session = self.get_session()
        try:
            search_query = session.query(Document).filter(
                or_(
                    Document.title.ilike(f"%{query}%"),
                    Document.content.ilike(f"%{query}%")
                )
            )
            
            if document_type:
                search_query = search_query.filter(Document.document_type == document_type)
                
            results = search_query.order_by(Document.created_at.desc()).all()
            self.set_cache(cache_key, results, ttl=300)  # 5 minutes cache
            return results
        finally:
            session.close()
            
    # Health check
    async def health_check(self) -> Dict[str, Any]:
        """Database health check."""
        try:
            session = self.get_session()
            # Test basic query
            result = session.execute(text("SELECT 1")).scalar()
            session.close()
            
            # Test Redis if available
            redis_status = "connected" if self.redis_client and self.redis_client.ping() else "disconnected"
            
            return {
                "status": "healthy",
                "database": "connected",
                "redis": redis_status,
                "cache_enabled": self.cache_enabled,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
    # Cleanup operations
    async def cleanup_old_data(self, days: int = 90):
        """Clean up old data."""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean up old metrics
            old_metrics = session.query(Metric).filter(Metric.timestamp < cutoff_date).count()
            session.query(Metric).filter(Metric.timestamp < cutoff_date).delete()
            
            # Clean up old notifications
            old_notifications = session.query(Notification).filter(
                and_(Notification.created_at < cutoff_date, Notification.is_read == True)
            ).count()
            session.query(Notification).filter(
                and_(Notification.created_at < cutoff_date, Notification.is_read == True)
            ).delete()
            
            session.commit()
            
            logger.info(f"Cleaned up {old_metrics} old metrics and {old_notifications} old notifications")
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error during cleanup: {str(e)}")
            raise
        finally:
            session.close()
            
    # Cache management
    async def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache."""
        if not self.redis_client:
            return
            
        try:
            if pattern:
                deleted_count = self.invalidate_pattern(pattern)
                logger.info(f"Cleared {deleted_count} cache entries matching pattern: {pattern}")
            else:
                self.redis_client.flushdb()
                logger.info("Cleared all cache entries")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis_client:
            return {"status": "redis_not_available"}
            
        try:
            info = self.redis_client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)) * 100
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}




























