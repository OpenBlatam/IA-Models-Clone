"""
Advanced Database Management API Endpoints
=========================================

Comprehensive API endpoints for database management, user operations, and data administration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
import redis

from ...schemas import (
    BusinessAgent, AgentRequest, AgentResponse, AgentAnalytics,
    AgentWorkflow, AgentCollaboration, AgentSettings,
    ErrorResponse, SuccessResponse
)
from ...exceptions import (
    DatabaseError, UserNotFoundError, AgentNotFoundError, WorkflowNotFoundError,
    create_agent_error, log_agent_error, handle_agent_error, get_error_response
)
from ...models import (
    db_manager, User, UserSession, BusinessAgent as AgentModel,
    AgentExecution, AgentAnalytics as AnalyticsModel, Workflow,
    WorkflowExecution, WorkflowAnalytics, AgentCollaboration,
    Notification, AuditLog, SystemMetrics, Configuration
)
from ...config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/database", tags=["Database Management"])


# Dependency injection
async def get_db_session() -> AsyncSession:
    """Get database session"""
    return await db_manager.get_session()


# User Management Endpoints
@router.post("/users", response_model=Dict[str, Any])
async def create_user(
    username: str = Query(..., description="Username"),
    email: str = Query(..., description="Email address"),
    password_hash: str = Query(..., description="Password hash"),
    first_name: str = Query(..., description="First name"),
    last_name: str = Query(..., description="Last name"),
    role: str = Query("user", description="User role"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Create new user
    
    - **username**: Username
    - **email**: Email address
    - **password_hash**: Password hash
    - **first_name**: First name
    - **last_name**: Last name
    - **role**: User role
    """
    try:
        # Create user data
        user_data = {
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "first_name": first_name,
            "last_name": last_name,
            "role": role,
            "is_active": True,
            "is_verified": False
        }
        
        # Create user
        user = await db_manager.create_user(user_data)
        
        return {
            "success": True,
            "message": "User created successfully",
            "data": {
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": user.role,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "created_at": user.created_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, username=username, email=email)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user(
    user_id: str = Path(..., description="User ID"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get user by ID
    
    - **user_id**: User ID
    """
    try:
        # Get user
        user = await db_manager.get_user_by_id(user_id)
        
        if not user:
            raise UserNotFoundError(
                "user_not_found",
                f"User {user_id} not found",
                {"user_id": user_id}
            )
        
        return {
            "success": True,
            "message": "User retrieved successfully",
            "data": {
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": user.role,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "created_at": user.created_at.isoformat(),
                "updated_at": user.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.get("/users", response_model=Dict[str, Any])
async def list_users(
    role: Optional[str] = Query(None, description="Filter by role"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(50, description="Number of users to return"),
    offset: int = Query(0, description="Number of users to skip"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    List users with filters
    
    - **role**: Filter by role
    - **is_active**: Filter by active status
    - **limit**: Number of users to return
    - **offset**: Number of users to skip
    """
    try:
        # Build query
        query = select(User)
        
        if role:
            query = query.where(User.role == role)
        
        if is_active is not None:
            query = query.where(User.is_active == is_active)
        
        # Execute query
        result = await db.execute(
            query.order_by(User.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        users = result.scalars().all()
        
        # Get total count
        count_query = select(func.count(User.id))
        if role:
            count_query = count_query.where(User.role == role)
        if is_active is not None:
            count_query = count_query.where(User.is_active == is_active)
        
        count_result = await db.execute(count_query)
        total_count = count_result.scalar()
        
        return {
            "success": True,
            "message": "Users retrieved successfully",
            "data": {
                "users": [
                    {
                        "user_id": str(user.id),
                        "username": user.username,
                        "email": user.email,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "role": user.role,
                        "is_active": user.is_active,
                        "is_verified": user.is_verified,
                        "last_login": user.last_login.isoformat() if user.last_login else None,
                        "created_at": user.created_at.isoformat()
                    }
                    for user in users
                ],
                "total_count": total_count,
                "limit": limit,
                "offset": offset
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# Agent Management Endpoints
@router.post("/agents", response_model=Dict[str, Any])
async def create_agent(
    name: str = Query(..., description="Agent name"),
    description: str = Query("", description="Agent description"),
    agent_type: str = Query(..., description="Agent type"),
    category: str = Query("", description="Agent category"),
    tags: List[str] = Query([], description="Agent tags"),
    configuration: Dict[str, Any] = Query({}, description="Agent configuration"),
    capabilities: List[str] = Query([], description="Agent capabilities"),
    created_by: str = Query(..., description="Creator user ID"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Create new business agent
    
    - **name**: Agent name
    - **description**: Agent description
    - **agent_type**: Agent type
    - **category**: Agent category
    - **tags**: Agent tags
    - **configuration**: Agent configuration
    - **capabilities**: Agent capabilities
    - **created_by**: Creator user ID
    """
    try:
        # Create agent data
        agent_data = {
            "name": name,
            "description": description,
            "agent_type": agent_type,
            "category": category,
            "tags": tags,
            "configuration": configuration,
            "capabilities": capabilities,
            "created_by": created_by
        }
        
        # Create agent
        agent = await db_manager.create_agent(agent_data)
        
        return {
            "success": True,
            "message": "Agent created successfully",
            "data": {
                "agent_id": str(agent.id),
                "name": agent.name,
                "description": agent.description,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "category": agent.category,
                "tags": agent.tags,
                "configuration": agent.configuration,
                "capabilities": agent.capabilities,
                "execution_count": agent.execution_count,
                "success_rate": agent.success_rate,
                "average_response_time": agent.average_response_time,
                "created_by": str(agent.created_by),
                "created_at": agent.created_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, name=name, created_by=created_by)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent(
    agent_id: str = Path(..., description="Agent ID"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get agent by ID
    
    - **agent_id**: Agent ID
    """
    try:
        # Get agent
        agent = await db_manager.get_agent_by_id(agent_id)
        
        if not agent:
            raise AgentNotFoundError(
                "agent_not_found",
                f"Agent {agent_id} not found",
                {"agent_id": agent_id}
            )
        
        return {
            "success": True,
            "message": "Agent retrieved successfully",
            "data": {
                "agent_id": str(agent.id),
                "name": agent.name,
                "description": agent.description,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "category": agent.category,
                "tags": agent.tags,
                "configuration": agent.configuration,
                "capabilities": agent.capabilities,
                "execution_count": agent.execution_count,
                "success_rate": agent.success_rate,
                "average_response_time": agent.average_response_time,
                "last_execution": agent.last_execution.isoformat() if agent.last_execution else None,
                "created_by": str(agent.created_by),
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, agent_id=agent_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.get("/agents", response_model=Dict[str, Any])
async def list_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    limit: int = Query(50, description="Number of agents to return"),
    offset: int = Query(0, description="Number of agents to skip"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    List agents with filters
    
    - **agent_type**: Filter by agent type
    - **status**: Filter by status
    - **category**: Filter by category
    - **created_by**: Filter by creator
    - **limit**: Number of agents to return
    - **offset**: Number of agents to skip
    """
    try:
        # Build query
        query = select(AgentModel)
        
        if agent_type:
            query = query.where(AgentModel.agent_type == agent_type)
        
        if status:
            query = query.where(AgentModel.status == status)
        
        if category:
            query = query.where(AgentModel.category == category)
        
        if created_by:
            query = query.where(AgentModel.created_by == created_by)
        
        # Execute query
        result = await db.execute(
            query.order_by(AgentModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        agents = result.scalars().all()
        
        # Get total count
        count_query = select(func.count(AgentModel.id))
        if agent_type:
            count_query = count_query.where(AgentModel.agent_type == agent_type)
        if status:
            count_query = count_query.where(AgentModel.status == status)
        if category:
            count_query = count_query.where(AgentModel.category == category)
        if created_by:
            count_query = count_query.where(AgentModel.created_by == created_by)
        
        count_result = await db.execute(count_query)
        total_count = count_result.scalar()
        
        return {
            "success": True,
            "message": "Agents retrieved successfully",
            "data": {
                "agents": [
                    {
                        "agent_id": str(agent.id),
                        "name": agent.name,
                        "description": agent.description,
                        "agent_type": agent.agent_type,
                        "status": agent.status,
                        "category": agent.category,
                        "tags": agent.tags,
                        "execution_count": agent.execution_count,
                        "success_rate": agent.success_rate,
                        "average_response_time": agent.average_response_time,
                        "created_by": str(agent.created_by),
                        "created_at": agent.created_at.isoformat()
                    }
                    for agent in agents
                ],
                "total_count": total_count,
                "limit": limit,
                "offset": offset
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# Workflow Management Endpoints
@router.post("/workflows", response_model=Dict[str, Any])
async def create_workflow(
    name: str = Query(..., description="Workflow name"),
    description: str = Query("", description="Workflow description"),
    workflow_type: str = Query(..., description="Workflow type"),
    category: str = Query("", description="Workflow category"),
    tags: List[str] = Query([], description="Workflow tags"),
    definition: Dict[str, Any] = Query({}, description="Workflow definition"),
    nodes: List[Dict[str, Any]] = Query([], description="Workflow nodes"),
    connections: List[Dict[str, Any]] = Query([], description="Workflow connections"),
    variables: Dict[str, Any] = Query({}, description="Workflow variables"),
    agent_id: Optional[str] = Query(None, description="Associated agent ID"),
    created_by: str = Query(..., description="Creator user ID"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Create new workflow
    
    - **name**: Workflow name
    - **description**: Workflow description
    - **workflow_type**: Workflow type
    - **category**: Workflow category
    - **tags**: Workflow tags
    - **definition**: Workflow definition
    - **nodes**: Workflow nodes
    - **connections**: Workflow connections
    - **variables**: Workflow variables
    - **agent_id**: Associated agent ID
    - **created_by**: Creator user ID
    """
    try:
        # Create workflow data
        workflow_data = {
            "name": name,
            "description": description,
            "workflow_type": workflow_type,
            "category": category,
            "tags": tags,
            "definition": definition,
            "nodes": nodes,
            "connections": connections,
            "variables": variables,
            "agent_id": agent_id,
            "created_by": created_by
        }
        
        # Create workflow
        workflow = await db_manager.create_workflow(workflow_data)
        
        return {
            "success": True,
            "message": "Workflow created successfully",
            "data": {
                "workflow_id": str(workflow.id),
                "name": workflow.name,
                "description": workflow.description,
                "workflow_type": workflow.workflow_type,
                "status": workflow.status,
                "category": workflow.category,
                "tags": workflow.tags,
                "definition": workflow.definition,
                "nodes": workflow.nodes,
                "connections": workflow.connections,
                "variables": workflow.variables,
                "execution_count": workflow.execution_count,
                "success_rate": workflow.success_rate,
                "average_duration": workflow.average_duration,
                "agent_id": str(workflow.agent_id) if workflow.agent_id else None,
                "created_by": str(workflow.created_by),
                "created_at": workflow.created_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, name=name, created_by=created_by)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.get("/workflows/{workflow_id}", response_model=Dict[str, Any])
async def get_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get workflow by ID
    
    - **workflow_id**: Workflow ID
    """
    try:
        # Get workflow
        workflow = await db_manager.get_workflow_by_id(workflow_id)
        
        if not workflow:
            raise WorkflowNotFoundError(
                "workflow_not_found",
                f"Workflow {workflow_id} not found",
                {"workflow_id": workflow_id}
            )
        
        return {
            "success": True,
            "message": "Workflow retrieved successfully",
            "data": {
                "workflow_id": str(workflow.id),
                "name": workflow.name,
                "description": workflow.description,
                "workflow_type": workflow.workflow_type,
                "status": workflow.status,
                "category": workflow.category,
                "tags": workflow.tags,
                "definition": workflow.definition,
                "nodes": workflow.nodes,
                "connections": workflow.connections,
                "variables": workflow.variables,
                "execution_count": workflow.execution_count,
                "success_rate": workflow.success_rate,
                "average_duration": workflow.average_duration,
                "last_execution": workflow.last_execution.isoformat() if workflow.last_execution else None,
                "agent_id": str(workflow.agent_id) if workflow.agent_id else None,
                "created_by": str(workflow.created_by),
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, workflow_id=workflow_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# Analytics Endpoints
@router.get("/analytics/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent_analytics(
    agent_id: str = Path(..., description="Agent ID"),
    days: int = Query(30, description="Number of days for analytics"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get agent analytics
    
    - **agent_id**: Agent ID
    - **days**: Number of days for analytics
    """
    try:
        # Get agent analytics
        analytics = await db_manager.get_agent_analytics(agent_id, days)
        
        return {
            "success": True,
            "message": "Agent analytics retrieved successfully",
            "data": {
                "agent_id": agent_id,
                "analytics": [
                    {
                        "date": analytics_item.date.isoformat(),
                        "execution_count": analytics_item.execution_count,
                        "success_count": analytics_item.success_count,
                        "failure_count": analytics_item.failure_count,
                        "average_response_time": analytics_item.average_response_time,
                        "min_response_time": analytics_item.min_response_time,
                        "max_response_time": analytics_item.max_response_time,
                        "cpu_usage_avg": analytics_item.cpu_usage_avg,
                        "memory_usage_avg": analytics_item.memory_usage_avg,
                        "network_usage_avg": analytics_item.network_usage_avg,
                        "error_rate": analytics_item.error_rate,
                        "throughput": analytics_item.throughput,
                        "efficiency_score": analytics_item.efficiency_score
                    }
                    for analytics_item in analytics
                ],
                "total_days": len(analytics)
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, agent_id=agent_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


# System Management Endpoints
@router.get("/health", response_model=Dict[str, Any])
async def database_health_check(
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Database health check
    """
    try:
        # Test database connection
        result = await db.execute(select(func.count(User.id)))
        user_count = result.scalar()
        
        result = await db.execute(select(func.count(AgentModel.id)))
        agent_count = result.scalar()
        
        result = await db.execute(select(func.count(Workflow.id)))
        workflow_count = result.scalar()
        
        return {
            "success": True,
            "message": "Database is healthy",
            "data": {
                "status": "healthy",
                "user_count": user_count,
                "agent_count": agent_count,
                "workflow_count": workflow_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.post("/maintenance/cleanup", response_model=Dict[str, Any])
async def database_cleanup(
    days_old: int = Query(30, description="Delete records older than N days"),
    user_id: str = Query(None, description="User ID for audit logging"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Database cleanup operation
    
    - **days_old**: Delete records older than N days
    - **user_id**: User ID for audit logging
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Clean up old audit logs
        audit_query = delete(AuditLog).where(AuditLog.timestamp < cutoff_date)
        audit_result = await db.execute(audit_query)
        audit_deleted = audit_result.rowcount
        
        # Clean up old notifications
        notification_query = delete(Notification).where(
            and_(
                Notification.created_at < cutoff_date,
                Notification.is_read == True
            )
        )
        notification_result = await db.execute(notification_query)
        notification_deleted = notification_result.rowcount
        
        # Clean up old system metrics
        metrics_query = delete(SystemMetrics).where(SystemMetrics.timestamp < cutoff_date)
        metrics_result = await db.execute(metrics_query)
        metrics_deleted = metrics_result.rowcount
        
        await db.commit()
        
        return {
            "success": True,
            "message": "Database cleanup completed successfully",
            "data": {
                "audit_logs_deleted": audit_deleted,
                "notifications_deleted": notification_deleted,
                "system_metrics_deleted": metrics_deleted,
                "total_deleted": audit_deleted + notification_deleted + metrics_deleted,
                "cutoff_date": cutoff_date.isoformat(),
                "cleaned_by": user_id,
                "cleaned_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e, user_id=user_id)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )


@router.get("/statistics", response_model=Dict[str, Any])
async def get_database_statistics(
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get database statistics
    """
    try:
        # Get counts for all main tables
        user_count = await db.execute(select(func.count(User.id)))
        user_count = user_count.scalar()
        
        agent_count = await db.execute(select(func.count(AgentModel.id)))
        agent_count = agent_count.scalar()
        
        workflow_count = await db.execute(select(func.count(Workflow.id)))
        workflow_count = workflow_count.scalar()
        
        execution_count = await db.execute(select(func.count(AgentExecution.id)))
        execution_count = execution_count.scalar()
        
        workflow_execution_count = await db.execute(select(func.count(WorkflowExecution.id)))
        workflow_execution_count = workflow_execution_count.scalar()
        
        notification_count = await db.execute(select(func.count(Notification.id)))
        notification_count = notification_count.scalar()
        
        audit_log_count = await db.execute(select(func.count(AuditLog.id)))
        audit_log_count = audit_log_count.scalar()
        
        return {
            "success": True,
            "message": "Database statistics retrieved successfully",
            "data": {
                "users": user_count,
                "agents": agent_count,
                "workflows": workflow_count,
                "agent_executions": execution_count,
                "workflow_executions": workflow_execution_count,
                "notifications": notification_count,
                "audit_logs": audit_log_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        error = handle_agent_error(e)
        log_agent_error(error)
        raise HTTPException(
            status_code=error.status_code,
            detail=get_error_response(error)
        )





























