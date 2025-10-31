from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
import logging
from datetime import datetime
from ..dependencies.auth import get_authenticated_user, require_permission
from ..routes.base import get_request_context, log_route_access
from ..schemas.base import BaseResponse, ErrorResponse
from ..pydantic_schemas import (
from ..version_control_roro import VersionControlService
from ..git_workflow import GitWorkflowManager
from typing import Any, List, Dict, Optional
import asyncio
"""
Version Control Router

This module contains routes for version control operations including
git management, version tracking, and rollback functionality for
product descriptions.
"""


# Import dependencies

# Import schemas
    VersionControlRequest,
    VersionControlResponse,
    GitOperationRequest,
    GitOperationResponse,
    VersionHistoryResponse,
    RollbackRequest,
    RollbackResponse
)

# Import services

# Initialize router
router = APIRouter(prefix="/version-control", tags=["version-control"])

# Logger
logger = logging.getLogger(__name__)

# Service instances
version_service = VersionControlService()
git_workflow = GitWorkflowManager()

# Route dependencies
async def get_version_service(
    context: Dict[str, Any] = Depends(get_request_context)
) -> VersionControlService:
    """Get version control service with dependencies."""
    return version_service

async def get_git_workflow(
    context: Dict[str, Any] = Depends(get_request_context)
) -> GitWorkflowManager:
    """Get git workflow manager from context."""
    return git_workflow

# Version Control Routes
@router.post("/commit", response_model=VersionControlResponse)
async def commit_changes(
    request: VersionControlRequest,
    background_tasks: BackgroundTasks,
    context: Dict[str, Any] = Depends(get_request_context),
    service: VersionControlService = Depends(get_version_service)
):
    """
    Commit changes to version control.
    
    This endpoint commits product description changes to git
    with proper version tracking and metadata.
    """
    try:
        # Log route access
        log_route_access(
            "commit_changes",
            user_id=context["user"].id if context["user"] else None,
            description_id=request.description_id
        )
        
        # Start performance monitoring
        context["performance_monitor"].start_operation("version_commit")
        
        # Commit changes
        result = await service.commit_changes(
            description_id=request.description_id,
            message=request.message,
            author=context["user"].username if context["user"] else "system",
            changes=request.changes
        )
        
        # End performance monitoring
        context["performance_monitor"].end_operation("version_commit")
        
        # Add background task for cleanup
        background_tasks.add_task(
            service.cleanup_temp_files,
            request.description_id
        )
        
        return VersionControlResponse(
            status="success",
            message="Changes committed successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error committing changes: {e}")
        context["error_monitor"].track_error("version_commit", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to commit changes"
        )

@router.get("/history/{description_id}", response_model=VersionHistoryResponse)
async def get_version_history(
    description_id: str,
    limit: int = 20,
    context: Dict[str, Any] = Depends(get_request_context),
    service: VersionControlService = Depends(get_version_service)
):
    """Get version history for a product description."""
    try:
        log_route_access("get_version_history", description_id=description_id)
        
        # Check cache first
        cache_key = f"version_history:{description_id}:{limit}"
        cached_result = await context["cache_manager"].get(cache_key)
        
        if cached_result:
            return VersionHistoryResponse(
                status="success",
                message="Version history retrieved from cache",
                data=cached_result,
                cached=True
            )
        
        # Get version history
        history = await service.get_version_history(
            description_id=description_id,
            limit=limit
        )
        
        # Cache the result
        await context["cache_manager"].set(cache_key, history, ttl=1800)
        
        return VersionHistoryResponse(
            status="success",
            message="Version history retrieved successfully",
            data=history,
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Error getting version history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve version history"
        )

@router.post("/rollback", response_model=RollbackResponse)
async def rollback_version(
    request: RollbackRequest,
    context: Dict[str, Any] = Depends(get_request_context),
    service: VersionControlService = Depends(get_version_service)
):
    """Rollback to a specific version."""
    try:
        log_route_access(
            "rollback_version",
            description_id=request.description_id,
            version=request.version
        )
        
        # Check permissions
        if context["user"] and not context["user"].is_admin:
            # Check if user owns the description
            description = await service.get_description_info(request.description_id)
            if not description or description.user_id != context["user"].id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to rollback this description"
                )
        
        # Perform rollback
        result = await service.rollback_to_version(
            description_id=request.description_id,
            version=request.version,
            user_id=context["user"].id if context["user"] else None
        )
        
        # Invalidate cache
        cache_key = f"version_history:{request.description_id}:*"
        await context["cache_manager"].delete_pattern(cache_key)
        
        return RollbackResponse(
            status="success",
            message=f"Successfully rolled back to version {request.version}",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rollback version"
        )

# Git Operations
@router.post("/git/init", response_model=GitOperationResponse)
async def initialize_git_repository(
    request: GitOperationRequest,
    context: Dict[str, Any] = Depends(get_request_context),
    git_workflow: GitWorkflowManager = Depends(get_git_workflow)
):
    """Initialize git repository for version control."""
    try:
        log_route_access("initialize_git_repository")
        
        # Check admin permissions
        if context["user"] and not context["user"].is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        result = await git_workflow.initialize_repository(
            repo_path=request.repo_path,
            remote_url=request.remote_url
        )
        
        return GitOperationResponse(
            status="success",
            message="Git repository initialized successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing git repository: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize git repository"
        )

@router.post("/git/push", response_model=GitOperationResponse)
async def push_changes(
    request: GitOperationRequest,
    context: Dict[str, Any] = Depends(get_request_context),
    git_workflow: GitWorkflowManager = Depends(get_git_workflow)
):
    """Push changes to remote repository."""
    try:
        log_route_access("push_changes")
        
        result = await git_workflow.push_changes(
            branch=request.branch or "main",
            force=request.force or False
        )
        
        return GitOperationResponse(
            status="success",
            message="Changes pushed successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error pushing changes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to push changes"
        )

@router.post("/git/pull", response_model=GitOperationResponse)
async def pull_changes(
    request: GitOperationRequest,
    context: Dict[str, Any] = Depends(get_request_context),
    git_workflow: GitWorkflowManager = Depends(get_git_workflow)
):
    """Pull changes from remote repository."""
    try:
        log_route_access("pull_changes")
        
        result = await git_workflow.pull_changes(
            branch=request.branch or "main"
        )
        
        return GitOperationResponse(
            status="success",
            message="Changes pulled successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error pulling changes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pull changes"
        )

@router.get("/git/status")
async def get_git_status(
    context: Dict[str, Any] = Depends(get_request_context),
    git_workflow: GitWorkflowManager = Depends(get_git_workflow)
):
    """Get git repository status."""
    try:
        log_route_access("get_git_status")
        
        status = await git_workflow.get_status()
        
        return {
            "status": "success",
            "message": "Git status retrieved successfully",
            "data": status
        }
        
    except Exception as e:
        logger.error(f"Error getting git status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get git status"
        )

@router.post("/git/branch", response_model=GitOperationResponse)
async def create_branch(
    request: GitOperationRequest,
    context: Dict[str, Any] = Depends(get_request_context),
    git_workflow: GitWorkflowManager = Depends(get_git_workflow)
):
    """Create a new git branch."""
    try:
        log_route_access("create_branch", branch=request.branch)
        
        result = await git_workflow.create_branch(
            branch_name=request.branch,
            checkout=request.checkout or True
        )
        
        return GitOperationResponse(
            status="success",
            message=f"Branch '{request.branch}' created successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error creating branch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create branch"
        )

@router.delete("/git/branch/{branch_name}", response_model=GitOperationResponse)
async def delete_branch(
    branch_name: str,
    force: bool = False,
    context: Dict[str, Any] = Depends(get_request_context),
    git_workflow: GitWorkflowManager = Depends(get_git_workflow)
):
    """Delete a git branch."""
    try:
        log_route_access("delete_branch", branch=branch_name)
        
        # Check admin permissions for branch deletion
        if context["user"] and not context["user"].is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for branch deletion"
            )
        
        result = await git_workflow.delete_branch(
            branch_name=branch_name,
            force=force
        )
        
        return GitOperationResponse(
            status="success",
            message=f"Branch '{branch_name}' deleted successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting branch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete branch"
        )

# Version Management
@router.get("/versions/{description_id}/latest")
async def get_latest_version(
    description_id: str,
    context: Dict[str, Any] = Depends(get_request_context),
    service: VersionControlService = Depends(get_version_service)
):
    """Get the latest version of a product description."""
    try:
        log_route_access("get_latest_version", description_id=description_id)
        
        latest = await service.get_latest_version(description_id)
        
        if not latest:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No versions found for this description"
            )
        
        return {
            "status": "success",
            "message": "Latest version retrieved successfully",
            "data": latest
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get latest version"
        )

@router.get("/versions/{description_id}/compare")
async def compare_versions(
    description_id: str,
    version1: str,
    version2: str,
    context: Dict[str, Any] = Depends(get_request_context),
    service: VersionControlService = Depends(get_version_service)
):
    """Compare two versions of a product description."""
    try:
        log_route_access(
            "compare_versions",
            description_id=description_id,
            version1=version1,
            version2=version2
        )
        
        comparison = await service.compare_versions(
            description_id=description_id,
            version1=version1,
            version2=version2
        )
        
        return {
            "status": "success",
            "message": "Version comparison completed",
            "data": comparison
        }
        
    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare versions"
        )

# Cleanup and Maintenance
@router.post("/cleanup", response_model=BaseResponse)
async def cleanup_version_control(
    context: Dict[str, Any] = Depends(get_request_context),
    service: VersionControlService = Depends(get_version_service)
):
    """Cleanup version control temporary files and old versions."""
    try:
        log_route_access("cleanup_version_control")
        
        # Check admin permissions
        if context["user"] and not context["user"].is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        cleanup_result = await service.cleanup_old_versions()
        
        return BaseResponse(
            status="success",
            message=f"Cleanup completed: {cleanup_result.files_removed} files removed, {cleanup_result.space_freed} bytes freed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform cleanup"
        ) 