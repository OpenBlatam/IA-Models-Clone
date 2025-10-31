"""
Workflow Service - Fast Implementation
======================================

Fast workflow service with AI integration.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from ..models.workflow import WorkflowChain, WorkflowNode
from ..core.database import get_database

logger = logging.getLogger(__name__)


class WorkflowService:
    """Fast workflow service with AI integration"""
    
    def __init__(self):
        self.ai_service = None
        self.cache_service = None
        self.notification_service = None
        self.analytics_service = None
    
    async def create_workflow(
        self,
        name: str,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None
    ) -> WorkflowChain:
        """Create workflow with AI defaults"""
        try:
            # Generate AI description if not provided
            if not description and self.ai_service:
                description = await self.ai_service.generate_description(name)
            
            # Create workflow
            workflow = WorkflowChain(
                name=name,
                description=description,
                config=config or {},
                status="draft"
            )
            
            # Save to database
            async with get_database() as db:
                db.add(workflow)
                await db.commit()
                await db.refresh(workflow)
            
            # Track analytics
            if self.analytics_service:
                await self.analytics_service.track_event(
                    "workflow_created",
                    {"workflow_id": workflow.id, "user_id": user_id}
                )
            
            logger.info(f"Workflow created: {workflow.id}")
            return workflow
        
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise
    
    async def get_workflow(self, workflow_id: int) -> Optional[WorkflowChain]:
        """Get workflow with caching"""
        try:
            # Try cache first
            if self.cache_service:
                cached = await self.cache_service.get(f"workflow:{workflow_id}")
                if cached:
                    return cached
            
            # Get from database
            async with get_database() as db:
                result = await db.execute(
                    select(WorkflowChain).where(WorkflowChain.id == workflow_id)
                )
                workflow = result.scalar_one_or_none()
            
            # Cache result
            if workflow and self.cache_service:
                await self.cache_service.set(f"workflow:{workflow_id}", workflow, ttl=300)
            
            return workflow
        
        except Exception as e:
            logger.error(f"Failed to get workflow: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: int) -> Dict[str, Any]:
        """Execute workflow with AI processing"""
        try:
            # Get workflow
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                raise ValueError("Workflow not found")
            
            # Get nodes
            async with get_database() as db:
                result = await db.execute(
                    select(WorkflowNode).where(WorkflowNode.workflow_id == workflow_id)
                )
                nodes = result.scalars().all()
            
            # Execute nodes
            results = {}
            for node in nodes:
                node_result = await self._execute_node(node)
                results[node.id] = node_result
            
            # Update workflow status
            await self.update_workflow(workflow_id, {"status": "completed"})
            
            logger.info(f"Workflow executed: {workflow_id}")
            return results
        
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            raise
    
    async def _execute_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute node with AI processing"""
        try:
            # Use AI service for processing
            if self.ai_service and node.node_type == "ai_processing":
                result = await self.ai_service.process_node(node)
                return {"success": True, "result": result}
            
            # Default processing
            return {
                "success": True,
                "result": f"Processed {node.name}",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to execute node {node.id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_workflow(
        self,
        workflow_id: int,
        updates: Dict[str, Any]
    ) -> Optional[WorkflowChain]:
        """Update workflow"""
        try:
            async with get_database() as db:
                result = await db.execute(
                    select(WorkflowChain).where(WorkflowChain.id == workflow_id)
                )
                workflow = result.scalar_one_or_none()
                
                if not workflow:
                    return None
                
                # Apply updates
                for key, value in updates.items():
                    if hasattr(workflow, key):
                        setattr(workflow, key, value)
                
                await db.commit()
                await db.refresh(workflow)
            
            # Invalidate cache
            if self.cache_service:
                await self.cache_service.delete(f"workflow:{workflow_id}")
            
            return workflow
        
        except Exception as e:
            logger.error(f"Failed to update workflow: {e}")
            raise


# Global workflow service instance
workflow_service = WorkflowService()