"""
Workflow Domain Service
========================

Domain service for workflow business logic and validation.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..entities.workflow_chain import WorkflowChain
from ..entities.workflow_node import WorkflowNode
from ..value_objects.workflow_id import WorkflowId
from ..value_objects.node_id import NodeId
from ..value_objects.workflow_status import WorkflowStatus
from ..value_objects.priority import Priority
from ..exceptions.workflow_exceptions import (
    WorkflowNotFoundException,
    InvalidWorkflowStatusTransition,
    WorkflowValidationException
)
from ..exceptions.node_exceptions import (
    NodeNotFoundException,
    InvalidNodeConfigurationException
)


logger = logging.getLogger(__name__)


@dataclass
class WorkflowStatistics:
    """Workflow statistics"""
    total_nodes: int
    active_nodes: int
    completed_nodes: int
    error_nodes: int
    average_quality_score: float
    total_word_count: int
    estimated_reading_time: int
    most_used_tags: List[Dict[str, Any]]


class WorkflowDomainService:
    """
    Workflow domain service
    
    Contains business logic that doesn't belong to entities
    and coordinates between multiple entities.
    """
    
    def __init__(self):
        self._max_nodes_per_workflow = 1000
        self._max_workflow_name_length = 255
        self._max_workflow_description_length = 1000
        self._max_node_title_length = 255
        self._max_node_content_length = 100000
    
    def validate_workflow_creation(
        self,
        name: str,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate workflow creation parameters"""
        if not name or not name.strip():
            raise WorkflowValidationException("Workflow name cannot be empty")
        
        if len(name) > self._max_workflow_name_length:
            raise WorkflowValidationException(
                f"Workflow name cannot exceed {self._max_workflow_name_length} characters"
            )
        
        if description and len(description) > self._max_workflow_description_length:
            raise WorkflowValidationException(
                f"Workflow description cannot exceed {self._max_workflow_description_length} characters"
            )
        
        # Validate settings if provided
        if settings:
            self._validate_workflow_settings(settings)
    
    def validate_workflow_update(
        self,
        workflow: WorkflowChain,
        updates: Dict[str, Any]
    ) -> None:
        """Validate workflow update parameters"""
        if "name" in updates:
            self.validate_workflow_creation(
                updates["name"],
                updates.get("description"),
                updates.get("settings")
            )
        
        if "status" in updates:
            self.validate_status_transition(workflow.status, WorkflowStatus(updates["status"]))
    
    def validate_status_transition(
        self,
        current_status: WorkflowStatus,
        new_status: WorkflowStatus
    ) -> None:
        """Validate workflow status transition"""
        valid_transitions = {
            WorkflowStatus.CREATED: [WorkflowStatus.ACTIVE, WorkflowStatus.CANCELLED, WorkflowStatus.DELETED],
            WorkflowStatus.ACTIVE: [WorkflowStatus.PAUSED, WorkflowStatus.COMPLETED, WorkflowStatus.ERROR, WorkflowStatus.CANCELLED],
            WorkflowStatus.PAUSED: [WorkflowStatus.ACTIVE, WorkflowStatus.CANCELLED, WorkflowStatus.DELETED],
            WorkflowStatus.COMPLETED: [WorkflowStatus.ACTIVE, WorkflowStatus.DELETED],
            WorkflowStatus.ERROR: [WorkflowStatus.ACTIVE, WorkflowStatus.CANCELLED, WorkflowStatus.DELETED],
            WorkflowStatus.CANCELLED: [WorkflowStatus.DELETED],
            WorkflowStatus.DELETED: []  # No transitions from deleted
        }
        
        if new_status not in valid_transitions.get(current_status, []):
            raise InvalidWorkflowStatusTransition(
                workflow_id=WorkflowId(),  # This would be the actual workflow ID
                current_status=current_status,
                target_status=new_status
            )
    
    def validate_node_creation(
        self,
        workflow: WorkflowChain,
        title: str,
        content: str,
        prompt: str,
        parent_id: Optional[NodeId] = None
    ) -> None:
        """Validate node creation parameters"""
        # Check workflow capacity
        if len(workflow.nodes) >= self._max_nodes_per_workflow:
            raise WorkflowValidationException(
                f"Workflow cannot have more than {self._max_nodes_per_workflow} nodes"
            )
        
        # Validate title
        if not title or not title.strip():
            raise WorkflowValidationException("Node title cannot be empty")
        
        if len(title) > self._max_node_title_length:
            raise WorkflowValidationException(
                f"Node title cannot exceed {self._max_node_title_length} characters"
            )
        
        # Validate content
        if not content or not content.strip():
            raise WorkflowValidationException("Node content cannot be empty")
        
        if len(content) > self._max_node_content_length:
            raise WorkflowValidationException(
                f"Node content cannot exceed {self._max_node_content_length} characters"
            )
        
        # Validate prompt
        if not prompt or not prompt.strip():
            raise WorkflowValidationException("Node prompt cannot be empty")
        
        # Validate parent node if specified
        if parent_id:
            parent_node = self._find_node_by_id(workflow, parent_id)
            if not parent_node:
                raise NodeNotFoundException(
                    workflow_id=workflow.id,
                    node_id=parent_id
                )
    
    def validate_node_update(
        self,
        workflow: WorkflowChain,
        node: WorkflowNode,
        updates: Dict[str, Any]
    ) -> None:
        """Validate node update parameters"""
        if "title" in updates:
            if not updates["title"] or not updates["title"].strip():
                raise WorkflowValidationException("Node title cannot be empty")
            
            if len(updates["title"]) > self._max_node_title_length:
                raise WorkflowValidationException(
                    f"Node title cannot exceed {self._max_node_title_length} characters"
                )
        
        if "content" in updates:
            if not updates["content"] or not updates["content"].strip():
                raise WorkflowValidationException("Node content cannot be empty")
            
            if len(updates["content"]) > self._max_node_content_length:
                raise WorkflowValidationException(
                    f"Node content cannot exceed {self._max_node_content_length} characters"
                )
        
        if "parent_id" in updates and updates["parent_id"]:
            parent_id = NodeId(updates["parent_id"])
            parent_node = self._find_node_by_id(workflow, parent_id)
            if not parent_node:
                raise NodeNotFoundException(
                    workflow_id=workflow.id,
                    node_id=parent_id
                )
            
            # Check for circular references
            if self._would_create_circular_reference(workflow, node.id, parent_id):
                raise WorkflowValidationException("Cannot create circular reference in node hierarchy")
    
    def calculate_workflow_statistics(self, workflow: WorkflowChain) -> WorkflowStatistics:
        """Calculate workflow statistics"""
        nodes = workflow.nodes
        
        total_nodes = len(nodes)
        active_nodes = len([n for n in nodes if n.status == WorkflowStatus.ACTIVE])
        completed_nodes = len([n for n in nodes if n.status == WorkflowStatus.COMPLETED])
        error_nodes = len([n for n in nodes if n.status == WorkflowStatus.ERROR])
        
        # Calculate average quality score
        quality_scores = [n.quality_scores.overall_score for n in nodes if n.quality_scores.overall_score is not None]
        average_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Calculate total word count
        total_word_count = sum(n.content_metrics.word_count for n in nodes if n.content_metrics.word_count)
        
        # Calculate estimated reading time
        estimated_reading_time = sum(n.content_metrics.reading_time_minutes for n in nodes if n.content_metrics.reading_time_minutes)
        
        # Find most used tags
        tag_counts = {}
        for node in nodes:
            for tag in node.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        most_used_tags = [
            {"tag": tag, "count": count}
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        return WorkflowStatistics(
            total_nodes=total_nodes,
            active_nodes=active_nodes,
            completed_nodes=completed_nodes,
            error_nodes=error_nodes,
            average_quality_score=average_quality_score,
            total_word_count=total_word_count,
            estimated_reading_time=estimated_reading_time,
            most_used_tags=most_used_tags
        )
    
    def can_add_node_to_workflow(self, workflow: WorkflowChain) -> bool:
        """Check if a node can be added to workflow"""
        return (
            workflow.status in [WorkflowStatus.CREATED, WorkflowStatus.ACTIVE] and
            len(workflow.nodes) < self._max_nodes_per_workflow
        )
    
    def can_update_workflow(self, workflow: WorkflowChain) -> bool:
        """Check if workflow can be updated"""
        return workflow.status != WorkflowStatus.DELETED
    
    def can_delete_workflow(self, workflow: WorkflowChain) -> bool:
        """Check if workflow can be deleted"""
        return workflow.status in [WorkflowStatus.CREATED, WorkflowStatus.CANCELLED, WorkflowStatus.ERROR]
    
    def get_workflow_health_score(self, workflow: WorkflowChain) -> float:
        """Calculate workflow health score (0-100)"""
        if not workflow.nodes:
            return 100.0  # Empty workflow is considered healthy
        
        # Calculate health based on node statuses
        total_nodes = len(workflow.nodes)
        error_nodes = len([n for n in workflow.nodes if n.status == WorkflowStatus.ERROR])
        completed_nodes = len([n for n in workflow.nodes if n.status == WorkflowStatus.COMPLETED])
        
        # Health score calculation
        error_penalty = (error_nodes / total_nodes) * 50  # Each error node reduces score by up to 50 points
        completion_bonus = (completed_nodes / total_nodes) * 20  # Completed nodes add up to 20 points
        
        health_score = 100 - error_penalty + completion_bonus
        return max(0.0, min(100.0, health_score))
    
    def get_workflow_complexity_score(self, workflow: WorkflowChain) -> float:
        """Calculate workflow complexity score (0-100)"""
        if not workflow.nodes:
            return 0.0
        
        # Factors that increase complexity
        node_count_factor = min(len(workflow.nodes) / 100, 1.0) * 30  # Up to 30 points for node count
        hierarchy_depth_factor = self._calculate_hierarchy_depth(workflow) * 20  # Up to 20 points for depth
        tag_diversity_factor = self._calculate_tag_diversity(workflow) * 20  # Up to 20 points for tag diversity
        content_length_factor = self._calculate_content_length_factor(workflow) * 30  # Up to 30 points for content
        
        complexity_score = node_count_factor + hierarchy_depth_factor + tag_diversity_factor + content_length_factor
        return min(100.0, complexity_score)
    
    def _validate_workflow_settings(self, settings: Dict[str, Any]) -> None:
        """Validate workflow settings"""
        # Validate specific settings
        if "max_nodes" in settings:
            max_nodes = settings["max_nodes"]
            if not isinstance(max_nodes, int) or max_nodes < 1 or max_nodes > self._max_nodes_per_workflow:
                raise WorkflowValidationException(
                    f"max_nodes must be an integer between 1 and {self._max_nodes_per_workflow}"
                )
        
        if "timeout" in settings:
            timeout = settings["timeout"]
            if not isinstance(timeout, int) or timeout < 1 or timeout > 3600:
                raise WorkflowValidationException("timeout must be an integer between 1 and 3600 seconds")
    
    def _find_node_by_id(self, workflow: WorkflowChain, node_id: NodeId) -> Optional[WorkflowNode]:
        """Find node by ID in workflow"""
        for node in workflow.nodes:
            if node.id == node_id:
                return node
        return None
    
    def _would_create_circular_reference(
        self,
        workflow: WorkflowChain,
        node_id: NodeId,
        new_parent_id: NodeId
    ) -> bool:
        """Check if setting parent would create circular reference"""
        if node_id == new_parent_id:
            return True
        
        # Check if new_parent_id is a descendant of node_id
        current_node = self._find_node_by_id(workflow, new_parent_id)
        while current_node and current_node.parent_id:
            if current_node.parent_id == node_id:
                return True
            current_node = self._find_node_by_id(workflow, current_node.parent_id)
        
        return False
    
    def _calculate_hierarchy_depth(self, workflow: WorkflowChain) -> float:
        """Calculate maximum hierarchy depth in workflow"""
        if not workflow.nodes:
            return 0.0
        
        # Build parent-child relationships
        children_map = {}
        for node in workflow.nodes:
            if node.parent_id:
                if node.parent_id not in children_map:
                    children_map[node.parent_id] = []
                children_map[node.parent_id].append(node.id)
        
        # Calculate depth for each root node
        max_depth = 0
        for node in workflow.nodes:
            if not node.parent_id:  # Root node
                depth = self._calculate_node_depth(node.id, children_map)
                max_depth = max(max_depth, depth)
        
        return min(max_depth / 10, 1.0)  # Normalize to 0-1 range
    
    def _calculate_node_depth(self, node_id: NodeId, children_map: Dict[NodeId, List[NodeId]]) -> int:
        """Calculate depth of a node in hierarchy"""
        if node_id not in children_map:
            return 1
        
        max_child_depth = 0
        for child_id in children_map[node_id]:
            child_depth = self._calculate_node_depth(child_id, children_map)
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth
    
    def _calculate_tag_diversity(self, workflow: WorkflowChain) -> float:
        """Calculate tag diversity in workflow"""
        if not workflow.nodes:
            return 0.0
        
        all_tags = set()
        for node in workflow.nodes:
            all_tags.update(node.tags)
        
        if not all_tags:
            return 0.0
        
        # Diversity is based on unique tags vs total tag usage
        total_tag_usage = sum(len(node.tags) for node in workflow.nodes)
        unique_tags = len(all_tags)
        
        return min(unique_tags / total_tag_usage, 1.0) if total_tag_usage > 0 else 0.0
    
    def _calculate_content_length_factor(self, workflow: WorkflowChain) -> float:
        """Calculate content length factor for complexity"""
        if not workflow.nodes:
            return 0.0
        
        total_content_length = sum(
            node.content_metrics.character_count or 0
            for node in workflow.nodes
        )
        
        # Normalize based on expected content length (100k characters = 1.0)
        return min(total_content_length / 100000, 1.0)




