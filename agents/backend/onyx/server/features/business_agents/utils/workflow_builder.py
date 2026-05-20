"""
Advanced Workflow Builder for Business Agents
===========================================

Comprehensive workflow builder with visual interface, automation, and AI assistance.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from sqlalchemy.ext.asyncio import AsyncSession
import redis

from ..schemas import (
    BusinessAgent, AgentWorkflow, WorkflowRequest, WorkflowResponse,
    WorkflowExecution, WorkflowAnalytics, WorkflowPerformance,
    ErrorResponse
)
from ..exceptions import (
    WorkflowBuilderError, WorkflowValidationError, WorkflowExecutionError,
    WorkflowNotFoundError, WorkflowPermissionDeniedError, WorkflowSystemError,
    create_agent_error, log_agent_error, handle_agent_error, get_error_response
)
from ..config import get_settings

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Workflow node types"""
    START = "start"
    END = "end"
    TASK = "task"
    DECISION = "decision"
    PARALLEL = "parallel"
    MERGE = "merge"
    LOOP = "loop"
    CONDITION = "condition"
    AGENT = "agent"
    API = "api"
    DATABASE = "database"
    NOTIFICATION = "notification"
    DELAY = "delay"
    SCRIPT = "script"
    CUSTOM = "custom"


class ConnectionType(Enum):
    """Workflow connection types"""
    SUCCESS = "success"
    FAILURE = "failure"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    MERGE = "merge"
    LOOP = "loop"


@dataclass
class WorkflowNode:
    """Workflow node definition"""
    node_id: str
    node_type: NodeType
    name: str
    description: str
    position: Tuple[float, float] = (0, 0)
    properties: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowConnection:
    """Workflow connection definition"""
    connection_id: str
    source_node: str
    target_node: str
    connection_type: ConnectionType
    condition: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTemplate:
    """Workflow template definition"""
    template_id: str
    name: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)
    nodes: List[WorkflowNode] = field(default_factory=list)
    connections: List[WorkflowConnection] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class WorkflowBuilder:
    """Advanced workflow builder for business agents"""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, WorkflowNode] = {}
        self.connections: Dict[str, WorkflowConnection] = {}
        self.templates: Dict[str, WorkflowTemplate] = {}
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        category: str = "general",
        tags: List[str] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Create a new workflow"""
        try:
            workflow_id = str(uuid4())
            
            # Create workflow structure
            workflow = {
                "workflow_id": workflow_id,
                "name": name,
                "description": description,
                "category": category,
                "tags": tags or [],
                "nodes": [],
                "connections": [],
                "variables": {},
                "metadata": {
                    "created_by": user_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "version": "1.0.0",
                    "status": "draft"
                }
            }
            
            # Store in database
            await self._store_workflow(workflow)
            
            # Cache workflow
            await self._cache_workflow(workflow_id, workflow)
            
            logger.info(f"Workflow created: {workflow_id} by user: {user_id}")
            
            return {
                "success": True,
                "message": "Workflow created successfully",
                "data": workflow
            }
            
        except Exception as e:
            error = handle_agent_error(e, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def add_node(
        self,
        workflow_id: str,
        node_type: NodeType,
        name: str,
        description: str,
        position: Tuple[float, float] = (0, 0),
        properties: Dict[str, Any] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Add a node to the workflow"""
        try:
            # Get workflow
            workflow = await self._get_workflow(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(
                    "workflow_not_found",
                    f"Workflow {workflow_id} not found",
                    {"workflow_id": workflow_id}
                )
            
            # Create node
            node_id = str(uuid4())
            node = WorkflowNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                description=description,
                position=position,
                properties=properties or {},
                metadata={
                    "created_by": user_id,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            # Add to workflow
            workflow["nodes"].append(node.__dict__)
            workflow["metadata"]["updated_at"] = datetime.utcnow().isoformat()
            
            # Update graph
            self.graph.add_node(node_id, **node.__dict__)
            self.nodes[node_id] = node
            
            # Store updated workflow
            await self._store_workflow(workflow)
            await self._cache_workflow(workflow_id, workflow)
            
            logger.info(f"Node added to workflow {workflow_id}: {node_id}")
            
            return {
                "success": True,
                "message": "Node added successfully",
                "data": {
                    "node_id": node_id,
                    "node": node.__dict__
                }
            }
            
        except Exception as e:
            error = handle_agent_error(e, workflow_id=workflow_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def add_connection(
        self,
        workflow_id: str,
        source_node: str,
        target_node: str,
        connection_type: ConnectionType,
        condition: Optional[str] = None,
        properties: Dict[str, Any] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Add a connection between nodes"""
        try:
            # Get workflow
            workflow = await self._get_workflow(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(
                    "workflow_not_found",
                    f"Workflow {workflow_id} not found",
                    {"workflow_id": workflow_id}
                )
            
            # Validate nodes exist
            source_exists = any(node["node_id"] == source_node for node in workflow["nodes"])
            target_exists = any(node["node_id"] == target_node for node in workflow["nodes"])
            
            if not source_exists or not target_exists:
                raise WorkflowValidationError(
                    "invalid_nodes",
                    "Source or target node does not exist",
                    {"source_node": source_node, "target_node": target_node}
                )
            
            # Create connection
            connection_id = str(uuid4())
            connection = WorkflowConnection(
                connection_id=connection_id,
                source_node=source_node,
                target_node=target_node,
                connection_type=connection_type,
                condition=condition,
                properties=properties or {},
                metadata={
                    "created_by": user_id,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            # Add to workflow
            workflow["connections"].append(connection.__dict__)
            workflow["metadata"]["updated_at"] = datetime.utcnow().isoformat()
            
            # Update graph
            self.graph.add_edge(source_node, target_node, **connection.__dict__)
            self.connections[connection_id] = connection
            
            # Store updated workflow
            await self._store_workflow(workflow)
            await self._cache_workflow(workflow_id, workflow)
            
            logger.info(f"Connection added to workflow {workflow_id}: {connection_id}")
            
            return {
                "success": True,
                "message": "Connection added successfully",
                "data": {
                    "connection_id": connection_id,
                    "connection": connection.__dict__
                }
            }
            
        except Exception as e:
            error = handle_agent_error(e, workflow_id=workflow_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def update_node(
        self,
        workflow_id: str,
        node_id: str,
        updates: Dict[str, Any],
        user_id: str = None
    ) -> Dict[str, Any]:
        """Update a workflow node"""
        try:
            # Get workflow
            workflow = await self._get_workflow(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(
                    "workflow_not_found",
                    f"Workflow {workflow_id} not found",
                    {"workflow_id": workflow_id}
                )
            
            # Find and update node
            node_found = False
            for i, node in enumerate(workflow["nodes"]):
                if node["node_id"] == node_id:
                    # Update node properties
                    for key, value in updates.items():
                        if key in ["name", "description", "position", "properties", "conditions", "retry_config", "timeout"]:
                            node[key] = value
                    
                    node["metadata"]["updated_at"] = datetime.utcnow().isoformat()
                    node["metadata"]["updated_by"] = user_id
                    node_found = True
                    break
            
            if not node_found:
                raise WorkflowNotFoundError(
                    "node_not_found",
                    f"Node {node_id} not found in workflow {workflow_id}",
                    {"node_id": node_id, "workflow_id": workflow_id}
                )
            
            workflow["metadata"]["updated_at"] = datetime.utcnow().isoformat()
            
            # Store updated workflow
            await self._store_workflow(workflow)
            await self._cache_workflow(workflow_id, workflow)
            
            logger.info(f"Node updated in workflow {workflow_id}: {node_id}")
            
            return {
                "success": True,
                "message": "Node updated successfully",
                "data": {"node_id": node_id, "updates": updates}
            }
            
        except Exception as e:
            error = handle_agent_error(e, workflow_id=workflow_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def delete_node(
        self,
        workflow_id: str,
        node_id: str,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Delete a workflow node"""
        try:
            # Get workflow
            workflow = await self._get_workflow(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(
                    "workflow_not_found",
                    f"Workflow {workflow_id} not found",
                    {"workflow_id": workflow_id}
                )
            
            # Remove node from workflow
            workflow["nodes"] = [node for node in workflow["nodes"] if node["node_id"] != node_id]
            
            # Remove connections involving this node
            workflow["connections"] = [
                conn for conn in workflow["connections"]
                if conn["source_node"] != node_id and conn["target_node"] != node_id
            ]
            
            workflow["metadata"]["updated_at"] = datetime.utcnow().isoformat()
            
            # Store updated workflow
            await self._store_workflow(workflow)
            await self._cache_workflow(workflow_id, workflow)
            
            logger.info(f"Node deleted from workflow {workflow_id}: {node_id}")
            
            return {
                "success": True,
                "message": "Node deleted successfully",
                "data": {"node_id": node_id}
            }
            
        except Exception as e:
            error = handle_agent_error(e, workflow_id=workflow_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def validate_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Validate workflow structure and logic"""
        try:
            # Get workflow
            workflow = await self._get_workflow(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(
                    "workflow_not_found",
                    f"Workflow {workflow_id} not found",
                    {"workflow_id": workflow_id}
                )
            
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
            
            # Check for start and end nodes
            start_nodes = [node for node in workflow["nodes"] if node["node_type"] == "start"]
            end_nodes = [node for node in workflow["nodes"] if node["node_type"] == "end"]
            
            if len(start_nodes) == 0:
                validation_results["errors"].append("Workflow must have at least one start node")
                validation_results["is_valid"] = False
            
            if len(start_nodes) > 1:
                validation_results["warnings"].append("Workflow has multiple start nodes")
            
            if len(end_nodes) == 0:
                validation_results["errors"].append("Workflow must have at least one end node")
                validation_results["is_valid"] = False
            
            # Check for orphaned nodes
            connected_nodes = set()
            for connection in workflow["connections"]:
                connected_nodes.add(connection["source_node"])
                connected_nodes.add(connection["target_node"])
            
            all_nodes = {node["node_id"] for node in workflow["nodes"]}
            orphaned_nodes = all_nodes - connected_nodes
            
            if orphaned_nodes:
                validation_results["warnings"].append(f"Orphaned nodes found: {list(orphaned_nodes)}")
            
            # Check for cycles
            graph = nx.DiGraph()
            for node in workflow["nodes"]:
                graph.add_node(node["node_id"])
            
            for connection in workflow["connections"]:
                graph.add_edge(connection["source_node"], connection["target_node"])
            
            try:
                cycles = list(nx.simple_cycles(graph))
                if cycles:
                    validation_results["warnings"].append(f"Cycles detected: {cycles}")
            except nx.NetworkXError:
                pass
            
            # Check node configurations
            for node in workflow["nodes"]:
                if node["node_type"] == "decision" and not node.get("conditions"):
                    validation_results["warnings"].append(f"Decision node {node['node_id']} has no conditions")
                
                if node["node_type"] == "agent" and not node.get("properties", {}).get("agent_id"):
                    validation_results["errors"].append(f"Agent node {node['node_id']} has no agent_id")
                    validation_results["is_valid"] = False
            
            logger.info(f"Workflow validation completed for {workflow_id}")
            
            return {
                "success": True,
                "message": "Workflow validation completed",
                "data": validation_results
            }
            
        except Exception as e:
            error = handle_agent_error(e, workflow_id=workflow_id)
            log_agent_error(error)
            raise error
    
    async def create_template(
        self,
        workflow_id: str,
        template_name: str,
        template_description: str,
        category: str = "general",
        tags: List[str] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Create a workflow template"""
        try:
            # Get workflow
            workflow = await self._get_workflow(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(
                    "workflow_not_found",
                    f"Workflow {workflow_id} not found",
                    {"workflow_id": workflow_id}
                )
            
            # Validate workflow before creating template
            validation = await self.validate_workflow(workflow_id)
            if not validation["data"]["is_valid"]:
                raise WorkflowValidationError(
                    "invalid_workflow",
                    "Cannot create template from invalid workflow",
                    {"errors": validation["data"]["errors"]}
                )
            
            # Create template
            template_id = str(uuid4())
            template = WorkflowTemplate(
                template_id=template_id,
                name=template_name,
                description=template_description,
                category=category,
                tags=tags or [],
                nodes=[WorkflowNode(**node) for node in workflow["nodes"]],
                connections=[WorkflowConnection(**conn) for conn in workflow["connections"]],
                variables=workflow.get("variables", {}),
                metadata={
                    "created_by": user_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "source_workflow_id": workflow_id
                }
            )
            
            # Store template
            await self._store_template(template)
            await self._cache_template(template_id, template)
            
            logger.info(f"Template created: {template_id} from workflow: {workflow_id}")
            
            return {
                "success": True,
                "message": "Template created successfully",
                "data": {
                    "template_id": template_id,
                    "template": template.__dict__
                }
            }
            
        except Exception as e:
            error = handle_agent_error(e, workflow_id=workflow_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def instantiate_template(
        self,
        template_id: str,
        workflow_name: str,
        workflow_description: str,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Create workflow from template"""
        try:
            # Get template
            template = await self._get_template(template_id)
            if not template:
                raise WorkflowNotFoundError(
                    "template_not_found",
                    f"Template {template_id} not found",
                    {"template_id": template_id}
                )
            
            # Create new workflow from template
            workflow_id = str(uuid4())
            workflow = {
                "workflow_id": workflow_id,
                "name": workflow_name,
                "description": workflow_description,
                "category": template.category,
                "tags": template.tags.copy(),
                "nodes": [node.__dict__ for node in template.nodes],
                "connections": [conn.__dict__ for conn in template.connections],
                "variables": template.variables.copy(),
                "metadata": {
                    "created_by": user_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "version": "1.0.0",
                    "status": "draft",
                    "template_id": template_id
                }
            }
            
            # Store workflow
            await self._store_workflow(workflow)
            await self._cache_workflow(workflow_id, workflow)
            
            logger.info(f"Workflow instantiated from template {template_id}: {workflow_id}")
            
            return {
                "success": True,
                "message": "Workflow created from template successfully",
                "data": workflow
            }
            
        except Exception as e:
            error = handle_agent_error(e, template_id=template_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def get_workflow_visualization(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow visualization data"""
        try:
            # Get workflow
            workflow = await self._get_workflow(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(
                    "workflow_not_found",
                    f"Workflow {workflow_id} not found",
                    {"workflow_id": workflow_id}
                )
            
            # Build graph for visualization
            graph = nx.DiGraph()
            
            # Add nodes
            for node in workflow["nodes"]:
                graph.add_node(
                    node["node_id"],
                    label=node["name"],
                    type=node["node_type"],
                    position=node.get("position", (0, 0)),
                    properties=node.get("properties", {})
                )
            
            # Add edges
            for connection in workflow["connections"]:
                graph.add_edge(
                    connection["source_node"],
                    connection["target_node"],
                    type=connection["connection_type"],
                    condition=connection.get("condition"),
                    properties=connection.get("properties", {})
                )
            
            # Generate layout
            try:
                pos = nx.spring_layout(graph, k=3, iterations=50)
            except:
                pos = {node: (i * 100, 0) for i, node in enumerate(graph.nodes())}
            
            # Prepare visualization data
            visualization = {
                "nodes": [
                    {
                        "id": node,
                        "label": graph.nodes[node]["label"],
                        "type": graph.nodes[node]["type"],
                        "position": pos[node],
                        "properties": graph.nodes[node]["properties"]
                    }
                    for node in graph.nodes()
                ],
                "edges": [
                    {
                        "id": f"{edge[0]}-{edge[1]}",
                        "source": edge[0],
                        "target": edge[1],
                        "type": graph.edges[edge]["type"],
                        "condition": graph.edges[edge].get("condition"),
                        "properties": graph.edges[edge].get("properties", {})
                    }
                    for edge in graph.edges()
                ],
                "metadata": {
                    "node_count": len(graph.nodes()),
                    "edge_count": len(graph.edges()),
                    "layout": "spring"
                }
            }
            
            return {
                "success": True,
                "message": "Workflow visualization generated",
                "data": visualization
            }
            
        except Exception as e:
            error = handle_agent_error(e, workflow_id=workflow_id)
            log_agent_error(error)
            raise error
    
    async def optimize_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Optimize workflow structure and performance"""
        try:
            # Get workflow
            workflow = await self._get_workflow(workflow_id)
            if not workflow:
                raise WorkflowNotFoundError(
                    "workflow_not_found",
                    f"Workflow {workflow_id} not found",
                    {"workflow_id": workflow_id}
                )
            
            optimizations = {
                "suggestions": [],
                "improvements": [],
                "performance_metrics": {}
            }
            
            # Analyze workflow structure
            graph = nx.DiGraph()
            for node in workflow["nodes"]:
                graph.add_node(node["node_id"], **node)
            
            for connection in workflow["connections"]:
                graph.add_edge(connection["source_node"], connection["target_node"], **connection)
            
            # Check for parallel execution opportunities
            for node in graph.nodes():
                successors = list(graph.successors(node))
                if len(successors) > 1:
                    # Check if successors can run in parallel
                    can_parallelize = True
                    for succ in successors:
                        if graph.nodes[succ].get("node_type") in ["decision", "merge"]:
                            can_parallelize = False
                            break
                    
                    if can_parallelize:
                        optimizations["suggestions"].append({
                            "type": "parallel_execution",
                            "node_id": node,
                            "description": f"Node {node} can execute successors in parallel",
                            "impact": "performance"
                        })
            
            # Check for redundant nodes
            node_types = {}
            for node in workflow["nodes"]:
                node_type = node["node_type"]
                if node_type not in node_types:
                    node_types[node_type] = []
                node_types[node_type].append(node)
            
            for node_type, nodes in node_types.items():
                if len(nodes) > 1 and node_type in ["delay", "notification"]:
                    optimizations["suggestions"].append({
                        "type": "consolidation",
                        "description": f"Consider consolidating {len(nodes)} {node_type} nodes",
                        "impact": "maintainability"
                    })
            
            # Calculate performance metrics
            optimizations["performance_metrics"] = {
                "node_count": len(graph.nodes()),
                "edge_count": len(graph.edges()),
                "complexity_score": len(graph.edges()) / max(len(graph.nodes()), 1),
                "parallel_opportunities": len([s for s in optimizations["suggestions"] if s["type"] == "parallel_execution"])
            }
            
            logger.info(f"Workflow optimization completed for {workflow_id}")
            
            return {
                "success": True,
                "message": "Workflow optimization completed",
                "data": optimizations
            }
            
        except Exception as e:
            error = handle_agent_error(e, workflow_id=workflow_id)
            log_agent_error(error)
            raise error
    
    # Helper methods
    async def _get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow from cache or database"""
        try:
            # Try cache first
            cached_workflow = await self._get_cached_workflow(workflow_id)
            if cached_workflow:
                return cached_workflow
            
            # Get from database
            # This would integrate with actual database
            # For now, return None
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}")
            return None
    
    async def _store_workflow(self, workflow: Dict[str, Any]) -> None:
        """Store workflow in database"""
        try:
            # This would integrate with actual database
            # For now, just log
            logger.info(f"Storing workflow: {workflow['workflow_id']}")
            
        except Exception as e:
            logger.error(f"Failed to store workflow: {e}")
            raise
    
    async def _cache_workflow(self, workflow_id: str, workflow: Dict[str, Any]) -> None:
        """Cache workflow in Redis"""
        try:
            cache_key = f"workflow:{workflow_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(workflow, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache workflow: {e}")
    
    async def _get_cached_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get cached workflow from Redis"""
        try:
            cache_key = f"workflow:{workflow_id}"
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            
        except Exception as e:
            logger.error(f"Failed to get cached workflow: {e}")
        
        return None
    
    async def _get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get template from cache or database"""
        try:
            # Try cache first
            cached_template = await self._get_cached_template(template_id)
            if cached_template:
                return cached_template
            
            # Get from database
            # This would integrate with actual database
            # For now, return None
            return None
            
        except Exception as e:
            logger.error(f"Failed to get template {template_id}: {e}")
            return None
    
    async def _store_template(self, template: WorkflowTemplate) -> None:
        """Store template in database"""
        try:
            # This would integrate with actual database
            # For now, just log
            logger.info(f"Storing template: {template.template_id}")
            
        except Exception as e:
            logger.error(f"Failed to store template: {e}")
            raise
    
    async def _cache_template(self, template_id: str, template: WorkflowTemplate) -> None:
        """Cache template in Redis"""
        try:
            cache_key = f"template:{template_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(template.__dict__, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache template: {e}")
    
    async def _get_cached_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get cached template from Redis"""
        try:
            cache_key = f"template:{template_id}"
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                template_data = json.loads(cached_data)
                return WorkflowTemplate(**template_data)
            
        except Exception as e:
            logger.error(f"Failed to get cached template: {e}")
        
        return None