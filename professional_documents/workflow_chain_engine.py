"""
Workflow Chain Engine
====================

Advanced workflow chain engine for complex document processing pipelines.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import json
import networkx as nx
from collections import defaultdict, deque
import traceback

logger = logging.getLogger(__name__)


class ChainStatus(str, Enum):
    """Chain execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):
    """Node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


class NodeType(str, Enum):
    """Node type."""
    PROCESSOR = "processor"
    CONDITION = "condition"
    PARALLEL = "parallel"
    MERGE = "merge"
    SPLIT = "split"
    DELAY = "delay"
    RETRY = "retry"
    CUSTOM = "custom"


@dataclass
class WorkflowNode:
    """Workflow node definition."""
    node_id: str
    name: str
    node_type: NodeType
    processor: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 300
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeExecution:
    """Node execution instance."""
    execution_id: str
    node_id: str
    chain_id: str
    status: NodeStatus
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    execution_time: float = 0.0


@dataclass
class WorkflowChain:
    """Workflow chain definition."""
    chain_id: str
    name: str
    description: str
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    execution_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class ChainExecution:
    """Chain execution instance."""
    execution_id: str
    chain_id: str
    status: ChainStatus
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    node_executions: Dict[str, NodeExecution] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0


class WorkflowChainEngine:
    """Advanced workflow chain engine."""
    
    def __init__(self):
        self.chains: Dict[str, WorkflowChain] = {}
        self.executions: Dict[str, ChainExecution] = {}
        self.execution_queue: deque = deque()
        self.running_executions: Dict[str, ChainExecution] = {}
        self.node_processors: Dict[str, Callable] = {}
        
        self._initialize_default_processors()
    
    def _initialize_default_processors(self):
        """Initialize default node processors."""
        
        # Document processing processors
        self.node_processors["document_analyzer"] = self._document_analyzer_processor
        self.node_processors["content_generator"] = self._content_generator_processor
        self.node_processors["quality_checker"] = self._quality_checker_processor
        self.node_processors["format_converter"] = self._format_converter_processor
        self.node_processors["approval_processor"] = self._approval_processor
        self.node_processors["notification_sender"] = self._notification_sender_processor
        
        # Utility processors
        self.node_processors["data_transformer"] = self._data_transformer_processor
        self.node_processors["condition_evaluator"] = self._condition_evaluator_processor
        self.node_processors["parallel_executor"] = self._parallel_executor_processor
        self.node_processors["delay_processor"] = self._delay_processor
        self.node_processors["retry_processor"] = self._retry_processor
    
    async def create_workflow_chain(
        self,
        name: str,
        description: str,
        nodes: List[Dict[str, Any]],
        variables: Dict[str, Any] = None
    ) -> WorkflowChain:
        """Create a new workflow chain."""
        
        chain_id = str(uuid4())
        
        # Create workflow nodes
        workflow_nodes = {}
        for node_data in nodes:
            node = WorkflowNode(
                node_id=node_data["node_id"],
                name=node_data["name"],
                node_type=NodeType(node_data["node_type"]),
                processor=self.node_processors.get(node_data.get("processor_name")),
                parameters=node_data.get("parameters", {}),
                conditions=node_data.get("conditions", {}),
                retry_config=node_data.get("retry_config", {}),
                timeout=node_data.get("timeout", 300),
                dependencies=node_data.get("dependencies", []),
                outputs=node_data.get("outputs", []),
                metadata=node_data.get("metadata", {})
            )
            workflow_nodes[node.node_id] = node
        
        # Create execution graph
        execution_graph = nx.DiGraph()
        
        # Add nodes to graph
        for node in workflow_nodes.values():
            execution_graph.add_node(node.node_id, node=node)
        
        # Add edges based on dependencies
        for node in workflow_nodes.values():
            for dep in node.dependencies:
                if dep in workflow_nodes:
                    execution_graph.add_edge(dep, node.node_id)
        
        # Validate graph
        if not nx.is_directed_acyclic_graph(execution_graph):
            raise ValueError("Workflow chain contains cycles")
        
        chain = WorkflowChain(
            chain_id=chain_id,
            name=name,
            description=description,
            nodes=workflow_nodes,
            execution_graph=execution_graph,
            variables=variables or {}
        )
        
        self.chains[chain_id] = chain
        
        logger.info(f"Created workflow chain: {name} ({chain_id})")
        
        return chain
    
    async def execute_chain(
        self,
        chain_id: str,
        input_data: Dict[str, Any] = None,
        variables: Dict[str, Any] = None
    ) -> ChainExecution:
        """Execute a workflow chain."""
        
        if chain_id not in self.chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        chain = self.chains[chain_id]
        
        # Create execution instance
        execution = ChainExecution(
            execution_id=str(uuid4()),
            chain_id=chain_id,
            status=ChainStatus.PENDING,
            input_data=input_data or {},
            variables={**chain.variables, **(variables or {})}
        )
        
        self.executions[execution.execution_id] = execution
        
        # Add to execution queue
        self.execution_queue.append(execution.execution_id)
        
        # Start execution
        asyncio.create_task(self._execute_chain_async(execution))
        
        logger.info(f"Started chain execution: {chain.name} ({execution.execution_id})")
        
        return execution
    
    async def _execute_chain_async(self, execution: ChainExecution):
        """Execute chain asynchronously."""
        
        try:
            execution.status = ChainStatus.RUNNING
            self.running_executions[execution.execution_id] = execution
            
            chain = self.chains[execution.chain_id]
            
            # Get topological order of nodes
            try:
                node_order = list(nx.topological_sort(chain.execution_graph))
            except nx.NetworkXError:
                raise ValueError("Invalid execution graph")
            
            # Execute nodes in order
            for node_id in node_order:
                if execution.status in [ChainStatus.FAILED, ChainStatus.CANCELLED]:
                    break
                
                node = chain.nodes[node_id]
                node_execution = await self._execute_node(execution, node)
                execution.node_executions[node_id] = node_execution
                
                # Check if node execution failed
                if node_execution.status == NodeStatus.FAILED:
                    execution.status = ChainStatus.FAILED
                    execution.error_message = node_execution.error_message
                    break
            
            # Mark as completed if all nodes succeeded
            if execution.status == ChainStatus.RUNNING:
                execution.status = ChainStatus.COMPLETED
                execution.completed_at = datetime.now()
                execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
        except Exception as e:
            execution.status = ChainStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            logger.error(f"Chain execution failed: {execution.execution_id} - {str(e)}")
        
        finally:
            # Remove from running executions
            if execution.execution_id in self.running_executions:
                del self.running_executions[execution.execution_id]
    
    async def _execute_node(self, execution: ChainExecution, node: WorkflowNode) -> NodeExecution:
        """Execute a single node."""
        
        node_execution = NodeExecution(
            execution_id=str(uuid4()),
            node_id=node.node_id,
            chain_id=execution.chain_id,
            status=NodeStatus.PENDING
        )
        
        try:
            # Check dependencies
            if not await self._check_node_dependencies(execution, node):
                node_execution.status = NodeStatus.WAITING
                return node_execution
            
            node_execution.status = NodeStatus.RUNNING
            node_execution.started_at = datetime.now()
            
            # Prepare input data
            input_data = await self._prepare_node_input(execution, node)
            node_execution.input_data = input_data
            
            # Execute based on node type
            if node.node_type == NodeType.PROCESSOR:
                output_data = await self._execute_processor_node(node, input_data, execution)
            elif node.node_type == NodeType.CONDITION:
                output_data = await self._execute_condition_node(node, input_data, execution)
            elif node.node_type == NodeType.PARALLEL:
                output_data = await self._execute_parallel_node(node, input_data, execution)
            elif node.node_type == NodeType.MERGE:
                output_data = await self._execute_merge_node(node, input_data, execution)
            elif node.node_type == NodeType.SPLIT:
                output_data = await self._execute_split_node(node, input_data, execution)
            elif node.node_type == NodeType.DELAY:
                output_data = await self._execute_delay_node(node, input_data, execution)
            elif node.node_type == NodeType.RETRY:
                output_data = await self._execute_retry_node(node, input_data, execution)
            else:
                output_data = await self._execute_custom_node(node, input_data, execution)
            
            node_execution.output_data = output_data
            node_execution.status = NodeStatus.COMPLETED
            node_execution.completed_at = datetime.now()
            node_execution.execution_time = (node_execution.completed_at - node_execution.started_at).total_seconds()
            
            # Update execution variables
            execution.variables.update(output_data.get("variables", {}))
            
        except Exception as e:
            node_execution.status = NodeStatus.FAILED
            node_execution.error_message = str(e)
            node_execution.completed_at = datetime.now()
            
            # Retry if configured
            if node_execution.retry_count < node.retry_config.get("max_retries", 0):
                node_execution.retry_count += 1
                node_execution.status = NodeStatus.PENDING
                await asyncio.sleep(node.retry_config.get("delay", 5))
                return await self._execute_node(execution, node)
        
        return node_execution
    
    async def _check_node_dependencies(self, execution: ChainExecution, node: WorkflowNode) -> bool:
        """Check if node dependencies are satisfied."""
        
        for dep_id in node.dependencies:
            if dep_id not in execution.node_executions:
                return False
            
            dep_execution = execution.node_executions[dep_id]
            if dep_execution.status != NodeStatus.COMPLETED:
                return False
        
        return True
    
    async def _prepare_node_input(self, execution: ChainExecution, node: WorkflowNode) -> Dict[str, Any]:
        """Prepare input data for node execution."""
        
        input_data = {
            "chain_input": execution.input_data,
            "variables": execution.variables,
            "node_parameters": node.parameters
        }
        
        # Add outputs from dependencies
        for dep_id in node.dependencies:
            if dep_id in execution.node_executions:
                dep_execution = execution.node_executions[dep_id]
                input_data[f"dep_{dep_id}"] = dep_execution.output_data
        
        return input_data
    
    async def _execute_processor_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: ChainExecution
    ) -> Dict[str, Any]:
        """Execute processor node."""
        
        if not node.processor:
            raise ValueError(f"No processor defined for node {node.node_id}")
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                node.processor(input_data, execution.variables),
                timeout=node.timeout
            )
            return result
        except asyncio.TimeoutError:
            raise ValueError(f"Node {node.node_id} execution timed out")
    
    async def _execute_condition_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: ChainExecution
    ) -> Dict[str, Any]:
        """Execute condition node."""
        
        conditions = node.conditions
        result = {"condition_result": False, "variables": {}}
        
        # Evaluate conditions
        for condition_name, condition_config in conditions.items():
            condition_result = await self._evaluate_condition(condition_config, input_data, execution.variables)
            result["variables"][f"condition_{condition_name}"] = condition_result
        
        # Overall condition result
        result["condition_result"] = all(result["variables"].values())
        
        return result
    
    async def _execute_parallel_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: ChainExecution
    ) -> Dict[str, Any]:
        """Execute parallel node."""
        
        parallel_tasks = node.parameters.get("tasks", [])
        results = []
        
        # Execute tasks in parallel
        tasks = []
        for task_config in parallel_tasks:
            task = asyncio.create_task(
                self._execute_parallel_task(task_config, input_data, execution.variables)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(parallel_results):
            if isinstance(result, Exception):
                results.append({"error": str(result), "task_index": i})
            else:
                results.append(result)
        
        return {"parallel_results": results}
    
    async def _execute_merge_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: ChainExecution
    ) -> Dict[str, Any]:
        """Execute merge node."""
        
        merge_strategy = node.parameters.get("strategy", "combine")
        merge_data = {}
        
        # Collect data from dependencies
        for dep_id in node.dependencies:
            if f"dep_{dep_id}" in input_data:
                dep_data = input_data[f"dep_{dep_id}"]
                if merge_strategy == "combine":
                    merge_data.update(dep_data)
                elif merge_strategy == "append":
                    for key, value in dep_data.items():
                        if key not in merge_data:
                            merge_data[key] = []
                        if isinstance(value, list):
                            merge_data[key].extend(value)
                        else:
                            merge_data[key].append(value)
        
        return {"merged_data": merge_data}
    
    async def _execute_split_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: ChainExecution
    ) -> Dict[str, Any]:
        """Execute split node."""
        
        split_config = node.parameters.get("split_config", {})
        split_data = input_data.get("chain_input", {})
        
        # Split data based on configuration
        split_results = {}
        for split_name, split_rule in split_config.items():
            if "field" in split_rule:
                field_name = split_rule["field"]
                if field_name in split_data:
                    split_results[split_name] = {field_name: split_data[field_name]}
        
        return {"split_results": split_results}
    
    async def _execute_delay_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: ChainExecution
    ) -> Dict[str, Any]:
        """Execute delay node."""
        
        delay_seconds = node.parameters.get("delay_seconds", 0)
        await asyncio.sleep(delay_seconds)
        
        return {"delayed": True, "delay_seconds": delay_seconds}
    
    async def _execute_retry_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: ChainExecution
    ) -> Dict[str, Any]:
        """Execute retry node."""
        
        retry_config = node.retry_config
        max_retries = retry_config.get("max_retries", 3)
        delay = retry_config.get("delay", 5)
        
        for attempt in range(max_retries + 1):
            try:
                # Execute the retry logic
                result = await self._execute_retry_logic(node, input_data, execution.variables)
                return {"retry_success": True, "attempts": attempt + 1, "result": result}
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise e
    
    async def _execute_custom_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: ChainExecution
    ) -> Dict[str, Any]:
        """Execute custom node."""
        
        custom_processor = node.processor
        if custom_processor:
            return await custom_processor(input_data, execution.variables)
        else:
            return {"custom_executed": True, "node_id": node.node_id}
    
    async def _evaluate_condition(
        self,
        condition_config: Dict[str, Any],
        input_data: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> bool:
        """Evaluate a condition."""
        
        condition_type = condition_config.get("type", "equals")
        field = condition_config.get("field")
        value = condition_config.get("value")
        
        # Get field value
        field_value = None
        if field in input_data:
            field_value = input_data[field]
        elif field in variables:
            field_value = variables[field]
        
        # Evaluate condition
        if condition_type == "equals":
            return field_value == value
        elif condition_type == "not_equals":
            return field_value != value
        elif condition_type == "greater_than":
            return field_value > value
        elif condition_type == "less_than":
            return field_value < value
        elif condition_type == "contains":
            return value in str(field_value)
        elif condition_type == "not_contains":
            return value not in str(field_value)
        else:
            return False
    
    async def _execute_parallel_task(
        self,
        task_config: Dict[str, Any],
        input_data: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a parallel task."""
        
        task_type = task_config.get("type", "custom")
        task_processor = self.node_processors.get(task_type)
        
        if task_processor:
            return await task_processor(input_data, variables)
        else:
            return {"task_type": task_type, "executed": True}
    
    async def _execute_retry_logic(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute retry logic."""
        
        retry_processor = node.processor
        if retry_processor:
            return await retry_processor(input_data, variables)
        else:
            return {"retry_logic_executed": True}
    
    # Default processors
    async def _document_analyzer_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Document analyzer processor."""
        content = input_data.get("chain_input", {}).get("content", "")
        
        return {
            "word_count": len(content.split()),
            "sentence_count": len(content.split('.')),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "analysis_complete": True
        }
    
    async def _content_generator_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Content generator processor."""
        template = input_data.get("node_parameters", {}).get("template", "default")
        
        return {
            "generated_content": f"Generated content using template: {template}",
            "template_used": template,
            "generation_complete": True
        }
    
    async def _quality_checker_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Quality checker processor."""
        content = input_data.get("chain_input", {}).get("content", "")
        
        # Simple quality metrics
        quality_score = min(100, len(content) / 10)  # Mock quality score
        
        return {
            "quality_score": quality_score,
            "quality_passed": quality_score >= 70,
            "quality_metrics": {
                "length_score": min(100, len(content) / 10),
                "structure_score": 85,
                "readability_score": 90
            }
        }
    
    async def _format_converter_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Format converter processor."""
        content = input_data.get("chain_input", {}).get("content", "")
        target_format = input_data.get("node_parameters", {}).get("target_format", "html")
        
        return {
            "converted_content": f"<{target_format}>{content}</{target_format}>",
            "target_format": target_format,
            "conversion_complete": True
        }
    
    async def _approval_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Approval processor."""
        content = input_data.get("chain_input", {}).get("content", "")
        
        # Mock approval logic
        approved = len(content) > 50  # Simple approval criteria
        
        return {
            "approved": approved,
            "approval_reason": "Content meets minimum length requirement" if approved else "Content too short",
            "approval_timestamp": datetime.now().isoformat()
        }
    
    async def _notification_sender_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Notification sender processor."""
        message = input_data.get("node_parameters", {}).get("message", "Workflow completed")
        recipients = input_data.get("node_parameters", {}).get("recipients", [])
        
        return {
            "notification_sent": True,
            "message": message,
            "recipients": recipients,
            "sent_at": datetime.now().isoformat()
        }
    
    async def _data_transformer_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Data transformer processor."""
        transformation = input_data.get("node_parameters", {}).get("transformation", "uppercase")
        data = input_data.get("chain_input", {}).get("data", "")
        
        if transformation == "uppercase":
            transformed_data = data.upper()
        elif transformation == "lowercase":
            transformed_data = data.lower()
        else:
            transformed_data = data
        
        return {
            "transformed_data": transformed_data,
            "transformation": transformation,
            "transformation_complete": True
        }
    
    async def _condition_evaluator_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Condition evaluator processor."""
        conditions = input_data.get("node_parameters", {}).get("conditions", {})
        
        results = {}
        for condition_name, condition_config in conditions.items():
            results[condition_name] = await self._evaluate_condition(condition_config, input_data, variables)
        
        return {
            "condition_results": results,
            "all_conditions_met": all(results.values())
        }
    
    async def _parallel_executor_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel executor processor."""
        tasks = input_data.get("node_parameters", {}).get("tasks", [])
        
        # Execute tasks in parallel
        task_results = []
        for task in tasks:
            task_result = await self._execute_parallel_task(task, input_data, variables)
            task_results.append(task_result)
        
        return {
            "parallel_results": task_results,
            "tasks_completed": len(task_results)
        }
    
    async def _delay_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Delay processor."""
        delay_seconds = input_data.get("node_parameters", {}).get("delay_seconds", 0)
        await asyncio.sleep(delay_seconds)
        
        return {
            "delay_completed": True,
            "delay_seconds": delay_seconds
        }
    
    async def _retry_processor(self, input_data: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Retry processor."""
        max_retries = input_data.get("node_parameters", {}).get("max_retries", 3)
        delay = input_data.get("node_parameters", {}).get("delay", 5)
        
        for attempt in range(max_retries + 1):
            try:
                # Mock retry logic
                if attempt < 2:  # Fail first two attempts
                    raise Exception(f"Attempt {attempt + 1} failed")
                else:
                    return {
                        "retry_success": True,
                        "attempts": attempt + 1,
                        "result": "Retry successful"
                    }
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise e
    
    async def get_chain_status(self, execution_id: str) -> Dict[str, Any]:
        """Get chain execution status."""
        
        if execution_id not in self.executions:
            raise ValueError(f"Execution {execution_id} not found")
        
        execution = self.executions[execution_id]
        chain = self.chains[execution.chain_id]
        
        # Calculate progress
        total_nodes = len(chain.nodes)
        completed_nodes = len([ne for ne in execution.node_executions.values() if ne.status == NodeStatus.COMPLETED])
        progress = (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0
        
        return {
            "execution_id": execution_id,
            "chain_id": execution.chain_id,
            "chain_name": chain.name,
            "status": execution.status.value,
            "progress": progress,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "execution_time": execution.execution_time,
            "error_message": execution.error_message,
            "node_executions": [
                {
                    "node_id": ne.node_id,
                    "status": ne.status.value,
                    "started_at": ne.started_at.isoformat() if ne.started_at else None,
                    "completed_at": ne.completed_at.isoformat() if ne.completed_at else None,
                    "execution_time": ne.execution_time,
                    "error_message": ne.error_message
                }
                for ne in execution.node_executions.values()
            ]
        }
    
    async def get_chain_analytics(self) -> Dict[str, Any]:
        """Get workflow chain analytics."""
        
        total_chains = len(self.chains)
        total_executions = len(self.executions)
        completed_executions = len([e for e in self.executions.values() if e.status == ChainStatus.COMPLETED])
        failed_executions = len([e for e in self.executions.values() if e.status == ChainStatus.FAILED])
        
        # Average execution time
        completed_times = [e.execution_time for e in self.executions.values() if e.execution_time > 0]
        avg_execution_time = sum(completed_times) / len(completed_times) if completed_times else 0
        
        # Node type distribution
        node_types = defaultdict(int)
        for chain in self.chains.values():
            for node in chain.nodes.values():
                node_types[node.node_type.value] += 1
        
        return {
            "total_chains": total_chains,
            "total_executions": total_executions,
            "completed_executions": completed_executions,
            "failed_executions": failed_executions,
            "success_rate": (completed_executions / total_executions * 100) if total_executions > 0 else 0,
            "average_execution_time": avg_execution_time,
            "node_type_distribution": dict(node_types),
            "running_executions": len(self.running_executions),
            "queued_executions": len(self.execution_queue)
        }



























