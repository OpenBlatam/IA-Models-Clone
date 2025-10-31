"""
Advanced Workflow Engine for BUL
Handles complex document generation pipelines with conditional logic, parallel processing, and error handling
"""

from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import json
import uuid
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Workflow step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    """Workflow status"""
    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class StepType(str, Enum):
    """Step types"""
    AI_GENERATION = "ai_generation"
    DATA_PROCESSING = "data_processing"
    VALIDATION = "validation"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    API_CALL = "api_call"
    FILE_OPERATION = "file_operation"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class ConditionOperator(str, Enum):
    """Condition operators"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    REGEX_MATCH = "regex_match"


@dataclass
class WorkflowContext:
    """Workflow execution context"""
    workflow_id: str
    execution_id: str
    user_id: str
    data: Dict[str, Any]
    variables: Dict[str, Any]
    start_time: datetime
    current_step: Optional[str] = None
    error_count: int = 0
    retry_count: int = 0


class Condition(BaseModel):
    """Condition definition"""
    field: str = Field(..., description="Field to evaluate")
    operator: ConditionOperator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    description: Optional[str] = Field(None, description="Human-readable description")


class WorkflowStep(BaseModel):
    """Workflow step definition"""
    id: str = Field(..., description="Unique step ID")
    name: str = Field(..., description="Step name")
    type: StepType = Field(..., description="Step type")
    description: str = Field(..., description="Step description")
    config: Dict[str, Any] = Field(default_factory=dict, description="Step configuration")
    conditions: List[Condition] = Field(default_factory=list, description="Execution conditions")
    retry_config: Dict[str, Any] = Field(default_factory=dict, description="Retry configuration")
    timeout: Optional[int] = Field(None, description="Step timeout in seconds")
    dependencies: List[str] = Field(default_factory=list, description="Step dependencies")
    parallel_steps: List[str] = Field(default_factory=list, description="Steps to run in parallel")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Step status")
    result: Optional[Dict[str, Any]] = Field(None, description="Step result")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[datetime] = Field(None, description="Step start time")
    completed_at: Optional[datetime] = Field(None, description="Step completion time")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class WorkflowDefinition(BaseModel):
    """Workflow definition"""
    id: str = Field(..., description="Unique workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Workflow variables")
    triggers: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow triggers")
    status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT, description="Workflow status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="Creator user ID")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WorkflowExecution(BaseModel):
    """Workflow execution instance"""
    id: str = Field(..., description="Unique execution ID")
    workflow_id: str = Field(..., description="Workflow definition ID")
    user_id: str = Field(..., description="User ID")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING, description="Execution status")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    results: Dict[str, Any] = Field(default_factory=dict, description="Execution results")
    errors: List[str] = Field(default_factory=list, description="Execution errors")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    duration: Optional[float] = Field(None, description="Execution duration in seconds")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Execution progress")
    current_step: Optional[str] = Field(None, description="Current executing step")
    step_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Step results")


class StepExecutor(ABC):
    """Abstract base class for step executors"""
    
    @abstractmethod
    async def execute(self, step: WorkflowStep, context: WorkflowContext) -> Dict[str, Any]:
        """Execute a workflow step"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate step configuration"""
        pass


class AIGenerationExecutor(StepExecutor):
    """AI content generation step executor"""
    
    async def execute(self, step: WorkflowStep, context: WorkflowContext) -> Dict[str, Any]:
        """Execute AI generation step"""
        config = step.config
        
        # Extract parameters
        prompt_template = config.get("prompt_template", "")
        model_id = config.get("model_id")
        max_tokens = config.get("max_tokens", 4000)
        temperature = config.get("temperature", 0.7)
        
        # Replace variables in prompt
        prompt = self._replace_variables(prompt_template, context.variables)
        
        # This would integrate with the model manager
        # For now, simulating the generation
        await asyncio.sleep(0.5)  # Simulate AI generation
        
        generated_content = f"Generated content for: {prompt[:100]}..."
        
        return {
            "content": generated_content,
            "model_used": model_id or "default",
            "tokens_used": len(generated_content.split()),
            "generation_time": 0.5
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate AI generation config"""
        required_fields = ["prompt_template"]
        return all(field in config for field in required_fields)
    
    def _replace_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """Replace variables in template"""
        for key, value in variables.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template


class DataProcessingExecutor(StepExecutor):
    """Data processing step executor"""
    
    async def execute(self, step: WorkflowStep, context: WorkflowContext) -> Dict[str, Any]:
        """Execute data processing step"""
        config = step.config
        
        operation = config.get("operation", "transform")
        input_data = context.data.get("input_data", {})
        
        if operation == "transform":
            # Transform data based on config
            transformed_data = self._transform_data(input_data, config.get("transform_rules", {}))
            return {"transformed_data": transformed_data}
        
        elif operation == "filter":
            # Filter data based on criteria
            filtered_data = self._filter_data(input_data, config.get("filter_criteria", {}))
            return {"filtered_data": filtered_data}
        
        elif operation == "aggregate":
            # Aggregate data
            aggregated_data = self._aggregate_data(input_data, config.get("aggregation_rules", {}))
            return {"aggregated_data": aggregated_data}
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate data processing config"""
        required_fields = ["operation"]
        return all(field in config for field in required_fields)
    
    def _transform_data(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data based on rules"""
        # Simple transformation logic
        transformed = {}
        for key, value in data.items():
            if key in rules:
                transform_rule = rules[key]
                if transform_rule.get("type") == "uppercase":
                    transformed[key] = str(value).upper()
                elif transform_rule.get("type") == "lowercase":
                    transformed[key] = str(value).lower()
                else:
                    transformed[key] = value
            else:
                transformed[key] = value
        return transformed
    
    def _filter_data(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data based on criteria"""
        filtered = {}
        for key, value in data.items():
            if key in criteria:
                filter_criteria = criteria[key]
                if self._evaluate_condition(value, filter_criteria):
                    filtered[key] = value
            else:
                filtered[key] = value
        return filtered
    
    def _aggregate_data(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data based on rules"""
        aggregated = {}
        for key, rule in rules.items():
            if rule.get("type") == "sum":
                aggregated[key] = sum(data.get(key, []))
            elif rule.get("type") == "count":
                aggregated[key] = len(data.get(key, []))
            elif rule.get("type") == "average":
                values = data.get(key, [])
                aggregated[key] = sum(values) / len(values) if values else 0
        return aggregated
    
    def _evaluate_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate a condition"""
        operator = condition.get("operator", "equals")
        expected_value = condition.get("value")
        
        if operator == "equals":
            return value == expected_value
        elif operator == "not_equals":
            return value != expected_value
        elif operator == "greater_than":
            return value > expected_value
        elif operator == "less_than":
            return value < expected_value
        elif operator == "contains":
            return expected_value in str(value)
        elif operator == "not_contains":
            return expected_value not in str(value)
        elif operator == "is_empty":
            return not value or value == ""
        elif operator == "is_not_empty":
            return bool(value) and value != ""
        
        return False


class ValidationExecutor(StepExecutor):
    """Validation step executor"""
    
    async def execute(self, step: WorkflowStep, context: WorkflowContext) -> Dict[str, Any]:
        """Execute validation step"""
        config = step.config
        
        validation_rules = config.get("validation_rules", [])
        data_to_validate = context.data.get("data_to_validate", {})
        
        validation_results = []
        is_valid = True
        
        for rule in validation_rules:
            field = rule.get("field")
            rule_type = rule.get("type")
            expected = rule.get("expected")
            
            if field not in data_to_validate:
                validation_results.append({
                    "field": field,
                    "status": "error",
                    "message": f"Field '{field}' not found"
                })
                is_valid = False
                continue
            
            value = data_to_validate[field]
            rule_result = self._validate_field(value, rule_type, expected)
            
            validation_results.append({
                "field": field,
                "status": "pass" if rule_result else "fail",
                "message": rule.get("message", f"Validation {rule_type} failed")
            })
            
            if not rule_result:
                is_valid = False
        
        return {
            "is_valid": is_valid,
            "validation_results": validation_results,
            "validated_fields": len(validation_rules)
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate validation config"""
        required_fields = ["validation_rules"]
        return all(field in config for field in required_fields)
    
    def _validate_field(self, value: Any, rule_type: str, expected: Any) -> bool:
        """Validate a single field"""
        if rule_type == "required":
            return value is not None and value != ""
        elif rule_type == "min_length":
            return len(str(value)) >= expected
        elif rule_type == "max_length":
            return len(str(value)) <= expected
        elif rule_type == "email":
            return "@" in str(value) and "." in str(value)
        elif rule_type == "number":
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        elif rule_type == "regex":
            import re
            return bool(re.match(expected, str(value)))
        
        return True


class WorkflowEngine:
    """Advanced Workflow Engine"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.step_executors: Dict[StepType, StepExecutor] = {}
        self._initialize_executors()
        self._load_default_workflows()
    
    def _initialize_executors(self):
        """Initialize step executors"""
        self.step_executors[StepType.AI_GENERATION] = AIGenerationExecutor()
        self.step_executors[StepType.DATA_PROCESSING] = DataProcessingExecutor()
        self.step_executors[StepType.VALIDATION] = ValidationExecutor()
    
    def _load_default_workflows(self):
        """Load default workflow templates"""
        default_workflows = [
            self._create_document_generation_workflow(),
            self._create_business_plan_workflow(),
            self._create_marketing_campaign_workflow(),
            self._create_financial_report_workflow(),
        ]
        
        for workflow in default_workflows:
            self.workflows[workflow.id] = workflow
        
        logger.info(f"Loaded {len(default_workflows)} default workflows")
    
    def _create_document_generation_workflow(self) -> WorkflowDefinition:
        """Create document generation workflow"""
        return WorkflowDefinition(
            id="document_generation_v1",
            name="Document Generation Pipeline",
            description="Complete document generation workflow with validation and formatting",
            steps=[
                WorkflowStep(
                    id="validate_input",
                    name="Validate Input Data",
                    type=StepType.VALIDATION,
                    description="Validate input data and requirements",
                    config={
                        "validation_rules": [
                            {"field": "document_type", "type": "required", "message": "Document type is required"},
                            {"field": "content_requirements", "type": "required", "message": "Content requirements are required"}
                        ]
                    }
                ),
                WorkflowStep(
                    id="generate_content",
                    name="Generate Content",
                    type=StepType.AI_GENERATION,
                    description="Generate document content using AI",
                    dependencies=["validate_input"],
                    config={
                        "prompt_template": "Generate a {document_type} document with the following requirements: {content_requirements}",
                        "model_id": "openai_gpt4",
                        "max_tokens": 4000,
                        "temperature": 0.7
                    }
                ),
                WorkflowStep(
                    id="format_document",
                    name="Format Document",
                    type=StepType.DATA_PROCESSING,
                    description="Format the generated content",
                    dependencies=["generate_content"],
                    config={
                        "operation": "transform",
                        "transform_rules": {
                            "content": {"type": "format_document"}
                        }
                    }
                ),
                WorkflowStep(
                    id="final_validation",
                    name="Final Validation",
                    type=StepType.VALIDATION,
                    description="Validate the final document",
                    dependencies=["format_document"],
                    config={
                        "validation_rules": [
                            {"field": "content", "type": "min_length", "expected": 100, "message": "Content too short"},
                            {"field": "format", "type": "required", "message": "Format validation required"}
                        ]
                    }
                )
            ],
            variables={
                "document_type": "business_plan",
                "content_requirements": "comprehensive business plan with financial projections"
            },
            created_by="system",
            tags=["document", "generation", "ai", "validation"]
        )
    
    def _create_business_plan_workflow(self) -> WorkflowDefinition:
        """Create business plan workflow"""
        return WorkflowDefinition(
            id="business_plan_v1",
            name="Business Plan Generation",
            description="Comprehensive business plan generation workflow",
            steps=[
                WorkflowStep(
                    id="collect_requirements",
                    name="Collect Requirements",
                    type=StepType.DATA_PROCESSING,
                    description="Collect and organize business requirements",
                    config={
                        "operation": "transform",
                        "transform_rules": {
                            "company_info": {"type": "organize"},
                            "market_data": {"type": "structure"}
                        }
                    }
                ),
                WorkflowStep(
                    id="generate_executive_summary",
                    name="Generate Executive Summary",
                    type=StepType.AI_GENERATION,
                    dependencies=["collect_requirements"],
                    config={
                        "prompt_template": "Create an executive summary for {company_name} in the {industry} industry",
                        "model_id": "openai_gpt4"
                    }
                ),
                WorkflowStep(
                    id="generate_market_analysis",
                    name="Generate Market Analysis",
                    type=StepType.AI_GENERATION,
                    dependencies=["collect_requirements"],
                    config={
                        "prompt_template": "Analyze the {industry} market for {company_name}",
                        "model_id": "openai_gpt4"
                    }
                ),
                WorkflowStep(
                    id="generate_financial_projections",
                    name="Generate Financial Projections",
                    type=StepType.AI_GENERATION,
                    dependencies=["collect_requirements"],
                    config={
                        "prompt_template": "Create 3-year financial projections for {company_name}",
                        "model_id": "openai_gpt4"
                    }
                ),
                WorkflowStep(
                    id="compile_business_plan",
                    name="Compile Business Plan",
                    type=StepType.DATA_PROCESSING,
                    dependencies=["generate_executive_summary", "generate_market_analysis", "generate_financial_projections"],
                    config={
                        "operation": "aggregate",
                        "aggregation_rules": {
                            "sections": {"type": "combine"}
                        }
                    }
                )
            ],
            variables={
                "company_name": "Your Company",
                "industry": "technology"
            },
            created_by="system",
            tags=["business", "plan", "financial", "analysis"]
        )
    
    def _create_marketing_campaign_workflow(self) -> WorkflowDefinition:
        """Create marketing campaign workflow"""
        return WorkflowDefinition(
            id="marketing_campaign_v1",
            name="Marketing Campaign Creation",
            description="Complete marketing campaign creation workflow",
            steps=[
                WorkflowStep(
                    id="analyze_audience",
                    name="Analyze Target Audience",
                    type=StepType.AI_GENERATION,
                    config={
                        "prompt_template": "Analyze the target audience for {product_name} in {market_segment}",
                        "model_id": "openai_gpt4"
                    }
                ),
                WorkflowStep(
                    id="create_campaign_strategy",
                    name="Create Campaign Strategy",
                    type=StepType.AI_GENERATION,
                    dependencies=["analyze_audience"],
                    config={
                        "prompt_template": "Create a marketing strategy for {product_name} targeting {audience}",
                        "model_id": "openai_gpt4"
                    }
                ),
                WorkflowStep(
                    id="generate_content",
                    name="Generate Marketing Content",
                    type=StepType.AI_GENERATION,
                    dependencies=["create_campaign_strategy"],
                    config={
                        "prompt_template": "Generate marketing content for {campaign_strategy}",
                        "model_id": "openai_gpt4"
                    }
                ),
                WorkflowStep(
                    id="create_budget_plan",
                    name="Create Budget Plan",
                    type=StepType.DATA_PROCESSING,
                    dependencies=["create_campaign_strategy"],
                    config={
                        "operation": "transform",
                        "transform_rules": {
                            "budget_allocation": {"type": "calculate"}
                        }
                    }
                )
            ],
            variables={
                "product_name": "Your Product",
                "market_segment": "B2B"
            },
            created_by="system",
            tags=["marketing", "campaign", "strategy", "content"]
        )
    
    def _create_financial_report_workflow(self) -> WorkflowDefinition:
        """Create financial report workflow"""
        return WorkflowDefinition(
            id="financial_report_v1",
            name="Financial Report Generation",
            description="Comprehensive financial report generation workflow",
            steps=[
                WorkflowStep(
                    id="collect_financial_data",
                    name="Collect Financial Data",
                    type=StepType.DATA_PROCESSING,
                    config={
                        "operation": "aggregate",
                        "aggregation_rules": {
                            "revenue": {"type": "sum"},
                            "expenses": {"type": "sum"},
                            "profit": {"type": "calculate"}
                        }
                    }
                ),
                WorkflowStep(
                    id="analyze_performance",
                    name="Analyze Financial Performance",
                    type=StepType.AI_GENERATION,
                    dependencies=["collect_financial_data"],
                    config={
                        "prompt_template": "Analyze the financial performance for {company_name} with revenue of ${revenue} and expenses of ${expenses}",
                        "model_id": "openai_gpt4"
                    }
                ),
                WorkflowStep(
                    id="generate_recommendations",
                    name="Generate Recommendations",
                    type=StepType.AI_GENERATION,
                    dependencies=["analyze_performance"],
                    config={
                        "prompt_template": "Provide strategic recommendations based on the financial analysis",
                        "model_id": "openai_gpt4"
                    }
                ),
                WorkflowStep(
                    id="format_report",
                    name="Format Report",
                    type=StepType.DATA_PROCESSING,
                    dependencies=["analyze_performance", "generate_recommendations"],
                    config={
                        "operation": "transform",
                        "transform_rules": {
                            "report_sections": {"type": "organize"}
                        }
                    }
                )
            ],
            variables={
                "company_name": "Your Company",
                "revenue": 1000000,
                "expenses": 750000
            },
            created_by="system",
            tags=["finance", "report", "analysis", "recommendations"]
        )
    
    async def create_workflow(self, workflow_data: Dict[str, Any]) -> WorkflowDefinition:
        """Create a new workflow"""
        workflow = WorkflowDefinition(**workflow_data)
        self.workflows[workflow.id] = workflow
        
        logger.info(f"Created workflow {workflow.id}")
        return workflow
    
    async def execute_workflow(
        self,
        workflow_id: str,
        user_id: str,
        context: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Execute a workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            user_id=user_id,
            context=context or {},
            status=WorkflowStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        self.executions[execution_id] = execution
        
        try:
            await self._execute_workflow_steps(workflow, execution)
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            execution.progress = 1.0
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.errors.append(str(e))
            execution.completed_at = datetime.utcnow()
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            logger.error(f"Workflow execution failed: {e}")
        
        return execution
    
    async def _execute_workflow_steps(
        self,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution
    ):
        """Execute workflow steps"""
        workflow_context = WorkflowContext(
            workflow_id=workflow.id,
            execution_id=execution.id,
            user_id=execution.user_id,
            data=execution.context,
            variables={**workflow.variables, **execution.context},
            start_time=execution.started_at
        )
        
        # Create step lookup
        step_lookup = {step.id: step for step in workflow.steps}
        
        # Track completed steps
        completed_steps = set()
        
        while len(completed_steps) < len(workflow.steps):
            # Find steps that can be executed
            executable_steps = []
            
            for step in workflow.steps:
                if step.id in completed_steps:
                    continue
                
                # Check if all dependencies are completed
                if all(dep in completed_steps for dep in step.dependencies):
                    # Check conditions
                    if await self._evaluate_step_conditions(step, workflow_context):
                        executable_steps.append(step)
            
            if not executable_steps:
                # Check if we're stuck
                remaining_steps = [s for s in workflow.steps if s.id not in completed_steps]
                if remaining_steps:
                    raise Exception(f"No executable steps found. Remaining: {[s.id for s in remaining_steps]}")
                break
            
            # Execute steps (parallel if possible)
            if len(executable_steps) == 1:
                await self._execute_step(executable_steps[0], workflow_context, execution)
                completed_steps.add(executable_steps[0].id)
            else:
                # Execute parallel steps
                tasks = [
                    self._execute_step(step, workflow_context, execution)
                    for step in executable_steps
                ]
                await asyncio.gather(*tasks)
                completed_steps.update(step.id for step in executable_steps)
            
            # Update progress
            execution.progress = len(completed_steps) / len(workflow.steps)
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
        execution: WorkflowExecution
    ):
        """Execute a single workflow step"""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow()
        context.current_step = step.id
        execution.current_step = step.id
        
        try:
            # Get executor for step type
            executor = self.step_executors.get(step.type)
            if not executor:
                raise Exception(f"No executor found for step type: {step.type}")
            
            # Validate step configuration
            if not executor.validate_config(step.config):
                raise Exception(f"Invalid configuration for step {step.id}")
            
            # Execute step
            result = await executor.execute(step, context)
            
            step.status = StepStatus.COMPLETED
            step.result = result
            step.completed_at = datetime.utcnow()
            step.execution_time = (step.completed_at - step.started_at).total_seconds()
            
            # Store result in execution
            execution.step_results[step.id] = result
            execution.results[step.id] = result
            
            logger.info(f"Step {step.id} completed successfully")
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            step.execution_time = (step.completed_at - step.started_at).total_seconds()
            
            execution.errors.append(f"Step {step.id} failed: {str(e)}")
            execution.step_results[step.id] = {"error": str(e)}
            
            logger.error(f"Step {step.id} failed: {e}")
            
            # Check retry configuration
            retry_config = step.retry_config
            if retry_config.get("enabled", False):
                max_retries = retry_config.get("max_retries", 3)
                if context.retry_count < max_retries:
                    context.retry_count += 1
                    logger.info(f"Retrying step {step.id} (attempt {context.retry_count})")
                    await asyncio.sleep(retry_config.get("delay", 1))
                    await self._execute_step(step, context, execution)
                else:
                    raise e
            else:
                raise e
    
    async def _evaluate_step_conditions(
        self,
        step: WorkflowStep,
        context: WorkflowContext
    ) -> bool:
        """Evaluate step execution conditions"""
        if not step.conditions:
            return True
        
        for condition in step.conditions:
            field_value = context.variables.get(condition.field)
            
            if not self._evaluate_condition(field_value, condition):
                return False
        
        return True
    
    def _evaluate_condition(self, value: Any, condition: Condition) -> bool:
        """Evaluate a single condition"""
        if condition.operator == ConditionOperator.EQUALS:
            return value == condition.value
        elif condition.operator == ConditionOperator.NOT_EQUALS:
            return value != condition.value
        elif condition.operator == ConditionOperator.GREATER_THAN:
            return value > condition.value
        elif condition.operator == ConditionOperator.LESS_THAN:
            return value < condition.value
        elif condition.operator == ConditionOperator.CONTAINS:
            return condition.value in str(value)
        elif condition.operator == ConditionOperator.NOT_CONTAINS:
            return condition.value not in str(value)
        elif condition.operator == ConditionOperator.IS_EMPTY:
            return not value or value == ""
        elif condition.operator == ConditionOperator.IS_NOT_EMPTY:
            return bool(value) and value != ""
        elif condition.operator == ConditionOperator.REGEX_MATCH:
            import re
            return bool(re.match(condition.value, str(value)))
        
        return True
    
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status"""
        return self.executions.get(execution_id)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution"""
        execution = self.executions.get(execution_id)
        if not execution:
            return False
        
        if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            return False
        
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        execution.duration = (execution.completed_at - execution.started_at).total_seconds()
        
        logger.info(f"Cancelled execution {execution_id}")
        return True
    
    async def get_workflow_analytics(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow analytics"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Get executions for this workflow
        workflow_executions = [
            exec for exec in self.executions.values()
            if exec.workflow_id == workflow_id
        ]
        
        if not workflow_executions:
            return {
                "workflow_id": workflow_id,
                "total_executions": 0,
                "success_rate": 0,
                "avg_duration": 0,
                "step_analytics": {}
            }
        
        # Calculate analytics
        total_executions = len(workflow_executions)
        successful_executions = len([e for e in workflow_executions if e.status == WorkflowStatus.COMPLETED])
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        durations = [e.duration for e in workflow_executions if e.duration is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Step analytics
        step_analytics = {}
        for step in workflow.steps:
            step_results = [e.step_results.get(step.id, {}) for e in workflow_executions]
            step_analytics[step.id] = {
                "execution_count": len([r for r in step_results if r]),
                "error_count": len([r for r in step_results if "error" in r]),
                "avg_execution_time": 0  # Would need to track this
            }
        
        return {
            "workflow_id": workflow_id,
            "total_executions": total_executions,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "step_analytics": step_analytics,
            "recent_executions": [
                {
                    "id": e.id,
                    "status": e.status.value,
                    "duration": e.duration,
                    "started_at": e.started_at.isoformat() if e.started_at else None
                }
                for e in workflow_executions[-10:]  # Last 10 executions
            ]
        }
    
    async def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None
    ) -> List[WorkflowDefinition]:
        """List workflows with optional filtering"""
        workflows = list(self.workflows.values())
        
        if status:
            workflows = [w for w in workflows if w.status == status]
        
        if tags:
            workflows = [w for w in workflows if any(tag in w.tags for tag in tags)]
        
        if created_by:
            workflows = [w for w in workflows if w.created_by == created_by]
        
        return workflows


# Global workflow engine instance
workflow_engine = WorkflowEngine()















