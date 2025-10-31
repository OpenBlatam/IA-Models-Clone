"""
Advanced Workflow Engine for Export IA
======================================

Comprehensive workflow automation system that orchestrates AI-enhanced document
processing, cosmic transcendence, blockchain verification, and real-time streaming.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import os
from pathlib import Path
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import websockets
import sseclient
import redis
from celery import Celery
import kombu

# Import all our components
from ..ai_enhanced.ai_export_engine import (
    AIEnhancedExportEngine, AIEnhancementConfig, AIEnhancementLevel
)
from ..cosmic_transcendence.cosmic_transcendence_engine import (
    CosmicTranscendenceEngine, CosmicConfiguration, TranscendenceLevel
)
from ..blockchain.document_verifier import (
    BlockchainDocumentVerifier, BlockchainConfig, DocumentIntegrityLevel
)
from ..export_ia_engine import ExportFormat, DocumentType, QualityLevel
from ..styling.professional_styler import ProfessionalStyler
from ..quality.quality_validator import QualityValidator

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class WorkflowStepType(Enum):
    """Types of workflow steps."""
    AI_ENHANCEMENT = "ai_enhancement"
    COSMIC_TRANSCENDENCE = "cosmic_transcendence"
    BLOCKCHAIN_VERIFICATION = "blockchain_verification"
    DOCUMENT_EXPORT = "document_export"
    QUALITY_VALIDATION = "quality_validation"
    STYLE_APPLICATION = "style_application"
    NOTIFICATION = "notification"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    CUSTOM = "custom"

class WorkflowPriority(Enum):
    """Workflow priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"

@dataclass
class WorkflowStep:
    """Individual workflow step definition."""
    id: str
    name: str
    step_type: WorkflowStepType
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 3
    parallel: bool = False
    condition: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None

@dataclass
class WorkflowExecution:
    """Workflow execution instance."""
    id: str
    workflow_id: str
    status: WorkflowStatus
    current_step: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    created_by: str = "system"
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    progress: float = 0.0

@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 3600  # 1 hour default
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    notifications: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class AdvancedWorkflowEngine:
    """Advanced workflow automation engine."""
    
    def __init__(self):
        # Initialize components
        self.ai_engine = AIEnhancedExportEngine()
        self.cosmic_engine = CosmicTranscendenceEngine()
        self.blockchain_verifier = BlockchainDocumentVerifier()
        self.styler = ProfessionalStyler()
        self.validator = QualityValidator()
        
        # Workflow management
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        self.step_handlers: Dict[WorkflowStepType, Callable] = {}
        
        # Real-time features
        self.redis_client = None
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.sse_clients: Dict[str, sseclient.SSEClient] = {}
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.celery_app = None
        
        # Initialize workflow system
        self._initialize_workflow_system()
        self._register_step_handlers()
        self._load_default_workflows()
        
        logger.info("Advanced Workflow Engine initialized")
    
    def _initialize_workflow_system(self):
        """Initialize workflow system components."""
        try:
            # Initialize Redis for real-time features
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
            
            # Initialize Celery for background processing
            try:
                self.celery_app = Celery('export_ia_workflows')
                self.celery_app.config_from_object({
                    'broker_url': 'redis://localhost:6379/1',
                    'result_backend': 'redis://localhost:6379/1',
                    'task_serializer': 'json',
                    'accept_content': ['json'],
                    'result_serializer': 'json',
                    'timezone': 'UTC',
                    'enable_utc': True,
                })
                logger.info("Celery initialized for background processing")
            except Exception as e:
                logger.warning(f"Celery initialization failed: {e}")
                self.celery_app = None
                
        except Exception as e:
            logger.error(f"Workflow system initialization failed: {e}")
    
    def _register_step_handlers(self):
        """Register step handlers for different workflow step types."""
        self.step_handlers = {
            WorkflowStepType.AI_ENHANCEMENT: self._handle_ai_enhancement,
            WorkflowStepType.COSMIC_TRANSCENDENCE: self._handle_cosmic_transcendence,
            WorkflowStepType.BLOCKCHAIN_VERIFICATION: self._handle_blockchain_verification,
            WorkflowStepType.DOCUMENT_EXPORT: self._handle_document_export,
            WorkflowStepType.QUALITY_VALIDATION: self._handle_quality_validation,
            WorkflowStepType.STYLE_APPLICATION: self._handle_style_application,
            WorkflowStepType.NOTIFICATION: self._handle_notification,
            WorkflowStepType.CONDITION: self._handle_condition,
            WorkflowStepType.PARALLEL: self._handle_parallel_execution,
            WorkflowStepType.SEQUENTIAL: self._handle_sequential_execution,
        }
    
    def _load_default_workflows(self):
        """Load default workflow definitions."""
        # AI-Enhanced Document Processing Workflow
        ai_workflow = WorkflowDefinition(
            id="ai_enhanced_document_processing",
            name="AI-Enhanced Document Processing",
            description="Complete AI-powered document processing workflow",
            version="1.0.0",
            steps=[
                WorkflowStep(
                    id="content_analysis",
                    name="AI Content Analysis",
                    step_type=WorkflowStepType.AI_ENHANCEMENT,
                    config={
                        "enhancement_level": "advanced",
                        "optimization_modes": ["grammar_correction", "style_enhancement", "readability_improvement"]
                    }
                ),
                WorkflowStep(
                    id="quality_validation",
                    name="Quality Validation",
                    step_type=WorkflowStepType.QUALITY_VALIDATION,
                    config={
                        "validation_level": "strict",
                        "min_quality_score": 0.8
                    },
                    dependencies=["content_analysis"]
                ),
                WorkflowStep(
                    id="style_application",
                    name="Professional Style Application",
                    step_type=WorkflowStepType.STYLE_APPLICATION,
                    config={
                        "style_level": "premium",
                        "custom_branding": True
                    },
                    dependencies=["quality_validation"]
                ),
                WorkflowStep(
                    id="document_export",
                    name="Document Export",
                    step_type=WorkflowStepType.DOCUMENT_EXPORT,
                    config={
                        "formats": ["pdf", "docx", "html"],
                        "quality_level": "professional"
                    },
                    dependencies=["style_application"]
                )
            ]
        )
        
        # Cosmic Transcendence Workflow
        cosmic_workflow = WorkflowDefinition(
            id="cosmic_transcendence_processing",
            name="Cosmic Transcendence Document Processing",
            description="Transcend documents to cosmic levels of perfection",
            version="1.0.0",
            steps=[
                WorkflowStep(
                    id="cosmic_analysis",
                    name="Cosmic Dimensional Analysis",
                    step_type=WorkflowStepType.COSMIC_TRANSCENDENCE,
                    config={
                        "transcendence_level": "cosmic",
                        "enable_energy_flow": True,
                        "enable_harmonic_resonance": True
                    }
                ),
                WorkflowStep(
                    id="cosmic_enhancement",
                    name="Cosmic Content Enhancement",
                    step_type=WorkflowStepType.AI_ENHANCEMENT,
                    config={
                        "enhancement_level": "enterprise",
                        "cosmic_optimization": True
                    },
                    dependencies=["cosmic_analysis"]
                ),
                WorkflowStep(
                    id="cosmic_verification",
                    name="Cosmic Blockchain Verification",
                    step_type=WorkflowStepType.BLOCKCHAIN_VERIFICATION,
                    config={
                        "integrity_level": "enterprise",
                        "network": "ethereum"
                    },
                    dependencies=["cosmic_enhancement"]
                )
            ]
        )
        
        # Enterprise Document Workflow
        enterprise_workflow = WorkflowDefinition(
            id="enterprise_document_workflow",
            name="Enterprise Document Workflow",
            description="Complete enterprise-grade document processing with all features",
            version="1.0.0",
            steps=[
                WorkflowStep(
                    id="parallel_processing",
                    name="Parallel AI and Cosmic Processing",
                    step_type=WorkflowStepType.PARALLEL,
                    config={
                        "parallel_steps": ["ai_enhancement", "cosmic_analysis"]
                    }
                ),
                WorkflowStep(
                    id="ai_enhancement",
                    name="AI Enhancement",
                    step_type=WorkflowStepType.AI_ENHANCEMENT,
                    config={
                        "enhancement_level": "enterprise"
                    },
                    parallel=True
                ),
                WorkflowStep(
                    id="cosmic_analysis",
                    name="Cosmic Analysis",
                    step_type=WorkflowStepType.COSMIC_TRANSCENDENCE,
                    config={
                        "transcendence_level": "transcendent"
                    },
                    parallel=True
                ),
                WorkflowStep(
                    id="quality_validation",
                    name="Comprehensive Quality Validation",
                    step_type=WorkflowStepType.QUALITY_VALIDATION,
                    config={
                        "validation_level": "enterprise",
                        "min_quality_score": 0.9
                    },
                    dependencies=["parallel_processing"]
                ),
                WorkflowStep(
                    id="blockchain_verification",
                    name="Blockchain Verification",
                    step_type=WorkflowStepType.BLOCKCHAIN_VERIFICATION,
                    config={
                        "integrity_level": "enterprise"
                    },
                    dependencies=["quality_validation"]
                ),
                WorkflowStep(
                    id="final_export",
                    name="Final Document Export",
                    step_type=WorkflowStepType.DOCUMENT_EXPORT,
                    config={
                        "formats": ["pdf", "docx", "html", "json"],
                        "quality_level": "enterprise"
                    },
                    dependencies=["blockchain_verification"]
                )
            ]
        )
        
        # Store workflows
        self.workflow_definitions[ai_workflow.id] = ai_workflow
        self.workflow_definitions[cosmic_workflow.id] = cosmic_workflow
        self.workflow_definitions[enterprise_workflow.id] = enterprise_workflow
        
        logger.info(f"Loaded {len(self.workflow_definitions)} default workflows")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        priority: WorkflowPriority = WorkflowPriority.NORMAL,
        created_by: str = "system"
    ) -> str:
        """Execute a workflow with given input data."""
        
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow_def = self.workflow_definitions[workflow_id]
        execution_id = str(uuid.uuid4())
        
        # Create workflow execution
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            created_by=created_by,
            priority=priority,
            context=input_data
        )
        
        self.workflow_executions[execution_id] = execution
        
        # Start workflow execution
        if self.celery_app:
            # Use Celery for background execution
            self.celery_app.send_task(
                'execute_workflow_task',
                args=[execution_id, workflow_id, input_data],
                priority=self._get_celery_priority(priority)
            )
        else:
            # Execute directly
            asyncio.create_task(self._execute_workflow_async(execution_id))
        
        logger.info(f"Workflow execution started: {execution_id}")
        return execution_id
    
    async def _execute_workflow_async(self, execution_id: str):
        """Execute workflow asynchronously."""
        execution = self.workflow_executions[execution_id]
        workflow_def = self.workflow_definitions[execution.workflow_id]
        
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now()
            
            # Execute workflow steps
            await self._execute_workflow_steps(execution, workflow_def)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.progress = 100.0
            
            logger.info(f"Workflow execution completed: {execution_id}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.errors.append(str(e))
            execution.completed_at = datetime.now()
            
            logger.error(f"Workflow execution failed: {execution_id}, error: {e}")
            logger.error(traceback.format_exc())
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution, workflow_def: WorkflowDefinition):
        """Execute workflow steps."""
        completed_steps = set()
        total_steps = len(workflow_def.steps)
        
        while len(completed_steps) < total_steps:
            # Find steps that can be executed
            ready_steps = []
            for step in workflow_def.steps:
                if step.id not in completed_steps:
                    # Check dependencies
                    if all(dep in completed_steps for dep in step.dependencies):
                        ready_steps.append(step)
            
            if not ready_steps:
                # Check for circular dependencies or missing steps
                remaining_steps = [s.id for s in workflow_def.steps if s.id not in completed_steps]
                raise ValueError(f"No ready steps found. Remaining: {remaining_steps}")
            
            # Execute ready steps
            if len(ready_steps) == 1:
                # Sequential execution
                step = ready_steps[0]
                await self._execute_step(execution, step)
                completed_steps.add(step.id)
            else:
                # Check for parallel execution
                parallel_steps = [s for s in ready_steps if s.parallel]
                if parallel_steps:
                    # Execute parallel steps
                    await self._execute_parallel_steps(execution, parallel_steps)
                    completed_steps.update(s.id for s in parallel_steps)
                else:
                    # Execute first ready step
                    step = ready_steps[0]
                    await self._execute_step(execution, step)
                    completed_steps.add(step.id)
            
            # Update progress
            execution.progress = (len(completed_steps) / total_steps) * 100
            
            # Publish progress update
            await self._publish_progress_update(execution)
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep):
        """Execute a single workflow step."""
        try:
            execution.current_step = step.id
            
            # Get step handler
            handler = self.step_handlers.get(step.step_type)
            if not handler:
                raise ValueError(f"No handler for step type: {step.step_type}")
            
            # Execute step with timeout
            result = await asyncio.wait_for(
                handler(execution, step),
                timeout=step.timeout
            )
            
            # Store result
            execution.results[step.id] = result
            
            logger.info(f"Step completed: {step.id}")
            
        except asyncio.TimeoutError:
            raise Exception(f"Step timed out: {step.id}")
        except Exception as e:
            # Retry logic
            if step.retry_count < step.max_retries:
                step.retry_count += 1
                logger.info(f"Retrying step {step.id} (attempt {step.retry_count + 1})")
                await asyncio.sleep(2 ** step.retry_count)  # Exponential backoff
                await self._execute_step(execution, step)
            else:
                raise Exception(f"Step failed after {step.max_retries} retries: {step.id}, error: {e}")
    
    async def _execute_parallel_steps(self, execution: WorkflowExecution, steps: List[WorkflowStep]):
        """Execute multiple steps in parallel."""
        tasks = []
        for step in steps:
            task = asyncio.create_task(self._execute_step(execution, step))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Step handlers
    async def _handle_ai_enhancement(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle AI enhancement step."""
        config = step.config
        content = execution.context.get("content", "")
        
        # Perform AI enhancement
        enhancement_level = AIEnhancementLevel(config.get("enhancement_level", "standard"))
        optimization_modes = config.get("optimization_modes", [])
        
        enhanced_content = await self.ai_engine.optimize_content(content, optimization_modes)
        quality_analysis = await self.ai_engine.analyze_content_quality(enhanced_content)
        
        return {
            "enhanced_content": enhanced_content,
            "quality_analysis": {
                "overall_score": quality_analysis.overall_score,
                "readability_score": quality_analysis.readability_score,
                "professional_tone_score": quality_analysis.professional_tone_score,
                "grammar_score": quality_analysis.grammar_score,
                "style_score": quality_analysis.style_score
            },
            "suggestions": quality_analysis.suggestions
        }
    
    async def _handle_cosmic_transcendence(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle cosmic transcendence step."""
        config = step.config
        title = execution.context.get("title", "Untitled Document")
        content = execution.context.get("content", "")
        
        # Perform cosmic transcendence
        transcendence_level = TranscendenceLevel(config.get("transcendence_level", "transcendent"))
        
        transcendent_doc = await self.cosmic_engine.transcend_document(
            title=title,
            content=content,
            target_transcendence=transcendence_level
        )
        
        return {
            "transcendent_document": {
                "id": transcendent_doc.id,
                "title": transcendent_doc.title,
                "content": transcendent_doc.content,
                "transcendence_level": transcendent_doc.transcendence_level.value,
                "overall_transcendence": transcendent_doc.overall_transcendence,
                "dimensional_scores": {
                    dim.value: score for dim, score in transcendent_doc.dimensional_scores.items()
                }
            }
        }
    
    async def _handle_blockchain_verification(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle blockchain verification step."""
        config = step.config
        content = execution.context.get("content", "")
        metadata = execution.context.get("metadata", {})
        
        # Perform blockchain verification
        integrity_level = DocumentIntegrityLevel(config.get("integrity_level", "standard"))
        
        verification = await self.blockchain_verifier.verify_document(
            document_id=execution.id,
            content=content,
            metadata=metadata,
            integrity_level=integrity_level
        )
        
        return {
            "verification": {
                "id": verification.id,
                "status": verification.verification_status.value,
                "integrity_level": verification.integrity_level.value,
                "document_hash": verification.document_hash.combined_hash,
                "blockchain_transaction": verification.blockchain_transaction.transaction_hash if verification.blockchain_transaction else None
            }
        }
    
    async def _handle_document_export(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle document export step."""
        config = step.config
        formats = config.get("formats", ["pdf"])
        quality_level = config.get("quality_level", "professional")
        
        # Get content from previous steps
        content = execution.context.get("content", "")
        if "ai_enhancement" in execution.results:
            content = execution.results["ai_enhancement"]["enhanced_content"]
        elif "cosmic_transcendence" in execution.results:
            content = execution.results["cosmic_transcendence"]["transcendent_document"]["content"]
        
        # Export documents
        exported_files = {}
        for format_type in formats:
            # This would use the actual export engine
            # For now, create a placeholder
            filename = f"export_{execution.id}.{format_type}"
            exported_files[format_type] = {
                "filename": filename,
                "path": f"exports/{filename}",
                "size": len(content)
            }
        
        return {
            "exported_files": exported_files,
            "formats": formats,
            "quality_level": quality_level
        }
    
    async def _handle_quality_validation(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle quality validation step."""
        config = step.config
        validation_level = config.get("validation_level", "standard")
        min_quality_score = config.get("min_quality_score", 0.7)
        
        # Get content for validation
        content = execution.context.get("content", "")
        if "ai_enhancement" in execution.results:
            content = execution.results["ai_enhancement"]["enhanced_content"]
        
        # Perform quality validation
        quality_report = await self.validator.validate_document(
            content={"content": content, "sections": []},
            validation_level=validation_level
        )
        
        # Check if quality meets minimum requirements
        quality_passed = quality_report.overall_score >= min_quality_score
        
        return {
            "quality_report": {
                "overall_score": quality_report.overall_score,
                "passed_validation": quality_report.passed_validation,
                "quality_passed": quality_passed,
                "metrics": {
                    metric.value: score for metric, score in quality_report.metrics.items()
                },
                "recommendations": quality_report.recommendations
            }
        }
    
    async def _handle_style_application(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle style application step."""
        config = step.config
        style_level = config.get("style_level", "professional")
        custom_branding = config.get("custom_branding", False)
        
        # Get content for styling
        content = execution.context.get("content", "")
        if "ai_enhancement" in execution.results:
            content = execution.results["ai_enhancement"]["enhanced_content"]
        
        # Apply professional styling
        style = self.styler.get_style(style_level)
        if not style:
            style = self.styler.list_styles()[0]  # Use first available style
        
        styled_content = self.styler.apply_style_to_content(
            content={"content": content, "sections": []},
            style=style,
            format_type="html"
        )
        
        return {
            "styled_content": styled_content,
            "style_applied": style.name,
            "style_level": style_level,
            "custom_branding": custom_branding
        }
    
    async def _handle_notification(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle notification step."""
        config = step.config
        message = config.get("message", "Workflow step completed")
        recipients = config.get("recipients", [])
        
        # Send notifications (simplified)
        notification_result = {
            "message": message,
            "recipients": recipients,
            "sent_at": datetime.now().isoformat(),
            "status": "sent"
        }
        
        return notification_result
    
    async def _handle_condition(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle conditional step."""
        config = step.config
        condition = config.get("condition", "true")
        
        # Evaluate condition (simplified)
        condition_result = True  # Would implement actual condition evaluation
        
        return {
            "condition": condition,
            "result": condition_result,
            "evaluated_at": datetime.now().isoformat()
        }
    
    async def _handle_parallel_execution(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle parallel execution step."""
        config = step.config
        parallel_steps = config.get("parallel_steps", [])
        
        # Execute parallel steps
        results = {}
        for step_id in parallel_steps:
            # Find the step definition
            workflow_def = self.workflow_definitions[execution.workflow_id]
            parallel_step = next((s for s in workflow_def.steps if s.id == step_id), None)
            
            if parallel_step:
                result = await self._execute_step(execution, parallel_step)
                results[step_id] = result
        
        return {
            "parallel_results": results,
            "executed_steps": parallel_steps
        }
    
    async def _handle_sequential_execution(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        """Handle sequential execution step."""
        config = step.config
        sequential_steps = config.get("sequential_steps", [])
        
        # Execute steps sequentially
        results = {}
        for step_id in sequential_steps:
            # Find the step definition
            workflow_def = self.workflow_definitions[execution.workflow_id]
            sequential_step = next((s for s in workflow_def.steps if s.id == step_id), None)
            
            if sequential_step:
                result = await self._execute_step(execution, sequential_step)
                results[step_id] = result
        
        return {
            "sequential_results": results,
            "executed_steps": sequential_steps
        }
    
    async def _publish_progress_update(self, execution: WorkflowExecution):
        """Publish progress update to real-time clients."""
        if self.redis_client:
            try:
                update_data = {
                    "execution_id": execution.id,
                    "status": execution.status.value,
                    "progress": execution.progress,
                    "current_step": execution.current_step,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.redis_client.publish(
                    f"workflow_progress:{execution.id}",
                    json.dumps(update_data)
                )
            except Exception as e:
                logger.error(f"Failed to publish progress update: {e}")
    
    def _get_celery_priority(self, priority: WorkflowPriority) -> int:
        """Convert workflow priority to Celery priority."""
        priority_map = {
            WorkflowPriority.LOW: 0,
            WorkflowPriority.NORMAL: 5,
            WorkflowPriority.HIGH: 8,
            WorkflowPriority.CRITICAL: 9,
            WorkflowPriority.URGENT: 10
        }
        return priority_map.get(priority, 5)
    
    # Workflow management methods
    def create_workflow(self, workflow_def: WorkflowDefinition) -> str:
        """Create a new workflow definition."""
        self.workflow_definitions[workflow_def.id] = workflow_def
        logger.info(f"Workflow created: {workflow_def.id}")
        return workflow_def.id
    
    def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID."""
        return self.workflow_executions.get(execution_id)
    
    def list_workflow_executions(self, status: Optional[WorkflowStatus] = None) -> List[WorkflowExecution]:
        """List workflow executions with optional status filter."""
        executions = list(self.workflow_executions.values())
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return sorted(executions, key=lambda x: x.started_at, reverse=True)
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics."""
        executions = list(self.workflow_executions.values())
        
        if not executions:
            return {"message": "No workflow executions found"}
        
        status_counts = {}
        for status in WorkflowStatus:
            status_counts[status.value] = len([e for e in executions if e.status == status])
        
        priority_counts = {}
        for priority in WorkflowPriority:
            priority_counts[priority.value] = len([e for e in executions if e.priority == priority])
        
        return {
            "total_executions": len(executions),
            "status_distribution": status_counts,
            "priority_distribution": priority_counts,
            "active_workflows": len([e for e in executions if e.status == WorkflowStatus.RUNNING]),
            "completed_workflows": len([e for e in executions if e.status == WorkflowStatus.COMPLETED]),
            "failed_workflows": len([e for e in executions if e.status == WorkflowStatus.FAILED]),
            "average_execution_time": self._calculate_average_execution_time(executions)
        }
    
    def _calculate_average_execution_time(self, executions: List[WorkflowExecution]) -> float:
        """Calculate average execution time for completed workflows."""
        completed_executions = [
            e for e in executions 
            if e.status == WorkflowStatus.COMPLETED and e.completed_at
        ]
        
        if not completed_executions:
            return 0.0
        
        total_time = sum(
            (e.completed_at - e.started_at).total_seconds()
            for e in completed_executions
        )
        
        return total_time / len(completed_executions)

# Global workflow engine instance
_global_workflow_engine: Optional[AdvancedWorkflowEngine] = None

def get_global_workflow_engine() -> AdvancedWorkflowEngine:
    """Get the global workflow engine instance."""
    global _global_workflow_engine
    if _global_workflow_engine is None:
        _global_workflow_engine = AdvancedWorkflowEngine()
    return _global_workflow_engine



























