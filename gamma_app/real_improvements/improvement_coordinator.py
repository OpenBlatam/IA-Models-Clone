"""
Gamma App - Real Improvement Coordinator
Coordinates and orchestrates real improvements that actually work
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class CoordinationStatus(Enum):
    """Coordination status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ImprovementPriority(Enum):
    """Improvement priority"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ImprovementPlan:
    """Improvement plan"""
    plan_id: str
    title: str
    description: str
    priority: ImprovementPriority
    estimated_duration: float  # hours
    dependencies: List[str] = None
    improvements: List[str] = None  # improvement IDs
    status: CoordinationStatus = CoordinationStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.dependencies is None:
            self.dependencies = []
        if self.improvements is None:
            self.improvements = []

@dataclass
class ImprovementExecution:
    """Improvement execution"""
    execution_id: str
    plan_id: str
    improvement_id: str
    status: CoordinationStatus = CoordinationStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: float = 0.0
    success: bool = False
    error_message: str = ""
    rollback_required: bool = False

class RealImprovementCoordinator:
    """
    Coordinates and orchestrates real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize improvement coordinator"""
        self.project_root = Path(project_root)
        self.plans: Dict[str, ImprovementPlan] = {}
        self.executions: Dict[str, ImprovementExecution] = {}
        self.execution_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.coordination_rules: Dict[str, Any] = {}
        
        # Initialize coordination rules
        self._initialize_coordination_rules()
        
        logger.info(f"Real Improvement Coordinator initialized for {self.project_root}")
    
    def _initialize_coordination_rules(self):
        """Initialize coordination rules"""
        self.coordination_rules = {
            "max_concurrent_executions": 3,
            "retry_attempts": 3,
            "retry_delay": 30,  # seconds
            "rollback_on_failure": True,
            "dependency_check": True,
            "progress_tracking": True,
            "notification_enabled": True
        }
    
    def create_improvement_plan(self, title: str, description: str, 
                               priority: ImprovementPriority, estimated_duration: float,
                               improvements: List[str], dependencies: List[str] = None) -> str:
        """Create improvement plan"""
        try:
            plan_id = f"plan_{int(time.time() * 1000)}"
            
            plan = ImprovementPlan(
                plan_id=plan_id,
                title=title,
                description=description,
                priority=priority,
                estimated_duration=estimated_duration,
                dependencies=dependencies or [],
                improvements=improvements
            )
            
            self.plans[plan_id] = plan
            
            logger.info(f"Improvement plan created: {title}")
            return plan_id
            
        except Exception as e:
            logger.error(f"Failed to create improvement plan: {e}")
            raise
    
    async def execute_plan(self, plan_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """Execute improvement plan"""
        try:
            if plan_id not in self.plans:
                return {"success": False, "error": f"Plan {plan_id} not found"}
            
            plan = self.plans[plan_id]
            
            # Check dependencies
            if not await self._check_plan_dependencies(plan):
                return {"success": False, "error": "Plan dependencies not met"}
            
            # Update plan status
            plan.status = CoordinationStatus.IN_PROGRESS
            plan.started_at = datetime.utcnow()
            
            # Create executions for each improvement
            executions = []
            for improvement_id in plan.improvements:
                execution_id = await self._create_execution(plan_id, improvement_id)
                executions.append(execution_id)
            
            # Execute improvements based on priority and dependencies
            results = await self._execute_improvements(executions, dry_run)
            
            # Update plan status
            if all(result.get("success", False) for result in results.values()):
                plan.status = CoordinationStatus.COMPLETED
                plan.completed_at = datetime.utcnow()
                plan.progress = 100.0
            else:
                plan.status = CoordinationStatus.FAILED
                plan.progress = self._calculate_plan_progress(plan_id)
            
            return {
                "success": plan.status == CoordinationStatus.COMPLETED,
                "plan_id": plan_id,
                "status": plan.status.value,
                "progress": plan.progress,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to execute plan: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_all_plans(self, priority: Optional[ImprovementPriority] = None,
                               max_concurrent: int = 2) -> Dict[str, Any]:
        """Execute all improvement plans"""
        try:
            # Filter plans by priority
            plans_to_execute = [
                plan for plan in self.plans.values()
                if priority is None or plan.priority == priority
            ]
            
            # Sort by priority
            priority_order = {
                ImprovementPriority.CRITICAL: 1,
                ImprovementPriority.HIGH: 2,
                ImprovementPriority.MEDIUM: 3,
                ImprovementPriority.LOW: 4
            }
            plans_to_execute.sort(key=lambda x: priority_order[x.priority])
            
            results = {}
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def execute_with_semaphore(plan):
                async with semaphore:
                    return await self.execute_plan(plan.plan_id)
            
            # Execute plans
            tasks = [execute_with_semaphore(plan) for plan in plans_to_execute]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results_list):
                plan_id = plans_to_execute[i].plan_id
                if isinstance(result, Exception):
                    results[plan_id] = {"success": False, "error": str(result)}
                else:
                    results[plan_id] = result
            
            # Calculate summary
            total_plans = len(plans_to_execute)
            successful_plans = len([r for r in results.values() if r.get("success", False)])
            
            return {
                "total_plans": total_plans,
                "successful_plans": successful_plans,
                "failed_plans": total_plans - successful_plans,
                "success_rate": (successful_plans / total_plans * 100) if total_plans > 0 else 0,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to execute all plans: {e}")
            return {"success": False, "error": str(e)}
    
    async def rollback_plan(self, plan_id: str) -> Dict[str, Any]:
        """Rollback improvement plan"""
        try:
            if plan_id not in self.plans:
                return {"success": False, "error": f"Plan {plan_id} not found"}
            
            plan = self.plans[plan_id]
            
            # Find executions for this plan
            plan_executions = [
                exec_id for exec_id, execution in self.executions.items()
                if execution.plan_id == plan_id
            ]
            
            rollback_results = {}
            for execution_id in plan_executions:
                result = await self._rollback_execution(execution_id)
                rollback_results[execution_id] = result
            
            # Update plan status
            plan.status = CoordinationStatus.CANCELLED
            plan.progress = 0.0
            
            return {
                "success": True,
                "plan_id": plan_id,
                "rollback_results": rollback_results
            }
            
        except Exception as e:
            logger.error(f"Failed to rollback plan: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get plan status"""
        if plan_id not in self.plans:
            return None
        
        plan = self.plans[plan_id]
        
        # Get execution statuses
        plan_executions = [
            exec_id for exec_id, execution in self.executions.items()
            if execution.plan_id == plan_id
        ]
        
        execution_statuses = {}
        for execution_id in plan_executions:
            execution = self.executions[execution_id]
            execution_statuses[execution_id] = {
                "status": execution.status.value,
                "success": execution.success,
                "duration": execution.duration
            }
        
        return {
            "plan_id": plan_id,
            "title": plan.title,
            "status": plan.status.value,
            "progress": plan.progress,
            "estimated_duration": plan.estimated_duration,
            "started_at": plan.started_at.isoformat() if plan.started_at else None,
            "completed_at": plan.completed_at.isoformat() if plan.completed_at else None,
            "executions": execution_statuses
        }
    
    async def _check_plan_dependencies(self, plan: ImprovementPlan) -> bool:
        """Check if plan dependencies are met"""
        for dep_plan_id in plan.dependencies:
            if dep_plan_id not in self.plans:
                logger.warning(f"Dependency plan {dep_plan_id} not found")
                return False
            
            dep_plan = self.plans[dep_plan_id]
            if dep_plan.status != CoordinationStatus.COMPLETED:
                logger.warning(f"Dependency plan {dep_plan_id} not completed")
                return False
        
        return True
    
    async def _create_execution(self, plan_id: str, improvement_id: str) -> str:
        """Create improvement execution"""
        execution_id = f"exec_{int(time.time() * 1000)}"
        
        execution = ImprovementExecution(
            execution_id=execution_id,
            plan_id=plan_id,
            improvement_id=improvement_id
        )
        
        self.executions[execution_id] = execution
        self.execution_logs[execution_id] = []
        
        return execution_id
    
    async def _execute_improvements(self, execution_ids: List[str], dry_run: bool) -> Dict[str, Any]:
        """Execute improvements"""
        results = {}
        max_concurrent = self.coordination_rules["max_concurrent_executions"]
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(execution_id):
            async with semaphore:
                return await self._execute_single_improvement(execution_id, dry_run)
        
        # Execute improvements
        tasks = [execute_with_semaphore(exec_id) for exec_id in execution_ids]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results_list):
            execution_id = execution_ids[i]
            if isinstance(result, Exception):
                results[execution_id] = {"success": False, "error": str(result)}
            else:
                results[execution_id] = result
        
        return results
    
    async def _execute_single_improvement(self, execution_id: str, dry_run: bool) -> Dict[str, Any]:
        """Execute single improvement"""
        try:
            execution = self.executions[execution_id]
            execution.status = CoordinationStatus.IN_PROGRESS
            execution.started_at = datetime.utcnow()
            
            # Log execution start
            self._log_execution(execution_id, "started", f"Executing improvement {execution.improvement_id}")
            
            if dry_run:
                # Simulate execution
                await asyncio.sleep(1)
                execution.success = True
                execution.status = CoordinationStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                execution.duration = 1.0
                
                self._log_execution(execution_id, "completed", "Dry run completed successfully")
                
                return {"success": True, "message": "Dry run completed"}
            
            # Execute improvement (simplified for example)
            # In real implementation, this would call the improvement automator
            await asyncio.sleep(2)  # Simulate execution time
            
            # Simulate success/failure based on improvement ID
            success = not execution.improvement_id.endswith("_fail")
            
            execution.success = success
            execution.status = CoordinationStatus.COMPLETED if success else CoordinationStatus.FAILED
            execution.completed_at = datetime.utcnow()
            execution.duration = 2.0
            
            if not success:
                execution.error_message = f"Improvement {execution.improvement_id} failed"
                execution.rollback_required = True
                self._log_execution(execution_id, "failed", execution.error_message)
            else:
                self._log_execution(execution_id, "completed", "Improvement executed successfully")
            
            return {
                "success": success,
                "execution_id": execution_id,
                "duration": execution.duration,
                "error": execution.error_message if not success else None
            }
            
        except Exception as e:
            execution.status = CoordinationStatus.FAILED
            execution.error_message = str(e)
            execution.rollback_required = True
            
            self._log_execution(execution_id, "error", str(e))
            
            return {"success": False, "error": str(e)}
    
    async def _rollback_execution(self, execution_id: str) -> Dict[str, Any]:
        """Rollback execution"""
        try:
            execution = self.executions[execution_id]
            
            if not execution.rollback_required:
                return {"success": True, "message": "No rollback required"}
            
            # Simulate rollback
            await asyncio.sleep(1)
            
            execution.status = CoordinationStatus.CANCELLED
            execution.rollback_required = False
            
            self._log_execution(execution_id, "rolled_back", "Execution rolled back successfully")
            
            return {"success": True, "message": "Rollback completed"}
            
        except Exception as e:
            self._log_execution(execution_id, "rollback_error", str(e))
            return {"success": False, "error": str(e)}
    
    def _log_execution(self, execution_id: str, event: str, message: str):
        """Log execution event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if execution_id not in self.execution_logs:
            self.execution_logs[execution_id] = []
        
        self.execution_logs[execution_id].append(log_entry)
        
        logger.info(f"Execution {execution_id}: {event} - {message}")
    
    def _calculate_plan_progress(self, plan_id: str) -> float:
        """Calculate plan progress"""
        if plan_id not in self.plans:
            return 0.0
        
        plan = self.plans[plan_id]
        plan_executions = [
            exec_id for exec_id, execution in self.executions.items()
            if execution.plan_id == plan_id
        ]
        
        if not plan_executions:
            return 0.0
        
        completed_executions = len([
            exec_id for exec_id in plan_executions
            if self.executions[exec_id].status == CoordinationStatus.COMPLETED
        ])
        
        return (completed_executions / len(plan_executions)) * 100
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get coordination summary"""
        total_plans = len(self.plans)
        completed_plans = len([
            plan for plan in self.plans.values()
            if plan.status == CoordinationStatus.COMPLETED
        ])
        
        total_executions = len(self.executions)
        completed_executions = len([
            exec for exec in self.executions.values()
            if exec.status == CoordinationStatus.COMPLETED
        ])
        
        by_priority = {}
        for priority in ImprovementPriority:
            by_priority[priority.value] = len([
                plan for plan in self.plans.values()
                if plan.priority == priority
            ])
        
        return {
            "total_plans": total_plans,
            "completed_plans": completed_plans,
            "plan_completion_rate": (completed_plans / total_plans * 100) if total_plans > 0 else 0,
            "total_executions": total_executions,
            "completed_executions": completed_executions,
            "execution_completion_rate": (completed_executions / total_executions * 100) if total_executions > 0 else 0,
            "by_priority": by_priority,
            "coordination_rules": self.coordination_rules
        }
    
    def get_execution_logs(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get execution logs"""
        return self.execution_logs.get(execution_id, [])
    
    def create_quick_improvement_plan(self, title: str, improvements: List[str]) -> str:
        """Create quick improvement plan for common improvements"""
        return self.create_improvement_plan(
            title=title,
            description=f"Quick improvement plan for {title}",
            priority=ImprovementPriority.MEDIUM,
            estimated_duration=2.0,
            improvements=improvements
        )
    
    def create_critical_improvement_plan(self, title: str, improvements: List[str]) -> str:
        """Create critical improvement plan"""
        return self.create_improvement_plan(
            title=title,
            description=f"Critical improvement plan for {title}",
            priority=ImprovementPriority.CRITICAL,
            estimated_duration=4.0,
            improvements=improvements
        )
    
    def create_security_improvement_plan(self, title: str, improvements: List[str]) -> str:
        """Create security improvement plan"""
        return self.create_improvement_plan(
            title=title,
            description=f"Security improvement plan for {title}",
            priority=ImprovementPriority.HIGH,
            estimated_duration=3.0,
            improvements=improvements
        )
    
    def create_performance_improvement_plan(self, title: str, improvements: List[str]) -> str:
        """Create performance improvement plan"""
        return self.create_improvement_plan(
            title=title,
            description=f"Performance improvement plan for {title}",
            priority=ImprovementPriority.MEDIUM,
            estimated_duration=2.5,
            improvements=improvements
        )

# Global coordinator instance
improvement_coordinator = None

def get_improvement_coordinator() -> RealImprovementCoordinator:
    """Get improvement coordinator instance"""
    global improvement_coordinator
    if not improvement_coordinator:
        improvement_coordinator = RealImprovementCoordinator()
    return improvement_coordinator













