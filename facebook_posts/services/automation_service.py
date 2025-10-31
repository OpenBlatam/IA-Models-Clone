"""
Advanced Automation Service for Facebook Posts API
Intelligent automation, scheduling, and business process orchestration
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog

from ..core.config import get_settings
from ..core.models import FacebookPost, PostStatus, ContentType, AudienceType, OptimizationLevel
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.monitoring import get_monitor, timed
from ..infrastructure.database import get_db_manager, PostRepository
from ..services.ai_service import get_ai_service
from ..services.analytics_service import get_analytics_service
from ..services.ml_service import get_ml_service
from ..services.optimization_service import get_optimization_service
from ..services.recommendation_service import get_recommendation_service
from ..services.notification_service import get_notification_service
from ..services.security_service import get_security_service
from ..services.workflow_service import get_workflow_service

logger = structlog.get_logger(__name__)


class AutomationType(Enum):
    """Automation type enumeration"""
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    CONDITIONAL = "conditional"
    CONTINUOUS = "continuous"
    BATCH = "batch"


class AutomationStatus(Enum):
    """Automation status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


class TriggerType(Enum):
    """Trigger type enumeration"""
    TIME = "time"
    EVENT = "event"
    CONDITION = "condition"
    WEBHOOK = "webhook"
    API = "api"
    MANUAL = "manual"


@dataclass
class AutomationRule:
    """Automation rule data structure"""
    id: str
    name: str
    description: str
    automation_type: AutomationType
    trigger_type: TriggerType
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    action_config: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutomationExecution:
    """Automation execution data structure"""
    id: str
    rule_id: str
    status: AutomationStatus
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Scheduler:
    """Advanced scheduling system"""
    
    def __init__(self):
        self.scheduled_tasks: Dict[str, asyncio.Task] = {}
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    @timed("schedule_task")
    async def schedule_task(
        self,
        task_id: str,
        task_func: Callable,
        schedule_time: datetime,
        repeat_interval: Optional[timedelta] = None,
        max_executions: Optional[int] = None
    ) -> bool:
        """Schedule a task for execution"""
        try:
            # Cancel existing task if it exists
            if task_id in self.scheduled_tasks:
                self.scheduled_tasks[task_id].cancel()
            
            # Create new scheduled task
            task = asyncio.create_task(
                self._execute_scheduled_task(
                    task_id, task_func, schedule_time, repeat_interval, max_executions
                )
            )
            
            self.scheduled_tasks[task_id] = task
            
            logger.info("Task scheduled", task_id=task_id, schedule_time=schedule_time)
            return True
            
        except Exception as e:
            logger.error("Failed to schedule task", task_id=task_id, error=str(e))
            return False
    
    async def _execute_scheduled_task(
        self,
        task_id: str,
        task_func: Callable,
        schedule_time: datetime,
        repeat_interval: Optional[timedelta] = None,
        max_executions: Optional[int] = None
    ):
        """Execute scheduled task"""
        execution_count = 0
        
        try:
            while True:
                # Calculate next execution time
                if execution_count == 0:
                    next_execution = schedule_time
                else:
                    if not repeat_interval:
                        break
                    next_execution = datetime.now() + repeat_interval
                
                # Wait until execution time
                wait_time = (next_execution - datetime.now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Check if task was cancelled
                if task_id not in self.scheduled_tasks:
                    break
                
                # Execute task
                try:
                    await task_func()
                    execution_count += 1
                    
                    logger.info("Scheduled task executed", task_id=task_id, execution_count=execution_count)
                    
                    # Check max executions
                    if max_executions and execution_count >= max_executions:
                        break
                        
                except Exception as e:
                    logger.error("Scheduled task execution failed", task_id=task_id, error=str(e))
                
                # If no repeat interval, break after first execution
                if not repeat_interval:
                    break
                    
        except asyncio.CancelledError:
            logger.info("Scheduled task cancelled", task_id=task_id)
        finally:
            # Clean up
            if task_id in self.scheduled_tasks:
                del self.scheduled_tasks[task_id]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        try:
            if task_id in self.scheduled_tasks:
                self.scheduled_tasks[task_id].cancel()
                del self.scheduled_tasks[task_id]
                logger.info("Scheduled task cancelled", task_id=task_id)
                return True
            return False
        except Exception as e:
            logger.error("Failed to cancel task", task_id=task_id, error=str(e))
            return False
    
    async def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """Get list of scheduled tasks"""
        tasks = []
        for task_id, task in self.scheduled_tasks.items():
            tasks.append({
                "task_id": task_id,
                "status": "running" if not task.done() else "completed",
                "cancelled": task.cancelled()
            })
        return tasks


class EventProcessor:
    """Advanced event processing system"""
    
    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    @timed("register_event_handler")
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info("Event handler registered", event_type=event_type)
    
    @timed("process_event")
    async def process_event(self, event_type: str, event_data: Dict[str, Any]):
        """Process an event"""
        try:
            if event_type not in self.event_handlers:
                logger.warning("No handlers for event type", event_type=event_type)
                return
            
            # Execute all handlers for this event type
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error("Event handler failed", event_type=event_type, error=str(e))
            
            logger.info("Event processed", event_type=event_type, handlers_count=len(self.event_handlers[event_type]))
            
        except Exception as e:
            logger.error("Event processing failed", event_type=event_type, error=str(e))
    
    @timed("emit_event")
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event"""
        # Store event in cache for persistence
        event_id = f"event_{int(time.time())}_{event_type}"
        await self.cache_manager.cache.set(
            f"event:{event_id}",
            {
                "type": event_type,
                "data": event_data,
                "timestamp": datetime.now().isoformat()
            },
            ttl=86400
        )
        
        # Process event
        await self.process_event(event_type, event_data)
        
        logger.info("Event emitted", event_type=event_type, event_id=event_id)


class ConditionEvaluator:
    """Advanced condition evaluation system"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    @timed("evaluate_condition")
    async def evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a condition against context"""
        try:
            condition_type = condition.get("type")
            
            if condition_type == "equals":
                return self._evaluate_equals(condition, context)
            elif condition_type == "not_equals":
                return self._evaluate_not_equals(condition, context)
            elif condition_type == "greater_than":
                return self._evaluate_greater_than(condition, context)
            elif condition_type == "less_than":
                return self._evaluate_less_than(condition, context)
            elif condition_type == "contains":
                return self._evaluate_contains(condition, context)
            elif condition_type == "not_contains":
                return self._evaluate_not_contains(condition, context)
            elif condition_type == "in_list":
                return self._evaluate_in_list(condition, context)
            elif condition_type == "not_in_list":
                return self._evaluate_not_in_list(condition, context)
            elif condition_type == "regex":
                return self._evaluate_regex(condition, context)
            elif condition_type == "time_range":
                return self._evaluate_time_range(condition, context)
            elif condition_type == "custom":
                return await self._evaluate_custom(condition, context)
            else:
                logger.error("Unknown condition type", condition_type=condition_type)
                return False
                
        except Exception as e:
            logger.error("Condition evaluation failed", condition=condition, error=str(e))
            return False
    
    def _evaluate_equals(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate equals condition"""
        field = condition.get("field")
        value = condition.get("value")
        context_value = context.get(field)
        return str(context_value) == str(value)
    
    def _evaluate_not_equals(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate not equals condition"""
        field = condition.get("field")
        value = condition.get("value")
        context_value = context.get(field)
        return str(context_value) != str(value)
    
    def _evaluate_greater_than(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate greater than condition"""
        field = condition.get("field")
        value = condition.get("value")
        context_value = context.get(field)
        try:
            return float(context_value) > float(value)
        except (ValueError, TypeError):
            return False
    
    def _evaluate_less_than(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate less than condition"""
        field = condition.get("field")
        value = condition.get("value")
        context_value = context.get(field)
        try:
            return float(context_value) < float(value)
        except (ValueError, TypeError):
            return False
    
    def _evaluate_contains(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate contains condition"""
        field = condition.get("field")
        value = condition.get("value")
        context_value = context.get(field)
        return str(value).lower() in str(context_value).lower()
    
    def _evaluate_not_contains(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate not contains condition"""
        field = condition.get("field")
        value = condition.get("value")
        context_value = context.get(field)
        return str(value).lower() not in str(context_value).lower()
    
    def _evaluate_in_list(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate in list condition"""
        field = condition.get("field")
        value_list = condition.get("value_list", [])
        context_value = context.get(field)
        return str(context_value) in [str(v) for v in value_list]
    
    def _evaluate_not_in_list(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate not in list condition"""
        field = condition.get("field")
        value_list = condition.get("value_list", [])
        context_value = context.get(field)
        return str(context_value) not in [str(v) for v in value_list]
    
    def _evaluate_regex(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate regex condition"""
        import re
        field = condition.get("field")
        pattern = condition.get("pattern")
        context_value = context.get(field)
        try:
            return bool(re.search(pattern, str(context_value)))
        except re.error:
            return False
    
    def _evaluate_time_range(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate time range condition"""
        field = condition.get("field")
        start_time = condition.get("start_time")
        end_time = condition.get("end_time")
        context_value = context.get(field)
        
        try:
            if isinstance(context_value, str):
                context_time = datetime.fromisoformat(context_value)
            else:
                context_time = context_value
            
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            
            return start_dt <= context_time <= end_dt
        except (ValueError, TypeError):
            return False
    
    async def _evaluate_custom(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate custom condition"""
        custom_function = condition.get("custom_function")
        if custom_function:
            # Mock custom function evaluation
            return await self._execute_custom_condition(custom_function, context)
        return False
    
    async def _execute_custom_condition(self, function_name: str, context: Dict[str, Any]) -> bool:
        """Execute custom condition function"""
        # Mock custom condition execution
        return True


class ActionExecutor:
    """Advanced action execution system"""
    
    def __init__(self):
        self.ai_service = get_ai_service()
        self.analytics_service = get_analytics_service()
        self.ml_service = get_ml_service()
        self.optimization_service = get_optimization_service()
        self.recommendation_service = get_recommendation_service()
        self.notification_service = get_notification_service()
        self.security_service = get_security_service()
        self.workflow_service = get_workflow_service()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    @timed("execute_action")
    async def execute_action(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action based on configuration"""
        try:
            action_type = action_config.get("type")
            
            if action_type == "ai_generate":
                return await self._execute_ai_generate(action_config, context)
            elif action_type == "ai_analyze":
                return await self._execute_ai_analyze(action_config, context)
            elif action_type == "ml_predict":
                return await self._execute_ml_predict(action_config, context)
            elif action_type == "optimize_content":
                return await self._execute_optimize_content(action_config, context)
            elif action_type == "send_notification":
                return await self._execute_send_notification(action_config, context)
            elif action_type == "schedule_post":
                return await self._execute_schedule_post(action_config, context)
            elif action_type == "publish_post":
                return await self._execute_publish_post(action_config, context)
            elif action_type == "track_analytics":
                return await self._execute_track_analytics(action_config, context)
            elif action_type == "execute_workflow":
                return await self._execute_workflow(action_config, context)
            elif action_type == "update_database":
                return await self._execute_update_database(action_config, context)
            elif action_type == "send_webhook":
                return await self._execute_send_webhook(action_config, context)
            elif action_type == "custom":
                return await self._execute_custom_action(action_config, context)
            else:
                raise ValueError(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error("Action execution failed", action_type=action_config.get("type"), error=str(e))
            raise
    
    async def _execute_ai_generate(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI content generation action"""
        # Get parameters from config or context
        topic = action_config.get("topic") or context.get("topic")
        audience_type = action_config.get("audience_type") or context.get("audience_type")
        content_type = action_config.get("content_type") or context.get("content_type")
        
        if not all([topic, audience_type, content_type]):
            raise ValueError("Missing required parameters for AI generation")
        
        # Generate content
        from ..core.models import PostRequest
        request = PostRequest(
            topic=topic,
            content_type=ContentType(content_type),
            audience_type=AudienceType(audience_type),
            tone=action_config.get("tone", "professional")
        )
        
        result = await self.ai_service.generate_content(request)
        
        return {
            "action_type": "ai_generate",
            "content": result.content if result else None,
            "metadata": result.metadata if result else {},
            "success": result is not None
        }
    
    async def _execute_ai_analyze(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI content analysis action"""
        content = action_config.get("content") or context.get("content")
        if not content:
            raise ValueError("Content is required for AI analysis")
        
        result = await self.ai_service.analyze_content(content)
        
        return {
            "action_type": "ai_analyze",
            "analysis": result.analysis if result else {},
            "scores": result.scores if result else {},
            "suggestions": result.suggestions if result else [],
            "success": result is not None
        }
    
    async def _execute_ml_predict(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ML prediction action"""
        content = action_config.get("content") or context.get("content")
        prediction_type = action_config.get("prediction_type", "engagement")
        
        if not content:
            raise ValueError("Content is required for ML prediction")
        
        # Make prediction
        if prediction_type == "engagement":
            result = await self.ml_service.predict_engagement(content, context)
        elif prediction_type == "reach":
            result = await self.ml_service.predict_reach(content, context)
        elif prediction_type == "clicks":
            result = await self.ml_service.predict_clicks(content, context)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        
        return {
            "action_type": "ml_predict",
            "prediction": result.predicted_value if result else 0,
            "confidence": result.confidence if result else 0,
            "prediction_type": prediction_type,
            "success": result is not None
        }
    
    async def _execute_optimize_content(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content optimization action"""
        content = action_config.get("content") or context.get("content")
        strategy = action_config.get("strategy", "engagement")
        
        if not content:
            raise ValueError("Content is required for optimization")
        
        result = await self.optimization_service.optimize_content(
            content=content,
            strategy=strategy,
            context=context
        )
        
        return {
            "action_type": "optimize_content",
            "optimized_content": result.optimized_content if result else content,
            "optimizations": result.optimizations if result else [],
            "improvement_score": result.improvement_score if result else 0,
            "success": result is not None
        }
    
    async def _execute_send_notification(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification sending action"""
        notification_type = action_config.get("notification_type", "email")
        priority = action_config.get("priority", "medium")
        title = action_config.get("title") or context.get("title")
        message = action_config.get("message") or context.get("message")
        recipient = action_config.get("recipient") or context.get("recipient")
        
        if not all([title, message, recipient]):
            raise ValueError("Missing required parameters for notification")
        
        notification_id = await self.notification_service.send_notification(
            notification_type=notification_type,
            priority=priority,
            title=title,
            message=message,
            recipient=recipient,
            metadata=action_config.get("metadata", {})
        )
        
        return {
            "action_type": "send_notification",
            "notification_id": notification_id,
            "notification_type": notification_type,
            "recipient": recipient,
            "success": True
        }
    
    async def _execute_schedule_post(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute post scheduling action"""
        content = action_config.get("content") or context.get("content")
        schedule_time = action_config.get("schedule_time") or context.get("schedule_time")
        
        if not content:
            raise ValueError("Content is required for post scheduling")
        
        # Mock post scheduling
        post_id = f"post_{int(time.time())}"
        
        return {
            "action_type": "schedule_post",
            "post_id": post_id,
            "content": content,
            "schedule_time": schedule_time,
            "status": "scheduled",
            "success": True
        }
    
    async def _execute_publish_post(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute post publishing action"""
        content = action_config.get("content") or context.get("content")
        platform = action_config.get("platform", "facebook")
        
        if not content:
            raise ValueError("Content is required for post publishing")
        
        # Mock post publishing
        post_id = f"published_post_{int(time.time())}"
        
        return {
            "action_type": "publish_post",
            "post_id": post_id,
            "content": content,
            "platform": platform,
            "status": "published",
            "published_at": datetime.now().isoformat(),
            "success": True
        }
    
    async def _execute_track_analytics(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analytics tracking action"""
        post_id = action_config.get("post_id") or context.get("post_id")
        metrics = action_config.get("metrics", ["engagement", "reach", "clicks"])
        
        if not post_id:
            raise ValueError("Post ID is required for analytics tracking")
        
        # Mock analytics tracking
        analytics_data = {
            "post_id": post_id,
            "metrics": {metric: 0 for metric in metrics},
            "tracked_at": datetime.now().isoformat()
        }
        
        return {
            "action_type": "track_analytics",
            "analytics_data": analytics_data,
            "success": True
        }
    
    async def _execute_workflow(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow action"""
        workflow_id = action_config.get("workflow_id")
        trigger_data = action_config.get("trigger_data", {})
        
        if not workflow_id:
            raise ValueError("Workflow ID is required")
        
        execution_id = await self.workflow_service.execute_workflow(
            workflow_id=workflow_id,
            trigger_data=trigger_data,
            context=context
        )
        
        return {
            "action_type": "execute_workflow",
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "success": True
        }
    
    async def _execute_update_database(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database update action"""
        table = action_config.get("table")
        data = action_config.get("data", {})
        operation = action_config.get("operation", "update")
        
        if not table:
            raise ValueError("Table is required for database update")
        
        # Mock database update
        return {
            "action_type": "update_database",
            "table": table,
            "operation": operation,
            "affected_rows": 1,
            "success": True
        }
    
    async def _execute_send_webhook(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute webhook sending action"""
        webhook_url = action_config.get("webhook_url")
        payload = action_config.get("payload", {})
        
        if not webhook_url:
            raise ValueError("Webhook URL is required")
        
        # Mock webhook sending
        return {
            "action_type": "send_webhook",
            "webhook_url": webhook_url,
            "status_code": 200,
            "success": True
        }
    
    async def _execute_custom_action(self, action_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom action"""
        custom_function = action_config.get("custom_function")
        custom_script = action_config.get("custom_script")
        
        if custom_function:
            result = await self._execute_custom_function(custom_function, context)
        elif custom_script:
            result = await self._execute_custom_script(custom_script, context)
        else:
            raise ValueError("Custom function or script is required for custom action")
        
        return {
            "action_type": "custom",
            "custom_result": result,
            "success": True
        }
    
    async def _execute_custom_function(self, function_name: str, context: Dict[str, Any]) -> Any:
        """Execute custom function"""
        # Mock custom function execution
        return f"Custom function {function_name} executed"
    
    async def _execute_custom_script(self, script: str, context: Dict[str, Any]) -> Any:
        """Execute custom script"""
        # Mock custom script execution
        return f"Custom script executed: {script[:50]}..."


class AutomationEngine:
    """Main automation engine"""
    
    def __init__(self):
        self.scheduler = Scheduler()
        self.event_processor = EventProcessor()
        self.condition_evaluator = ConditionEvaluator()
        self.action_executor = ActionExecutor()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
        self.automation_rules: Dict[str, AutomationRule] = {}
        self.active_executions: Dict[str, AutomationExecution] = {}
        self._load_default_rules()
        self._setup_event_handlers()
    
    def _load_default_rules(self):
        """Load default automation rules"""
        default_rules = [
            AutomationRule(
                id="high_engagement_alert",
                name="High Engagement Alert",
                description="Send alert when post engagement is high",
                automation_type=AutomationType.EVENT_DRIVEN,
                trigger_type=TriggerType.EVENT,
                trigger_config={"event_type": "post_engagement_high"},
                action_config={
                    "type": "send_notification",
                    "notification_type": "email",
                    "priority": "high",
                    "title": "High Engagement Alert",
                    "message": "Your post is performing exceptionally well!",
                    "recipient": "{{user_email}}"
                },
                conditions=[
                    {"type": "greater_than", "field": "engagement_rate", "value": 0.8}
                ]
            ),
            AutomationRule(
                id="daily_content_generation",
                name="Daily Content Generation",
                description="Generate content daily at 9 AM",
                automation_type=AutomationType.SCHEDULED,
                trigger_type=TriggerType.TIME,
                trigger_config={
                    "schedule_time": "09:00",
                    "repeat_interval": "daily"
                },
                action_config={
                    "type": "ai_generate",
                    "topic": "{{daily_topic}}",
                    "audience_type": "professionals",
                    "content_type": "educational"
                }
            ),
            AutomationRule(
                id="low_engagement_optimization",
                name="Low Engagement Optimization",
                description="Optimize content when engagement is low",
                automation_type=AutomationType.CONDITIONAL,
                trigger_type=TriggerType.CONDITION,
                trigger_config={"check_interval": 3600},  # Check every hour
                action_config={
                    "type": "optimize_content",
                    "strategy": "engagement"
                },
                conditions=[
                    {"type": "less_than", "field": "engagement_rate", "value": 0.2}
                ]
            )
        ]
        
        for rule in default_rules:
            self.automation_rules[rule.id] = rule
    
    def _setup_event_handlers(self):
        """Setup event handlers"""
        self.event_processor.register_event_handler("post_engagement_high", self._handle_high_engagement)
        self.event_processor.register_event_handler("post_engagement_low", self._handle_low_engagement)
        self.event_processor.register_event_handler("post_published", self._handle_post_published)
        self.event_processor.register_event_handler("user_registered", self._handle_user_registered)
    
    async def _handle_high_engagement(self, event_data: Dict[str, Any]):
        """Handle high engagement event"""
        await self._process_automation_rule("high_engagement_alert", event_data)
    
    async def _handle_low_engagement(self, event_data: Dict[str, Any]):
        """Handle low engagement event"""
        await self._process_automation_rule("low_engagement_optimization", event_data)
    
    async def _handle_post_published(self, event_data: Dict[str, Any]):
        """Handle post published event"""
        # Start analytics tracking
        await self.action_executor.execute_action({
            "type": "track_analytics",
            "post_id": event_data.get("post_id"),
            "metrics": ["engagement", "reach", "clicks"]
        }, event_data)
    
    async def _handle_user_registered(self, event_data: Dict[str, Any]):
        """Handle user registered event"""
        # Send welcome notification
        await self.action_executor.execute_action({
            "type": "send_notification",
            "notification_type": "email",
            "priority": "medium",
            "title": "Welcome to Facebook Posts API!",
            "message": "Thank you for registering. Get started with our AI-powered content generation.",
            "recipient": event_data.get("user_email")
        }, event_data)
    
    @timed("process_automation_rule")
    async def _process_automation_rule(self, rule_id: str, trigger_data: Dict[str, Any]):
        """Process an automation rule"""
        try:
            rule = self.automation_rules.get(rule_id)
            if not rule or not rule.enabled:
                return
            
            # Evaluate conditions
            if rule.conditions:
                context = {**trigger_data}
                for condition in rule.conditions:
                    if not await self.condition_evaluator.evaluate_condition(condition, context):
                        logger.info("Condition not met", rule_id=rule_id, condition=condition)
                        return
            
            # Create execution
            execution = AutomationExecution(
                id=f"exec_{int(time.time())}_{rule_id}",
                rule_id=rule_id,
                status=AutomationStatus.ACTIVE,
                trigger_data=trigger_data
            )
            
            self.active_executions[execution.id] = execution
            
            # Execute action
            result = await self.action_executor.execute_action(rule.action_config, trigger_data)
            
            execution.status = AutomationStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.result = result
            
            # Update rule statistics
            rule.last_executed = datetime.now()
            rule.execution_count += 1
            rule.success_count += 1
            
            logger.info("Automation rule executed", rule_id=rule_id, execution_id=execution.id)
            
        except Exception as e:
            if execution:
                execution.status = AutomationStatus.ERROR
                execution.error_message = str(e)
                execution.completed_at = datetime.now()
            
            rule.error_count += 1
            
            logger.error("Automation rule execution failed", rule_id=rule_id, error=str(e))
    
    @timed("create_automation_rule")
    async def create_automation_rule(self, rule: AutomationRule) -> bool:
        """Create a new automation rule"""
        try:
            self.automation_rules[rule.id] = rule
            
            # Setup scheduling if needed
            if rule.automation_type == AutomationType.SCHEDULED:
                await self._setup_scheduled_automation(rule)
            
            logger.info("Automation rule created", rule_id=rule.id)
            return True
            
        except Exception as e:
            logger.error("Failed to create automation rule", rule_id=rule.id, error=str(e))
            return False
    
    async def _setup_scheduled_automation(self, rule: AutomationRule):
        """Setup scheduled automation"""
        schedule_time = rule.trigger_config.get("schedule_time")
        repeat_interval = rule.trigger_config.get("repeat_interval")
        
        if schedule_time:
            # Parse schedule time
            if isinstance(schedule_time, str):
                # Assume format "HH:MM"
                hour, minute = map(int, schedule_time.split(":"))
                next_execution = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_execution <= datetime.now():
                    next_execution += timedelta(days=1)
            else:
                next_execution = schedule_time
            
            # Parse repeat interval
            interval = None
            if repeat_interval == "daily":
                interval = timedelta(days=1)
            elif repeat_interval == "weekly":
                interval = timedelta(weeks=1)
            elif repeat_interval == "hourly":
                interval = timedelta(hours=1)
            
            # Schedule task
            await self.scheduler.schedule_task(
                task_id=f"automation_{rule.id}",
                task_func=lambda: self._process_automation_rule(rule.id, {}),
                schedule_time=next_execution,
                repeat_interval=interval
            )
    
    @timed("emit_event")
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event"""
        await self.event_processor.emit_event(event_type, event_data)
    
    async def get_automation_rules(self) -> List[Dict[str, Any]]:
        """Get all automation rules"""
        rules = []
        for rule_id, rule in self.automation_rules.items():
            rules.append({
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "automation_type": rule.automation_type.value,
                "trigger_type": rule.trigger_type.value,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "execution_count": rule.execution_count,
                "success_count": rule.success_count,
                "error_count": rule.error_count,
                "last_executed": rule.last_executed.isoformat() if rule.last_executed else None,
                "created_at": rule.created_at.isoformat()
            })
        return rules
    
    async def get_automation_executions(self, rule_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get automation executions"""
        executions = []
        for exec_id, execution in self.active_executions.items():
            if rule_id and execution.rule_id != rule_id:
                continue
            
            executions.append({
                "id": execution.id,
                "rule_id": execution.rule_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "error_message": execution.error_message,
                "result": execution.result
            })
        return executions


class AutomationService:
    """Main automation service orchestrator"""
    
    def __init__(self):
        self.automation_engine = AutomationEngine()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    @timed("automation_service_emit_event")
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event"""
        await self.automation_engine.emit_event(event_type, event_data)
    
    @timed("automation_service_create_rule")
    async def create_automation_rule(self, rule: AutomationRule) -> bool:
        """Create automation rule"""
        return await self.automation_engine.create_automation_rule(rule)
    
    async def get_automation_rules(self) -> List[Dict[str, Any]]:
        """Get automation rules"""
        return await self.automation_engine.get_automation_rules()
    
    async def get_automation_executions(self, rule_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get automation executions"""
        return await self.automation_engine.get_automation_executions(rule_id)
    
    async def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """Get scheduled tasks"""
        return await self.automation_engine.scheduler.get_scheduled_tasks()


# Global automation service instance
_automation_service: Optional[AutomationService] = None


def get_automation_service() -> AutomationService:
    """Get global automation service instance"""
    global _automation_service
    
    if _automation_service is None:
        _automation_service = AutomationService()
    
    return _automation_service


# Export all classes and functions
__all__ = [
    # Enums
    'AutomationType',
    'AutomationStatus',
    'TriggerType',
    
    # Data classes
    'AutomationRule',
    'AutomationExecution',
    
    # Services
    'Scheduler',
    'EventProcessor',
    'ConditionEvaluator',
    'ActionExecutor',
    'AutomationEngine',
    'AutomationService',
    
    # Utility functions
    'get_automation_service',
]





























