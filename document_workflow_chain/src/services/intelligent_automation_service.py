"""
Intelligent Automation Service - Advanced Implementation
======================================================

Advanced intelligent automation service with AI-powered decision making and automation.
"""

from __future__ import annotations
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

from .ai_service import ai_service
from .workflow_service import workflow_service
from .analytics_service import analytics_service
from .notification_service import notification_service

logger = logging.getLogger(__name__)


class AutomationTrigger(str, Enum):
    """Automation trigger enumeration"""
    SCHEDULED = "scheduled"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    MANUAL = "manual"
    API_CALL = "api_call"
    WEBHOOK = "webhook"
    DATA_CHANGE = "data_change"
    THRESHOLD_REACHED = "threshold_reached"


class AutomationAction(str, Enum):
    """Automation action enumeration"""
    SEND_NOTIFICATION = "send_notification"
    EXECUTE_WORKFLOW = "execute_workflow"
    GENERATE_REPORT = "generate_report"
    UPDATE_DATA = "update_data"
    CREATE_TASK = "create_task"
    AI_ANALYSIS = "ai_analysis"
    AI_GENERATION = "ai_generation"
    SCHEDULE_TASK = "schedule_task"
    SEND_EMAIL = "send_email"
    UPDATE_STATUS = "update_status"


class AutomationStatus(str, Enum):
    """Automation status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIGGERED = "triggered"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class IntelligentAutomationService:
    """Advanced intelligent automation service with AI-powered decision making"""
    
    def __init__(self):
        self.automations = {}
        self.automation_rules = {}
        self.automation_templates = {}
        self.automation_stats = {
            "total_automations": 0,
            "active_automations": 0,
            "triggered_automations": 0,
            "completed_automations": 0,
            "failed_automations": 0,
            "automations_by_trigger": {trigger.value: 0 for trigger in AutomationTrigger},
            "automations_by_action": {action.value: 0 for action in AutomationAction}
        }
        
        # Initialize automation templates
        self._initialize_automation_templates()
    
    def _initialize_automation_templates(self):
        """Initialize automation templates"""
        try:
            # Content Generation Automation
            self.automation_templates["content_generation"] = {
                "name": "AI Content Generation Automation",
                "description": "Automatically generate content based on triggers",
                "trigger": {
                    "type": AutomationTrigger.EVENT_BASED.value,
                    "conditions": {
                        "event_type": "content_request",
                        "priority": "high"
                    }
                },
                "actions": [
                    {
                        "type": AutomationAction.AI_GENERATION.value,
                        "parameters": {
                            "prompt_template": "Generate content for: {topic}",
                            "max_tokens": 1000,
                            "temperature": 0.7
                        }
                    },
                    {
                        "type": AutomationAction.SEND_NOTIFICATION.value,
                        "parameters": {
                            "channel": "email",
                            "template": "content_generated"
                        }
                    }
                ],
                "ai_decision_making": True,
                "learning_enabled": True
            }
            
            # Workflow Optimization Automation
            self.automation_templates["workflow_optimization"] = {
                "name": "Workflow Optimization Automation",
                "description": "Automatically optimize workflows based on performance",
                "trigger": {
                    "type": AutomationTrigger.CONDITION_BASED.value,
                    "conditions": {
                        "metric": "workflow_performance",
                        "operator": "less_than",
                        "threshold": 0.8
                    }
                },
                "actions": [
                    {
                        "type": AutomationAction.AI_ANALYSIS.value,
                        "parameters": {
                            "analysis_type": "performance_optimization",
                            "focus": "workflow_efficiency"
                        }
                    },
                    {
                        "type": AutomationAction.UPDATE_DATA.value,
                        "parameters": {
                            "target": "workflow_configuration",
                            "optimization": "ai_recommended"
                        }
                    }
                ],
                "ai_decision_making": True,
                "learning_enabled": True
            }
            
            # Report Generation Automation
            self.automation_templates["report_generation"] = {
                "name": "Automated Report Generation",
                "description": "Generate reports automatically based on schedule",
                "trigger": {
                    "type": AutomationTrigger.SCHEDULED.value,
                    "schedule": "0 9 * * 1"  # Every Monday at 9 AM
                },
                "actions": [
                    {
                        "type": AutomationAction.GENERATE_REPORT.value,
                        "parameters": {
                            "report_type": "weekly_summary",
                            "include_ai_insights": True
                        }
                    },
                    {
                        "type": AutomationAction.SEND_EMAIL.value,
                        "parameters": {
                            "recipients": ["admin@company.com"],
                            "subject": "Weekly Report - {date}"
                        }
                    }
                ],
                "ai_decision_making": False,
                "learning_enabled": False
            }
            
            logger.info("Automation templates initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize automation templates: {e}")
    
    async def create_automation(
        self,
        name: str,
        description: str,
        trigger: Dict[str, Any],
        actions: List[Dict[str, Any]],
        ai_decision_making: bool = False,
        learning_enabled: bool = False,
        template_name: Optional[str] = None
    ) -> str:
        """Create intelligent automation"""
        try:
            automation_id = f"automation_{len(self.automations) + 1}"
            
            # Use template if provided
            if template_name and template_name in self.automation_templates:
                template = self.automation_templates[template_name]
                trigger = template["trigger"]
                actions = template["actions"]
                ai_decision_making = template.get("ai_decision_making", ai_decision_making)
                learning_enabled = template.get("learning_enabled", learning_enabled)
            
            # Create automation
            automation = {
                "id": automation_id,
                "name": name,
                "description": description,
                "trigger": trigger,
                "actions": actions,
                "ai_decision_making": ai_decision_making,
                "learning_enabled": learning_enabled,
                "status": AutomationStatus.ACTIVE.value,
                "created_at": datetime.utcnow().isoformat(),
                "last_triggered": None,
                "last_executed": None,
                "execution_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "learning_data": {},
                "metadata": {}
            }
            
            self.automations[automation_id] = automation
            self.automation_stats["total_automations"] += 1
            self.automation_stats["active_automations"] += 1
            self.automation_stats["automations_by_trigger"][trigger["type"]] += 1
            
            # Count actions
            for action in actions:
                self.automation_stats["automations_by_action"][action["type"]] += 1
            
            logger.info(f"Automation created: {automation_id} - {name}")
            return automation_id
        
        except Exception as e:
            logger.error(f"Failed to create automation: {e}")
            raise
    
    async def trigger_automation(
        self,
        automation_id: str,
        trigger_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trigger automation execution"""
        try:
            if automation_id not in self.automations:
                raise ValueError(f"Automation not found: {automation_id}")
            
            automation = self.automations[automation_id]
            
            if automation["status"] != AutomationStatus.ACTIVE.value:
                raise ValueError(f"Automation is not active: {automation_id}")
            
            # Check trigger conditions
            if not await self._check_trigger_conditions(automation["trigger"], trigger_data):
                return {
                    "automation_id": automation_id,
                    "triggered": False,
                    "reason": "Trigger conditions not met"
                }
            
            # Update automation status
            automation["status"] = AutomationStatus.TRIGGERED.value
            automation["last_triggered"] = datetime.utcnow().isoformat()
            self.automation_stats["triggered_automations"] += 1
            
            # Execute automation
            result = await self._execute_automation(automation, trigger_data, context or {})
            
            return {
                "automation_id": automation_id,
                "triggered": True,
                "execution_result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to trigger automation: {e}")
            if automation_id in self.automations:
                self.automations[automation_id]["failure_count"] += 1
            raise
    
    async def _check_trigger_conditions(self, trigger: Dict[str, Any], trigger_data: Dict[str, Any]) -> bool:
        """Check if trigger conditions are met"""
        try:
            trigger_type = trigger["type"]
            
            if trigger_type == AutomationTrigger.EVENT_BASED.value:
                return await self._check_event_conditions(trigger.get("conditions", {}), trigger_data)
            
            elif trigger_type == AutomationTrigger.CONDITION_BASED.value:
                return await self._check_condition_based(trigger.get("conditions", {}), trigger_data)
            
            elif trigger_type == AutomationTrigger.MANUAL.value:
                return True  # Manual triggers are always valid
            
            elif trigger_type == AutomationTrigger.API_CALL.value:
                return await self._check_api_call_conditions(trigger.get("conditions", {}), trigger_data)
            
            elif trigger_type == AutomationTrigger.WEBHOOK.value:
                return await self._check_webhook_conditions(trigger.get("conditions", {}), trigger_data)
            
            elif trigger_type == AutomationTrigger.DATA_CHANGE.value:
                return await self._check_data_change_conditions(trigger.get("conditions", {}), trigger_data)
            
            elif trigger_type == AutomationTrigger.THRESHOLD_REACHED.value:
                return await self._check_threshold_conditions(trigger.get("conditions", {}), trigger_data)
            
            else:
                return False
        
        except Exception as e:
            logger.error(f"Failed to check trigger conditions: {e}")
            return False
    
    async def _check_event_conditions(self, conditions: Dict[str, Any], trigger_data: Dict[str, Any]) -> bool:
        """Check event-based conditions"""
        try:
            for key, expected_value in conditions.items():
                if key not in trigger_data or trigger_data[key] != expected_value:
                    return False
            return True
        
        except Exception as e:
            logger.error(f"Failed to check event conditions: {e}")
            return False
    
    async def _check_condition_based(self, conditions: Dict[str, Any], trigger_data: Dict[str, Any]) -> bool:
        """Check condition-based triggers"""
        try:
            metric = conditions.get("metric")
            operator = conditions.get("operator")
            threshold = conditions.get("threshold")
            
            if not all([metric, operator, threshold]):
                return False
            
            # Get current metric value (simplified)
            current_value = trigger_data.get(metric, 0)
            
            if operator == "greater_than":
                return current_value > threshold
            elif operator == "less_than":
                return current_value < threshold
            elif operator == "equals":
                return current_value == threshold
            elif operator == "not_equals":
                return current_value != threshold
            else:
                return False
        
        except Exception as e:
            logger.error(f"Failed to check condition-based triggers: {e}")
            return False
    
    async def _check_api_call_conditions(self, conditions: Dict[str, Any], trigger_data: Dict[str, Any]) -> bool:
        """Check API call conditions"""
        try:
            # Simplified API call condition checking
            return trigger_data.get("api_endpoint") == conditions.get("endpoint")
        
        except Exception as e:
            logger.error(f"Failed to check API call conditions: {e}")
            return False
    
    async def _check_webhook_conditions(self, conditions: Dict[str, Any], trigger_data: Dict[str, Any]) -> bool:
        """Check webhook conditions"""
        try:
            # Simplified webhook condition checking
            return trigger_data.get("webhook_source") == conditions.get("source")
        
        except Exception as e:
            logger.error(f"Failed to check webhook conditions: {e}")
            return False
    
    async def _check_data_change_conditions(self, conditions: Dict[str, Any], trigger_data: Dict[str, Any]) -> bool:
        """Check data change conditions"""
        try:
            # Simplified data change condition checking
            return trigger_data.get("data_type") == conditions.get("data_type")
        
        except Exception as e:
            logger.error(f"Failed to check data change conditions: {e}")
            return False
    
    async def _check_threshold_conditions(self, conditions: Dict[str, Any], trigger_data: Dict[str, Any]) -> bool:
        """Check threshold conditions"""
        try:
            # Similar to condition-based but for specific thresholds
            return await self._check_condition_based(conditions, trigger_data)
        
        except Exception as e:
            logger.error(f"Failed to check threshold conditions: {e}")
            return False
    
    async def _execute_automation(
        self,
        automation: Dict[str, Any],
        trigger_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute automation actions"""
        try:
            automation["status"] = AutomationStatus.EXECUTING.value
            automation["execution_count"] += 1
            
            results = []
            
            # Execute each action
            for action in automation["actions"]:
                try:
                    # AI decision making if enabled
                    if automation["ai_decision_making"]:
                        should_execute = await self._ai_decision_making(action, trigger_data, context)
                        if not should_execute:
                            continue
                    
                    # Execute action
                    result = await self._execute_action(action, trigger_data, context)
                    results.append(result)
                    
                    # Learning if enabled
                    if automation["learning_enabled"]:
                        await self._update_learning_data(automation, action, result)
                
                except Exception as e:
                    error_result = {
                        "action": action["type"],
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    results.append(error_result)
                    logger.error(f"Action execution failed: {e}")
            
            # Update automation status
            success_count = len([r for r in results if r.get("success", False)])
            if success_count == len(results):
                automation["status"] = AutomationStatus.COMPLETED.value
                automation["success_count"] += 1
                self.automation_stats["completed_automations"] += 1
            else:
                automation["status"] = AutomationStatus.FAILED.value
                automation["failure_count"] += 1
                self.automation_stats["failed_automations"] += 1
            
            automation["last_executed"] = datetime.utcnow().isoformat()
            
            return {
                "automation_id": automation["id"],
                "execution_results": results,
                "success_count": success_count,
                "total_actions": len(results),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to execute automation: {e}")
            automation["status"] = AutomationStatus.FAILED.value
            automation["failure_count"] += 1
            self.automation_stats["failed_automations"] += 1
            raise
    
    async def _ai_decision_making(
        self,
        action: Dict[str, Any],
        trigger_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """AI-powered decision making for automation actions"""
        try:
            # Create decision prompt
            prompt = f"""
            Should this automation action be executed?
            
            Action: {action['type']}
            Parameters: {action.get('parameters', {})}
            Trigger Data: {trigger_data}
            Context: {context}
            
            Respond with 'yes' or 'no' and a brief reason.
            """
            
            # Get AI decision
            decision = await ai_service.generate_content(
                prompt=prompt,
                provider="openai",
                max_tokens=100,
                temperature=0.3
            )
            
            # Parse decision
            decision_lower = decision.lower()
            should_execute = "yes" in decision_lower and "no" not in decision_lower
            
            logger.info(f"AI decision for action {action['type']}: {should_execute} - {decision}")
            return should_execute
        
        except Exception as e:
            logger.error(f"AI decision making failed: {e}")
            return True  # Default to executing if AI fails
    
    async def _execute_action(
        self,
        action: Dict[str, Any],
        trigger_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute individual automation action"""
        try:
            action_type = action["type"]
            parameters = action.get("parameters", {})
            
            if action_type == AutomationAction.SEND_NOTIFICATION.value:
                result = await self._execute_send_notification(parameters, trigger_data, context)
            
            elif action_type == AutomationAction.EXECUTE_WORKFLOW.value:
                result = await self._execute_workflow(parameters, trigger_data, context)
            
            elif action_type == AutomationAction.GENERATE_REPORT.value:
                result = await self._execute_generate_report(parameters, trigger_data, context)
            
            elif action_type == AutomationAction.AI_ANALYSIS.value:
                result = await self._execute_ai_analysis(parameters, trigger_data, context)
            
            elif action_type == AutomationAction.AI_GENERATION.value:
                result = await self._execute_ai_generation(parameters, trigger_data, context)
            
            elif action_type == AutomationAction.SEND_EMAIL.value:
                result = await self._execute_send_email(parameters, trigger_data, context)
            
            else:
                result = {
                    "action": action_type,
                    "success": False,
                    "error": f"Unknown action type: {action_type}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                "action": action["type"],
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _execute_send_notification(
        self,
        parameters: Dict[str, Any],
        trigger_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute send notification action"""
        try:
            # Send notification using notification service
            result = await notification_service.send_notification(
                channel=parameters.get("channel", "email"),
                recipient=parameters.get("recipient", "admin@company.com"),
                subject=parameters.get("subject", "Automation Notification"),
                message=parameters.get("message", "Automation executed successfully"),
                priority=parameters.get("priority", "normal")
            )
            
            return {
                "action": AutomationAction.SEND_NOTIFICATION.value,
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Send notification action failed: {e}")
            return {
                "action": AutomationAction.SEND_NOTIFICATION.value,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _execute_workflow(
        self,
        parameters: Dict[str, Any],
        trigger_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow action"""
        try:
            # Execute workflow using workflow service
            workflow_id = parameters.get("workflow_id")
            if not workflow_id:
                raise ValueError("Workflow ID is required")
            
            result = await workflow_service.execute_workflow(workflow_id)
            
            return {
                "action": AutomationAction.EXECUTE_WORKFLOW.value,
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Execute workflow action failed: {e}")
            return {
                "action": AutomationAction.EXECUTE_WORKFLOW.value,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _execute_ai_analysis(
        self,
        parameters: Dict[str, Any],
        trigger_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute AI analysis action"""
        try:
            # Perform AI analysis
            analysis_type = parameters.get("analysis_type", "general")
            text = trigger_data.get("text", "")
            
            result = await ai_service.analyze_text(
                text=text,
                analysis_type=analysis_type,
                provider=parameters.get("provider", "openai")
            )
            
            return {
                "action": AutomationAction.AI_ANALYSIS.value,
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"AI analysis action failed: {e}")
            return {
                "action": AutomationAction.AI_ANALYSIS.value,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _execute_ai_generation(
        self,
        parameters: Dict[str, Any],
        trigger_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute AI generation action"""
        try:
            # Generate content using AI
            prompt = parameters.get("prompt_template", "Generate content")
            # Replace placeholders in prompt
            for key, value in trigger_data.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))
            
            result = await ai_service.generate_content(
                prompt=prompt,
                provider=parameters.get("provider", "openai"),
                max_tokens=parameters.get("max_tokens", 1000),
                temperature=parameters.get("temperature", 0.7)
            )
            
            return {
                "action": AutomationAction.AI_GENERATION.value,
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"AI generation action failed: {e}")
            return {
                "action": AutomationAction.AI_GENERATION.value,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _execute_generate_report(
        self,
        parameters: Dict[str, Any],
        trigger_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute generate report action"""
        try:
            # Generate report (simplified)
            report_type = parameters.get("report_type", "general")
            include_ai_insights = parameters.get("include_ai_insights", False)
            
            report_data = {
                "type": report_type,
                "generated_at": datetime.utcnow().isoformat(),
                "data": trigger_data,
                "ai_insights": None
            }
            
            if include_ai_insights:
                insights = await ai_service.analyze_text(
                    text=str(trigger_data),
                    analysis_type="insights",
                    provider="openai"
                )
                report_data["ai_insights"] = insights
            
            return {
                "action": AutomationAction.GENERATE_REPORT.value,
                "success": True,
                "result": report_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Generate report action failed: {e}")
            return {
                "action": AutomationAction.GENERATE_REPORT.value,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _execute_send_email(
        self,
        parameters: Dict[str, Any],
        trigger_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute send email action"""
        try:
            # Send email using notification service
            recipients = parameters.get("recipients", ["admin@company.com"])
            subject = parameters.get("subject", "Automation Email")
            message = parameters.get("message", "Automation executed successfully")
            
            # Replace placeholders
            for key, value in trigger_data.items():
                subject = subject.replace(f"{{{key}}}", str(value))
                message = message.replace(f"{{{key}}}", str(value))
            
            result = await notification_service.send_notification(
                channel="email",
                recipient=recipients[0] if isinstance(recipients, list) else recipients,
                subject=subject,
                message=message,
                priority="normal"
            )
            
            return {
                "action": AutomationAction.SEND_EMAIL.value,
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Send email action failed: {e}")
            return {
                "action": AutomationAction.SEND_EMAIL.value,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _update_learning_data(
        self,
        automation: Dict[str, Any],
        action: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Update learning data for automation"""
        try:
            if "learning_data" not in automation:
                automation["learning_data"] = {}
            
            action_type = action["type"]
            if action_type not in automation["learning_data"]:
                automation["learning_data"][action_type] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "success_rate": 0.0,
                    "last_updated": None
                }
            
            learning_data = automation["learning_data"][action_type]
            
            if result.get("success", False):
                learning_data["success_count"] += 1
            else:
                learning_data["failure_count"] += 1
            
            total = learning_data["success_count"] + learning_data["failure_count"]
            learning_data["success_rate"] = learning_data["success_count"] / total if total > 0 else 0.0
            learning_data["last_updated"] = datetime.utcnow().isoformat()
        
        except Exception as e:
            logger.error(f"Failed to update learning data: {e}")
    
    async def get_automation(self, automation_id: str) -> Optional[Dict[str, Any]]:
        """Get automation information"""
        try:
            return self.automations.get(automation_id)
        
        except Exception as e:
            logger.error(f"Failed to get automation: {e}")
            return None
    
    async def list_automations(
        self,
        status: Optional[AutomationStatus] = None,
        trigger_type: Optional[AutomationTrigger] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List automations with filtering"""
        try:
            filtered_automations = []
            
            for automation in self.automations.values():
                if status and automation["status"] != status.value:
                    continue
                if trigger_type and automation["trigger"]["type"] != trigger_type.value:
                    continue
                
                filtered_automations.append({
                    "id": automation["id"],
                    "name": automation["name"],
                    "description": automation["description"],
                    "status": automation["status"],
                    "trigger_type": automation["trigger"]["type"],
                    "ai_decision_making": automation["ai_decision_making"],
                    "learning_enabled": automation["learning_enabled"],
                    "created_at": automation["created_at"],
                    "last_triggered": automation["last_triggered"],
                    "execution_count": automation["execution_count"],
                    "success_count": automation["success_count"],
                    "failure_count": automation["failure_count"]
                })
            
            # Sort by created_at (newest first)
            filtered_automations.sort(key=lambda x: x["created_at"], reverse=True)
            
            return filtered_automations[:limit]
        
        except Exception as e:
            logger.error(f"Failed to list automations: {e}")
            return []
    
    async def get_automation_templates(self) -> Dict[str, Any]:
        """Get available automation templates"""
        try:
            return {
                "templates": self.automation_templates,
                "total_templates": len(self.automation_templates),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get automation templates: {e}")
            return {"error": str(e)}
    
    async def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation service statistics"""
        try:
            return {
                "total_automations": self.automation_stats["total_automations"],
                "active_automations": self.automation_stats["active_automations"],
                "triggered_automations": self.automation_stats["triggered_automations"],
                "completed_automations": self.automation_stats["completed_automations"],
                "failed_automations": self.automation_stats["failed_automations"],
                "automations_by_trigger": self.automation_stats["automations_by_trigger"],
                "automations_by_action": self.automation_stats["automations_by_action"],
                "available_templates": len(self.automation_templates),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get automation stats: {e}")
            return {"error": str(e)}


# Global intelligent automation service instance
intelligent_automation_service = IntelligentAutomationService()

