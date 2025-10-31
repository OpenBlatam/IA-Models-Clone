"""
Notification Service
====================

Advanced notification system for workflow alerts, document completion, and system events.
"""

import asyncio
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging
import httpx

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    PUSH = "push"
    IN_APP = "in_app"

class NotificationPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class NotificationTemplate:
    id: str
    name: str
    notification_type: NotificationType
    subject_template: str
    body_template: str
    variables: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class NotificationRequest:
    id: str
    notification_type: NotificationType
    recipients: List[str]
    subject: str
    body: str
    priority: NotificationPriority
    template_id: Optional[str]
    variables: Dict[str, Any]
    scheduled_at: Optional[datetime]
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class NotificationResult:
    id: str
    request_id: str
    status: str
    sent_at: datetime
    error_message: Optional[str]
    delivery_attempts: int
    metadata: Dict[str, Any]

class NotificationService:
    """
    Advanced notification service for business agents system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates: Dict[str, NotificationTemplate] = {}
        self.requests: Dict[str, NotificationRequest] = {}
        self.results: Dict[str, NotificationResult] = {}
        
        # Initialize notification providers
        self._initialize_providers()
        
        # Load default templates
        self._load_default_templates()
        
    def _initialize_providers(self):
        """Initialize notification providers."""
        
        # Email provider
        self.smtp_config = {
            "host": self.config.get("smtp_host"),
            "port": self.config.get("smtp_port", 587),
            "username": self.config.get("smtp_username"),
            "password": self.config.get("smtp_password"),
            "use_tls": self.config.get("smtp_use_tls", True)
        }
        
        # Slack provider
        self.slack_webhook_url = self.config.get("slack_webhook_url")
        
        # Teams provider
        self.teams_webhook_url = self.config.get("teams_webhook_url")
        
        # SMS provider (placeholder)
        self.sms_config = {
            "provider": self.config.get("sms_provider"),
            "api_key": self.config.get("sms_api_key"),
            "from_number": self.config.get("sms_from_number")
        }
        
    def _load_default_templates(self):
        """Load default notification templates."""
        
        # Workflow completion template
        workflow_complete_template = NotificationTemplate(
            id="workflow_complete",
            name="Workflow Completion",
            notification_type=NotificationType.EMAIL,
            subject_template="Workflow '{workflow_name}' Completed",
            body_template="""
            <h2>Workflow Completed Successfully</h2>
            <p><strong>Workflow:</strong> {workflow_name}</p>
            <p><strong>Business Area:</strong> {business_area}</p>
            <p><strong>Duration:</strong> {duration}</p>
            <p><strong>Steps Completed:</strong> {steps_completed}/{total_steps}</p>
            <p><strong>Status:</strong> {status}</p>
            
            <h3>Results Summary:</h3>
            <ul>
            {results_summary}
            </ul>
            
            <p>You can view the full details in the Business Agents System.</p>
            """,
            variables=["workflow_name", "business_area", "duration", "steps_completed", "total_steps", "status", "results_summary"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.templates[workflow_complete_template.id] = workflow_complete_template
        
        # Document generation template
        document_ready_template = NotificationTemplate(
            id="document_ready",
            name="Document Ready",
            notification_type=NotificationType.EMAIL,
            subject_template="Document '{document_title}' is Ready",
            body_template="""
            <h2>Document Generated Successfully</h2>
            <p><strong>Document:</strong> {document_title}</p>
            <p><strong>Type:</strong> {document_type}</p>
            <p><strong>Business Area:</strong> {business_area}</p>
            <p><strong>Format:</strong> {format}</p>
            <p><strong>Size:</strong> {size_bytes} bytes</p>
            
            <p><a href="{download_url}">Download Document</a></p>
            
            <p>The document has been generated and is ready for use.</p>
            """,
            variables=["document_title", "document_type", "business_area", "format", "size_bytes", "download_url"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.templates[document_ready_template.id] = document_ready_template
        
        # Workflow error template
        workflow_error_template = NotificationTemplate(
            id="workflow_error",
            name="Workflow Error",
            notification_type=NotificationType.EMAIL,
            subject_template="Workflow '{workflow_name}' Failed",
            body_template="""
            <h2>Workflow Execution Failed</h2>
            <p><strong>Workflow:</strong> {workflow_name}</p>
            <p><strong>Business Area:</strong> {business_area}</p>
            <p><strong>Failed Step:</strong> {failed_step}</p>
            <p><strong>Error:</strong> {error_message}</p>
            <p><strong>Time:</strong> {error_time}</p>
            
            <h3>Next Steps:</h3>
            <ul>
            <li>Review the error details</li>
            <li>Check the workflow configuration</li>
            <li>Retry the workflow if appropriate</li>
            </ul>
            
            <p>Please investigate and resolve this issue.</p>
            """,
            variables=["workflow_name", "business_area", "failed_step", "error_message", "error_time"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.templates[workflow_error_template.id] = workflow_error_template
        
        # Agent capability completion template
        agent_complete_template = NotificationTemplate(
            id="agent_complete",
            name="Agent Capability Complete",
            notification_type=NotificationType.SLACK,
            subject_template="",
            body_template="""
            🤖 *Agent Capability Completed*
            
            *Agent:* {agent_name}
            *Capability:* {capability_name}
            *Business Area:* {business_area}
            *Duration:* {duration}
            *Status:* {status}
            
            *Results:*
            {results_summary}
            
            *Next Steps:* {next_steps}
            """,
            variables=["agent_name", "capability_name", "business_area", "duration", "status", "results_summary", "next_steps"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.templates[agent_complete_template.id] = agent_complete_template
        
    async def send_notification(
        self,
        notification_type: NotificationType,
        recipients: List[str],
        subject: str,
        body: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        template_id: Optional[str] = None,
        variables: Dict[str, Any] = None,
        scheduled_at: Optional[datetime] = None
    ) -> NotificationResult:
        """Send a notification."""
        
        request_id = f"notif_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.requests)}"
        
        request = NotificationRequest(
            id=request_id,
            notification_type=notification_type,
            recipients=recipients,
            subject=subject,
            body=body,
            priority=priority,
            template_id=template_id,
            variables=variables or {},
            scheduled_at=scheduled_at,
            created_at=datetime.now(),
            metadata={}
        )
        
        self.requests[request_id] = request
        
        try:
            # Send notification based on type
            if notification_type == NotificationType.EMAIL:
                result = await self._send_email(request)
            elif notification_type == NotificationType.SLACK:
                result = await self._send_slack(request)
            elif notification_type == NotificationType.TEAMS:
                result = await self._send_teams(request)
            elif notification_type == NotificationType.SMS:
                result = await self._send_sms(request)
            elif notification_type == NotificationType.WEBHOOK:
                result = await self._send_webhook(request)
            else:
                result = NotificationResult(
                    id=f"result_{request_id}",
                    request_id=request_id,
                    status="unsupported",
                    sent_at=datetime.now(),
                    error_message=f"Unsupported notification type: {notification_type}",
                    delivery_attempts=1,
                    metadata={}
                )
            
            self.results[result.id] = result
            return result
            
        except Exception as e:
            logger.error(f"Notification sending failed: {str(e)}")
            result = NotificationResult(
                id=f"result_{request_id}",
                request_id=request_id,
                status="failed",
                sent_at=datetime.now(),
                error_message=str(e),
                delivery_attempts=1,
                metadata={}
            )
            self.results[result.id] = result
            return result
            
    async def send_template_notification(
        self,
        template_id: str,
        recipients: List[str],
        variables: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> NotificationResult:
        """Send notification using a template."""
        
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Replace variables in template
        subject = self._replace_variables(template.subject_template, variables)
        body = self._replace_variables(template.body_template, variables)
        
        return await self.send_notification(
            notification_type=template.notification_type,
            recipients=recipients,
            subject=subject,
            body=body,
            priority=priority,
            template_id=template_id,
            variables=variables
        )
        
    def _replace_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """Replace variables in template string."""
        
        result = template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        
        return result
        
    async def _send_email(self, request: NotificationRequest) -> NotificationResult:
        """Send email notification."""
        
        try:
            if not all([self.smtp_config["host"], self.smtp_config["username"], self.smtp_config["password"]]):
                raise ValueError("SMTP configuration incomplete")
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = request.subject
            msg['From'] = self.smtp_config["username"]
            msg['To'] = ', '.join(request.recipients)
            
            # Add HTML body
            html_body = MIMEText(request.body, 'html')
            msg.attach(html_body)
            
            # Send email
            with smtplib.SMTP(self.smtp_config["host"], self.smtp_config["port"]) as server:
                if self.smtp_config["use_tls"]:
                    server.starttls()
                server.login(self.smtp_config["username"], self.smtp_config["password"])
                server.send_message(msg)
            
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="sent",
                sent_at=datetime.now(),
                error_message=None,
                delivery_attempts=1,
                metadata={"recipients_count": len(request.recipients)}
            )
            
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="failed",
                sent_at=datetime.now(),
                error_message=str(e),
                delivery_attempts=1,
                metadata={}
            )
            
    async def _send_slack(self, request: NotificationRequest) -> NotificationResult:
        """Send Slack notification."""
        
        try:
            if not self.slack_webhook_url:
                raise ValueError("Slack webhook URL not configured")
            
            # Format message for Slack
            slack_message = {
                "text": request.subject,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": request.body
                        }
                    }
                ]
            }
            
            # Send to Slack
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.slack_webhook_url,
                    json=slack_message,
                    timeout=30
                )
                response.raise_for_status()
            
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="sent",
                sent_at=datetime.now(),
                error_message=None,
                delivery_attempts=1,
                metadata={"slack_response": response.status_code}
            )
            
        except Exception as e:
            logger.error(f"Slack notification failed: {str(e)}")
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="failed",
                sent_at=datetime.now(),
                error_message=str(e),
                delivery_attempts=1,
                metadata={}
            )
            
    async def _send_teams(self, request: NotificationRequest) -> NotificationResult:
        """Send Microsoft Teams notification."""
        
        try:
            if not self.teams_webhook_url:
                raise ValueError("Teams webhook URL not configured")
            
            # Format message for Teams
            teams_message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "0076D7",
                "summary": request.subject,
                "sections": [
                    {
                        "activityTitle": request.subject,
                        "activitySubtitle": "Business Agents System",
                        "text": request.body,
                        "markdown": True
                    }
                ]
            }
            
            # Send to Teams
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.teams_webhook_url,
                    json=teams_message,
                    timeout=30
                )
                response.raise_for_status()
            
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="sent",
                sent_at=datetime.now(),
                error_message=None,
                delivery_attempts=1,
                metadata={"teams_response": response.status_code}
            )
            
        except Exception as e:
            logger.error(f"Teams notification failed: {str(e)}")
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="failed",
                sent_at=datetime.now(),
                error_message=str(e),
                delivery_attempts=1,
                metadata={}
            )
            
    async def _send_sms(self, request: NotificationRequest) -> NotificationResult:
        """Send SMS notification."""
        
        try:
            # Placeholder for SMS implementation
            # This would integrate with SMS providers like Twilio, AWS SNS, etc.
            
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="sent",
                sent_at=datetime.now(),
                error_message=None,
                delivery_attempts=1,
                metadata={"sms_provider": "placeholder"}
            )
            
        except Exception as e:
            logger.error(f"SMS sending failed: {str(e)}")
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="failed",
                sent_at=datetime.now(),
                error_message=str(e),
                delivery_attempts=1,
                metadata={}
            )
            
    async def _send_webhook(self, request: NotificationRequest) -> NotificationResult:
        """Send webhook notification."""
        
        try:
            webhook_url = request.metadata.get("webhook_url")
            if not webhook_url:
                raise ValueError("Webhook URL not provided")
            
            # Prepare webhook payload
            payload = {
                "notification_id": request.id,
                "type": request.notification_type.value,
                "subject": request.subject,
                "body": request.body,
                "priority": request.priority.value,
                "recipients": request.recipients,
                "timestamp": request.created_at.isoformat(),
                "metadata": request.metadata
            }
            
            # Send webhook
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
            
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="sent",
                sent_at=datetime.now(),
                error_message=None,
                delivery_attempts=1,
                metadata={"webhook_response": response.status_code}
            )
            
        except Exception as e:
            logger.error(f"Webhook sending failed: {str(e)}")
            return NotificationResult(
                id=f"result_{request.id}",
                request_id=request.id,
                status="failed",
                sent_at=datetime.now(),
                error_message=str(e),
                delivery_attempts=1,
                metadata={}
            )
            
    def get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """Get notification template by ID."""
        return self.templates.get(template_id)
        
    def list_templates(self) -> List[NotificationTemplate]:
        """List all notification templates."""
        return list(self.templates.values())
        
    def create_template(
        self,
        name: str,
        notification_type: NotificationType,
        subject_template: str,
        body_template: str,
        variables: List[str]
    ) -> NotificationTemplate:
        """Create a new notification template."""
        
        template_id = f"template_{len(self.templates) + 1}"
        
        template = NotificationTemplate(
            id=template_id,
            name=name,
            notification_type=notification_type,
            subject_template=subject_template,
            body_template=body_template,
            variables=variables,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.templates[template_id] = template
        return template
        
    def get_notification_result(self, result_id: str) -> Optional[NotificationResult]:
        """Get notification result by ID."""
        return self.results.get(result_id)
        
    def list_notification_results(
        self,
        status: Optional[str] = None,
        notification_type: Optional[NotificationType] = None
    ) -> List[NotificationResult]:
        """List notification results with optional filters."""
        
        results = list(self.results.values())
        
        if status:
            results = [r for r in results if r.status == status]
            
        if notification_type:
            # Get request to check notification type
            filtered_results = []
            for result in results:
                request = self.requests.get(result.request_id)
                if request and request.notification_type == notification_type:
                    filtered_results.append(result)
            results = filtered_results
            
        return results
        
    async def send_workflow_completion_notification(
        self,
        workflow_name: str,
        business_area: str,
        duration: str,
        steps_completed: int,
        total_steps: int,
        status: str,
        results_summary: str,
        recipients: List[str]
    ) -> NotificationResult:
        """Send workflow completion notification."""
        
        variables = {
            "workflow_name": workflow_name,
            "business_area": business_area,
            "duration": duration,
            "steps_completed": steps_completed,
            "total_steps": total_steps,
            "status": status,
            "results_summary": results_summary
        }
        
        return await self.send_template_notification(
            template_id="workflow_complete",
            recipients=recipients,
            variables=variables,
            priority=NotificationPriority.HIGH if status == "failed" else NotificationPriority.NORMAL
        )
        
    async def send_document_ready_notification(
        self,
        document_title: str,
        document_type: str,
        business_area: str,
        format: str,
        size_bytes: int,
        download_url: str,
        recipients: List[str]
    ) -> NotificationResult:
        """Send document ready notification."""
        
        variables = {
            "document_title": document_title,
            "document_type": document_type,
            "business_area": business_area,
            "format": format,
            "size_bytes": size_bytes,
            "download_url": download_url
        }
        
        return await self.send_template_notification(
            template_id="document_ready",
            recipients=recipients,
            variables=variables
        )
        
    async def send_workflow_error_notification(
        self,
        workflow_name: str,
        business_area: str,
        failed_step: str,
        error_message: str,
        recipients: List[str]
    ) -> NotificationResult:
        """Send workflow error notification."""
        
        variables = {
            "workflow_name": workflow_name,
            "business_area": business_area,
            "failed_step": failed_step,
            "error_message": error_message,
            "error_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return await self.send_template_notification(
            template_id="workflow_error",
            recipients=recipients,
            variables=variables,
            priority=NotificationPriority.URGENT
        )





























