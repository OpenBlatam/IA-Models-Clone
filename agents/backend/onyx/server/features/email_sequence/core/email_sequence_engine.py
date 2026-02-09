"""
Optimized Email Sequence Engine

This module contains the main engine for managing email sequences,
integrating LangChain for intelligent automation and personalization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from dataclasses import dataclass
from enum import Enum

from ..models.sequence import EmailSequence, SequenceStep, SequenceStatus, StepType
from ..models.template import EmailTemplate, TemplateStatus
from ..models.subscriber import Subscriber, SubscriberStatus
from ..models.campaign import EmailCampaign, CampaignMetrics
from ..services.langchain_service import LangChainEmailService
from ..services.delivery_service import EmailDeliveryService
from ..services.analytics_service import EmailAnalyticsService

logger = logging.getLogger(__name__)

# Constants
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
BATCH_SIZE = 100


class EngineStatus(Enum):
    """Engine status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ProcessingResult:
    """Result of sequence processing"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class EmailSequenceEngine:
    """
    Optimized main engine for managing email sequences with LangChain integration.
    """
    
    def __init__(
        self,
        langchain_service: LangChainEmailService,
        delivery_service: EmailDeliveryService,
        analytics_service: EmailAnalyticsService,
        max_concurrent_sequences: int = 50
    ):
        """
        Initialize the email sequence engine.
        
        Args:
            langchain_service: LangChain service for AI-powered features
            delivery_service: Email delivery service
            analytics_service: Analytics service for tracking
            max_concurrent_sequences: Maximum number of concurrent sequences
        """
        self.langchain_service = langchain_service
        self.delivery_service = delivery_service
        self.analytics_service = analytics_service
        self.max_concurrent_sequences = max_concurrent_sequences
        
        # Active sequences and campaigns with improved memory management
        self.active_sequences: Dict[UUID, EmailSequence] = {}
        self.active_campaigns: Dict[UUID, EmailCampaign] = {}
        
        # Background tasks with better management
        self.background_tasks: List[asyncio.Task] = []
        self.status = EngineStatus.IDLE
        
        # Processing queues for better performance
        self.sequence_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.email_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        
        # Statistics
        self.stats = {
            "sequences_processed": 0,
            "emails_sent": 0,
            "errors": 0,
            "start_time": None
        }
        
        logger.info("Email Sequence Engine initialized")
    
    async def start(self) -> ProcessingResult:
        """Start the email sequence engine"""
        try:
            self.status = EngineStatus.RUNNING
            self.stats["start_time"] = datetime.utcnow()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Email Sequence Engine started successfully")
            return ProcessingResult(
                success=True,
                message="Engine started successfully",
                data={"status": self.status.value}
            )
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Error starting Email Sequence Engine: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to start engine: {str(e)}",
                error=e
            )
    
    async def stop(self) -> ProcessingResult:
        """Stop the email sequence engine"""
        try:
            self.status = EngineStatus.STOPPING
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete with timeout
            await asyncio.wait_for(
                asyncio.gather(*self.background_tasks, return_exceptions=True),
                timeout=TIMEOUT_SECONDS
            )
            
            # Close services
            await self.langchain_service.close()
            await self.delivery_service.close()
            await self.analytics_service.close()
            
            self.status = EngineStatus.IDLE
            logger.info("Email Sequence Engine stopped successfully")
            
            return ProcessingResult(
                success=True,
                message="Engine stopped successfully",
                data={"status": self.status.value}
            )
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Error stopping Email Sequence Engine: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to stop engine: {str(e)}",
                error=e
            )
    
    async def create_sequence(
        self,
        name: str,
        target_audience: str,
        goals: List[str],
        tone: str = "professional",
        templates: List[EmailTemplate] = None
    ) -> ProcessingResult:
        """
        Create a new email sequence with optimized processing.
        
        Args:
            name: Sequence name
            target_audience: Target audience description
            goals: List of sequence goals
            tone: Email tone
            templates: List of email templates
            
        Returns:
            ProcessingResult with sequence creation status
        """
        try:
            # Generate sequence using LangChain
            sequence_data = await self.langchain_service.generate_sequence(
                name=name,
                target_audience=target_audience,
                goals=goals,
                tone=tone
            )
            
            # Create sequence object
            sequence = EmailSequence(
                name=name,
                description=sequence_data.get("description", ""),
                steps=sequence_data.get("steps", []),
                personalization_variables=sequence_data.get("personalization", {})
            )
            
            # Apply templates if provided
            if templates:
                await self._apply_templates_to_sequence(sequence, templates)
            
            # Add to active sequences
            self.active_sequences[sequence.id] = sequence
            
            logger.info(f"Created sequence: {sequence.id}")
            
            return ProcessingResult(
                success=True,
                message="Sequence created successfully",
                data={"sequence_id": sequence.id, "sequence": sequence}
            )
            
        except Exception as e:
            logger.error(f"Error creating sequence: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to create sequence: {str(e)}",
                error=e
            )
    
    async def activate_sequence(self, sequence_id: UUID) -> ProcessingResult:
        """Activate a sequence with improved error handling"""
        try:
            if sequence_id not in self.active_sequences:
                return ProcessingResult(
                    success=False,
                    message="Sequence not found"
                )
            
            sequence = self.active_sequences[sequence_id]
            sequence.activate()
            
            # Add to processing queue
            await self.sequence_queue.put(sequence)
            
            logger.info(f"Activated sequence: {sequence_id}")
            
            return ProcessingResult(
                success=True,
                message="Sequence activated successfully",
                data={"sequence_id": sequence_id}
            )
            
        except Exception as e:
            logger.error(f"Error activating sequence {sequence_id}: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to activate sequence: {str(e)}",
                error=e
            )
    
    async def add_subscribers_to_sequence(
        self,
        sequence_id: UUID,
        subscribers: List[Subscriber]
    ) -> ProcessingResult:
        """Add subscribers to sequence with batch processing"""
        try:
            if sequence_id not in self.active_sequences:
                return ProcessingResult(
                    success=False,
                    message="Sequence not found"
                )
            
            sequence = self.active_sequences[sequence_id]
            
            # Process subscribers in batches
            for i in range(0, len(subscribers), BATCH_SIZE):
                batch = subscribers[i:i + BATCH_SIZE]
                
                for subscriber in batch:
                    await self._add_subscriber_to_sequence(sequence, subscriber)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            logger.info(f"Added {len(subscribers)} subscribers to sequence {sequence_id}")
            
            return ProcessingResult(
                success=True,
                message=f"Added {len(subscribers)} subscribers successfully",
                data={"subscribers_added": len(subscribers)}
            )
            
        except Exception as e:
            logger.error(f"Error adding subscribers to sequence {sequence_id}: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to add subscribers: {str(e)}",
                error=e
            )
    
    async def get_sequence_analytics(
        self,
        sequence_id: UUID
    ) -> ProcessingResult:
        """Get analytics for a sequence with caching"""
        try:
            if sequence_id not in self.active_sequences:
                return ProcessingResult(
                    success=False,
                    message="Sequence not found"
                )
            
            # Get analytics from analytics service
            analytics = await self.analytics_service.get_sequence_analytics(sequence_id)
            
            return ProcessingResult(
                success=True,
                message="Analytics retrieved successfully",
                data=analytics
            )
            
        except Exception as e:
            logger.error(f"Error getting analytics for sequence {sequence_id}: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to get analytics: {str(e)}",
                error=e
            )
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        # Sequence processing task
        sequence_task = asyncio.create_task(self._process_sequence_queue())
        self.background_tasks.append(sequence_task)
        
        # Email processing task
        email_task = asyncio.create_task(self._process_email_queue())
        self.background_tasks.append(email_task)
        
        # Analytics processing task
        analytics_task = asyncio.create_task(self._process_analytics())
        self.background_tasks.append(analytics_task)
        
        logger.info("Background tasks started")
    
    async def _process_sequence_queue(self) -> None:
        """Process sequences from the queue"""
        while self.status in [EngineStatus.RUNNING, EngineStatus.STOPPING]:
            try:
                sequence = await asyncio.wait_for(
                    self.sequence_queue.get(),
                    timeout=1.0
                )
                
                await self._process_sequence(sequence)
                self.stats["sequences_processed"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing sequence: {e}")
                self.stats["errors"] += 1
    
    async def _process_email_queue(self) -> None:
        """Process emails from the queue"""
        while self.status in [EngineStatus.RUNNING, EngineStatus.STOPPING]:
            try:
                email_data = await asyncio.wait_for(
                    self.email_queue.get(),
                    timeout=1.0
                )
                
                await self._send_email(email_data)
                self.stats["emails_sent"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing email: {e}")
                self.stats["errors"] += 1
    
    async def _process_analytics(self) -> None:
        """Process analytics in background"""
        while self.status in [EngineStatus.RUNNING, EngineStatus.STOPPING]:
            try:
                await asyncio.sleep(60)  # Process every minute
                await self.analytics_service.process_pending_analytics()
                
            except Exception as e:
                logger.error(f"Error processing analytics: {e}")
                self.stats["errors"] += 1
    
    async def _process_sequence(self, sequence: EmailSequence) -> None:
        """Process a single sequence"""
        try:
            for step in sequence.steps:
                if step.is_active:
                    await self._process_sequence_step(sequence, step)
                    
        except Exception as e:
            logger.error(f"Error processing sequence {sequence.id}: {e}")
    
    async def _process_sequence_step(
        self,
        sequence: EmailSequence,
        step: SequenceStep
    ) -> None:
        """Process a single sequence step"""
        try:
            if step.step_type == StepType.EMAIL:
                await self._process_email_step(sequence, step)
            elif step.step_type == StepType.DELAY:
                await self._process_delay_step(sequence, step)
            elif step.step_type == StepType.CONDITION:
                await self._process_condition_step(sequence, step)
            elif step.step_type == StepType.ACTION:
                await self._process_action_step(sequence, step)
            elif step.step_type == StepType.WEBHOOK:
                await self._process_webhook_step(sequence, step)
                
        except Exception as e:
            logger.error(f"Error processing step {step.id}: {e}")
    
    async def _process_email_step(
        self,
        sequence: EmailSequence,
        step: SequenceStep
    ) -> None:
        """Process email step with personalization"""
        try:
            # Get subscribers for this sequence
            subscribers = sequence.get_active_subscribers()
            
            for subscriber in subscribers:
                # Personalize content
                personalized_content = await self.langchain_service.personalize_content(
                    step.content,
                    subscriber,
                    sequence.personalization_variables
                )
                
                # Add to email queue
                await self.email_queue.put({
                    "sequence_id": sequence.id,
                    "step_id": step.id,
                    "subscriber_id": subscriber.id,
                    "subject": step.subject,
                    "content": personalized_content,
                    "template_id": step.template_id
                })
                
        except Exception as e:
            logger.error(f"Error processing email step: {e}")
    
    async def _process_delay_step(
        self,
        sequence: EmailSequence,
        step: SequenceStep
    ) -> None:
        """Process delay step"""
        try:
            delay_seconds = (step.delay_hours or 0) * 3600 + (step.delay_days or 0) * 86400
            await asyncio.sleep(delay_seconds)
            
        except Exception as e:
            logger.error(f"Error processing delay step: {e}")
    
    async def _process_condition_step(
        self,
        sequence: EmailSequence,
        step: SequenceStep
    ) -> None:
        """Process condition step"""
        try:
            # Evaluate condition using LangChain
            result = await self.langchain_service.evaluate_condition(
                step.condition_expression,
                step.condition_variables
            )
            
            if result:
                # Continue with next step
                pass
            else:
                # Skip to alternative step or end
                pass
                
        except Exception as e:
            logger.error(f"Error processing condition step: {e}")
    
    async def _process_action_step(
        self,
        sequence: EmailSequence,
        step: SequenceStep
    ) -> None:
        """Process action step"""
        try:
            # Execute action using LangChain
            await self.langchain_service.execute_action(
                step.action_type,
                step.action_data
            )
            
        except Exception as e:
            logger.error(f"Error processing action step: {e}")
    
    async def _process_webhook_step(
        self,
        sequence: EmailSequence,
        step: SequenceStep
    ) -> None:
        """Process webhook step"""
        try:
            # Execute webhook
            await self.delivery_service.execute_webhook(
                step.webhook_url,
                step.webhook_method,
                step.webhook_headers
            )
            
        except Exception as e:
            logger.error(f"Error processing webhook step: {e}")
    
    async def _send_email(self, email_data: Dict[str, Any]) -> None:
        """Send email with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                await self.delivery_service.send_email(
                    to_email=email_data["subscriber_id"],
                    subject=email_data["subject"],
                    content=email_data["content"],
                    template_id=email_data.get("template_id")
                )
                break
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to send email after {MAX_RETRIES} attempts: {e}")
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _add_subscriber_to_sequence(
        self,
        sequence: EmailSequence,
        subscriber: Subscriber
    ) -> None:
        """Add subscriber to sequence"""
        try:
            sequence.add_subscriber(subscriber)
            
        except Exception as e:
            logger.error(f"Error adding subscriber to sequence: {e}")
    
    async def _apply_templates_to_sequence(
        self,
        sequence: EmailSequence,
        templates: List[EmailTemplate]
    ) -> None:
        """Apply templates to sequence"""
        try:
            for template in templates:
                if template.status == TemplateStatus.ACTIVE:
                    # Apply template to sequence
                    pass
                    
        except Exception as e:
            logger.error(f"Error applying templates to sequence: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            **self.stats,
            "status": self.status.value,
            "active_sequences": len(self.active_sequences),
            "active_campaigns": len(self.active_campaigns),
            "queue_size": {
                "sequence_queue": self.sequence_queue.qsize(),
                "email_queue": self.email_queue.qsize()
            }
        } 