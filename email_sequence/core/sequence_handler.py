"""
Optimized Sequence Handler for Email Sequence System

Manages tokenized sequences, batch processing, and sequence optimization
for efficient text processing and model training.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from uuid import UUID

from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate

logger = logging.getLogger(__name__)

# Constants
MAX_CONNECTIONS = 1000
MAX_RETRIES = 3
BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 512


@dataclass
class SequenceBatch:
    """Optimized batch of tokenized sequences"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None


@dataclass
class SequenceConfig:
    """Configuration for sequence handling"""
    batch_size: int = BATCH_SIZE
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    collate_fn: Optional[callable] = None


class OptimizedEmailSequenceDataset(Dataset):
    """Optimized dataset for email sequences with memory efficiency"""
    
    def __init__(
        self, 
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        config: SequenceConfig
    ):
        self.sequences = sequences
        self.subscribers = subscribers
        self.templates = templates
        self.config = config
        
        # Pre-process data for efficiency
        self._preprocess_data()
    
    def _preprocess_data(self) -> None:
        """Preprocess data for efficient access"""
        self.processed_data = []
        
        for i, sequence in enumerate(self.sequences):
            subscriber = self.subscribers[i % len(self.subscribers)]
            template = self.templates[i % len(self.templates)]
            
            # Process each step
            for step in sequence.steps:
                if step.is_active:
                    self.processed_data.append({
                        "sequence_id": sequence.id,
                        "step_order": step.order,
                        "subscriber_id": subscriber.id,
                        "template_id": template.id,
                        "step_data": self._extract_step_data(step),
                        "subscriber_data": self._extract_subscriber_data(subscriber),
                        "template_data": self._extract_template_data(template)
                    })
    
    def _extract_step_data(self, step: SequenceStep) -> Dict[str, Any]:
        """Extract relevant data from step"""
        return {
            "step_type": step.step_type,
            "subject": step.subject,
            "content": step.content,
            "delay_hours": step.delay_hours,
            "delay_days": step.delay_days,
            "condition_expression": step.condition_expression,
            "action_type": step.action_type,
            "webhook_url": step.webhook_url
        }
    
    def _extract_subscriber_data(self, subscriber: Subscriber) -> Dict[str, Any]:
        """Extract relevant data from subscriber"""
        return {
            "email": subscriber.email,
            "first_name": subscriber.first_name,
            "last_name": subscriber.last_name,
            "status": subscriber.status,
            "preferences": subscriber.preferences
        }
    
    def _extract_template_data(self, template: EmailTemplate) -> Dict[str, Any]:
        """Extract relevant data from template"""
        return {
            "name": template.name,
            "subject": template.subject,
            "html_content": template.html_content,
            "text_content": template.text_content,
            "status": template.status
        }
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with error handling"""
        try:
            return self.processed_data[idx]
        except IndexError:
            logger.error(f"Index {idx} out of range for dataset")
            return {}


class OptimizedSequenceHandler:
    """Optimized handler for sequence processing"""
    
    def __init__(self, config: SequenceConfig):
        self.config = config
        self.sequences: Dict[UUID, EmailSequence] = {}
        self.subscribers: Dict[UUID, Subscriber] = {}
        self.templates: Dict[UUID, EmailTemplate] = {}
        
        # Processing queues
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.results_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        
        # Statistics
        self.stats = {
            "sequences_processed": 0,
            "steps_processed": 0,
            "errors": 0,
            "processing_time": 0.0
        }
    
    async def add_sequence(self, sequence: EmailSequence) -> bool:
        """Add sequence to handler"""
        try:
            self.sequences[sequence.id] = sequence
            await self.processing_queue.put(sequence)
            return True
        except Exception as e:
            logger.error(f"Error adding sequence: {e}")
            return False
    
    async def add_subscriber(self, subscriber: Subscriber) -> bool:
        """Add subscriber to handler"""
        try:
            self.subscribers[subscriber.id] = subscriber
            return True
        except Exception as e:
            logger.error(f"Error adding subscriber: {e}")
            return False
    
    async def add_template(self, template: EmailTemplate) -> bool:
        """Add template to handler"""
        try:
            self.templates[template.id] = template
            return True
        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False
    
    async def process_sequences(self) -> None:
        """Process all sequences in queue"""
        while not self.processing_queue.empty():
            try:
                sequence = await self.processing_queue.get()
                await self._process_single_sequence(sequence)
                self.stats["sequences_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing sequence: {e}")
                self.stats["errors"] += 1
    
    async def _process_single_sequence(self, sequence: EmailSequence) -> None:
        """Process a single sequence"""
        try:
            for step in sequence.steps:
                if step.is_active:
                    await self._process_step(sequence, step)
                    self.stats["steps_processed"] += 1
                    
        except Exception as e:
            logger.error(f"Error processing sequence {sequence.id}: {e}")
    
    async def _process_step(self, sequence: EmailSequence, step: SequenceStep) -> None:
        """Process a single step"""
        try:
            if step.step_type == "email":
                await self._process_email_step(sequence, step)
            elif step.step_type == "delay":
                await self._process_delay_step(sequence, step)
            elif step.step_type == "condition":
                await self._process_condition_step(sequence, step)
            elif step.step_type == "action":
                await self._process_action_step(sequence, step)
            elif step.step_type == "webhook":
                await self._process_webhook_step(sequence, step)
                
        except Exception as e:
            logger.error(f"Error processing step {step.id}: {e}")
    
    async def _process_email_step(self, sequence: EmailSequence, step: SequenceStep) -> None:
        """Process email step with optimization"""
        try:
            # Get active subscribers for this sequence
            active_subscribers = [
                sub for sub in self.subscribers.values()
                if sub.status == "active"
            ]
            
            # Process in batches for efficiency
            for i in range(0, len(active_subscribers), self.config.batch_size):
                batch = active_subscribers[i:i + self.config.batch_size]
                
                # Process batch
                for subscriber in batch:
                    await self._send_email_to_subscriber(sequence, step, subscriber)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error processing email step: {e}")
    
    async def _process_delay_step(self, sequence: EmailSequence, step: SequenceStep) -> None:
        """Process delay step"""
        try:
            delay_seconds = (step.delay_hours or 0) * 3600 + (step.delay_days or 0) * 86400
            await asyncio.sleep(delay_seconds)
            
        except Exception as e:
            logger.error(f"Error processing delay step: {e}")
    
    async def _process_condition_step(self, sequence: EmailSequence, step: SequenceStep) -> None:
        """Process condition step"""
        try:
            # Evaluate condition (simplified for optimization)
            condition_result = self._evaluate_condition(step.condition_expression)
            
            if condition_result:
                # Continue with next step
                pass
            else:
                # Skip to alternative step
                pass
                
        except Exception as e:
            logger.error(f"Error processing condition step: {e}")
    
    async def _process_action_step(self, sequence: EmailSequence, step: SequenceStep) -> None:
        """Process action step"""
        try:
            # Execute action (simplified for optimization)
            await self._execute_action(step.action_type, step.action_data)
            
        except Exception as e:
            logger.error(f"Error processing action step: {e}")
    
    async def _process_webhook_step(self, sequence: EmailSequence, step: SequenceStep) -> None:
        """Process webhook step"""
        try:
            # Execute webhook (simplified for optimization)
            await self._execute_webhook(step.webhook_url, step.webhook_method)
            
        except Exception as e:
            logger.error(f"Error processing webhook step: {e}")
    
    async def _send_email_to_subscriber(
        self,
        sequence: EmailSequence,
        step: SequenceStep,
        subscriber: Subscriber
    ) -> None:
        """Send email to subscriber with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                # Personalize content
                personalized_content = self._personalize_content(
                    step.content,
                    subscriber,
                    sequence.personalization_variables
                )
                
                # Send email (simplified for optimization)
                await self._send_email(
                    to_email=subscriber.email,
                    subject=step.subject,
                    content=personalized_content
                )
                break
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to send email after {MAX_RETRIES} attempts: {e}")
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _personalize_content(
        self,
        content: str,
        subscriber: Subscriber,
        variables: Optional[Dict[str, Any]]
    ) -> str:
        """Personalize content with optimization"""
        try:
            if not variables:
                return content
            
            # Simple variable replacement for optimization
            personalized = content
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                personalized = personalized.replace(placeholder, str(value))
            
            # Add subscriber-specific personalization
            personalized = personalized.replace("{{first_name}}", subscriber.first_name or "")
            personalized = personalized.replace("{{last_name}}", subscriber.last_name or "")
            personalized = personalized.replace("{{email}}", subscriber.email)
            
            return personalized
            
        except Exception as e:
            logger.error(f"Error personalizing content: {e}")
            return content
    
    def _evaluate_condition(self, condition_expression: Optional[str]) -> bool:
        """Evaluate condition (simplified)"""
        try:
            if not condition_expression:
                return True
            
            # Simple condition evaluation for optimization
            # In a real implementation, this would use a proper expression evaluator
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def _execute_action(self, action_type: Optional[str], action_data: Optional[Dict[str, Any]]) -> None:
        """Execute action (simplified)"""
        try:
            if not action_type:
                return
            
            # Simple action execution for optimization
            logger.info(f"Executing action: {action_type}")
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    
    async def _execute_webhook(self, webhook_url: Optional[str], method: Optional[str]) -> None:
        """Execute webhook (simplified)"""
        try:
            if not webhook_url:
                return
            
            # Simple webhook execution for optimization
            logger.info(f"Executing webhook: {webhook_url}")
            
        except Exception as e:
            logger.error(f"Error executing webhook: {e}")
    
    async def _send_email(self, to_email: str, subject: str, content: str) -> None:
        """Send email (simplified)"""
        try:
            # Simple email sending for optimization
            logger.info(f"Sending email to {to_email}: {subject}")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            **self.stats,
            "total_sequences": len(self.sequences),
            "total_subscribers": len(self.subscribers),
            "total_templates": len(self.templates),
            "queue_size": self.processing_queue.qsize()
        }
    
    def create_dataset(self) -> OptimizedEmailSequenceDataset:
        """Create dataset from current data"""
        sequences_list = list(self.sequences.values())
        subscribers_list = list(self.subscribers.values())
        templates_list = list(self.templates.values())
        
        return OptimizedEmailSequenceDataset(
            sequences=sequences_list,
            subscribers=subscribers_list,
            templates=templates_list,
            config=self.config
        )
    
    def create_dataloader(self, dataset: OptimizedEmailSequenceDataset) -> DataLoader:
        """Create dataloader from dataset"""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            collate_fn=self.config.collate_fn
        ) 