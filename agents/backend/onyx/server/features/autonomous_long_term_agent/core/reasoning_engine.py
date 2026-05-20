"""
Reasoning Engine for Long-Horizon Reasoning
Implements concepts from WebResearcher paper
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..infrastructure.storage.knowledge_base import KnowledgeBase
from ..infrastructure.openrouter.client import OpenRouterClient
from .async_helpers import safe_async_call
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ReasoningContext:
    """Context for reasoning operations"""
    instruction: str
    relevant_knowledge: List[str]
    metadata: Dict[str, Any]


@dataclass
class ReasoningResult:
    """Result of reasoning operation"""
    response: str
    tokens_used: int
    reasoning_steps: List[str]
    confidence: float


class ReasoningEngine:
    """
    Long-horizon reasoning engine
    Implements concepts from WebResearcher paper
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        openrouter_client: OpenRouterClient
    ):
        self.knowledge_base = knowledge_base
        self.openrouter_client = openrouter_client
    
    async def reason(
        self,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """
        Perform long-horizon reasoning
        
        Args:
            instruction: Task instruction
            metadata: Optional metadata
        
        Returns:
            ReasoningResult with response and metadata
        """
        # Retrieve relevant knowledge
        relevant_knowledge = await self._retrieve_knowledge(instruction)
        
        # Build context
        context = ReasoningContext(
            instruction=instruction,
            relevant_knowledge=relevant_knowledge,
            metadata=metadata or {}
        )
        
        # Generate response
        result = await self._generate_response(context)
        
        return result
    
    async def _retrieve_knowledge(
        self,
        instruction: str,
        limit: int = 5
    ) -> List[str]:
        """Retrieve relevant knowledge from knowledge base"""
        knowledge_entries = await safe_async_call(
            self.knowledge_base.search_knowledge,
            instruction,
            limit=limit,
            default=[],
            error_message="Error retrieving knowledge from knowledge base"
        )
        
        if knowledge_entries:
            return [entry.content for entry in knowledge_entries]
        return []
    
    async def _generate_response(
        self,
        context: ReasoningContext
    ) -> ReasoningResult:
        """Generate response using OpenRouter with context"""
        # Build system prompt with knowledge context
        knowledge_context = "\n".join(context.relevant_knowledge) if context.relevant_knowledge else "No previous knowledge available."
        
        system_prompt = f"""You are an autonomous long-term agent operating continuously.
You have access to accumulated knowledge from previous operations.

Previous Knowledge:
{knowledge_context}

Current Instruction: {context.instruction}

Provide a thoughtful response considering long-term implications and learning opportunities."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context.instruction}
        ]
        
        # Call OpenRouter
        result = await self.openrouter_client.chat_completion(
            model=settings.openrouter_model,
            messages=messages,
            temperature=settings.openrouter_temperature,
            max_tokens=settings.openrouter_max_tokens,
            http_referer=settings.openrouter_http_referer,
            app_name="Autonomous Long-Term Agent"
        )
        
        if not result:
            # If result is None/empty, raise an error
            raise ValueError("OpenRouter returned empty result")
        
        return ReasoningResult(
            response=result.get("response", ""),
            tokens_used=result.get("tokens_used", 0),
            reasoning_steps=result.get("reasoning_steps", []),
            confidence=result.get("confidence", 0.8)
        )

