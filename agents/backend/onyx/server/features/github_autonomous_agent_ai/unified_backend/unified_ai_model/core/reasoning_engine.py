"""
Reasoning Engine for Long-Horizon Reasoning
Implements concepts from WebResearcher paper
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .knowledge_base import KnowledgeBase
from .llm_service import LLMService
from .async_helpers import safe_async_call
from ..config import get_config

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
        llm_service: LLMService
    ):
        self.knowledge_base = knowledge_base
        self.llm_service = llm_service
        self.config = get_config()
    
    async def reason(
        self,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None,
        context_data: Optional[str] = None
    ) -> ReasoningResult:
        """
        Perform long-horizon reasoning with self-correction
        
        Args:
            instruction: Task instruction
            metadata: Optional metadata
            context_data: Optional additional context (e.g. similar experiences)
        
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
        
        # 1. Generate Initial Draft
        draft_response = await self._generate_response(context, context_data)
        
        # 2. Critique Draft
        critique = await self._critique_response(instruction, draft_response.response)
        
        # 3. Refine if necessary
        if critique["needs_improvement"]:
            logger.info(f"Refining response based on critique: {critique['feedback']}")
            final_response = await self._refine_response(
                instruction, 
                draft_response.response, 
                critique["feedback"]
            )
            
            # Update result
            draft_response.response = final_response.content
            draft_response.tokens_used += final_response.usage.get("total_tokens", 0) if final_response.usage else 0
            draft_response.reasoning_steps.append(f"Critique: {critique['feedback']}")
            draft_response.reasoning_steps.append("Refined response")
        
        return draft_response

    async def _critique_response(self, instruction: str, response: str) -> Dict[str, Any]:
        """Critique the generated response"""
        prompt = f"""Critique the following response to the instruction: "{instruction}"
        
        Response:
        {response}
        
        Identify any logical errors, missing information, or potential issues.
        Return JSON with:
        - needs_improvement: boolean
        - feedback: string (concise critique)
        """
        
        result = await self.llm_service.generate(
            prompt=prompt,
            model=self.config.default_model,
            temperature=0.3
        )
        
        try:
            # Simple parsing - in production use structured output
            content = result.content.lower()
            needs_improvement = "true" in content or "yes" in content
            return {
                "needs_improvement": needs_improvement,
                "feedback": result.content
            }
        except Exception:
            return {"needs_improvement": False, "feedback": ""}

    async def _refine_response(self, instruction: str, original_response: str, feedback: str) -> Any:
        """Refine response based on feedback"""
        prompt = f"""Refine the following response based on the feedback.
        
        Instruction: {instruction}
        Original Response: {original_response}
        Feedback: {feedback}
        
        Provide the improved response only."""
        
        return await self.llm_service.generate(
            prompt=prompt,
            model=self.config.default_model,
            temperature=0.7
        )
    
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
        context: ReasoningContext,
        context_data: Optional[str] = None
    ) -> ReasoningResult:
        """Generate response using LLMService with context"""
        # Build system prompt with knowledge context
        knowledge_context = "\n".join(context.relevant_knowledge) if context.relevant_knowledge else "No previous knowledge available."
        
        system_prompt = f"""You are an autonomous long-term agent operating continuously.
You have access to accumulated knowledge from previous operations.

Previous Knowledge:
{knowledge_context}

{context_data if context_data else ""}

Current Instruction: {context.instruction}

Provide a thoughtful response considering long-term implications and learning opportunities."""
        
        # Call LLMService
        response = await self.llm_service.generate(
            prompt=context.instruction,
            system_prompt=system_prompt,
            model=self.config.default_model,
            temperature=0.7
        )
        
        if not response.success:
            raise ValueError(f"LLM generation failed: {response.error}")
        
        return ReasoningResult(
            response=response.content,
            tokens_used=response.usage.get("total_tokens", 0) if response.usage else 0,
            reasoning_steps=[], # LLMService doesn't return steps explicitly unless structured
            confidence=0.8
        )
