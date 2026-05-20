"""
Prompt Optimizer
================
Optimizes prompts using OpenRouter
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Optimizes prompts using OpenRouter.
    """
    
    def __init__(self, openrouter_client: Optional[Any] = None, enabled: bool = True):
        """
        Initialize prompt optimizer.
        
        Args:
            openrouter_client: Optional OpenRouter client instance
            enabled: Whether optimization is enabled
        """
        self.openrouter_client = openrouter_client
        self.enabled = enabled and openrouter_client is not None
    
    async def optimize(
        self,
        prompt: str,
        character_name: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Optimize prompt using OpenRouter.
        
        Args:
            prompt: Original prompt to optimize
            character_name: Optional character name for context
            context: Optional additional context
            
        Returns:
            Tuple of (optimized_prompt, was_optimized)
        """
        if not self.enabled:
            return prompt, False
        
        try:
            # Build optimization context
            optimization_prompt = self._build_optimization_prompt(
                prompt, character_name, context
            )
            
            # Call OpenRouter
            response = await self.openrouter_client.chat.completions.create(
                model="openai/gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at optimizing prompts for AI image generation. "
                                 "Create concise, detailed prompts that will produce high-quality images."
                    },
                    {
                        "role": "user",
                        "content": optimization_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            optimized = response.choices[0].message.content.strip()
            
            if optimized and len(optimized) > 0:
                logger.info(f"Prompt optimized: {prompt[:50]}... -> {optimized[:50]}...")
                return optimized, True
            else:
                logger.warning("OpenRouter returned empty optimization, using original")
                return prompt, False
                
        except Exception as e:
            logger.warning(f"Failed to optimize prompt with OpenRouter: {e}")
            return prompt, False
    
    def _build_optimization_prompt(
        self,
        prompt: str,
        character_name: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Build optimization prompt for OpenRouter.
        
        Args:
            prompt: Original prompt
            character_name: Optional character name
            context: Optional context
            
        Returns:
            Optimization prompt text
        """
        parts = []
        
        if context:
            parts.append(f"Context: {context}")
        
        if character_name:
            parts.append(f"Character: {character_name}")
        
        parts.append(f"Original prompt: {prompt}")
        parts.append(
            "Please optimize this prompt for AI image generation. "
            "Make it more detailed and specific while keeping it concise. "
            "Focus on visual details that will improve the quality of the generated image."
        )
        
        return "\n\n".join(parts)

