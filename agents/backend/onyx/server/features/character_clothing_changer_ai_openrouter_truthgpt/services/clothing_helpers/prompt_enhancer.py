"""
Prompt Enhancer
===============
Enhances prompts using TruthGPT
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """
    Enhances prompts using TruthGPT.
    """
    
    def __init__(self, truthgpt_client: Optional[Any] = None, enabled: bool = True):
        """
        Initialize prompt enhancer.
        
        Args:
            truthgpt_client: Optional TruthGPT client instance
            enabled: Whether enhancement is enabled
        """
        self.truthgpt_client = truthgpt_client
        self.enabled = enabled and truthgpt_client is not None
    
    async def enhance(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, bool]:
        """
        Enhance prompt using TruthGPT.
        
        Args:
            prompt: Prompt to enhance
            context: Optional context dictionary
            
        Returns:
            Tuple of (enhanced_prompt, was_enhanced)
        """
        if not self.enabled:
            return prompt, False
        
        try:
            # Check if TruthGPT is ready
            if not hasattr(self.truthgpt_client, 'is_ready') or not self.truthgpt_client.is_ready():
                logger.debug("TruthGPT not ready, skipping enhancement")
                return prompt, False
            
            # Build enhancement request
            enhancement_data = {
                "query": prompt,
                "context": context or {}
            }
            
            # Call TruthGPT
            result = await self.truthgpt_client.enhance_query(enhancement_data)
            
            if result and isinstance(result, dict):
                enhanced = result.get("enhanced_query") or result.get("query")
                if enhanced and enhanced != prompt:
                    logger.info(f"Prompt enhanced: {prompt[:50]}... -> {enhanced[:50]}...")
                    return enhanced, True
            
            logger.debug("TruthGPT did not enhance prompt, using original")
            return prompt, False
            
        except Exception as e:
            logger.warning(f"Failed to enhance prompt with TruthGPT: {e}")
            return prompt, False

