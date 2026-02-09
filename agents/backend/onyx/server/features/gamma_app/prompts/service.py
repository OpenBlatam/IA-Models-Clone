"""
Prompt Service Implementation
"""

from typing import Dict, Any, Optional
import logging

from .base import PromptBase, Prompt, PromptTemplate, PromptVersion

logger = logging.getLogger(__name__)


class PromptService(PromptBase):
    """Prompt service implementation"""
    
    def __init__(self, db=None, llm_service=None):
        """Initialize prompt service"""
        self.db = db
        self.llm_service = llm_service
        self._prompts: dict = {}
        self._templates: dict = {}
    
    async def get_prompt(
        self,
        name: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get prompt by name"""
        try:
            prompt = self._prompts.get(name)
            if not prompt:
                raise ValueError(f"Prompt '{name}' not found")
            
            content = prompt.content
            
            # Render variables if provided
            if variables:
                for key, value in variables.items():
                    content = content.replace(f"{{{key}}}", str(value))
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting prompt: {e}")
            raise
    
    async def create_prompt(self, prompt: Prompt) -> bool:
        """Create prompt"""
        try:
            self._prompts[prompt.name] = prompt
            return True
            
        except Exception as e:
            logger.error(f"Error creating prompt: {e}")
            return False
    
    async def render_template(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any]
    ) -> str:
        """Render prompt template"""
        try:
            content = template.template
            
            for key, value in variables.items():
                content = content.replace(f"{{{key}}}", str(value))
            
            return content
            
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            raise

