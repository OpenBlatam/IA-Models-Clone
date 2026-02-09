"""
Prompts Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4


class PromptTemplate:
    """Prompt template"""
    
    def __init__(
        self,
        name: str,
        template: str,
        variables: Optional[List[str]] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.template = template
        self.variables = variables or []
        self.created_at = datetime.utcnow()


class PromptVersion:
    """Prompt version"""
    
    def __init__(
        self,
        prompt_id: str,
        version: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.prompt_id = prompt_id
        self.version = version
        self.content = content
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()


class Prompt:
    """Prompt definition"""
    
    def __init__(
        self,
        name: str,
        content: str,
        prompt_type: str = "standard"
    ):
        self.id = str(uuid4())
        self.name = name
        self.content = content
        self.prompt_type = prompt_type
        self.versions: List[PromptVersion] = []
        self.created_at = datetime.utcnow()


class PromptBase(ABC):
    """Base interface for prompts"""
    
    @abstractmethod
    async def get_prompt(
        self,
        name: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get prompt by name"""
        pass
    
    @abstractmethod
    async def create_prompt(self, prompt: Prompt) -> bool:
        """Create prompt"""
        pass
    
    @abstractmethod
    async def render_template(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any]
    ) -> str:
        """Render prompt template"""
        pass

