"""
Content Generator - Shared content generation utilities
========================================================

Provides common utilities for generating document content using LangChain.
"""

import logging
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class ContentGenerator:
    """Shared utilities for generating document content."""
    
    @staticmethod
    async def generate_content(
        prompt_template: ChatPromptTemplate,
        llm: ChatOpenAI,
        output_parser: StrOutputParser,
        task_data: Dict[str, Any],
        context: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate document content using LangChain chain.
        
        Args:
            prompt_template: LangChain prompt template
            llm: ChatOpenAI instance
            output_parser: Output parser instance
            task_data: Dictionary with task data (business_area, document_type, query, etc.)
            context: Optional additional context string
            
        Returns:
            Generated content or None if generation fails
        """
        try:
            chain = prompt_template | llm | output_parser
            
            invoke_data = {
                "business_area": task_data.get("business_area", ""),
                "document_type": task_data.get("document_type", ""),
                "query": task_data.get("query", ""),
                "context": context or task_data.get("context", "")
            }
            
            if "target_audience" in task_data:
                invoke_data["target_audience"] = task_data["target_audience"]
            if "language" in task_data:
                invoke_data["language"] = task_data["language"]
            if "tone" in task_data:
                invoke_data["tone"] = task_data["tone"]
            if "variation_number" in task_data:
                invoke_data["variation_number"] = task_data["variation_number"]
            if "previous_content" in task_data:
                invoke_data["previous_content"] = task_data["previous_content"]
            
            content = await chain.ainvoke(invoke_data)
            return content
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}", exc_info=True)
            return None
    
    @staticmethod
    def build_task_context(
        task_id: str,
        priority: int,
        created_at: Optional[str] = None,
        quality_target: Optional[float] = None
    ) -> str:
        """
        Build context string for task.
        
        Args:
            task_id: Task identifier
            priority: Task priority level
            created_at: Optional creation timestamp
            quality_target: Optional quality threshold
            
        Returns:
            Formatted context string
        """
        context_parts = [
            f"Task ID: {task_id}",
            f"Priority: {priority}"
        ]
        
        if created_at:
            context_parts.append(f"Created: {created_at}")
        
        if quality_target:
            context_parts.append(f"Quality Target: {quality_target}")
        
        return ", ".join(context_parts)

