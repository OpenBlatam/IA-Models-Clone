"""
AI Document Generation Service
==============================

Service for generating document content using AI models.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from .types import SectionData, DocumentTone, DocumentLength, LanguageCode

from .models import DocumentTemplate, DocumentType, DocumentSection
from .templates import template_manager
from .constants import (
    LENGTH_GUIDELINES, 
    TONE_GUIDELINES,
    DEFAULT_AI_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    AI_SIMULATION_DELAY_SECONDS
)
from .mock_content import get_mock_content_for_type, detect_document_type_from_prompt
from .exceptions import AIServiceError, DocumentGenerationError

logger = logging.getLogger(__name__)


class AIDocumentGenerator:
    """AI-powered document content generator."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_AI_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    async def generate_document_content(
        self,
        query: str,
        template: DocumentTemplate,
        document_type: DocumentType,
        tone: DocumentTone = "professional",
        length: DocumentLength = "medium",
        language: LanguageCode = "en",
        additional_requirements: Optional[str] = None
    ) -> List[SectionData]:
        """Generate document content using AI."""
        
        try:
            # Create the prompt for AI generation
            prompt = self._create_generation_prompt(
                query=query,
                template=template,
                document_type=document_type,
                tone=tone,
                length=length,
                language=language,
                additional_requirements=additional_requirements
            )
            
            # Generate content using AI (simulated for now)
            # In production, this would call OpenAI, Anthropic, or other AI service
            content = await self._call_ai_service(prompt)
            
            # Parse and structure the generated content
            sections = self._parse_ai_response(content, template)
            
            return sections
            
        except AIServiceError:
            raise
        except Exception as e:
            logger.error(f"Error generating document content: {str(e)}")
            raise DocumentGenerationError(f"Failed to generate document content: {str(e)}") from e
    
    def _create_generation_prompt(
        self,
        query: str,
        template: DocumentTemplate,
        document_type: DocumentType,
        tone: DocumentTone,
        length: DocumentLength,
        language: LanguageCode,
        additional_requirements: Optional[str]
    ) -> str:
        """Create a comprehensive prompt for AI generation."""
        
        prompt = f"""
You are a professional document writer specializing in creating high-quality {document_type.value} documents.

TASK: Generate a comprehensive {document_type.value} based on the following request:

USER REQUEST: {query}

DOCUMENT REQUIREMENTS:
- Document Type: {document_type.value}
- Template: {template.name}
- Tone: {tone} ({TONE_GUIDELINES.get(tone, '')})
- Length: {length} ({LENGTH_GUIDELINES.get(length, '')})
- Language: {language}

REQUIRED SECTIONS (in order):
{chr(10).join(f"- {section}" for section in template.sections)}

ADDITIONAL REQUIREMENTS:
{additional_requirements or "None specified"}

INSTRUCTIONS:
1. Generate content for each required section
2. Ensure content is relevant to the user's request
3. Maintain consistency in tone and style throughout
4. Include specific, actionable information where appropriate
5. Use proper formatting and structure
6. Make the content professional and engaging

OUTPUT FORMAT:
Return a JSON array where each element represents a section with the following structure:
{{
    "title": "Section Title",
    "content": "Detailed content for this section...",
    "level": 1,
    "metadata": {{
        "word_count": 150,
        "key_points": ["point1", "point2", "point3"]
    }}
}}

Generate comprehensive, professional content that addresses the user's request effectively.
"""
        
        return prompt
    
    async def _call_ai_service(self, prompt: str) -> str:
        """Call the AI service to generate content."""
        # This is a simulation - in production, you would integrate with:
        # - OpenAI GPT-4
        # - Anthropic Claude
        # - Google Gemini
        # - Azure OpenAI
        # - Local models like Llama, Mistral, etc.
        
        # Simulate AI response with realistic content
        await asyncio.sleep(AI_SIMULATION_DELAY_SECONDS)
        
        # For demonstration, return structured content
        # In production, this would be the actual AI response
        return self._generate_mock_ai_response(prompt)
    
    def _generate_mock_ai_response(self, prompt: str) -> str:
        """Generate mock AI response for demonstration."""
        doc_type = detect_document_type_from_prompt(prompt)
        content = get_mock_content_for_type(doc_type)
        return json.dumps(content)
    
    def _parse_ai_response(self, response: str, template: DocumentTemplate) -> List[SectionData]:
        """Parse AI response into structured sections."""
        try:
            # Parse JSON response
            sections_data = json.loads(response)
            
            # Validate and structure sections using list comprehension
            sections = [
                {
                    "title": section_data.get("title", f"Section {i+1}"),
                    "content": section_data.get("content", ""),
                    "level": section_data.get("level", 1),
                    "metadata": section_data.get("metadata", {})
                }
                for i, section_data in enumerate(sections_data)
            ]
            
            return sections
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {str(e)}")
            raise AIServiceError(f"Invalid AI response format: {str(e)}") from e
    
    def _generate_fallback_content(
        self, 
        query: str, 
        template: DocumentTemplate, 
        document_type: DocumentType
    ) -> List[SectionData]:
        """Generate fallback content when AI generation fails."""
        query_preview = query[:100] if query else ""
        
        return [
            {
                "title": section_name,
                "content": (
                    f"This section will contain detailed information about {section_name.lower()}. "
                    f"The content will be tailored to address the specific requirements: {query_preview}..."
                ),
                "level": 1,
                "metadata": {
                    "word_count": len(
                        f"This section will contain detailed information about {section_name.lower()}. "
                        f"The content will be tailored to address the specific requirements: {query_preview}..."
                        .split()
                    ),
                    "key_points": [f"Key point {j+1}" for j in range(3)],
                    "fallback": True
                }
            }
            for section_name in template.sections
        ]
    
    def set_model_config(self, model_name: str, max_tokens: int, temperature: float):
        """Configure AI model parameters."""
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info(f"AI model configured: {model_name}, max_tokens: {max_tokens}, temperature: {temperature}")























