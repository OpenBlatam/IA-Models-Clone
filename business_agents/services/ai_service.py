"""
AI Service for Business Agents
==============================

Advanced AI integration service for enhanced document generation and workflow execution.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
import httpx
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"

@dataclass
class AIRequest:
    prompt: str
    context: Dict[str, Any]
    provider: AIProvider
    model: str
    max_tokens: int = 2000
    temperature: float = 0.7
    system_message: Optional[str] = None

@dataclass
class AIResponse:
    content: str
    provider: AIProvider
    model: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]

class AIService:
    """
    Advanced AI service for business agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        
        # Initialize AI clients
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize AI service clients."""
        
        # OpenAI
        if self.config.get("openai_api_key"):
            openai.api_key = self.config["openai_api_key"]
            self.openai_client = openai
            
        # Anthropic
        if self.config.get("anthropic_api_key"):
            self.anthropic_client = Anthropic(
                api_key=self.config["anthropic_api_key"]
            )
            
        # Google (placeholder)
        if self.config.get("google_api_key"):
            # Initialize Google AI client
            pass
            
    async def generate_content(
        self,
        request: AIRequest,
        business_area: str,
        document_type: str
    ) -> AIResponse:
        """Generate content using AI."""
        
        try:
            # Enhance prompt with business context
            enhanced_prompt = self._enhance_prompt(
                request.prompt,
                business_area,
                document_type,
                request.context
            )
            
            # Select best provider
            provider = self._select_provider(request.provider, document_type)
            
            # Generate content
            if provider == AIProvider.OPENAI:
                response = await self._generate_openai(enhanced_prompt, request)
            elif provider == AIProvider.ANTHROPIC:
                response = await self._generate_anthropic(enhanced_prompt, request)
            elif provider == AIProvider.GOOGLE:
                response = await self._generate_google(enhanced_prompt, request)
            else:
                response = await self._generate_local(enhanced_prompt, request)
                
            return response
            
        except Exception as e:
            logger.error(f"AI content generation failed: {str(e)}")
            raise
            
    def _enhance_prompt(
        self,
        prompt: str,
        business_area: str,
        document_type: str,
        context: Dict[str, Any]
    ) -> str:
        """Enhance prompt with business context."""
        
        # Business area specific enhancements
        area_context = self._get_business_area_context(business_area)
        
        # Document type specific enhancements
        doc_context = self._get_document_type_context(document_type)
        
        # Build enhanced prompt
        enhanced_prompt = f"""
Business Context:
- Area: {business_area.title()}
- Document Type: {document_type.replace('_', ' ').title()}
- Area Guidelines: {area_context}
- Document Guidelines: {doc_context}

User Context:
{json.dumps(context, indent=2)}

Original Request:
{prompt}

Please generate high-quality, professional content that follows best practices for {business_area} {document_type.replace('_', ' ')}. 
Ensure the content is actionable, well-structured, and tailored to the provided context.
"""
        
        return enhanced_prompt
        
    def _get_business_area_context(self, business_area: str) -> str:
        """Get context guidelines for business area."""
        
        contexts = {
            "marketing": """
            - Focus on customer value and market positioning
            - Include data-driven insights and metrics
            - Emphasize brand consistency and messaging
            - Consider multi-channel strategies
            """,
            "sales": """
            - Focus on customer pain points and solutions
            - Include clear value propositions
            - Emphasize relationship building
            - Consider sales funnel optimization
            """,
            "operations": """
            - Focus on efficiency and process optimization
            - Include measurable KPIs and metrics
            - Emphasize scalability and automation
            - Consider risk management and compliance
            """,
            "hr": """
            - Focus on employee experience and development
            - Include diversity and inclusion considerations
            - Emphasize legal compliance and best practices
            - Consider organizational culture and values
            """,
            "finance": """
            - Focus on financial accuracy and compliance
            - Include risk assessment and mitigation
            - Emphasize ROI and cost optimization
            - Consider regulatory requirements
            """,
            "legal": """
            - Focus on compliance and risk mitigation
            - Include clear legal language and terms
            - Emphasize protection of interests
            - Consider jurisdiction-specific requirements
            """,
            "technical": """
            - Focus on technical accuracy and clarity
            - Include implementation details and examples
            - Emphasize maintainability and scalability
            - Consider security and performance
            """,
            "content": """
            - Focus on audience engagement and value
            - Include SEO optimization and readability
            - Emphasize brand voice and consistency
            - Consider multi-format adaptability
            """
        }
        
        return contexts.get(business_area, "Focus on professional quality and business value.")
        
    def _get_document_type_context(self, document_type: str) -> str:
        """Get context guidelines for document type."""
        
        contexts = {
            "business_plan": """
            - Include executive summary with key highlights
            - Provide detailed market analysis and competitive landscape
            - Include financial projections and funding requirements
            - Emphasize scalability and growth potential
            """,
            "marketing_strategy": """
            - Include target audience analysis and personas
            - Provide channel-specific strategies and tactics
            - Include budget allocation and ROI projections
            - Emphasize measurement and optimization
            """,
            "sales_proposal": """
            - Include problem statement and solution overview
            - Provide clear pricing and value proposition
            - Include implementation timeline and next steps
            - Emphasize benefits and ROI for the client
            """,
            "financial_report": """
            - Include executive summary with key findings
            - Provide detailed financial analysis and trends
            - Include recommendations and action items
            - Emphasize accuracy and compliance
            """,
            "operational_manual": """
            - Include clear step-by-step procedures
            - Provide troubleshooting and FAQ sections
            - Include safety and compliance guidelines
            - Emphasize usability and clarity
            """,
            "hr_policy": """
            - Include clear policy statements and procedures
            - Provide examples and scenarios
            - Include compliance and legal considerations
            - Emphasize fairness and consistency
            """,
            "technical_specification": """
            - Include detailed technical requirements
            - Provide implementation guidelines and examples
            - Include testing and validation procedures
            - Emphasize accuracy and completeness
            """,
            "project_proposal": """
            - Include project overview and objectives
            - Provide detailed timeline and resource requirements
            - Include risk assessment and mitigation strategies
            - Emphasize value and feasibility
            """,
            "contract": """
            - Include clear terms and conditions
            - Provide detailed scope and deliverables
            - Include payment terms and timelines
            - Emphasize legal protection and clarity
            """,
            "presentation": """
            - Include compelling opening and closing
            - Provide clear structure and flow
            - Include visual elements and data visualization
            - Emphasize engagement and persuasion
            """,
            "email_template": """
            - Include compelling subject line and opening
            - Provide clear call-to-action
            - Include personalization elements
            - Emphasize deliverability and engagement
            """,
            "social_media_post": """
            - Include engaging and shareable content
            - Provide relevant hashtags and mentions
            - Include visual elements and formatting
            - Emphasize platform-specific optimization
            """,
            "blog_post": """
            - Include SEO-optimized title and structure
            - Provide valuable and actionable content
            - Include internal and external links
            - Emphasize readability and engagement
            """,
            "press_release": """
            - Include newsworthy headline and lead
            - Provide clear and concise information
            - Include quotes and contact information
            - Emphasize newsworthiness and clarity
            """,
            "user_manual": """
            - Include clear and simple instructions
            - Provide troubleshooting and FAQ sections
            - Include visual aids and examples
            - Emphasize usability and accessibility
            """,
            "training_material": """
            - Include learning objectives and outcomes
            - Provide interactive and engaging content
            - Include assessments and feedback mechanisms
            - Emphasize practical application and retention
            """
        }
        
        return contexts.get(document_type, "Focus on clarity, completeness, and professional quality.")
        
    def _select_provider(
        self,
        requested_provider: AIProvider,
        document_type: str
    ) -> AIProvider:
        """Select the best AI provider for the task."""
        
        # Provider capabilities by document type
        provider_capabilities = {
            "business_plan": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "marketing_strategy": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "sales_proposal": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "financial_report": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "operational_manual": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "hr_policy": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "technical_specification": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "project_proposal": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "contract": [AIProvider.ANTHROPIC, AIProvider.OPENAI],
            "presentation": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "email_template": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "social_media_post": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "blog_post": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "press_release": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "user_manual": [AIProvider.OPENAI, AIProvider.ANTHROPIC],
            "training_material": [AIProvider.OPENAI, AIProvider.ANTHROPIC]
        }
        
        # Get preferred providers for document type
        preferred_providers = provider_capabilities.get(document_type, [AIProvider.OPENAI])
        
        # Check if requested provider is available and preferred
        if requested_provider in preferred_providers and self._is_provider_available(requested_provider):
            return requested_provider
            
        # Fallback to first available preferred provider
        for provider in preferred_providers:
            if self._is_provider_available(provider):
                return provider
                
        # Final fallback
        return AIProvider.OPENAI if self._is_provider_available(AIProvider.OPENAI) else AIProvider.LOCAL
        
    def _is_provider_available(self, provider: AIProvider) -> bool:
        """Check if AI provider is available."""
        
        if provider == AIProvider.OPENAI:
            return self.openai_client is not None
        elif provider == AIProvider.ANTHROPIC:
            return self.anthropic_client is not None
        elif provider == AIProvider.GOOGLE:
            return self.google_client is not None
        else:
            return True  # Local provider is always available
            
    async def _generate_openai(self, prompt: str, request: AIRequest) -> AIResponse:
        """Generate content using OpenAI."""
        
        try:
            response = await self.openai_client.ChatCompletion.acreate(
                model=request.model,
                messages=[
                    {"role": "system", "content": request.system_message or "You are a professional business consultant and content creator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            return AIResponse(
                content=response.choices[0].message.content,
                provider=AIProvider.OPENAI,
                model=request.model,
                usage=response.usage,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise
            
    async def _generate_anthropic(self, prompt: str, request: AIRequest) -> AIResponse:
        """Generate content using Anthropic."""
        
        try:
            response = await self.anthropic_client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system_message or "You are a professional business consultant and content creator.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return AIResponse(
                content=response.content[0].text,
                provider=AIProvider.ANTHROPIC,
                model=request.model,
                usage=response.usage,
                metadata={"stop_reason": response.stop_reason}
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {str(e)}")
            raise
            
    async def _generate_google(self, prompt: str, request: AIRequest) -> AIResponse:
        """Generate content using Google AI."""
        
        # Placeholder for Google AI integration
        # This would integrate with Google's AI services
        
        return AIResponse(
            content="Google AI integration not yet implemented.",
            provider=AIProvider.GOOGLE,
            model=request.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            metadata={}
        )
        
    async def _generate_local(self, prompt: str, request: AIRequest) -> AIResponse:
        """Generate content using local AI model."""
        
        # Placeholder for local AI integration
        # This would integrate with local models like Ollama, etc.
        
        return AIResponse(
            content="Local AI integration not yet implemented.",
            provider=AIProvider.LOCAL,
            model=request.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            metadata={}
        )
        
    async def analyze_content_quality(
        self,
        content: str,
        document_type: str,
        business_area: str
    ) -> Dict[str, Any]:
        """Analyze content quality and provide feedback."""
        
        try:
            analysis_prompt = f"""
            Analyze the following {business_area} {document_type.replace('_', ' ')} content for quality, completeness, and effectiveness.
            
            Content:
            {content}
            
            Please provide:
            1. Overall quality score (1-10)
            2. Strengths and weaknesses
            3. Suggestions for improvement
            4. Missing elements
            5. Compliance with best practices
            
            Format your response as JSON.
            """
            
            request = AIRequest(
                prompt=analysis_prompt,
                context={"document_type": document_type, "business_area": business_area},
                provider=AIProvider.OPENAI,
                model="gpt-4",
                max_tokens=1000,
                temperature=0.3
            )
            
            response = await self.generate_content(request, business_area, document_type)
            
            # Parse JSON response
            try:
                analysis = json.loads(response.content)
                return analysis
            except json.JSONDecodeError:
                return {
                    "quality_score": 7,
                    "strengths": ["Content generated successfully"],
                    "weaknesses": ["Unable to parse detailed analysis"],
                    "suggestions": ["Review content manually"],
                    "missing_elements": [],
                    "compliance": "Partial"
                }
                
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return {
                "quality_score": 5,
                "strengths": [],
                "weaknesses": ["Analysis failed"],
                "suggestions": ["Manual review recommended"],
                "missing_elements": [],
                "compliance": "Unknown"
            }
            
    async def generate_variations(
        self,
        content: str,
        document_type: str,
        business_area: str,
        count: int = 3
    ) -> List[str]:
        """Generate variations of content."""
        
        try:
            variations = []
            
            for i in range(count):
                variation_prompt = f"""
                Create a variation of the following {business_area} {document_type.replace('_', ' ')} content.
                Maintain the same key information but vary the style, tone, or approach.
                
                Original Content:
                {content}
                
                Variation {i+1}: Create a different version that maintains quality and completeness.
                """
                
                request = AIRequest(
                    prompt=variation_prompt,
                    context={"document_type": document_type, "business_area": business_area},
                    provider=AIProvider.OPENAI,
                    model="gpt-4",
                    max_tokens=2000,
                    temperature=0.8
                )
                
                response = await self.generate_content(request, business_area, document_type)
                variations.append(response.content)
                
            return variations
            
        except Exception as e:
            logger.error(f"Content variation generation failed: {str(e)}")
            return [content]  # Return original content as fallback
            
    async def optimize_for_seo(
        self,
        content: str,
        keywords: List[str],
        document_type: str
    ) -> str:
        """Optimize content for SEO."""
        
        try:
            seo_prompt = f"""
            Optimize the following {document_type.replace('_', ' ')} content for SEO.
            
            Target Keywords: {', '.join(keywords)}
            
            Content:
            {content}
            
            Please:
            1. Integrate keywords naturally
            2. Optimize headings and structure
            3. Improve readability
            4. Add relevant internal linking suggestions
            5. Maintain content quality and value
            
            Return the optimized content.
            """
            
            request = AIRequest(
                prompt=seo_prompt,
                context={"keywords": keywords, "document_type": document_type},
                provider=AIProvider.OPENAI,
                model="gpt-4",
                max_tokens=2000,
                temperature=0.5
            )
            
            response = await self.generate_content(request, "content", document_type)
            return response.content
            
        except Exception as e:
            logger.error(f"SEO optimization failed: {str(e)}")
            return content  # Return original content as fallback





























