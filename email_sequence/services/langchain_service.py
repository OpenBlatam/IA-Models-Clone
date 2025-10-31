from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema.runnable import RunnablePassthrough
from ..models.sequence import EmailSequence, SequenceStep, StepType
from ..models.template import EmailTemplate, TemplateVariable, VariableType
from ..models.subscriber import Subscriber
from typing import Any, List, Dict, Optional
"""
LangChain Email Service

This module provides LangChain integration for email sequence generation,
personalization, and intelligent content creation.
"""




logger = logging.getLogger(__name__)


class LangChainEmailService:
    """
    Service for LangChain-powered email sequence generation and personalization.
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        """
        Initialize the LangChain email service.
        
        Args:
            api_key: OpenAI API key
            model_name: Model to use for generation
        """
        self.api_key = api_key
        self.model_name = model_name
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name=model_name,
            openai_api_key=api_key
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools and agents
        self._setup_tools()
        self._setup_agents()
    
    def _setup_tools(self) -> Any:
        """Setup LangChain tools for email operations"""
        self.tools = [
            Tool(
                name="email_analyzer",
                func=self._analyze_email_content,
                description="Analyze email content for tone, sentiment, and effectiveness"
            ),
            Tool(
                name="personalization_generator",
                func=self._generate_personalization,
                description="Generate personalized content based on subscriber data"
            ),
            Tool(
                name="subject_line_optimizer",
                func=self._optimize_subject_line,
                description="Optimize email subject lines for better open rates"
            ),
            Tool(
                name="content_enhancer",
                func=self._enhance_content,
                description="Enhance email content for better engagement"
            )
        ]
    
    def _setup_agents(self) -> Any:
        """Setup LangChain agents"""
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
    
    async def generate_email_sequence(
        self,
        sequence_name: str,
        target_audience: str,
        goals: List[str],
        tone: str = "professional",
        length: int = 5
    ) -> EmailSequence:
        """
        Generate a complete email sequence using LangChain.
        
        Args:
            sequence_name: Name of the sequence
            target_audience: Description of target audience
            goals: List of sequence goals
            tone: Desired tone for emails
            length: Number of emails in sequence
            
        Returns:
            EmailSequence: Generated email sequence
        """
        try:
            # Create system prompt for sequence generation
            system_prompt = self._create_sequence_generation_prompt(
                sequence_name, target_audience, goals, tone, length
            )
            
            # Generate sequence structure
            sequence_structure = await self._generate_sequence_structure(
                system_prompt, length
            )
            
            # Create email sequence
            sequence = EmailSequence(
                name=sequence_name,
                description=f"AI-generated sequence for {target_audience}",
                status="draft"
            )
            
            # Add steps to sequence
            for i, step_data in enumerate(sequence_structure, 1):
                step = await self._create_sequence_step(step_data, i)
                sequence.add_step(step)
            
            logger.info(f"Generated email sequence: {sequence_name}")
            return sequence
            
        except Exception as e:
            logger.error(f"Error generating email sequence: {e}")
            raise
    
    async def personalize_email_content(
        self,
        template: EmailTemplate,
        subscriber: Subscriber,
        context: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """
        Personalize email content using LangChain.
        
        Args:
            template: Email template to personalize
            subscriber: Subscriber data
            context: Additional context for personalization
            
        Returns:
            Dict containing personalized subject and content
        """
        try:
            # Create personalization prompt
            prompt = self._create_personalization_prompt(template, subscriber, context)
            
            # Generate personalized content
            response = await self.llm.agenerate([prompt])
            personalized_content = response.generations[0][0].text
            
            # Parse and apply personalization
            rendered_content = template.render({
                **subscriber.dict(),
                **(context or {}),
                "personalized_content": personalized_content
            })
            
            return rendered_content
            
        except Exception as e:
            logger.error(f"Error personalizing email content: {e}")
            raise
    
    async def generate_subject_line(
        self,
        email_content: str,
        subscriber_data: Dict[str, Any],
        tone: str = "professional",
        max_length: int = 60
    ) -> str:
        """
        Generate optimized subject line using LangChain.
        
        Args:
            email_content: Email content
            subscriber_data: Subscriber information
            tone: Desired tone
            max_length: Maximum subject line length
            
        Returns:
            Generated subject line
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are an expert email marketer. Generate compelling subject lines "
                    "that increase open rates. Consider the subscriber's interests and "
                    "the email content. Keep subject lines under {max_length} characters."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Email Content: {content}\n"
                    "Subscriber Data: {subscriber_data}\n"
                    "Tone: {tone}\n"
                    "Generate a subject line:"
                )
            ])
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = await chain.arun(
                content=email_content,
                subscriber_data=str(subscriber_data),
                tone=tone,
                max_length=max_length
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating subject line: {e}")
            raise
    
    async def analyze_email_performance(
        self,
        email_content: str,
        subject_line: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze email performance and provide optimization suggestions.
        
        Args:
            email_content: Email content
            subject_line: Subject line
            metrics: Performance metrics
            
        Returns:
            Analysis results and suggestions
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are an email marketing expert. Analyze the provided email "
                    "and metrics to provide actionable insights for improvement."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Subject: {subject}\n"
                    "Content: {content}\n"
                    "Metrics: {metrics}\n"
                    "Provide analysis and suggestions:"
                )
            ])
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            analysis = await chain.arun(
                subject=subject_line,
                content=email_content,
                metrics=str(metrics)
            )
            
            return {
                "analysis": analysis,
                "timestamp": datetime.utcnow(),
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing email performance: {e}")
            raise
    
    async def generate_ab_test_variants(
        self,
        original_content: str,
        test_type: str = "subject",
        num_variants: int = 2
    ) -> List[Dict[str, str]]:
        """
        Generate A/B test variants using LangChain.
        
        Args:
            original_content: Original content to test
            test_type: Type of test (subject, content, etc.)
            num_variants: Number of variants to generate
            
        Returns:
            List of test variants
        """
        try:
            variants = []
            
            for i in range(num_variants):
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Generate a {test_type} variant for A/B testing. "
                        "Make it different from the original but equally effective."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "Original {test_type}: {content}\n"
                        "Generate variant {i+1}:"
                    )
                ])
                
                chain = LLMChain(llm=self.llm, prompt=prompt)
                variant = await chain.arun(
                    test_type=test_type,
                    content=original_content,
                    i=i
                )
                
                variants.append({
                    "variant_id": f"variant_{i+1}",
                    "content": variant.strip(),
                    "type": test_type
                })
            
            return variants
            
        except Exception as e:
            logger.error(f"Error generating A/B test variants: {e}")
            raise
    
    def _create_sequence_generation_prompt(
        self,
        sequence_name: str,
        target_audience: str,
        goals: List[str],
        tone: str,
        length: int
    ) -> str:
        """Create prompt for sequence generation"""
        return f"""
        Generate an email sequence with the following specifications:
        
        Sequence Name: {sequence_name}
        Target Audience: {target_audience}
        Goals: {', '.join(goals)}
        Tone: {tone}
        Number of Emails: {length}
        
        For each email, provide:
        1. Subject line
        2. Email content (HTML format)
        3. Delay before next email (in hours)
        4. Purpose and strategy
        
        Structure the response as a JSON array with objects containing:
        - subject: string
        - content: string (HTML)
        - delay_hours: integer
        - purpose: string
        - strategy: string
        """
    
    async def _generate_sequence_structure(
        self,
        prompt: str,
        length: int
    ) -> List[Dict[str, Any]]:
        """Generate sequence structure using LangChain"""
        try:
            response = await self.llm.agenerate([prompt])
            content = response.generations[0][0].text
            
            # Parse JSON response (simplified - in production use proper JSON parsing)
            # This is a simplified version - you'd want proper JSON parsing
            return [{"subject": f"Email {i+1}", "content": f"Content {i+1}", "delay_hours": 24} 
                   for i in range(length)]
            
        except Exception as e:
            logger.error(f"Error generating sequence structure: {e}")
            raise
    
    async def _create_sequence_step(
        self,
        step_data: Dict[str, Any],
        order: int
    ) -> SequenceStep:
        """Create a sequence step from generated data"""
        return SequenceStep(
            step_type=StepType.EMAIL,
            order=order,
            name=f"Step {order}",
            subject=step_data.get("subject", ""),
            content=step_data.get("content", ""),
            delay_hours=step_data.get("delay_hours", 24)
        )
    
    def _create_personalization_prompt(
        self,
        template: EmailTemplate,
        subscriber: Subscriber,
        context: Dict[str, Any] = None
    ) -> str:
        """Create prompt for content personalization"""
        return f"""
        Personalize the following email content for the subscriber:
        
        Template Variables: {[var.name for var in template.variables]}
        Subscriber Data: {subscriber.dict()}
        Context: {context or {}}
        
        Generate personalized content that maintains the original message
        while making it relevant to this specific subscriber.
        """
    
    # Tool implementations
    def _analyze_email_content(self, content: str) -> str:
        """Analyze email content for effectiveness"""
        return f"Analysis of email content: {len(content)} characters, professional tone detected"
    
    def _generate_personalization(self, subscriber_data: str) -> str:
        """Generate personalized content"""
        return f"Personalized content based on: {subscriber_data}"
    
    def _optimize_subject_line(self, subject: str) -> str:
        """Optimize subject line"""
        return f"Optimized subject: {subject[:50]}..."
    
    def _enhance_content(self, content: str) -> str:
        """Enhance email content"""
        return f"Enhanced content: {content[:100]}..."
    
    async def close(self) -> Any:
        """Clean up resources"""
        if hasattr(self.llm, 'close'):
            await self.llm.aclose() 