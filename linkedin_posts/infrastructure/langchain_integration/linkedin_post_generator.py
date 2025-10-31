from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
from typing import Dict, List, Optional, Any
from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel, Field
from ...core.entities.linkedin_post import LinkedInPost, PostType, ContentTone
from ...shared.logging import get_logger
from typing import Any, List, Dict, Optional
import logging
"""
LinkedIn Post Generator with LangChain
=====================================

AI-powered LinkedIn post generation using LangChain framework.
"""





logger = get_logger(__name__)


class GeneratedPost(BaseModel):
    """Generated post content structure."""
    
    title: str = Field(description="Post title")
    content: str = Field(description="Main post content")
    summary: Optional[str] = Field(description="Post summary")
    hashtags: List[str] = Field(description="Relevant hashtags")
    keywords: List[str] = Field(description="Target keywords")
    call_to_action: Optional[str] = Field(description="Call to action")
    estimated_engagement: float = Field(description="Estimated engagement score")


class LinkedInPostGenerator:
    """
    LinkedIn Post Generator using LangChain.
    
    Generates high-quality LinkedIn posts using AI with various
    customization options and optimization features.
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        """Initialize the post generator."""
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name=model_name,
            openai_api_key=api_key
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self._setup_prompts()
        self._setup_tools()
    
    def _setup_prompts(self) -> Any:
        """Setup prompt templates."""
        
        # System prompt for LinkedIn post generation
        self.system_prompt = SystemMessagePromptTemplate.from_template(
            """You are an expert LinkedIn content creator and social media strategist. 
            Your task is to create engaging, professional LinkedIn posts that drive engagement and reach.
            
            Key guidelines:
            - Write in a {tone} tone
            - Include relevant hashtags (3-5 recommended)
            - Use storytelling and personal experiences when appropriate
            - Include a clear call-to-action
            - Optimize for LinkedIn's algorithm
            - Keep content authentic and valuable
            - Use bullet points and formatting for readability
            - Target audience: {target_audience}
            - Industry focus: {industry}
            
            Post type: {post_type}
            Target keywords: {keywords}
            """
        )
        
        # Human prompt template
        self.human_prompt = HumanMessagePromptTemplate.from_template(
            """Create a LinkedIn post based on the following information:
            
            Topic: {topic}
            Key points to cover: {key_points}
            Target audience: {target_audience}
            Industry: {industry}
            Tone: {tone}
            Post type: {post_type}
            Keywords to include: {keywords}
            
            Additional context: {additional_context}
            
            Please generate a compelling LinkedIn post that follows best practices and drives engagement."""
        )
        
        # Create the full prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            self.system_prompt,
            self.human_prompt
        ])
        
        # Output parser
        self.output_parser = PydanticOutputParser(pydantic_object=GeneratedPost)
    
    def _setup_tools(self) -> Any:
        """Setup LangChain tools for enhanced generation."""
        
        # Hashtag research tool
        self.hashtag_tool = Tool(
            name="hashtag_research",
            func=self._research_hashtags,
            description="Research trending and relevant hashtags for a given topic"
        )
        
        # Content optimization tool
        self.optimization_tool = Tool(
            name="content_optimization",
            func=self._optimize_content,
            description="Optimize content for better engagement and reach"
        )
        
        # Engagement prediction tool
        self.engagement_tool = Tool(
            name="engagement_prediction",
            func=self._predict_engagement,
            description="Predict engagement metrics for a given post"
        )
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=[self.hashtag_tool, self.optimization_tool, self.engagement_tool],
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
    
    async def generate_post(
        self,
        topic: str,
        key_points: List[str],
        target_audience: str,
        industry: str,
        tone: ContentTone = ContentTone.PROFESSIONAL,
        post_type: PostType = PostType.TEXT,
        keywords: Optional[List[str]] = None,
        additional_context: Optional[str] = None,
        user_id: UUID = None
    ) -> Dict[str, Any]:
        """
        Generate a LinkedIn post using LangChain.
        
        Args:
            topic: Main topic of the post
            key_points: Key points to cover
            target_audience: Target audience description
            industry: Industry focus
            tone: Content tone
            post_type: Type of post
            keywords: Target keywords
            additional_context: Additional context
            user_id: User ID for personalization
            
        Returns:
            Dictionary with generated post data
        """
        try:
            logger.info(f"Generating LinkedIn post for topic: {topic}")
            
            # Prepare input data
            input_data = {
                "topic": topic,
                "key_points": "\n".join(f"- {point}" for point in key_points),
                "target_audience": target_audience,
                "industry": industry,
                "tone": tone.value,
                "post_type": post_type.value,
                "keywords": ", ".join(keywords or []),
                "additional_context": additional_context or "",
            }
            
            # Generate post using LangChain
            chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template,
                output_parser=self.output_parser
            )
            
            # Execute the chain
            result = await chain.arun(input_data)
            
            # Parse the result
            generated_post = self.output_parser.parse(result)
            
            # Enhance with additional tools
            enhanced_hashtags = await self._research_hashtags(topic, industry)
            optimized_content = await self._optimize_content(generated_post.content)
            engagement_prediction = await self._predict_engagement(generated_post.content)
            
            # Create response
            response = {
                "title": generated_post.title,
                "content": optimized_content,
                "summary": generated_post.summary,
                "hashtags": enhanced_hashtags,
                "keywords": generated_post.keywords,
                "call_to_action": generated_post.call_to_action,
                "estimated_engagement": engagement_prediction,
                "langchain_data": {
                    "prompt"f": self.prompt_template",
                    "model": self.llm.model_name,
                    "parameters": {
                        "temperature": self.llm.temperature,
                        "max_tokens": getattr(self.llm, 'max_tokens', None),
                    }
                }
            }
            
            logger.info(f"Successfully generated LinkedIn post: {generated_post.title}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating LinkedIn post: {e}")
            raise
    
    async def generate_multiple_variants(
        self,
        topic: str,
        key_points: List[str],
        target_audience: str,
        industry: str,
        num_variants: int = 3,
        tone: ContentTone = ContentTone.PROFESSIONAL,
        post_type: PostType = PostType.TEXT,
        keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple post variants for A/B testing.
        
        Args:
            topic: Main topic
            key_points: Key points to cover
            target_audience: Target audience
            industry: Industry focus
            num_variants: Number of variants to generate
            tone: Content tone
            post_type: Post type
            keywords: Target keywords
            
        Returns:
            List of generated post variants
        """
        variants = []
        
        for i in range(num_variants):
            # Vary the tone slightly for different variants
            variant_tone = self._get_variant_tone(tone, i)
            
            variant = await self.generate_post(
                topic=topic,
                key_points=key_points,
                target_audience=target_audience,
                industry=industry,
                tone=variant_tone,
                post_type=post_type,
                keywords=keywords,
                additional_context=f"Variant {i+1} - Focus on different angle"
            )
            
            variant["variant_id"] = f"variant_{i+1}"
            variant["tone"] = variant_tone.value
            variants.append(variant)
        
        return variants
    
    async def _research_hashtags(self, topic: str, industry: str) -> List[str]:
        """Research relevant hashtags for the topic."""
        try:
            prompt = f"""
            Research trending and relevant hashtags for LinkedIn posts about:
            Topic: {topic}
            Industry: {industry}
            
            Return 5-7 relevant hashtags that are:
            1. Popular in the industry
            2. Relevant to the topic
            3. Not overly saturated
            4. Professional and appropriate
            
            Format as a list of hashtags only.
            """
            
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            hashtags_text = response.generations[0][0].text.strip()
            
            # Parse hashtags
            hashtags = [tag.strip() for tag in hashtags_text.split('\n') if tag.strip().startswith('#')]
            return hashtags[:7]  # Limit to 7 hashtags
            
        except Exception as e:
            logger.error(f"Error researching hashtags: {e}")
            return []
    
    async def _optimize_content(self, content: str) -> str:
        """Optimize content for better engagement."""
        try:
            prompt = f"""
            Optimize this LinkedIn post content for maximum engagement:
            
            {content}
            
            Optimization guidelines:
            1. Improve readability with better formatting
            2. Add engaging hooks and storytelling elements
            3. Include clear call-to-action
            4. Optimize for LinkedIn's algorithm
            5. Make it more shareable and comment-worthy
            6. Keep it authentic and valuable
            
            Return the optimized content only.
            """
            
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            optimized_content = response.generations[0][0].text.strip()
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            return content
    
    async def _predict_engagement(self, content: str) -> float:
        """Predict engagement score for the content."""
        try:
            prompt = f"""
            Analyze this LinkedIn post content and predict its engagement potential:
            
            {content}
            
            Consider factors like:
            - Content quality and relevance
            - Emotional appeal
            - Call-to-action effectiveness
            - Readability and formatting
            - Trending topic relevance
            
            Return a score from 0-100 where:
            0-20: Low engagement potential
            21-40: Below average
            41-60: Average
            61-80: Above average
            81-100: High engagement potential
            
            Return only the numerical score.
            """
            
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            score_text = response.generations[0][0].text.strip()
            
            # Extract numerical score
            try:
                score = float(score_text.split()[0])
                return max(0, min(100, score))  # Clamp between 0-100
            except (ValueError, IndexError):
                return 50.0  # Default score
                
        except Exception as e:
            logger.error(f"Error predicting engagement: {e}")
            return 50.0
    
    def _get_variant_tone(self, base_tone: ContentTone, variant_index: int) -> ContentTone:
        """Get variant tone for A/B testing."""
        tones = list(ContentTone)
        base_index = tones.index(base_tone)
        
        # Cycle through different tones
        variant_index = (base_index + variant_index) % len(tones)
        return tones[variant_index]
    
    async def generate_industry_specific_post(
        self,
        topic: str,
        industry: str,
        company_size: str = "startup",
        target_role: str = "professionals"
    ) -> Dict[str, Any]:
        """
        Generate industry-specific LinkedIn post.
        
        Args:
            topic: Main topic
            industry: Industry focus
            company_size: Company size (startup, mid-size, enterprise)
            target_role: Target role (professionals, executives, etc.)
            
        Returns:
            Generated post data
        """
        
        # Industry-specific prompts
        industry_prompts = {
            "technology": "Focus on innovation, digital transformation, and tech trends",
            "finance": "Emphasize financial insights, market analysis, and investment strategies",
            "healthcare": "Highlight patient care, medical advancements, and healthcare innovation",
            "education": "Focus on learning, skill development, and educational technology",
            "marketing": "Emphasize brand building, customer engagement, and marketing strategies",
            "consulting": "Highlight problem-solving, business strategy, and client success"
        }
        
        additional_context = industry_prompts.get(industry.lower(), "")
        
        return await self.generate_post(
            topic=topic,
            key_points=[f"Industry insights for {industry}"],
            target_audience=f"{target_role} in {company_size} {industry} companies",
            industry=industry,
            additional_context=additional_context
        )
    
    async def generate_storytelling_post(
        self,
        personal_experience: str,
        lesson_learned: str,
        industry_application: str
    ) -> Dict[str, Any]:
        """
        Generate storytelling LinkedIn post.
        
        Args:
            personal_experience: Personal story or experience
            lesson_learned: Key lesson from the experience
            industry_application: How it applies to the industry
            
        Returns:
            Generated post data
        """
        
        return await self.generate_post(
            topic="Personal Growth Story",
            key_points=[
                personal_experience,
                lesson_learned,
                industry_application
            ],
            target_audience="Professionals seeking growth and inspiration",
            industry="Professional Development",
            tone=ContentTone.INSPIRATIONAL,
            additional_context="Focus on authentic storytelling and vulnerability"
        )
    
    async def generate_thought_leadership_post(
        self,
        industry_trend: str,
        analysis: str,
        prediction: str
    ) -> Dict[str, Any]:
        """
        Generate thought leadership LinkedIn post.
        
        Args:
            industry_trend: Current industry trend
            analysis: Analysis of the trend
            prediction: Future prediction
            
        Returns:
            Generated post data
        """
        
        return await self.generate_post(
            topic=f"Thought Leadership: {industry_trend}",
            key_points=[
                industry_trend,
                analysis,
                prediction
            ],
            target_audience="Industry leaders and decision makers",
            industry="Thought Leadership",
            tone=ContentTone.AUTHORITATIVE,
            additional_context="Position as industry expert with unique insights"
        ) 