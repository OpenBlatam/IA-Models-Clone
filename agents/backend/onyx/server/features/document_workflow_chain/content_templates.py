"""
Content Templates System
========================

This module provides predefined content templates for different types
of documents, including blog posts, articles, tutorials, and more.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ContentTemplate:
    """Content template definition"""
    id: str
    name: str
    description: str
    category: str
    template_structure: Dict[str, Any]
    prompt_template: str
    target_word_count: int
    seo_keywords: List[str]
    metadata: Dict[str, Any]

class ContentTemplateManager:
    """Manager for content templates"""
    
    def __init__(self):
        self.templates: Dict[str, ContentTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default content templates"""
        
        # Blog Post Template
        blog_post_template = ContentTemplate(
            id="blog_post",
            name="Blog Post",
            description="Standard blog post with introduction, body, and conclusion",
            category="blogging",
            template_structure={
                "introduction": {
                    "length": "10-15%",
                    "elements": ["hook", "problem_statement", "preview"]
                },
                "body": {
                    "length": "70-80%",
                    "sections": ["main_points", "examples", "data"]
                },
                "conclusion": {
                    "length": "10-15%",
                    "elements": ["summary", "call_to_action"]
                }
            },
            prompt_template="""
            Write a comprehensive blog post about "{topic}" with the following structure:
            
            Introduction (10-15% of content):
            - Start with a compelling hook that grabs attention
            - Identify the problem or challenge your readers face
            - Preview what they'll learn from this post
            
            Body (70-80% of content):
            - Present 3-5 main points with clear explanations
            - Include relevant examples, case studies, or data
            - Use subheadings to organize information
            - Add actionable insights and practical tips
            
            Conclusion (10-15% of content):
            - Summarize the key takeaways
            - Include a clear call-to-action
            - Encourage reader engagement
            
            Target length: {word_count} words
            Tone: {tone}
            Target audience: {audience}
            """,
            target_word_count=1500,
            seo_keywords=["blog", "guide", "tips", "how to"],
            metadata={
                "readability_target": 0.7,
                "engagement_elements": ["questions", "examples", "lists"],
                "seo_optimized": True
            }
        )
        
        # Tutorial Template
        tutorial_template = ContentTemplate(
            id="tutorial",
            name="Step-by-Step Tutorial",
            description="Detailed tutorial with clear steps and instructions",
            category="education",
            template_structure={
                "introduction": {
                    "length": "15%",
                    "elements": ["overview", "prerequisites", "outcomes"]
                },
                "steps": {
                    "length": "75%",
                    "format": "numbered_steps",
                    "elements": ["explanation", "example", "tip"]
                },
                "conclusion": {
                    "length": "10%",
                    "elements": ["summary", "next_steps", "resources"]
                }
            },
            prompt_template="""
            Create a detailed step-by-step tutorial about "{topic}" with the following structure:
            
            Introduction (15% of content):
            - Provide a clear overview of what the tutorial covers
            - List any prerequisites or tools needed
            - Explain what the reader will achieve by the end
            
            Step-by-Step Instructions (75% of content):
            - Break down the process into clear, numbered steps
            - Include detailed explanations for each step
            - Provide examples or screenshots where helpful
            - Add tips and warnings for common mistakes
            
            Conclusion (10% of content):
            - Summarize what was accomplished
            - Suggest next steps or advanced topics
            - Provide additional resources or links
            
            Target length: {word_count} words
            Difficulty level: {difficulty}
            Target audience: {audience}
            """,
            target_word_count=2000,
            seo_keywords=["tutorial", "how to", "step by step", "guide"],
            metadata={
                "readability_target": 0.8,
                "engagement_elements": ["steps", "examples", "tips"],
                "seo_optimized": True
            }
        )
        
        # Product Description Template
        product_template = ContentTemplate(
            id="product_description",
            name="Product Description",
            description="Compelling product description with features and benefits",
            category="ecommerce",
            template_structure={
                "headline": {
                    "length": "5%",
                    "elements": ["product_name", "key_benefit"]
                },
                "features": {
                    "length": "40%",
                    "format": "bullet_points",
                    "elements": ["feature", "benefit"]
                },
                "description": {
                    "length": "40%",
                    "elements": ["problem_solution", "use_cases"]
                },
                "call_to_action": {
                    "length": "15%",
                    "elements": ["urgency", "action"]
                }
            },
            prompt_template="""
            Write a compelling product description for "{product_name}" with the following structure:
            
            Headline (5% of content):
            - Product name with key benefit
            - Attention-grabbing statement
            
            Key Features (40% of content):
            - List 5-7 main features as bullet points
            - Focus on benefits, not just features
            - Use action-oriented language
            
            Detailed Description (40% of content):
            - Explain the problem this product solves
            - Describe how it works
            - Include use cases and scenarios
            - Address potential objections
            
            Call to Action (15% of content):
            - Create urgency or scarcity
            - Clear action steps
            - Risk-free guarantee or trial
            
            Target length: {word_count} words
            Product category: {category}
            Target audience: {audience}
            """,
            target_word_count=800,
            seo_keywords=["product", "features", "benefits", "buy"],
            metadata={
                "readability_target": 0.6,
                "engagement_elements": ["benefits", "social_proof", "urgency"],
                "seo_optimized": True
            }
        )
        
        # News Article Template
        news_template = ContentTemplate(
            id="news_article",
            name="News Article",
            description="Objective news article with facts and analysis",
            category="journalism",
            template_structure={
                "lead": {
                    "length": "20%",
                    "elements": ["who", "what", "when", "where", "why"]
                },
                "body": {
                    "length": "70%",
                    "format": "inverted_pyramid",
                    "elements": ["facts", "quotes", "context"]
                },
                "conclusion": {
                    "length": "10%",
                    "elements": ["implications", "future_developments"]
                }
            },
            prompt_template="""
            Write a professional news article about "{topic}" following journalistic standards:
            
            Lead Paragraph (20% of content):
            - Answer the 5 W's: Who, What, When, Where, Why
            - Most important information first
            - Clear and concise statement
            
            Body (70% of content):
            - Present facts in order of importance
            - Include relevant quotes from sources
            - Provide context and background
            - Use objective, neutral tone
            
            Conclusion (10% of content):
            - Discuss implications of the news
            - Mention future developments or next steps
            - Avoid editorializing
            
            Target length: {word_count} words
            News category: {category}
            Target audience: {audience}
            """,
            target_word_count=1000,
            seo_keywords=["news", "breaking", "update", "report"],
            metadata={
                "readability_target": 0.7,
                "engagement_elements": ["facts", "quotes", "timeliness"],
                "seo_optimized": True
            }
        )
        
        # Social Media Post Template
        social_template = ContentTemplate(
            id="social_media_post",
            name="Social Media Post",
            description="Engaging social media post with hashtags and call-to-action",
            category="social_media",
            template_structure={
                "hook": {
                    "length": "30%",
                    "elements": ["attention_grabber", "question"]
                },
                "content": {
                    "length": "50%",
                    "elements": ["value", "story", "tip"]
                },
                "call_to_action": {
                    "length": "20%",
                    "elements": ["engagement", "hashtags"]
                }
            },
            prompt_template="""
            Create an engaging social media post about "{topic}" with the following structure:
            
            Hook (30% of content):
            - Start with an attention-grabbing statement or question
            - Make it relatable to your audience
            - Create curiosity or urgency
            
            Content (50% of content):
            - Provide valuable information or insight
            - Tell a brief story or share a tip
            - Keep it conversational and authentic
            - Use emojis appropriately
            
            Call to Action (20% of content):
            - Encourage engagement (likes, comments, shares)
            - Include relevant hashtags (3-5)
            - Ask a question to start conversation
            
            Target length: {word_count} words
            Platform: {platform}
            Target audience: {audience}
            """,
            target_word_count=200,
            seo_keywords=["social", "engagement", "viral", "trending"],
            metadata={
                "readability_target": 0.8,
                "engagement_elements": ["questions", "emojis", "hashtags"],
                "seo_optimized": False
            }
        )
        
        # Add templates to manager
        self.templates = {
            "blog_post": blog_post_template,
            "tutorial": tutorial_template,
            "product_description": product_template,
            "news_article": news_template,
            "social_media_post": social_template
        }
        
        logger.info(f"Initialized {len(self.templates)} content templates")
    
    def get_template(self, template_id: str) -> Optional[ContentTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: str) -> List[ContentTemplate]:
        """Get all templates in a category"""
        return [template for template in self.templates.values() 
                if template.category == category]
    
    def get_all_templates(self) -> List[ContentTemplate]:
        """Get all available templates"""
        return list(self.templates.values())
    
    def create_custom_template(
        self,
        template_id: str,
        name: str,
        description: str,
        category: str,
        template_structure: Dict[str, Any],
        prompt_template: str,
        target_word_count: int,
        seo_keywords: List[str],
        metadata: Dict[str, Any] = None
    ) -> ContentTemplate:
        """Create a custom template"""
        template = ContentTemplate(
            id=template_id,
            name=name,
            description=description,
            category=category,
            template_structure=template_structure,
            prompt_template=prompt_template,
            target_word_count=target_word_count,
            seo_keywords=seo_keywords,
            metadata=metadata or {}
        )
        
        self.templates[template_id] = template
        logger.info(f"Created custom template: {template_id}")
        return template
    
    def generate_prompt_from_template(
        self,
        template_id: str,
        topic: str,
        **kwargs
    ) -> str:
        """Generate a prompt using a template"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Default values
        defaults = {
            "word_count": template.target_word_count,
            "tone": "professional",
            "audience": "general",
            "difficulty": "beginner",
            "category": "general",
            "platform": "general"
        }
        
        # Merge with provided values
        params = {**defaults, **kwargs, "topic": topic}
        
        # Format the prompt template
        try:
            prompt = template.prompt_template.format(**params)
            return prompt
        except KeyError as e:
            logger.error(f"Missing parameter for template {template_id}: {e}")
            raise
    
    def get_template_recommendations(
        self,
        content_type: str,
        word_count: int,
        audience: str = "general"
    ) -> List[ContentTemplate]:
        """Get template recommendations based on requirements"""
        recommendations = []
        
        for template in self.templates.values():
            score = 0
            
            # Word count compatibility
            if abs(template.target_word_count - word_count) < 500:
                score += 2
            elif abs(template.target_word_count - word_count) < 1000:
                score += 1
            
            # Category match
            if content_type.lower() in template.category.lower():
                score += 3
            elif content_type.lower() in template.name.lower():
                score += 2
            
            # Audience compatibility
            if audience in template.metadata.get("target_audiences", []):
                score += 1
            
            if score > 0:
                recommendations.append((template, score))
        
        # Sort by score and return templates
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [template for template, score in recommendations]

# Global template manager
template_manager = ContentTemplateManager()

# Example usage
if __name__ == "__main__":
    async def test_templates():
        print("🧪 Testing Content Templates")
        print("=" * 40)
        
        # Get all templates
        templates = template_manager.get_all_templates()
        print(f"Available templates: {len(templates)}")
        
        for template in templates:
            print(f"- {template.name} ({template.category})")
        
        # Generate prompt from template
        prompt = template_manager.generate_prompt_from_template(
            "blog_post",
            "Artificial Intelligence in Marketing",
            word_count=1200,
            tone="conversational",
            audience="marketing professionals"
        )
        
        print(f"\nGenerated prompt length: {len(prompt)} characters")
        print(f"First 200 characters: {prompt[:200]}...")
        
        # Get recommendations
        recommendations = template_manager.get_template_recommendations(
            "tutorial",
            1500,
            "beginners"
        )
        
        print(f"\nRecommendations for tutorial (1500 words, beginners):")
        for template in recommendations[:3]:
            print(f"- {template.name}")
    
    asyncio.run(test_templates())


