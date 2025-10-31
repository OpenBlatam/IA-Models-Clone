from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import re
from typing import Dict, List, Optional, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage
from ...shared.logging import get_logger
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Content Optimizer for LinkedIn Posts
===================================

AI-powered content optimization using LangChain for better LinkedIn engagement.
"""



logger = get_logger(__name__)


class ContentOptimizer:
    """
    Content optimizer for LinkedIn posts using LangChain.
    
    Provides various optimization techniques to improve
    post engagement and readability.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize the content optimizer."""
        self.llm = llm
        self._setup_optimization_chains()
    
    def _setup_optimization_chains(self) -> Any:
        """Setup optimization chains."""
        
        # Readability optimization
        self.readability_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content"],
                template="""Improve the readability of this LinkedIn post:

{content}

Focus on:
- Breaking up long paragraphs
- Using bullet points and lists
- Adding clear section breaks
- Improving sentence structure
- Making it more scannable

Return the optimized content only."""
            )
        )
        
        # Engagement optimization
        self.engagement_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content", "target_audience"],
                template="""Optimize this LinkedIn post for maximum engagement:

{content}

Target Audience: {target_audience}

Optimization techniques:
- Add compelling hooks
- Include questions to encourage comments
- Use storytelling elements
- Add relevant statistics or data
- Improve call-to-action
- Make it more shareable

Return the optimized content only."""
            )
        )
        
        # SEO optimization
        self.seo_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content", "keywords"],
                template="""Optimize this LinkedIn post for SEO and keyword targeting:

{content}

Target Keywords: {keywords}

Optimization techniques:
- Naturally incorporate keywords
- Improve keyword density
- Add relevant hashtags
- Optimize for LinkedIn's algorithm
- Include trending topics when relevant

Return the optimized content only."""
            )
        )
    
    async def optimize_readability(self, content: str) -> str:
        """Optimize content for better readability."""
        try:
            result = await self.readability_chain.arun(content=content)
            return result.strip()
        except Exception as e:
            logger.error(f"Error optimizing readability: {e}")
            return content
    
    async def optimize_engagement(self, content: str, target_audience: str) -> str:
        """Optimize content for better engagement."""
        try:
            result = await self.engagement_chain.arun(
                content=content,
                target_audience=target_audience
            )
            return result.strip()
        except Exception as e:
            logger.error(f"Error optimizing engagement: {e}")
            return content
    
    async def optimize_seo(self, content: str, keywords: List[str]) -> str:
        """Optimize content for SEO and keyword targeting."""
        try:
            keywords_str = ", ".join(keywords)
            result = await self.seo_chain.arun(
                content=content,
                keywords=keywords_str
            )
            return result.strip()
        except Exception as e:
            logger.error(f"Error optimizing SEO: {e}")
            return content
    
    async def optimize_comprehensive(
        self,
        content: str,
        target_audience: str,
        keywords: List[str]
    ) -> Dict[str, str]:
        """Perform comprehensive content optimization."""
        try:
            # Optimize in parallel
            readability_task = self.optimize_readability(content)
            engagement_task = self.optimize_engagement(content, target_audience)
            seo_task = self.optimize_seo(content, keywords)
            
            # Wait for all optimizations
            optimized_readability, optimized_engagement, optimized_seo = await asyncio.gather(
                readability_task, engagement_task, seo_task
            )
            
            # Combine optimizations
            final_optimized = await self._combine_optimizations(
                optimized_readability, optimized_engagement, optimized_seo
            )
            
            return {
                "original": content,
                "readability_optimized": optimized_readability,
                "engagement_optimized": optimized_engagement,
                "seo_optimized": optimized_seo,
                "final_optimized": final_optimized
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive optimization: {e}")
            return {"original": content, "final_optimized": content}
    
    async def _combine_optimizations(
        self,
        readability_content: str,
        engagement_content: str,
        seo_content: str
    ) -> str:
        """Combine different optimization approaches."""
        try:
            prompt = f"""
            Combine the best elements from these three optimized versions of a LinkedIn post:

            Readability Optimized:
            {readability_content}

            Engagement Optimized:
            {engagement_content}

            SEO Optimized:
            {seo_content}

            Create a final version that:
            1. Maintains excellent readability
            2. Maximizes engagement potential
            3. Includes SEO best practices
            4. Flows naturally and cohesively
            5. Preserves the original message

            Return the final optimized content only.
            """
            
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            logger.error(f"Error combining optimizations: {e}")
            return readability_content  # Fallback to readability version
    
    def analyze_content_structure(self, content: str) -> Dict[str, any]:
        """Analyze content structure and provide insights."""
        try:
            # Basic text analysis
            words = content.split()
            sentences = re.split(r'[.!?]+', content)
            paragraphs = content.split('\n\n')
            
            # Calculate metrics
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            paragraph_count = len([p for p in paragraphs if p.strip()])
            
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            avg_paragraph_length = word_count / paragraph_count if paragraph_count > 0 else 0
            
            # Analyze structure
            has_bullet_points = bool(re.search(r'[•\-\*]\s', content))
            has_hashtags = bool(re.search(r'#\w+', content))
            has_questions = bool(re.search(r'\?', content))
            has_call_to_action = bool(re.search(r'\b(comment|share|like|follow|connect|learn|discover)\b', content, re.IGNORECASE))
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_paragraph_length": round(avg_paragraph_length, 2),
                "has_bullet_points": has_bullet_points,
                "has_hashtags": has_hashtags,
                "has_questions": has_questions,
                "has_call_to_action": has_call_to_action,
                "readability_score": self._calculate_readability_score(content),
                "engagement_potential": self._calculate_engagement_potential(content)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content structure: {e}")
            return {}
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score (0-100)."""
        try:
            words = content.split()
            sentences = re.split(r'[.!?]+', content)
            syllables = self._count_syllables(content)
            
            if not words or not sentences:
                return 0.0
            
            # Flesch Reading Ease formula
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-100 scale
            return max(0, min(100, flesch_score))
            
        except Exception:
            return 50.0  # Default score
    
    def _calculate_engagement_potential(self, content: str) -> float:
        """Calculate engagement potential score (0-100)."""
        try:
            score = 50.0  # Base score
            
            # Factors that increase engagement
            if re.search(r'\?', content):  # Questions
                score += 15
            if re.search(r'#\w+', content):  # Hashtags
                score += 10
            if re.search(r'[•\-\*]\s', content):  # Bullet points
                score += 10
            if re.search(r'\b(comment|share|like|follow|connect)\b', content, re.IGNORECASE):  # CTAs
                score += 15
            if len(content.split()) > 100:  # Substantial content
                score += 10
            
            # Factors that decrease engagement
            if len(content.split()) > 500:  # Too long
                score -= 10
            if not re.search(r'[.!?]', content):  # No punctuation
                score -= 20
            
            return max(0, min(100, score))
            
        except Exception:
            return 50.0  # Default score
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (approximate)."""
        try:
            # Simple syllable counting
            text = text.lower()
            count = 0
            vowels = "aeiouy"
            on_vowel = False
            
            for char in text:
                is_vowel = char in vowels
                if is_vowel and not on_vowel:
                    count += 1
                on_vowel = is_vowel
            
            return count
        except Exception:
            return len(text.split())  # Fallback
    
    async def suggest_improvements(self, content: str, target_audience: str) -> List[str]:
        """Suggest specific improvements for the content."""
        try:
            prompt = f"""
            Analyze this LinkedIn post and suggest specific improvements:

            Content: {content}
            Target Audience: {target_audience}

            Provide 3-5 specific, actionable suggestions for improvement.
            Focus on:
            - Engagement optimization
            - Readability improvements
            - Call-to-action effectiveness
            - Content structure
            - Audience relevance

            Format as a numbered list of suggestions.
            """
            
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            suggestions_text = response.generations[0][0].text.strip()
            
            # Parse suggestions
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering/bullets
                    suggestion = re.sub(r'^\d+\.\s*|^\-\s*', '', line)
                    if suggestion:
                        suggestions.append(suggestion)
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting improvements: {e}")
            return []
    
    async def create_ab_test_variants(self, content: str, num_variants: int = 3) -> List[str]:
        """Create A/B test variants of the content."""
        try:
            variants = []
            
            variant_types = [
                "different opening hook",
                "alternative call-to-action",
                "different tone (more casual)",
                "different structure with bullet points",
                "alternative hashtag strategy"
            ]
            
            for i in range(min(num_variants, len(variant_types))):
                variant_type = variant_types[i]
                
                prompt = f"""
                Create an A/B test variant of this LinkedIn post:

                Original Content: {content}
                Variant Type: {variant_type}

                Create a variant that tests the {variant_type} while maintaining the core message and value.
                Make it distinctly different but equally effective.

                Return the variant content only.
                """
                
                response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
                variant = response.generations[0][0].text.strip()
                variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"Error creating A/B test variants: {e}")
            return [] 