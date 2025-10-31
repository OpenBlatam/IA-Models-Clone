"""
Intelligent Content Generation System
====================================

This module provides advanced AI-powered content generation with machine learning,
adaptive prompting, and intelligent content optimization.
"""

import asyncio
import logging
import json
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, Counter
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class GenerationContext:
    """Context for intelligent content generation"""
    topic: str
    target_audience: str
    content_type: str
    tone: str
    length_preference: str
    quality_requirements: List[str]
    previous_content: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class GenerationResult:
    """Result of intelligent content generation"""
    content: str
    title: str
    metadata: Dict[str, Any]
    quality_score: float
    engagement_prediction: float
    seo_score: float
    readability_score: float
    generation_time: float
    tokens_used: int
    confidence_score: float
    suggestions: List[str] = field(default_factory=list)

@dataclass
class LearningPattern:
    """Pattern learned from content generation"""
    pattern_type: str
    pattern_data: Dict[str, Any]
    success_rate: float
    usage_count: int
    last_used: datetime
    effectiveness_score: float

class IntelligentGenerator:
    """Advanced AI content generator with learning capabilities"""
    
    def __init__(self):
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.content_templates: Dict[str, Dict[str, Any]] = {}
        self.quality_indicators = self._initialize_quality_indicators()
        self.engagement_predictors = self._initialize_engagement_predictors()
        
    def _initialize_quality_indicators(self) -> Dict[str, List[str]]:
        """Initialize quality indicators for content assessment"""
        return {
            "high_quality": [
                "comprehensive", "detailed", "insightful", "well-researched",
                "professional", "engaging", "clear", "structured"
            ],
            "medium_quality": [
                "informative", "useful", "relevant", "organized",
                "readable", "helpful", "practical"
            ],
            "low_quality": [
                "basic", "simple", "generic", "repetitive",
                "unclear", "disorganized", "superficial"
            ]
        }
    
    def _initialize_engagement_predictors(self) -> Dict[str, List[str]]:
        """Initialize engagement prediction indicators"""
        return {
            "high_engagement": [
                "story", "example", "case study", "personal experience",
                "question", "challenge", "solution", "insight"
            ],
            "medium_engagement": [
                "tip", "guide", "tutorial", "how-to",
                "benefit", "advantage", "feature"
            ],
            "low_engagement": [
                "definition", "overview", "summary", "list",
                "description", "explanation"
            ]
        }
    
    async def generate_intelligent_content(
        self,
        context: GenerationContext,
        ai_client: Any,
        optimization_level: str = "balanced"
    ) -> GenerationResult:
        """
        Generate content using intelligent AI with learning capabilities
        
        Args:
            context: Generation context
            ai_client: AI client for content generation
            optimization_level: Level of optimization (basic, balanced, advanced)
            
        Returns:
            GenerationResult: Generated content with metadata
        """
        try:
            start_time = datetime.now()
            
            # Analyze context and apply learning
            optimized_context = await self._optimize_context_with_learning(context)
            
            # Generate adaptive prompt
            adaptive_prompt = await self._create_adaptive_prompt(optimized_context)
            
            # Generate content using AI
            raw_content = await self._generate_with_ai(ai_client, adaptive_prompt, context)
            
            # Post-process and optimize content
            optimized_content = await self._post_process_content(raw_content, context)
            
            # Generate title
            title = await self._generate_intelligent_title(optimized_content, context)
            
            # Analyze quality and engagement
            quality_metrics = await self._analyze_content_quality(optimized_content)
            engagement_prediction = await self._predict_engagement(optimized_content, context)
            
            # Calculate final scores
            final_scores = await self._calculate_final_scores(
                optimized_content, quality_metrics, engagement_prediction
            )
            
            # Generate suggestions
            suggestions = await self._generate_improvement_suggestions(
                optimized_content, final_scores, context
            )
            
            # Record performance for learning
            await self._record_generation_performance(
                context, optimized_content, final_scores, start_time
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return GenerationResult(
                content=optimized_content,
                title=title,
                metadata={
                    "context": context.__dict__,
                    "optimization_level": optimization_level,
                    "learning_patterns_used": list(optimized_context.get("applied_patterns", [])),
                    "generation_timestamp": start_time.isoformat()
                },
                quality_score=final_scores["quality"],
                engagement_prediction=final_scores["engagement"],
                seo_score=final_scores["seo"],
                readability_score=final_scores["readability"],
                generation_time=generation_time,
                tokens_used=final_scores.get("tokens_used", 0),
                confidence_score=final_scores["confidence"],
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error in intelligent content generation: {str(e)}")
            return GenerationResult(
                content="",
                title="",
                metadata={"error": str(e)},
                quality_score=0.0,
                engagement_prediction=0.0,
                seo_score=0.0,
                readability_score=0.0,
                generation_time=0.0,
                tokens_used=0,
                confidence_score=0.0,
                suggestions=["Error in content generation"]
            )
    
    async def _optimize_context_with_learning(self, context: GenerationContext) -> Dict[str, Any]:
        """Optimize context using learned patterns"""
        try:
            optimized = context.__dict__.copy()
            applied_patterns = []
            
            # Apply learned patterns for similar contexts
            context_key = self._generate_context_key(context)
            similar_patterns = await self._find_similar_patterns(context_key)
            
            for pattern in similar_patterns:
                if pattern.effectiveness_score > 0.7:  # Only use highly effective patterns
                    optimized = await self._apply_pattern(optimized, pattern)
                    applied_patterns.append(pattern.pattern_type)
            
            # Add user preferences
            user_id = context.user_preferences.get("user_id", "default")
            if user_id in self.user_preferences:
                user_prefs = self.user_preferences[user_id]
                optimized.update(user_prefs)
            
            optimized["applied_patterns"] = applied_patterns
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing context: {str(e)}")
            return context.__dict__
    
    async def _create_adaptive_prompt(self, context: Dict[str, Any]) -> str:
        """Create adaptive prompt based on context and learning"""
        try:
            base_prompt = f"Write a {context.get('content_type', 'article')} about {context.get('topic', 'the topic')}"
            
            # Add audience-specific elements
            audience = context.get('target_audience', 'general')
            if audience != 'general':
                base_prompt += f" for {audience} audience"
            
            # Add tone requirements
            tone = context.get('tone', 'professional')
            base_prompt += f" with a {tone} tone"
            
            # Add length requirements
            length = context.get('length_preference', 'medium')
            length_mapping = {
                'short': '300-500 words',
                'medium': '800-1200 words',
                'long': '1500-2500 words'
            }
            base_prompt += f" ({length_mapping.get(length, '800-1200 words')})"
            
            # Add quality requirements
            quality_reqs = context.get('quality_requirements', [])
            if quality_reqs:
                base_prompt += f". Ensure the content is {', '.join(quality_reqs)}"
            
            # Add learned optimizations
            if context.get('applied_patterns'):
                base_prompt += self._add_pattern_optimizations(context['applied_patterns'])
            
            # Add structure requirements
            base_prompt += ". Structure the content with clear sections, engaging introduction, and actionable conclusion."
            
            return base_prompt
            
        except Exception as e:
            logger.error(f"Error creating adaptive prompt: {str(e)}")
            return f"Write a comprehensive article about {context.get('topic', 'the topic')}"
    
    async def _generate_with_ai(self, ai_client: Any, prompt: str, context: GenerationContext) -> str:
        """Generate content using AI client"""
        try:
            # Use AI client to generate content
            if hasattr(ai_client, 'generate_text'):
                response = await ai_client.generate_text(prompt)
                return response.get('content', '') if isinstance(response, dict) else str(response)
            else:
                # Fallback for different AI client interfaces
                return await ai_client.generate(prompt)
                
        except Exception as e:
            logger.error(f"Error generating with AI: {str(e)}")
            return f"Error generating content: {str(e)}"
    
    async def _post_process_content(self, content: str, context: GenerationContext) -> str:
        """Post-process and optimize generated content"""
        try:
            # Clean up content
            processed = content.strip()
            
            # Ensure proper structure
            if not processed.startswith('#'):
                processed = f"# {context.topic}\n\n{processed}"
            
            # Add engaging elements based on context
            if context.tone == 'engaging':
                processed = await self._add_engaging_elements(processed)
            
            # Optimize for readability
            processed = await self._optimize_readability(processed)
            
            # Add call-to-action if appropriate
            if context.content_type in ['blog', 'article', 'guide']:
                processed = await self._add_call_to_action(processed, context)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error post-processing content: {str(e)}")
            return content
    
    async def _generate_intelligent_title(self, content: str, context: GenerationContext) -> str:
        """Generate intelligent title based on content and context"""
        try:
            # Extract key phrases from content
            key_phrases = await self._extract_key_phrases(content)
            
            # Generate title based on content type and tone
            if context.content_type == 'blog':
                title = await self._generate_blog_title(key_phrases, context.tone)
            elif context.content_type == 'article':
                title = await self._generate_article_title(key_phrases, context.tone)
            else:
                title = await self._generate_generic_title(key_phrases, context.tone)
            
            # Optimize title for engagement
            title = await self._optimize_title_engagement(title, context)
            
            return title
            
        except Exception as e:
            logger.error(f"Error generating title: {str(e)}")
            return context.topic.title()
    
    async def _analyze_content_quality(self, content: str) -> Dict[str, float]:
        """Analyze content quality using multiple metrics"""
        try:
            metrics = {}
            
            # Word count and structure
            word_count = len(content.split())
            metrics['word_count'] = word_count
            
            # Sentence structure
            sentences = re.split(r'[.!?]+', content)
            avg_sentence_length = word_count / len(sentences) if sentences else 0
            metrics['avg_sentence_length'] = avg_sentence_length
            
            # Paragraph structure
            paragraphs = content.split('\n\n')
            metrics['paragraph_count'] = len(paragraphs)
            
            # Quality indicators
            quality_score = 0.0
            for category, indicators in self.quality_indicators.items():
                matches = sum(1 for indicator in indicators if indicator in content.lower())
                if category == 'high_quality':
                    quality_score += matches * 0.3
                elif category == 'medium_quality':
                    quality_score += matches * 0.2
                else:
                    quality_score -= matches * 0.1
            
            metrics['quality_score'] = min(max(quality_score, 0.0), 1.0)
            
            # Readability score (simplified Flesch Reading Ease)
            readability = await self._calculate_readability(content)
            metrics['readability_score'] = readability
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing content quality: {str(e)}")
            return {"quality_score": 0.5, "readability_score": 0.5}
    
    async def _predict_engagement(self, content: str, context: GenerationContext) -> float:
        """Predict content engagement based on various factors"""
        try:
            engagement_score = 0.0
            
            # Check for engagement indicators
            for category, indicators in self.engagement_predictors.items():
                matches = sum(1 for indicator in indicators if indicator in content.lower())
                if category == 'high_engagement':
                    engagement_score += matches * 0.3
                elif category == 'medium_engagement':
                    engagement_score += matches * 0.2
                else:
                    engagement_score += matches * 0.1
            
            # Check for interactive elements
            interactive_elements = ['question', '?', 'challenge', 'think', 'consider']
            interactive_matches = sum(1 for element in interactive_elements if element in content.lower())
            engagement_score += interactive_matches * 0.1
            
            # Check for storytelling elements
            story_elements = ['story', 'example', 'case', 'experience', 'scenario']
            story_matches = sum(1 for element in story_elements if element in content.lower())
            engagement_score += story_matches * 0.15
            
            # Normalize score
            engagement_score = min(max(engagement_score, 0.0), 1.0)
            
            return engagement_score
            
        except Exception as e:
            logger.error(f"Error predicting engagement: {str(e)}")
            return 0.5
    
    async def _calculate_final_scores(
        self,
        content: str,
        quality_metrics: Dict[str, float],
        engagement_prediction: float
    ) -> Dict[str, float]:
        """Calculate final scores for the generated content"""
        try:
            scores = {}
            
            # Quality score
            scores['quality'] = quality_metrics.get('quality_score', 0.5)
            
            # Engagement score
            scores['engagement'] = engagement_prediction
            
            # SEO score (simplified)
            seo_score = await self._calculate_seo_score(content)
            scores['seo'] = seo_score
            
            # Readability score
            scores['readability'] = quality_metrics.get('readability_score', 0.5)
            
            # Confidence score (combination of all scores)
            scores['confidence'] = (
                scores['quality'] * 0.3 +
                scores['engagement'] * 0.25 +
                scores['seo'] * 0.2 +
                scores['readability'] * 0.25
            )
            
            # Token estimation
            scores['tokens_used'] = len(content) // 4
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating final scores: {str(e)}")
            return {
                'quality': 0.5,
                'engagement': 0.5,
                'seo': 0.5,
                'readability': 0.5,
                'confidence': 0.5,
                'tokens_used': 0
            }
    
    async def _generate_improvement_suggestions(
        self,
        content: str,
        scores: Dict[str, float],
        context: GenerationContext
    ) -> List[str]:
        """Generate suggestions for content improvement"""
        try:
            suggestions = []
            
            # Quality suggestions
            if scores['quality'] < 0.6:
                suggestions.append("Add more detailed explanations and examples to improve content quality")
            
            # Engagement suggestions
            if scores['engagement'] < 0.5:
                suggestions.append("Include more interactive elements like questions or case studies")
            
            # SEO suggestions
            if scores['seo'] < 0.6:
                suggestions.append("Optimize headings and add more relevant keywords")
            
            # Readability suggestions
            if scores['readability'] < 0.5:
                suggestions.append("Simplify sentence structure and use shorter paragraphs")
            
            # Length suggestions
            word_count = len(content.split())
            if word_count < 300:
                suggestions.append("Expand content with more detailed information")
            elif word_count > 2000:
                suggestions.append("Consider breaking content into multiple sections")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return ["Review content for overall quality and engagement"]
    
    async def _record_generation_performance(
        self,
        context: GenerationContext,
        content: str,
        scores: Dict[str, float],
        start_time: datetime
    ):
        """Record generation performance for learning"""
        try:
            performance_record = {
                "timestamp": start_time.isoformat(),
                "context": context.__dict__,
                "content_length": len(content),
                "scores": scores,
                "generation_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.performance_history.append(performance_record)
            
            # Update learning patterns
            await self._update_learning_patterns(context, scores)
            
            # Keep only recent history (last 1000 records)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error recording performance: {str(e)}")
    
    async def _update_learning_patterns(
        self,
        context: GenerationContext,
        scores: Dict[str, float]
    ):
        """Update learning patterns based on performance"""
        try:
            context_key = self._generate_context_key(context)
            
            if context_key not in self.learning_patterns:
                self.learning_patterns[context_key] = LearningPattern(
                    pattern_type=context_key,
                    pattern_data=context.__dict__,
                    success_rate=0.0,
                    usage_count=0,
                    last_used=datetime.now(),
                    effectiveness_score=0.0
                )
            
            pattern = self.learning_patterns[context_key]
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            
            # Update success rate based on confidence score
            confidence = scores.get('confidence', 0.5)
            pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + confidence) / pattern.usage_count
            
            # Update effectiveness score
            pattern.effectiveness_score = (
                pattern.success_rate * 0.6 +
                (1.0 / (1.0 + pattern.usage_count)) * 0.4  # Decay factor
            )
            
        except Exception as e:
            logger.error(f"Error updating learning patterns: {str(e)}")
    
    def _generate_context_key(self, context: GenerationContext) -> str:
        """Generate a key for context-based learning"""
        key_elements = [
            context.content_type,
            context.tone,
            context.length_preference,
            context.target_audience
        ]
        return hashlib.md5('|'.join(key_elements).encode()).hexdigest()[:16]
    
    async def _find_similar_patterns(self, context_key: str) -> List[LearningPattern]:
        """Find similar learning patterns"""
        try:
            similar_patterns = []
            for pattern in self.learning_patterns.values():
                if pattern.effectiveness_score > 0.5:
                    similar_patterns.append(pattern)
            
            # Sort by effectiveness
            similar_patterns.sort(key=lambda p: p.effectiveness_score, reverse=True)
            return similar_patterns[:5]  # Return top 5 patterns
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {str(e)}")
            return []
    
    async def _apply_pattern(self, context: Dict[str, Any], pattern: LearningPattern) -> Dict[str, Any]:
        """Apply a learning pattern to context"""
        try:
            # Apply pattern-specific optimizations
            if pattern.pattern_type.startswith('blog'):
                context['quality_requirements'] = context.get('quality_requirements', []) + ['engaging', 'shareable']
            elif pattern.pattern_type.startswith('article'):
                context['quality_requirements'] = context.get('quality_requirements', []) + ['comprehensive', 'well-researched']
            
            return context
            
        except Exception as e:
            logger.error(f"Error applying pattern: {str(e)}")
            return context
    
    def _add_pattern_optimizations(self, applied_patterns: List[str]) -> str:
        """Add optimizations based on applied patterns"""
        optimizations = []
        
        for pattern in applied_patterns:
            if 'blog' in pattern:
                optimizations.append("include engaging examples and personal insights")
            elif 'article' in pattern:
                optimizations.append("provide comprehensive coverage with data and statistics")
            elif 'guide' in pattern:
                optimizations.append("structure as step-by-step instructions with clear headings")
        
        if optimizations:
            return f". {'. '.join(optimizations)}"
        return ""
    
    async def _add_engaging_elements(self, content: str) -> str:
        """Add engaging elements to content"""
        try:
            # Add questions if not present
            if '?' not in content:
                content = f"Have you ever wondered about this topic? {content}"
            
            # Add examples if not present
            if 'example' not in content.lower():
                content += "\n\nFor example, consider this scenario..."
            
            return content
            
        except Exception as e:
            logger.error(f"Error adding engaging elements: {str(e)}")
            return content
    
    async def _optimize_readability(self, content: str) -> str:
        """Optimize content for readability"""
        try:
            # Split long sentences
            sentences = re.split(r'[.!?]+', content)
            optimized_sentences = []
            
            for sentence in sentences:
                if len(sentence.split()) > 25:  # Long sentence
                    # Split at conjunctions
                    parts = re.split(r'\s+(and|but|or|so|yet|for|nor)\s+', sentence)
                    if len(parts) > 1:
                        optimized_sentences.extend([part.strip() for part in parts if part.strip()])
                    else:
                        optimized_sentences.append(sentence.strip())
                else:
                    optimized_sentences.append(sentence.strip())
            
            return '. '.join(optimized_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Error optimizing readability: {str(e)}")
            return content
    
    async def _add_call_to_action(self, content: str, context: GenerationContext) -> str:
        """Add appropriate call-to-action to content"""
        try:
            cta_options = {
                'blog': "What are your thoughts on this topic? Share your experience in the comments below!",
                'article': "For more insights on this topic, explore our related articles and resources.",
                'guide': "Ready to get started? Follow these steps and let us know how it goes!"
            }
            
            cta = cta_options.get(context.content_type, "What questions do you have about this topic?")
            return f"{content}\n\n{cta}"
            
        except Exception as e:
            logger.error(f"Error adding call-to-action: {str(e)}")
            return content
    
    async def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content"""
        try:
            # Simple key phrase extraction
            words = re.findall(r'\b\w+\b', content.lower())
            word_freq = Counter(words)
            
            # Filter out common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
            
            key_phrases = [word for word, freq in word_freq.most_common(10) 
                          if word not in common_words and len(word) > 3]
            
            return key_phrases[:5]  # Return top 5 key phrases
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            return []
    
    async def _generate_blog_title(self, key_phrases: List[str], tone: str) -> str:
        """Generate blog title"""
        try:
            if not key_phrases:
                return "Insights and Tips"
            
            main_phrase = key_phrases[0].title()
            
            if tone == 'engaging':
                return f"The Ultimate Guide to {main_phrase}: Everything You Need to Know"
            elif tone == 'professional':
                return f"Understanding {main_phrase}: A Comprehensive Analysis"
            else:
                return f"{main_phrase}: Key Insights and Best Practices"
                
        except Exception as e:
            logger.error(f"Error generating blog title: {str(e)}")
            return "Blog Post Title"
    
    async def _generate_article_title(self, key_phrases: List[str], tone: str) -> str:
        """Generate article title"""
        try:
            if not key_phrases:
                return "Comprehensive Analysis"
            
            main_phrase = key_phrases[0].title()
            
            if tone == 'professional':
                return f"An In-Depth Analysis of {main_phrase}"
            elif tone == 'engaging':
                return f"Discovering {main_phrase}: What You Need to Know"
            else:
                return f"{main_phrase}: A Detailed Examination"
                
        except Exception as e:
            logger.error(f"Error generating article title: {str(e)}")
            return "Article Title"
    
    async def _generate_generic_title(self, key_phrases: List[str], tone: str) -> str:
        """Generate generic title"""
        try:
            if not key_phrases:
                return "Content Title"
            
            main_phrase = key_phrases[0].title()
            return f"{main_phrase}: Insights and Information"
            
        except Exception as e:
            logger.error(f"Error generating generic title: {str(e)}")
            return "Content Title"
    
    async def _optimize_title_engagement(self, title: str, context: GenerationContext) -> str:
        """Optimize title for engagement"""
        try:
            # Add engagement words if tone is engaging
            if context.tone == 'engaging':
                engagement_words = ['Ultimate', 'Complete', 'Essential', 'Must-Know', 'Proven']
                if not any(word in title for word in engagement_words):
                    title = f"Ultimate {title}"
            
            # Add numbers if appropriate
            if context.content_type == 'guide' and 'step' not in title.lower():
                title = f"5-Step {title}"
            
            return title
            
        except Exception as e:
            logger.error(f"Error optimizing title engagement: {str(e)}")
            return title
    
    async def _calculate_readability(self, content: str) -> float:
        """Calculate readability score (simplified)"""
        try:
            sentences = re.split(r'[.!?]+', content)
            words = content.split()
            
            if not sentences or not words:
                return 0.5
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simplified readability calculation
            readability = 1.0 - (avg_sentence_length / 50.0) - (avg_word_length / 10.0)
            return max(0.0, min(1.0, readability))
            
        except Exception as e:
            logger.error(f"Error calculating readability: {str(e)}")
            return 0.5
    
    async def _calculate_seo_score(self, content: str) -> float:
        """Calculate SEO score (simplified)"""
        try:
            score = 0.0
            
            # Check for headings
            if '#' in content:
                score += 0.2
            
            # Check for keywords in title/headings
            if any(word in content.lower() for word in ['introduction', 'conclusion', 'summary']):
                score += 0.2
            
            # Check for internal structure
            if len(content.split('\n\n')) > 3:  # Multiple paragraphs
                score += 0.2
            
            # Check for word count (SEO-friendly length)
            word_count = len(content.split())
            if 300 <= word_count <= 2000:
                score += 0.2
            elif word_count > 2000:
                score += 0.1
            
            # Check for engaging elements
            if any(word in content.lower() for word in ['example', 'case study', 'tip', 'guide']):
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating SEO score: {str(e)}")
            return 0.5

# Global instance
intelligent_generator = IntelligentGenerator()

# Example usage
if __name__ == "__main__":
    async def test_intelligent_generation():
        print("🧠 Testing Intelligent Content Generation")
        print("=" * 50)
        
        # Create test context
        context = GenerationContext(
            topic="Artificial Intelligence in Marketing",
            target_audience="marketing professionals",
            content_type="blog",
            tone="engaging",
            length_preference="medium",
            quality_requirements=["comprehensive", "practical"],
            user_preferences={"user_id": "test_user"}
        )
        
        # Mock AI client
        class MockAIClient:
            async def generate_text(self, prompt):
                return {
                    'content': f"Generated content about {context.topic}. This is a comprehensive article that covers all aspects of the topic with practical examples and actionable insights."
                }
        
        ai_client = MockAIClient()
        
        # Generate content
        result = await intelligent_generator.generate_intelligent_content(
            context, ai_client, optimization_level="advanced"
        )
        
        print(f"Generated Title: {result.title}")
        print(f"Content Length: {len(result.content)} characters")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Engagement Prediction: {result.engagement_prediction:.2f}")
        print(f"SEO Score: {result.seo_score:.2f}")
        print(f"Readability Score: {result.readability_score:.2f}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Generation Time: {result.generation_time:.2f}s")
        print(f"Suggestions: {result.suggestions}")
    
    asyncio.run(test_intelligent_generation())


