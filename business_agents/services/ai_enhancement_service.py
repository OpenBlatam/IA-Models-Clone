"""
AI Enhancement Service
======================

Advanced AI-powered enhancement service for business agents with multi-model support,
content optimization, and intelligent automation.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import openai
from anthropic import Anthropic
import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)

class EnhancementType(Enum):
    CONTENT_OPTIMIZATION = "content_optimization"
    SEO_ENHANCEMENT = "seo_enhancement"
    TONE_ADJUSTMENT = "tone_adjustment"
    STRUCTURE_IMPROVEMENT = "structure_improvement"
    GRAMMAR_CORRECTION = "grammar_correction"
    READABILITY_ENHANCEMENT = "readability_enhancement"
    PERSONALIZATION = "personalization"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXPANSION = "expansion"

class ContentQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class EnhancementRequest:
    content: str
    enhancement_type: EnhancementType
    target_audience: Optional[str] = None
    business_context: Optional[Dict[str, Any]] = None
    quality_requirements: Optional[Dict[str, Any]] = None
    language: str = "en"
    tone: Optional[str] = None
    keywords: Optional[List[str]] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None

@dataclass
class EnhancementResult:
    original_content: str
    enhanced_content: str
    enhancement_type: EnhancementType
    quality_score: float
    improvements: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    confidence_score: float

@dataclass
class ContentAnalysis:
    readability_score: float
    sentiment_score: float
    keyword_density: Dict[str, float]
    structure_score: float
    grammar_score: float
    overall_quality: ContentQuality
    recommendations: List[str]

class AIEnhancementService:
    """
    Advanced AI-powered content enhancement service.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize AI clients
        self._initialize_clients()
        
        # Initialize text processing tools
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Enhancement templates
        self.enhancement_templates = self._load_enhancement_templates()
        
    def _initialize_clients(self):
        """Initialize AI service clients."""
        
        if self.config.get("openai_api_key"):
            openai.api_key = self.config["openai_api_key"]
            self.openai_client = openai
            
        if self.config.get("anthropic_api_key"):
            self.anthropic_client = Anthropic(
                api_key=self.config["anthropic_api_key"]
            )
            
    def _load_enhancement_templates(self) -> Dict[EnhancementType, str]:
        """Load enhancement templates for different types."""
        
        return {
            EnhancementType.CONTENT_OPTIMIZATION: """
            Optimize the following content for better engagement and clarity:
            
            Content: {content}
            Target Audience: {target_audience}
            Business Context: {business_context}
            
            Please:
            1. Improve clarity and readability
            2. Enhance engagement and persuasiveness
            3. Maintain the original message and intent
            4. Use active voice where appropriate
            5. Add compelling calls-to-action if relevant
            
            Return the optimized content.
            """,
            
            EnhancementType.SEO_ENHANCEMENT: """
            Optimize the following content for SEO:
            
            Content: {content}
            Target Keywords: {keywords}
            Business Context: {business_context}
            
            Please:
            1. Integrate keywords naturally
            2. Optimize headings and structure
            3. Improve meta descriptions
            4. Add relevant internal linking suggestions
            5. Enhance readability for both users and search engines
            
            Return the SEO-optimized content.
            """,
            
            EnhancementType.TONE_ADJUSTMENT: """
            Adjust the tone of the following content:
            
            Content: {content}
            Desired Tone: {tone}
            Target Audience: {target_audience}
            
            Please:
            1. Maintain the core message
            2. Adjust language and style to match the desired tone
            3. Ensure consistency throughout
            4. Keep it appropriate for the target audience
            
            Return the tone-adjusted content.
            """,
            
            EnhancementType.STRUCTURE_IMPROVEMENT: """
            Improve the structure and organization of the following content:
            
            Content: {content}
            Content Type: {content_type}
            
            Please:
            1. Create clear headings and subheadings
            2. Organize information logically
            3. Use bullet points and lists where appropriate
            4. Ensure smooth transitions between sections
            5. Add a compelling introduction and conclusion
            
            Return the restructured content.
            """,
            
            EnhancementType.READABILITY_ENHANCEMENT: """
            Enhance the readability of the following content:
            
            Content: {content}
            Target Reading Level: {reading_level}
            
            Please:
            1. Simplify complex sentences
            2. Use shorter paragraphs
            3. Replace jargon with simpler terms
            4. Add transitional phrases
            5. Ensure consistent terminology
            
            Return the readability-enhanced content.
            """,
            
            EnhancementType.PERSONALIZATION: """
            Personalize the following content for the target audience:
            
            Content: {content}
            Target Audience: {target_audience}
            Personalization Data: {personalization_data}
            
            Please:
            1. Use audience-specific language and references
            2. Address their specific pain points
            3. Include relevant examples and case studies
            4. Adjust the value proposition
            5. Make it feel tailored and relevant
            
            Return the personalized content.
            """,
            
            EnhancementType.SUMMARIZATION: """
            Create a concise summary of the following content:
            
            Content: {content}
            Target Length: {target_length}
            Key Points to Include: {key_points}
            
            Please:
            1. Capture the main ideas and key points
            2. Maintain the original meaning
            3. Use clear and concise language
            4. Include important details and statistics
            5. Ensure the summary is self-contained
            
            Return the summary.
            """,
            
            EnhancementType.EXPANSION: """
            Expand the following content with additional details and insights:
            
            Content: {content}
            Expansion Focus: {expansion_focus}
            Target Length: {target_length}
            
            Please:
            1. Add relevant details and examples
            2. Provide additional context and background
            3. Include supporting evidence and data
            4. Maintain the original structure and flow
            5. Ensure all additions are valuable and relevant
            
            Return the expanded content.
            """
        }
        
    async def enhance_content(self, request: EnhancementRequest) -> EnhancementResult:
        """Enhance content using AI."""
        
        start_time = datetime.now()
        
        try:
            # Analyze original content
            analysis = await self._analyze_content(request.content)
            
            # Get enhancement template
            template = self.enhancement_templates.get(request.enhancement_type)
            if not template:
                raise ValueError(f"No template found for enhancement type: {request.enhancement_type}")
            
            # Prepare enhancement prompt
            prompt = self._prepare_enhancement_prompt(template, request, analysis)
            
            # Generate enhanced content
            enhanced_content = await self._generate_enhanced_content(prompt, request)
            
            # Analyze enhanced content
            enhanced_analysis = await self._analyze_content(enhanced_content)
            
            # Calculate quality improvement
            quality_improvement = self._calculate_quality_improvement(analysis, enhanced_analysis)
            
            # Generate improvements and suggestions
            improvements = self._generate_improvements(analysis, enhanced_analysis)
            suggestions = self._generate_suggestions(enhanced_analysis)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(analysis, enhanced_analysis, quality_improvement)
            
            return EnhancementResult(
                original_content=request.content,
                enhanced_content=enhanced_content,
                enhancement_type=request.enhancement_type,
                quality_score=enhanced_analysis.overall_quality.value,
                improvements=improvements,
                suggestions=suggestions,
                metadata={
                    "original_analysis": asdict(analysis),
                    "enhanced_analysis": asdict(enhanced_analysis),
                    "quality_improvement": quality_improvement,
                    "enhancement_parameters": asdict(request)
                },
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Content enhancement failed: {str(e)}")
            raise
            
    async def _analyze_content(self, content: str) -> ContentAnalysis:
        """Analyze content quality and characteristics."""
        
        # Readability score (simplified Flesch Reading Ease)
        readability_score = self._calculate_readability_score(content)
        
        # Sentiment score
        sentiment_score = self._calculate_sentiment_score(content)
        
        # Keyword density
        keyword_density = self._calculate_keyword_density(content)
        
        # Structure score
        structure_score = self._calculate_structure_score(content)
        
        # Grammar score (simplified)
        grammar_score = self._calculate_grammar_score(content)
        
        # Overall quality assessment
        overall_quality = self._assess_overall_quality(
            readability_score, sentiment_score, structure_score, grammar_score
        )
        
        # Generate recommendations
        recommendations = self._generate_content_recommendations(
            readability_score, sentiment_score, structure_score, grammar_score
        )
        
        return ContentAnalysis(
            readability_score=readability_score,
            sentiment_score=sentiment_score,
            keyword_density=keyword_density,
            structure_score=structure_score,
            grammar_score=grammar_score,
            overall_quality=overall_quality,
            recommendations=recommendations
        )
        
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score (0-100, higher is more readable)."""
        
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
            
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0, min(100, score))
        
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
                
        if word.endswith('e'):
            syllable_count -= 1
            
        return max(1, syllable_count)
        
    def _calculate_sentiment_score(self, content: str) -> float:
        """Calculate sentiment score (-1 to 1, positive is good)."""
        
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding', 'superb']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'worst', 'hate']
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
            
        sentiment = (positive_count - negative_count) / total_words
        return max(-1, min(1, sentiment))
        
    def _calculate_keyword_density(self, content: str) -> Dict[str, float]:
        """Calculate keyword density."""
        
        words = content.lower().split()
        word_count = {}
        
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 2:  # Ignore short words
                word_count[word] = word_count.get(word, 0) + 1
                
        total_words = len(words)
        density = {word: count / total_words for word, count in word_count.items() if count > 1}
        
        # Return top 10 keywords
        return dict(sorted(density.items(), key=lambda x: x[1], reverse=True)[:10])
        
    def _calculate_structure_score(self, content: str) -> float:
        """Calculate content structure score."""
        
        score = 0.0
        
        # Check for headings
        if re.search(r'^#+\s', content, re.MULTILINE):
            score += 0.3
            
        # Check for lists
        if re.search(r'^\s*[-*+]\s', content, re.MULTILINE) or re.search(r'^\s*\d+\.\s', content, re.MULTILINE):
            score += 0.2
            
        # Check for paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.2
            
        # Check for transitions
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 'consequently']
        if any(word in content.lower() for word in transition_words):
            score += 0.1
            
        # Check for conclusion
        conclusion_words = ['in conclusion', 'to summarize', 'in summary', 'finally', 'overall']
        if any(phrase in content.lower() for phrase in conclusion_words):
            score += 0.2
            
        return min(1.0, score)
        
    def _calculate_grammar_score(self, content: str) -> float:
        """Calculate grammar score (simplified)."""
        
        # Basic grammar checks
        score = 1.0
        
        # Check for sentence capitalization
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                score -= 0.1
                
        # Check for proper punctuation
        if not content.endswith(('.', '!', '?')):
            score -= 0.1
            
        # Check for double spaces
        if '  ' in content:
            score -= 0.1
            
        # Check for common grammar issues
        grammar_issues = [
            r'\byour\s+you\b',  # your you
            r'\bthere\s+their\b',  # there their
            r'\bits\s+it\'s\b',  # its it's
        ]
        
        for pattern in grammar_issues:
            if re.search(pattern, content, re.IGNORECASE):
                score -= 0.1
                
        return max(0.0, score)
        
    def _assess_overall_quality(self, readability: float, sentiment: float, structure: float, grammar: float) -> ContentQuality:
        """Assess overall content quality."""
        
        # Weighted average
        overall_score = (readability * 0.3 + structure * 0.3 + grammar * 0.3 + (sentiment + 1) * 0.1 * 50)
        
        if overall_score >= 80:
            return ContentQuality.EXCELLENT
        elif overall_score >= 65:
            return ContentQuality.GOOD
        elif overall_score >= 50:
            return ContentQuality.FAIR
        else:
            return ContentQuality.POOR
            
    def _generate_content_recommendations(self, readability: float, sentiment: float, structure: float, grammar: float) -> List[str]:
        """Generate content improvement recommendations."""
        
        recommendations = []
        
        if readability < 60:
            recommendations.append("Improve readability by using shorter sentences and simpler words")
            
        if sentiment < -0.2:
            recommendations.append("Consider using more positive language to improve sentiment")
            
        if structure < 0.5:
            recommendations.append("Improve structure by adding headings, lists, and clear organization")
            
        if grammar < 0.8:
            recommendations.append("Review grammar and punctuation for better clarity")
            
        if not recommendations:
            recommendations.append("Content quality is good - consider minor optimizations for even better results")
            
        return recommendations
        
    def _prepare_enhancement_prompt(self, template: str, request: EnhancementRequest, analysis: ContentAnalysis) -> str:
        """Prepare enhancement prompt with context."""
        
        context = {
            "content": request.content,
            "target_audience": request.target_audience or "general audience",
            "business_context": json.dumps(request.business_context or {}),
            "keywords": ", ".join(request.keywords or []),
            "tone": request.tone or "professional",
            "content_type": "business document",
            "reading_level": "intermediate",
            "personalization_data": json.dumps(request.business_context or {}),
            "key_points": "main ideas and key insights",
            "target_length": request.max_length or "original length",
            "expansion_focus": "additional details and insights"
        }
        
        return template.format(**context)
        
    async def _generate_enhanced_content(self, prompt: str, request: EnhancementRequest) -> str:
        """Generate enhanced content using AI."""
        
        try:
            if self.openai_client:
                response = await self.openai_client.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a professional content enhancement specialist. Provide high-quality, improved content that maintains the original message while enhancing clarity, engagement, and effectiveness."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=request.max_length or 2000,
                    temperature=0.7
                )
                return response.choices[0].message.content
                
            elif self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=request.max_length or 2000,
                    temperature=0.7,
                    system="You are a professional content enhancement specialist. Provide high-quality, improved content that maintains the original message while enhancing clarity, engagement, and effectiveness.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            else:
                # Fallback to simple enhancement
                return self._simple_enhancement(request.content, request.enhancement_type)
                
        except Exception as e:
            logger.error(f"AI content generation failed: {str(e)}")
            return self._simple_enhancement(request.content, request.enhancement_type)
            
    def _simple_enhancement(self, content: str, enhancement_type: EnhancementType) -> str:
        """Simple fallback enhancement when AI is not available."""
        
        if enhancement_type == EnhancementType.GRAMMAR_CORRECTION:
            # Basic grammar corrections
            content = re.sub(r'\s+', ' ', content)  # Remove extra spaces
            content = content.strip()
            if not content.endswith(('.', '!', '?')):
                content += '.'
                
        elif enhancement_type == EnhancementType.READABILITY_ENHANCEMENT:
            # Break long sentences
            sentences = content.split('. ')
            enhanced_sentences = []
            for sentence in sentences:
                if len(sentence.split()) > 20:
                    # Split long sentences
                    words = sentence.split()
                    mid = len(words) // 2
                    enhanced_sentences.append(' '.join(words[:mid]) + '.')
                    enhanced_sentences.append(' '.join(words[mid:]))
                else:
                    enhanced_sentences.append(sentence)
            content = '. '.join(enhanced_sentences)
            
        return content
        
    def _calculate_quality_improvement(self, original: ContentAnalysis, enhanced: ContentAnalysis) -> Dict[str, float]:
        """Calculate quality improvement metrics."""
        
        return {
            "readability_improvement": enhanced.readability_score - original.readability_score,
            "sentiment_improvement": enhanced.sentiment_score - original.sentiment_score,
            "structure_improvement": enhanced.structure_score - original.structure_score,
            "grammar_improvement": enhanced.grammar_score - original.grammar_score,
            "overall_improvement": self._calculate_overall_improvement(original, enhanced)
        }
        
    def _calculate_overall_improvement(self, original: ContentAnalysis, enhanced: ContentAnalysis) -> float:
        """Calculate overall improvement score."""
        
        original_score = (
            original.readability_score * 0.3 +
            original.structure_score * 100 * 0.3 +
            original.grammar_score * 100 * 0.3 +
            (original.sentiment_score + 1) * 50 * 0.1
        )
        
        enhanced_score = (
            enhanced.readability_score * 0.3 +
            enhanced.structure_score * 100 * 0.3 +
            enhanced.grammar_score * 100 * 0.3 +
            (enhanced.sentiment_score + 1) * 50 * 0.1
        )
        
        return enhanced_score - original_score
        
    def _generate_improvements(self, original: ContentAnalysis, enhanced: ContentAnalysis) -> List[str]:
        """Generate list of improvements made."""
        
        improvements = []
        
        if enhanced.readability_score > original.readability_score:
            improvements.append(f"Improved readability by {enhanced.readability_score - original.readability_score:.1f} points")
            
        if enhanced.structure_score > original.structure_score:
            improvements.append("Enhanced content structure and organization")
            
        if enhanced.grammar_score > original.grammar_score:
            improvements.append("Improved grammar and punctuation")
            
        if enhanced.sentiment_score > original.sentiment_score:
            improvements.append("Enhanced positive sentiment and tone")
            
        if enhanced.overall_quality.value != original.overall_quality.value:
            improvements.append(f"Upgraded overall quality from {original.overall_quality.value} to {enhanced.overall_quality.value}")
            
        return improvements
        
    def _generate_suggestions(self, analysis: ContentAnalysis) -> List[str]:
        """Generate additional suggestions for further improvement."""
        
        suggestions = []
        
        if analysis.readability_score < 70:
            suggestions.append("Consider using more bullet points and shorter paragraphs")
            
        if analysis.structure_score < 0.7:
            suggestions.append("Add more headings and subheadings for better organization")
            
        if analysis.sentiment_score < 0:
            suggestions.append("Include more positive language and success stories")
            
        if analysis.grammar_score < 0.9:
            suggestions.append("Review for grammar and punctuation consistency")
            
        return suggestions
        
    def _calculate_confidence_score(self, original: ContentAnalysis, enhanced: ContentAnalysis, improvement: Dict[str, float]) -> float:
        """Calculate confidence score for the enhancement."""
        
        # Base confidence on improvement magnitude and consistency
        base_confidence = 0.5
        
        # Increase confidence based on improvements
        if improvement["overall_improvement"] > 10:
            base_confidence += 0.3
        elif improvement["overall_improvement"] > 5:
            base_confidence += 0.2
        elif improvement["overall_improvement"] > 0:
            base_confidence += 0.1
            
        # Increase confidence if multiple aspects improved
        improvements_count = sum(1 for v in improvement.values() if v > 0)
        base_confidence += improvements_count * 0.05
        
        # Decrease confidence if quality degraded
        if improvement["overall_improvement"] < -5:
            base_confidence -= 0.2
            
        return max(0.0, min(1.0, base_confidence))
        
    async def batch_enhance_content(self, requests: List[EnhancementRequest]) -> List[EnhancementResult]:
        """Enhance multiple content pieces in batch."""
        
        tasks = [self.enhance_content(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        enhanced_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch enhancement failed for request {i}: {str(result)}")
                # Create a fallback result
                enhanced_results.append(EnhancementResult(
                    original_content=requests[i].content,
                    enhanced_content=requests[i].content,
                    enhancement_type=requests[i].enhancement_type,
                    quality_score=0.0,
                    improvements=[],
                    suggestions=["Enhancement failed - please try again"],
                    metadata={"error": str(result)},
                    processing_time=0.0,
                    confidence_score=0.0
                ))
            else:
                enhanced_results.append(result)
                
        return enhanced_results
        
    async def get_enhancement_suggestions(self, content: str, business_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get suggestions for content enhancement."""
        
        analysis = await self._analyze_content(content)
        suggestions = []
        
        # Suggest enhancements based on analysis
        if analysis.readability_score < 60:
            suggestions.append({
                "type": EnhancementType.READABILITY_ENHANCEMENT,
                "priority": "high",
                "description": "Content readability is low - consider simplifying language and structure",
                "expected_improvement": "15-25% readability increase"
            })
            
        if analysis.structure_score < 0.5:
            suggestions.append({
                "type": EnhancementType.STRUCTURE_IMPROVEMENT,
                "priority": "high",
                "description": "Content structure needs improvement - add headings and better organization",
                "expected_improvement": "Better user experience and comprehension"
            })
            
        if analysis.sentiment_score < -0.1:
            suggestions.append({
                "type": EnhancementType.TONE_ADJUSTMENT,
                "priority": "medium",
                "description": "Content tone could be more positive and engaging",
                "expected_improvement": "Improved audience engagement"
            })
            
        if analysis.grammar_score < 0.8:
            suggestions.append({
                "type": EnhancementType.GRAMMAR_CORRECTION,
                "priority": "high",
                "description": "Grammar and punctuation need attention",
                "expected_improvement": "Professional appearance and clarity"
            })
            
        # Business context specific suggestions
        if business_context:
            if business_context.get("seo_required", False):
                suggestions.append({
                    "type": EnhancementType.SEO_ENHANCEMENT,
                    "priority": "medium",
                    "description": "Optimize content for search engines",
                    "expected_improvement": "Better search engine visibility"
                })
                
            if business_context.get("personalization_required", False):
                suggestions.append({
                    "type": EnhancementType.PERSONALIZATION,
                    "priority": "medium",
                    "description": "Personalize content for target audience",
                    "expected_improvement": "Higher engagement and conversion"
                })
                
        return suggestions





























