"""
Content Optimizer - AI-powered content optimization and enhancement
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re
from collections import Counter

import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from sentence_transformers import SentenceTransformer
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSuggestion:
    """Content optimization suggestion"""
    suggestion_type: str
    priority: int  # 1-5, 5 being highest priority
    description: str
    original_text: str
    suggested_text: str
    improvement_score: float
    category: str  # readability, seo, engagement, grammar, style


@dataclass
class ContentOptimization:
    """Content optimization result"""
    content_id: str
    original_content: str
    optimized_content: str
    optimization_score: float
    suggestions: List[OptimizationSuggestion]
    improvements: Dict[str, float]
    seo_improvements: Dict[str, Any]
    readability_improvements: Dict[str, Any]
    engagement_improvements: Dict[str, Any]
    optimization_timestamp: datetime


@dataclass
class SEOAnalysis:
    """SEO analysis result"""
    content_id: str
    title_optimization: Dict[str, Any]
    meta_description: str
    keyword_density: Dict[str, float]
    heading_structure: Dict[str, Any]
    internal_links: List[str]
    external_links: List[str]
    image_alt_texts: List[str]
    seo_score: float
    recommendations: List[str]
    analysis_timestamp: datetime


class ContentOptimizer:
    """AI-powered content optimizer"""
    
    def __init__(self):
        self.grammar_checker = None
        self.style_improver = None
        self.seo_optimizer = None
        self.readability_enhancer = None
        self.engagement_booster = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        
    async def initialize(self) -> None:
        """Initialize optimization models"""
        try:
            logger.info("Initializing Content Optimizer...")
            
            # Load models asynchronously
            await asyncio.gather(
                self._load_grammar_checker(),
                self._load_style_improver(),
                self._load_seo_optimizer(),
                self._load_readability_enhancer(),
                self._load_engagement_booster()
            )
            
            self.models_loaded = True
            logger.info("Content Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Content Optimizer: {e}")
            raise
    
    async def _load_grammar_checker(self) -> None:
        """Load grammar checking model"""
        try:
            # Using a text generation model for grammar improvement
            self.grammar_checker = pipeline(
                "text2text-generation",
                model="t5-base",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load grammar checker: {e}")
            self.grammar_checker = None
    
    async def _load_style_improver(self) -> None:
        """Load style improvement model"""
        try:
            self.style_improver = pipeline(
                "text2text-generation",
                model="facebook/bart-base",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load style improver: {e}")
            self.style_improver = None
    
    async def _load_seo_optimizer(self) -> None:
        """Load SEO optimization model"""
        try:
            # Using a text generation model for SEO optimization
            self.seo_optimizer = pipeline(
                "text2text-generation",
                model="t5-small",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load SEO optimizer: {e}")
            self.seo_optimizer = None
    
    async def _load_readability_enhancer(self) -> None:
        """Load readability enhancement model"""
        try:
            self.readability_enhancer = pipeline(
                "text2text-generation",
                model="t5-base",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load readability enhancer: {e}")
            self.readability_enhancer = None
    
    async def _load_engagement_booster(self) -> None:
        """Load engagement boosting model"""
        try:
            self.engagement_booster = pipeline(
                "text2text-generation",
                model="facebook/bart-base",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load engagement booster: {e}")
            self.engagement_booster = None
    
    async def optimize_content(
        self, 
        content: str, 
        content_id: str = "",
        optimization_goals: List[str] = None
    ) -> ContentOptimization:
        """Optimize content based on specified goals"""
        
        if not self.models_loaded:
            raise Exception("Optimization models not loaded. Call initialize() first.")
        
        if optimization_goals is None:
            optimization_goals = ["readability", "seo", "engagement", "grammar"]
        
        try:
            # Generate optimization suggestions
            suggestions = []
            improvements = {}
            
            # Readability optimization
            if "readability" in optimization_goals:
                readability_suggestions, readability_improvements = await self._optimize_readability(content)
                suggestions.extend(readability_suggestions)
                improvements["readability"] = readability_improvements
            
            # SEO optimization
            if "seo" in optimization_goals:
                seo_suggestions, seo_improvements = await self._optimize_seo(content)
                suggestions.extend(seo_suggestions)
                improvements["seo"] = seo_improvements
            
            # Engagement optimization
            if "engagement" in optimization_goals:
                engagement_suggestions, engagement_improvements = await self._optimize_engagement(content)
                suggestions.extend(engagement_suggestions)
                improvements["engagement"] = engagement_improvements
            
            # Grammar optimization
            if "grammar" in optimization_goals:
                grammar_suggestions, grammar_improvements = await self._optimize_grammar(content)
                suggestions.extend(grammar_suggestions)
                improvements["grammar"] = grammar_improvements
            
            # Apply optimizations to create optimized content
            optimized_content = await self._apply_optimizations(content, suggestions)
            
            # Calculate overall optimization score
            optimization_score = await self._calculate_optimization_score(improvements)
            
            # Separate improvements by category
            seo_improvements = improvements.get("seo", {})
            readability_improvements = improvements.get("readability", {})
            engagement_improvements = improvements.get("engagement", {})
            
            return ContentOptimization(
                content_id=content_id,
                original_content=content,
                optimized_content=optimized_content,
                optimization_score=optimization_score,
                suggestions=suggestions,
                improvements=improvements,
                seo_improvements=seo_improvements,
                readability_improvements=readability_improvements,
                engagement_improvements=engagement_improvements,
                optimization_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in content optimization: {e}")
            raise
    
    async def _optimize_readability(self, content: str) -> Tuple[List[OptimizationSuggestion], Dict[str, float]]:
        """Optimize content for readability"""
        suggestions = []
        improvements = {}
        
        try:
            # Analyze current readability
            current_readability = await self._calculate_readability_score(content)
            
            # Identify readability issues
            sentences = re.split(r'[.!?]+', content)
            words = content.split()
            
            # Check for long sentences
            long_sentences = [s for s in sentences if len(s.split()) > 25]
            if long_sentences:
                for sentence in long_sentences[:3]:  # Limit to 3 suggestions
                    # Simple sentence splitting suggestion
                    words_in_sentence = sentence.split()
                    if len(words_in_sentence) > 25:
                        mid_point = len(words_in_sentence) // 2
                        suggested_text = ' '.join(words_in_sentence[:mid_point]) + '. ' + ' '.join(words_in_sentence[mid_point:])
                        
                        suggestions.append(OptimizationSuggestion(
                            suggestion_type="sentence_length",
                            priority=3,
                            description="Split long sentence for better readability",
                            original_text=sentence.strip(),
                            suggested_text=suggested_text,
                            improvement_score=0.3,
                            category="readability"
                        ))
            
            # Check for complex words
            complex_words = await self._identify_complex_words(content)
            if complex_words:
                for word in complex_words[:3]:  # Limit to 3 suggestions
                    simpler_word = await self._find_simpler_word(word)
                    if simpler_word:
                        suggestions.append(OptimizationSuggestion(
                            suggestion_type="word_complexity",
                            priority=2,
                            description=f"Replace complex word with simpler alternative",
                            original_text=word,
                            suggested_text=simpler_word,
                            improvement_score=0.2,
                            category="readability"
                        ))
            
            # Calculate potential improvement
            potential_readability = min(100, current_readability + len(suggestions) * 5)
            improvements["readability_score"] = potential_readability - current_readability
            improvements["current_readability"] = current_readability
            improvements["potential_readability"] = potential_readability
            
        except Exception as e:
            logger.warning(f"Readability optimization failed: {e}")
        
        return suggestions, improvements
    
    async def _optimize_seo(self, content: str) -> Tuple[List[OptimizationSuggestion], Dict[str, Any]]:
        """Optimize content for SEO"""
        suggestions = []
        improvements = {}
        
        try:
            # Analyze current SEO
            seo_analysis = await self._analyze_seo(content)
            
            # Check for title optimization
            if not seo_analysis.get("has_title"):
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="missing_title",
                    priority=5,
                    description="Add a compelling title for better SEO",
                    original_text="",
                    suggested_text=await self._generate_title(content),
                    improvement_score=0.4,
                    category="seo"
                ))
            
            # Check for meta description
            if not seo_analysis.get("has_meta_description"):
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="missing_meta_description",
                    priority=4,
                    description="Add a meta description for better search visibility",
                    original_text="",
                    suggested_text=await self._generate_meta_description(content),
                    improvement_score=0.3,
                    category="seo"
                ))
            
            # Check for heading structure
            if seo_analysis.get("heading_count", 0) < 2:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="heading_structure",
                    priority=3,
                    description="Add more headings to improve content structure",
                    original_text="",
                    suggested_text="Add H2 and H3 headings to break up content",
                    improvement_score=0.2,
                    category="seo"
                ))
            
            # Check for keyword density
            keyword_density = seo_analysis.get("keyword_density", {})
            if not keyword_density:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="keyword_optimization",
                    priority=3,
                    description="Add relevant keywords to improve search ranking",
                    original_text="",
                    suggested_text="Include relevant keywords naturally throughout the content",
                    improvement_score=0.3,
                    category="seo"
                ))
            
            improvements["seo_score"] = seo_analysis.get("seo_score", 0)
            improvements["current_seo"] = seo_analysis
            improvements["potential_seo"] = seo_analysis.get("seo_score", 0) + len(suggestions) * 10
            
        except Exception as e:
            logger.warning(f"SEO optimization failed: {e}")
        
        return suggestions, improvements
    
    async def _optimize_engagement(self, content: str) -> Tuple[List[OptimizationSuggestion], Dict[str, Any]]:
        """Optimize content for engagement"""
        suggestions = []
        improvements = {}
        
        try:
            # Analyze current engagement factors
            engagement_analysis = await self._analyze_engagement(content)
            
            # Check for hooks and attention-grabbing elements
            if not engagement_analysis.get("has_hook"):
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="missing_hook",
                    priority=4,
                    description="Add an engaging opening hook to capture attention",
                    original_text="",
                    suggested_text=await self._generate_hook(content),
                    improvement_score=0.4,
                    category="engagement"
                ))
            
            # Check for call-to-action
            if not engagement_analysis.get("has_cta"):
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="missing_cta",
                    priority=3,
                    description="Add a clear call-to-action to encourage reader engagement",
                    original_text="",
                    suggested_text=await self._generate_cta(content),
                    improvement_score=0.3,
                    category="engagement"
                ))
            
            # Check for questions and interactive elements
            if engagement_analysis.get("question_count", 0) < 2:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="interactive_elements",
                    priority=2,
                    description="Add questions or interactive elements to increase engagement",
                    original_text="",
                    suggested_text="Include rhetorical questions or prompts for reader interaction",
                    improvement_score=0.2,
                    category="engagement"
                ))
            
            # Check for emotional language
            if engagement_analysis.get("emotional_score", 0) < 0.3:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="emotional_language",
                    priority=2,
                    description="Add emotional language to increase reader connection",
                    original_text="",
                    suggested_text="Use more emotional and descriptive language",
                    improvement_score=0.2,
                    category="engagement"
                ))
            
            improvements["engagement_score"] = engagement_analysis.get("engagement_score", 0)
            improvements["current_engagement"] = engagement_analysis
            improvements["potential_engagement"] = engagement_analysis.get("engagement_score", 0) + len(suggestions) * 0.1
            
        except Exception as e:
            logger.warning(f"Engagement optimization failed: {e}")
        
        return suggestions, improvements
    
    async def _optimize_grammar(self, content: str) -> Tuple[List[OptimizationSuggestion], Dict[str, Any]]:
        """Optimize content for grammar and style"""
        suggestions = []
        improvements = {}
        
        try:
            # Analyze current grammar
            grammar_analysis = await self._analyze_grammar(content)
            
            # Check for common grammar issues
            if grammar_analysis.get("passive_voice_count", 0) > 3:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="passive_voice",
                    priority=2,
                    description="Reduce passive voice for more direct and engaging writing",
                    original_text="",
                    suggested_text="Convert passive voice to active voice where possible",
                    improvement_score=0.2,
                    category="grammar"
                ))
            
            # Check for repetitive words
            repetitive_words = grammar_analysis.get("repetitive_words", [])
            if repetitive_words:
                for word in repetitive_words[:2]:  # Limit to 2 suggestions
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type="word_repetition",
                        priority=2,
                        description=f"Replace repetitive use of '{word}' with synonyms",
                        original_text=word,
                        suggested_text=await self._find_synonym(word),
                        improvement_score=0.1,
                        category="grammar"
                    ))
            
            # Check for sentence variety
            if grammar_analysis.get("sentence_variety_score", 0) < 0.5:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="sentence_variety",
                    priority=2,
                    description="Vary sentence length and structure for better flow",
                    original_text="",
                    suggested_text="Mix short and long sentences, and vary sentence beginnings",
                    improvement_score=0.2,
                    category="grammar"
                ))
            
            improvements["grammar_score"] = grammar_analysis.get("grammar_score", 0)
            improvements["current_grammar"] = grammar_analysis
            improvements["potential_grammar"] = grammar_analysis.get("grammar_score", 0) + len(suggestions) * 0.1
            
        except Exception as e:
            logger.warning(f"Grammar optimization failed: {e}")
        
        return suggestions, improvements
    
    async def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score"""
        try:
            sentences = re.split(r'[.!?]+', content)
            words = content.split()
            
            if not sentences or not words:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Flesch Reading Ease
            readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            return max(0.0, min(100.0, readability))
            
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return 0.0
    
    async def _identify_complex_words(self, content: str) -> List[str]:
        """Identify complex words in content"""
        try:
            words = content.lower().split()
            complex_words = []
            
            # Simple heuristic: words longer than 8 characters or with 3+ syllables
            for word in words:
                if len(word) > 8 or self._count_syllables(word) >= 3:
                    complex_words.append(word)
            
            return list(set(complex_words))[:10]  # Return top 10 unique complex words
            
        except Exception as e:
            logger.warning(f"Complex word identification failed: {e}")
            return []
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    async def _find_simpler_word(self, word: str) -> Optional[str]:
        """Find a simpler alternative for a complex word"""
        # Simple word replacement dictionary
        word_replacements = {
            "utilize": "use",
            "facilitate": "help",
            "implement": "put in place",
            "comprehensive": "complete",
            "substantial": "large",
            "demonstrate": "show",
            "establish": "set up",
            "acquire": "get",
            "commence": "start",
            "terminate": "end"
        }
        
        return word_replacements.get(word.lower())
    
    async def _analyze_seo(self, content: str) -> Dict[str, Any]:
        """Analyze SEO aspects of content"""
        try:
            # Check for title (simple heuristic: first line or first 60 chars)
            has_title = len(content.split('\n')[0]) < 60
            
            # Check for meta description (placeholder)
            has_meta_description = False
            
            # Count headings (simple heuristic: lines starting with # or all caps)
            heading_count = len([line for line in content.split('\n') 
                               if line.strip().startswith('#') or line.strip().isupper()])
            
            # Extract keywords (simple word frequency)
            words = content.lower().split()
            word_freq = Counter(words)
            keywords = {word: count for word, count in word_freq.items() 
                       if len(word) > 3 and count > 1}
            
            # Calculate SEO score
            seo_score = 0
            if has_title:
                seo_score += 20
            if has_meta_description:
                seo_score += 20
            if heading_count >= 2:
                seo_score += 20
            if keywords:
                seo_score += 20
            if len(content) > 300:
                seo_score += 20
            
            return {
                "has_title": has_title,
                "has_meta_description": has_meta_description,
                "heading_count": heading_count,
                "keyword_density": keywords,
                "seo_score": seo_score
            }
            
        except Exception as e:
            logger.warning(f"SEO analysis failed: {e}")
            return {"seo_score": 0}
    
    async def _analyze_engagement(self, content: str) -> Dict[str, Any]:
        """Analyze engagement factors in content"""
        try:
            # Check for hooks (questions, statistics, bold statements)
            has_hook = any([
                content.startswith(('Did you know', 'Have you ever', 'What if')),
                '?' in content[:100],
                any(word in content[:100].lower() for word in ['amazing', 'incredible', 'shocking'])
            ])
            
            # Check for call-to-action
            cta_indicators = ['click here', 'learn more', 'find out', 'discover', 'get started']
            has_cta = any(cta in content.lower() for cta in cta_indicators)
            
            # Count questions
            question_count = content.count('?')
            
            # Calculate emotional score (simple heuristic)
            emotional_words = ['love', 'hate', 'amazing', 'terrible', 'excited', 'worried', 'happy', 'sad']
            emotional_count = sum(content.lower().count(word) for word in emotional_words)
            emotional_score = min(1.0, emotional_count / len(content.split()) * 100)
            
            # Calculate engagement score
            engagement_score = 0
            if has_hook:
                engagement_score += 0.3
            if has_cta:
                engagement_score += 0.3
            if question_count >= 2:
                engagement_score += 0.2
            if emotional_score > 0.1:
                engagement_score += 0.2
            
            return {
                "has_hook": has_hook,
                "has_cta": has_cta,
                "question_count": question_count,
                "emotional_score": emotional_score,
                "engagement_score": engagement_score
            }
            
        except Exception as e:
            logger.warning(f"Engagement analysis failed: {e}")
            return {"engagement_score": 0}
    
    async def _analyze_grammar(self, content: str) -> Dict[str, Any]:
        """Analyze grammar and style aspects"""
        try:
            sentences = content.split('.')
            
            # Count passive voice (simple heuristic)
            passive_indicators = ['was', 'were', 'been', 'being']
            passive_voice_count = sum(content.lower().count(indicator) for indicator in passive_indicators)
            
            # Find repetitive words
            words = content.lower().split()
            word_freq = Counter(words)
            repetitive_words = [word for word, count in word_freq.items() 
                              if count > 3 and len(word) > 3]
            
            # Calculate sentence variety score
            sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
            if sentence_lengths:
                avg_length = sum(sentence_lengths) / len(sentence_lengths)
                length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
                sentence_variety_score = min(1.0, length_variance / (avg_length ** 2))
            else:
                sentence_variety_score = 0
            
            # Calculate grammar score
            grammar_score = 0.5  # Base score
            if passive_voice_count < 5:
                grammar_score += 0.2
            if len(repetitive_words) < 3:
                grammar_score += 0.2
            if sentence_variety_score > 0.3:
                grammar_score += 0.1
            
            return {
                "passive_voice_count": passive_voice_count,
                "repetitive_words": repetitive_words,
                "sentence_variety_score": sentence_variety_score,
                "grammar_score": grammar_score
            }
            
        except Exception as e:
            logger.warning(f"Grammar analysis failed: {e}")
            return {"grammar_score": 0.5}
    
    async def _generate_title(self, content: str) -> str:
        """Generate a title for content"""
        # Simple title generation from first sentence
        first_sentence = content.split('.')[0]
        if len(first_sentence) > 60:
            return first_sentence[:57] + "..."
        return first_sentence
    
    async def _generate_meta_description(self, content: str) -> str:
        """Generate a meta description for content"""
        # Simple meta description generation
        words = content.split()
        if len(words) > 20:
            return ' '.join(words[:20]) + "..."
        return content
    
    async def _generate_hook(self, content: str) -> str:
        """Generate an engaging hook for content"""
        # Simple hook generation
        return "Did you know that " + content.split('.')[0].lower()
    
    async def _generate_cta(self, content: str) -> str:
        """Generate a call-to-action for content"""
        # Simple CTA generation
        return "Learn more about this topic and discover how it can benefit you."
    
    async def _find_synonym(self, word: str) -> str:
        """Find a synonym for a word"""
        # Simple synonym dictionary
        synonyms = {
            "good": "excellent",
            "bad": "poor",
            "big": "large",
            "small": "tiny",
            "fast": "quick",
            "slow": "gradual",
            "new": "recent",
            "old": "ancient",
            "important": "crucial",
            "easy": "simple"
        }
        
        return synonyms.get(word.lower(), word)
    
    async def _apply_optimizations(self, content: str, suggestions: List[OptimizationSuggestion]) -> str:
        """Apply optimization suggestions to content"""
        optimized_content = content
        
        # Apply high-priority suggestions first
        sorted_suggestions = sorted(suggestions, key=lambda x: x.priority, reverse=True)
        
        for suggestion in sorted_suggestions[:5]:  # Apply top 5 suggestions
            if suggestion.original_text and suggestion.suggested_text:
                optimized_content = optimized_content.replace(
                    suggestion.original_text, 
                    suggestion.suggested_text
                )
        
        return optimized_content
    
    async def _calculate_optimization_score(self, improvements: Dict[str, Any]) -> float:
        """Calculate overall optimization score"""
        try:
            total_score = 0
            count = 0
            
            for category, improvement in improvements.items():
                if isinstance(improvement, dict) and "score" in improvement:
                    total_score += improvement["score"]
                    count += 1
            
            return total_score / count if count > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Optimization score calculation failed: {e}")
            return 0.0
    
    async def analyze_seo(self, content: str, content_id: str = "") -> SEOAnalysis:
        """Perform comprehensive SEO analysis"""
        
        try:
            # Analyze SEO aspects
            seo_data = await self._analyze_seo(content)
            
            # Generate title optimization
            title_optimization = {
                "current_title": content.split('\n')[0] if content else "",
                "suggested_title": await self._generate_title(content),
                "title_length": len(content.split('\n')[0]) if content else 0,
                "optimal_length": 50,
                "is_optimal": len(content.split('\n')[0]) <= 60 if content else False
            }
            
            # Generate meta description
            meta_description = await self._generate_meta_description(content)
            
            # Analyze keyword density
            words = content.lower().split()
            word_freq = Counter(words)
            keyword_density = {word: (count / len(words)) * 100 
                             for word, count in word_freq.items() 
                             if len(word) > 3 and count > 1}
            
            # Analyze heading structure
            lines = content.split('\n')
            headings = [line.strip() for line in lines 
                       if line.strip().startswith('#') or line.strip().isupper()]
            
            heading_structure = {
                "total_headings": len(headings),
                "heading_levels": [line.count('#') for line in headings if line.startswith('#')],
                "has_h1": any(line.startswith('# ') for line in headings),
                "has_h2": any(line.startswith('## ') for line in headings)
            }
            
            # Extract links (simple heuristic)
            internal_links = []
            external_links = []
            
            # Extract image alt texts (simple heuristic)
            image_alt_texts = []
            
            # Generate SEO recommendations
            recommendations = []
            if not title_optimization["is_optimal"]:
                recommendations.append("Optimize title length (recommended: 50-60 characters)")
            if not seo_data.get("has_meta_description"):
                recommendations.append("Add a compelling meta description")
            if heading_structure["total_headings"] < 2:
                recommendations.append("Add more headings to improve content structure")
            if not keyword_density:
                recommendations.append("Include relevant keywords naturally throughout content")
            
            return SEOAnalysis(
                content_id=content_id,
                title_optimization=title_optimization,
                meta_description=meta_description,
                keyword_density=keyword_density,
                heading_structure=heading_structure,
                internal_links=internal_links,
                external_links=external_links,
                image_alt_texts=image_alt_texts,
                seo_score=seo_data.get("seo_score", 0),
                recommendations=recommendations,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in SEO analysis: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of content optimizer"""
        return {
            "status": "healthy" if self.models_loaded else "unhealthy",
            "models_loaded": self.models_loaded,
            "device": self.device,
            "available_models": {
                "grammar_checker": self.grammar_checker is not None,
                "style_improver": self.style_improver is not None,
                "seo_optimizer": self.seo_optimizer is not None,
                "readability_enhancer": self.readability_enhancer is not None,
                "engagement_booster": self.engagement_booster is not None
            },
            "timestamp": datetime.now().isoformat()
        }


# Global content optimizer instance
content_optimizer = ContentOptimizer()


async def initialize_content_optimizer() -> None:
    """Initialize the global content optimizer"""
    await content_optimizer.initialize()


async def optimize_content(
    content: str, 
    content_id: str = "", 
    optimization_goals: List[str] = None
) -> ContentOptimization:
    """Optimize content using AI"""
    return await content_optimizer.optimize_content(content, content_id, optimization_goals)


async def analyze_seo(content: str, content_id: str = "") -> SEOAnalysis:
    """Analyze SEO aspects of content"""
    return await content_optimizer.analyze_seo(content, content_id)


async def get_content_optimizer_health() -> Dict[str, Any]:
    """Get content optimizer health status"""
    return await content_optimizer.health_check()




