"""
Optimization Core
================

Core optimization system for TruthGPT-based document generation.
Implements advanced optimization techniques for continuous improvement.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import numpy as np
from dataclasses import dataclass

from ..models.schemas import OptimizationLevel
from ..utils.metrics_collector import MetricsCollector
from ..utils.optimization_engine import OptimizationEngine
from ..utils.learning_system import LearningSystem

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of optimization process."""
    optimized_content: str
    optimization_score: float
    improvements_applied: List[str]
    metrics: Dict[str, Any]
    optimization_time: float

class OptimizationCore:
    """
    Optimization Core for TruthGPT system.
    
    Features:
    - Content optimization
    - Prompt optimization
    - Performance optimization
    - Continuous learning
    - A/B testing
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.optimization_engine = OptimizationEngine()
        self.learning_system = LearningSystem()
        self.optimization_history = []
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize the optimization core."""
        logger.info("Initializing Optimization Core...")
        
        try:
            await self.metrics_collector.initialize()
            await self.optimization_engine.initialize()
            await self.learning_system.initialize()
            
            # Initialize performance metrics
            self.performance_metrics = {
                "total_optimizations": 0,
                "successful_optimizations": 0,
                "average_improvement": 0.0,
                "optimization_speed": 0.0
            }
            
            logger.info("Optimization Core initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Optimization Core: {str(e)}")
            raise
    
    async def optimize_document(
        self, 
        document: Dict[str, Any], 
        optimization_level: OptimizationLevel
    ) -> OptimizationResult:
        """
        Optimize a document using TruthGPT techniques.
        
        Args:
            document: Document to optimize
            optimization_level: Level of optimization to apply
            
        Returns:
            Optimization result with improved document
        """
        try:
            logger.info(f"Optimizing document with level: {optimization_level}")
            start_time = datetime.utcnow()
            
            # Get current metrics
            current_metrics = await self._analyze_document_metrics(document)
            
            # Apply optimization techniques based on level
            optimization_techniques = self._get_optimization_techniques(optimization_level)
            
            optimized_content = document["content"]
            improvements_applied = []
            
            # Apply each optimization technique
            for technique in optimization_techniques:
                try:
                    result = await self._apply_optimization_technique(
                        optimized_content, 
                        technique, 
                        current_metrics
                    )
                    
                    if result["improved"]:
                        optimized_content = result["content"]
                        improvements_applied.append(technique["name"])
                        logger.info(f"Applied optimization: {technique['name']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to apply optimization {technique['name']}: {str(e)}")
                    continue
            
            # Calculate optimization metrics
            optimization_time = (datetime.utcnow() - start_time).total_seconds()
            optimization_score = await self._calculate_optimization_score(
                document["content"], 
                optimized_content, 
                current_metrics
            )
            
            # Create optimization result
            result = OptimizationResult(
                optimized_content=optimized_content,
                optimization_score=optimization_score,
                improvements_applied=improvements_applied,
                metrics=current_metrics,
                optimization_time=optimization_time
            )
            
            # Store optimization history
            await self._store_optimization_history(document, result)
            
            # Update performance metrics
            await self._update_performance_metrics(result)
            
            logger.info(f"Document optimization completed. Score: {optimization_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to optimize document: {str(e)}")
            raise
    
    async def _analyze_document_metrics(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document metrics for optimization."""
        try:
            content = document["content"]
            
            # Basic metrics
            metrics = {
                "length": len(content),
                "word_count": len(content.split()),
                "sentence_count": len(content.split('.')),
                "paragraph_count": len(content.split('\n\n')),
                "readability_score": self._calculate_readability(content),
                "structure_score": self._analyze_structure(content),
                "coherence_score": self._analyze_coherence(content),
                "clarity_score": self._analyze_clarity(content)
            }
            
            # Advanced metrics
            metrics.update({
                "vocabulary_diversity": self._calculate_vocabulary_diversity(content),
                "information_density": self._calculate_information_density(content),
                "engagement_score": self._calculate_engagement_score(content)
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to analyze document metrics: {str(e)}")
            return {}
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score."""
        try:
            words = content.split()
            sentences = content.split('.')
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            avg_words_per_sentence = len(words) / len(sentences)
            avg_syllables_per_word = self._calculate_avg_syllables(words)
            
            # Simple readability formula
            readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            return max(0.0, min(1.0, readability / 100))
            
        except Exception as e:
            logger.error(f"Failed to calculate readability: {str(e)}")
            return 0.5
    
    def _calculate_avg_syllables(self, words: List[str]) -> float:
        """Calculate average syllables per word."""
        try:
            total_syllables = 0
            for word in words:
                total_syllables += self._count_syllables(word)
            return total_syllables / len(words) if words else 0
        except Exception as e:
            logger.error(f"Failed to calculate syllables: {str(e)}")
            return 2.0  # Default average
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        try:
            word = word.lower()
            vowels = "aeiouy"
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            
            # Handle silent 'e'
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            return max(1, syllable_count)
        except Exception as e:
            logger.error(f"Failed to count syllables: {str(e)}")
            return 1
    
    def _analyze_structure(self, content: str) -> float:
        """Analyze content structure."""
        try:
            structure_indicators = {
                "paragraphs": content.count('\n\n'),
                "headers": content.count('#'),
                "lists": content.count('*') + content.count('-'),
                "numbered_lists": content.count('1.') + content.count('2.'),
                "bold_text": content.count('**'),
                "italic_text": content.count('*'),
                "links": content.count('['),
                "code_blocks": content.count('```')
            }
            
            # Calculate structure score
            total_indicators = sum(structure_indicators.values())
            if total_indicators > 0:
                return min(1.0, total_indicators / 20)  # Normalize to 0-1
            return 0.5
            
        except Exception as e:
            logger.error(f"Failed to analyze structure: {str(e)}")
            return 0.5
    
    def _analyze_coherence(self, content: str) -> float:
        """Analyze content coherence."""
        try:
            sentences = content.split('.')
            if len(sentences) < 2:
                return 0.5
            
            # Simple coherence analysis
            # Check for transition words and logical flow
            transition_words = [
                'however', 'therefore', 'moreover', 'furthermore', 'additionally',
                'consequently', 'meanwhile', 'subsequently', 'finally', 'in conclusion'
            ]
            
            transition_count = sum(1 for word in transition_words if word in content.lower())
            coherence_score = min(1.0, transition_count / len(sentences))
            
            return coherence_score
            
        except Exception as e:
            logger.error(f"Failed to analyze coherence: {str(e)}")
            return 0.5
    
    def _analyze_clarity(self, content: str) -> float:
        """Analyze content clarity."""
        try:
            # Check for clarity indicators
            clarity_indicators = {
                "short_sentences": sum(1 for s in content.split('.') if len(s.split()) <= 15),
                "active_voice": content.count(' is ') + content.count(' are ') + content.count(' was ') + content.count(' were '),
                "concrete_words": len([w for w in content.split() if len(w) > 6]),
                "avoid_jargon": len([w for w in content.split() if w.isupper() and len(w) > 2])
            }
            
            # Calculate clarity score
            total_sentences = len(content.split('.'))
            if total_sentences == 0:
                return 0.5
            
            clarity_score = (
                clarity_indicators["short_sentences"] / total_sentences * 0.4 +
                (1 - min(1.0, clarity_indicators["active_voice"] / total_sentences)) * 0.3 +
                min(1.0, clarity_indicators["concrete_words"] / len(content.split())) * 0.2 +
                (1 - min(1.0, clarity_indicators["avoid_jargon"] / total_sentences)) * 0.1
            )
            
            return min(1.0, clarity_score)
            
        except Exception as e:
            logger.error(f"Failed to analyze clarity: {str(e)}")
            return 0.5
    
    def _calculate_vocabulary_diversity(self, content: str) -> float:
        """Calculate vocabulary diversity."""
        try:
            words = content.lower().split()
            if len(words) == 0:
                return 0.0
            
            unique_words = set(words)
            diversity = len(unique_words) / len(words)
            return min(1.0, diversity)
            
        except Exception as e:
            logger.error(f"Failed to calculate vocabulary diversity: {str(e)}")
            return 0.5
    
    def _calculate_information_density(self, content: str) -> float:
        """Calculate information density."""
        try:
            # Simple information density calculation
            words = content.split()
            if len(words) == 0:
                return 0.0
            
            # Count information-rich words (longer words, technical terms)
            info_words = [w for w in words if len(w) > 6 or w.isupper()]
            density = len(info_words) / len(words)
            
            return min(1.0, density)
            
        except Exception as e:
            logger.error(f"Failed to calculate information density: {str(e)}")
            return 0.5
    
    def _calculate_engagement_score(self, content: str) -> float:
        """Calculate engagement score."""
        try:
            # Check for engagement indicators
            engagement_indicators = {
                "questions": content.count('?'),
                "exclamations": content.count('!'),
                "quotes": content.count('"'),
                "examples": content.count('for example') + content.count('such as'),
                "call_to_action": content.count('click') + content.count('learn more') + content.count('discover')
            }
            
            total_indicators = sum(engagement_indicators.values())
            word_count = len(content.split())
            
            if word_count == 0:
                return 0.0
            
            engagement_score = min(1.0, total_indicators / (word_count / 100))
            return engagement_score
            
        except Exception as e:
            logger.error(f"Failed to calculate engagement score: {str(e)}")
            return 0.5
    
    def _get_optimization_techniques(self, optimization_level: OptimizationLevel) -> List[Dict[str, Any]]:
        """Get optimization techniques based on level."""
        techniques = {
            OptimizationLevel.BASIC: [
                {"name": "sentence_optimization", "weight": 0.3},
                {"name": "paragraph_structure", "weight": 0.2},
                {"name": "basic_clarity", "weight": 0.5}
            ],
            OptimizationLevel.ADVANCED: [
                {"name": "sentence_optimization", "weight": 0.2},
                {"name": "paragraph_structure", "weight": 0.2},
                {"name": "basic_clarity", "weight": 0.2},
                {"name": "coherence_improvement", "weight": 0.2},
                {"name": "engagement_enhancement", "weight": 0.2}
            ],
            OptimizationLevel.EXPERT: [
                {"name": "sentence_optimization", "weight": 0.15},
                {"name": "paragraph_structure", "weight": 0.15},
                {"name": "basic_clarity", "weight": 0.15},
                {"name": "coherence_improvement", "weight": 0.15},
                {"name": "engagement_enhancement", "weight": 0.15},
                {"name": "advanced_optimization", "weight": 0.15},
                {"name": "personalization", "weight": 0.1}
            ]
        }
        
        return techniques.get(optimization_level, techniques[OptimizationLevel.BASIC])
    
    async def _apply_optimization_technique(
        self, 
        content: str, 
        technique: Dict[str, Any], 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a specific optimization technique."""
        try:
            technique_name = technique["name"]
            
            if technique_name == "sentence_optimization":
                return await self._optimize_sentences(content, metrics)
            elif technique_name == "paragraph_structure":
                return await self._optimize_paragraph_structure(content, metrics)
            elif technique_name == "basic_clarity":
                return await self._optimize_clarity(content, metrics)
            elif technique_name == "coherence_improvement":
                return await self._improve_coherence(content, metrics)
            elif technique_name == "engagement_enhancement":
                return await self._enhance_engagement(content, metrics)
            elif technique_name == "advanced_optimization":
                return await self._advanced_optimization(content, metrics)
            elif technique_name == "personalization":
                return await self._personalize_content(content, metrics)
            else:
                return {"improved": False, "content": content}
                
        except Exception as e:
            logger.error(f"Failed to apply optimization technique {technique_name}: {str(e)}")
            return {"improved": False, "content": content}
    
    async def _optimize_sentences(self, content: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize sentence structure."""
        try:
            sentences = content.split('.')
            optimized_sentences = []
            
            for sentence in sentences:
                if sentence.strip():
                    # Simple sentence optimization
                    optimized_sentence = sentence.strip()
                    
                    # Remove redundant words
                    redundant_words = ['very', 'really', 'quite', 'rather', 'somewhat']
                    for word in redundant_words:
                        optimized_sentence = optimized_sentence.replace(f' {word} ', ' ')
                    
                    # Improve sentence structure
                    if len(optimized_sentence.split()) > 25:
                        # Split long sentences
                        words = optimized_sentence.split()
                        mid_point = len(words) // 2
                        optimized_sentences.append(' '.join(words[:mid_point]))
                        optimized_sentences.append(' '.join(words[mid_point:]))
                    else:
                        optimized_sentences.append(optimized_sentence)
            
            optimized_content = '. '.join(optimized_sentences)
            improved = optimized_content != content
            
            return {"improved": improved, "content": optimized_content}
            
        except Exception as e:
            logger.error(f"Failed to optimize sentences: {str(e)}")
            return {"improved": False, "content": content}
    
    async def _optimize_paragraph_structure(self, content: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize paragraph structure."""
        try:
            paragraphs = content.split('\n\n')
            optimized_paragraphs = []
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Ensure paragraphs have proper structure
                    sentences = paragraph.split('.')
                    if len(sentences) > 1:
                        # Add topic sentence if missing
                        if not any(sentence.strip().startswith(('The', 'This', 'In', 'For', 'When', 'Where', 'Why', 'How')) for sentence in sentences):
                            optimized_paragraph = f"This section covers {sentences[0].strip().lower()}." + ' '.join(sentences[1:])
                        else:
                            optimized_paragraph = paragraph
                    else:
                        optimized_paragraph = paragraph
                    
                    optimized_paragraphs.append(optimized_paragraph)
            
            optimized_content = '\n\n'.join(optimized_paragraphs)
            improved = optimized_content != content
            
            return {"improved": improved, "content": optimized_content}
            
        except Exception as e:
            logger.error(f"Failed to optimize paragraph structure: {str(e)}")
            return {"improved": False, "content": content}
    
    async def _optimize_clarity(self, content: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content clarity."""
        try:
            # Replace complex words with simpler alternatives
            replacements = {
                'utilize': 'use',
                'facilitate': 'help',
                'implement': 'put in place',
                'leverage': 'use',
                'optimize': 'improve',
                'enhance': 'improve',
                'demonstrate': 'show',
                'illustrate': 'show',
                'consequently': 'so',
                'furthermore': 'also',
                'moreover': 'also'
            }
            
            optimized_content = content
            for complex_word, simple_word in replacements.items():
                optimized_content = optimized_content.replace(complex_word, simple_word)
            
            improved = optimized_content != content
            return {"improved": improved, "content": optimized_content}
            
        except Exception as e:
            logger.error(f"Failed to optimize clarity: {str(e)}")
            return {"improved": False, "content": content}
    
    async def _improve_coherence(self, content: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Improve content coherence."""
        try:
            # Add transition words between paragraphs
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                optimized_paragraphs = [paragraphs[0]]
                
                transition_words = ['Additionally', 'Furthermore', 'Moreover', 'However', 'Therefore', 'Consequently']
                
                for i, paragraph in enumerate(paragraphs[1:], 1):
                    if paragraph.strip():
                        # Add transition word
                        transition = transition_words[i % len(transition_words)]
                        optimized_paragraph = f"{transition}, {paragraph.strip().lower()}"
                        optimized_paragraphs.append(optimized_paragraph)
                
                optimized_content = '\n\n'.join(optimized_paragraphs)
                improved = optimized_content != content
            else:
                optimized_content = content
                improved = False
            
            return {"improved": improved, "content": optimized_content}
            
        except Exception as e:
            logger.error(f"Failed to improve coherence: {str(e)}")
            return {"improved": False, "content": content}
    
    async def _enhance_engagement(self, content: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance content engagement."""
        try:
            # Add engaging elements
            optimized_content = content
            
            # Add questions to engage readers
            if '?' not in content and len(content.split()) > 100:
                sentences = content.split('.')
                if len(sentences) > 2:
                    # Add a question after the first paragraph
                    first_paragraph = sentences[0]
                    if len(first_paragraph.split()) > 10:
                        question = f"Have you ever wondered about {first_paragraph.split()[0].lower()}?"
                        optimized_content = f"{first_paragraph}. {question}. " + '. '.join(sentences[1:])
            
            # Add call-to-action if appropriate
            if 'conclusion' in content.lower() or 'summary' in content.lower():
                cta = "This information can help you make better decisions."
                optimized_content += f" {cta}"
            
            improved = optimized_content != content
            return {"improved": improved, "content": optimized_content}
            
        except Exception as e:
            logger.error(f"Failed to enhance engagement: {str(e)}")
            return {"improved": False, "content": content}
    
    async def _advanced_optimization(self, content: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply advanced optimization techniques."""
        try:
            # This would implement more sophisticated optimization
            # For now, apply multiple basic optimizations
            optimized_content = content
            
            # Apply multiple optimization passes
            for _ in range(2):
                result = await self._optimize_sentences(optimized_content, metrics)
                if result["improved"]:
                    optimized_content = result["content"]
                
                result = await self._optimize_clarity(optimized_content, metrics)
                if result["improved"]:
                    optimized_content = result["content"]
            
            improved = optimized_content != content
            return {"improved": improved, "content": optimized_content}
            
        except Exception as e:
            logger.error(f"Failed to apply advanced optimization: {str(e)}")
            return {"improved": False, "content": content}
    
    async def _personalize_content(self, content: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize content based on metrics."""
        try:
            # Simple personalization based on content analysis
            optimized_content = content
            
            # Adjust tone based on content type
            if metrics.get("word_count", 0) > 500:
                # For longer content, add more structure
                if '##' not in content:
                    optimized_content = f"## Overview\n\n{content}"
            
            improved = optimized_content != content
            return {"improved": improved, "content": optimized_content}
            
        except Exception as e:
            logger.error(f"Failed to personalize content: {str(e)}")
            return {"improved": False, "content": content}
    
    async def _calculate_optimization_score(
        self, 
        original_content: str, 
        optimized_content: str, 
        metrics: Dict[str, Any]
    ) -> float:
        """Calculate optimization score."""
        try:
            # Calculate improvement metrics
            original_metrics = await self._analyze_document_metrics({"content": original_content})
            optimized_metrics = await self._analyze_document_metrics({"content": optimized_content})
            
            # Calculate score based on improvements
            score_components = []
            
            # Readability improvement
            readability_improvement = optimized_metrics.get("readability_score", 0) - original_metrics.get("readability_score", 0)
            score_components.append(max(0, readability_improvement))
            
            # Structure improvement
            structure_improvement = optimized_metrics.get("structure_score", 0) - original_metrics.get("structure_score", 0)
            score_components.append(max(0, structure_improvement))
            
            # Clarity improvement
            clarity_improvement = optimized_metrics.get("clarity_score", 0) - original_metrics.get("clarity_score", 0)
            score_components.append(max(0, clarity_improvement))
            
            # Calculate overall score
            if score_components:
                overall_score = sum(score_components) / len(score_components)
            else:
                overall_score = 0.0
            
            return min(1.0, max(0.0, overall_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate optimization score: {str(e)}")
            return 0.0
    
    async def _store_optimization_history(self, document: Dict[str, Any], result: OptimizationResult):
        """Store optimization history."""
        try:
            history_entry = {
                "document_id": document.get("id", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
                "optimization_score": result.optimization_score,
                "improvements_applied": result.improvements_applied,
                "optimization_time": result.optimization_time,
                "original_metrics": result.metrics
            }
            
            self.optimization_history.append(history_entry)
            
            # Keep only recent history
            if len(self.optimization_history) > 1000:
                self.optimization_history = self.optimization_history[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to store optimization history: {str(e)}")
    
    async def _update_performance_metrics(self, result: OptimizationResult):
        """Update performance metrics."""
        try:
            self.performance_metrics["total_optimizations"] += 1
            
            if result.optimization_score > 0.1:  # Consider successful if score > 0.1
                self.performance_metrics["successful_optimizations"] += 1
            
            # Update average improvement
            total = self.performance_metrics["total_optimizations"]
            current_avg = self.performance_metrics["average_improvement"]
            new_avg = ((current_avg * (total - 1)) + result.optimization_score) / total
            self.performance_metrics["average_improvement"] = new_avg
            
            # Update optimization speed
            if result.optimization_time > 0:
                speed = 1.0 / result.optimization_time
                total = self.performance_metrics["total_optimizations"]
                current_speed = self.performance_metrics["optimization_speed"]
                new_speed = ((current_speed * (total - 1)) + speed) / total
                self.performance_metrics["optimization_speed"] = new_speed
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {str(e)}")
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics."""
        return self.performance_metrics.copy()
    
    async def get_optimization_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history[-limit:]
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.metrics_collector.cleanup()
            await self.optimization_engine.cleanup()
            await self.learning_system.cleanup()
            logger.info("Optimization Core cleaned up successfully")
        except Exception as e:
            logger.error(f"Failed to cleanup Optimization Core: {str(e)}")











