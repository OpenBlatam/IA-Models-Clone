"""
AI Optimization and Prompt Engineering System
============================================

This module provides advanced AI optimization features including prompt engineering,
context optimization, and intelligent prompt chaining for better content generation.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PromptOptimization:
    """Prompt optimization result"""
    original_prompt: str
    optimized_prompt: str
    optimization_type: str
    improvement_score: float
    tokens_saved: int
    expected_quality_improvement: float
    metadata: Dict[str, Any]

@dataclass
class ContextWindow:
    """Context window management"""
    max_tokens: int
    used_tokens: int
    available_tokens: int
    compression_ratio: float
    priority_sections: List[str]

class PromptOptimizer:
    """Advanced prompt optimization system"""
    
    def __init__(self):
        self.optimization_patterns = {
            "remove_redundancy": self._remove_redundant_phrases,
            "improve_clarity": self._improve_clarity,
            "add_structure": self._add_structural_elements,
            "optimize_length": self._optimize_prompt_length,
            "enhance_specificity": self._enhance_specificity
        }
        
        self.quality_indicators = {
            "specific_instructions": ["write", "create", "generate", "develop"],
            "context_providers": ["about", "regarding", "concerning", "on the topic of"],
            "quality_markers": ["detailed", "comprehensive", "professional", "engaging"],
            "structure_indicators": ["introduction", "body", "conclusion", "sections"]
        }
    
    async def optimize_prompt(
        self,
        prompt: str,
        target_length: Optional[int] = None,
        optimization_goals: List[str] = None
    ) -> PromptOptimization:
        """
        Optimize a prompt for better AI generation
        
        Args:
            prompt: Original prompt to optimize
            target_length: Target length in tokens
            optimization_goals: List of optimization goals
            
        Returns:
            PromptOptimization: Optimization result
        """
        try:
            original_tokens = self._estimate_tokens(prompt)
            optimization_goals = optimization_goals or ["improve_clarity", "add_structure"]
            
            optimized_prompt = prompt
            total_improvement = 0.0
            tokens_saved = 0
            
            # Apply optimization techniques
            for goal in optimization_goals:
                if goal in self.optimization_patterns:
                    optimized_prompt, improvement = await self.optimization_patterns[goal](
                        optimized_prompt, target_length
                    )
                    total_improvement += improvement
            
            # Calculate final metrics
            final_tokens = self._estimate_tokens(optimized_prompt)
            tokens_saved = original_tokens - final_tokens
            
            # Calculate quality improvement
            quality_improvement = self._calculate_quality_improvement(
                prompt, optimized_prompt
            )
            
            return PromptOptimization(
                original_prompt=prompt,
                optimized_prompt=optimized_prompt,
                optimization_type="multi_goal",
                improvement_score=total_improvement,
                tokens_saved=tokens_saved,
                expected_quality_improvement=quality_improvement,
                metadata={
                    "original_tokens": original_tokens,
                    "final_tokens": final_tokens,
                    "optimization_goals": optimization_goals,
                    "optimization_timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            return PromptOptimization(
                original_prompt=prompt,
                optimized_prompt=prompt,
                optimization_type="error",
                improvement_score=0.0,
                tokens_saved=0,
                expected_quality_improvement=0.0,
                metadata={"error": str(e)}
            )
    
    async def _remove_redundant_phrases(self, prompt: str, target_length: Optional[int]) -> Tuple[str, float]:
        """Remove redundant phrases from prompt"""
        try:
            # Common redundant phrases
            redundant_patterns = [
                r'\b(please|kindly|would you|could you)\b',
                r'\b(very|really|quite|rather)\s+(good|bad|important|useful)',
                r'\b(in order to|so as to|for the purpose of)\b',
                r'\b(due to the fact that|because of the fact that)\b',
                r'\b(at this point in time|at the present time)\b'
            ]
            
            optimized = prompt
            improvements = 0
            
            for pattern in redundant_patterns:
                matches = len(re.findall(pattern, optimized, re.IGNORECASE))
                optimized = re.sub(pattern, '', optimized, flags=re.IGNORECASE)
                improvements += matches * 0.1
            
            # Clean up extra whitespace
            optimized = re.sub(r'\s+', ' ', optimized).strip()
            
            return optimized, improvements
            
        except Exception as e:
            logger.error(f"Error removing redundant phrases: {str(e)}")
            return prompt, 0.0
    
    async def _improve_clarity(self, prompt: str, target_length: Optional[int]) -> Tuple[str, float]:
        """Improve prompt clarity and specificity"""
        try:
            improvements = 0.0
            optimized = prompt
            
            # Add specific instructions if missing
            if not any(indicator in optimized.lower() for indicator in self.quality_indicators["specific_instructions"]):
                optimized = f"Write a {optimized}"
                improvements += 0.2
            
            # Add structure indicators if missing
            if not any(indicator in optimized.lower() for indicator in self.quality_indicators["structure_indicators"]):
                optimized += " Structure the content with clear sections and logical flow."
                improvements += 0.15
            
            # Add quality markers if missing
            if not any(marker in optimized.lower() for marker in self.quality_indicators["quality_markers"]):
                optimized = optimized.replace("content", "comprehensive and engaging content")
                improvements += 0.1
            
            return optimized, improvements
            
        except Exception as e:
            logger.error(f"Error improving clarity: {str(e)}")
            return prompt, 0.0
    
    async def _add_structural_elements(self, prompt: str, target_length: Optional[int]) -> Tuple[str, float]:
        """Add structural elements to prompt"""
        try:
            improvements = 0.0
            optimized = prompt
            
            # Add length guidance if not present
            if "words" not in optimized.lower() and "length" not in optimized.lower():
                optimized += " Aim for approximately 800-1200 words."
                improvements += 0.1
            
            # Add tone guidance if not present
            if "tone" not in optimized.lower() and "style" not in optimized.lower():
                optimized += " Use a professional yet engaging tone."
                improvements += 0.1
            
            # Add audience guidance if not present
            if "audience" not in optimized.lower() and "readers" not in optimized.lower():
                optimized += " Write for a general audience with varying levels of expertise."
                improvements += 0.1
            
            return optimized, improvements
            
        except Exception as e:
            logger.error(f"Error adding structural elements: {str(e)}")
            return prompt, 0.0
    
    async def _optimize_prompt_length(self, prompt: str, target_length: Optional[int]) -> Tuple[str, float]:
        """Optimize prompt length for target token count"""
        try:
            if not target_length:
                return prompt, 0.0
            
            current_tokens = self._estimate_tokens(prompt)
            if current_tokens <= target_length:
                return prompt, 0.0
            
            # Calculate compression needed
            compression_ratio = target_length / current_tokens
            improvements = 0.0
            
            if compression_ratio < 0.8:  # Need significant compression
                # Remove less important words
                optimized = self._compress_prompt(prompt, compression_ratio)
                improvements = 0.2
            else:
                # Minor optimizations
                optimized = self._minor_optimize(prompt)
                improvements = 0.1
            
            return optimized, improvements
            
        except Exception as e:
            logger.error(f"Error optimizing prompt length: {str(e)}")
            return prompt, 0.0
    
    async def _enhance_specificity(self, prompt: str, target_length: Optional[int]) -> Tuple[str, float]:
        """Enhance prompt specificity and detail"""
        try:
            improvements = 0.0
            optimized = prompt
            
            # Add specific examples if generic
            if "example" not in optimized.lower() and "instance" not in optimized.lower():
                optimized += " Include specific examples and case studies where relevant."
                improvements += 0.15
            
            # Add data requirements if not present
            if "data" not in optimized.lower() and "statistics" not in optimized.lower():
                optimized += " Support key points with relevant data and statistics."
                improvements += 0.1
            
            # Add actionable elements
            if "action" not in optimized.lower() and "step" not in optimized.lower():
                optimized += " Include actionable insights and practical applications."
                improvements += 0.1
            
            return optimized, improvements
            
        except Exception as e:
            logger.error(f"Error enhancing specificity: {str(e)}")
            return prompt, 0.0
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _calculate_quality_improvement(self, original: str, optimized: str) -> float:
        """Calculate expected quality improvement"""
        try:
            score = 0.0
            
            # Check for specific instructions
            original_instructions = sum(1 for indicator in self.quality_indicators["specific_instructions"] 
                                     if indicator in original.lower())
            optimized_instructions = sum(1 for indicator in self.quality_indicators["specific_instructions"] 
                                       if indicator in optimized.lower())
            
            if optimized_instructions > original_instructions:
                score += 0.2
            
            # Check for structure indicators
            original_structure = sum(1 for indicator in self.quality_indicators["structure_indicators"] 
                                   if indicator in original.lower())
            optimized_structure = sum(1 for indicator in self.quality_indicators["structure_indicators"] 
                                    if indicator in optimized.lower())
            
            if optimized_structure > original_structure:
                score += 0.15
            
            # Check for quality markers
            original_quality = sum(1 for marker in self.quality_indicators["quality_markers"] 
                                 if marker in original.lower())
            optimized_quality = sum(1 for marker in self.quality_indicators["quality_markers"] 
                                  if marker in optimized.lower())
            
            if optimized_quality > original_quality:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality improvement: {str(e)}")
            return 0.0
    
    def _compress_prompt(self, prompt: str, compression_ratio: float) -> str:
        """Compress prompt while maintaining key information"""
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', prompt)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Keep most important sentences
            target_sentences = max(1, int(len(sentences) * compression_ratio))
            
            # Simple importance scoring (longer sentences with key words are more important)
            scored_sentences = []
            for sentence in sentences:
                score = len(sentence)  # Length as base score
                # Add points for important words
                for word_list in self.quality_indicators.values():
                    score += sum(1 for word in word_list if word in sentence.lower()) * 10
                scored_sentences.append((score, sentence))
            
            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            selected_sentences = [sentence for _, sentence in scored_sentences[:target_sentences]]
            
            return '. '.join(selected_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Error compressing prompt: {str(e)}")
            return prompt
    
    def _minor_optimize(self, prompt: str) -> str:
        """Apply minor optimizations to prompt"""
        try:
            optimized = prompt
            
            # Remove extra spaces
            optimized = re.sub(r'\s+', ' ', optimized)
            
            # Remove redundant words
            optimized = re.sub(r'\b(the the|a a|an an)\b', lambda m: m.group(1).split()[0], optimized)
            
            # Simplify complex phrases
            replacements = {
                'in order to': 'to',
                'due to the fact that': 'because',
                'at this point in time': 'now',
                'for the purpose of': 'to'
            }
            
            for old, new in replacements.items():
                optimized = optimized.replace(old, new)
            
            return optimized.strip()
            
        except Exception as e:
            logger.error(f"Error in minor optimization: {str(e)}")
            return prompt

class ContextOptimizer:
    """Context window optimization system"""
    
    def __init__(self, max_context_tokens: int = 8000):
        self.max_context_tokens = max_context_tokens
        self.compression_strategies = {
            "summarize_old_content": self._summarize_old_content,
            "extract_key_points": self._extract_key_points,
            "compress_metadata": self._compress_metadata,
            "prioritize_recent": self._prioritize_recent_content
        }
    
    async def optimize_context(
        self,
        content_history: List[Dict[str, Any]],
        current_prompt: str,
        target_tokens: Optional[int] = None
    ) -> ContextWindow:
        """
        Optimize context window for AI generation
        
        Args:
            content_history: List of previous content items
            current_prompt: Current prompt to optimize for
            target_tokens: Target token count
            
        Returns:
            ContextWindow: Optimized context window
        """
        try:
            target_tokens = target_tokens or self.max_context_tokens
            
            # Calculate current usage
            current_tokens = self._estimate_tokens(current_prompt)
            history_tokens = sum(self._estimate_tokens(item.get('content', '')) 
                               for item in content_history)
            
            total_tokens = current_tokens + history_tokens
            available_tokens = target_tokens - current_tokens
            
            if total_tokens <= target_tokens:
                return ContextWindow(
                    max_tokens=target_tokens,
                    used_tokens=total_tokens,
                    available_tokens=target_tokens - total_tokens,
                    compression_ratio=1.0,
                    priority_sections=["full_history"]
                )
            
            # Optimize content history
            optimized_history = await self._optimize_content_history(
                content_history, available_tokens
            )
            
            # Calculate compression ratio
            optimized_tokens = sum(self._estimate_tokens(item.get('content', '')) 
                                 for item in optimized_history)
            compression_ratio = optimized_tokens / history_tokens if history_tokens > 0 else 1.0
            
            return ContextWindow(
                max_tokens=target_tokens,
                used_tokens=current_tokens + optimized_tokens,
                available_tokens=target_tokens - (current_tokens + optimized_tokens),
                compression_ratio=compression_ratio,
                priority_sections=["recent_content", "key_points", "summaries"]
            )
            
        except Exception as e:
            logger.error(f"Error optimizing context: {str(e)}")
            return ContextWindow(
                max_tokens=target_tokens or self.max_context_tokens,
                used_tokens=0,
                available_tokens=target_tokens or self.max_context_tokens,
                compression_ratio=1.0,
                priority_sections=[]
            )
    
    async def _optimize_content_history(
        self,
        content_history: List[Dict[str, Any]],
        available_tokens: int
    ) -> List[Dict[str, Any]]:
        """Optimize content history to fit available tokens"""
        try:
            if not content_history:
                return []
            
            # Sort by recency (most recent first)
            sorted_history = sorted(content_history, 
                                  key=lambda x: x.get('timestamp', ''), 
                                  reverse=True)
            
            optimized_history = []
            used_tokens = 0
            
            for item in sorted_history:
                content = item.get('content', '')
                content_tokens = self._estimate_tokens(content)
                
                if used_tokens + content_tokens <= available_tokens:
                    # Can fit full content
                    optimized_history.append(item)
                    used_tokens += content_tokens
                else:
                    # Need to compress
                    remaining_tokens = available_tokens - used_tokens
                    if remaining_tokens > 100:  # Minimum viable content
                        compressed_item = await self._compress_content_item(item, remaining_tokens)
                        optimized_history.append(compressed_item)
                        used_tokens += self._estimate_tokens(compressed_item.get('content', ''))
                    break
            
            return optimized_history
            
        except Exception as e:
            logger.error(f"Error optimizing content history: {str(e)}")
            return content_history[:3]  # Fallback to first 3 items
    
    async def _compress_content_item(
        self,
        item: Dict[str, Any],
        target_tokens: int
    ) -> Dict[str, Any]:
        """Compress a single content item"""
        try:
            content = item.get('content', '')
            current_tokens = self._estimate_tokens(content)
            
            if current_tokens <= target_tokens:
                return item
            
            # Extract key sentences
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Score sentences by importance
            scored_sentences = []
            for sentence in sentences:
                score = len(sentence)  # Base score
                # Add points for important words
                important_words = ['important', 'key', 'main', 'primary', 'essential', 'critical']
                score += sum(1 for word in important_words if word in sentence.lower()) * 5
                scored_sentences.append((score, sentence))
            
            # Select top sentences that fit
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            selected_sentences = []
            used_tokens = 0
            
            for _, sentence in scored_sentences:
                sentence_tokens = self._estimate_tokens(sentence)
                if used_tokens + sentence_tokens <= target_tokens:
                    selected_sentences.append(sentence)
                    used_tokens += sentence_tokens
                else:
                    break
            
            # Create compressed item
            compressed_content = '. '.join(selected_sentences) + '.'
            compressed_item = item.copy()
            compressed_item['content'] = compressed_content
            compressed_item['compressed'] = True
            compressed_item['original_length'] = len(content)
            compressed_item['compressed_length'] = len(compressed_content)
            
            return compressed_item
            
        except Exception as e:
            logger.error(f"Error compressing content item: {str(e)}")
            return item
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(text) // 4
    
    async def _summarize_old_content(self, content: str) -> str:
        """Summarize old content"""
        # Simple summarization - take first and last sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 2:
            return content
        
        summary = sentences[0] + '. ' + sentences[-1] + '.'
        return summary
    
    async def _extract_key_points(self, content: str) -> str:
        """Extract key points from content"""
        # Look for sentences with key indicators
        sentences = re.split(r'[.!?]+', content)
        key_sentences = []
        
        key_indicators = ['important', 'key', 'main', 'primary', 'essential', 'critical', 'note']
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in key_indicators):
                key_sentences.append(sentence.strip())
        
        return '. '.join(key_sentences[:3]) + '.' if key_sentences else content[:200] + '...'
    
    async def _compress_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compress metadata to save tokens"""
        # Keep only essential metadata
        essential_keys = ['id', 'title', 'timestamp', 'quality_score']
        return {k: v for k, v in metadata.items() if k in essential_keys}
    
    async def _prioritize_recent_content(self, content_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize recent content in history"""
        return sorted(content_history, key=lambda x: x.get('timestamp', ''), reverse=True)

# Global instances
prompt_optimizer = PromptOptimizer()
context_optimizer = ContextOptimizer()

# Example usage
if __name__ == "__main__":
    async def test_optimization():
        print("🧪 Testing AI Optimization")
        print("=" * 40)
        
        # Test prompt optimization
        sample_prompt = """
        Please write a very comprehensive and detailed article about artificial intelligence 
        and its applications in the field of marketing. The article should be quite long 
        and really informative for the readers. Kindly include examples and make it very engaging.
        """
        
        optimization = await prompt_optimizer.optimize_prompt(
            sample_prompt,
            target_length=100,
            optimization_goals=["remove_redundancy", "improve_clarity", "add_structure"]
        )
        
        print(f"Original prompt: {optimization.original_prompt[:100]}...")
        print(f"Optimized prompt: {optimization.optimized_prompt[:100]}...")
        print(f"Improvement score: {optimization.improvement_score:.2f}")
        print(f"Tokens saved: {optimization.tokens_saved}")
        print(f"Quality improvement: {optimization.expected_quality_improvement:.2f}")
        
        # Test context optimization
        content_history = [
            {"content": "This is a long article about AI...", "timestamp": "2024-01-01"},
            {"content": "Another detailed piece about machine learning...", "timestamp": "2024-01-02"},
            {"content": "A comprehensive guide to deep learning...", "timestamp": "2024-01-03"}
        ]
        
        context_window = await context_optimizer.optimize_context(
            content_history,
            "Write about neural networks",
            target_tokens=500
        )
        
        print(f"\nContext optimization:")
        print(f"Max tokens: {context_window.max_tokens}")
        print(f"Used tokens: {context_window.used_tokens}")
        print(f"Available tokens: {context_window.available_tokens}")
        print(f"Compression ratio: {context_window.compression_ratio:.2f}")
    
    asyncio.run(test_optimization())


