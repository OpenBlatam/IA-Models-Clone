"""
Prompt Optimizer
================

Advanced prompt optimization system for better LLM performance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationTechnique(str, Enum):
    """Prompt optimization techniques."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    SELF_CONSISTENCY = "self_consistency"
    REACT = "react"
    TREE_OF_THOUGHT = "tree_of_thought"
    PROMPT_CHAINING = "prompt_chaining"

@dataclass
class OptimizedPrompt:
    """Optimized prompt result."""
    original_prompt: str
    optimized_prompt: str
    techniques_applied: List[str]
    optimization_score: float
    metadata: Dict[str, Any]
    timestamp: datetime

class PromptOptimizer:
    """
    Advanced prompt optimization system.
    
    Features:
    - Multiple optimization techniques
    - A/B testing
    - Performance tracking
    - Template system
    - Context awareness
    - Quality assessment
    """
    
    def __init__(self):
        self.optimization_templates = {}
        self.performance_history = []
        self.technique_effectiveness = {}
        
    async def initialize(self):
        """Initialize prompt optimizer."""
        logger.info("Initializing Prompt Optimizer...")
        
        try:
            # Load optimization templates
            await self._load_optimization_templates()
            
            # Initialize technique effectiveness tracking
            self.technique_effectiveness = {
                technique.value: {'success_rate': 0.5, 'avg_improvement': 0.0, 'usage_count': 0}
                for technique in OptimizationTechnique
            }
            
            logger.info("Prompt Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prompt Optimizer: {str(e)}")
            raise
    
    async def _load_optimization_templates(self):
        """Load optimization templates."""
        try:
            self.optimization_templates = {
                OptimizationTechnique.CHAIN_OF_THOUGHT: {
                    'template': "Let's think step by step.\n\n{original_prompt}\n\nPlease provide your reasoning step by step.",
                    'description': "Encourages step-by-step reasoning"
                },
                OptimizationTechnique.FEW_SHOT: {
                    'template': "Here are some examples:\n\n{examples}\n\nNow, {original_prompt}",
                    'description': "Provides examples for better understanding"
                },
                OptimizationTechnique.SELF_CONSISTENCY: {
                    'template': "{original_prompt}\n\nPlease think about this from multiple perspectives and ensure your answer is internally consistent.",
                    'description': "Encourages self-consistency checking"
                },
                OptimizationTechnique.REACT: {
                    'template': "You are a helpful assistant. You can think, act, and observe.\n\nThought: I need to {original_prompt}\nAction: [action to take]\nObservation: [result of action]\nThought: [reflection on observation]\nAction: [next action]\n...",
                    'description': "ReAct (Reasoning + Acting) framework"
                },
                OptimizationTechnique.TREE_OF_THOUGHT: {
                    'template': "Let's explore this problem systematically:\n\n{original_prompt}\n\nConsider multiple approaches and evaluate each one before providing your final answer.",
                    'description': "Encourages systematic exploration"
                },
                OptimizationTechnique.PROMPT_CHAINING: {
                    'template': "First, let's break this down:\n\nStep 1: {step1_prompt}\nStep 2: {step2_prompt}\nStep 3: {step3_prompt}\n\nNow, {original_prompt}",
                    'description': "Breaks complex tasks into steps"
                }
            }
            
            logger.info("Optimization templates loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load optimization templates: {str(e)}")
    
    async def optimize_prompt(
        self,
        original_prompt: str,
        techniques: Optional[List[OptimizationTechnique]] = None,
        context: Optional[Dict[str, Any]] = None,
        examples: Optional[List[str]] = None
    ) -> OptimizedPrompt:
        """
        Optimize a prompt using specified techniques.
        
        Args:
            original_prompt: Original prompt to optimize
            techniques: List of optimization techniques to apply
            context: Additional context for optimization
            examples: Examples for few-shot learning
            
        Returns:
            Optimized prompt result
        """
        try:
            if not techniques:
                # Auto-select techniques based on prompt characteristics
                techniques = await self._auto_select_techniques(original_prompt, context)
            
            optimized_prompt = original_prompt
            applied_techniques = []
            optimization_score = 0.0
            
            # Apply techniques in sequence
            for technique in techniques:
                try:
                    optimized_prompt = await self._apply_technique(
                        optimized_prompt,
                        technique,
                        context,
                        examples
                    )
                    applied_techniques.append(technique.value)
                    
                    # Update optimization score
                    technique_effectiveness = self.technique_effectiveness.get(technique.value, {})
                    optimization_score += technique_effectiveness.get('avg_improvement', 0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to apply technique {technique}: {str(e)}")
                    continue
            
            # Normalize optimization score
            optimization_score = min(1.0, optimization_score)
            
            # Create result
            result = OptimizedPrompt(
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                techniques_applied=applied_techniques,
                optimization_score=optimization_score,
                metadata={
                    'context': context or {},
                    'examples_count': len(examples) if examples else 0,
                    'techniques_count': len(applied_techniques)
                },
                timestamp=datetime.utcnow()
            )
            
            # Store performance data
            self.performance_history.append(result)
            
            logger.info(f"Optimized prompt with {len(applied_techniques)} techniques")
            return result
            
        except Exception as e:
            logger.error(f"Failed to optimize prompt: {str(e)}")
            # Return original prompt if optimization fails
            return OptimizedPrompt(
                original_prompt=original_prompt,
                optimized_prompt=original_prompt,
                techniques_applied=[],
                optimization_score=0.0,
                metadata={'error': str(e)},
                timestamp=datetime.utcnow()
            )
    
    async def _auto_select_techniques(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationTechnique]:
        """Auto-select optimization techniques based on prompt characteristics."""
        try:
            techniques = []
            
            # Analyze prompt characteristics
            prompt_lower = prompt.lower()
            word_count = len(prompt.split())
            
            # Chain of thought for complex reasoning
            if any(keyword in prompt_lower for keyword in ['analyze', 'explain', 'reason', 'why', 'how', 'calculate']):
                techniques.append(OptimizationTechnique.CHAIN_OF_THOUGHT)
            
            # Few-shot for examples
            if any(keyword in prompt_lower for keyword in ['example', 'similar', 'like', 'such as']):
                techniques.append(OptimizationTechnique.FEW_SHOT)
            
            # Self-consistency for verification tasks
            if any(keyword in prompt_lower for keyword in ['verify', 'check', 'validate', 'confirm']):
                techniques.append(OptimizationTechnique.SELF_CONSISTENCY)
            
            # ReAct for action-oriented tasks
            if any(keyword in prompt_lower for keyword in ['do', 'perform', 'execute', 'action', 'task']):
                techniques.append(OptimizationTechnique.REACT)
            
            # Tree of thought for exploration
            if any(keyword in prompt_lower for keyword in ['explore', 'consider', 'evaluate', 'compare']):
                techniques.append(OptimizationTechnique.TREE_OF_THOUGHT)
            
            # Prompt chaining for complex tasks
            if word_count > 100 or any(keyword in prompt_lower for keyword in ['multiple', 'several', 'various']):
                techniques.append(OptimizationTechnique.PROMPT_CHAINING)
            
            # Default to chain of thought if no specific techniques identified
            if not techniques:
                techniques.append(OptimizationTechnique.CHAIN_OF_THOUGHT)
            
            return techniques
            
        except Exception as e:
            logger.error(f"Failed to auto-select techniques: {str(e)}")
            return [OptimizationTechnique.CHAIN_OF_THOUGHT]
    
    async def _apply_technique(
        self,
        prompt: str,
        technique: OptimizationTechnique,
        context: Optional[Dict[str, Any]] = None,
        examples: Optional[List[str]] = None
    ) -> str:
        """Apply a specific optimization technique."""
        try:
            template_info = self.optimization_templates.get(technique)
            if not template_info:
                return prompt
            
            template = template_info['template']
            
            if technique == OptimizationTechnique.CHAIN_OF_THOUGHT:
                return template.format(original_prompt=prompt)
            
            elif technique == OptimizationTechnique.FEW_SHOT:
                if examples:
                    examples_text = '\n\n'.join(f"Example {i+1}: {ex}" for i, ex in enumerate(examples[:3]))
                    return template.format(original_prompt=prompt, examples=examples_text)
                else:
                    return template.format(original_prompt=prompt, examples="[No examples provided]")
            
            elif technique == OptimizationTechnique.SELF_CONSISTENCY:
                return template.format(original_prompt=prompt)
            
            elif technique == OptimizationTechnique.REACT:
                return template.format(original_prompt=prompt)
            
            elif technique == OptimizationTechnique.TREE_OF_THOUGHT:
                return template.format(original_prompt=prompt)
            
            elif technique == OptimizationTechnique.PROMPT_CHAINING:
                # Break down the prompt into steps
                steps = await self._break_down_prompt(prompt)
                return template.format(
                    original_prompt=prompt,
                    step1_prompt=steps[0] if len(steps) > 0 else "Identify the main topic",
                    step2_prompt=steps[1] if len(steps) > 1 else "Analyze the key components",
                    step3_prompt=steps[2] if len(steps) > 2 else "Provide a comprehensive answer"
                )
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to apply technique {technique}: {str(e)}")
            return prompt
    
    async def _break_down_prompt(self, prompt: str) -> List[str]:
        """Break down a complex prompt into steps."""
        try:
            # Simple step extraction based on keywords
            steps = []
            
            # Look for numbered steps
            step_pattern = r'(?:step\s*\d+|first|second|third|next|then|finally)'
            matches = re.finditer(step_pattern, prompt, re.IGNORECASE)
            
            for match in matches:
                start = match.start()
                # Find the end of this step
                next_match = None
                for next_match in re.finditer(step_pattern, prompt[start+1:], re.IGNORECASE):
                    break
                
                if next_match:
                    end = start + next_match.start() + 1
                else:
                    end = len(prompt)
                
                step_text = prompt[start:end].strip()
                if step_text:
                    steps.append(step_text)
            
            # If no steps found, create generic steps
            if not steps:
                steps = [
                    "Understand the requirements",
                    "Analyze the key points",
                    "Provide a comprehensive response"
                ]
            
            return steps[:3]  # Limit to 3 steps
            
        except Exception as e:
            logger.error(f"Failed to break down prompt: {str(e)}")
            return ["Process the request", "Provide the answer"]
    
    async def evaluate_optimization(
        self,
        original_prompt: str,
        optimized_prompt: str,
        response_quality: float
    ):
        """Evaluate the effectiveness of prompt optimization."""
        try:
            # Calculate improvement
            improvement = response_quality - 0.5  # Assuming baseline of 0.5
            
            # Update technique effectiveness
            for result in self.performance_history[-10:]:  # Recent results
                for technique in result.techniques_applied:
                    if technique in self.technique_effectiveness:
                        current = self.technique_effectiveness[technique]
                        usage_count = current['usage_count'] + 1
                        
                        # Update success rate
                        success_rate = current['success_rate']
                        new_success_rate = (success_rate * (usage_count - 1) + (1 if improvement > 0 else 0)) / usage_count
                        
                        # Update average improvement
                        avg_improvement = current['avg_improvement']
                        new_avg_improvement = (avg_improvement * (usage_count - 1) + improvement) / usage_count
                        
                        self.technique_effectiveness[technique] = {
                            'success_rate': new_success_rate,
                            'avg_improvement': new_avg_improvement,
                            'usage_count': usage_count
                        }
            
            logger.info(f"Updated technique effectiveness based on quality score: {response_quality}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate optimization: {str(e)}")
    
    async def get_optimization_suggestions(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get suggestions for prompt optimization."""
        try:
            suggestions = []
            
            # Analyze prompt characteristics
            word_count = len(prompt.split())
            sentence_count = len(re.split(r'[.!?]+', prompt))
            
            # Length suggestions
            if word_count < 10:
                suggestions.append("Consider adding more context and specific instructions")
            elif word_count > 500:
                suggestions.append("Consider breaking down the prompt into smaller, more focused parts")
            
            # Structure suggestions
            if sentence_count < 2:
                suggestions.append("Consider adding more detailed instructions or examples")
            
            # Technique suggestions based on effectiveness
            for technique, effectiveness in self.technique_effectiveness.items():
                if effectiveness['success_rate'] > 0.7 and effectiveness['avg_improvement'] > 0.1:
                    technique_name = technique.replace('_', ' ').title()
                    suggestions.append(f"Consider using {technique_name} technique for better results")
            
            # Context-specific suggestions
            if context:
                if context.get('task_type') == 'creative':
                    suggestions.append("Add examples of creative outputs to guide the model")
                elif context.get('task_type') == 'analytical':
                    suggestions.append("Use step-by-step reasoning to improve analytical accuracy")
                elif context.get('task_type') == 'factual':
                    suggestions.append("Add verification steps to ensure factual accuracy")
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Failed to get optimization suggestions: {str(e)}")
            return []
    
    async def get_technique_effectiveness(self) -> Dict[str, Any]:
        """Get effectiveness statistics for optimization techniques."""
        try:
            return {
                'techniques': self.technique_effectiveness,
                'total_optimizations': len(self.performance_history),
                'average_optimization_score': sum(r.optimization_score for r in self.performance_history) / len(self.performance_history) if self.performance_history else 0.0,
                'most_used_technique': max(self.technique_effectiveness.items(), key=lambda x: x[1]['usage_count'])[0] if self.technique_effectiveness else None,
                'most_effective_technique': max(self.technique_effectiveness.items(), key=lambda x: x[1]['avg_improvement'])[0] if self.technique_effectiveness else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get technique effectiveness: {str(e)}")
            return {}
    
    async def get_optimization_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get optimization history."""
        try:
            recent_results = self.performance_history[-limit:]
            
            return [
                {
                    'original_prompt': result.original_prompt[:100] + '...' if len(result.original_prompt) > 100 else result.original_prompt,
                    'optimized_prompt': result.optimized_prompt[:100] + '...' if len(result.optimized_prompt) > 100 else result.optimized_prompt,
                    'techniques_applied': result.techniques_applied,
                    'optimization_score': result.optimization_score,
                    'timestamp': result.timestamp.isoformat(),
                    'metadata': result.metadata
                }
                for result in recent_results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get optimization history: {str(e)}")
            return []
    
    async def cleanup(self):
        """Cleanup prompt optimizer."""
        try:
            logger.info("Prompt Optimizer cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Prompt Optimizer: {str(e)}")











