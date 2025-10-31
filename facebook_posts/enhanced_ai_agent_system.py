import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import threading
import asyncio
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import hashlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our existing components
from custom_nn_modules import (
    FacebookContentAnalysisTransformer, MultiModalFacebookAnalyzer,
    TemporalEngagementPredictor, AdaptiveContentOptimizer, FacebookDiffusionUNet
)
from enhanced_performance_engine import EnhancedPerformanceOptimizationEngine, EnhancedPerformanceConfig


@dataclass
class EnhancedAgentConfig:
    """Enhanced configuration for AI agents"""
    # Agent parameters
    num_agents: int = 5
    agent_types: List[str] = field(default_factory=lambda: [
        'content_optimizer', 'engagement_analyzer', 'trend_predictor', 
        'audience_targeter', 'performance_monitor'
    ])
    
    # Learning parameters
    learning_rate: float = 0.001
    exploration_rate: float = 0.1
    memory_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 100
    
    # Decision making
    confidence_threshold: float = 0.7
    consensus_threshold: float = 0.6
    max_decision_time: float = 5.0
    
    # Communication
    enable_agent_communication: bool = True
    communication_frequency: int = 50
    knowledge_sharing: bool = True
    
    # Autonomy
    enable_autonomous_mode: bool = True
    autonomous_decision_threshold: float = 0.8
    human_oversight_threshold: float = 0.3
    
    # Specialization
    enable_specialization: bool = True
    specialization_learning_rate: float = 0.002
    cross_training_enabled: bool = True


class AgentMemory:
    """Advanced memory system for AI agents"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.short_term = deque(maxlen=1000)  # Recent experiences
        self.long_term = deque(maxlen=max_size)  # Important experiences
        self.knowledge_base = {}  # Learned patterns and rules
        
        # Memory consolidation
        self.consolidation_threshold = 100
        self.importance_scores = defaultdict(float)
    
    def add_experience(self, experience: Dict[str, Any], importance: float = 1.0):
        """Add new experience to memory"""
        experience['timestamp'] = time.time()
        experience['importance'] = importance
        
        # Add to short-term memory
        self.short_term.append(experience)
        
        # Update importance score
        self.importance_scores[hash(str(experience))] += importance
        
        # Consolidate if needed
        if len(self.short_term) >= self.consolidation_threshold:
            self._consolidate_memory()
    
    def _consolidate_memory(self):
        """Consolidate short-term memories to long-term"""
        important_experiences = []
        
        for exp in self.short_term:
            exp_hash = hash(str(exp))
            if self.importance_scores[exp_hash] > 0.5:  # Threshold for importance
                important_experiences.append(exp)
        
        # Move important experiences to long-term memory
        for exp in important_experiences:
            if len(self.long_term) < self.max_size:
                self.long_term.append(exp)
        
        # Clear short-term memory
        self.short_term.clear()
        
        # Update knowledge base
        self._update_knowledge_base()
    
    def _update_knowledge_base(self):
        """Update knowledge base with patterns from experiences"""
        if not self.long_term:
            return
        
        # Analyze patterns in long-term memory
        patterns = self._extract_patterns()
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_type not in self.knowledge_base:
                self.knowledge_base[pattern_type] = []
            
            self.knowledge_base[pattern_type].append({
                'pattern': pattern_data,
                'timestamp': time.time(),
                'confidence': pattern_data.get('confidence', 0.5)
            })
    
    def _extract_patterns(self) -> Dict[str, Any]:
        """Extract patterns from experiences"""
        patterns = {}
        
        # Content engagement patterns
        engagement_data = [exp for exp in self.long_term if 'engagement_score' in exp]
        if engagement_data:
            scores = [exp['engagement_score'] for exp in engagement_data]
            patterns['engagement'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'trend': 'increasing' if len(scores) > 1 and scores[-1] > scores[0] else 'stable',
                'confidence': min(0.9, len(scores) / 100)
            }
        
        # Content type patterns
        content_types = defaultdict(list)
        for exp in self.long_term:
            if 'content_type' in exp and 'engagement_score' in exp:
                content_types[exp['content_type']].append(exp['engagement_score'])
        
        for content_type, scores in content_types.items():
            patterns[f'content_type_{content_type}'] = {
                'mean_engagement': np.mean(scores),
                'count': len(scores),
                'confidence': min(0.9, len(scores) / 50)
            }
        
        return patterns
    
    def get_relevant_experiences(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get experiences relevant to a query"""
        relevant_experiences = []
        
        for exp in self.long_term:
            relevance_score = self._calculate_relevance(exp, query)
            if relevance_score > 0.3:  # Relevance threshold
                relevant_experiences.append((exp, relevance_score))
        
        # Sort by relevance and return top results
        relevant_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in relevant_experiences[:limit]]
    
    def _calculate_relevance(self, experience: Dict[str, Any], query: Dict[str, Any]) -> float:
        """Calculate relevance score between experience and query"""
        relevance = 0.0
        
        # Content type relevance
        if 'content_type' in experience and 'content_type' in query:
            if experience['content_type'] == query['content_type']:
                relevance += 0.3
        
        # Engagement score relevance
        if 'engagement_score' in experience and 'target_engagement' in query:
            score_diff = abs(experience['engagement_score'] - query['target_engagement'])
            relevance += max(0, 0.2 - score_diff)
        
        # Temporal relevance
        if 'timestamp' in experience:
            age_hours = (time.time() - experience['timestamp']) / 3600
            if age_hours < 24:  # Recent experiences are more relevant
                relevance += 0.2
            elif age_hours < 168:  # Within a week
                relevance += 0.1
        
        return min(1.0, relevance)
    
    def get_knowledge(self, knowledge_type: str) -> List[Dict[str, Any]]:
        """Get knowledge of a specific type"""
        return self.knowledge_base.get(knowledge_type, [])


class AgentBrain:
    """Neural network brain for AI agents"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        self.batch_size = 32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)
    
    def learn(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor):
        """Learn from experience"""
        # Store experience
        self.experience_buffer.append((state, action, reward, next_state))
        
        # Learn if we have enough experiences
        if len(self.experience_buffer) >= self.batch_size:
            self._update_network()
    
    def _update_network(self):
        """Update network weights using experience replay"""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.experience_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        
        # Current Q-values
        current_q = self.network(states)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.network(next_states)
            target_q = rewards + 0.99 * torch.max(next_q, dim=1)[0]
        
        # Loss and update
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_action(self, state: torch.Tensor, exploration_rate: float = 0.1) -> torch.Tensor:
        """Get action with exploration"""
        if random.random() < exploration_rate:
            # Random action
            return torch.rand_like(state)
        else:
            # Greedy action
            with torch.no_grad():
                return self.network(state)


class AIAgent:
    """Enhanced AI agent with advanced capabilities"""
    
    def __init__(self, agent_id: str, agent_type: str, config: EnhancedAgentConfig):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        
        # Core components
        self.memory = AgentMemory(config.memory_size)
        self.brain = AgentBrain(input_size=64, hidden_size=128, output_size=32)
        
        # Agent state
        self.specialization = {}
        self.performance_history = deque(maxlen=1000)
        self.decision_history = deque(maxlen=1000)
        
        # Communication
        self.communication_buffer = deque(maxlen=100)
        self.knowledge_shared = set()
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_decisions = 0
        self.learning_rate = config.learning_rate
        
        # Initialize specialization
        if config.enable_specialization:
            self._initialize_specialization()
    
    def _initialize_specialization(self):
        """Initialize agent specialization"""
        if self.agent_type == 'content_optimizer':
            self.specialization = {
                'content_types': ['Post', 'Story', 'Reel'],
                'optimization_strategies': ['engagement', 'viral', 'conversion'],
                'expertise_level': 0.8
            }
        elif self.agent_type == 'engagement_analyzer':
            self.specialization = {
                'metrics': ['likes', 'comments', 'shares', 'reach'],
                'analysis_depth': 'deep',
                'expertise_level': 0.9
            }
        elif self.agent_type == 'trend_predictor':
            self.specialization = {
                'prediction_horizon': '7_days',
                'confidence_threshold': 0.7,
                'expertise_level': 0.85
            }
        elif self.agent_type == 'audience_targeter':
            self.specialization = {
                'demographics': ['age', 'location', 'interests'],
                'targeting_strategies': ['lookalike', 'custom', 'broad'],
                'expertise_level': 0.75
            }
        elif self.agent_type == 'performance_monitor':
            self.specialization = {
                'monitoring_frequency': 'real_time',
                'alert_thresholds': ['critical', 'warning', 'info'],
                'expertise_level': 0.9
            }
    
    def process_content(self, content: str, content_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process content and make decisions"""
        start_time = time.time()
        
        # Create state representation
        state = self._create_state_representation(content, content_type, context)
        
        # Get decision from brain
        decision = self._make_decision(state)
        
        # Apply specialization knowledge
        enhanced_decision = self._apply_specialization(decision, content_type)
        
        # Record decision
        decision_record = {
            'timestamp': time.time(),
            'content': content,
            'content_type': content_type,
            'decision': enhanced_decision,
            'processing_time': time.time() - start_time,
            'confidence': enhanced_decision.get('confidence', 0.5)
        }
        
        self.decision_history.append(decision_record)
        self.total_decisions += 1
        
        return enhanced_decision
    
    def _create_state_representation(self, content: str, content_type: str, context: Dict[str, Any] = None) -> torch.Tensor:
        """Create neural network input state"""
        # Simple feature extraction (in production, use proper NLP features)
        features = []
        
        # Content length
        features.append(min(1.0, len(content) / 1000))
        
        # Content type encoding
        content_type_encoding = {
            'Post': [1, 0, 0, 0, 0],
            'Story': [0, 1, 0, 0, 0],
            'Reel': [0, 0, 1, 0, 0],
            'Video': [0, 0, 0, 1, 0],
            'Image': [0, 0, 0, 0, 1]
        }
        features.extend(content_type_encoding.get(content_type, [0, 0, 0, 0, 0]))
        
        # Context features
        if context:
            features.append(context.get('time_of_day', 0.5))
            features.append(context.get('day_of_week', 0.5))
            features.append(context.get('audience_size', 0.5))
        else:
            features.extend([0.5, 0.5, 0.5])
        
        # Memory-based features
        relevant_experiences = self.memory.get_relevant_experiences({
            'content_type': content_type
        }, limit=5)
        
        if relevant_experiences:
            avg_engagement = np.mean([exp.get('engagement_score', 0.5) for exp in relevant_experiences])
            features.append(avg_engagement)
        else:
            features.append(0.5)
        
        # Pad to 64 features
        while len(features) < 64:
            features.append(0.0)
        
        return torch.tensor(features[:64], dtype=torch.float32)
    
    def _make_decision(self, state: torch.Tensor) -> Dict[str, Any]:
        """Make decision using the agent's brain"""
        with torch.no_grad():
            output = self.brain.forward(state)
        
        # Convert output to decision
        decision = {
            'optimization_score': float(output[0]),
            'engagement_prediction': float(output[1]),
            'viral_potential': float(output[2]),
            'audience_match': float(output[3]),
            'confidence': float(output[4]),
            'recommended_actions': self._generate_recommendations(output)
        }
        
        return decision
    
    def _generate_recommendations(self, output: torch.Tensor) -> List[str]:
        """Generate action recommendations based on output"""
        recommendations = []
        
        if output[0] > 0.7:  # High optimization score
            recommendations.append("Content is well-optimized")
        elif output[0] < 0.3:  # Low optimization score
            recommendations.append("Content needs significant optimization")
        
        if output[1] > 0.8:  # High engagement prediction
            recommendations.append("Expected high engagement")
        elif output[1] < 0.4:  # Low engagement prediction
            recommendations.append("Consider revising for better engagement")
        
        if output[2] > 0.7:  # High viral potential
            recommendations.append("Content has viral potential")
        
        if output[3] < 0.5:  # Low audience match
            recommendations.append("Consider adjusting target audience")
        
        return recommendations
    
    def _apply_specialization(self, decision: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """Apply agent specialization to enhance decision"""
        enhanced_decision = decision.copy()
        
        if self.agent_type == 'content_optimizer':
            enhanced_decision['specialized_advice'] = self._get_optimization_advice(content_type)
            enhanced_decision['confidence'] *= self.specialization.get('expertise_level', 0.8)
        
        elif self.agent_type == 'engagement_analyzer':
            enhanced_decision['engagement_breakdown'] = self._analyze_engagement_factors(content_type)
            enhanced_decision['confidence'] *= self.specialization.get('expertise_level', 0.9)
        
        elif self.agent_type == 'trend_predictor':
            enhanced_decision['trend_analysis'] = self._predict_trends(content_type)
            enhanced_decision['confidence'] *= self.specialization.get('expertise_level', 0.85)
        
        elif self.agent_type == 'audience_targeter':
            enhanced_decision['audience_insights'] = self._analyze_audience(content_type)
            enhanced_decision['confidence'] *= self.specialization.get('expertise_level', 0.75)
        
        elif self.agent_type == 'performance_monitor':
            enhanced_decision['performance_metrics'] = self._get_performance_metrics(content_type)
            enhanced_decision['confidence'] *= self.specialization.get('expertise_level', 0.9)
        
        return enhanced_decision
    
    def _get_optimization_advice(self, content_type: str) -> List[str]:
        """Get optimization advice based on content type"""
        advice = []
        
        if content_type == 'Post':
            advice.extend([
                "Use engaging headlines",
                "Include relevant hashtags",
                "Add call-to-action",
                "Use high-quality images"
            ])
        elif content_type == 'Story':
            advice.extend([
                "Keep it short and engaging",
                "Use interactive elements",
                "Include behind-the-scenes content",
                "Use engaging stickers"
            ])
        elif content_type == 'Reel':
            advice.extend([
                "Start with a hook",
                "Keep it under 30 seconds",
                "Use trending music",
                "Include trending hashtags"
            ])
        
        return advice
    
    def _analyze_engagement_factors(self, content_type: str) -> Dict[str, float]:
        """Analyze engagement factors"""
        factors = {
            'content_quality': 0.8,
            'timing': 0.7,
            'audience_relevance': 0.6,
            'trend_alignment': 0.5,
            'call_to_action': 0.4
        }
        
        # Adjust based on content type
        if content_type == 'Story':
            factors['timing'] *= 1.2  # Stories are more time-sensitive
        elif content_type == 'Reel':
            factors['trend_alignment'] *= 1.3  # Reels benefit from trends
        
        return factors
    
    def _predict_trends(self, content_type: str) -> Dict[str, Any]:
        """Predict content trends"""
        trends = {
            'trending_topics': ['AI', 'Sustainability', 'Wellness'],
            'content_formats': ['Short-form video', 'Interactive posts', 'User-generated content'],
            'engagement_patterns': ['Peak hours: 7-9 AM, 5-7 PM', 'Weekend boost: +15%'],
            'confidence': 0.75
        }
        
        return trends
    
    def _analyze_audience(self, content_type: str) -> Dict[str, Any]:
        """Analyze target audience"""
        audience = {
            'demographics': {
                'age_range': '25-45',
                'primary_location': 'Urban areas',
                'interests': ['Technology', 'Business', 'Lifestyle']
            },
            'behavior_patterns': {
                'active_hours': 'Evening',
                'content_preferences': 'Visual and interactive',
                'engagement_style': 'Comment and share'
            },
            'growth_potential': 0.8
        }
        
        return audience
    
    def _get_performance_metrics(self, content_type: str) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            'current_performance': {
                'engagement_rate': 0.045,
                'reach_growth': 0.12,
                'conversion_rate': 0.023
            },
            'benchmarks': {
                'industry_average': 0.038,
                'top_performers': 0.067,
                'our_position': 'Above average'
            },
            'trends': {
                'weekly_growth': 0.08,
                'monthly_growth': 0.25
            }
        }
        
        return metrics
    
    def learn_from_feedback(self, content_id: str, actual_performance: Dict[str, Any]):
        """Learn from actual performance feedback"""
        # Find the decision that was made for this content
        decision = None
        for dec in self.decision_history:
            if dec.get('content_id') == content_id:
                decision = dec
                break
        
        if not decision:
            return
        
        # Calculate reward based on performance
        predicted_engagement = decision['decision'].get('engagement_prediction', 0.5)
        actual_engagement = actual_performance.get('engagement_score', 0.5)
        
        # Reward is based on prediction accuracy
        reward = 1.0 - abs(predicted_engagement - actual_engagement)
        
        # Create state representation for learning
        state = self._create_state_representation(
            decision['content'], 
            decision['content_type']
        )
        
        # Create next state (could be based on actual performance)
        next_state = state.clone()
        
        # Learn
        self.brain.learn(state, torch.tensor([predicted_engagement]), reward, next_state)
        
        # Update performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'predicted': predicted_engagement,
            'actual': actual_engagement,
            'reward': reward,
            'accuracy': 1.0 - abs(predicted_engagement - actual_engagement)
        })
        
        # Update success rate
        if reward > 0.7:  # High accuracy threshold
            self.successful_decisions += 1
    
    def communicate_with_agents(self, other_agents: List['AIAgent']):
        """Communicate with other agents to share knowledge"""
        if not self.config.enable_agent_communication:
            return
        
        # Share relevant experiences
        for agent in other_agents:
            if agent.agent_id == self.agent_id:
                continue
            
            # Share high-value experiences
            high_value_experiences = [
                exp for exp in self.memory.long_term 
                if exp.get('importance', 0) > 0.8
            ]
            
            for exp in high_value_experiences[:3]:  # Share top 3
                exp_hash = hash(str(exp))
                if exp_hash not in self.knowledge_shared:
                    agent.receive_knowledge(exp)
                    self.knowledge_shared.add(exp_hash)
        
        # Share specialized knowledge
        if self.config.knowledge_sharing:
            specialized_knowledge = self._extract_specialized_knowledge()
            for agent in other_agents:
                agent.receive_specialized_knowledge(self.agent_type, specialized_knowledge)
    
    def receive_knowledge(self, experience: Dict[str, Any]):
        """Receive knowledge from another agent"""
        # Add to memory with lower importance (it's second-hand knowledge)
        self.memory.add_experience(experience, importance=0.5)
    
    def receive_specialized_knowledge(self, agent_type: str, knowledge: Dict[str, Any]):
        """Receive specialized knowledge from another agent"""
        # Store specialized knowledge
        if agent_type not in self.specialization:
            self.specialization[agent_type] = {}
        
        self.specialization[agent_type].update(knowledge)
    
    def _extract_specialized_knowledge(self) -> Dict[str, Any]:
        """Extract specialized knowledge to share"""
        knowledge = {}
        
        if self.agent_type == 'content_optimizer':
            knowledge['optimization_patterns'] = self.memory.get_knowledge('engagement')
            knowledge['content_type_insights'] = {
                k: v for k, v in self.memory.knowledge_base.items() 
                if k.startswith('content_type_')
            }
        
        elif self.agent_type == 'engagement_analyzer':
            knowledge['engagement_patterns'] = self.memory.get_knowledge('engagement')
            knowledge['performance_insights'] = self.performance_history[-10:] if self.performance_history else []
        
        return knowledge
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'total_decisions': self.total_decisions,
            'successful_decisions': self.successful_decisions,
            'success_rate': self.successful_decisions / max(self.total_decisions, 1),
            'specialization': self.specialization,
            'memory_stats': {
                'short_term_size': len(self.memory.short_term),
                'long_term_size': len(self.memory.long_term),
                'knowledge_base_size': len(self.memory.knowledge_base)
            },
            'performance_history': list(self.performance_history)[-10:],
            'learning_rate': self.learning_rate
        }


class EnhancedAIAgentSystem:
    """Enhanced AI agent system with multiple intelligent agents"""
    
    def __init__(self, config: EnhancedAgentConfig):
        self.config = config
        self.agents = {}
        self.performance_engine = None
        
        # Initialize agents
        self._initialize_agents()
        
        # Communication and coordination
        self.communication_thread = None
        if config.enable_agent_communication:
            self.communication_thread = threading.Thread(target=self._communication_loop, daemon=True)
            self.communication_thread.start()
        
        # System monitoring
        self.system_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'average_response_time': 0.0,
            'agent_performance': {}
        }
    
    def _initialize_agents(self):
        """Initialize all AI agents"""
        for i, agent_type in enumerate(self.config.agent_types):
            agent_id = f"{agent_type}_{i+1}"
            self.agents[agent_id] = AIAgent(agent_id, agent_type, self.config)
    
    def process_content_with_consensus(self, content: str, content_type: str, 
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process content using consensus from multiple agents"""
        start_time = time.time()
        self.system_stats['total_requests'] += 1
        
        try:
            # Get decisions from all agents
            agent_decisions = {}
            for agent_id, agent in self.agents.items():
                decision = agent.process_content(content, content_type, context)
                agent_decisions[agent_id] = decision
            
            # Calculate consensus decision
            consensus_decision = self._calculate_consensus(agent_decisions)
            
            # Apply autonomous decision if confidence is high enough
            if (self.config.enable_autonomous_mode and 
                consensus_decision['confidence'] > self.config.autonomous_decision_threshold):
                final_decision = consensus_decision
                decision_source = 'autonomous'
            else:
                # Require human oversight
                final_decision = consensus_decision
                final_decision['requires_human_oversight'] = True
                decision_source = 'human_oversight'
            
            # Record system performance
            response_time = time.time() - start_time
            self.system_stats['average_response_time'] = (
                (self.system_stats['average_response_time'] * (self.system_stats['total_requests'] - 1) + response_time) /
                self.system_stats['total_requests']
            )
            
            self.system_stats['successful_requests'] += 1
            
            # Return comprehensive result
            result = {
                'content': content,
                'content_type': content_type,
                'final_decision': final_decision,
                'agent_decisions': agent_decisions,
                'consensus_metrics': {
                    'confidence': consensus_decision['confidence'],
                    'agreement_level': self._calculate_agreement_level(agent_decisions),
                    'decision_source': decision_source
                },
                'system_performance': {
                    'response_time': response_time,
                    'agents_used': len(self.agents),
                    'consensus_threshold_met': consensus_decision['confidence'] > self.config.consensus_threshold
                }
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing content: {e}")
            self.system_stats['total_requests'] -= 1  # Don't count failed requests
            return {
                'error': str(e),
                'content': content,
                'content_type': content_type
            }
    
    def _calculate_consensus(self, agent_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus decision from multiple agents"""
        if not agent_decisions:
            return {}
        
        # Extract key metrics
        optimization_scores = []
        engagement_predictions = []
        viral_potentials = []
        confidences = []
        
        for decision in agent_decisions.values():
            if 'optimization_score' in decision:
                optimization_scores.append(decision['optimization_score'])
            if 'engagement_prediction' in decision:
                engagement_predictions.append(decision['engagement_prediction'])
            if 'viral_potential' in decision:
                viral_potentials.append(decision['viral_potential'])
            if 'confidence' in decision:
                confidences.append(decision['confidence'])
        
        # Calculate weighted averages based on agent confidence
        if confidences:
            weights = np.array(confidences) / sum(confidences)
            
            consensus = {
                'optimization_score': float(np.average(optimization_scores, weights=weights)) if optimization_scores else 0.5,
                'engagement_prediction': float(np.average(engagement_predictions, weights=weights)) if engagement_predictions else 0.5,
                'viral_potential': float(np.average(viral_potentials, weights=weights)) if viral_potentials else 0.5,
                'confidence': float(np.mean(confidences)),
                'agent_count': len(agent_decisions)
            }
        else:
            consensus = {
                'optimization_score': float(np.mean(optimization_scores)) if optimization_scores else 0.5,
                'engagement_prediction': float(np.mean(engagement_predictions)) if engagement_predictions else 0.5,
                'viral_potential': float(np.mean(viral_potentials)) if viral_potentials else 0.5,
                'confidence': 0.5,
                'agent_count': len(agent_decisions)
            }
        
        # Add recommendations
        all_recommendations = []
        for decision in agent_decisions.values():
            if 'recommended_actions' in decision:
                all_recommendations.extend(decision['recommended_actions'])
        
        # Remove duplicates and rank by frequency
        recommendation_counts = defaultdict(int)
        for rec in all_recommendations:
            recommendation_counts[rec] += 1
        
        ranked_recommendations = sorted(
            recommendation_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        consensus['top_recommendations'] = [rec for rec, count in ranked_recommendations[:5]]
        
        return consensus
    
    def _calculate_agreement_level(self, agent_decisions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate level of agreement between agents"""
        if len(agent_decisions) < 2:
            return 1.0
        
        # Calculate agreement on key metrics
        agreements = []
        
        for metric in ['optimization_score', 'engagement_prediction', 'viral_potential']:
            values = []
            for decision in agent_decisions.values():
                if metric in decision:
                    values.append(decision[metric])
            
            if len(values) > 1:
                # Calculate coefficient of variation (lower = more agreement)
                mean_val = np.mean(values)
                std_val = np.std(values)
                if mean_val > 0:
                    cv = std_val / mean_val
                    agreement = max(0, 1 - cv)  # Convert to agreement score
                    agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.5
    
    def _communication_loop(self):
        """Background loop for agent communication"""
        while True:
            try:
                # Agents communicate every N decisions
                stats = self._get_system_stats()
                if stats['total_requests'] % self.config.communication_frequency == 0:
                    agent_list = list(self.agents.values())
                    for agent in agent_list:
                        agent.communicate_with_agents(agent_list)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Error in communication loop: {e}")
                time.sleep(30)
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get basic system statistics"""
        return {
            'total_requests': getattr(self, '_total_requests', 0),
            'successful_requests': getattr(self, '_successful_requests', 0),
            'average_response_time': getattr(self, '_average_response_time', 0.0),
            'agents_active': len(self.agents),
            'system_status': 'active'
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = agent.get_agent_stats()
        
        stats = self._get_system_stats()
        return {
            'system_overview': stats,
            'agent_performance': agent_stats,
            'consensus_metrics': {
                'total_consensus_decisions': sum(1 for req in range(stats['total_requests']) 
                                               if req % 10 == 0),  # Approximate
                'average_confidence': 0.75,  # Placeholder
                'agreement_trends': 'increasing'  # Placeholder
            },
            'system_health': {
                'agents_active': len(self.agents),
                'communication_enabled': self.config.enable_agent_communication,
                'autonomous_mode': self.config.enable_autonomous_mode
            }
        }
    
    def update_agent_performance(self, content_id: str, actual_performance: Dict[str, Any]):
        """Update agent performance based on actual results"""
        for agent in self.agents.values():
            agent.learn_from_feedback(content_id, actual_performance)
    
    def cleanup(self):
        """Cleanup system resources"""
        if self.communication_thread:
            self.communication_thread.join(timeout=5)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = EnhancedAgentConfig(
        num_agents=5,
        enable_agent_communication=True,
        enable_autonomous_mode=True,
        enable_specialization=True
    )
    
    # Initialize system
    system = EnhancedAIAgentSystem(config)
    
    try:
        print("🚀 Testing Enhanced AI Agent System...")
        
        # Test content processing
        test_content = "Check out our amazing new AI-powered content optimization tool!"
        test_context = {
            'time_of_day': 0.7,  # Evening
            'day_of_week': 0.8,  # Weekend
            'audience_size': 0.6  # Medium audience
        }
        
        # Process content
        result = system.process_content_with_consensus(test_content, "Post", test_context)
        
        print(f"✅ Content processed successfully!")
        print(f"📊 Final Decision:")
        print(f"  - Optimization Score: {result['final_decision']['optimization_score']:.3f}")
        print(f"  - Engagement Prediction: {result['final_decision']['engagement_prediction']:.3f}")
        print(f"  - Viral Potential: {result['final_decision']['viral_potential']:.3f}")
        print(f"  - Confidence: {result['final_decision']['confidence']:.3f}")
        
        print(f"\n🤖 Agent Decisions: {len(result['agent_decisions'])} agents participated")
        print(f"📈 Consensus Metrics:")
        print(f"  - Agreement Level: {result['consensus_metrics']['agreement_level']:.3f}")
        print(f"  - Decision Source: {result['consensus_metrics']['decision_source']}")
        
        print(f"\n⚡ System Performance:")
        print(f"  - Response Time: {result['system_performance']['response_time']:.3f}s")
        print(f"  - Agents Used: {result['system_performance']['agents_used']}")
        
        # Get system stats
        stats = system.get_system_stats()
        print(f"\n📊 System Statistics:")
        print(f"  - Total Requests: {stats['system_overview']['total_requests']}")
        print(f"  - Success Rate: {stats['system_overview']['successful_requests'] / max(stats['system_overview']['total_requests'], 1):.3f}")
        print(f"  - Average Response Time: {stats['system_overview']['average_response_time']:.3f}s")
        
        # Test agent communication
        print(f"\n🤝 Testing agent communication...")
        time.sleep(2)  # Allow communication to happen
        
        # Simulate performance feedback
        print(f"\n📈 Simulating performance feedback...")
        system.update_agent_performance("test_content_123", {
            'engagement_score': 0.75,
            'viral_score': 0.6,
            'reach': 1000
        })
        
        print(f"✅ Performance feedback processed!")
        
    finally:
        # Cleanup
        system.cleanup()
        print("\n✨ Enhanced AI Agent System test completed!")
