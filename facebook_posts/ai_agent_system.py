import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import asyncio
import json
import random
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import our existing components
from .custom_nn_modules import (
    FacebookContentAnalysisTransformer, MultiModalFacebookAnalyzer,
    TemporalEngagementPredictor, AdaptiveContentOptimizer
)
from .performance_optimization_engine import HighPerformanceOptimizationEngine, PerformanceConfig


class AgentType(Enum):
    """Types of AI agents"""
    CONTENT_ANALYZER = "content_analyzer"
    ENGAGEMENT_PREDICTOR = "engagement_predictor"
    OPTIMIZATION_SPECIALIST = "optimization_specialist"
    PERFORMANCE_MONITOR = "performance_monitor"
    STRATEGY_COORDINATOR = "strategy_coordinator"


class AgentState(Enum):
    """Agent states"""
    IDLE = "idle"
    ACTIVE = "active"
    LEARNING = "learning"
    COLLABORATING = "collaborating"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for AI agents"""
    agent_type: AgentType
    learning_rate: float = 0.001
    memory_size: int = 1000
    decision_threshold: float = 0.7
    collaboration_enabled: bool = True
    autonomous_mode: bool = True
    max_actions_per_cycle: int = 10
    confidence_threshold: float = 0.8


@dataclass
class AgentMemory:
    """Memory system for agents"""
    experiences: deque = field(default_factory=lambda: deque(maxlen=1000))
    decisions: deque = field(default_factory=lambda: deque(maxlen=500))
    outcomes: deque = field(default_factory=lambda: deque(maxlen=500))
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    
    def add_experience(self, experience: Dict[str, Any]):
        """Add new experience to memory"""
        experience['timestamp'] = datetime.now()
        self.experiences.append(experience)
    
    def add_decision(self, decision: Dict[str, Any]):
        """Add decision to memory"""
        decision['timestamp'] = datetime.now()
        self.decisions.append(decision)
    
    def add_outcome(self, outcome: Dict[str, Any]):
        """Add outcome to memory"""
        outcome['timestamp'] = datetime.now()
        self.outcomes.append(outcome)
    
    def get_recent_experiences(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent experiences"""
        return list(self.experiences)[-count:]
    
    def get_successful_decisions(self) -> List[Dict[str, Any]]:
        """Get successful decisions"""
        return [d for d in self.decisions if d.get('success', False)]
    
    def update_knowledge(self, key: str, value: Any):
        """Update knowledge base"""
        self.knowledge_base[key] = value


class BaseAgent:
    """Base class for all AI agents"""
    
    def __init__(self, config: AgentConfig, agent_id: str):
        self.config = config
        self.agent_id = agent_id
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.performance_metrics = defaultdict(float)
        
        # Initialize logging
        self.logger = logging.getLogger(f"Agent_{agent_id}")
        self.logger.setLevel(logging.INFO)
        
        # Agent capabilities
        self.capabilities = self._initialize_capabilities()
        
        # Learning parameters
        self.learning_rate = config.learning_rate
        self.confidence = 0.5
        self.experience_count = 0
        
        self.logger.info(f"Agent {agent_id} initialized with type {config.agent_type}")
    
    def _initialize_capabilities(self) -> Dict[str, Any]:
        """Initialize agent capabilities"""
        return {
            'can_analyze': True,
            'can_predict': True,
            'can_optimize': True,
            'can_learn': True,
            'can_collaborate': self.config.collaboration_enabled
        }
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return results"""
        self.state = AgentState.ACTIVE
        
        try:
            # Analyze task
            analysis = await self._analyze_task(task)
            
            # Make decision
            decision = await self._make_decision(analysis)
            
            # Execute action
            result = await self._execute_action(decision)
            
            # Learn from outcome
            await self._learn_from_outcome(task, decision, result)
            
            # Update performance
            self._update_performance(result)
            
            return result
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Error processing task: {e}")
            return {'error': str(e), 'agent_id': self.agent_id}
        
        finally:
            self.state = AgentState.IDLE
    
    async def _analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze incoming task"""
        analysis = {
            'task_type': task.get('type', 'unknown'),
            'complexity': self._assess_complexity(task),
            'priority': self._assess_priority(task),
            'confidence': self.confidence,
            'timestamp': datetime.now()
        }
        
        self.memory.add_experience({
            'type': 'task_analysis',
            'task': task,
            'analysis': analysis
        })
        
        return analysis
    
    async def _make_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision based on analysis"""
        if analysis['confidence'] < self.config.confidence_threshold:
            # Request collaboration or more information
            decision = {
                'action': 'request_collaboration',
                'reason': 'Low confidence in analysis',
                'confidence': analysis['confidence']
            }
        else:
            # Make autonomous decision
            decision = {
                'action': 'execute_optimization',
                'strategy': self._select_strategy(analysis),
                'confidence': analysis['confidence']
            }
        
        self.memory.add_decision(decision)
        return decision
    
    async def _execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decided action"""
        if decision['action'] == 'request_collaboration':
            return await self._request_collaboration(decision)
        elif decision['action'] == 'execute_optimization':
            return await self._execute_optimization(decision)
        else:
            return {'error': 'Unknown action', 'decision': decision}
    
    async def _learn_from_outcome(self, task: Dict[str, Any], decision: Dict[str, Any], 
                                 result: Dict[str, Any]):
        """Learn from the outcome of actions"""
        outcome = {
            'task': task,
            'decision': decision,
            'result': result,
            'success': 'error' not in result,
            'performance_score': self._calculate_performance_score(result)
        }
        
        self.memory.add_outcome(outcome)
        
        # Update confidence based on success
        if outcome['success']:
            self.confidence = min(1.0, self.confidence + self.learning_rate)
        else:
            self.confidence = max(0.1, self.confidence - self.learning_rate)
        
        self.experience_count += 1
        
        # Update knowledge base
        self._update_knowledge_base(outcome)
    
    def _assess_complexity(self, task: Dict[str, Any]) -> float:
        """Assess task complexity"""
        # Simple complexity assessment based on task properties
        complexity_factors = [
            len(task.get('content', '')),
            len(task.get('requirements', [])),
            task.get('urgency', 1)
        ]
        return min(1.0, sum(complexity_factors) / 10.0)
    
    def _assess_priority(self, task: Dict[str, Any]) -> float:
        """Assess task priority"""
        priority_factors = [
            task.get('urgency', 0.5),
            task.get('importance', 0.5),
            task.get('business_value', 0.5)
        ]
        return sum(priority_factors) / len(priority_factors)
    
    def _select_strategy(self, analysis: Dict[str, Any]) -> str:
        """Select optimization strategy"""
        if analysis['complexity'] > 0.8:
            return 'advanced_optimization'
        elif analysis['complexity'] > 0.5:
            return 'standard_optimization'
        else:
            return 'basic_optimization'
    
    async def _request_collaboration(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Request collaboration with other agents"""
        self.state = AgentState.COLLABORATING
        
        # Simulate collaboration request
        collaboration_result = {
            'action': 'collaboration_requested',
            'agents_contacted': ['agent_2', 'agent_3'],
            'waiting_for_response': True,
            'timestamp': datetime.now()
        }
        
        return collaboration_result
    
    async def _execute_optimization(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization action"""
        strategy = decision.get('strategy', 'basic_optimization')
        
        # Simulate optimization execution
        optimization_result = {
            'action': 'optimization_executed',
            'strategy': strategy,
            'confidence': decision['confidence'],
            'estimated_improvement': random.uniform(0.1, 0.3),
            'execution_time': random.uniform(0.5, 2.0),
            'timestamp': datetime.now()
        }
        
        return optimization_result
    
    def _calculate_performance_score(self, result: Dict[str, Any]) -> float:
        """Calculate performance score from result"""
        if 'error' in result:
            return 0.0
        
        # Calculate score based on result properties
        score_factors = [
            result.get('confidence', 0.5),
            result.get('estimated_improvement', 0.0),
            1.0 - result.get('execution_time', 1.0) / 5.0  # Normalize execution time
        ]
        
        return sum(score_factors) / len(score_factors)
    
    def _update_knowledge_base(self, outcome: Dict[str, Any]):
        """Update knowledge base with new information"""
        key = f"{outcome['task'].get('type', 'unknown')}_{outcome['success']}"
        current_knowledge = self.memory.knowledge_base.get(key, {'count': 0, 'avg_score': 0.0})
        
        current_knowledge['count'] += 1
        current_knowledge['avg_score'] = (
            (current_knowledge['avg_score'] * (current_knowledge['count'] - 1) + 
             outcome['performance_score']) / current_knowledge['count']
        )
        
        self.memory.update_knowledge(key, current_knowledge)
    
    def _update_performance(self, result: Dict[str, Any]):
        """Update performance metrics"""
        if 'error' not in result:
            self.performance_metrics['success_rate'] = (
                (self.performance_metrics['success_rate'] * self.experience_count + 1) / 
                (self.experience_count + 1)
            )
            self.performance_metrics['avg_confidence'] = (
                (self.performance_metrics['avg_confidence'] * self.experience_count + 
                 result.get('confidence', 0.5)) / (self.experience_count + 1)
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'type': self.config.agent_type.value,
            'state': self.state.value,
            'confidence': self.confidence,
            'experience_count': self.experience_count,
            'performance_metrics': dict(self.performance_metrics),
            'memory_size': len(self.memory.experiences)
        }


class ContentAnalyzerAgent(BaseAgent):
    """Specialized agent for content analysis"""
    
    def __init__(self, config: AgentConfig, agent_id: str, model: FacebookContentAnalysisTransformer):
        super().__init__(config, agent_id)
        self.model = model
        self.analysis_specializations = ['text_analysis', 'sentiment_analysis', 'engagement_prediction']
    
    async def _analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced task analysis for content"""
        base_analysis = await super()._analyze_task(task)
        
        # Add content-specific analysis
        content = task.get('content', '')
        content_analysis = {
            'text_length': len(content),
            'sentiment_score': self._analyze_sentiment(content),
            'engagement_potential': self._predict_engagement(content),
            'content_quality': self._assess_content_quality(content)
        }
        
        base_analysis.update(content_analysis)
        return base_analysis
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze text sentiment"""
        # Simple sentiment analysis (in production, use proper NLP models)
        positive_words = ['good', 'great', 'amazing', 'excellent', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in word in negative_words if word in text_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.5
        return positive_count / (positive_count + negative_count)
    
    def _predict_engagement(self, text: str) -> float:
        """Predict engagement potential"""
        # Simple engagement prediction based on text features
        features = [
            len(text) / 1000,  # Length factor
            text.count('!') / max(len(text), 1),  # Exclamation factor
            text.count('?') / max(len(text), 1),  # Question factor
            len([word for word in text.split() if len(word) > 5]) / max(len(text.split()), 1)  # Long words
        ]
        
        return min(1.0, sum(features) / len(features))
    
    def _assess_content_quality(self, text: str) -> float:
        """Assess content quality"""
        # Simple quality assessment
        quality_factors = [
            len(text) > 50,  # Minimum length
            len(text.split()) > 10,  # Minimum word count
            text.count('.') > 0,  # Has sentences
            not text.isupper(),  # Not all caps
        ]
        
        return sum(quality_factors) / len(quality_factors)


class OptimizationSpecialistAgent(BaseAgent):
    """Specialized agent for content optimization"""
    
    def __init__(self, config: AgentConfig, agent_id: str, model: AdaptiveContentOptimizer):
        super().__init__(config, agent_id)
        self.model = model
        self.optimization_strategies = {
            'basic_optimization': self._basic_optimization,
            'standard_optimization': self._standard_optimization,
            'advanced_optimization': self._advanced_optimization
        }
    
    async def _execute_optimization(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization with specialized strategies"""
        strategy_name = decision.get('strategy', 'basic_optimization')
        strategy_func = self.optimization_strategies.get(strategy_name, self._basic_optimization)
        
        return await strategy_func(decision)
    
    async def _basic_optimization(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Basic optimization strategy"""
        return {
            'action': 'basic_optimization_executed',
            'strategy': 'basic_optimization',
            'suggestions': [
                'Add relevant hashtags',
                'Include call-to-action',
                'Optimize posting time'
            ],
            'estimated_improvement': 0.1,
            'execution_time': 0.5,
            'timestamp': datetime.now()
        }
    
    async def _standard_optimization(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Standard optimization strategy"""
        return {
            'action': 'standard_optimization_executed',
            'strategy': 'standard_optimization',
            'suggestions': [
                'Analyze audience demographics',
                'A/B test different headlines',
                'Optimize image placement',
                'Include trending topics'
            ],
            'estimated_improvement': 0.2,
            'execution_time': 1.0,
            'timestamp': datetime.now()
        }
    
    async def _advanced_optimization(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced optimization strategy"""
        return {
            'action': 'advanced_optimization_executed',
            'strategy': 'advanced_optimization',
            'suggestions': [
                'Multi-modal content analysis',
                'Predictive engagement modeling',
                'Dynamic content adaptation',
                'Cross-platform optimization',
                'Real-time performance monitoring'
            ],
            'estimated_improvement': 0.3,
            'execution_time': 2.0,
            'timestamp': datetime.now()
        }


class AgentCoordinator:
    """Coordinates multiple AI agents"""
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.coordination_rules = self._initialize_coordination_rules()
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.successful_collaborations = 0
        self.agent_performance = defaultdict(list)
        
        self.logger = logging.getLogger("AgentCoordinator")
        self.logger.setLevel(logging.INFO)
    
    def _initialize_coordination_rules(self) -> Dict[str, Any]:
        """Initialize coordination rules"""
        return {
            'max_agents_per_task': 3,
            'collaboration_threshold': 0.7,
            'timeout_seconds': 30,
            'retry_attempts': 3
        }
    
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit task for processing"""
        task_id = f"task_{int(time.time() * 1000)}"
        task['task_id'] = task_id
        task['submitted_at'] = datetime.now()
        
        await self.task_queue.put(task)
        self.logger.info(f"Task {task_id} submitted for processing")
        
        return task_id
    
    async def process_tasks(self):
        """Main task processing loop"""
        while True:
            try:
                task = await self.task_queue.get()
                result = await self._process_task_with_agents(task)
                await self.result_queue.put(result)
                
                self.total_tasks_processed += 1
                self.logger.info(f"Task {task['task_id']} processed successfully")
                
            except Exception as e:
                self.logger.error(f"Error processing task: {e}")
    
    async def _process_task_with_agents(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task with multiple agents"""
        # Select appropriate agents
        selected_agents = self._select_agents_for_task(task)
        
        # Process with primary agent
        primary_agent = selected_agents[0]
        primary_result = await primary_agent.process_task(task)
        
        # Check if collaboration is needed
        if primary_result.get('action') == 'collaboration_requested':
            collaboration_result = await self._handle_collaboration(task, selected_agents[1:])
            primary_result['collaboration'] = collaboration_result
            self.successful_collaborations += 1
        
        # Aggregate results
        final_result = {
            'task_id': task['task_id'],
            'primary_agent': primary_agent.agent_id,
            'result': primary_result,
            'agents_involved': [agent.agent_id for agent in selected_agents],
            'processing_time': (datetime.now() - task['submitted_at']).total_seconds(),
            'timestamp': datetime.now()
        }
        
        # Update agent performance
        self._update_agent_performance(primary_agent.agent_id, final_result)
        
        return final_result
    
    def _select_agents_for_task(self, task: Dict[str, Any]) -> List[BaseAgent]:
        """Select appropriate agents for task"""
        task_type = task.get('type', 'general')
        
        # Simple agent selection logic
        if task_type == 'content_analysis':
            return [agent for agent in self.agents.values() 
                   if agent.config.agent_type == AgentType.CONTENT_ANALYZER]
        elif task_type == 'optimization':
            return [agent for agent in self.agents.values() 
                   if agent.config.agent_type == AgentType.OPTIMIZATION_SPECIALIST]
        else:
            # Return all agents for general tasks
            return list(self.agents.values())
    
    async def _handle_collaboration(self, task: Dict[str, Any], 
                                  collaborating_agents: List[BaseAgent]) -> Dict[str, Any]:
        """Handle collaboration between agents"""
        collaboration_results = []
        
        # Process with collaborating agents
        for agent in collaborating_agents:
            try:
                result = await agent.process_task(task)
                collaboration_results.append({
                    'agent_id': agent.agent_id,
                    'result': result
                })
            except Exception as e:
                self.logger.error(f"Collaboration error with agent {agent.agent_id}: {e}")
        
        return {
            'collaborating_agents': len(collaborating_agents),
            'results': collaboration_results,
            'consensus_reached': len(collaboration_results) > 0
        }
    
    def _update_agent_performance(self, agent_id: str, result: Dict[str, Any]):
        """Update agent performance tracking"""
        performance_score = 1.0 if 'error' not in result['result'] else 0.0
        self.agent_performance[agent_id].append({
            'timestamp': datetime.now(),
            'performance_score': performance_score,
            'processing_time': result['processing_time']
        })
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            'total_tasks_processed': self.total_tasks_processed,
            'successful_collaborations': self.successful_collaborations,
            'active_agents': len([a for a in self.agents.values() if a.state != AgentState.ERROR]),
            'agent_performance': {
                agent_id: {
                    'avg_performance': np.mean([p['performance_score'] for p in performances]),
                    'total_tasks': len(performances),
                    'avg_processing_time': np.mean([p['processing_time'] for p in performances])
                }
                for agent_id, performances in self.agent_performance.items()
            }
        }
    
    def get_agent_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        return {
            agent_id: agent.get_status()
            for agent_id, agent in self.agents.items()
        }


class AIAgentSystem:
    """Complete AI agent system for Facebook optimization"""
    
    def __init__(self, performance_engine: HighPerformanceOptimizationEngine):
        self.performance_engine = performance_engine
        self.coordinator = None
        self.agents = {}
        
        # Initialize logging
        self.logger = logging.getLogger("AIAgentSystem")
        self.logger.setLevel(logging.INFO)
        
        # Initialize agents and coordinator
        self._initialize_agents()
        self._initialize_coordinator()
        
        self.logger.info("AI Agent System initialized")
    
    def _initialize_agents(self):
        """Initialize AI agents"""
        # Create models for agents
        content_model = FacebookContentAnalysisTransformer()
        optimization_model = AdaptiveContentOptimizer()
        
        # Create agents
        agent_configs = [
            (AgentType.CONTENT_ANALYZER, "content_analyzer_1", content_model),
            (AgentType.OPTIMIZATION_SPECIALIST, "optimization_specialist_1", optimization_model),
            (AgentType.ENGAGEMENT_PREDICTOR, "engagement_predictor_1", None),
            (AgentType.PERFORMANCE_MONITOR, "performance_monitor_1", None),
            (AgentType.STRATEGY_COORDINATOR, "strategy_coordinator_1", None)
        ]
        
        for agent_type, agent_id, model in agent_configs:
            config = AgentConfig(
                agent_type=agent_type,
                learning_rate=0.001,
                collaboration_enabled=True,
                autonomous_mode=True
            )
            
            if agent_type == AgentType.CONTENT_ANALYZER:
                agent = ContentAnalyzerAgent(config, agent_id, model)
            elif agent_type == AgentType.OPTIMIZATION_SPECIALIST:
                agent = OptimizationSpecialistAgent(config, agent_id, model)
            else:
                agent = BaseAgent(config, agent_id)
            
            self.agents[agent_id] = agent
    
    def _initialize_coordinator(self):
        """Initialize agent coordinator"""
        self.coordinator = AgentCoordinator(list(self.agents.values()))
    
    async def optimize_content_with_agents(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content using AI agents"""
        # Submit task to coordinator
        task_id = await self.coordinator.submit_task({
            'type': 'content_optimization',
            'content': content,
            'priority': content.get('priority', 0.5),
            'urgency': content.get('urgency', 0.5)
        })
        
        # Wait for result
        result = await self.coordinator.result_queue.get()
        
        return result
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            'coordinator_stats': self.coordinator.get_coordinator_stats(),
            'agent_statuses': self.coordinator.get_agent_statuses(),
            'performance_engine_stats': self.performance_engine.get_performance_stats()
        }
    
    async def start_autonomous_mode(self):
        """Start autonomous operation mode"""
        self.logger.info("Starting autonomous mode")
        
        # Start task processing
        asyncio.create_task(self.coordinator.process_tasks())
        
        # Start autonomous task generation
        asyncio.create_task(self._autonomous_task_generation())
    
    async def _autonomous_task_generation(self):
        """Generate tasks autonomously"""
        while True:
            try:
                # Generate autonomous tasks based on system state
                task = await self._generate_autonomous_task()
                if task:
                    await self.coordinator.submit_task(task)
                
                await asyncio.sleep(60)  # Generate task every minute
                
            except Exception as e:
                self.logger.error(f"Error in autonomous task generation: {e}")
                await asyncio.sleep(60)
    
    async def _generate_autonomous_task(self) -> Optional[Dict[str, Any]]:
        """Generate autonomous task based on system state"""
        # Simple autonomous task generation
        # In production, this would be more sophisticated
        if random.random() < 0.1:  # 10% chance to generate task
            return {
                'type': 'performance_monitoring',
                'priority': 0.3,
                'urgency': 0.2,
                'autonomous': True
            }
        return None


def create_ai_agent_system(performance_config: Optional[PerformanceConfig] = None) -> AIAgentSystem:
    """Create and configure AI agent system"""
    if performance_config is None:
        performance_config = PerformanceConfig()
    
    performance_engine = HighPerformanceOptimizationEngine(performance_config)
    return AIAgentSystem(performance_engine)


async def main():
    """Demonstrate AI agent system"""
    
    # Create AI agent system
    system = create_ai_agent_system()
    
    # Start autonomous mode
    await system.start_autonomous_mode()
    
    # Create sample content
    sample_content = {
        'id': 'content_123',
        'text': 'Check out our amazing new product! It will revolutionize your life!',
        'type': 'post',
        'priority': 0.8,
        'urgency': 0.6
    }
    
    # Optimize content with agents
    print("Optimizing content with AI agents...")
    result = await system.optimize_content_with_agents(sample_content)
    
    print("Optimization Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Get system status
    status = await system.get_system_status()
    print("\nSystem Status:")
    print(json.dumps(status, indent=2, default=str))
    
    return system


if __name__ == "__main__":
    asyncio.run(main())


