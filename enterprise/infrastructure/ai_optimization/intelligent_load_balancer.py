from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import random
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
from typing import Any, List, Dict, Optional
"""
Intelligent Load Balancer with AI
=================================

AI-powered load balancing with neural networks and reinforcement learning:
- Neural network-based load prediction
- Reinforcement learning for optimal routing
- Real-time performance adaptation
- Predictive scaling decisions
- Smart health detection
"""


logger = logging.getLogger(__name__)

class LoadBalancingDecision(Enum):
    ROUTE_TO_INSTANCE = "route"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MARK_UNHEALTHY = "mark_unhealthy"
    REDISTRIBUTE = "redistribute"


@dataclass
class InstanceMetrics:
    """Metrics for a service instance."""
    instance_id: str
    cpu_usage: float
    memory_usage: float
    active_connections: int
    response_time: float
    error_rate: float
    throughput: float
    timestamp: datetime
    health_score: float = 1.0


@dataclass
class LoadPrediction:
    """Load prediction result."""
    instance_id: str
    predicted_load: float
    confidence: float
    recommendation: LoadBalancingDecision
    reasoning: str


class NeuralLoadBalancer:
    """Neural network-based load balancer."""
    
    def __init__(self, learning_rate: float = 0.001):
        
    """__init__ function."""
self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.training_data: List[Tuple[List[float], float]] = []
        self.prediction_history: deque = deque(maxlen=1000)
        self.is_trained = False
        
    def _extract_features(self, metrics: InstanceMetrics, global_metrics: Dict[str, Any]) -> List[float]:
        """Extract features for neural network."""
        features = [
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.active_connections / 1000.0,  # Normalized
            metrics.response_time / 1000.0,  # Normalized to seconds
            metrics.error_rate,
            metrics.throughput / 100.0,  # Normalized
            metrics.health_score,
            
            # Time-based features
            datetime.now().hour / 24.0,
            datetime.now().weekday() / 7.0,
            
            # Global context
            global_metrics.get('total_requests', 0) / 10000.0,
            global_metrics.get('avg_response_time', 0) / 1000.0,
            global_metrics.get('total_instances', 1) / 10.0,
        ]
        
        return features
    
    def record_performance(self, metrics: InstanceMetrics, actual_load: float, global_metrics: Dict[str, Any]):
        """Record performance data for training."""
        features = self._extract_features(metrics, global_metrics)
        self.training_data.append((features, actual_load))
        
        # Maintain limited training data
        if len(self.training_data) > 5000:
            self.training_data = self.training_data[-3000:]
        
        # Retrain periodically
        if len(self.training_data) % 200 == 0 and len(self.training_data) >= 100:
            asyncio.create_task(self._train_model())
    
    async def _train_model(self) -> Any:
        """Train the neural network model."""
        try:
            
            if len(self.training_data) < 50:
                return
            
            # Prepare data
            X = np.array([features for features, _ in self.training_data])
            y = np.array([load for _, load in self.training_data])
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train neural network
            self.model = MLPRegressor(
                hidden_layer_sizes=(50, 25),
                activation='relu',
                solver='adam',
                learning_rate_init=self.learning_rate,
                max_iter=500,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            self.is_trained = True
            
            logger.info(f"Neural load balancer trained - R² train: {train_score:.3f}, test: {test_score:.3f}")
            
        except ImportError:
            logger.warning("scikit-learn not available for neural load balancing")
        except Exception as e:
            logger.error(f"Failed to train neural load balancer: {e}")
    
    def predict_load(self, metrics: InstanceMetrics, global_metrics: Dict[str, Any]) -> LoadPrediction:
        """Predict optimal load for an instance."""
        if not self.is_trained or not self.model or not self.scaler:
            # Fallback to heuristic-based prediction
            return self._heuristic_prediction(metrics)
        
        try:
            features = self._extract_features(metrics, global_metrics)
            features_scaled = self.scaler.transform([features])
            predicted_load = self.model.predict(features_scaled)[0]
            
            # Calculate confidence based on recent prediction accuracy
            confidence = self._calculate_confidence()
            
            # Determine recommendation
            recommendation = self._make_recommendation(metrics, predicted_load)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(metrics, predicted_load, recommendation)
            
            return LoadPrediction(
                instance_id=metrics.instance_id,
                predicted_load=float(predicted_load),
                confidence=confidence,
                recommendation=recommendation,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Neural load prediction failed: {e}")
            return self._heuristic_prediction(metrics)
    
    def _heuristic_prediction(self, metrics: InstanceMetrics) -> LoadPrediction:
        """Fallback heuristic-based prediction."""
        # Simple heuristic based on current metrics
        load_score = (
            metrics.cpu_usage * 0.3 +
            metrics.memory_usage * 0.2 +
            (metrics.active_connections / 100.0) * 0.2 +
            metrics.error_rate * 100 * 0.3
        )
        
        if load_score > 0.8:
            recommendation = LoadBalancingDecision.SCALE_UP
        elif load_score < 0.2:
            recommendation = LoadBalancingDecision.SCALE_DOWN
        else:
            recommendation = LoadBalancingDecision.ROUTE_TO_INSTANCE
        
        return LoadPrediction(
            instance_id=metrics.instance_id,
            predicted_load=load_score,
            confidence=0.6,  # Medium confidence for heuristic
            recommendation=recommendation,
            reasoning=f"Heuristic-based: load_score={load_score:.2f}"
        )
    
    def _calculate_confidence(self) -> float:
        """Calculate prediction confidence based on historical accuracy."""
        if len(self.prediction_history) < 10:
            return 0.5
        
        # Simple confidence calculation based on recent prediction errors
        recent_errors = [abs(pred - actual) for pred, actual in self.prediction_history[-20:]]
        avg_error = np.mean(recent_errors) if recent_errors else 0.5
        confidence = max(0.1, 1.0 - avg_error)
        return confidence
    
    def _make_recommendation(self, metrics: InstanceMetrics, predicted_load: float) -> LoadBalancingDecision:
        """Make load balancing recommendation based on prediction."""
        if metrics.health_score < 0.5:
            return LoadBalancingDecision.MARK_UNHEALTHY
        elif predicted_load > 0.9:
            return LoadBalancingDecision.SCALE_UP
        elif predicted_load < 0.1:
            return LoadBalancingDecision.SCALE_DOWN
        elif predicted_load > 0.7:
            return LoadBalancingDecision.REDISTRIBUTE
        else:
            return LoadBalancingDecision.ROUTE_TO_INSTANCE
    
    def _generate_reasoning(self, metrics: InstanceMetrics, predicted_load: float, recommendation: LoadBalancingDecision) -> str:
        """Generate human-readable reasoning for the decision."""
        factors = []
        
        if metrics.cpu_usage > 0.8:
            factors.append(f"high CPU ({metrics.cpu_usage:.1%})")
        if metrics.memory_usage > 0.8:
            factors.append(f"high memory ({metrics.memory_usage:.1%})")
        if metrics.error_rate > 0.05:
            factors.append(f"high error rate ({metrics.error_rate:.1%})")
        if metrics.response_time > 1000:
            factors.append(f"slow response ({metrics.response_time:.0f}ms)")
        
        factor_str = ", ".join(factors) if factors else "normal metrics"
        
        return f"Predicted load: {predicted_load:.2f}, factors: {factor_str}, action: {recommendation.value}"


class RLLoadBalancer:
    """Reinforcement Learning-based load balancer."""
    
    def __init__(self, epsilon: float = 0.1, learning_rate: float = 0.1, discount_factor: float = 0.9):
        
    """__init__ function."""
self.epsilon = epsilon  # Exploration rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Q-table: state -> action -> Q-value
        self.q_table: Dict[str, Dict[LoadBalancingDecision, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        self.state_history: deque = deque(maxlen=1000)
        self.action_history: deque = deque(maxlen=1000)
        self.reward_history: deque = deque(maxlen=1000)
        
        self.total_episodes = 0
    
    def _discretize_state(self, metrics: InstanceMetrics) -> str:
        """Convert continuous metrics to discrete state."""
        cpu_bucket = int(metrics.cpu_usage * 10)  # 0-10
        memory_bucket = int(metrics.memory_usage * 10)  # 0-10
        response_bucket = min(int(metrics.response_time / 100), 10)  # 0-10
        error_bucket = min(int(metrics.error_rate * 100), 10)  # 0-10
        
        return f"{cpu_bucket}_{memory_bucket}_{response_bucket}_{error_bucket}"
    
    def choose_action(self, metrics: InstanceMetrics) -> LoadBalancingDecision:
        """Choose action using epsilon-greedy policy."""
        state = self._discretize_state(metrics)
        
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(list(LoadBalancingDecision))
        else:
            # Exploit: choose best known action
            q_values = self.q_table[state]
            if not q_values:
                return random.choice(list(LoadBalancingDecision))
            
            best_action = max(q_values.keys(), key=lambda a: q_values[a])
            return best_action
    
    def update_q_value(self, state: str, action: LoadBalancingDecision, reward: float, next_state: str):
        """Update Q-value using Q-learning algorithm."""
        current_q = self.q_table[state][action]
        
        # Find maximum Q-value for next state
        next_q_values = self.q_table[next_state]
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, metrics_before: InstanceMetrics, metrics_after: InstanceMetrics, action: LoadBalancingDecision) -> float:
        """Calculate reward based on performance improvement."""
        # Reward components
        response_time_improvement = (metrics_before.response_time - metrics_after.response_time) / 1000.0
        error_rate_improvement = (metrics_before.error_rate - metrics_after.error_rate) * 100
        throughput_improvement = (metrics_after.throughput - metrics_before.throughput) / 100.0
        
        # Resource utilization penalty
        resource_penalty = (metrics_after.cpu_usage + metrics_after.memory_usage) / 2.0
        
        # Calculate total reward
        reward = (
            response_time_improvement * 2.0 +
            error_rate_improvement * 3.0 +
            throughput_improvement * 1.0 -
            resource_penalty * 0.5
        )
        
        # Action-specific bonuses/penalties
        if action == LoadBalancingDecision.SCALE_UP and metrics_before.cpu_usage > 0.8:
            reward += 1.0  # Good scaling decision
        elif action == LoadBalancingDecision.SCALE_DOWN and metrics_before.cpu_usage < 0.2:
            reward += 0.5  # Good resource optimization
        elif action == LoadBalancingDecision.MARK_UNHEALTHY and metrics_before.error_rate > 0.1:
            reward += 2.0  # Good health detection
        
        return reward
    
    def train_episode(self, initial_metrics: InstanceMetrics, action: LoadBalancingDecision, final_metrics: InstanceMetrics):
        """Train the RL agent with one episode."""
        initial_state = self._discretize_state(initial_metrics)
        final_state = self._discretize_state(final_metrics)
        
        reward = self.calculate_reward(initial_metrics, final_metrics, action)
        
        # Update Q-value
        self.update_q_value(initial_state, action, reward, final_state)
        
        # Record episode
        self.state_history.append(initial_state)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.total_episodes += 1
        
        # Decay exploration rate
        if self.total_episodes % 100 == 0:
            self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of learned policy."""
        if not self.q_table:
            return {"message": "No policy learned yet"}
        
        # Find most common states and their best actions
        policy_summary = {}
        for state, actions in self.q_table.items():
            if actions:
                best_action = max(actions.keys(), key=lambda a: actions[a])
                best_q_value = actions[best_action]
                policy_summary[state] = {
                    "best_action": best_action.value,
                    "q_value": best_q_value,
                    "action_count": len(actions)
                }
        
        return {
            "total_states": len(self.q_table),
            "total_episodes": self.total_episodes,
            "current_epsilon": self.epsilon,
            "avg_reward": np.mean(list(self.reward_history)) if self.reward_history else 0,
            "top_policies": dict(list(policy_summary.items())[:10])
        }


class AILoadBalancer:
    """Complete AI-powered load balancer combining neural networks and RL."""
    
    def __init__(self) -> Any:
        self.neural_balancer = NeuralLoadBalancer()
        self.rl_balancer = RLLoadBalancer()
        self.instance_metrics: Dict[str, InstanceMetrics] = {}
        self.global_metrics: Dict[str, Any] = {}
        self.decision_history: deque = deque(maxlen=1000)
        
    async async def route_request(self, available_instances: List[str], request_context: Dict[str, Any] = None) -> str:
        """Route request to optimal instance using AI."""
        if not available_instances:
            raise ValueError("No instances available")
        
        if len(available_instances) == 1:
            return available_instances[0]
        
        # Get predictions for all instances
        predictions = []
        for instance_id in available_instances:
            if instance_id in self.instance_metrics:
                metrics = self.instance_metrics[instance_id]
                
                # Get neural network prediction
                neural_prediction = self.neural_balancer.predict_load(metrics, self.global_metrics)
                
                # Get RL recommendation
                rl_action = self.rl_balancer.choose_action(metrics)
                
                # Combine predictions (weighted)
                combined_score = self._combine_predictions(neural_prediction, rl_action, metrics)
                
                predictions.append((instance_id, combined_score, neural_prediction, rl_action))
        
        if not predictions:
            # Fallback to random selection
            return random.choice(available_instances)
        
        # Select best instance
        best_instance = min(predictions, key=lambda x: x[1])
        selected_instance_id = best_instance[0]
        
        # Record decision
        self.decision_history.append({
            'timestamp': datetime.utcnow(),
            'selected_instance': selected_instance_id,
            'predictions': predictions,
            'request_context': request_context
        })
        
        return selected_instance_id
    
    def _combine_predictions(self, neural_prediction: LoadPrediction, rl_action: LoadBalancingDecision, metrics: InstanceMetrics) -> float:
        """Combine neural network and RL predictions into a single score."""
        # Neural network score (lower is better for routing)
        neural_score = neural_prediction.predicted_load * neural_prediction.confidence
        
        # RL score adjustment
        rl_score_adjustment = 0.0
        if rl_action == LoadBalancingDecision.ROUTE_TO_INSTANCE:
            rl_score_adjustment = -0.2  # Encourage routing
        elif rl_action == LoadBalancingDecision.MARK_UNHEALTHY:
            rl_score_adjustment = 2.0  # Strongly discourage
        elif rl_action in [LoadBalancingDecision.SCALE_UP, LoadBalancingDecision.REDISTRIBUTE]:
            rl_score_adjustment = 0.3  # Slightly discourage
        
        # Health score consideration
        health_penalty = (1.0 - metrics.health_score) * 0.5
        
        # Combined score (lower is better)
        combined_score = neural_score + rl_score_adjustment + health_penalty
        
        return combined_score
    
    def update_instance_metrics(self, instance_id: str, metrics: InstanceMetrics):
        """Update metrics for an instance."""
        old_metrics = self.instance_metrics.get(instance_id)
        self.instance_metrics[instance_id] = metrics
        
        # Record for neural network training
        if old_metrics:
            # Calculate actual load based on metrics change
            actual_load = (
                metrics.cpu_usage * 0.4 +
                metrics.memory_usage * 0.3 +
                (metrics.active_connections / 100.0) * 0.3
            )
            
            self.neural_balancer.record_performance(old_metrics, actual_load, self.global_metrics)
    
    def update_global_metrics(self, metrics: Dict[str, Any]):
        """Update global system metrics."""
        self.global_metrics.update(metrics)
    
    def train_rl_agent(self, instance_id: str, action: LoadBalancingDecision, result_metrics: InstanceMetrics):
        """Train the RL agent with outcome data."""
        if instance_id in self.instance_metrics:
            initial_metrics = self.instance_metrics[instance_id]
            self.rl_balancer.train_episode(initial_metrics, action, result_metrics)
    
    def get_ai_insights(self) -> Dict[str, Any]:
        """Get insights from the AI load balancer."""
        return {
            "neural_balancer": {
                "is_trained": self.neural_balancer.is_trained,
                "training_data_size": len(self.neural_balancer.training_data),
                "prediction_history_size": len(self.neural_balancer.prediction_history)
            },
            "rl_balancer": self.rl_balancer.get_policy_summary(),
            "instance_count": len(self.instance_metrics),
            "decision_history_size": len(self.decision_history),
            "global_metrics": self.global_metrics
        } 