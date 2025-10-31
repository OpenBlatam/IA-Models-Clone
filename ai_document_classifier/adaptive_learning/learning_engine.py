"""
Advanced Adaptive Learning Engine
================================

Intelligent learning system that continuously improves and adapts
based on user feedback, performance data, and changing patterns.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import statistics
import math
import pickle
import threading
import time

logger = logging.getLogger(__name__)

class LearningType(Enum):
    """Learning types"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    FEDERATED = "federated"
    CONTINUOUS = "continuous"
    INCREMENTAL = "incremental"
    META = "meta"

class AdaptationStrategy(Enum):
    """Adaptation strategies"""
    GRADUAL = "gradual"
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    TRIGGERED = "triggered"
    BATCH = "batch"
    STREAMING = "streaming"
    ENSEMBLE = "ensemble"
    TRANSFER = "transfer"

class PerformanceMetric(Enum):
    """Performance metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    LOSS = "loss"
    CONVERGENCE = "convergence"
    STABILITY = "stability"

@dataclass
class LearningData:
    """Learning data point"""
    id: str
    input_data: Any
    expected_output: Optional[Any]
    actual_output: Optional[Any]
    feedback: Optional[float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningSession:
    """Learning session"""
    id: str
    session_type: LearningType
    start_time: datetime
    end_time: Optional[datetime]
    data_points: List[LearningData]
    performance_metrics: Dict[PerformanceMetric, float]
    model_updates: List[Dict[str, Any]]
    status: str  # active, completed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdaptationEvent:
    """Adaptation event"""
    id: str
    trigger_type: str
    trigger_data: Dict[str, Any]
    adaptation_strategy: AdaptationStrategy
    model_changes: Dict[str, Any]
    performance_impact: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningProgress:
    """Learning progress tracking"""
    id: str
    metric_name: str
    current_value: float
    target_value: float
    improvement_rate: float
    convergence_status: str
    last_updated: datetime
    history: List[Tuple[datetime, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedAdaptiveLearningEngine:
    """
    Advanced adaptive learning engine
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize adaptive learning engine
        
        Args:
            models_dir: Directory for model storage
        """
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Learning data storage
        self.learning_sessions: Dict[str, LearningSession] = {}
        self.adaptation_events: List[AdaptationEvent] = []
        self.learning_progress: Dict[str, LearningProgress] = {}
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.model_versions: Dict[str, List[str]] = defaultdict(list)
        
        # Learning configuration
        self.learning_config = self._initialize_learning_config()
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.feedback_buffer: deque = deque(maxlen=10000)
        
        # Learning algorithms
        self.learning_algorithms = self._initialize_learning_algorithms()
        
        # Adaptation triggers
        self.adaptation_triggers = self._initialize_adaptation_triggers()
        
        # Background learning thread
        self.learning_thread = None
        self.learning_active = False
        
        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_learning_config(self) -> Dict[str, Any]:
        """Initialize learning configuration"""
        return {
            "learning_rates": {
                "supervised": 0.01,
                "unsupervised": 0.001,
                "reinforcement": 0.1,
                "transfer": 0.005
            },
            "batch_sizes": {
                "small": 32,
                "medium": 64,
                "large": 128
            },
            "adaptation_thresholds": {
                "performance_degradation": 0.05,  # 5% drop
                "feedback_negative": 0.3,  # 30% negative feedback
                "data_drift": 0.1,  # 10% distribution change
                "concept_drift": 0.15  # 15% concept change
            },
            "learning_schedules": {
                "continuous": True,
                "batch_interval": 3600,  # 1 hour
                "evaluation_interval": 1800,  # 30 minutes
                "adaptation_interval": 7200  # 2 hours
            }
        }
    
    def _initialize_learning_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize learning algorithms"""
        return {
            "online_learning": {
                "type": "incremental",
                "algorithm": "stochastic_gradient_descent",
                "parameters": {"learning_rate": 0.01, "momentum": 0.9}
            },
            "ensemble_learning": {
                "type": "ensemble",
                "algorithm": "adaptive_boosting",
                "parameters": {"n_estimators": 100, "learning_rate": 1.0}
            },
            "transfer_learning": {
                "type": "transfer",
                "algorithm": "fine_tuning",
                "parameters": {"freeze_layers": 5, "learning_rate": 0.001}
            },
            "reinforcement_learning": {
                "type": "reinforcement",
                "algorithm": "q_learning",
                "parameters": {"epsilon": 0.1, "gamma": 0.95}
            }
        }
    
    def _initialize_adaptation_triggers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize adaptation triggers"""
        return {
            "performance_degradation": {
                "metric": "accuracy",
                "threshold": 0.05,
                "window_size": 100,
                "strategy": AdaptationStrategy.IMMEDIATE
            },
            "negative_feedback": {
                "metric": "user_satisfaction",
                "threshold": 0.3,
                "window_size": 50,
                "strategy": AdaptationStrategy.GRADUAL
            },
            "data_drift": {
                "metric": "distribution_similarity",
                "threshold": 0.1,
                "window_size": 200,
                "strategy": AdaptationStrategy.SCHEDULED
            },
            "concept_drift": {
                "metric": "concept_similarity",
                "threshold": 0.15,
                "window_size": 100,
                "strategy": AdaptationStrategy.TRIGGERED
            }
        }
    
    def _initialize_default_models(self):
        """Initialize default models"""
        # Initialize basic classification model
        self.models["document_classifier"] = {
            "type": "classification",
            "algorithm": "random_forest",
            "parameters": {"n_estimators": 100, "max_depth": 10},
            "performance": {"accuracy": 0.85, "precision": 0.82, "recall": 0.80},
            "last_updated": datetime.now()
        }
        
        # Initialize sentiment analysis model
        self.models["sentiment_analyzer"] = {
            "type": "sentiment",
            "algorithm": "naive_bayes",
            "parameters": {"alpha": 1.0},
            "performance": {"accuracy": 0.78, "f1_score": 0.76},
            "last_updated": datetime.now()
        }
        
        # Initialize recommendation model
        self.models["recommender"] = {
            "type": "recommendation",
            "algorithm": "collaborative_filtering",
            "parameters": {"n_factors": 50, "regularization": 0.01},
            "performance": {"rmse": 0.85, "mae": 0.65},
            "last_updated": datetime.now()
        }
    
    async def start_learning(self):
        """Start continuous learning"""
        if self.learning_active:
            logger.warning("Learning is already active")
            return
        
        self.learning_active = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        logger.info("Adaptive learning started")
    
    async def stop_learning(self):
        """Stop continuous learning"""
        self.learning_active = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        
        logger.info("Adaptive learning stopped")
    
    def _learning_loop(self):
        """Main learning loop"""
        while self.learning_active:
            try:
                # Process feedback buffer
                self._process_feedback_buffer()
                
                # Check adaptation triggers
                self._check_adaptation_triggers()
                
                # Update learning progress
                self._update_learning_progress()
                
                # Sleep for evaluation interval
                time.sleep(self.learning_config["learning_schedules"]["evaluation_interval"])
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    async def add_learning_data(self, input_data: Any, expected_output: Optional[Any] = None, actual_output: Optional[Any] = None, feedback: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add learning data point
        
        Args:
            input_data: Input data for learning
            expected_output: Expected output (for supervised learning)
            actual_output: Actual output from model
            feedback: User feedback score
            metadata: Additional metadata
            
        Returns:
            Learning data ID
        """
        if metadata is None:
            metadata = {}
        
        learning_data = LearningData(
            id=str(uuid.uuid4()),
            input_data=input_data,
            expected_output=expected_output,
            actual_output=actual_output,
            feedback=feedback,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Add to feedback buffer
        self.feedback_buffer.append(learning_data)
        
        logger.info(f"Added learning data point: {learning_data.id}")
        
        return learning_data.id
    
    async def create_learning_session(self, session_type: LearningType, model_name: str) -> LearningSession:
        """
        Create new learning session
        
        Args:
            session_type: Type of learning session
            model_name: Name of model to train
            
        Returns:
            Created learning session
        """
        session = LearningSession(
            id=str(uuid.uuid4()),
            session_type=session_type,
            start_time=datetime.now(),
            end_time=None,
            data_points=[],
            performance_metrics={},
            model_updates=[],
            status="active"
        )
        
        self.learning_sessions[session.id] = session
        
        logger.info(f"Created learning session: {session.id} ({session_type.value})")
        
        return session
    
    async def train_model(self, session_id: str, model_name: str, training_data: List[LearningData]) -> Dict[str, Any]:
        """
        Train model in learning session
        
        Args:
            session_id: Learning session ID
            model_name: Name of model to train
            training_data: Training data
            
        Returns:
            Training results
        """
        if session_id not in self.learning_sessions:
            raise ValueError(f"Learning session not found: {session_id}")
        
        session = self.learning_sessions[session_id]
        
        # Simulate model training
        training_results = {
            "model_name": model_name,
            "training_samples": len(training_data),
            "training_time": np.random.uniform(10, 60),  # seconds
            "performance_improvement": np.random.uniform(0.01, 0.05),
            "convergence_achieved": True,
            "final_metrics": {
                "accuracy": np.random.uniform(0.85, 0.95),
                "precision": np.random.uniform(0.80, 0.90),
                "recall": np.random.uniform(0.75, 0.85),
                "f1_score": np.random.uniform(0.77, 0.87)
            }
        }
        
        # Update session
        session.data_points.extend(training_data)
        session.performance_metrics = training_results["final_metrics"]
        session.model_updates.append(training_results)
        
        # Update model
        if model_name in self.models:
            self.models[model_name]["performance"] = training_results["final_metrics"]
            self.models[model_name]["last_updated"] = datetime.now()
            
            # Save model version
            version_id = str(uuid.uuid4())
            self.model_versions[model_name].append(version_id)
        
        logger.info(f"Trained model {model_name} in session {session_id}")
        
        return training_results
    
    def _process_feedback_buffer(self):
        """Process feedback buffer for learning"""
        if len(self.feedback_buffer) < 10:  # Minimum batch size
            return
        
        # Extract recent feedback
        recent_feedback = list(self.feedback_buffer)[-100:]  # Last 100 items
        
        # Analyze feedback patterns
        feedback_scores = [item.feedback for item in recent_feedback if item.feedback is not None]
        
        if feedback_scores:
            avg_feedback = statistics.mean(feedback_scores)
            negative_feedback_ratio = len([f for f in feedback_scores if f < 3.0]) / len(feedback_scores)
            
            # Check if adaptation is needed
            if negative_feedback_ratio > self.learning_config["adaptation_thresholds"]["feedback_negative"]:
                self._trigger_adaptation("negative_feedback", {
                    "negative_ratio": negative_feedback_ratio,
                    "avg_feedback": avg_feedback
                })
    
    def _check_adaptation_triggers(self):
        """Check all adaptation triggers"""
        for trigger_name, trigger_config in self.adaptation_triggers.items():
            if self._evaluate_trigger(trigger_name, trigger_config):
                self._trigger_adaptation(trigger_name, trigger_config)
    
    def _evaluate_trigger(self, trigger_name: str, trigger_config: Dict[str, Any]) -> bool:
        """Evaluate if adaptation trigger is met"""
        try:
            if trigger_name == "performance_degradation":
                return self._check_performance_degradation(trigger_config)
            elif trigger_name == "negative_feedback":
                return self._check_negative_feedback(trigger_config)
            elif trigger_name == "data_drift":
                return self._check_data_drift(trigger_config)
            elif trigger_name == "concept_drift":
                return self._check_concept_drift(trigger_config)
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating trigger {trigger_name}: {e}")
            return False
    
    def _check_performance_degradation(self, config: Dict[str, Any]) -> bool:
        """Check for performance degradation"""
        if len(self.performance_history) < config["window_size"]:
            return False
        
        recent_performance = list(self.performance_history)[-config["window_size"]:]
        avg_recent = statistics.mean(recent_performance)
        
        # Compare with baseline (first half of history)
        if len(self.performance_history) >= config["window_size"] * 2:
            baseline = list(self.performance_history)[:config["window_size"]]
            avg_baseline = statistics.mean(baseline)
            
            degradation = (avg_baseline - avg_recent) / avg_baseline
            return degradation > config["threshold"]
        
        return False
    
    def _check_negative_feedback(self, config: Dict[str, Any]) -> bool:
        """Check for negative feedback patterns"""
        recent_feedback = list(self.feedback_buffer)[-config["window_size"]:]
        feedback_scores = [item.feedback for item in recent_feedback if item.feedback is not None]
        
        if len(feedback_scores) < 10:
            return False
        
        negative_ratio = len([f for f in feedback_scores if f < 3.0]) / len(feedback_scores)
        return negative_ratio > config["threshold"]
    
    def _check_data_drift(self, config: Dict[str, Any]) -> bool:
        """Check for data distribution drift"""
        # Simplified drift detection
        # In practice, you'd use statistical tests like KS test or PSI
        recent_data = list(self.feedback_buffer)[-config["window_size"]:]
        
        if len(recent_data) < 20:
            return False
        
        # Simulate drift detection
        drift_score = np.random.uniform(0.05, 0.2)
        return drift_score > config["threshold"]
    
    def _check_concept_drift(self, config: Dict[str, Any]) -> bool:
        """Check for concept drift"""
        # Simplified concept drift detection
        # In practice, you'd use more sophisticated methods
        recent_data = list(self.feedback_buffer)[-config["window_size"]:]
        
        if len(recent_data) < 20:
            return False
        
        # Simulate concept drift detection
        concept_drift_score = np.random.uniform(0.1, 0.25)
        return concept_drift_score > config["threshold"]
    
    def _trigger_adaptation(self, trigger_type: str, trigger_data: Dict[str, Any]):
        """Trigger model adaptation"""
        try:
            # Create adaptation event
            event = AdaptationEvent(
                id=str(uuid.uuid4()),
                trigger_type=trigger_type,
                trigger_data=trigger_data,
                adaptation_strategy=self.adaptation_triggers[trigger_type]["strategy"],
                model_changes={},
                performance_impact=0.0,
                timestamp=datetime.now()
            )
            
            # Perform adaptation based on strategy
            if event.adaptation_strategy == AdaptationStrategy.IMMEDIATE:
                self._immediate_adaptation(event)
            elif event.adaptation_strategy == AdaptationStrategy.GRADUAL:
                self._gradual_adaptation(event)
            elif event.adaptation_strategy == AdaptationStrategy.SCHEDULED:
                self._scheduled_adaptation(event)
            elif event.adaptation_strategy == AdaptationStrategy.TRIGGERED:
                self._triggered_adaptation(event)
            
            self.adaptation_events.append(event)
            
            logger.info(f"Triggered adaptation: {trigger_type} ({event.adaptation_strategy.value})")
            
        except Exception as e:
            logger.error(f"Error triggering adaptation: {e}")
    
    def _immediate_adaptation(self, event: AdaptationEvent):
        """Perform immediate adaptation"""
        # Update learning rates
        for model_name in self.models:
            if model_name in self.models:
                self.models[model_name]["learning_rate"] *= 1.1  # Increase learning rate
        
        event.model_changes = {"learning_rate_increase": 0.1}
        event.performance_impact = 0.05
    
    def _gradual_adaptation(self, event: AdaptationEvent):
        """Perform gradual adaptation"""
        # Gradually adjust model parameters
        for model_name in self.models:
            if model_name in self.models:
                self.models[model_name]["regularization"] *= 0.95  # Decrease regularization
        
        event.model_changes = {"regularization_decrease": 0.05}
        event.performance_impact = 0.02
    
    def _scheduled_adaptation(self, event: AdaptationEvent):
        """Perform scheduled adaptation"""
        # Schedule model retraining
        event.model_changes = {"retraining_scheduled": True}
        event.performance_impact = 0.03
    
    def _triggered_adaptation(self, event: AdaptationEvent):
        """Perform triggered adaptation"""
        # Trigger specific model updates
        event.model_changes = {"model_update_triggered": True}
        event.performance_impact = 0.04
    
    def _update_learning_progress(self):
        """Update learning progress tracking"""
        for model_name, model_data in self.models.items():
            if model_name not in self.learning_progress:
                self.learning_progress[model_name] = LearningProgress(
                    id=str(uuid.uuid4()),
                    metric_name="accuracy",
                    current_value=model_data["performance"]["accuracy"],
                    target_value=0.95,
                    improvement_rate=0.0,
                    convergence_status="learning",
                    last_updated=datetime.now(),
                    history=[]
                )
            
            progress = self.learning_progress[model_name]
            current_accuracy = model_data["performance"]["accuracy"]
            
            # Update progress
            progress.current_value = current_accuracy
            progress.last_updated = datetime.now()
            progress.history.append((datetime.now(), current_accuracy))
            
            # Calculate improvement rate
            if len(progress.history) >= 2:
                recent_values = [h[1] for h in progress.history[-5:]]
                if len(recent_values) >= 2:
                    progress.improvement_rate = (recent_values[-1] - recent_values[0]) / len(recent_values)
            
            # Update convergence status
            if current_accuracy >= progress.target_value:
                progress.convergence_status = "converged"
            elif progress.improvement_rate > 0.01:
                progress.convergence_status = "improving"
            elif progress.improvement_rate < -0.01:
                progress.convergence_status = "declining"
            else:
                progress.convergence_status = "stable"
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        total_sessions = len(self.learning_sessions)
        active_sessions = len([s for s in self.learning_sessions.values() if s.status == "active"])
        total_adaptations = len(self.adaptation_events)
        
        # Model performance
        model_performance = {}
        for model_name, model_data in self.models.items():
            model_performance[model_name] = {
                "accuracy": model_data["performance"]["accuracy"],
                "last_updated": model_data["last_updated"].isoformat()
            }
        
        # Learning progress
        progress_summary = {}
        for model_name, progress in self.learning_progress.items():
            progress_summary[model_name] = {
                "current_value": progress.current_value,
                "target_value": progress.target_value,
                "improvement_rate": progress.improvement_rate,
                "convergence_status": progress.convergence_status
            }
        
        return {
            "total_learning_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_adaptations": total_adaptations,
            "model_performance": model_performance,
            "learning_progress": progress_summary,
            "feedback_buffer_size": len(self.feedback_buffer),
            "performance_history_size": len(self.performance_history),
            "learning_active": self.learning_active
        }
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model by name"""
        return self.models.get(model_name)
    
    def get_learning_session(self, session_id: str) -> Optional[LearningSession]:
        """Get learning session by ID"""
        return self.learning_sessions.get(session_id)
    
    def get_adaptation_events(self, limit: int = 100) -> List[AdaptationEvent]:
        """Get recent adaptation events"""
        return self.adaptation_events[-limit:]
    
    def get_learning_progress(self, model_name: str) -> Optional[LearningProgress]:
        """Get learning progress for model"""
        return self.learning_progress.get(model_name)

# Example usage
if __name__ == "__main__":
    # Initialize adaptive learning engine
    learning_engine = AdvancedAdaptiveLearningEngine()
    
    # Start learning
    asyncio.run(learning_engine.start_learning())
    
    # Add learning data
    for i in range(50):
        feedback = np.random.uniform(1, 5)
        await learning_engine.add_learning_data(
            input_data=f"document_{i}",
            expected_output="novel",
            actual_output="novel" if feedback > 3 else "contract",
            feedback=feedback
        )
    
    # Create learning session
    session = await learning_engine.create_learning_session(LearningType.SUPERVISED, "document_classifier")
    
    # Simulate training
    training_data = [
        LearningData(
            id=str(uuid.uuid4()),
            input_data=f"training_doc_{i}",
            expected_output="novel",
            actual_output="novel",
            feedback=4.5,
            timestamp=datetime.now()
        )
        for i in range(20)
    ]
    
    training_results = await learning_engine.train_model(session.id, "document_classifier", training_data)
    
    print("Training Results:")
    print(f"Training Samples: {training_results['training_samples']}")
    print(f"Performance Improvement: {training_results['performance_improvement']:.3f}")
    print(f"Final Accuracy: {training_results['final_metrics']['accuracy']:.3f}")
    
    # Get statistics
    stats = await learning_engine.get_learning_statistics()
    
    print(f"\nLearning Statistics:")
    print(f"Total Sessions: {stats['total_learning_sessions']}")
    print(f"Active Sessions: {stats['active_sessions']}")
    print(f"Total Adaptations: {stats['total_adaptations']}")
    print(f"Learning Active: {stats['learning_active']}")
    
    # Get model performance
    model = learning_engine.get_model("document_classifier")
    if model:
        print(f"\nModel Performance:")
        print(f"Accuracy: {model['performance']['accuracy']:.3f}")
        print(f"Last Updated: {model['last_updated']}")
    
    # Stop learning
    asyncio.run(learning_engine.stop_learning())
    
    print("\nAdvanced Adaptive Learning Engine initialized successfully")

























