#!/usr/bin/env python3
"""
Advanced Real-Time Learning System for Frontier Model Training
Provides comprehensive online learning, streaming data processing, and adaptive model updates.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import kafka
from kafka import KafkaProducer, KafkaConsumer
import redis
import websockets
import asyncio
import aiohttp
import socket
import ssl
import cryptography
from cryptography.fernet import Fernet
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import GPUtil
import joblib
import pickle
from collections import defaultdict, deque
import queue
import threading
import warnings
warnings.filterwarnings('ignore')

console = Console()

class LearningMode(Enum):
    """Learning modes."""
    ONLINE = "online"
    BATCH = "batch"
    STREAMING = "streaming"
    INCREMENTAL = "incremental"
    CONTINUOUS = "continuous"
    ADAPTIVE = "adaptive"
    REACTIVE = "reactive"
    PROACTIVE = "proactive"

class DataSource(Enum):
    """Data sources."""
    KAFKA = "kafka"
    REDIS = "redis"
    WEBSOCKET = "websocket"
    HTTP_STREAM = "http_stream"
    FILE_STREAM = "file_stream"
    DATABASE_STREAM = "database_stream"
    SENSOR_DATA = "sensor_data"
    USER_INTERACTION = "user_interaction"

class UpdateStrategy(Enum):
    """Update strategies."""
    GRADIENT_DESCENT = "gradient_descent"
    STOCHASTIC_GRADIENT = "stochastic_gradient"
    ADAPTIVE_GRADIENT = "adaptive_gradient"
    MOMENTUM = "momentum"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ONLINE_LEARNING = "online_learning"

class DriftDetectionMethod(Enum):
    """Drift detection methods."""
    STATISTICAL = "statistical"
    DISTRIBUTION_BASED = "distribution_based"
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    ADAPTIVE_WINDOW = "adaptive_window"
    ENSEMBLE_DRIFT = "ensemble_drift"

@dataclass
class RealTimeConfig:
    """Real-time learning configuration."""
    learning_mode: LearningMode = LearningMode.ONLINE
    data_source: DataSource = DataSource.KAFKA
    update_strategy: UpdateStrategy = UpdateStrategy.ADAM
    drift_detection: DriftDetectionMethod = DriftDetectionMethod.STATISTICAL
    learning_rate: float = 0.001
    batch_size: int = 32
    update_frequency: float = 1.0  # seconds
    drift_threshold: float = 0.1
    adaptation_rate: float = 0.01
    memory_size: int = 10000
    enable_online_evaluation: bool = True
    enable_drift_detection: bool = True
    enable_model_adaptation: bool = True
    enable_performance_monitoring: bool = True
    enable_data_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_ensemble_learning: bool = True
    enable_transfer_learning: bool = True
    device: str = "auto"

@dataclass
class StreamingData:
    """Streaming data point."""
    data_id: str
    features: np.ndarray
    target: Optional[float] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

@dataclass
class LearningEvent:
    """Learning event."""
    event_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    performance_impact: float

@dataclass
class RealTimeResult:
    """Real-time learning result."""
    result_id: str
    model_performance: Dict[str, float]
    drift_detected: bool
    adaptation_applied: bool
    learning_events: List[LearningEvent]
    streaming_metrics: Dict[str, Any]
    created_at: datetime

class DataStreamProcessor:
    """Data stream processing engine."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data buffers
        self.data_buffer = deque(maxlen=config.memory_size)
        self.feature_buffer = deque(maxlen=config.memory_size)
        self.target_buffer = deque(maxlen=config.memory_size)
        
        # Processing queues
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Data sources
        self.data_sources = {}
        self._init_data_sources()
    
    def _init_data_sources(self):
        """Initialize data sources."""
        if self.config.data_source == DataSource.KAFKA:
            self._init_kafka()
        elif self.config.data_source == DataSource.REDIS:
            self._init_redis()
        elif self.config.data_source == DataSource.WEBSOCKET:
            self._init_websocket()
        elif self.config.data_source == DataSource.HTTP_STREAM:
            self._init_http_stream()
    
    def _init_kafka(self):
        """Initialize Kafka consumer."""
        try:
            self.kafka_consumer = KafkaConsumer(
                'real_time_data',
                bootstrap_servers=['localhost:9092'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            console.print("[green]Kafka consumer initialized[/green]")
        except Exception as e:
            self.logger.warning(f"Kafka initialization failed: {e}")
            self.kafka_consumer = None
    
    def _init_redis(self):
        """Initialize Redis client."""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            console.print("[green]Redis client initialized[/green]")
        except Exception as e:
            self.logger.warning(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    def _init_websocket(self):
        """Initialize WebSocket client."""
        self.websocket_url = "ws://localhost:8080/stream"
        console.print("[green]WebSocket client initialized[/green]")
    
    def _init_http_stream(self):
        """Initialize HTTP stream client."""
        self.http_stream_url = "http://localhost:8080/api/stream"
        console.print("[green]HTTP stream client initialized[/green]")
    
    def start_streaming(self):
        """Start data streaming."""
        console.print("[blue]Starting data streaming...[/blue]")
        
        if self.config.data_source == DataSource.KAFKA and self.kafka_consumer:
            self._stream_from_kafka()
        elif self.config.data_source == DataSource.REDIS and self.redis_client:
            self._stream_from_redis()
        elif self.config.data_source == DataSource.WEBSOCKET:
            asyncio.run(self._stream_from_websocket())
        elif self.config.data_source == DataSource.HTTP_STREAM:
            asyncio.run(self._stream_from_http())
        else:
            self._stream_synthetic_data()
    
    def _stream_from_kafka(self):
        """Stream data from Kafka."""
        for message in self.kafka_consumer:
            try:
                data = message.value
                streaming_data = self._parse_data(data)
                self._add_to_buffer(streaming_data)
            except Exception as e:
                self.logger.error(f"Kafka streaming error: {e}")
    
    def _stream_from_redis(self):
        """Stream data from Redis."""
        while True:
            try:
                # Get data from Redis list
                data = self.redis_client.lpop('real_time_data')
                if data:
                    data_dict = json.loads(data)
                    streaming_data = self._parse_data(data_dict)
                    self._add_to_buffer(streaming_data)
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Redis streaming error: {e}")
                time.sleep(1)
    
    async def _stream_from_websocket(self):
        """Stream data from WebSocket."""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                async for message in websocket:
                    data = json.loads(message)
                    streaming_data = self._parse_data(data)
                    self._add_to_buffer(streaming_data)
        except Exception as e:
            self.logger.error(f"WebSocket streaming error: {e}")
    
    async def _stream_from_http(self):
        """Stream data from HTTP stream."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.http_stream_url) as response:
                    async for line in response.content:
                        if line:
                            data = json.loads(line.decode())
                            streaming_data = self._parse_data(data)
                            self._add_to_buffer(streaming_data)
        except Exception as e:
            self.logger.error(f"HTTP streaming error: {e}")
    
    def _stream_synthetic_data(self):
        """Stream synthetic data for testing."""
        console.print("[yellow]Streaming synthetic data[/yellow]")
        
        while True:
            # Generate synthetic data
            features = np.random.randn(10)
            target = np.random.randint(0, 2)
            
            streaming_data = StreamingData(
                data_id=f"synthetic_{int(time.time())}",
                features=features,
                target=target,
                timestamp=datetime.now(),
                metadata={'source': 'synthetic'}
            )
            
            self._add_to_buffer(streaming_data)
            time.sleep(0.1)  # 10 samples per second
    
    def _parse_data(self, data: Dict[str, Any]) -> StreamingData:
        """Parse incoming data."""
        return StreamingData(
            data_id=data.get('id', f"data_{int(time.time())}"),
            features=np.array(data.get('features', [])),
            target=data.get('target'),
            timestamp=datetime.now(),
            metadata=data.get('metadata', {})
        )
    
    def _add_to_buffer(self, streaming_data: StreamingData):
        """Add data to buffer."""
        self.data_buffer.append(streaming_data)
        self.feature_buffer.append(streaming_data.features)
        
        if streaming_data.target is not None:
            self.target_buffer.append(streaming_data.target)
        
        # Add to processing queue
        self.processing_queue.put(streaming_data)
    
    def get_latest_data(self, n: int = 1) -> List[StreamingData]:
        """Get latest n data points."""
        return list(self.data_buffer)[-n:]
    
    def get_data_batch(self, batch_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get data batch for training."""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Get latest data
        latest_data = list(self.data_buffer)[-batch_size:]
        
        if not latest_data:
            return np.array([]), np.array([])
        
        features = np.array([data.features for data in latest_data])
        targets = np.array([data.target for data in latest_data if data.target is not None])
        
        return features, targets

class DriftDetector:
    """Drift detection engine."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Drift detection state
        self.reference_data = deque(maxlen=1000)
        self.drift_history = []
        self.drift_threshold = config.drift_threshold
        
        # Statistical measures
        self.reference_mean = None
        self.reference_std = None
        self.reference_distribution = None
    
    def detect_drift(self, new_data: np.ndarray) -> Dict[str, Any]:
        """Detect drift in new data."""
        drift_result = {
            'drift_detected': False,
            'drift_type': None,
            'drift_score': 0.0,
            'confidence': 0.0,
            'timestamp': datetime.now()
        }
        
        if self.config.drift_detection == DriftDetectionMethod.STATISTICAL:
            drift_result = self._statistical_drift_detection(new_data)
        elif self.config.drift_detection == DriftDetectionMethod.DISTRIBUTION_BASED:
            drift_result = self._distribution_drift_detection(new_data)
        elif self.config.drift_detection == DriftDetectionMethod.CONCEPT_DRIFT:
            drift_result = self._concept_drift_detection(new_data)
        else:
            drift_result = self._statistical_drift_detection(new_data)
        
        # Update drift history
        self.drift_history.append(drift_result)
        
        return drift_result
    
    def _statistical_drift_detection(self, new_data: np.ndarray) -> Dict[str, Any]:
        """Statistical drift detection."""
        drift_result = {
            'drift_detected': False,
            'drift_type': 'statistical',
            'drift_score': 0.0,
            'confidence': 0.0,
            'timestamp': datetime.now()
        }
        
        if len(self.reference_data) < 10:
            # Not enough reference data
            self.reference_data.append(new_data)
            return drift_result
        
        # Calculate reference statistics
        if self.reference_mean is None:
            reference_array = np.array(list(self.reference_data))
            self.reference_mean = np.mean(reference_array, axis=0)
            self.reference_std = np.std(reference_array, axis=0)
        
        # Calculate drift score
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)
        
        # Statistical distance
        mean_diff = np.abs(np.mean(new_data, axis=0) - self.reference_mean)
        std_diff = np.abs(np.std(new_data, axis=0) - self.reference_std)
        
        drift_score = np.mean(mean_diff / (self.reference_std + 1e-8)) + np.mean(std_diff / (self.reference_std + 1e-8))
        
        drift_result['drift_score'] = drift_score
        drift_result['confidence'] = min(1.0, drift_score / self.drift_threshold)
        
        if drift_score > self.drift_threshold:
            drift_result['drift_detected'] = True
        
        # Update reference data
        self.reference_data.append(new_data)
        
        return drift_result
    
    def _distribution_drift_detection(self, new_data: np.ndarray) -> Dict[str, Any]:
        """Distribution-based drift detection."""
        drift_result = {
            'drift_detected': False,
            'drift_type': 'distribution',
            'drift_score': 0.0,
            'confidence': 0.0,
            'timestamp': datetime.now()
        }
        
        if len(self.reference_data) < 20:
            self.reference_data.append(new_data)
            return drift_result
        
        # Calculate KL divergence or other distribution distance
        try:
            from scipy import stats
            
            reference_array = np.array(list(self.reference_data))
            
            # Calculate KL divergence for each feature
            kl_divergences = []
            for i in range(min(new_data.shape[1], reference_array.shape[1])):
                ref_feature = reference_array[:, i]
                new_feature = new_data[:, i] if len(new_data.shape) > 1 else new_data
                
                # Create histograms
                ref_hist, ref_bins = np.histogram(ref_feature, bins=10, density=True)
                new_hist, new_bins = np.histogram(new_feature, bins=ref_bins, density=True)
                
                # Calculate KL divergence
                kl_div = stats.entropy(new_hist + 1e-8, ref_hist + 1e-8)
                kl_divergences.append(kl_div)
            
            drift_score = np.mean(kl_divergences)
            drift_result['drift_score'] = drift_score
            drift_result['confidence'] = min(1.0, drift_score / self.drift_threshold)
            
            if drift_score > self.drift_threshold:
                drift_result['drift_detected'] = True
            
        except ImportError:
            # Fallback to statistical method
            drift_result = self._statistical_drift_detection(new_data)
        
        # Update reference data
        self.reference_data.append(new_data)
        
        return drift_result
    
    def _concept_drift_detection(self, new_data: np.ndarray) -> Dict[str, Any]:
        """Concept drift detection."""
        drift_result = {
            'drift_detected': False,
            'drift_type': 'concept',
            'drift_score': 0.0,
            'confidence': 0.0,
            'timestamp': datetime.now()
        }
        
        # Simplified concept drift detection
        # In practice, this would use model performance metrics
        if len(self.reference_data) < 10:
            self.reference_data.append(new_data)
            return drift_result
        
        # Calculate concept drift score based on feature relationships
        reference_array = np.array(list(self.reference_data))
        
        # Calculate correlation changes
        if len(new_data.shape) > 1 and new_data.shape[1] > 1:
            ref_corr = np.corrcoef(reference_array.T)
            new_corr = np.corrcoef(new_data.T)
            
            corr_diff = np.abs(ref_corr - new_corr)
            drift_score = np.mean(corr_diff)
            
            drift_result['drift_score'] = drift_score
            drift_result['confidence'] = min(1.0, drift_score / self.drift_threshold)
            
            if drift_score > self.drift_threshold:
                drift_result['drift_detected'] = True
        
        # Update reference data
        self.reference_data.append(new_data)
        
        return drift_result

class OnlineLearner:
    """Online learning engine."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Model and optimizer
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Learning state
        self.learning_history = []
        self.performance_history = []
        self.adaptation_history = []
    
    def initialize_model(self, input_size: int, output_size: int):
        """Initialize the model."""
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        ).to(self.device)
        
        # Initialize optimizer
        if self.config.update_strategy == UpdateStrategy.ADAM:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.update_strategy == UpdateStrategy.SGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.update_strategy == UpdateStrategy.RMSPROP:
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        console.print("[green]Model initialized for online learning[/green]")
    
    def update_model(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Update model with new data."""
        if self.model is None:
            return {'error': 'Model not initialized'}
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features).to(self.device)
        targets_tensor = torch.LongTensor(targets).to(self.device)
        
        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(features_tensor)
        loss = self.criterion(outputs, targets_tensor)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate performance metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == targets_tensor).float().mean().item()
        
        # Record learning event
        learning_event = {
            'timestamp': datetime.now(),
            'loss': loss.item(),
            'accuracy': accuracy,
            'data_size': len(features)
        }
        
        self.learning_history.append(learning_event)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'data_size': len(features)
        }
    
    def adapt_to_drift(self, drift_info: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt model to detected drift."""
        adaptation_result = {
            'adaptation_applied': False,
            'adaptation_type': None,
            'performance_change': 0.0,
            'timestamp': datetime.now()
        }
        
        if not drift_info.get('drift_detected', False):
            return adaptation_result
        
        drift_type = drift_info.get('drift_type', 'unknown')
        drift_score = drift_info.get('drift_score', 0.0)
        
        # Apply different adaptation strategies based on drift type
        if drift_type == 'statistical':
            adaptation_result = self._adapt_to_statistical_drift(drift_score)
        elif drift_type == 'distribution':
            adaptation_result = self._adapt_to_distribution_drift(drift_score)
        elif drift_type == 'concept':
            adaptation_result = self._adapt_to_concept_drift(drift_score)
        else:
            adaptation_result = self._adapt_to_statistical_drift(drift_score)
        
        # Record adaptation
        self.adaptation_history.append(adaptation_result)
        
        return adaptation_result
    
    def _adapt_to_statistical_drift(self, drift_score: float) -> Dict[str, Any]:
        """Adapt to statistical drift."""
        adaptation_result = {
            'adaptation_applied': True,
            'adaptation_type': 'statistical',
            'performance_change': 0.0,
            'timestamp': datetime.now()
        }
        
        # Increase learning rate temporarily
        if drift_score > self.config.drift_threshold * 2:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 1.5
        
        # Add regularization
        if drift_score > self.config.drift_threshold * 3:
            # Add weight decay
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = 1e-4
        
        return adaptation_result
    
    def _adapt_to_distribution_drift(self, drift_score: float) -> Dict[str, Any]:
        """Adapt to distribution drift."""
        adaptation_result = {
            'adaptation_applied': True,
            'adaptation_type': 'distribution',
            'performance_change': 0.0,
            'timestamp': datetime.now()
        }
        
        # Reset optimizer state
        if drift_score > self.config.drift_threshold * 2:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        return adaptation_result
    
    def _adapt_to_concept_drift(self, drift_score: float) -> Dict[str, Any]:
        """Adapt to concept drift."""
        adaptation_result = {
            'adaptation_applied': True,
            'adaptation_type': 'concept',
            'performance_change': 0.0,
            'timestamp': datetime.now()
        }
        
        # Reinitialize some layers
        if drift_score > self.config.drift_threshold * 3:
            # Reinitialize last layer
            for layer in self.model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        return adaptation_result
    
    def evaluate_model(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            return {'error': 'Model not initialized'}
        
        self.model.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            targets_tensor = torch.LongTensor(targets).to(self.device)
            
            outputs = self.model(features_tensor)
            loss = self.criterion(outputs, targets_tensor)
            
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == targets_tensor).float().mean().item()
            
            # Calculate additional metrics
            precision = precision_score(targets, predictions.cpu().numpy(), average='weighted')
            recall = recall_score(targets, predictions.cpu().numpy(), average='weighted')
            f1 = f1_score(targets, predictions.cpu().numpy(), average='weighted')
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

class RealTimeLearningSystem:
    """Main real-time learning system."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_processor = DataStreamProcessor(config)
        self.drift_detector = DriftDetector(config)
        self.online_learner = OnlineLearner(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # System state
        self.is_running = False
        self.learning_thread = None
        self.monitoring_thread = None
        
        # Results storage
        self.real_time_results: Dict[str, RealTimeResult] = {}
    
    def _init_database(self) -> str:
        """Initialize real-time learning database."""
        db_path = Path("./real_time_learning.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    performance_impact REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS real_time_results (
                    result_id TEXT PRIMARY KEY,
                    model_performance TEXT NOT NULL,
                    drift_detected BOOLEAN NOT NULL,
                    adaptation_applied BOOLEAN NOT NULL,
                    learning_events TEXT NOT NULL,
                    streaming_metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def start_learning(self, input_size: int = 10, output_size: int = 2):
        """Start real-time learning."""
        console.print("[blue]Starting real-time learning system...[/blue]")
        
        # Initialize model
        self.online_learner.initialize_model(input_size, output_size)
        
        # Start data streaming
        self.data_processor.start_streaming()
        
        # Start learning loop
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.start()
        
        # Start monitoring
        if self.config.enable_performance_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.start()
        
        console.print("[green]Real-time learning system started[/green]")
    
    def stop_learning(self):
        """Stop real-time learning."""
        console.print("[blue]Stopping real-time learning system...[/blue]")
        
        self.is_running = False
        
        if self.learning_thread:
            self.learning_thread.join()
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        console.print("[green]Real-time learning system stopped[/green]")
    
    def _learning_loop(self):
        """Main learning loop."""
        last_update_time = time.time()
        
        while self.is_running:
            try:
                # Check if it's time for an update
                current_time = time.time()
                if current_time - last_update_time >= self.config.update_frequency:
                    
                    # Get latest data
                    features, targets = self.data_processor.get_data_batch()
                    
                    if len(features) > 0 and len(targets) > 0:
                        # Detect drift
                        drift_result = self.drift_detector.detect_drift(features)
                        
                        # Update model
                        update_result = self.online_learner.update_model(features, targets)
                        
                        # Adapt to drift if detected
                        adaptation_result = None
                        if drift_result['drift_detected']:
                            adaptation_result = self.online_learner.adapt_to_drift(drift_result)
                        
                        # Create learning event
                        learning_event = LearningEvent(
                            event_id=f"event_{int(time.time())}",
                            event_type="model_update",
                            data={
                                'update_result': update_result,
                                'drift_result': drift_result,
                                'adaptation_result': adaptation_result
                            },
                            timestamp=datetime.now(),
                            performance_impact=update_result.get('accuracy', 0.0)
                        )
                        
                        # Store result
                        self._store_learning_event(learning_event)
                        
                        last_update_time = current_time
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Learning loop error: {e}")
                time.sleep(1)
    
    def _monitoring_loop(self):
        """Monitoring loop."""
        while self.is_running:
            try:
                # Monitor system performance
                system_metrics = self._get_system_metrics()
                
                # Log metrics
                self.logger.info(f"System metrics: {system_metrics}")
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(1)
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            'data_buffer_size': len(self.data_processor.data_buffer),
            'learning_events_count': len(self.online_learner.learning_history),
            'drift_detections_count': len(self.drift_detector.drift_history),
            'adaptations_count': len(self.online_learner.adaptation_history),
            'system_uptime': time.time() - (self.online_learner.learning_history[0]['timestamp'].timestamp() if self.online_learner.learning_history else time.time())
        }
    
    def _store_learning_event(self, learning_event: LearningEvent):
        """Store learning event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO learning_events 
                (event_id, event_type, data, timestamp, performance_impact)
                VALUES (?, ?, ?, ?, ?)
            """, (
                learning_event.event_id,
                learning_event.event_type,
                json.dumps(learning_event.data),
                learning_event.timestamp.isoformat(),
                learning_event.performance_impact
            ))
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get learning summary."""
        if not self.online_learner.learning_history:
            return {'total_events': 0}
        
        # Calculate summary statistics
        total_events = len(self.online_learner.learning_history)
        total_drift_detections = len(self.drift_detector.drift_history)
        total_adaptations = len(self.online_learner.adaptation_history)
        
        # Calculate average performance
        accuracies = [event['accuracy'] for event in self.online_learner.learning_history]
        losses = [event['loss'] for event in self.online_learner.learning_history]
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Calculate drift rate
        drift_rate = total_drift_detections / total_events if total_events > 0 else 0.0
        
        return {
            'total_events': total_events,
            'total_drift_detections': total_drift_detections,
            'total_adaptations': total_adaptations,
            'average_accuracy': avg_accuracy,
            'average_loss': avg_loss,
            'drift_rate': drift_rate,
            'system_uptime': time.time() - (self.online_learner.learning_history[0]['timestamp'].timestamp() if self.online_learner.learning_history else time.time())
        }
    
    def visualize_learning_progress(self, output_path: str = None) -> str:
        """Visualize learning progress."""
        if output_path is None:
            output_path = f"real_time_learning_{int(time.time())}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Learning progress
        if self.online_learner.learning_history:
            timestamps = [event['timestamp'] for event in self.online_learner.learning_history]
            accuracies = [event['accuracy'] for event in self.online_learner.learning_history]
            losses = [event['loss'] for event in self.online_learner.learning_history]
            
            axes[0, 0].plot(timestamps, accuracies, 'b-', label='Accuracy')
            axes[0, 0].set_title('Learning Progress - Accuracy')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(timestamps, losses, 'r-', label='Loss')
            axes[0, 1].set_title('Learning Progress - Loss')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Drift detection
        if self.drift_detector.drift_history:
            drift_timestamps = [drift['timestamp'] for drift in self.drift_detector.drift_history]
            drift_scores = [drift['drift_score'] for drift in self.drift_detector.drift_history]
            drift_detected = [drift['drift_detected'] for drift in self.drift_detector.drift_history]
            
            axes[1, 0].plot(drift_timestamps, drift_scores, 'g-', label='Drift Score')
            axes[1, 0].axhline(y=self.config.drift_threshold, color='r', linestyle='--', label='Threshold')
            axes[1, 0].set_title('Drift Detection')
            axes[1, 0].set_ylabel('Drift Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Drift events
            drift_events = sum(drift_detected)
            no_drift_events = len(drift_detected) - drift_events
            
            axes[1, 1].pie([drift_events, no_drift_events], 
                          labels=['Drift Detected', 'No Drift'], 
                          autopct='%1.1f%%',
                          colors=['red', 'green'])
            axes[1, 1].set_title('Drift Events Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Learning progress visualization saved: {output_path}[/green]")
        return output_path

def main():
    """Main function for real-time learning CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Learning System")
    parser.add_argument("--learning-mode", type=str,
                       choices=["online", "streaming", "incremental", "continuous"],
                       default="online", help="Learning mode")
    parser.add_argument("--data-source", type=str,
                       choices=["kafka", "redis", "websocket", "http_stream"],
                       default="kafka", help="Data source")
    parser.add_argument("--update-strategy", type=str,
                       choices=["adam", "sgd", "rmsprop"],
                       default="adam", help="Update strategy")
    parser.add_argument("--drift-detection", type=str,
                       choices=["statistical", "distribution_based", "concept_drift"],
                       default="statistical", help="Drift detection method")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--update-frequency", type=float, default=1.0,
                       help="Update frequency in seconds")
    parser.add_argument("--drift-threshold", type=float, default=0.1,
                       help="Drift threshold")
    parser.add_argument("--input-size", type=int, default=10,
                       help="Input size")
    parser.add_argument("--output-size", type=int, default=2,
                       help="Output size")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create real-time learning configuration
    config = RealTimeConfig(
        learning_mode=LearningMode(args.learning_mode),
        data_source=DataSource(args.data_source),
        update_strategy=UpdateStrategy(args.update_strategy),
        drift_detection=DriftDetectionMethod(args.drift_detection),
        learning_rate=args.learning_rate,
        update_frequency=args.update_frequency,
        drift_threshold=args.drift_threshold,
        device=args.device
    )
    
    # Create real-time learning system
    rt_system = RealTimeLearningSystem(config)
    
    # Start learning
    rt_system.start_learning(input_size=args.input_size, output_size=args.output_size)
    
    try:
        # Run for a while
        console.print("[blue]Real-time learning running... Press Ctrl+C to stop[/blue]")
        time.sleep(60)  # Run for 60 seconds
        
    except KeyboardInterrupt:
        console.print("[yellow]Stopping real-time learning...[/yellow]")
    
    finally:
        # Stop learning
        rt_system.stop_learning()
        
        # Show summary
        summary = rt_system.get_learning_summary()
        console.print(f"[green]Learning summary: {summary}[/green]")
        
        # Create visualization
        rt_system.visualize_learning_progress()

if __name__ == "__main__":
    main()
