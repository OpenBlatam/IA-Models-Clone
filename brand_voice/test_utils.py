"""
Test Utilities for Brand Voice AI System
========================================

This module provides utility functions and mock data for testing
the Brand Voice AI system components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

# Import configuration classes
from brand_ai_transformer import BrandTransformerConfig
from brand_ai_training import TrainingConfig
from brand_ai_serving import ServingConfig
from brand_ai_advanced_models import AdvancedModelsConfig
from brand_ai_optimization import OptimizationConfig
from brand_ai_deployment import DeploymentConfig
from brand_ai_computer_vision import ComputerVisionConfig
from brand_ai_monitoring import MonitoringConfig
from brand_ai_trend_prediction import TrendPredictionConfig
from brand_ai_multilingual import MultilingualConfig
from brand_ai_sentiment_analysis import SentimentConfig
from brand_ai_competitive_intelligence import CompetitiveIntelligenceConfig
from brand_ai_automation_system import AutomationConfig
from brand_ai_voice_cloning import VoiceCloningConfig
from brand_ai_collaboration_platform import CollaborationConfig
from brand_ai_performance_prediction import PerformancePredictionConfig
from brand_ai_blockchain_verification import BlockchainConfig
from brand_ai_crisis_management import CrisisManagementConfig

def create_mock_config(config_type: str = "default") -> Dict[str, Any]:
    """Create mock configuration for testing"""
    
    base_config = {
        # Common settings
        "device": "cpu",
        "batch_size": 2,  # Small for testing
        "learning_rate": 1e-4,
        "num_epochs": 1,  # Minimal for testing
        "max_sequence_length": 128,  # Shorter for testing
        "embedding_dim": 256,  # Smaller for testing
        "hidden_dim": 512,
        "num_attention_heads": 4,
        "num_layers": 2,
        "dropout_rate": 0.1,
        
        # Database settings
        "redis_url": "redis://localhost:6379",
        "sqlite_path": ":memory:",  # In-memory database for testing
        "postgres_url": "postgresql://test:test@localhost/test_db",
        
        # API settings
        "openai_api_key": "test_key",
        "huggingface_token": "test_token",
        
        # Experiment tracking
        "wandb_project": "test-project",
        "mlflow_tracking_uri": "http://localhost:5000",
        
        # Security
        "jwt_secret": "test_secret_key",
        "encryption_key": "test_encryption_key_32_bytes_long",
        
        # File paths
        "temp_dir": tempfile.mkdtemp(),
        "test_data_dir": tempfile.mkdtemp(),
        "model_cache_dir": tempfile.mkdtemp(),
        
        # Testing flags
        "test_mode": True,
        "mock_external_apis": True,
        "disable_gpu": True,
        "fast_mode": True
    }
    
    if config_type == "transformer":
        return {**base_config, **{
            "transformer_models": ["gpt2"],
            "diffusion_models": ["runwayml/stable-diffusion-v1-5"],
            "vision_models": ["openai/clip-vit-base-patch32"],
            "embedding_models": ["sentence-transformers/all-MiniLM-L6-v2"]
        }}
    
    elif config_type == "training":
        return {**base_config, **{
            "training_data_path": tempfile.mkdtemp(),
            "validation_data_path": tempfile.mkdtemp(),
            "model_save_path": tempfile.mkdtemp(),
            "checkpoint_interval": 1,
            "early_stopping_patience": 2
        }}
    
    elif config_type == "serving":
        return {**base_config, **{
            "api_host": "localhost",
            "api_port": 8000,
            "max_concurrent_requests": 10,
            "request_timeout": 30,
            "enable_gradio": True,
            "gradio_port": 7860
        }}
    
    elif config_type == "deployment":
        return {**base_config, **{
            "deployment_platform": "docker",
            "container_registry": "localhost:5000",
            "namespace": "test",
            "replicas": 1,
            "resource_limits": {
                "cpu": "100m",
                "memory": "256Mi"
            }
        }}
    
    elif config_type == "monitoring":
        return {**base_config, **{
            "prometheus_url": "http://localhost:9090",
            "grafana_url": "http://localhost:3000",
            "jaeger_url": "http://localhost:16686",
            "metrics_interval": 10,
            "alert_thresholds": {
                "cpu_usage": 80,
                "memory_usage": 85,
                "response_time": 1000
            }
        }}
    
    else:
        return base_config

def create_test_data(data_type: str = "text", size: int = 10) -> Union[List[str], List[Dict], np.ndarray]:
    """Create test data for different types"""
    
    if data_type == "text":
        return [
            "This is a test brand description for a technology company.",
            "Our company focuses on innovation and customer satisfaction.",
            "We provide cutting-edge solutions for modern businesses.",
            "Quality and reliability are our core values.",
            "Customer success is our top priority.",
            "We are committed to sustainable business practices.",
            "Innovation drives everything we do at our company.",
            "Our team consists of industry experts and professionals.",
            "We deliver exceptional results for our clients.",
            "Trust and transparency guide our business operations."
        ]
    
    elif data_type == "brand_data":
        return [
            {
                "brand_name": "TechCorp",
                "description": "Leading technology company",
                "industry": "Technology",
                "founded": "2010",
                "employees": 1000,
                "revenue": 100000000,
                "content": [
                    "We are a leading technology company",
                    "Innovation is at our core",
                    "Customer satisfaction is our priority"
                ]
            },
            {
                "brand_name": "EcoGreen",
                "description": "Sustainable energy solutions",
                "industry": "Energy",
                "founded": "2015",
                "employees": 500,
                "revenue": 50000000,
                "content": [
                    "We provide sustainable energy solutions",
                    "Environmental responsibility is key",
                    "Green technology for the future"
                ]
            }
        ]
    
    elif data_type == "sentiment_data":
        return [
            {"text": "I love this brand! Amazing products.", "sentiment": "positive"},
            {"text": "Terrible service, very disappointed.", "sentiment": "negative"},
            {"text": "The product is okay, nothing special.", "sentiment": "neutral"},
            {"text": "Excellent quality and fast delivery!", "sentiment": "positive"},
            {"text": "Worst experience ever, avoid this brand.", "sentiment": "negative"}
        ]
    
    elif data_type == "time_series":
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        values = np.random.normal(100, 10, len(dates))
        return pd.DataFrame({
            'date': dates,
            'value': values,
            'brand_id': 'test_brand'
        })
    
    elif data_type == "numerical":
        return np.random.normal(0, 1, (size, 10))
    
    else:
        return ["Test data item " + str(i) for i in range(size)]

def create_test_images(count: int = 3, size: Tuple[int, int] = (224, 224)) -> List[str]:
    """Create test images for computer vision testing"""
    image_paths = []
    
    for i in range(count):
        # Create a simple test image
        img = Image.new('RGB', size, color=(i * 80, 100, 150))
        draw = ImageDraw.Draw(img)
        
        # Add some simple shapes
        draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=(255, 255, 255), width=2)
        draw.text((size[0]//2, size[1]//2), f"Test {i+1}", fill=(255, 255, 255))
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix='.png')
        img.save(temp_path)
        image_paths.append(temp_path)
    
    return image_paths

def create_test_audio(duration: float = 2.0, sample_rate: int = 22050) -> str:
    """Create test audio file for voice cloning testing"""
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(audio_data))
    audio_data = audio_data + noise
    
    # Save to temporary file
    temp_path = tempfile.mktemp(suffix='.wav')
    sf.write(temp_path, audio_data, sample_rate)
    
    return temp_path

def create_mock_models(model_type: str = "transformer") -> Dict[str, Any]:
    """Create mock models for testing"""
    
    if model_type == "transformer":
        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 256)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(256, 4, 512, 0.1),
                    num_layers=2
                )
                self.classifier = nn.Linear(256, 2)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                x = x.mean(dim=1)
                return self.classifier(x)
        
        return {
            'model': MockTransformer(),
            'tokenizer': Mock(),
            'config': {'vocab_size': 1000, 'hidden_size': 256}
        }
    
    elif model_type == "vision":
        class MockVisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, 1, 1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Linear(64, 10)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return {
            'model': MockVisionModel(),
            'processor': Mock(),
            'config': {'num_classes': 10}
        }
    
    elif model_type == "audio":
        class MockAudioModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1d = nn.Conv1d(1, 64, 3, 1, 1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Linear(64, 5)
            
            def forward(self, x):
                x = self.conv1d(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return {
            'model': MockAudioModel(),
            'processor': Mock(),
            'config': {'num_classes': 5}
        }
    
    else:
        return {
            'model': nn.Linear(10, 1),
            'config': {}
        }

def create_mock_database() -> Mock:
    """Create mock database for testing"""
    mock_db = Mock()
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_db.add.return_value = None
    mock_db.commit.return_value = None
    mock_db.rollback.return_value = None
    return mock_db

def create_mock_redis() -> Mock:
    """Create mock Redis client for testing"""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    return mock_redis

def create_mock_web3() -> Mock:
    """Create mock Web3 client for testing"""
    mock_web3 = Mock()
    mock_web3.eth.get_transaction_count.return_value = 0
    mock_web3.eth.gas_price = 20000000000
    mock_web3.eth.send_raw_transaction.return_value = b'tx_hash'
    mock_web3.eth.wait_for_transaction_receipt.return_value = Mock(contractAddress='0x123')
    return mock_web3

def create_mock_http_client() -> Mock:
    """Create mock HTTP client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text = "Success"
    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response
    return mock_client

def create_mock_ai_models() -> Dict[str, Mock]:
    """Create mock AI models for testing"""
    return {
        'transformer': Mock(),
        'vision': Mock(),
        'audio': Mock(),
        'sentiment': Mock(),
        'classification': Mock()
    }

def create_test_workflow_definition() -> Dict[str, Any]:
    """Create test workflow definition"""
    return {
        "workflow_id": "test_workflow_001",
        "name": "Test Brand Analysis Workflow",
        "description": "A test workflow for brand analysis",
        "workflow_type": "brand_analysis",
        "trigger_type": "manual",
        "trigger_config": {},
        "tasks": [
            {
                "task_id": "analyze_content",
                "task_type": "content_analysis",
                "parameters": {
                    "content_type": "brand_description",
                    "analysis_depth": "comprehensive"
                },
                "dependencies": []
            },
            {
                "task_id": "generate_report",
                "task_type": "report_generation",
                "parameters": {
                    "report_type": "brand_analysis",
                    "format": "pdf"
                },
                "dependencies": ["analyze_content"]
            }
        ],
        "settings": {
            "timeout": 300,
            "retry_attempts": 3,
            "notifications": True
        }
    }

def create_test_crisis_data() -> Dict[str, Any]:
    """Create test crisis data"""
    return {
        "crisis_id": "crisis_001",
        "brand_id": "test_brand",
        "crisis_type": "reputation_damage",
        "severity": "high",
        "mentions": [
            {
                "text": "This brand is terrible! Avoid at all costs!",
                "timestamp": datetime.now(),
                "source": "twitter",
                "author_type": "customer",
                "sentiment": -0.8,
                "reach": 1000
            },
            {
                "text": "Worst experience ever with this company",
                "timestamp": datetime.now(),
                "source": "facebook",
                "author_type": "customer",
                "sentiment": -0.9,
                "reach": 500
            }
        ],
        "metrics": {
            "sentiment_score": -0.85,
            "volume_score": 1500,
            "velocity_score": 2.5,
            "severity_score": 0.8
        }
    }

def create_test_verification_data() -> Dict[str, Any]:
    """Create test verification data"""
    return {
        "verification_id": "verify_001",
        "brand_id": "test_brand",
        "verification_type": "brand_identity",
        "verification_data": {
            "brand_name": "TestBrand",
            "trademark": "TestBrand Logo",
            "description": "Test brand verification",
            "documents": ["trademark_certificate.pdf", "business_license.pdf"]
        },
        "blockchain_hash": "0x1234567890abcdef",
        "smart_contract_address": "0xabcdef1234567890",
        "status": "verified",
        "created_at": datetime.now(),
        "verified_at": datetime.now()
    }

def create_test_performance_data() -> Dict[str, Any]:
    """Create test performance data"""
    return {
        "brand_id": "test_brand",
        "metrics": {
            "revenue": 1000000,
            "market_share": 0.15,
            "brand_awareness": 0.65,
            "customer_satisfaction": 0.8,
            "social_media_engagement": 0.45,
            "website_traffic": 50000,
            "conversion_rate": 0.03,
            "customer_lifetime_value": 2500
        },
        "historical_data": pd.DataFrame({
            'date': pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D'),
            'revenue': np.random.normal(1000000, 100000, 30),
            'engagement': np.random.normal(0.45, 0.05, 30),
            'sentiment': np.random.normal(0.7, 0.1, 30)
        }),
        "predictions": {
            "revenue_30_days": 1100000,
            "market_share_30_days": 0.16,
            "brand_awareness_30_days": 0.68
        }
    }

def create_test_collaboration_data() -> Dict[str, Any]:
    """Create test collaboration data"""
    return {
        "session_id": "session_001",
        "name": "Test Collaboration Session",
        "description": "Testing collaboration features",
        "session_type": "brainstorming",
        "participants": [
            {
                "user_id": "user_001",
                "username": "testuser1",
                "role": "creator",
                "is_online": True
            },
            {
                "user_id": "user_002",
                "username": "testuser2",
                "role": "collaborator",
                "is_online": True
            }
        ],
        "ai_assistants": [
            {
                "assistant_id": "strategy_assistant",
                "name": "Brand Strategy AI",
                "role": "Strategic Planning",
                "is_active": True
            }
        ],
        "content": [
            {
                "content_id": "content_001",
                "type": "text",
                "title": "Brand Strategy Ideas",
                "content": "Here are some ideas for our brand strategy...",
                "creator": "user_001",
                "created_at": datetime.now()
            }
        ]
    }

def cleanup_test_files(file_paths: List[str]):
    """Clean up test files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove test file {file_path}: {e}")

def cleanup_test_directories(dir_paths: List[str]):
    """Clean up test directories"""
    import shutil
    for dir_path in dir_paths:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Warning: Could not remove test directory {dir_path}: {e}")

def assert_model_output_valid(output: Any, expected_keys: List[str] = None) -> bool:
    """Assert that model output is valid"""
    if output is None:
        return False
    
    if expected_keys:
        if isinstance(output, dict):
            for key in expected_keys:
                if key not in output:
                    return False
        else:
            # For non-dict outputs, check if it has the expected attributes
            for key in expected_keys:
                if not hasattr(output, key):
                    return False
    
    return True

def assert_performance_acceptable(execution_time: float, max_time: float = 10.0) -> bool:
    """Assert that performance is acceptable"""
    return execution_time <= max_time

def assert_memory_usage_acceptable(memory_usage: float, max_memory: float = 1024.0) -> bool:
    """Assert that memory usage is acceptable (in MB)"""
    return memory_usage <= max_memory

def create_mock_external_api_responses() -> Dict[str, Any]:
    """Create mock responses for external APIs"""
    return {
        "twitter": {
            "status": "success",
            "data": [
                {
                    "id": "1234567890",
                    "text": "Great product from @TestBrand!",
                    "created_at": datetime.now().isoformat(),
                    "user": {
                        "id": "user123",
                        "username": "testuser",
                        "followers_count": 1000
                    }
                }
            ]
        },
        "facebook": {
            "status": "success",
            "data": [
                {
                    "id": "post123",
                    "message": "Love this brand!",
                    "created_time": datetime.now().isoformat(),
                    "from": {
                        "id": "user456",
                        "name": "Test User"
                    }
                }
            ]
        },
        "news": {
            "status": "success",
            "articles": [
                {
                    "title": "TestBrand announces new product",
                    "url": "https://example.com/article1",
                    "publishedAt": datetime.now().isoformat(),
                    "source": {
                        "name": "Tech News"
                    }
                }
            ]
        }
    }

def create_test_benchmark_data() -> Dict[str, Any]:
    """Create test benchmark data for performance testing"""
    return {
        "text_analysis": {
            "input_sizes": [100, 500, 1000, 5000],
            "expected_times": [0.1, 0.5, 1.0, 5.0],
            "expected_memory": [50, 100, 200, 500]
        },
        "image_analysis": {
            "input_sizes": [(224, 224), (512, 512), (1024, 1024)],
            "expected_times": [0.2, 0.8, 3.0],
            "expected_memory": [100, 200, 400]
        },
        "audio_analysis": {
            "input_durations": [1.0, 5.0, 10.0, 30.0],
            "expected_times": [0.3, 1.5, 3.0, 9.0],
            "expected_memory": [80, 150, 300, 600]
        }
    }

# Test data generators for specific modules
def generate_brand_analysis_test_data() -> Dict[str, Any]:
    """Generate test data for brand analysis"""
    return {
        "brand_name": "TestBrand",
        "industry": "Technology",
        "content": create_test_data("text", 5),
        "images": create_test_images(2),
        "social_media_data": create_mock_external_api_responses(),
        "competitor_data": [
            {
                "name": "Competitor1",
                "market_share": 0.25,
                "strengths": ["Innovation", "Brand recognition"],
                "weaknesses": ["High prices", "Limited reach"]
            }
        ],
        "market_data": {
            "market_size": 1000000000,
            "growth_rate": 0.05,
            "trends": ["AI adoption", "Sustainability", "Digital transformation"]
        }
    }

def generate_training_test_data() -> Dict[str, Any]:
    """Generate test data for training"""
    return {
        "training_data": create_test_data("brand_data", 10),
        "validation_data": create_test_data("brand_data", 5),
        "test_data": create_test_data("brand_data", 3),
        "labels": ["positive", "negative", "neutral"] * 6,
        "features": create_test_data("numerical", 18),
        "metadata": {
            "data_source": "synthetic",
            "preprocessing": "normalized",
            "augmentation": "none"
        }
    }

def generate_deployment_test_data() -> Dict[str, Any]:
    """Generate test data for deployment"""
    return {
        "service_config": {
            "name": "brand-ai-service",
            "image": "brand-ai:latest",
            "replicas": 3,
            "resources": {
                "requests": {"cpu": "100m", "memory": "256Mi"},
                "limits": {"cpu": "500m", "memory": "1Gi"}
            }
        },
        "ingress_config": {
            "host": "brand-ai.example.com",
            "tls": True,
            "certificate": "brand-ai-tls"
        },
        "monitoring_config": {
            "enabled": True,
            "metrics_path": "/metrics",
            "scrape_interval": "30s"
        }
    }

# Utility functions for test assertions
def assert_brand_analysis_result(result: Any) -> bool:
    """Assert that brand analysis result is valid"""
    required_fields = ['sentiment', 'keywords', 'confidence', 'recommendations']
    return assert_model_output_valid(result, required_fields)

def assert_training_result(result: Any) -> bool:
    """Assert that training result is valid"""
    required_fields = ['model', 'metrics', 'training_time', 'validation_score']
    return assert_model_output_valid(result, required_fields)

def assert_deployment_result(result: Any) -> bool:
    """Assert that deployment result is valid"""
    required_fields = ['status', 'endpoints', 'health_check', 'logs']
    return assert_model_output_valid(result, required_fields)

def assert_monitoring_result(result: Any) -> bool:
    """Assert that monitoring result is valid"""
    required_fields = ['metrics', 'alerts', 'health_status', 'performance']
    return assert_model_output_valid(result, required_fields)

# Test environment setup
def setup_test_environment():
    """Set up test environment"""
    # Create temporary directories
    temp_dirs = [
        tempfile.mkdtemp(prefix="brand_ai_test_"),
        tempfile.mkdtemp(prefix="brand_ai_models_"),
        tempfile.mkdtemp(prefix="brand_ai_data_")
    ]
    
    # Set environment variables for testing
    os.environ['BRAND_AI_TEST_MODE'] = 'true'
    os.environ['BRAND_AI_MOCK_APIS'] = 'true'
    os.environ['BRAND_AI_DISABLE_GPU'] = 'true'
    
    return temp_dirs

def teardown_test_environment(temp_dirs: List[str]):
    """Tear down test environment"""
    # Clean up temporary directories
    cleanup_test_directories(temp_dirs)
    
    # Remove test environment variables
    test_env_vars = ['BRAND_AI_TEST_MODE', 'BRAND_AI_MOCK_APIS', 'BRAND_AI_DISABLE_GPU']
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]

# Mock classes for testing
class MockModel:
    """Mock model class for testing"""
    def __init__(self, output_shape=(1, 10)):
        self.output_shape = output_shape
        self.trained = False
    
    def forward(self, x):
        return torch.randn(self.output_shape)
    
    def train(self):
        self.trained = True
    
    def eval(self):
        pass

class MockTokenizer:
    """Mock tokenizer class for testing"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
    
    def encode(self, text, **kwargs):
        return [1, 2, 3, 4, 5]  # Mock token IDs
    
    def decode(self, token_ids, **kwargs):
        return "Mock decoded text"
    
    def __call__(self, text, **kwargs):
        return {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }

class MockDataset:
    """Mock dataset class for testing"""
    def __init__(self, size=100):
        self.size = size
        self.data = [f"Sample {i}" for i in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'text': self.data[idx],
            'label': idx % 3,
            'id': idx
        }

# Performance testing utilities
def measure_execution_time(func):
    """Decorator to measure execution time"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {execution_time:.2f} seconds")
        return result, execution_time
    
    return wrapper

def measure_memory_usage(func):
    """Decorator to measure memory usage"""
    import psutil
    import os
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory
        
        print(f"Memory usage for {func.__name__}: {memory_usage:.2f} MB")
        return result, memory_usage
    
    return wrapper

# Test data validation
def validate_test_data(data: Any, data_type: str) -> bool:
    """Validate test data structure"""
    if data_type == "text" and isinstance(data, list):
        return all(isinstance(item, str) for item in data)
    elif data_type == "brand_data" and isinstance(data, list):
        required_fields = ['brand_name', 'description', 'content']
        return all(all(field in item for field in required_fields) for item in data)
    elif data_type == "images" and isinstance(data, list):
        return all(isinstance(item, str) and os.path.exists(item) for item in data)
    elif data_type == "audio" and isinstance(data, str):
        return os.path.exists(data)
    else:
        return False

# Export all utility functions
__all__ = [
    'create_mock_config',
    'create_test_data',
    'create_test_images',
    'create_test_audio',
    'create_mock_models',
    'create_mock_database',
    'create_mock_redis',
    'create_mock_web3',
    'create_mock_http_client',
    'create_mock_ai_models',
    'create_test_workflow_definition',
    'create_test_crisis_data',
    'create_test_verification_data',
    'create_test_performance_data',
    'create_test_collaboration_data',
    'cleanup_test_files',
    'cleanup_test_directories',
    'assert_model_output_valid',
    'assert_performance_acceptable',
    'assert_memory_usage_acceptable',
    'create_mock_external_api_responses',
    'create_test_benchmark_data',
    'generate_brand_analysis_test_data',
    'generate_training_test_data',
    'generate_deployment_test_data',
    'assert_brand_analysis_result',
    'assert_training_result',
    'assert_deployment_result',
    'assert_monitoring_result',
    'setup_test_environment',
    'teardown_test_environment',
    'MockModel',
    'MockTokenizer',
    'MockDataset',
    'measure_execution_time',
    'measure_memory_usage',
    'validate_test_data'
]
























