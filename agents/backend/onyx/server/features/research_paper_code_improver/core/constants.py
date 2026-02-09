"""
Constants - Constantes compartidas para módulos de deep learning
==================================================================
"""

from enum import Enum
from typing import Dict, Any

# Device defaults
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "float32"

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_WORKERS = 4

# Optimization defaults
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_EPS = 1e-8

# Scheduler defaults
DEFAULT_STEP_SIZE = 10
DEFAULT_GAMMA = 0.1
DEFAULT_T_MAX = 10

# Early stopping defaults
DEFAULT_PATIENCE = 10
DEFAULT_MIN_DELTA = 0.0

# Gradient clipping defaults
DEFAULT_MAX_GRAD_NORM = 1.0

# Mixed precision defaults
DEFAULT_INIT_SCALE = 2.0 ** 16
DEFAULT_GROWTH_FACTOR = 2.0
DEFAULT_BACKOFF_FACTOR = 0.5

# Model defaults
DEFAULT_DROPOUT = 0.1
DEFAULT_HIDDEN_SIZE = 768

# Evaluation defaults
DEFAULT_METRIC_NAMES = ["accuracy", "loss", "f1_score", "precision", "recall"]

# File paths
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_LOG_DIR = "./logs"
DEFAULT_MODEL_DIR = "./models"
DEFAULT_DATA_DIR = "./data"

# Performance thresholds
LATENCY_THRESHOLD_MS = 100.0
MEMORY_THRESHOLD_MB = 1000.0
ACCURACY_THRESHOLD = 0.8

# Cache defaults
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 3600  # seconds

# Batch defaults
DEFAULT_MAX_BATCH_SIZE = 128
DEFAULT_PREFETCH_FACTOR = 2

# Reproducibility
DEFAULT_SEED = 42

# Cost defaults (USD)
TRAINING_COST_PER_GPU_HOUR = 2.0
INFERENCE_COST_PER_1K = 0.01
STORAGE_COST_PER_GB_MONTH = 0.023




