"""
Model Constants
===============

Constants used throughout the character consistency model.
"""

# Default model configuration
DEFAULT_MODEL_ID = "black-forest-labs/flux2-dev"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_EMBEDDING_DIM = 768

# Pooling constants
POOLING_METHODS_COUNT = 3  # CLS, mean, attention
POOLING_COMBINE_WEIGHT = 1.0 / POOLING_METHODS_COUNT

# Aggregation constants
CROSS_ATTENTION_BLEND_WEIGHT = 0.3
FUSION_BASE_WEIGHT = 0.7
FUSION_METHODS_COUNT = 3  # mean, max, attention

# Architecture constants
MIN_INTERMEDIATE_SIZE_MULTIPLIER = 3
MIN_ATTENTION_HEADS = 8
ATTENTION_HEAD_DIM = 64

# Optimization constants
DROPOUT_RATE = 0.1
ATTENTION_SLICE_SIZE = 1

