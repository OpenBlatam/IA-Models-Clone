"""
Constants for Clothing Changer Model
====================================
"""

# Model IDs
DEFAULT_MODEL_ID = "black-forest-labs/flux2-dev"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

# Embedding dimensions
CHARACTER_EMBEDDING_DIM = 768
CLOTHING_EMBEDDING_DIM = 512
COMBINED_EMBEDDING_DIM = CHARACTER_EMBEDDING_DIM + CLOTHING_EMBEDDING_DIM

# Architecture constants
MIN_INTERMEDIATE_SIZE_MULTIPLIER = 3
MIN_ATTENTION_HEADS = 8
ATTENTION_HEAD_DIM = 64
DROPOUT_RATE = 0.1

# Pooling weights
CLS_POOLING_WEIGHT = 0.3
MEAN_POOLING_WEIGHT = 0.2
MAX_POOLING_WEIGHT = 0.2
ATTN_POOLING_WEIGHT = 0.3

# Default generation parameters
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_STRENGTH = 0.8
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, bad anatomy"

# Mask detection
MASK_DETECTION_THRESHOLD = 0.4
CLOTHING_REGION_RATIO = 0.6  # Lower 60% of image for clothing

# Image processing
MAX_IMAGE_SIZE = 1024
MIN_IMAGE_SIZE = 256

# ComfyUI
COMFYUI_VERSION = "1.0.0"
TENSOR_FORMAT = "safetensors"


