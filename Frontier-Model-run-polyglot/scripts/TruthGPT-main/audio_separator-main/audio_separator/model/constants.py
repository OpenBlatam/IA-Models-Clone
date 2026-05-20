"""
Constants for audio separator models.
Refactored to centralize constants.
"""

# ════════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_NUM_SOURCES = 4
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512

# ════════════════════════════════════════════════════════════════════════════
# VALID MODEL TYPES
# ════════════════════════════════════════════════════════════════════════════

VALID_MODEL_TYPES = ["demucs", "spleeter", "lalal", "hybrid"]
DEFAULT_MODEL_TYPE = "demucs"

# ════════════════════════════════════════════════════════════════════════════
# ERROR CODES
# ════════════════════════════════════════════════════════════════════════════

ERROR_CODE_INVALID_MODEL_TYPE = "INVALID_MODEL_TYPE"
ERROR_CODE_UNEXPECTED_MODEL_TYPE = "UNEXPECTED_MODEL_TYPE"
ERROR_CODE_MODEL_BUILD_FAILED = "MODEL_BUILD_FAILED"
ERROR_CODE_INVALID_NUM_SOURCES = "INVALID_NUM_SOURCES"
ERROR_CODE_INVALID_SAMPLE_RATE = "INVALID_SAMPLE_RATE"
ERROR_CODE_INVALID_N_FFT = "INVALID_N_FFT"
ERROR_CODE_INVALID_HOP_LENGTH = "INVALID_HOP_LENGTH"
ERROR_CODE_EMPTY_AUDIO = "EMPTY_AUDIO"
ERROR_CODE_INVALID_AUDIO_TYPE = "INVALID_AUDIO_TYPE"
ERROR_CODE_INVALID_AUDIO_DIMENSIONS = "INVALID_AUDIO_DIMENSIONS"
ERROR_CODE_EMPTY_SEPARATED = "EMPTY_SEPARATED"
ERROR_CODE_INVALID_TENSOR_TYPE = "INVALID_TENSOR_TYPE"

