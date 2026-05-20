"""
Constants for utilities.
Refactored to centralize all utility constants.
"""

# ════════════════════════════════════════════════════════════════════════════
# AUDIO FORMATS
# ════════════════════════════════════════════════════════════════════════════

SUPPORTED_AUDIO_FORMATS = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"
}

SUPPORTED_OUTPUT_FORMATS = {
    ".wav", ".mp3", ".flac", ".ogg"
}

# ════════════════════════════════════════════════════════════════════════════
# SAMPLE RATES
# ════════════════════════════════════════════════════════════════════════════

COMMON_SAMPLE_RATES = [8000, 11025, 16000, 22050, 44100, 48000, 96000]
DEFAULT_SAMPLE_RATE = 44100

# ════════════════════════════════════════════════════════════════════════════
# AUDIO PROCESSING
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_SILENCE_THRESHOLD = 0.01
DEFAULT_MIN_SILENCE_DURATION = 0.1
DEFAULT_TARGET_PEAK = 0.95
DEFAULT_TARGET_RMS = 0.1
DEFAULT_FADE_DURATION = 0.5
MAX_NUM_SOURCES = 10

# ════════════════════════════════════════════════════════════════════════════
# DEVICE
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_DEVICE = "auto"
VALID_DEVICES = ["auto", "cpu", "cuda", "mps"]

# ════════════════════════════════════════════════════════════════════════════
# CACHE
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_CACHE_DIR_NAME = ".audio_separator"
DEFAULT_CACHE_EXTENSION = ".pkl"

# ════════════════════════════════════════════════════════════════════════════
# ERROR CODES
# ════════════════════════════════════════════════════════════════════════════

ERROR_CODE_FILE_NOT_FOUND = "FILE_NOT_FOUND"
ERROR_CODE_NOT_A_FILE = "NOT_A_FILE"
ERROR_CODE_UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
ERROR_CODE_UNSUPPORTED_OUTPUT_FORMAT = "UNSUPPORTED_OUTPUT_FORMAT"
ERROR_CODE_INVALID_SAMPLE_RATE = "INVALID_SAMPLE_RATE"
ERROR_CODE_INVALID_NUM_SOURCES = "INVALID_NUM_SOURCES"
ERROR_CODE_INVALID_AUDIO_TYPE = "INVALID_AUDIO_TYPE"
ERROR_CODE_EMPTY_AUDIO = "EMPTY_AUDIO"
ERROR_CODE_INVALID_AUDIO_DIMENSIONS = "INVALID_AUDIO_DIMENSIONS"
ERROR_CODE_NAN_IN_AUDIO = "NAN_IN_AUDIO"
ERROR_CODE_INF_IN_AUDIO = "INF_IN_AUDIO"
ERROR_CODE_NOT_A_DIRECTORY = "NOT_A_DIRECTORY"
ERROR_CODE_DIRECTORY_CREATION_FAILED = "DIRECTORY_CREATION_FAILED"

