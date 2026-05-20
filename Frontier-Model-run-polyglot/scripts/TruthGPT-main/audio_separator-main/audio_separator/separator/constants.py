"""
Constants for audio separators.
Refactored to centralize constants and improve maintainability.
"""

# ════════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_MODEL_TYPE = "demucs"
DEFAULT_OUTPUT_FORMAT = "wav"

# ════════════════════════════════════════════════════════════════════════════
# VALID VALUES
# ════════════════════════════════════════════════════════════════════════════

VALID_MODEL_TYPES = ["demucs", "spleeter", "lalal", "hybrid"]
VALID_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000, 96000]
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
SUPPORTED_OUTPUT_FORMATS = {'wav', 'mp3', 'flac', 'ogg'}

# ════════════════════════════════════════════════════════════════════════════
# ERROR CODES
# ════════════════════════════════════════════════════════════════════════════

# File errors
ERROR_CODE_FILE_NOT_FOUND = "FILE_NOT_FOUND"
ERROR_CODE_NOT_A_FILE = "NOT_A_FILE"
ERROR_CODE_INVALID_OUTPUT_DIR = "INVALID_OUTPUT_DIR"
ERROR_CODE_SAVE_FAILED = "SAVE_FAILED"

# Validation errors
ERROR_CODE_INVALID_SAMPLE_RATE = "INVALID_SAMPLE_RATE"
ERROR_CODE_INVALID_SAMPLE_RATE_TYPE = "INVALID_SAMPLE_RATE_TYPE"
ERROR_CODE_INVALID_MODEL_TYPE = "INVALID_MODEL_TYPE"
ERROR_CODE_INVALID_AUDIO_TYPE = "INVALID_AUDIO_TYPE"
ERROR_CODE_EMPTY_AUDIO = "EMPTY_AUDIO"

# Processing errors
ERROR_CODE_SEPARATION_FAILED = "SEPARATION_FAILED"
ERROR_CODE_SEPARATION_PIPELINE_FAILED = "SEPARATION_PIPELINE_FAILED"
ERROR_CODE_PREPROCESS_FAILED = "PREPROCESS_FAILED"
ERROR_CODE_POSTPROCESS_FAILED = "POSTPROCESS_FAILED"

# Initialization errors
ERROR_CODE_INIT_FAILED = "INIT_FAILED"

# ════════════════════════════════════════════════════════════════════════════
# SEPARATION SOURCES
# ════════════════════════════════════════════════════════════════════════════

# Common source names
SOURCE_VOCALS = "vocals"
SOURCE_DRUMS = "drums"
SOURCE_BASS = "bass"
SOURCE_OTHER = "other"
SOURCE_PIANO = "piano"
SOURCE_GUITAR = "guitar"

# Default source names for 4-stem separation
DEFAULT_4_STEM_SOURCES = [SOURCE_VOCALS, SOURCE_DRUMS, SOURCE_BASS, SOURCE_OTHER]

# ════════════════════════════════════════════════════════════════════════════
# FILE NAMING
# ════════════════════════════════════════════════════════════════════════════

OUTPUT_FILENAME_SEPARATOR = "_"
DEFAULT_OUTPUT_DIR_NAME = "separated"

