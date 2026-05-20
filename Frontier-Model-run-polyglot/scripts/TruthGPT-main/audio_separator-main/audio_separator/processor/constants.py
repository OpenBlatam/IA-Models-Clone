"""
Constants for audio processors.
Refactored to centralize constants.
"""

# ════════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_NORMALIZE = True
DEFAULT_TRIM_SILENCE = False
DEFAULT_DENOISE = False

# ════════════════════════════════════════════════════════════════════════════
# ERROR CODES
# ════════════════════════════════════════════════════════════════════════════

# Loader errors
ERROR_CODE_FILE_NOT_FOUND = "FILE_NOT_FOUND"
ERROR_CODE_NOT_A_FILE = "NOT_A_FILE"
ERROR_CODE_LIBROSA_LOAD_FAILED = "LIBROSA_LOAD_FAILED"
ERROR_CODE_SOUNDFILE_LOAD_FAILED = "SOUNDFILE_LOAD_FAILED"
ERROR_CODE_SCIPY_LOAD_FAILED = "SCIPY_LOAD_FAILED"
ERROR_CODE_NO_AUDIO_LIBRARY = "NO_AUDIO_LIBRARY"
ERROR_CODE_AUDIO_LOAD_FAILED = "AUDIO_LOAD_FAILED"

# Saver errors
ERROR_CODE_SAVE_FAILED = "SAVE_FAILED"
ERROR_CODE_UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"

# Processor errors
ERROR_CODE_PREPROCESS_FAILED = "PREPROCESS_FAILED"
ERROR_CODE_POSTPROCESS_FAILED = "POSTPROCESS_FAILED"
ERROR_CODE_EMPTY_AUDIO = "EMPTY_AUDIO"
ERROR_CODE_INVALID_AUDIO_TYPE = "INVALID_AUDIO_TYPE"
ERROR_CODE_INVALID_DIMENSIONS = "INVALID_DIMENSIONS"
ERROR_CODE_NAN_IN_AUDIO = "NAN_IN_AUDIO"
ERROR_CODE_INF_IN_AUDIO = "INF_IN_AUDIO"
ERROR_CODE_EMPTY_SEPARATED = "EMPTY_SEPARATED"
ERROR_CODE_INVALID_TENSOR_TYPE = "INVALID_TENSOR_TYPE"

# ════════════════════════════════════════════════════════════════════════════
# AUDIO LIBRARIES
# ════════════════════════════════════════════════════════════════════════════

# Preferred library order for loading
AUDIO_LOAD_LIBRARIES = ["librosa", "soundfile", "scipy.io.wavfile"]

# Preferred library order for saving
AUDIO_SAVE_LIBRARIES = ["soundfile", "scipy.io.wavfile"]

# ════════════════════════════════════════════════════════════════════════════
# AUDIO FORMATS
# ════════════════════════════════════════════════════════════════════════════

SUPPORTED_LOAD_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
SUPPORTED_SAVE_FORMATS = {'.wav', '.flac', '.ogg'}

# Default format
DEFAULT_AUDIO_FORMAT = 'wav'

# Audio conversion constants
INT16_MAX = 32768.0
INT32_MAX = 2147483648.0
AUDIO_CLIP_MIN = -1.0
AUDIO_CLIP_MAX = 1.0

