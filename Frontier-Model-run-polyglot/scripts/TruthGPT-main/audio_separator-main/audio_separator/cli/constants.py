"""
Constants for CLI.
Refactored to centralize CLI constants.
"""

# ════════════════════════════════════════════════════════════════════════════
# DEFAULT VALUES
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "demucs"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_EXTENSIONS = [".mp3", ".wav", ".flac"]

# ════════════════════════════════════════════════════════════════════════════
# VALID VALUES
# ════════════════════════════════════════════════════════════════════════════

VALID_MODELS = ["demucs", "spleeter", "lalal", "hybrid"]

# ════════════════════════════════════════════════════════════════════════════
# MESSAGES
# ════════════════════════════════════════════════════════════════════════════

MSG_FILE_NOT_FOUND = "Error: File not found: {path}"
MSG_SUCCESS_SEPARATED = "\nSuccessfully separated into {count} sources:"
MSG_NO_FILES_FOUND = "No audio files found in {dir}"
MSG_PROCESSING_COMPLETE = "Processing complete: {success}/{total} files processed successfully"

