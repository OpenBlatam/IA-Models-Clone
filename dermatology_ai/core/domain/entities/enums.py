from enum import Enum


class SkinType(str, Enum):
    DRY = "dry"
    OILY = "oily"
    COMBINATION = "combination"
    SENSITIVE = "sensitive"
    NORMAL = "normal"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"















