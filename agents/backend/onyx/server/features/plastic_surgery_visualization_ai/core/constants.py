"""Application constants."""

from api.schemas.visualization import SurgeryType

# Surgery type metadata
SURGERY_TYPES_METADATA = {
    SurgeryType.RHINOPLASTY: {
        "name": "Rhinoplasty",
        "description": "Nose reshaping surgery",
        "areas": ["nose"]
    },
    SurgeryType.FACELIFT: {
        "name": "Facelift",
        "description": "Facial rejuvenation surgery",
        "areas": ["cheeks", "jawline", "neck"]
    },
    SurgeryType.BLEPHAROPLASTY: {
        "name": "Blepharoplasty",
        "description": "Eyelid surgery",
        "areas": ["upper_eyelids", "lower_eyelids"]
    },
    SurgeryType.LIPOSUCTION: {
        "name": "Liposuction",
        "description": "Fat removal surgery",
        "areas": ["body"]
    },
    SurgeryType.BREAST_AUGMENTATION: {
        "name": "Breast Augmentation",
        "description": "Breast enhancement surgery",
        "areas": ["chest"]
    },
    SurgeryType.CHIN_AUGMENTATION: {
        "name": "Chin Augmentation",
        "description": "Chin enhancement surgery",
        "areas": ["chin"]
    }
}

# API version
API_VERSION = "1.0.0"

# Service name
SERVICE_NAME = "Plastic Surgery Visualization AI"

# Default values
DEFAULT_INTENSITY = 0.5
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_RATE_LIMIT_PER_MINUTE = 60

# Image constraints
MIN_IMAGE_DIMENSION = 100
MAX_IMAGE_DIMENSION = 10000

