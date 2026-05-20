"""
Constants
=========
"""

# Valid training goals
VALID_TRAINING_GOALS = [
    "obedience",
    "agility",
    "behavior",
    "socialization",
    "tricks",
    "house-training",
    "leash-training",
    "recall",
    "stay",
    "down",
    "heel"
]

# Valid experience levels
VALID_EXPERIENCE_LEVELS = ["beginner", "intermediate", "advanced"]

# Valid dog sizes
VALID_DOG_SIZES = ["small", "medium", "large", "giant"]

# Rate limits (requests per minute)
RATE_LIMITS = {
    "coach": 10,
    "training-plan": 5,
    "analyze-behavior": 10,
    "chat": 20,
    "default": 100  # per hour
}

