"""
Application Constants

Centralized constants for the Lovable Community feature.
This module contains all magic numbers, default values, and configuration constants.
"""

# Ranking Algorithm Constants
DEFAULT_VOTE_WEIGHT = 2.0
DEFAULT_REMIX_WEIGHT = 3.0
DEFAULT_VIEW_WEIGHT = 0.1
MIN_AGE_HOURS = 0.1  # Minimum age in hours for score calculation
HOURS_PER_DAY = 24  # Hours in a day for time decay calculation
DEFAULT_TRENDING_HOURS = 24  # Default time window for trending scores
SCORE_DECIMAL_PLACES = 2  # Decimal places for score rounding

# Pagination Constants
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE = 1000
MAX_PAGE_SIZE = 100
MIN_PAGE = 1
MIN_PAGE_SIZE = 1

# Tag Constants
MAX_TAGS_PER_CHAT = 10  # Maximum number of tags allowed per chat

# Time Constants
SECONDS_PER_HOUR = 3600  # Seconds in an hour

# Validation Constants
MIN_CHAT_AGE_HOURS = 0.1  # Minimum age in hours for validation

# Schema Validation Constants
MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 1000
MAX_CHAT_CONTENT_LENGTH = 50000
MAX_TAG_LENGTH = 50  # Maximum length for individual tag
MIN_TITLE_LENGTH = 1
MIN_CHAT_CONTENT_LENGTH = 1

__all__ = [
    # Ranking
    "DEFAULT_VOTE_WEIGHT",
    "DEFAULT_REMIX_WEIGHT",
    "DEFAULT_VIEW_WEIGHT",
    "MIN_AGE_HOURS",
    "HOURS_PER_DAY",
    "DEFAULT_TRENDING_HOURS",
    "SCORE_DECIMAL_PLACES",
    # Pagination
    "DEFAULT_PAGE",
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE",
    "MAX_PAGE_SIZE",
    "MIN_PAGE",
    "MIN_PAGE_SIZE",
    # Tags
    "MAX_TAGS_PER_CHAT",
    # Time
    "SECONDS_PER_HOUR",
    # Validation
    "MIN_CHAT_AGE_HOURS",
    # Schema Validation
    "MAX_TITLE_LENGTH",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_CHAT_CONTENT_LENGTH",
    "MAX_TAG_LENGTH",
    "MIN_TITLE_LENGTH",
    "MIN_CHAT_CONTENT_LENGTH",
]

