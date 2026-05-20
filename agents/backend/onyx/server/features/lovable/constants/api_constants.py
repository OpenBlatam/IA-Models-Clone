"""
API Constants for Lovable Community SAM3.
"""

from typing import Dict

# Pagination constants
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MIN_PAGE_SIZE = 1

# Limit constants
DEFAULT_LIMIT = 50
MAX_LIMIT = 1000

# Trending periods (in hours)
TRENDING_PERIODS: Dict[str, float] = {
    "hour": 1.0,
    "day": 24.0,
    "week": 168.0,
    "month": 720.0
}

# Sort options
SORT_OPTIONS = {
    "score": "score",
    "created_at": "created_at",
    "vote_count": "vote_count",
    "relevance": "relevance",
    "trending": "trending"
}

# Share platforms
SHARE_PLATFORMS = [
    "twitter",
    "facebook",
    "linkedin",
    "whatsapp",
    "telegram",
    "copy_link",
    "other"
]

# Report types
REPORT_TYPES = [
    "spam",
    "harassment",
    "inappropriate",
    "copyright",
    "misinformation",
    "other"
]

# Report statuses
REPORT_STATUSES = [
    "pending",
    "reviewed",
    "resolved",
    "dismissed"
]

# Content types
CONTENT_TYPES = [
    "chat",
    "comment"
]

# Validation limits
MAX_TAG_LENGTH = 50
MAX_USER_ID_LENGTH = 100
MAX_CHAT_ID_LENGTH = 100
MAX_COMMENT_LENGTH = 2000
MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 1000

# Export limits
MAX_EXPORT_CHATS = 1000
MAX_EXPORT_COMMENTS = 1000
MAX_EXPORT_VOTES = 1000
MAX_EXPORT_BOOKMARKS = 1000

# Tag service limits
MAX_TAG_CHATS_LIMIT = 1000
DEFAULT_TAG_LIMIT = 50
DEFAULT_TRENDING_TAG_LIMIT = 20

# Chat service limits
MAX_RANKING_CHATS_LIMIT = 1000
MAX_REMIXES_LIMIT = 100
MAX_FOLLOWING_LIMIT = 1000






