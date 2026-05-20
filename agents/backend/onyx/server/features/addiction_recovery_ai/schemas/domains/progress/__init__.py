"""
Progress domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.progress import (
        LogEntryRequest,
        LogEntryResponse,
        ProgressResponse,
        StatsResponse,
        TimelineResponse
    )
    
    def register_schemas():
        register_schema("progress", "LogEntryRequest", LogEntryRequest)
        register_schema("progress", "LogEntryResponse", LogEntryResponse)
        register_schema("progress", "ProgressResponse", ProgressResponse)
        register_schema("progress", "StatsResponse", StatsResponse)
        register_schema("progress", "TimelineResponse", TimelineResponse)
except ImportError:
    pass



