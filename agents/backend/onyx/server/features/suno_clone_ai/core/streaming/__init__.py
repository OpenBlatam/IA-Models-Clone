"""
Streaming Module

Provides:
- Data streaming utilities
- Stream processing
- Streaming pipelines
"""

from .stream_processor import (
    StreamProcessor,
    process_stream,
    create_stream_pipeline
)

from .data_stream import (
    DataStream,
    create_data_stream
)

__all__ = [
    # Stream processing
    "StreamProcessor",
    "process_stream",
    "create_stream_pipeline",
    # Data stream
    "DataStream",
    "create_data_stream"
]



