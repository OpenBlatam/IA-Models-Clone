from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .console_reporter import (
from .html_reporter import (
from .json_reporter import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Reporting module for cybersecurity testing results.

This module provides tools for:
- Console reporting (terminal output)
- HTML reporting (web-based reports)
- JSON reporting (structured data export)

All tools follow cybersecurity principles:
- Functional programming approach
- Async for I/O operations, def for CPU operations
- Type hints and Pydantic validation
- RORO pattern for tool interfaces
"""

    ConsoleReporter,
    ConsoleReportConfig,
    ConsoleReportResult
)

    HTMLReporter,
    HTMLReportConfig,
    HTMLReportResult
)

    JSONReporter,
    JSONReportConfig,
    JSONReportResult
)

__all__ = [
    # Console Reporting
    'ConsoleReporter',
    'ConsoleReportConfig',
    'ConsoleReportResult',
    
    # HTML Reporting
    'HTMLReporter',
    'HTMLReportConfig',
    'HTMLReportResult',
    
    # JSON Reporting
    'JSONReporter',
    'JSONReportConfig',
    'JSONReportResult'
] 