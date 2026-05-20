from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .console_reporter import generate_console_report
from .html_reporter import generate_html_report
from .json_reporter import generate_json_report
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Reporting module for cybersecurity assessment results.
Supports console, HTML, and JSON output formats.
"""


__all__ = [
    "generate_console_report",
    "generate_html_report", 
    "generate_json_report",
] 