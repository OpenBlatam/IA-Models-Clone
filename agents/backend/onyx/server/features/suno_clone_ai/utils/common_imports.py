"""
Common imports for consistent usage across the codebase.

This module provides commonly used imports in a centralized location
to reduce duplication and ensure consistency.
"""

# Standard library
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path

# FastAPI
from fastapi import APIRouter, FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse, ORJSONResponse

# Pydantic
from pydantic import BaseModel, Field, validator

# Common patterns
logger = logging.getLogger(__name__)

__all__ = [
    # Standard library
    "logging",
    "Dict",
    "Any",
    "Optional",
    "List",
    "Union",
    "Callable",
    "Path",
    # FastAPI
    "APIRouter",
    "FastAPI",
    "HTTPException",
    "Depends",
    "status",
    "JSONResponse",
    "ORJSONResponse",
    # Pydantic
    "BaseModel",
    "Field",
    "validator",
    # Common
    "logger"
]

