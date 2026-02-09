"""
Type Aliases

Type aliases for better code readability.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Image types
ImageArray = np.ndarray
ImageBytes = bytes
ImagePath = str
ImageBase64 = str
ImageData = Union[ImageArray, ImageBytes, ImagePath, ImageBase64]

# Coordinates
Point = Tuple[int, int]
BoundingBox = Tuple[int, int, int, int]  # (x, y, width, height)
Resolution = Tuple[int, int]  # (width, height)

# Quality
QualityScore = float  # 0.0 to 100.0
Confidence = float  # 0.0 to 1.0
AnomalyScore = float  # 0.0 to 1.0

# Collections
DefectList = List[Any]  # List[Defect]
AnomalyList = List[Any]  # List[Anomaly]
InspectionList = List[Any]  # List[Inspection]

# Configuration
ConfigDict = Dict[str, Any]
MetadataDict = Dict[str, Any]

# API
RequestData = Dict[str, Any]
ResponseData = Dict[str, Any]

# Processing
BatchSize = int
Timeout = float



