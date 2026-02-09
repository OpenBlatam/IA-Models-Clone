"""
Test Helpers

Utility functions for testing.
"""

import numpy as np
from typing import Optional, Tuple
from datetime import datetime

from ..domain import ImageMetadata, QualityScore, Inspection
from ..domain.entities import Defect, DefectType, DefectSeverity, DefectLocation
from ..domain.entities import Anomaly, AnomalyType, AnomalySeverity, AnomalyLocation


def create_test_image(
    width: int = 640,
    height: int = 480,
    channels: int = 3,
    dtype: type = np.uint8
) -> np.ndarray:
    """
    Create a test image.
    
    Args:
        width: Image width
        height: Image height
        channels: Number of channels
        dtype: Data type
    
    Returns:
        Test image array
    """
    if channels == 1:
        shape = (height, width)
    else:
        shape = (height, width, channels)
    
    return np.random.randint(0, 255, shape, dtype=dtype)


def create_test_image_metadata(
    width: int = 640,
    height: int = 480,
    channels: int = 3
) -> ImageMetadata:
    """
    Create test image metadata.
    
    Args:
        width: Image width
        height: Image height
        channels: Number of channels
    
    Returns:
        ImageMetadata instance
    """
    return ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        source="test",
    )


def create_test_quality_score(
    score: float = 85.0,
    defects_count: int = 0,
    anomalies_count: int = 0
) -> QualityScore:
    """
    Create test quality score.
    
    Args:
        score: Quality score
        defects_count: Number of defects
        anomalies_count: Number of anomalies
    
    Returns:
        QualityScore instance
    """
    return QualityScore(
        score=score,
        defects_count=defects_count,
        anomalies_count=anomalies_count,
    )


def create_test_defect(
    defect_type: DefectType = DefectType.SCRATCH,
    severity: DefectSeverity = DefectSeverity.MINOR,
    x: int = 100,
    y: int = 100,
    width: int = 50,
    height: int = 50,
    confidence: float = 0.8
) -> Defect:
    """
    Create test defect.
    
    Args:
        defect_type: Type of defect
        severity: Severity level
        x: X coordinate
        y: Y coordinate
        width: Width
        height: Height
        confidence: Confidence score
    
    Returns:
        Defect instance
    """
    import uuid
    
    location = DefectLocation(x=x, y=y, width=width, height=height)
    
    return Defect(
        id=str(uuid.uuid4()),
        type=defect_type,
        severity=severity,
        location=location,
        confidence=confidence,
    )


def create_test_anomaly(
    anomaly_type: AnomalyType = AnomalyType.STATISTICAL,
    severity: AnomalySeverity = AnomalySeverity.MEDIUM,
    x: int = 200,
    y: int = 200,
    width: int = 100,
    height: int = 100,
    score: float = 0.7
) -> Anomaly:
    """
    Create test anomaly.
    
    Args:
        anomaly_type: Type of anomaly
        severity: Severity level
        x: X coordinate
        y: Y coordinate
        width: Width
        height: Height
        score: Anomaly score
    
    Returns:
        Anomaly instance
    """
    import uuid
    
    location = AnomalyLocation(x=x, y=y, width=width, height=height)
    
    return Anomaly(
        id=str(uuid.uuid4()),
        type=anomaly_type,
        severity=severity,
        location=location,
        score=score,
    )


def create_test_inspection(
    image_width: int = 640,
    image_height: int = 480,
    quality_score: float = 85.0,
    num_defects: int = 0,
    num_anomalies: int = 0
) -> Inspection:
    """
    Create test inspection.
    
    Args:
        image_width: Image width
        image_height: Image height
        quality_score: Quality score
        num_defects: Number of defects
        num_anomalies: Number of anomalies
    
    Returns:
        Inspection instance
    """
    from ..domain.services import InspectionService
    
    # Create image metadata
    image_metadata = create_test_image_metadata(image_width, image_height)
    
    # Create inspection service
    service = InspectionService()
    
    # Create inspection
    inspection = service.create_inspection(image_metadata)
    
    # Add defects
    for i in range(num_defects):
        defect = create_test_defect(
            x=50 + i * 100,
            y=50 + i * 100,
        )
        inspection = service.add_defect_to_inspection(inspection, defect)
    
    # Add anomalies
    for i in range(num_anomalies):
        anomaly = create_test_anomaly(
            x=100 + i * 150,
            y=100 + i * 150,
        )
        inspection = service.add_anomaly_to_inspection(inspection, anomaly)
    
    # Complete inspection
    inspection = service.complete_inspection(inspection)
    
    return inspection


def assert_quality_score_valid(quality_score: QualityScore):
    """
    Assert quality score is valid.
    
    Args:
        quality_score: Quality score to validate
    
    Raises:
        AssertionError: If invalid
    """
    assert 0.0 <= quality_score.score <= 100.0, \
        f"Quality score must be between 0 and 100, got {quality_score.score}"
    assert quality_score.defects_count >= 0, \
        f"Defects count cannot be negative, got {quality_score.defects_count}"
    assert quality_score.anomalies_count >= 0, \
        f"Anomalies count cannot be negative, got {quality_score.anomalies_count}"


def assert_inspection_valid(inspection: Inspection):
    """
    Assert inspection is valid.
    
    Args:
        inspection: Inspection to validate
    
    Raises:
        AssertionError: If invalid
    """
    assert_quality_score_valid(inspection.quality_score)
    assert len(inspection.defects) == inspection.quality_score.defects_count, \
        "Defects count mismatch"
    assert len(inspection.anomalies) == inspection.quality_score.anomalies_count, \
        "Anomalies count mismatch"
    assert inspection.image_metadata.width > 0, "Invalid image width"
    assert inspection.image_metadata.height > 0, "Invalid image height"



