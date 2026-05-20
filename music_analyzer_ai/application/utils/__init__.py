"""
Application-level utility functions for use cases.
"""

from .validation_helpers import (
    validate_string_not_empty,
    validate_numeric_range,
)
from .data_extractors import (
    extract_track_id,
    extract_track_name,
    extract_artists,
    extract_album_name,
    safe_get_nested,
)
from .dto_converters import (
    convert_dict_to_recommendation_dto,
    convert_dict_to_track_analysis_dto,
    convert_dict_list_to_recommendation_dtos,
    convert_dict_list_to_track_analysis_dtos,
)

__all__ = [
    # Validation
    "validate_string_not_empty",
    "validate_numeric_range",
    # Data extraction
    "extract_track_id",
    "extract_track_name",
    "extract_artists",
    "extract_album_name",
    "safe_get_nested",
    # DTO conversion
    "convert_dict_to_recommendation_dto",
    "convert_dict_to_track_analysis_dto",
    "convert_dict_list_to_recommendation_dtos",
    "convert_dict_list_to_track_analysis_dtos",
]








