"""
Transformation Testing Helpers
Specialized helpers for data transformation testing
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock


class TransformationTestHelpers:
    """Helpers for transformation testing"""
    
    @staticmethod
    def create_mock_transformer(
        transform_func: Optional[Callable] = None
    ) -> Mock:
        """Create mock transformer"""
        transformer = Mock()
        
        if transform_func:
            transformer.transform = Mock(side_effect=transform_func)
        else:
            transformer.transform = Mock(return_value={})
        
        return transformer
    
    @staticmethod
    def assert_transformation_applied(
        transformer: Mock,
        input_data: Any,
        expected_output: Optional[Any] = None
    ):
        """Assert transformation was applied"""
        assert transformer.transform.called, "Transformation was not applied"
        
        if expected_output is not None:
            call_args = transformer.transform.call_args
            if call_args:
                result = transformer.transform(*call_args[0], **call_args[1])
                assert result == expected_output, \
                    f"Transformation result {result} does not match expected {expected_output}"


class MappingHelpers:
    """Helpers for mapping testing"""
    
    @staticmethod
    def create_field_mapping(
        source_fields: List[str],
        target_fields: List[str]
    ) -> Dict[str, str]:
        """Create field mapping dictionary"""
        return dict(zip(source_fields, target_fields))
    
    @staticmethod
    def assert_fields_mapped(
        source: Dict[str, Any],
        target: Dict[str, Any],
        mapping: Dict[str, str]
    ):
        """Assert fields were mapped correctly"""
        for source_field, target_field in mapping.items():
            assert source_field in source, \
                f"Source field {source_field} not found"
            assert target_field in target, \
                f"Target field {target_field} not found"
            assert source[source_field] == target[target_field], \
                f"Field mapping mismatch: {source_field} -> {target_field}"


class NormalizationHelpers:
    """Helpers for normalization testing"""
    
    @staticmethod
    def normalize_data(
        data: Any,
        normalizer: Callable
    ) -> Any:
        """Normalize data using normalizer function"""
        return normalizer(data)
    
    @staticmethod
    def assert_data_normalized(
        original: Any,
        normalized: Any,
        expected_format: Optional[str] = None
    ):
        """Assert data was normalized"""
        assert normalized is not None, "Normalized data is None"
        
        if expected_format == "lowercase" and isinstance(normalized, str):
            assert normalized.islower(), "Data was not normalized to lowercase"
        elif expected_format == "uppercase" and isinstance(normalized, str):
            assert normalized.isupper(), "Data was not normalized to uppercase"


# Convenience exports
create_mock_transformer = TransformationTestHelpers.create_mock_transformer
assert_transformation_applied = TransformationTestHelpers.assert_transformation_applied

create_field_mapping = MappingHelpers.create_field_mapping
assert_fields_mapped = MappingHelpers.assert_fields_mapped

normalize_data = NormalizationHelpers.normalize_data
assert_data_normalized = NormalizationHelpers.assert_data_normalized



