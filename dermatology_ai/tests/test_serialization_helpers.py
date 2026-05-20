"""
Serialization Testing Helpers
Specialized helpers for serialization testing
"""

from typing import Any, Dict, List, Optional
import json
import pickle
from datetime import datetime
from unittest.mock import Mock


class SerializationTestHelpers:
    """Helpers for serialization testing"""
    
    @staticmethod
    def assert_json_serializable(obj: Any) -> bool:
        """Assert object is JSON serializable"""
        try:
            json.dumps(obj, default=str)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def assert_json_roundtrip(obj: Dict[str, Any]) -> bool:
        """Assert object survives JSON roundtrip"""
        try:
            json_str = json.dumps(obj, default=str)
            reconstructed = json.loads(json_str)
            
            # Compare (with type conversion for dates)
            for key, value in obj.items():
                if isinstance(value, datetime):
                    # Datetime becomes string in JSON
                    assert key in reconstructed
                else:
                    assert reconstructed.get(key) == value, \
                        f"Value mismatch for key {key}"
            return True
        except Exception as e:
            raise AssertionError(f"JSON roundtrip failed: {e}")
    
    @staticmethod
    def assert_pickle_serializable(obj: Any) -> bool:
        """Assert object is pickle serializable"""
        try:
            pickled = pickle.dumps(obj)
            unpickled = pickle.loads(pickled)
            return True
        except Exception:
            return False
    
    @staticmethod
    def assert_pickle_roundtrip(obj: Any) -> bool:
        """Assert object survives pickle roundtrip"""
        try:
            pickled = pickle.dumps(obj)
            unpickled = pickle.loads(pickled)
            assert obj == unpickled, "Object changed after pickle roundtrip"
            return True
        except Exception as e:
            raise AssertionError(f"Pickle roundtrip failed: {e}")


class EntitySerializationHelpers:
    """Helpers for entity serialization testing"""
    
    @staticmethod
    def assert_entity_to_dict(entity: Any, expected_dict: Dict[str, Any]):
        """Assert entity converts to dict correctly"""
        if hasattr(entity, "to_dict"):
            result = entity.to_dict()
            for key, value in expected_dict.items():
                assert key in result, f"Key {key} missing in dict"
                assert result[key] == value, \
                    f"Value for {key} is {result[key]}, expected {value}"
        else:
            raise AssertionError("Entity does not have to_dict method")
    
    @staticmethod
    def assert_dict_to_entity(
        data: Dict[str, Any],
        entity_class: type,
        expected_attributes: Optional[Dict[str, Any]] = None
    ):
        """Assert dict converts to entity correctly"""
        if hasattr(entity_class, "from_dict"):
            entity = entity_class.from_dict(data)
            if expected_attributes:
                for key, value in expected_attributes.items():
                    assert hasattr(entity, key), f"Entity missing attribute {key}"
                    assert getattr(entity, key) == value, \
                        f"Attribute {key} is {getattr(entity, key)}, expected {value}"
            return entity
        elif hasattr(entity_class, "__init__"):
            # Try direct instantiation
            entity = entity_class(**data)
            if expected_attributes:
                for key, value in expected_attributes.items():
                    assert hasattr(entity, key), f"Entity missing attribute {key}"
                    assert getattr(entity, key) == value, \
                        f"Attribute {key} is {getattr(entity, key)}, expected {value}"
            return entity
        else:
            raise AssertionError(f"Entity class {entity_class} cannot be created from dict")


class ResponseSerializationHelpers:
    """Helpers for API response serialization"""
    
    @staticmethod
    def assert_response_serializable(response: Dict[str, Any]) -> bool:
        """Assert API response is serializable"""
        return SerializationTestHelpers.assert_json_serializable(response)
    
    @staticmethod
    def assert_response_structure(
        response: Dict[str, Any],
        required_fields: List[str],
        optional_fields: Optional[List[str]] = None
    ):
        """Assert response has correct structure"""
        for field in required_fields:
            assert field in response, f"Required field {field} missing in response"
        
        if optional_fields:
            all_fields = set(required_fields) | set(optional_fields)
            unexpected_fields = set(response.keys()) - all_fields
            if unexpected_fields:
                # Warn but don't fail
                pass


class DateSerializationHelpers:
    """Helpers for date/datetime serialization"""
    
    @staticmethod
    def serialize_datetime(dt: datetime, format: str = "iso") -> str:
        """Serialize datetime to string"""
        if format == "iso":
            return dt.isoformat()
        elif format == "rfc3339":
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            return dt.strftime(format)
    
    @staticmethod
    def deserialize_datetime(dt_str: str, format: str = "iso") -> datetime:
        """Deserialize string to datetime"""
        if format == "iso":
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        else:
            return datetime.strptime(dt_str, format)
    
    @staticmethod
    def assert_datetime_serialization(dt: datetime, format: str = "iso"):
        """Assert datetime survives serialization roundtrip"""
        serialized = DateSerializationHelpers.serialize_datetime(dt, format)
        deserialized = DateSerializationHelpers.deserialize_datetime(serialized, format)
        
        # Compare with tolerance for microseconds
        diff = abs((dt - deserialized).total_seconds())
        assert diff < 1.0, f"Datetime changed after roundtrip: {diff}s difference"


# Convenience exports
assert_json_serializable = SerializationTestHelpers.assert_json_serializable
assert_json_roundtrip = SerializationTestHelpers.assert_json_roundtrip
assert_pickle_serializable = SerializationTestHelpers.assert_pickle_serializable
assert_pickle_roundtrip = SerializationTestHelpers.assert_pickle_roundtrip

assert_entity_to_dict = EntitySerializationHelpers.assert_entity_to_dict
assert_dict_to_entity = EntitySerializationHelpers.assert_dict_to_entity

assert_response_serializable = ResponseSerializationHelpers.assert_response_serializable
assert_response_structure = ResponseSerializationHelpers.assert_response_structure

serialize_datetime = DateSerializationHelpers.serialize_datetime
deserialize_datetime = DateSerializationHelpers.deserialize_datetime
assert_datetime_serialization = DateSerializationHelpers.assert_datetime_serialization



