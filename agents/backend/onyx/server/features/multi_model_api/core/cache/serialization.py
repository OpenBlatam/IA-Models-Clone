"""
Serialization module for cache system
Handles serialization/deserialization with multiple format support
"""

import json
import pickle
from typing import Any

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import rapidjson
    RAPIDJSON_AVAILABLE = True
except ImportError:
    RAPIDJSON_AVAILABLE = False


class Serializer:
    """Handles serialization with smart format detection"""
    
    @staticmethod
    def serialize(value: Any) -> bytes:
        """Serialize value for storage with smart type detection and optimal format
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized bytes
        """
        if MSGPACK_AVAILABLE:
            try:
                return msgpack.packb(value, use_bin_type=True, strict_types=False)
            except (TypeError, ValueError):
                pass
        
        if isinstance(value, (str, int, float, bool, type(None), dict, list)):
            if ORJSON_AVAILABLE:
                try:
                    return orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY)
                except (TypeError, ValueError):
                    if RAPIDJSON_AVAILABLE:
                        try:
                            return rapidjson.dumps(value).encode('utf-8')
                        except Exception:
                            pass
                    return json.dumps(value, default=str).encode('utf-8')
            elif RAPIDJSON_AVAILABLE:
                try:
                    return rapidjson.dumps(value).encode('utf-8')
                except Exception:
                    return json.dumps(value, default=str).encode('utf-8')
            return json.dumps(value, default=str).encode('utf-8')
        
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def deserialize(value: bytes) -> Any:
        """Deserialize value from storage with smart format detection
        
        Args:
            value: Serialized bytes
            
        Returns:
            Deserialized value
        """
        if isinstance(value, str):
            value = value.encode('utf-8')
        
        if not value:
            return None
        
        if MSGPACK_AVAILABLE:
            try:
                first_byte = value[0]
                if (0x80 <= first_byte <= 0xff) or first_byte in (0xcc, 0xcd, 0xce, 0xcf, 0xd0, 0xd1, 0xd2, 0xd3):
                    return msgpack.unpackb(value, raw=False, strict_map_key=False)
            except Exception:
                pass
        
        if ORJSON_AVAILABLE:
            try:
                return orjson.loads(value)
            except Exception:
                pass
        
        if RAPIDJSON_AVAILABLE:
            try:
                if isinstance(value, bytes):
                    return rapidjson.loads(value.decode('utf-8'))
                return rapidjson.loads(value)
            except Exception:
                pass
        
        try:
            if isinstance(value, bytes):
                return json.loads(value.decode('utf-8'))
            return json.loads(value)
        except Exception:
            try:
                return pickle.loads(value)
            except Exception:
                if isinstance(value, bytes):
                    return value.decode('utf-8', errors='ignore')
                return value
    
    @staticmethod
    def get_available_formats() -> dict:
        """Get information about available serialization formats"""
        return {
            "orjson": ORJSON_AVAILABLE,
            "msgpack": MSGPACK_AVAILABLE,
            "rapidjson": RAPIDJSON_AVAILABLE
        }

