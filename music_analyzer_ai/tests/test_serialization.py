"""
Tests de serialización y deserialización
"""

import pytest
import json
from unittest.mock import Mock
from datetime import datetime


class TestJSONSerialization:
    """Tests de serialización JSON"""
    
    def test_serialize_analysis_to_json(self):
        """Test de serialización de análisis a JSON"""
        analysis = {
            "track_basic_info": {
                "name": "Test Track",
                "artists": ["Test Artist"],
                "duration_ms": 200000
            },
            "musical_analysis": {
                "key_signature": "C major",
                "tempo": {"bpm": 120.0}
            }
        }
        
        json_str = json.dumps(analysis, indent=2, ensure_ascii=False)
        
        assert isinstance(json_str, str)
        assert "Test Track" in json_str
        assert "C major" in json_str
        
        # Debe ser JSON válido
        parsed = json.loads(json_str)
        assert parsed == analysis
    
    def test_serialize_with_datetime(self):
        """Test de serialización con datetime"""
        data = {
            "timestamp": datetime.now(),
            "analysis": {"key": "value"}
        }
        
        def serialize_with_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        json_str = json.dumps(data, default=serialize_with_datetime)
        
        assert isinstance(json_str, str)
        assert "timestamp" in json_str
    
    def test_deserialize_from_json(self):
        """Test de deserialización desde JSON"""
        json_str = '{"name": "Test", "value": 123}'
        
        data = json.loads(json_str)
        
        assert data["name"] == "Test"
        assert data["value"] == 123


class TestDataTransformation:
    """Tests de transformación de datos"""
    
    def test_transform_spotify_to_internal_format(self):
        """Test de transformación de formato Spotify a interno"""
        spotify_format = {
            "tracks": {
                "items": [
                    {
                        "id": "123",
                        "name": "Track",
                        "artists": [{"name": "Artist"}]
                    }
                ]
            }
        }
        
        def transform(data):
            if "tracks" in data and "items" in data["tracks"]:
                return [item for item in data["tracks"]["items"]]
            return []
        
        result = transform(spotify_format)
        
        assert len(result) == 1
        assert result[0]["id"] == "123"
    
    def test_normalize_data_structure(self):
        """Test de normalización de estructura de datos"""
        def normalize(data):
            normalized = {}
            for key, value in data.items():
                # Normalizar keys a lowercase
                normalized[key.lower()] = value
            return normalized
        
        data = {"TrackName": "Test", "ArtistName": "Artist"}
        result = normalize(data)
        
        assert "trackname" in result
        assert "artistname" in result


class TestDataValidation:
    """Tests de validación de datos serializados"""
    
    def test_validate_json_structure(self):
        """Test de validación de estructura JSON"""
        def validate_structure(data, schema):
            for key in schema:
                if key not in data:
                    return False
            return True
        
        data = {"name": "Test", "artists": ["Artist"]}
        schema = ["name", "artists"]
        
        assert validate_structure(data, schema) == True
        
        incomplete_data = {"name": "Test"}
        assert validate_structure(incomplete_data, schema) == False
    
    def test_validate_data_types(self):
        """Test de validación de tipos de datos"""
        def validate_types(data, expected_types):
            for key, expected_type in expected_types.items():
                if key not in data:
                    return False
                if not isinstance(data[key], expected_type):
                    return False
            return True
        
        data = {
            "name": "Test",
            "duration": 200000,
            "popularity": 80
        }
        expected = {
            "name": str,
            "duration": int,
            "popularity": int
        }
        
        assert validate_types(data, expected) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

