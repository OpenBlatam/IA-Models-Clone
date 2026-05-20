"""
Tests de transformación de datos
"""

import pytest
from unittest.mock import Mock
import json


class TestDataTransformation:
    """Tests de transformación de datos"""
    
    def test_normalize_data(self):
        """Test de normalización de datos"""
        def normalize_data(data):
            normalized = {}
            
            for key, value in data.items():
                # Normalizar key
                normalized_key = key.lower().replace(" ", "_")
                
                # Normalizar value
                if isinstance(value, str):
                    normalized_value = value.strip()
                elif isinstance(value, (int, float)):
                    normalized_value = float(value)
                else:
                    normalized_value = value
                
                normalized[normalized_key] = normalized_value
            
            return normalized
        
        data = {
            "Track Name": "  Rock Song  ",
            "Duration": 180,
            "Artist": "Artist Name"
        }
        
        normalized = normalize_data(data)
        
        assert "track_name" in normalized
        assert normalized["track_name"] == "Rock Song"
        assert normalized["duration"] == 180.0
    
    def test_transform_spotify_format(self):
        """Test de transformación desde formato Spotify"""
        def transform_from_spotify(spotify_data):
            return {
                "id": spotify_data.get("id"),
                "name": spotify_data.get("name"),
                "artists": [artist.get("name") for artist in spotify_data.get("artists", [])],
                "duration_ms": spotify_data.get("duration_ms"),
                "album": spotify_data.get("album", {}).get("name")
            }
        
        spotify_data = {
            "id": "123",
            "name": "Track Name",
            "artists": [{"name": "Artist 1"}, {"name": "Artist 2"}],
            "duration_ms": 180000,
            "album": {"name": "Album Name"}
        }
        
        transformed = transform_from_spotify(spotify_data)
        
        assert transformed["id"] == "123"
        assert transformed["name"] == "Track Name"
        assert len(transformed["artists"]) == 2
        assert transformed["duration_ms"] == 180000
    
    def test_flatten_nested_data(self):
        """Test de aplanamiento de datos anidados"""
        def flatten_data(data, prefix=""):
            flattened = {}
            
            for key, value in data.items():
                new_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    flattened.update(flatten_data(value, new_key))
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            flattened.update(flatten_data(item, f"{new_key}[{i}]"))
                        else:
                            flattened[f"{new_key}[{i}]"] = item
                else:
                    flattened[new_key] = value
            
            return flattened
        
        nested_data = {
            "track": {
                "name": "Song",
                "artists": [
                    {"name": "Artist 1"},
                    {"name": "Artist 2"}
                ]
            }
        }
        
        flattened = flatten_data(nested_data)
        
        assert "track.name" in flattened
        assert flattened["track.name"] == "Song"
        assert "track.artists[0].name" in flattened
        assert flattened["track.artists[0].name"] == "Artist 1"


class TestDataMapping:
    """Tests de mapeo de datos"""
    
    def test_map_fields(self):
        """Test de mapeo de campos"""
        def map_fields(data, field_mapping):
            mapped = {}
            
            for old_key, new_key in field_mapping.items():
                if old_key in data:
                    mapped[new_key] = data[old_key]
            
            return mapped
        
        data = {
            "track_id": "123",
            "track_name": "Song",
            "duration": 180
        }
        
        mapping = {
            "track_id": "id",
            "track_name": "name",
            "duration": "duration_seconds"
        }
        
        mapped = map_fields(data, mapping)
        
        assert mapped["id"] == "123"
        assert mapped["name"] == "Song"
        assert mapped["duration_seconds"] == 180
    
    def test_map_with_transformation(self):
        """Test de mapeo con transformación"""
        def map_with_transform(data, mapping):
            mapped = {}
            
            for old_key, config in mapping.items():
                if old_key in data:
                    new_key = config.get("target", old_key)
                    transform = config.get("transform")
                    
                    value = data[old_key]
                    if transform:
                        value = transform(value)
                    
                    mapped[new_key] = value
            
            return mapped
        
        data = {
            "duration_ms": 180000,
            "tempo": 120.5
        }
        
        mapping = {
            "duration_ms": {
                "target": "duration_seconds",
                "transform": lambda x: x / 1000
            },
            "tempo": {
                "target": "bpm",
                "transform": lambda x: int(round(x))
            }
        }
        
        mapped = map_with_transform(data, mapping)
        
        assert mapped["duration_seconds"] == 180.0
        assert mapped["bpm"] == 121


class TestDataValidation:
    """Tests de validación de datos transformados"""
    
    def test_validate_transformed_data(self):
        """Test de validación de datos transformados"""
        def validate_transformed(data, schema):
            errors = []
            
            for field, rules in schema.items():
                if field not in data:
                    if rules.get("required", False):
                        errors.append(f"Missing required field: {field}")
                    continue
                
                value = data[field]
                
                # Validar tipo
                expected_type = rules.get("type")
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Field {field} must be {expected_type.__name__}")
                
                # Validar rango
                if "min" in rules and value < rules["min"]:
                    errors.append(f"Field {field} must be >= {rules['min']}")
                if "max" in rules and value > rules["max"]:
                    errors.append(f"Field {field} must be <= {rules['max']}")
            
            return {"valid": len(errors) == 0, "errors": errors}
        
        schema = {
            "id": {"type": str, "required": True},
            "name": {"type": str, "required": True},
            "duration": {"type": (int, float), "min": 0}
        }
        
        valid_data = {
            "id": "123",
            "name": "Song",
            "duration": 180
        }
        
        result = validate_transformed(valid_data, schema)
        assert result["valid"] == True
        
        invalid_data = {
            "id": "123",
            "duration": -10  # Negativo, inválido
        }
        
        result = validate_transformed(invalid_data, schema)
        assert result["valid"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

