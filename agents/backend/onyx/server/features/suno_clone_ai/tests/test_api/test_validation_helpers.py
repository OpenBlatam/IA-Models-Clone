"""
Tests para helpers de validación
"""

import pytest
import uuid
from api.utils.validation_helpers import (
    validate_uuid_list,
    parse_comma_separated_ids,
    validate_prompt_length
)


@pytest.mark.unit
@pytest.mark.api
class TestValidateUUIDList:
    """Tests para validate_uuid_list"""
    
    def test_validate_uuid_list_valid(self):
        """Test de validación de lista válida"""
        ids = [str(uuid.uuid4()) for _ in range(5)]
        result = validate_uuid_list(ids)
        
        assert result == ids
    
    def test_validate_uuid_list_empty(self):
        """Test de lista vacía"""
        with pytest.raises(ValueError, match="required"):
            validate_uuid_list([])
    
    def test_validate_uuid_list_max_items(self):
        """Test de exceder máximo de items"""
        ids = [str(uuid.uuid4()) for _ in range(51)]
        
        with pytest.raises(ValueError, match="Maximum"):
            validate_uuid_list(ids, max_items=50)
    
    def test_validate_uuid_list_invalid_format(self):
        """Test de formato inválido"""
        ids = ["not-a-uuid", "also-not-a-uuid"]
        
        with pytest.raises(ValueError, match="Invalid"):
            validate_uuid_list(ids)
    
    def test_validate_uuid_list_mixed(self):
        """Test de lista mixta válida e inválida"""
        valid_id = str(uuid.uuid4())
        ids = [valid_id, "invalid-uuid"]
        
        with pytest.raises(ValueError):
            validate_uuid_list(ids)
    
    def test_validate_uuid_list_strips_whitespace(self):
        """Test de que elimina espacios en blanco"""
        valid_id = str(uuid.uuid4())
        ids = [f" {valid_id} ", f"\t{valid_id}\t"]
        result = validate_uuid_list(ids)
        
        assert len(result) == 2
        assert all(uuid.UUID(id_str) for id_str in result)


@pytest.mark.unit
@pytest.mark.api
class TestParseCommaSeparatedIds:
    """Tests para parse_comma_separated_ids"""
    
    def test_parse_comma_separated_ids_valid(self):
        """Test de parseo de IDs válidos"""
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        ids_string = f"{id1}, {id2}"
        
        result = parse_comma_separated_ids(ids_string)
        
        assert len(result) == 2
        assert id1 in result
        assert id2 in result
    
    def test_parse_comma_separated_ids_empty(self):
        """Test de string vacío"""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_comma_separated_ids("")
    
    def test_parse_comma_separated_ids_whitespace(self):
        """Test de string con solo espacios"""
        with pytest.raises(ValueError):
            parse_comma_separated_ids("   ")
    
    def test_parse_comma_separated_ids_max_items(self):
        """Test de exceder máximo de items"""
        ids = [str(uuid.uuid4()) for _ in range(51)]
        ids_string = ", ".join(ids)
        
        with pytest.raises(ValueError):
            parse_comma_separated_ids(ids_string, max_items=50)
    
    def test_parse_comma_separated_ids_invalid(self):
        """Test de IDs inválidos"""
        with pytest.raises(ValueError):
            parse_comma_separated_ids("invalid-uuid-1, invalid-uuid-2")


@pytest.mark.unit
@pytest.mark.api
class TestValidatePromptLength:
    """Tests para validate_prompt_length"""
    
    def test_validate_prompt_length_valid(self):
        """Test de prompt válido"""
        prompt = "This is a valid prompt"
        result = validate_prompt_length(prompt)
        
        assert result == prompt
    
    def test_validate_prompt_length_empty(self):
        """Test de prompt vacío"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt_length("")
    
    def test_validate_prompt_length_whitespace_only(self):
        """Test de prompt con solo espacios"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt_length("   ")
    
    def test_validate_prompt_length_too_long(self):
        """Test de prompt muy largo"""
        prompt = "x" * 501
        
        with pytest.raises(ValueError, match="exceeds"):
            validate_prompt_length(prompt, max_length=500)
    
    def test_validate_prompt_length_strips_whitespace(self):
        """Test de que elimina espacios en blanco"""
        prompt = "  valid prompt  "
        result = validate_prompt_length(prompt)
        
        assert result == "valid prompt"
    
    def test_validate_prompt_length_not_string(self):
        """Test de que no es string"""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_prompt_length(123)
    
    def test_validate_prompt_length_custom_max(self):
        """Test de max_length personalizado"""
        prompt = "x" * 1000
        result = validate_prompt_length(prompt, max_length=2000)
        
        assert len(result) == 1000



