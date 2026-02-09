"""
Tests refactorizados para helpers de validación
Usando clases base y helpers para eliminar duplicación
"""

import pytest
import uuid
from api.utils.validation_helpers import (
    validate_uuid_list,
    parse_comma_separated_ids,
    validate_prompt_length
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestValidateUUIDListRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para validate_uuid_list"""
    
    def test_validate_uuid_list_valid(self):
        """Test de validación de lista válida"""
        ids = [str(uuid.uuid4()) for _ in range(5)]
        result = validate_uuid_list(ids)
        
        assert result == ids
    
    def test_validate_uuid_list_empty(self):
        """Test de lista vacía"""
        with pytest.raises(ValueError, match="required"):
            validate_uuid_list([])
    
    @pytest.mark.parametrize("count,max_items,should_raise", [
        (50, 50, False),
        (51, 50, True),
        (100, 50, True)
    ])
    def test_validate_uuid_list_max_items(self, count, max_items, should_raise):
        """Test de máximo de items"""
        ids = [str(uuid.uuid4()) for _ in range(count)]
        
        if should_raise:
            with pytest.raises(ValueError, match="Maximum"):
                validate_uuid_list(ids, max_items=max_items)
        else:
            result = validate_uuid_list(ids, max_items=max_items)
            assert len(result) == count
    
    def test_validate_uuid_list_invalid_format(self):
        """Test de formato inválido"""
        ids = ["not-a-uuid", "also-not-a-uuid"]
        
        with pytest.raises(ValueError, match="Invalid"):
            validate_uuid_list(ids)
    
    def test_validate_uuid_list_strips_whitespace(self):
        """Test de que elimina espacios en blanco"""
        valid_id = str(uuid.uuid4())
        ids = [f" {valid_id} ", f"\t{valid_id}\t"]
        result = validate_uuid_list(ids)
        
        assert len(result) == 2
        assert all(uuid.UUID(id_str) for id_str in result)


class TestParseCommaSeparatedIdsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para parse_comma_separated_ids"""
    
    def test_parse_comma_separated_ids_valid(self):
        """Test de parseo de IDs válidos"""
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        ids_string = f"{id1}, {id2}"
        
        result = parse_comma_separated_ids(ids_string)
        
        assert len(result) == 2
        assert id1 in result
        assert id2 in result
    
    @pytest.mark.parametrize("ids_string,should_raise", [
        ("", True),
        ("   ", True),
        (f"{uuid.uuid4()}", False),
        (f"{uuid.uuid4()}, {uuid.uuid4()}", False)
    ])
    def test_parse_comma_separated_ids_validation(self, ids_string, should_raise):
        """Test de validación de string"""
        if should_raise:
            with pytest.raises(ValueError):
                parse_comma_separated_ids(ids_string)
        else:
            result = parse_comma_separated_ids(ids_string)
            assert len(result) > 0


class TestValidatePromptLengthRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para validate_prompt_length"""
    
    @pytest.mark.parametrize("prompt,max_length,should_raise", [
        ("Valid prompt", 500, False),
        ("", 500, True),
        ("   ", 500, True),
        ("x" * 501, 500, True),
        ("x" * 1000, 2000, False)
    ])
    def test_validate_prompt_length(self, prompt, max_length, should_raise):
        """Test de validación de longitud de prompt"""
        if should_raise:
            with pytest.raises(ValueError):
                validate_prompt_length(prompt, max_length=max_length)
        else:
            result = validate_prompt_length(prompt, max_length=max_length)
            assert result == prompt.strip()
    
    def test_validate_prompt_length_strips_whitespace(self):
        """Test de que elimina espacios en blanco"""
        prompt = "  valid prompt  "
        result = validate_prompt_length(prompt)
        
        assert result == "valid prompt"
    
    def test_validate_prompt_length_not_string(self):
        """Test de que no es string"""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_prompt_length(123)



