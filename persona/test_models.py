from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from uuid6 import UUID
from agents.backend.onyx.server.features.persona.models import Persona
from agents.backend.onyx.server.features.persona.schemas import PersonaCreate, PersonaRead

import uuid6
from typing import Any, List, Dict, Optional
import logging
import asyncio
def test_persona_valid():
    
    """test_persona_valid function."""
persona = Persona(name="Ada Lovelace", description="Pioneer", attributes={"field": "math"})
    assert isinstance(persona.id, UUID)
    assert persona.name == "Ada Lovelace"
    assert persona.attributes == {"field": "math"}

def test_persona_empty_name_raises():
    
    """test_persona_empty_name_raises function."""
with pytest.raises(ValueError):
        Persona(name="   ", description="Empty", attributes={})

def test_persona_attributes_not_dict():
    
    """test_persona_attributes_not_dict function."""
with pytest.raises(ValueError):
        Persona(name="Alan", description="Test", attributes=["not", "a", "dict"])

def test_persona_serialization():
    
    """test_persona_serialization function."""
persona = Persona(name="Grace", description=None, attributes={})
    data = persona.model_dump_json()
    assert '"name":"Grace"' in data

def test_persona_create_valid():
    
    """test_persona_create_valid function."""
schema = PersonaCreate(name="Ada", description="Test")
    assert schema.name == "Ada"

def test_persona_create_invalid_name():
    
    """test_persona_create_invalid_name function."""
with pytest.raises(ValueError):
        PersonaCreate(name="", description="Test")

def test_persona_read_valid():
    
    """test_persona_read_valid function."""
    schema = PersonaRead(id=uuid6.uuid7(), name="Alan", description=None, attributes={})
    assert schema.name == "Alan"
    assert isinstance(schema.id, type(uuid6.uuid7())) 