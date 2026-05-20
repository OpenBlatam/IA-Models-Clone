from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from uuid6 import UUID, uuid7
from agents.backend.onyx.server.features.folder.models import Folder
from agents.backend.onyx.server.features.folder.schemas import FolderCreate, FolderRead

from typing import Any, List, Dict, Optional
import logging
import asyncio
def test_folder_valid():
    
    """test_folder_valid function."""
folder = Folder(name="Root", parent_id=None)
    assert isinstance(folder.id, UUID)
    assert folder.name == "Root"
    assert folder.parent_id is None

def test_folder_empty_name_raises():
    
    """test_folder_empty_name_raises function."""
with pytest.raises(ValueError):
        Folder(name=" ", parent_id=None)

def test_folder_serialization():
    
    """test_folder_serialization function."""
folder = Folder(name="Root", parent_id=None)
    data = folder.model_dump_json()
    assert '"name":"Root"' in data

def test_folder_create_valid():
    
    """test_folder_create_valid function."""
schema = FolderCreate(name="Root", parent_id=None)
    assert schema.name == "Root"

def test_folder_create_invalid_name():
    
    """test_folder_create_invalid_name function."""
with pytest.raises(ValueError):
        FolderCreate(name=" ", parent_id=None)

def test_folder_read_valid():
    
    """test_folder_read_valid function."""
schema = FolderRead(id=uuid7(), name="Root", parent_id=None)
    assert schema.name == "Root"
    assert isinstance(schema.id, UUID) 