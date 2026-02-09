"""
Tests for StorageService
"""

import pytest
from pathlib import Path

from ..services.storage_service import StorageService
from ..core.exceptions import StorageError, NotFoundError
from .conftest import sample_store_design


class TestStorageService:
    """Test cases for StorageService"""
    
    def test_save_design(self, storage_service: StorageService, sample_store_design):
        """Test saving a design"""
        result = storage_service.save_design(sample_store_design)
        assert result is True
        
        # Verify file exists
        file_path = storage_service.storage_path / f"{sample_store_design.store_id}.json"
        assert file_path.exists()
    
    def test_load_design(self, storage_service: StorageService, sample_store_design):
        """Test loading a design"""
        # Save first
        storage_service.save_design(sample_store_design)
        
        # Load
        loaded = storage_service.load_design(sample_store_design.store_id)
        assert loaded is not None
        assert loaded.store_id == sample_store_design.store_id
        assert loaded.store_name == sample_store_design.store_name
    
    def test_load_design_not_found(self, storage_service: StorageService):
        """Test loading non-existent design"""
        with pytest.raises(NotFoundError):
            storage_service.load_design("non_existent_id")
    
    def test_list_designs(self, storage_service: StorageService, sample_store_design):
        """Test listing all designs"""
        # Save a design
        storage_service.save_design(sample_store_design)
        
        # List designs
        designs = storage_service.list_designs()
        assert len(designs) >= 1
        assert any(d.store_id == sample_store_design.store_id for d in designs)
    
    def test_delete_design(self, storage_service: StorageService, sample_store_design):
        """Test deleting a design"""
        # Save first
        storage_service.save_design(sample_store_design)
        
        # Delete
        result = storage_service.delete_design(sample_store_design.store_id)
        assert result is True
        
        # Verify file is gone
        file_path = storage_service.storage_path / f"{sample_store_design.store_id}.json"
        assert not file_path.exists()
        
        # Verify can't load
        with pytest.raises(NotFoundError):
            storage_service.load_design(sample_store_design.store_id)








