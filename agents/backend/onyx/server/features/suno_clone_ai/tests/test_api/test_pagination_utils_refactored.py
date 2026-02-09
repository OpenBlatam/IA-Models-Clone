"""
Tests refactorizados para utilidades de paginación
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from api.pagination import (
    PaginationParams,
    PaginatedResponse,
    create_paginated_response
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestPaginationParamsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para PaginationParams"""
    
    def test_pagination_params_default(self):
        """Test de parámetros por defecto"""
        params = PaginationParams()
        
        assert params.limit == 50
        assert params.offset == 0
    
    def test_pagination_params_custom(self):
        """Test de parámetros personalizados"""
        params = PaginationParams(limit=25, offset=10)
        
        assert params.limit == 25
        assert params.offset == 10
    
    @pytest.mark.parametrize("limit,should_raise", [
        (0, True),
        (101, True),
        (1, False),
        (100, False),
        (50, False)
    ])
    def test_pagination_params_validation_limit(self, limit, should_raise):
        """Test de validación de límite"""
        if should_raise:
            with pytest.raises(ValueError):
                PaginationParams(limit=limit)
        else:
            params = PaginationParams(limit=limit)
            assert params.limit == limit
    
    def test_pagination_params_validation_negative_offset(self):
        """Test de validación de offset negativo"""
        with pytest.raises(ValueError):
            PaginationParams(offset=-1)


class TestPaginatedResponseRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para PaginatedResponse"""
    
    @pytest.fixture
    def sample_items(self):
        """Items de ejemplo"""
        return [{"id": i} for i in range(10)]
    
    def test_paginated_response_creation(self, sample_items):
        """Test de creación de respuesta paginada"""
        response = PaginatedResponse(
            items=sample_items,
            total=100,
            limit=10,
            offset=0
        )
        
        assert len(response.items) == 10
        assert response.total == 100
        assert response.limit == 10
        assert response.offset == 0
        assert response.has_more is True
    
    @pytest.mark.parametrize("total,limit,offset,expected_has_more", [
        (100, 10, 0, True),
        (10, 10, 0, False),
        (95, 10, 90, False),
        (100, 10, 90, True)
    ])
    def test_paginated_response_has_more(self, sample_items, total, limit, offset, expected_has_more):
        """Test de cálculo de has_more"""
        response = PaginatedResponse(
            items=sample_items[:min(limit, len(sample_items))],
            total=total,
            limit=limit,
            offset=offset
        )
        
        assert response.has_more == expected_has_more
    
    @pytest.mark.parametrize("offset,limit,total,expected_next", [
        (0, 10, 100, 10),
        (90, 10, 100, None),
        (50, 10, 100, 60)
    ])
    def test_paginated_response_next_offset(self, sample_items, offset, limit, total, expected_next):
        """Test de cálculo de siguiente offset"""
        response = PaginatedResponse(
            items=sample_items[:min(limit, len(sample_items))],
            total=total,
            limit=limit,
            offset=offset
        )
        
        assert response.next_offset == expected_next
    
    @pytest.mark.parametrize("offset,limit,expected_prev", [
        (0, 10, None),
        (20, 10, 10),
        (5, 10, 0),
        (15, 10, 5)
    ])
    def test_paginated_response_prev_offset(self, sample_items, offset, limit, expected_prev):
        """Test de cálculo de offset anterior"""
        response = PaginatedResponse(
            items=sample_items[:min(limit, len(sample_items))],
            total=100,
            limit=limit,
            offset=offset
        )
        
        assert response.prev_offset == expected_prev


class TestCreatePaginatedResponseRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para create_paginated_response"""
    
    @pytest.mark.parametrize("items_count,total,limit,offset,expected_has_more", [
        (10, 100, 10, 0, True),
        (10, 10, 10, 0, False),
        (5, 95, 10, 90, False),
        (0, 0, 10, 0, False)
    ])
    def test_create_paginated_response(self, items_count, total, limit, offset, expected_has_more):
        """Test de creación de respuesta paginada con diferentes parámetros"""
        items = [{"id": i} for i in range(items_count)]
        response = create_paginated_response(
            items=items,
            total=total,
            limit=limit,
            offset=offset
        )
        
        assert isinstance(response, PaginatedResponse)
        assert len(response.items) == items_count
        assert response.total == total
        assert response.has_more == expected_has_more



