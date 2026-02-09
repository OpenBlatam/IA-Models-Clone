"""
Tests para utilidades de paginación
"""

import pytest
from api.pagination import (
    PaginationParams,
    PaginatedResponse,
    create_paginated_response
)


@pytest.mark.unit
@pytest.mark.api
class TestPaginationParams:
    """Tests para PaginationParams"""
    
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
    
    def test_pagination_params_validation_min_limit(self):
        """Test de validación de límite mínimo"""
        with pytest.raises(ValueError):
            PaginationParams(limit=0)
    
    def test_pagination_params_validation_max_limit(self):
        """Test de validación de límite máximo"""
        with pytest.raises(ValueError):
            PaginationParams(limit=101)
    
    def test_pagination_params_validation_negative_offset(self):
        """Test de validación de offset negativo"""
        with pytest.raises(ValueError):
            PaginationParams(offset=-1)


@pytest.mark.unit
@pytest.mark.api
class TestPaginatedResponse:
    """Tests para PaginatedResponse"""
    
    def test_paginated_response_creation(self):
        """Test de creación de respuesta paginada"""
        items = [{"id": i} for i in range(10)]
        response = PaginatedResponse(
            items=items,
            total=100,
            limit=10,
            offset=0
        )
        
        assert len(response.items) == 10
        assert response.total == 100
        assert response.limit == 10
        assert response.offset == 0
        assert response.has_more is True
    
    def test_paginated_response_no_more(self):
        """Test de respuesta sin más páginas"""
        items = [{"id": i} for i in range(10)]
        response = PaginatedResponse(
            items=items,
            total=10,
            limit=10,
            offset=0
        )
        
        assert response.has_more is False
    
    def test_paginated_response_next_offset(self):
        """Test de cálculo de siguiente offset"""
        items = [{"id": i} for i in range(10)]
        response = PaginatedResponse(
            items=items,
            total=100,
            limit=10,
            offset=0
        )
        
        assert response.next_offset == 10
    
    def test_paginated_response_next_offset_none(self):
        """Test de siguiente offset cuando no hay más"""
        items = [{"id": i} for i in range(10)]
        response = PaginatedResponse(
            items=items,
            total=10,
            limit=10,
            offset=0
        )
        
        assert response.next_offset is None
    
    def test_paginated_response_prev_offset(self):
        """Test de cálculo de offset anterior"""
        items = [{"id": i} for i in range(10)]
        response = PaginatedResponse(
            items=items,
            total=100,
            limit=10,
            offset=20
        )
        
        assert response.prev_offset == 10
    
    def test_paginated_response_prev_offset_none(self):
        """Test de offset anterior cuando está en el inicio"""
        items = [{"id": i} for i in range(10)]
        response = PaginatedResponse(
            items=items,
            total=100,
            limit=10,
            offset=0
        )
        
        assert response.prev_offset is None
    
    def test_paginated_response_prev_offset_boundary(self):
        """Test de offset anterior en límite"""
        items = [{"id": i} for i in range(10)]
        response = PaginatedResponse(
            items=items,
            total=100,
            limit=10,
            offset=5
        )
        
        assert response.prev_offset == 0


@pytest.mark.unit
@pytest.mark.api
class TestCreatePaginatedResponse:
    """Tests para create_paginated_response"""
    
    def test_create_paginated_response(self):
        """Test de creación de respuesta paginada"""
        items = [{"id": i} for i in range(10)]
        response = create_paginated_response(
            items=items,
            total=100,
            limit=10,
            offset=0
        )
        
        assert isinstance(response, PaginatedResponse)
        assert len(response.items) == 10
        assert response.total == 100
        assert response.has_more is True
    
    def test_create_paginated_response_last_page(self):
        """Test de creación de última página"""
        items = [{"id": i} for i in range(5)]
        response = create_paginated_response(
            items=items,
            total=95,
            limit=10,
            offset=90
        )
        
        assert response.has_more is False
    
    def test_create_paginated_response_empty(self):
        """Test de creación con lista vacía"""
        response = create_paginated_response(
            items=[],
            total=0,
            limit=10,
            offset=0
        )
        
        assert len(response.items) == 0
        assert response.has_more is False



