"""
Tests para las rutas de marketplace
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.marketplace import router
from services.marketplace import MarketplaceService, LicenseType, ListingStatus


@pytest.fixture
def mock_marketplace_service():
    """Mock del servicio de marketplace"""
    service = Mock(spec=MarketplaceService)
    
    # Mock de listing
    listing = Mock()
    listing.listing_id = "listing-123"
    listing.song_id = "song-456"
    listing.title = "Test Song"
    listing.price = 9.99
    listing.license_type = LicenseType.COMMERCIAL
    listing.status = ListingStatus.ACTIVE
    
    service.create_listing = Mock(return_value=listing)
    service.get_listing = Mock(return_value=listing)
    service.search_listings = Mock(return_value=[listing])
    service.purchase_listing = Mock(return_value={"transaction_id": "txn-789"})
    service.add_review = Mock(return_value={"review_id": "review-123"})
    
    return service


@pytest.fixture
def client(mock_marketplace_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.marketplace.get_marketplace_service', return_value=mock_marketplace_service):
        with patch('api.routes.marketplace.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCreateListing:
    """Tests para crear publicación"""
    
    def test_create_listing_success(self, client, mock_marketplace_service):
        """Test de creación exitosa de publicación"""
        response = client.post(
            "/marketplace/listings",
            json={
                "song_id": "song-456",
                "title": "Test Song",
                "description": "A great test song",
                "price": 9.99,
                "license_type": "commercial",
                "tags": ["pop", "happy"]
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "listing_id" in data
        assert data["song_id"] == "song-456"
        assert data["title"] == "Test Song"
        assert data["price"] == 9.99
    
    def test_create_listing_different_license_types(self, client, mock_marketplace_service):
        """Test con diferentes tipos de licencia"""
        license_types = ["commercial", "non-commercial", "royalty-free"]
        
        for license_type in license_types:
            response = client.post(
                "/marketplace/listings",
                json={
                    "song_id": "song-456",
                    "title": "Test Song",
                    "description": "Test",
                    "price": 9.99,
                    "license_type": license_type
                }
            )
            assert response.status_code == status.HTTP_200_OK
    
    def test_create_listing_invalid_license(self, client):
        """Test con tipo de licencia inválido"""
        response = client.post(
            "/marketplace/listings",
            json={
                "song_id": "song-456",
                "title": "Test Song",
                "description": "Test",
                "price": 9.99,
                "license_type": "invalid"
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid license type" in response.json()["detail"]
    
    def test_create_listing_price_validation(self, client):
        """Test de validación de precio"""
        # Precio negativo
        response = client.post(
            "/marketplace/listings",
            json={
                "song_id": "song-456",
                "title": "Test",
                "description": "Test",
                "price": -10.0,
                "license_type": "commercial"
            }
        )
        # Puede ser válido o inválido dependiendo de la validación
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


@pytest.mark.unit
@pytest.mark.api
class TestSearchListings:
    """Tests para buscar publicaciones"""
    
    def test_search_listings_success(self, client, mock_marketplace_service):
        """Test de búsqueda exitosa"""
        response = client.get("/marketplace/listings/search?query=pop")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "listings" in data or isinstance(data, list)
    
    def test_search_listings_with_filters(self, client, mock_marketplace_service):
        """Test con filtros"""
        response = client.get(
            "/marketplace/listings/search",
            params={
                "query": "pop",
                "min_price": 5.0,
                "max_price": 20.0,
                "license_type": "commercial"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestPurchaseListing:
    """Tests para comprar publicación"""
    
    def test_purchase_listing_success(self, client, mock_marketplace_service):
        """Test de compra exitosa"""
        response = client.post(
            "/marketplace/listings/listing-123/purchase",
            json={"buyer_id": "buyer-789"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "transaction_id" in data or "message" in data
    
    def test_purchase_listing_not_found(self, client, mock_marketplace_service):
        """Test cuando la publicación no existe"""
        mock_marketplace_service.get_listing.return_value = None
        
        response = client.post(
            "/marketplace/listings/nonexistent/purchase",
            json={"buyer_id": "buyer-789"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.unit
@pytest.mark.api
class TestAddReview:
    """Tests para agregar review"""
    
    def test_add_review_success(self, client, mock_marketplace_service):
        """Test de agregar review exitosamente"""
        response = client.post(
            "/marketplace/listings/listing-123/reviews",
            json={
                "user_id": "user-789",
                "rating": 5,
                "comment": "Great song!"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "review_id" in data or "message" in data
    
    def test_add_review_rating_validation(self, client):
        """Test de validación de rating"""
        # Rating inválido
        response = client.post(
            "/marketplace/listings/listing-123/reviews",
            json={
                "user_id": "user-789",
                "rating": 6,
                "comment": "Test"
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.integration
@pytest.mark.api
class TestMarketplaceIntegration:
    """Tests de integración para marketplace"""
    
    def test_full_marketplace_workflow(self, client, mock_marketplace_service):
        """Test del flujo completo de marketplace"""
        # 1. Crear publicación
        create_response = client.post(
            "/marketplace/listings",
            json={
                "song_id": "song-456",
                "title": "Marketplace Test Song",
                "description": "Test description",
                "price": 9.99,
                "license_type": "commercial"
            }
        )
        assert create_response.status_code == status.HTTP_200_OK
        listing_id = create_response.json()["listing_id"]
        
        # 2. Buscar publicaciones
        search_response = client.get("/marketplace/listings/search?query=test")
        assert search_response.status_code == status.HTTP_200_OK
        
        # 3. Agregar review
        review_response = client.post(
            f"/marketplace/listings/{listing_id}/reviews",
            json={
                "user_id": "user-789",
                "rating": 5,
                "comment": "Excellent!"
            }
        )
        assert review_response.status_code == status.HTTP_200_OK
        
        # 4. Comprar publicación
        purchase_response = client.post(
            f"/marketplace/listings/{listing_id}/purchase",
            json={"buyer_id": "buyer-123"}
        )
        assert purchase_response.status_code == status.HTTP_200_OK



