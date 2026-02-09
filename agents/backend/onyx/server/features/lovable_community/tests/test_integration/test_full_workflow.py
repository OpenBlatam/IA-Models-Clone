"""
Tests de integración para flujos completos de Lovable Community
"""

import pytest
from datetime import datetime
from tests.helpers.test_helpers import (
    create_publish_request,
    create_remix_request,
    create_vote_request,
    generate_user_id
)
from tests.helpers.advanced_helpers import (
    DataFactory,
    TestDataBuilder,
    PerformanceHelper
)
from tests.helpers.assertion_helpers import (
    assert_chat_response_valid,
    assert_chat_list_valid
)


class TestFullChatLifecycle:
    """Tests del ciclo de vida completo de un chat"""
    
    @pytest.mark.integration
    def test_chat_lifecycle_complete(self, chat_service, sample_user_id):
        """Test del ciclo completo: publish -> vote -> remix -> update -> delete"""
        # 1. Publicar chat
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Lifecycle Test Chat",
            chat_content='{"messages": [{"role": "user", "content": "Hello"}]}',
            description="Testing full lifecycle",
            tags=["test", "lifecycle"]
        )
        
        assert_chat_valid(chat)
        chat_id = chat.id
        
        # 2. Obtener chat
        retrieved = chat_service.get_chat(chat_id)
        assert retrieved is not None
        assert retrieved.id == chat_id
        
        # 3. Votar
        vote = chat_service.vote_chat(chat_id, "voter-1", "upvote")
        assert vote.vote_type == "upvote"
        
        chat_service.db.refresh(retrieved)
        assert retrieved.vote_count == 1
        
        # 4. Crear remix
        remix_chat, remix = chat_service.remix_chat(
            original_chat_id=chat_id,
            user_id="remixer-1",
            title="Remix: Lifecycle Test",
            chat_content='{"messages": [{"role": "user", "content": "Remix"}]}'
        )
        
        assert remix.original_chat_id == chat_id
        chat_service.db.refresh(retrieved)
        assert retrieved.remix_count == 1
        
        # 5. Actualizar
        updated = chat_service.update_chat(
            chat_id=chat_id,
            user_id=sample_user_id,
            title="Updated Lifecycle Test",
            description="Updated description"
        )
        
        assert updated.title == "Updated Lifecycle Test"
        
        # 6. Eliminar
        deleted = chat_service.delete_chat(chat_id, sample_user_id)
        assert deleted is True
        
        # Verificar eliminación
        final_check = chat_service.get_chat(chat_id)
        assert final_check is None


class TestBatchOperations:
    """Tests de operaciones en lote"""
    
    @pytest.mark.integration
    def test_batch_feature_operation(self, chat_service, sample_user_id):
        """Test de operación en lote: feature múltiples chats"""
        # Crear múltiples chats
        chat_ids = []
        for i in range(10):
            chat = chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Batch Chat {i}",
                chat_content="{}"
            )
            chat_ids.append(chat.id)
        
        # Feature en lote
        result = chat_service.bulk_operation(chat_ids, "feature")
        
        assert result["operation"] == "feature"
        assert result["successful"] == 10
        assert result["failed"] == 0
        
        # Verificar que están featured
        for chat_id in chat_ids:
            chat = chat_service.get_chat(chat_id)
            assert chat.is_featured is True
    
    @pytest.mark.integration
    def test_batch_delete_operation(self, chat_service, sample_user_id):
        """Test de operación en lote: delete múltiples chats"""
        chat_ids = []
        for i in range(5):
            chat = chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"To Delete {i}",
                chat_content="{}"
            )
            chat_ids.append(chat.id)
        
        result = chat_service.bulk_operation(
            chat_ids,
            "delete",
            user_id=sample_user_id
        )
        
        assert result["successful"] == 5
        
        # Verificar eliminación
        for chat_id in chat_ids:
            chat = chat_service.get_chat(chat_id)
            assert chat is None


class TestSearchAndRanking:
    """Tests de búsqueda y ranking"""
    
    @pytest.mark.integration
    def test_search_with_ranking(self, chat_service, sample_user_id):
        """Test de búsqueda con ranking por score"""
        # Crear chats con diferentes scores
        chats = []
        for i in range(5):
            chat = chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Ranked Chat {i}",
                chat_content="{}",
                tags=["ranked"]
            )
            # Agregar votos para variar scores
            for j in range(i):
                chat_service.vote_chat(chat.id, f"voter-{j}", "upvote")
            chats.append(chat)
        
        # Buscar y verificar orden
        results, total = chat_service.search_chats(
            tags=["ranked"],
            sort_by="score",
            order="desc"
        )
        
        assert total >= 5
        
        # Verificar que están ordenados por score descendente
        scores = [chat.score for chat in results[:5]]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.integration
    def test_trending_chats(self, chat_service, sample_user_id):
        """Test de chats trending"""
        # Crear chats recientes con engagement
        for i in range(10):
            chat = chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Trending {i}",
                chat_content="{}"
            )
            # Agregar engagement
            for j in range(i + 1):
                chat_service.vote_chat(chat.id, f"voter-{j}", "upvote")
        
        trending = chat_service.get_trending_chats(period="day", limit=10)
        
        assert len(trending) <= 10
        assert all(chat.is_public for chat in trending)


class TestUserProfileAndAnalytics:
    """Tests de perfiles de usuario y analytics"""
    
    @pytest.mark.integration
    def test_user_profile_complete(self, chat_service, sample_user_id):
        """Test de perfil completo de usuario"""
        # Crear contenido del usuario
        for i in range(5):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"User Chat {i}",
                chat_content="{}"
            )
        
        # Crear remixes
        original = chat_service.publish_chat(
            user_id="other-user",
            title="Original",
            chat_content="{}"
        )
        
        for i in range(3):
            chat_service.remix_chat(
                original_chat_id=original.id,
                user_id=sample_user_id,
                title=f"Remix {i}",
                chat_content="{}"
            )
        
        # Obtener perfil
        profile = chat_service.get_user_profile(sample_user_id)
        
        assert profile["user_id"] == sample_user_id
        assert profile["total_chats"] == 5
        assert profile["total_remixes"] == 3
    
    @pytest.mark.integration
    def test_analytics_aggregation(self, chat_service):
        """Test de agregación de analytics"""
        # Crear contenido variado
        user_ids = [f"user-{i}" for i in range(5)]
        
        for user_id in user_ids:
            for i in range(3):
                chat = chat_service.publish_chat(
                    user_id=user_id,
                    title=f"Analytics Chat {i}",
                    chat_content="{}",
                    tags=["analytics"]
                )
                # Agregar engagement
                chat_service.vote_chat(chat.id, "voter-1", "upvote")
        
        analytics = chat_service.get_analytics()
        
        assert analytics["total_chats"] >= 15
        assert analytics["total_users"] >= 5
        assert analytics["total_votes"] >= 15


class TestConcurrentOperations:
    """Tests de operaciones concurrentes"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_concurrent_votes(self, chat_service, sample_user_id):
        """Test de votos concurrentes en el mismo chat"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Concurrent Test",
            chat_content="{}"
        )
        
        # Simular votos concurrentes
        initial_votes = chat.vote_count
        
        for i in range(10):
            chat_service.vote_chat(chat.id, f"voter-{i}", "upvote")
        
        chat_service.db.refresh(chat)
        
        # Verificar que todos los votos se registraron
        assert chat.vote_count == initial_votes + 10
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_concurrent_remixes(self, chat_service, sample_user_id):
        """Test de remixes concurrentes"""
        original = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Original for Remixes",
            chat_content="{}"
        )
        
        initial_remixes = original.remix_count
        
        # Crear múltiples remixes
        for i in range(5):
            chat_service.remix_chat(
                original_chat_id=original.id,
                user_id=f"remixer-{i}",
                title=f"Remix {i}",
                chat_content="{}"
            )
        
        chat_service.db.refresh(original)
        
        assert original.remix_count == initial_remixes + 5


class TestPerformanceScenarios:
    """Tests de escenarios de performance"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.performance
    def test_large_search_result(self, chat_service, sample_user_id):
        """Test de búsqueda con muchos resultados"""
        # Crear muchos chats
        for i in range(100):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Search Test {i}",
                chat_content="{}",
                tags=["search-test"]
            )
        
        # Medir tiempo de búsqueda
        start = datetime.utcnow()
        results, total = chat_service.search_chats(
            tags=["search-test"],
            page_size=50
        )
        elapsed = (datetime.utcnow() - start).total_seconds()
        
        assert total >= 100
        assert len(results) == 50
        assert elapsed < 5.0  # Debe ser rápido
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.performance
    def test_ranking_calculation_performance(self, chat_service, ranking_service):
        """Test de performance del cálculo de ranking"""
        # Crear chats con diferentes métricas
        chats = []
        for i in range(50):
            chat = chat_service.publish_chat(
                user_id=f"user-{i % 10}",
                title=f"Ranking Test {i}",
                chat_content="{}"
            )
            chats.append(chat)
        
        # Medir tiempo de cálculo de scores
        start = datetime.utcnow()
        for chat in chats:
            score = ranking_service.calculate_score(
                chat.vote_count,
                chat.remix_count,
                chat.view_count,
                chat.created_at
            )
            chat.score = score
        elapsed = (datetime.utcnow() - start).total_seconds()
        
        assert elapsed < 1.0  # Debe ser muy rápido

