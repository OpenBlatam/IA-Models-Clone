"""
Tests avanzados para ChatService
"""

import pytest
from datetime import datetime, timedelta
from tests.helpers.test_helpers import create_chat_dict, generate_chat_id, generate_user_id
from services import ChatService
from exceptions import ChatNotFoundError, InvalidChatError


class TestChatServiceAdvanced:
    """Tests avanzados para ChatService"""
    
    @pytest.mark.unit
    def test_update_chat_success(self, chat_service, sample_user_id):
        """Test exitoso de actualización"""
        # Publicar chat
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Original Title",
            chat_content="{}"
        )
        
        # Actualizar
        updated = chat_service.update_chat(
            chat_id=chat.id,
            user_id=sample_user_id,
            title="Updated Title",
            description="Updated description",
            tags=["updated", "tags"]
        )
        
        assert updated.title == "Updated Title"
        assert updated.description == "Updated description"
        assert "updated" in updated.tags.lower()
    
    @pytest.mark.unit
    def test_update_chat_not_owner(self, chat_service, sample_user_id):
        """Test actualizando chat de otro usuario"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        with pytest.raises(InvalidChatError):
            chat_service.update_chat(
                chat_id=chat.id,
                user_id="other-user",
                title="Hacked Title"
            )
    
    @pytest.mark.unit
    def test_delete_chat_success(self, chat_service, sample_user_id):
        """Test exitoso de eliminación"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="To Delete",
            chat_content="{}"
        )
        
        result = chat_service.delete_chat(chat.id, sample_user_id)
        
        assert result is True
        
        # Verificar que fue eliminado
        deleted = chat_service.get_chat(chat.id)
        assert deleted is None
    
    @pytest.mark.unit
    def test_delete_chat_not_owner(self, chat_service, sample_user_id):
        """Test eliminando chat de otro usuario"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        with pytest.raises(InvalidChatError):
            chat_service.delete_chat(chat.id, "other-user")
    
    @pytest.mark.unit
    def test_feature_chat(self, chat_service, sample_user_id):
        """Test destacar chat"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        featured = chat_service.feature_chat(chat.id, True)
        
        assert featured.is_featured is True
    
    @pytest.mark.unit
    def test_get_user_profile(self, chat_service, sample_user_id):
        """Test obtener perfil de usuario"""
        # Publicar algunos chats
        for i in range(3):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Chat {i}",
                chat_content="{}"
            )
        
        profile = chat_service.get_user_profile(sample_user_id)
        
        assert profile["user_id"] == sample_user_id
        assert profile["total_chats"] == 3
        assert isinstance(profile["total_chats"], int)
    
    @pytest.mark.unit
    def test_get_trending_chats(self, chat_service, sample_user_id):
        """Test obtener chats trending"""
        # Publicar chats recientes
        for i in range(5):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Trending Chat {i}",
                chat_content="{}"
            )
        
        trending = chat_service.get_trending_chats(period="day", limit=10)
        
        assert isinstance(trending, list)
        assert len(trending) <= 10
    
    @pytest.mark.unit
    def test_get_trending_chats_invalid_period(self, chat_service):
        """Test con período inválido"""
        with pytest.raises(InvalidChatError):
            chat_service.get_trending_chats(period="invalid")
    
    @pytest.mark.unit
    def test_get_analytics(self, chat_service, sample_user_id):
        """Test obtener analytics"""
        # Publicar algunos chats
        for i in range(5):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Chat {i}",
                chat_content="{}",
                tags=["test"]
            )
        
        analytics = chat_service.get_analytics()
        
        assert "total_chats" in analytics
        assert "total_users" in analytics
        assert analytics["total_chats"] >= 5
    
    @pytest.mark.unit
    def test_get_chat_stats_detailed(self, chat_service, sample_user_id):
        """Test obtener estadísticas detalladas"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        # Agregar algunos votos
        chat_service.vote_chat(chat.id, "voter-1", "upvote")
        chat_service.vote_chat(chat.id, "voter-2", "upvote")
        chat_service.vote_chat(chat.id, "voter-3", "downvote")
        
        stats = chat_service.get_chat_stats_detailed(chat.id)
        
        assert stats["chat_id"] == chat.id
        assert stats["upvote_count"] == 2
        assert stats["downvote_count"] == 1
        assert "rank" in stats
        assert stats["rank"] >= 1
    
    @pytest.mark.unit
    def test_bulk_operation_feature(self, chat_service, sample_user_id):
        """Test operación en lote: feature"""
        # Crear chats
        chat_ids = []
        for i in range(5):
            chat = chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Chat {i}",
                chat_content="{}"
            )
            chat_ids.append(chat.id)
        
        result = chat_service.bulk_operation(chat_ids, "feature")
        
        assert result["operation"] == "feature"
        assert result["successful"] == 5
        assert result["failed"] == 0
    
    @pytest.mark.unit
    def test_bulk_operation_delete(self, chat_service, sample_user_id):
        """Test operación en lote: delete"""
        chat_ids = []
        for i in range(3):
            chat = chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Chat {i}",
                chat_content="{}"
            )
            chat_ids.append(chat.id)
        
        result = chat_service.bulk_operation(chat_ids, "delete", user_id=sample_user_id)
        
        assert result["operation"] == "delete"
        assert result["successful"] == 3
    
    @pytest.mark.unit
    def test_bulk_operation_invalid_operation(self, chat_service):
        """Test con operación inválida"""
        with pytest.raises(InvalidChatError):
            chat_service.bulk_operation(["chat-1"], "invalid_operation")
    
    @pytest.mark.unit
    def test_bulk_operation_empty_list(self, chat_service):
        """Test con lista vacía"""
        with pytest.raises(InvalidChatError):
            chat_service.bulk_operation([], "feature")
    
    @pytest.mark.unit
    def test_get_remixes(self, chat_service, sample_user_id):
        """Test obtener remixes de un chat"""
        original = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Original",
            chat_content="{}"
        )
        
        # Crear remixes
        for i in range(3):
            chat_service.remix_chat(
                original_chat_id=original.id,
                user_id=f"remixer-{i}",
                title=f"Remix {i}",
                chat_content="{}"
            )
        
        remixes = chat_service.get_remixes(original.id)
        
        assert len(remixes) == 3
        assert all(remix.original_chat_id == original.id for remix in remixes)
    
    @pytest.mark.unit
    def test_get_remixes_not_found(self, chat_service):
        """Test obtener remixes de chat inexistente"""
        with pytest.raises(ChatNotFoundError):
            chat_service.get_remixes("nonexistent-id")
    
    @pytest.mark.unit
    def test_get_user_vote(self, chat_service, sample_user_id):
        """Test obtener voto de usuario"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        # Votar
        chat_service.vote_chat(chat.id, "voter-123", "upvote")
        
        # Obtener voto
        vote = chat_service.get_user_vote(chat.id, "voter-123")
        
        assert vote is not None
        assert vote.vote_type == "upvote"
    
    @pytest.mark.unit
    def test_get_user_vote_not_found(self, chat_service, sample_user_id):
        """Test obtener voto inexistente"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        vote = chat_service.get_user_vote(chat.id, "non-voter")
        
        assert vote is None

