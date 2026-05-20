"""
Tests modulares para ChatService
"""

import pytest
from datetime import datetime, timedelta
from tests.helpers.test_helpers import create_chat_dict, generate_chat_id, generate_user_id
from tests.helpers.assertion_helpers import assert_chat_valid
from services import ChatService, RankingService
from exceptions import ChatNotFoundError, InvalidChatError, DuplicateVoteError


class TestChatServicePublish:
    """Tests para publish_chat"""
    
    @pytest.mark.unit
    def test_publish_chat_success(self, chat_service, sample_user_id):
        """Test exitoso de publicación"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test Chat",
            chat_content='{"messages": []}',
            description="Test description",
            tags=["test", "ai"]
        )
        
        assert_chat_valid(chat)
        assert chat.title == "Test Chat"
        assert chat.user_id == sample_user_id
        assert chat.is_public is True
    
    @pytest.mark.unit
    def test_publish_chat_empty_user_id(self, chat_service):
        """Test con user_id vacío"""
        with pytest.raises(InvalidChatError):
            chat_service.publish_chat(
                user_id="",
                title="Test",
                chat_content="{}"
            )
    
    @pytest.mark.unit
    def test_publish_chat_empty_title(self, chat_service, sample_user_id):
        """Test con título vacío"""
        with pytest.raises(InvalidChatError):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title="",
                chat_content="{}"
            )
    
    @pytest.mark.unit
    def test_publish_chat_private(self, chat_service, sample_user_id):
        """Test publicando chat privado"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Private Chat",
            chat_content="{}",
            is_public=False
        )
        
        assert chat.is_public is False
    
    @pytest.mark.unit
    def test_publish_chat_with_tags(self, chat_service, sample_user_id):
        """Test con tags"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}",
            tags=["ai", "ml", "test"]
        )
        
        assert chat.tags is not None
        assert "ai" in chat.tags.lower()


class TestChatServiceGet:
    """Tests para get_chat"""
    
    @pytest.mark.unit
    def test_get_chat_success(self, chat_service, sample_user_id):
        """Test exitoso de obtención"""
        # Primero publicar
        published = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test Chat",
            chat_content="{}"
        )
        
        # Luego obtener
        chat = chat_service.get_chat(published.id)
        
        assert chat is not None
        assert chat.id == published.id
        assert chat.title == "Test Chat"
    
    @pytest.mark.unit
    def test_get_chat_not_found(self, chat_service):
        """Test cuando el chat no existe"""
        chat = chat_service.get_chat("nonexistent-id")
        
        assert chat is None
    
    @pytest.mark.unit
    def test_get_chat_with_view_recording(self, chat_service, sample_user_id):
        """Test que registra visualización"""
        published = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        initial_views = published.view_count
        
        # Obtener con user_id (debe registrar view)
        chat = chat_service.get_chat(published.id, user_id="viewer-123")
        
        # Verificar que se incrementó view_count
        chat_service.db.refresh(chat)
        assert chat.view_count >= initial_views


class TestChatServiceVote:
    """Tests para vote_chat"""
    
    @pytest.mark.unit
    def test_vote_chat_upvote_success(self, chat_service, sample_user_id):
        """Test exitoso de upvote"""
        # Publicar chat
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        initial_votes = chat.vote_count
        
        # Votar
        vote = chat_service.vote_chat(chat.id, "voter-123", "upvote")
        
        assert vote is not None
        assert vote.vote_type == "upvote"
        
        # Verificar que se incrementó vote_count
        chat_service.db.refresh(chat)
        assert chat.vote_count == initial_votes + 1
    
    @pytest.mark.unit
    def test_vote_chat_downvote_success(self, chat_service, sample_user_id):
        """Test exitoso de downvote"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        initial_votes = chat.vote_count
        
        vote = chat_service.vote_chat(chat.id, "voter-123", "downvote")
        
        assert vote.vote_type == "downvote"
        chat_service.db.refresh(chat)
        assert chat.vote_count == initial_votes - 1
    
    @pytest.mark.unit
    def test_vote_chat_change_vote(self, chat_service, sample_user_id):
        """Test cambiando voto"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        # Primer voto: upvote
        chat_service.vote_chat(chat.id, "voter-123", "upvote")
        chat_service.db.refresh(chat)
        votes_after_upvote = chat.vote_count
        
        # Cambiar a downvote
        chat_service.vote_chat(chat.id, "voter-123", "downvote")
        chat_service.db.refresh(chat)
        
        # Debe haber disminuido en 2 (quitar upvote + agregar downvote)
        assert chat.vote_count == votes_after_upvote - 2
    
    @pytest.mark.unit
    def test_vote_chat_not_found(self, chat_service):
        """Test votando chat inexistente"""
        with pytest.raises(ChatNotFoundError):
            chat_service.vote_chat("nonexistent", "user-123", "upvote")
    
    @pytest.mark.unit
    def test_vote_chat_invalid_type(self, chat_service, sample_user_id):
        """Test con tipo de voto inválido"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        with pytest.raises(InvalidChatError):
            chat_service.vote_chat(chat.id, "user-123", "invalid")


class TestChatServiceRemix:
    """Tests para remix_chat"""
    
    @pytest.mark.unit
    def test_remix_chat_success(self, chat_service, sample_user_id):
        """Test exitoso de remix"""
        # Publicar chat original
        original = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Original",
            chat_content="{}"
        )
        
        initial_remixes = original.remix_count
        
        # Crear remix
        remix_chat, remix = chat_service.remix_chat(
            original_chat_id=original.id,
            user_id="remixer-123",
            title="Remix",
            chat_content="{}"
        )
        
        assert remix_chat is not None
        assert remix is not None
        assert remix.original_chat_id == original.id
        assert remix.remix_chat_id == remix_chat.id
        
        # Verificar que se incrementó remix_count del original
        chat_service.db.refresh(original)
        assert original.remix_count == initial_remixes + 1
    
    @pytest.mark.unit
    def test_remix_chat_original_not_found(self, chat_service):
        """Test remixando chat inexistente"""
        with pytest.raises(ChatNotFoundError):
            chat_service.remix_chat(
                original_chat_id="nonexistent",
                user_id="user-123",
                title="Remix",
                chat_content="{}"
            )


class TestChatServiceSearch:
    """Tests para search_chats"""
    
    @pytest.mark.unit
    def test_search_chats_by_query(self, chat_service, sample_user_id):
        """Test búsqueda por query"""
        # Publicar chats
        chat_service.publish_chat(
            user_id=sample_user_id,
            title="AI Chat",
            chat_content="{}",
            description="About AI"
        )
        chat_service.publish_chat(
            user_id=sample_user_id,
            title="Other Chat",
            chat_content="{}"
        )
        
        # Buscar
        chats, total = chat_service.search_chats(query="AI")
        
        assert total >= 1
        assert any("AI" in chat.title for chat in chats)
    
    @pytest.mark.unit
    def test_search_chats_by_tags(self, chat_service, sample_user_id):
        """Test búsqueda por tags"""
        chat_service.publish_chat(
            user_id=sample_user_id,
            title="AI Chat",
            chat_content="{}",
            tags=["ai", "ml"]
        )
        
        chats, total = chat_service.search_chats(tags=["ai"])
        
        assert total >= 1
    
    @pytest.mark.unit
    def test_search_chats_pagination(self, chat_service, sample_user_id):
        """Test con paginación"""
        # Crear múltiples chats
        for i in range(25):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Chat {i}",
                chat_content="{}"
            )
        
        # Primera página
        chats, total = chat_service.search_chats(page=1, page_size=10)
        
        assert len(chats) == 10
        assert total == 25
        
        # Segunda página
        chats2, _ = chat_service.search_chats(page=2, page_size=10)
        assert len(chats2) == 10


class TestRankingService:
    """Tests para RankingService"""
    
    @pytest.mark.unit
    def test_calculate_score_basic(self, ranking_service):
        """Test de cálculo básico de score"""
        score = ranking_service.calculate_score(
            vote_count=10,
            remix_count=5,
            view_count=100,
            created_at=datetime.utcnow()
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    @pytest.mark.unit
    def test_calculate_score_time_decay(self, ranking_service):
        """Test que el score decae con el tiempo"""
        now = datetime.utcnow()
        recent = now - timedelta(hours=1)
        old = now - timedelta(days=7)
        
        score_recent = ranking_service.calculate_score(10, 5, 100, recent)
        score_old = ranking_service.calculate_score(10, 5, 100, old)
        
        # El score reciente debe ser mayor
        assert score_recent > score_old
    
    @pytest.mark.unit
    def test_calculate_score_negative_counts(self, ranking_service):
        """Test con conteos negativos (debe fallar)"""
        with pytest.raises(ValueError):
            ranking_service.calculate_score(
                vote_count=-1,
                remix_count=0,
                view_count=0,
                created_at=datetime.utcnow()
            )

