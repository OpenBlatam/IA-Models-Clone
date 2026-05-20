"""
Tests modulares para validación de schemas
"""

import pytest
from pydantic import ValidationError
from tests.helpers.test_helpers import (
    create_publish_request,
    create_remix_request,
    create_vote_request,
    create_search_request
)
from schemas import (
    PublishChatRequest,
    RemixChatRequest,
    VoteRequest,
    SearchRequest,
    UpdateChatRequest,
    CommentRequest
)


class TestPublishChatRequest:
    """Tests para PublishChatRequest"""
    
    @pytest.mark.unit
    def test_publish_request_valid(self):
        """Test con request válido"""
        request_data = create_publish_request()
        request = PublishChatRequest(**request_data)
        
        assert request.title == request_data["title"]
        assert request.chat_content == request_data["chat_content"]
        assert request.is_public == request_data["is_public"]
    
    @pytest.mark.unit
    def test_publish_request_empty_title(self):
        """Test con título vacío"""
        request_data = create_publish_request(title="")
        
        with pytest.raises(ValidationError):
            PublishChatRequest(**request_data)
    
    @pytest.mark.unit
    def test_publish_request_title_too_long(self):
        """Test con título muy largo"""
        request_data = create_publish_request(title="A" * 201)
        
        with pytest.raises(ValidationError):
            PublishChatRequest(**request_data)
    
    @pytest.mark.unit
    def test_publish_request_empty_content(self):
        """Test con contenido vacío"""
        request_data = create_publish_request(chat_content="")
        
        with pytest.raises(ValidationError):
            PublishChatRequest(**request_data)
    
    @pytest.mark.unit
    def test_publish_request_content_too_long(self):
        """Test con contenido muy largo"""
        request_data = create_publish_request(chat_content="A" * 50001)
        
        with pytest.raises(ValidationError):
            PublishChatRequest(**request_data)
    
    @pytest.mark.unit
    def test_publish_request_too_many_tags(self):
        """Test con demasiados tags"""
        request_data = create_publish_request(tags=[f"tag{i}" for i in range(11)])
        
        with pytest.raises(ValidationError):
            PublishChatRequest(**request_data)
    
    @pytest.mark.unit
    def test_publish_request_tags_sanitization(self):
        """Test que los tags se sanitizan correctamente"""
        request_data = create_publish_request(
            tags=["  TAG1  ", "Tag2", "TAG1", "tag3"]  # Duplicados y espacios
        )
        request = PublishChatRequest(**request_data)
        
        # Debe eliminar duplicados y normalizar
        assert request.tags is not None
        assert len(request.tags) <= 3
    
    @pytest.mark.unit
    def test_publish_request_description_optional(self):
        """Test sin descripción"""
        request_data = create_publish_request(description=None)
        request = PublishChatRequest(**request_data)
        
        assert request.description is None
    
    @pytest.mark.unit
    def test_publish_request_description_too_long(self):
        """Test con descripción muy larga"""
        request_data = create_publish_request(description="A" * 1001)
        
        with pytest.raises(ValidationError):
            PublishChatRequest(**request_data)


class TestRemixChatRequest:
    """Tests para RemixChatRequest"""
    
    @pytest.mark.unit
    def test_remix_request_valid(self):
        """Test con request válido"""
        request_data = create_remix_request(original_chat_id="original-123")
        request = RemixChatRequest(**request_data)
        
        assert request.original_chat_id == "original-123"
        assert request.title == request_data["title"]
    
    @pytest.mark.unit
    def test_remix_request_empty_original_id(self):
        """Test con ID original vacío"""
        request_data = create_remix_request(original_chat_id="")
        
        with pytest.raises(ValidationError):
            RemixChatRequest(**request_data)
    
    @pytest.mark.unit
    def test_remix_request_reuses_validators(self):
        """Test que reutiliza validadores de PublishChatRequest"""
        request_data = create_remix_request(
            original_chat_id="original-123",
            title=""  # Título vacío
        )
        
        with pytest.raises(ValidationError):
            RemixChatRequest(**request_data)


class TestVoteRequest:
    """Tests para VoteRequest"""
    
    @pytest.mark.unit
    def test_vote_request_valid_upvote(self):
        """Test con upvote válido"""
        request_data = create_vote_request(chat_id="chat-123", vote_type="upvote")
        request = VoteRequest(**request_data)
        
        assert request.chat_id == "chat-123"
        assert request.vote_type == "upvote"
    
    @pytest.mark.unit
    def test_vote_request_valid_downvote(self):
        """Test con downvote válido"""
        request_data = create_vote_request(chat_id="chat-123", vote_type="downvote")
        request = VoteRequest(**request_data)
        
        assert request.vote_type == "downvote"
    
    @pytest.mark.unit
    def test_vote_request_empty_chat_id(self):
        """Test con chat_id vacío"""
        request_data = create_vote_request(chat_id="")
        
        with pytest.raises(ValidationError):
            VoteRequest(**request_data)
    
    @pytest.mark.unit
    def test_vote_request_invalid_type(self):
        """Test con tipo de voto inválido"""
        request_data = {
            "chat_id": "chat-123",
            "vote_type": "invalid"
        }
        
        with pytest.raises(ValidationError):
            VoteRequest(**request_data)


class TestSearchRequest:
    """Tests para SearchRequest"""
    
    @pytest.mark.unit
    def test_search_request_valid(self):
        """Test con request válido"""
        request_data = create_search_request(query="test", tags=["ai"])
        request = SearchRequest(**request_data)
        
        assert request.query == "test"
        assert request.tags == ["ai"]
    
    @pytest.mark.unit
    def test_search_request_empty(self):
        """Test sin criterios (debe permitir para listar todos)"""
        request_data = create_search_request()
        request = SearchRequest(**request_data)
        
        assert request.query is None
        assert request.tags is None
    
    @pytest.mark.unit
    def test_search_request_query_too_long(self):
        """Test con query muy larga"""
        request_data = create_search_request(query="A" * 201)
        
        with pytest.raises(ValidationError):
            SearchRequest(**request_data)
    
    @pytest.mark.unit
    def test_search_request_invalid_sort_by(self):
        """Test con sort_by inválido"""
        request_data = create_search_request(sort_by="invalid")
        
        with pytest.raises(ValidationError):
            SearchRequest(**request_data)
    
    @pytest.mark.unit
    def test_search_request_invalid_order(self):
        """Test con order inválido"""
        request_data = create_search_request(order="invalid")
        
        with pytest.raises(ValidationError):
            SearchRequest(**request_data)
    
    @pytest.mark.unit
    def test_search_request_page_validation(self):
        """Test de validación de página"""
        # Página mínima
        request_data = create_search_request(page=1)
        request = SearchRequest(**request_data)
        assert request.page == 1
        
        # Página máxima
        request_data = create_search_request(page=1000)
        request = SearchRequest(**request_data)
        assert request.page == 1000
        
        # Página inválida
        request_data = create_search_request(page=0)
        with pytest.raises(ValidationError):
            SearchRequest(**request_data)


class TestUpdateChatRequest:
    """Tests para UpdateChatRequest"""
    
    @pytest.mark.unit
    def test_update_request_valid(self):
        """Test con request válido"""
        request = UpdateChatRequest(title="Updated Title")
        
        assert request.title == "Updated Title"
        assert request.description is None
    
    @pytest.mark.unit
    def test_update_request_all_fields(self):
        """Test actualizando todos los campos"""
        request = UpdateChatRequest(
            title="New Title",
            description="New Description",
            tags=["new", "tags"],
            is_public=False
        )
        
        assert request.title == "New Title"
        assert request.description == "New Description"
        assert request.tags == ["new", "tags"]
        assert request.is_public is False
    
    @pytest.mark.unit
    def test_update_request_no_fields(self):
        """Test sin campos (debe fallar)"""
        with pytest.raises(ValidationError):
            UpdateChatRequest()
    
    @pytest.mark.unit
    def test_update_request_empty_title(self):
        """Test con título vacío"""
        with pytest.raises(ValidationError):
            UpdateChatRequest(title="")


class TestCommentRequest:
    """Tests para CommentRequest"""
    
    @pytest.mark.unit
    def test_comment_request_valid(self):
        """Test con request válido"""
        request = CommentRequest(content="Great chat!")
        
        assert request.content == "Great chat!"
        assert request.parent_comment_id is None
    
    @pytest.mark.unit
    def test_comment_request_with_parent(self):
        """Test con comentario padre"""
        request = CommentRequest(
            content="Reply to comment",
            parent_comment_id="parent-123"
        )
        
        assert request.parent_comment_id == "parent-123"
    
    @pytest.mark.unit
    def test_comment_request_empty_content(self):
        """Test con contenido vacío"""
        with pytest.raises(ValidationError):
            CommentRequest(content="")
    
    @pytest.mark.unit
    def test_comment_request_content_too_long(self):
        """Test con contenido muy largo"""
        with pytest.raises(ValidationError):
            CommentRequest(content="A" * 2001)

