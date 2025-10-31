"""
Tests for service layer
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ..services.blog_service import BlogService
from ..services.user_service import UserService
from ..services.comment_service import CommentService
from ..models.schemas import BlogPostCreate, UserCreate, CommentCreate, PostCategory


@pytest.mark.asyncio
class TestBlogService:
    """Test class for blog service."""
    
    async def test_create_blog_post(self, test_session: AsyncSession, sample_blog_post_data: dict):
        """Test creating a blog post through service."""
        service = BlogService(test_session)
        
        post_data = BlogPostCreate(**sample_blog_post_data)
        author_id = "test-author-id"
        
        post = await service.create_post(post_data, author_id)
        
        assert post.title == sample_blog_post_data["title"]
        assert post.content == sample_blog_post_data["content"]
        assert post.author_id == author_id
        assert post.id is not None
        assert post.slug is not None
    
    async def test_get_blog_post(self, test_session: AsyncSession, sample_blog_post_data: dict):
        """Test getting a blog post through service."""
        service = BlogService(test_session)
        
        # Create a post first
        post_data = BlogPostCreate(**sample_blog_post_data)
        author_id = "test-author-id"
        created_post = await service.create_post(post_data, author_id)
        
        # Get the post
        retrieved_post = await service.get_post(created_post.id)
        
        assert retrieved_post.id == created_post.id
        assert retrieved_post.title == created_post.title
        assert retrieved_post.content == created_post.content
    
    async def test_get_blog_post_not_found(self, test_session: AsyncSession):
        """Test getting a non-existent blog post."""
        service = BlogService(test_session)
        
        with pytest.raises(Exception):  # Should raise NotFoundError
            await service.get_post(99999)
    
    async def test_list_blog_posts(self, test_session: AsyncSession, sample_blog_post_data: dict):
        """Test listing blog posts through service."""
        service = BlogService(test_session)
        
        # Create multiple posts
        for i in range(3):
            post_data = BlogPostCreate(
                title=f"Test Post {i}",
                content=f"Content for test post {i}",
                category=PostCategory.TECHNOLOGY
            )
            await service.create_post(post_data, f"author-{i}")
        
        # List posts
        from ..models.schemas import PaginationParams
        pagination = PaginationParams(page=1, size=10)
        posts, total = await service.list_posts(pagination)
        
        assert len(posts) == 3
        assert total == 3
        assert all(post.title.startswith("Test Post") for post in posts)
    
    async def test_search_blog_posts(self, test_session: AsyncSession, sample_blog_post_data: dict):
        """Test searching blog posts through service."""
        service = BlogService(test_session)
        
        # Create a post
        post_data = BlogPostCreate(**sample_blog_post_data)
        await service.create_post(post_data, "test-author")
        
        # Search posts
        from ..models.schemas import SearchParams, PaginationParams
        search_params = SearchParams(query="test")
        pagination = PaginationParams(page=1, size=10)
        
        posts, total = await service.search_posts(search_params, pagination)
        
        assert total >= 1
        assert any("test" in post.title.lower() for post in posts)


@pytest.mark.asyncio
class TestUserService:
    """Test class for user service."""
    
    async def test_create_user(self, test_session: AsyncSession, sample_user_data: dict):
        """Test creating a user through service."""
        service = UserService(test_session)
        
        user_data = UserCreate(**sample_user_data)
        user = await service.create_user(user_data)
        
        assert user.email == sample_user_data["email"]
        assert user.username == sample_user_data["username"]
        assert user.full_name == sample_user_data["full_name"]
        assert user.id is not None
    
    async def test_get_user_by_email(self, test_session: AsyncSession, sample_user_data: dict):
        """Test getting user by email through service."""
        service = UserService(test_session)
        
        # Create user first
        user_data = UserCreate(**sample_user_data)
        created_user = await service.create_user(user_data)
        
        # Get user by email
        retrieved_user = await service.get_user_by_email(sample_user_data["email"])
        
        assert retrieved_user is not None
        assert retrieved_user.id == created_user.id
        assert retrieved_user.email == created_user.email
    
    async def test_authenticate_user(self, test_session: AsyncSession, sample_user_data: dict):
        """Test user authentication through service."""
        service = UserService(test_session)
        
        # Create user first
        user_data = UserCreate(**sample_user_data)
        await service.create_user(user_data)
        
        # Authenticate user
        authenticated_user = await service.authenticate_user(
            sample_user_data["email"],
            sample_user_data["password"]
        )
        
        assert authenticated_user is not None
        assert authenticated_user.email == sample_user_data["email"]


@pytest.mark.asyncio
class TestCommentService:
    """Test class for comment service."""
    
    async def test_create_comment(self, test_session: AsyncSession, sample_comment_data: dict):
        """Test creating a comment through service."""
        service = CommentService(test_session)
        
        # First create a blog post
        blog_service = BlogService(test_session)
        post_data = BlogPostCreate(
            title="Test Post for Comment",
            content="Test content for comment testing",
            category=PostCategory.TECHNOLOGY
        )
        post = await blog_service.create_post(post_data, "test-author")
        
        # Create comment
        comment_data = CommentCreate(**sample_comment_data)
        comment = await service.create_comment(comment_data, post.id, "test-user")
        
        assert comment.content == sample_comment_data["content"]
        assert comment.post_id == post.id
        assert comment.author_id == "test-user"
        assert comment.id is not None
    
    async def test_list_comments(self, test_session: AsyncSession, sample_comment_data: dict):
        """Test listing comments through service."""
        service = CommentService(test_session)
        
        # Create blog post and comments
        blog_service = BlogService(test_session)
        post_data = BlogPostCreate(
            title="Test Post for Comments",
            content="Test content",
            category=PostCategory.TECHNOLOGY
        )
        post = await blog_service.create_post(post_data, "test-author")
        
        # Create multiple comments
        for i in range(3):
            comment_data = CommentCreate(content=f"Test comment {i}")
            await service.create_comment(comment_data, post.id, f"user-{i}")
        
        # List comments
        from ..models.schemas import PaginationParams
        pagination = PaginationParams(page=1, size=10)
        comments, total = await service.list_comments(post.id, pagination)
        
        assert len(comments) == 3
        assert total == 3






























