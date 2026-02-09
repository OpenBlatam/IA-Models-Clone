from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import sys, pathlib
import pytest
from uuid import uuid4
from datetime import datetime, timedelta
from onyx.server.features.linkedin_posts.core.domain.entities.linkedin_post_refactored import LinkedInPost, PostStatus, PostType, PostTone
from onyx.server.features.linkedin_posts.core.domain.value_objects.content import Content
from onyx.server.features.linkedin_posts.core.domain.value_objects.author import Author
from onyx.server.features.linkedin_posts.core.domain.exceptions.post_exceptions import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
sys.path.append(str(pathlib.Path(__file__).resolve().parents[4]))


    ContentValidationError,
    InvalidPostStateError,
    PostAlreadyPublishedError,
)

author = Author(uuid4(), "Test Author")

def test_create_draft_post():
    
    """test_create_draft_post function."""
content = Content("This is a valid LinkedIn post content with more than ten characters.")
    post = LinkedInPost(content=content, author=author)
    assert post.status == PostStatus.DRAFT
    assert post.is_draft()
    assert not post.is_published()


def test_publish_post_changes_status():
    
    """test_publish_post_changes_status function."""
content = Content("Publishing a draft post should change its status to published.")
    post = LinkedInPost(content=content, author=author)
    post.publish()
    assert post.status == PostStatus.PUBLISHED
    assert post.published_at is not None
    assert post.is_published()


def test_publish_already_published_post_raises():
    
    """test_publish_already_published_post_raises function."""
content = Content("Duplicate publish should raise error.")
    post = LinkedInPost(content=content, author=author)
    post.publish()
    with pytest.raises(PostAlreadyPublishedError):
        post.publish()


def test_update_content_on_published_post_fails():
    
    """test_update_content_on_published_post_fails function."""
content = Content("Original content")
    post = LinkedInPost(content=content, author=author)
    post.publish()
    with pytest.raises(InvalidPostStateError):
        post.update_content("New content that should fail")


def test_invalid_content_too_short():
    
    """test_invalid_content_too_short function."""
with pytest.raises(ValueError):
        Content("short")


def test_schedule_post():
    
    """test_schedule_post function."""
content = Content("Scheduling a draft post for future publication.")
    post = LinkedInPost(content=content, author=author)
    future_time = datetime.utcnow() + timedelta(hours=1)
    post.schedule(future_time)
    assert post.status == PostStatus.SCHEDULED
    assert post.metadata.scheduled_at == future_time 