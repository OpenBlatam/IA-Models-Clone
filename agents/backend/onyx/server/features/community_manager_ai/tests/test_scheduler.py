"""
Tests for Post Scheduler
========================
"""

import pytest
from datetime import datetime, timedelta
from community_manager_ai.core.scheduler import PostScheduler


def test_add_post():
    """Test agregar un post"""
    scheduler = PostScheduler()
    
    post_data = {
        "content": "Test post",
        "platforms": ["facebook"],
        "scheduled_time": datetime.now() + timedelta(hours=1)
    }
    
    post_id = scheduler.add_post(post_data)
    assert post_id is not None
    
    post = scheduler.get_post(post_id)
    assert post is not None
    assert post["content"] == "Test post"


def test_get_pending_posts():
    """Test obtener posts pendientes"""
    scheduler = PostScheduler()
    
    # Agregar post en el pasado (debe aparecer como pendiente)
    past_time = datetime.now() - timedelta(minutes=5)
    post_data = {
        "content": "Past post",
        "platforms": ["facebook"],
        "scheduled_time": past_time
    }
    scheduler.add_post(post_data)
    
    # Agregar post en el futuro (no debe aparecer)
    future_time = datetime.now() + timedelta(hours=1)
    post_data = {
        "content": "Future post",
        "platforms": ["facebook"],
        "scheduled_time": future_time
    }
    scheduler.add_post(post_data)
    
    pending = scheduler.get_pending_posts()
    assert len(pending) >= 1
    assert any(p["content"] == "Past post" for p in pending)


def test_mark_as_published():
    """Test marcar post como publicado"""
    scheduler = PostScheduler()
    
    post_data = {
        "content": "Test post",
        "platforms": ["facebook"],
        "scheduled_time": datetime.now()
    }
    
    post_id = scheduler.add_post(post_data)
    results = {"facebook": {"status": "success"}}
    
    scheduler.mark_as_published(post_id, results)
    
    post = scheduler.get_post(post_id)
    assert post["status"] == "published"
    assert "published_at" in post


def test_cancel_post():
    """Test cancelar un post"""
    scheduler = PostScheduler()
    
    post_data = {
        "content": "Test post",
        "platforms": ["facebook"],
        "scheduled_time": datetime.now() + timedelta(hours=1)
    }
    
    post_id = scheduler.add_post(post_data)
    success = scheduler.cancel_post(post_id)
    
    assert success is True
    
    post = scheduler.get_post(post_id)
    assert post["status"] == "cancelled"




