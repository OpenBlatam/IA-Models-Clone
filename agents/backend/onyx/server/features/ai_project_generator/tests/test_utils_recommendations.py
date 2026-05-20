"""
Tests for RecommendationEngine utility
"""

import pytest
from ..utils.recommendation_engine import RecommendationEngine


class TestRecommendationEngine:
    """Test suite for RecommendationEngine"""

    def test_init(self):
        """Test RecommendationEngine initialization"""
        engine = RecommendationEngine()
        assert engine.project_patterns == {}
        assert engine.user_preferences == {}

    def test_learn_from_project(self):
        """Test learning from a project"""
        engine = RecommendationEngine()
        
        project_info = {
            "ai_type": "chat",
            "features": ["auth", "database", "websocket"],
            "backend_framework": "fastapi",
            "frontend_framework": "react"
        }
        
        engine.learn_from_project(project_info, success=True)
        
        assert "chat" in engine.project_patterns
        pattern = engine.project_patterns["chat"]
        assert pattern["count"] == 1
        assert pattern["features"]["auth"] == 1
        assert pattern["features"]["database"] == 1
        assert pattern["success_rate"] == 1.0

    def test_learn_from_multiple_projects(self):
        """Test learning from multiple projects"""
        engine = RecommendationEngine()
        
        for i in range(10):
            project_info = {
                "ai_type": "chat",
                "features": ["auth"] if i % 2 == 0 else ["database"],
                "backend_framework": "fastapi",
                "frontend_framework": "react"
            }
            engine.learn_from_project(project_info, success=True)
        
        pattern = engine.project_patterns["chat"]
        assert pattern["count"] == 10
        assert pattern["features"]["auth"] == 5
        assert pattern["features"]["database"] == 5

    def test_learn_success_rate(self):
        """Test learning success rate"""
        engine = RecommendationEngine()
        
        project_info = {
            "ai_type": "vision",
            "features": [],
            "backend_framework": "fastapi",
            "frontend_framework": "react"
        }
        
        # Learn from 5 successful and 2 failed
        for i in range(5):
            engine.learn_from_project(project_info, success=True)
        for i in range(2):
            engine.learn_from_project(project_info, success=False)
        
        pattern = engine.project_patterns["vision"]
        assert pattern["count"] == 7
        assert pattern["success_rate"] == pytest.approx(5/7, abs=0.1)

    def test_recommend_features(self):
        """Test recommending features"""
        engine = RecommendationEngine()
        
        # Learn from projects
        for i in range(10):
            project_info = {
                "ai_type": "chat",
                "features": ["auth", "database"] if i < 7 else ["websocket"],
                "backend_framework": "fastapi",
                "frontend_framework": "react"
            }
            engine.learn_from_project(project_info, success=True)
        
        recommendations = engine.recommend_features("chat", limit=3)
        
        assert len(recommendations) <= 3
        assert "auth" in recommendations or "database" in recommendations

    def test_recommend_features_unknown_type(self):
        """Test recommending features for unknown AI type"""
        engine = RecommendationEngine()
        
        recommendations = engine.recommend_features("unknown_type")
        
        assert recommendations == []

    def test_recommend_framework(self):
        """Test recommending framework"""
        engine = RecommendationEngine()
        
        # Learn from projects
        for i in range(10):
            project_info = {
                "ai_type": "chat",
                "features": [],
                "backend_framework": "fastapi" if i < 8 else "flask",
                "frontend_framework": "react"
            }
            engine.learn_from_project(project_info, success=True)
        
        recommendation = engine.recommend_framework("chat")
        
        assert recommendation is not None
        assert "fastapi" in recommendation.lower()

    def test_recommend_similar_projects(self):
        """Test recommending similar projects"""
        engine = RecommendationEngine()
        
        # Learn from projects
        projects = [
            {"ai_type": "chat", "features": ["auth"], "backend_framework": "fastapi"},
            {"ai_type": "chat", "features": ["auth", "database"], "backend_framework": "fastapi"},
            {"ai_type": "vision", "features": ["file_upload"], "backend_framework": "fastapi"},
        ]
        
        for project in projects:
            engine.learn_from_project(project, success=True)
        
        similar = engine.recommend_similar_projects(
            {"ai_type": "chat", "features": ["auth"]},
            limit=2
        )
        
        assert len(similar) <= 2

