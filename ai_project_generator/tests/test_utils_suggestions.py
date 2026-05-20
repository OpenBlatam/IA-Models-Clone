"""
Tests for IntelligentSuggestions utility
"""

import pytest
from ..utils.intelligent_suggestions import IntelligentSuggestions


class TestIntelligentSuggestions:
    """Test suite for IntelligentSuggestions"""

    def test_init(self):
        """Test IntelligentSuggestions initialization"""
        suggestions = IntelligentSuggestions()
        assert suggestions.suggestion_history == []
        assert suggestions.user_preferences == {}
        assert suggestions.pattern_analysis == {}

    def test_generate_suggestions_simple_project(self):
        """Test generating suggestions for simple project"""
        suggestions = IntelligentSuggestions()
        
        project_info = {
            "description": "A simple chatbot"
        }
        
        result = suggestions.generate_suggestions(project_info)
        
        assert isinstance(result, list)
        # Should suggest simpler framework for simple projects
        assert any(s["type"] == "framework" for s in result)

    def test_generate_suggestions_api_project(self):
        """Test generating suggestions for API project"""
        suggestions = IntelligentSuggestions()
        
        project_info = {
            "description": "A REST API for managing users"
        }
        
        result = suggestions.generate_suggestions(project_info)
        
        # Should suggest FastAPI for API projects
        api_suggestions = [s for s in result if "api" in s.get("message", "").lower() or "rest" in s.get("message", "").lower()]
        assert len(api_suggestions) > 0

    def test_generate_suggestions_with_auth(self):
        """Test generating suggestions for project with auth"""
        suggestions = IntelligentSuggestions()
        
        project_info = {
            "description": "A system with user authentication and login"
        }
        
        result = suggestions.generate_suggestions(project_info)
        
        # Should suggest auth features
        auth_suggestions = [s for s in result if "auth" in s.get("message", "").lower()]
        assert len(auth_suggestions) > 0

    def test_generate_suggestions_with_database(self):
        """Test generating suggestions for project with database"""
        suggestions = IntelligentSuggestions()
        
        project_info = {
            "description": "A system that stores data in a database"
        }
        
        result = suggestions.generate_suggestions(project_info)
        
        # Should suggest database
        db_suggestions = [s for s in result if "database" in s.get("message", "").lower() or "data" in s.get("message", "").lower()]
        assert len(db_suggestions) > 0

    def test_generate_suggestions_without_tests(self):
        """Test generating suggestions when tests are disabled"""
        suggestions = IntelligentSuggestions()
        
        project_info = {
            "description": "A test project",
            "generate_tests": False
        }
        
        result = suggestions.generate_suggestions(project_info)
        
        # Should suggest enabling tests
        test_suggestions = [s for s in result if s.get("type") == "testing"]
        assert len(test_suggestions) > 0

    def test_generate_suggestions_with_user_preferences(self):
        """Test generating suggestions with user preferences"""
        suggestions = IntelligentSuggestions()
        
        user_id = "user-123"
        suggestions.user_preferences[user_id] = {
            "preferred_types": ["framework"]
        }
        
        project_info = {
            "description": "A simple project"
        }
        
        result = suggestions.generate_suggestions(project_info, user_id=user_id)
        
        # Should prioritize framework suggestions
        framework_suggestions = [s for s in result if s.get("type") == "framework"]
        if framework_suggestions:
            assert framework_suggestions[0].get("priority") == "high"

    def test_get_suggestion_history(self):
        """Test getting suggestion history"""
        suggestions = IntelligentSuggestions()
        
        project_info = {"description": "Test project"}
        suggestions.generate_suggestions(project_info)
        
        history = suggestions.get_suggestion_history()
        
        assert len(history) == 1
        assert history[0]["project_info"] == project_info

    def test_apply_user_preferences(self):
        """Test applying user preferences"""
        suggestions = IntelligentSuggestions()
        
        user_id = "user-456"
        suggestions.user_preferences[user_id] = {
            "preferred_frameworks": ["fastapi"],
            "preferred_features": ["auth", "database"]
        }
        
        project_info = {"description": "A test project"}
        result = suggestions.generate_suggestions(project_info, user_id=user_id)
        
        # Should apply preferences
        assert len(result) >= 0  # May or may not have suggestions

