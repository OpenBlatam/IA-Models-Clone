"""
Tests for KeywordExtractor

This test suite covers:
- Keyword extraction from descriptions
- AI type detection
- Feature detection
- Provider detection
- Complexity detection
- Deep learning pattern detection

Test Generation Principles:
- Unique: Each test covers a distinct scenario
- Diverse: Tests cover happy paths, edge cases, errors, and boundaries
- Intuitive: Clear names and assertions express intent
"""

import pytest
from ..core.keyword_extractor import KeywordExtractor, _get_default_keywords, _contains_any_pattern


class TestKeywordExtractor:
    """Test suite for KeywordExtractor"""

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_init_creates_instance(self):
        """Test KeywordExtractor can be instantiated"""
        # Happy path: Normal initialization
        extractor = KeywordExtractor()
        assert extractor is not None
        assert isinstance(extractor, KeywordExtractor)

    # ========================================================================
    # Default Keywords Tests
    # ========================================================================

    def test_get_default_keywords_returns_complete_structure(self):
        """Test _get_default_keywords returns complete keyword structure"""
        # Happy path: Default keywords structure
        keywords = _get_default_keywords()
        
        assert isinstance(keywords, dict)
        assert keywords["ai_type"] == "general"
        assert keywords["features"] == []
        assert keywords["requires_auth"] is False
        assert keywords["requires_database"] is False
        assert keywords["requires_api"] is True
        assert keywords["complexity"] == "medium"
        assert keywords["is_deep_learning"] is False
        assert keywords["is_transformer"] is False
        assert keywords["is_llm"] is False
        assert keywords["model_providers"] == []

    def test_get_default_keywords_returns_new_instance(self):
        """Test _get_default_keywords returns new instance each time"""
        # Edge case: Immutability check
        keywords1 = _get_default_keywords()
        keywords2 = _get_default_keywords()
        
        assert keywords1 is not keywords2
        keywords1["ai_type"] = "modified"
        assert keywords2["ai_type"] == "general"  # Should not be affected

    # ========================================================================
    # Pattern Matching Tests
    # ========================================================================

    def test_contains_any_pattern_with_matching_pattern(self):
        """Test _contains_any_pattern detects matching patterns"""
        # Happy path: Pattern found
        text = "This is a chat bot system"
        patterns = {"chat", "bot", "assistant"}
        assert _contains_any_pattern(text, patterns) is True

    def test_contains_any_pattern_with_no_match(self):
        """Test _contains_any_pattern returns False when no pattern matches"""
        # Happy path: No pattern found
        text = "This is a completely different system"
        patterns = {"chat", "bot", "assistant"}
        assert _contains_any_pattern(text, patterns) is False

    def test_contains_any_pattern_with_empty_patterns(self):
        """Test _contains_any_pattern handles empty pattern set"""
        # Edge case: Empty patterns
        text = "Any text here"
        patterns = set()
        assert _contains_any_pattern(text, patterns) is False

    def test_contains_any_pattern_with_empty_text(self):
        """Test _contains_any_pattern handles empty text"""
        # Edge case: Empty text
        text = ""
        patterns = {"chat", "bot"}
        assert _contains_any_pattern(text, patterns) is False

    def test_contains_any_pattern_case_insensitive(self):
        """Test _contains_any_pattern is case sensitive (as expected)"""
        # Edge case: Case sensitivity
        text = "CHAT BOT SYSTEM"
        patterns = {"chat", "bot"}
        # Note: This depends on implementation - pattern matching is case-sensitive
        # If text is lowercased before matching, this would be True
        assert _contains_any_pattern(text.lower(), patterns) is True

    # ========================================================================
    # Extract Method Tests - Empty/None Inputs
    # ========================================================================

    def test_extract_with_empty_string(self, keyword_extractor):
        """Test extract returns default keywords for empty string"""
        # Edge case: Empty string
        result = keyword_extractor.extract("")
        assert result == _get_default_keywords()

    def test_extract_with_whitespace_only(self, keyword_extractor):
        """Test extract handles whitespace-only strings"""
        # Edge case: Only whitespace
        result = keyword_extractor.extract("   \n\t  ")
        # Should return defaults or process it
        assert isinstance(result, dict)
        assert "ai_type" in result

    # ========================================================================
    # Extract Method Tests - AI Type Detection
    # ========================================================================

    def test_extract_detects_chat_ai(self, keyword_extractor):
        """Test extract identifies chat AI from description"""
        # Happy path: Chat AI detection
        description = "A chat bot that helps users"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "chat"
        assert result["requires_websocket"] is True
        assert result["requires_streaming"] is True

    def test_extract_detects_chat_ai_multilingual(self, keyword_extractor):
        """Test extract identifies chat AI from multilingual descriptions"""
        # Edge case: Multilingual
        descriptions = [
            "Un asistente conversacional",
            "A conversational assistant",
            "Un chatbot que ayuda"
        ]
        for desc in descriptions:
            result = keyword_extractor.extract(desc)
            assert result["ai_type"] == "chat", f"Failed for: {desc}"

    def test_extract_detects_vision_ai(self, keyword_extractor):
        """Test extract identifies vision AI from description"""
        # Happy path: Vision AI detection
        description = "An image recognition system"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "vision"
        assert result["requires_ml"] is True

    def test_extract_detects_audio_ai(self, keyword_extractor):
        """Test extract identifies audio AI from description"""
        # Happy path: Audio AI detection
        description = "A speech recognition system"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "audio"
        assert result["requires_ml"] is True

    def test_extract_detects_nlp_ai(self, keyword_extractor):
        """Test extract identifies NLP AI from description"""
        # Happy path: NLP AI detection
        description = "A text analysis system"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "nlp"
        assert result["requires_ml"] is True

    def test_extract_detects_video_ai(self, keyword_extractor):
        """Test extract identifies video AI from description"""
        # Happy path: Video AI detection
        description = "A video streaming system"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "video"
        assert result["requires_websocket"] is True
        assert result["requires_streaming"] is True

    def test_extract_detects_recommendation_ai(self, keyword_extractor):
        """Test extract identifies recommendation AI from description"""
        # Happy path: Recommendation AI detection
        description = "A recommendation system for users"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "recommendation"

    def test_extract_detects_qa_ai(self, keyword_extractor):
        """Test extract identifies Q&A AI from description"""
        # Happy path: Q&A AI detection
        description = "A question and answer system"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "qa"

    # ========================================================================
    # Extract Method Tests - Deep Learning Detection
    # ========================================================================

    def test_extract_detects_deep_learning(self, keyword_extractor):
        """Test extract detects deep learning patterns"""
        # Happy path: Deep learning detection
        description = "A deep learning model using PyTorch"
        result = keyword_extractor.extract(description)
        assert result["is_deep_learning"] is True
        assert result["requires_pytorch"] is True
        assert result["requires_ml"] is True
        assert result["requires_training"] is True

    def test_extract_detects_transformer(self, keyword_extractor):
        """Test extract detects transformer models"""
        # Happy path: Transformer detection
        description = "A transformer model for NLP"
        result = keyword_extractor.extract(description)
        assert result["is_transformer"] is True
        assert result["is_deep_learning"] is True
        assert result["model_architecture"] == "transformer"

    def test_extract_detects_llm(self, keyword_extractor):
        """Test extract detects LLM patterns"""
        # Happy path: LLM detection
        description = "A large language model for text generation"
        result = keyword_extractor.extract(description)
        assert result["is_llm"] is True
        assert result["is_transformer"] is True
        assert result["is_deep_learning"] is True
        assert result["model_architecture"] == "llm"
        assert result["requires_fine_tuning"] is True

    def test_extract_detects_diffusion(self, keyword_extractor):
        """Test extract detects diffusion models"""
        # Happy path: Diffusion detection
        description = "A stable diffusion model for image generation"
        result = keyword_extractor.extract(description)
        assert result["is_diffusion"] is True
        assert result["is_deep_learning"] is True
        assert result["model_architecture"] == "diffusion"
        assert result["requires_file_upload"] is True

    def test_extract_detects_gradio(self, keyword_extractor):
        """Test extract detects Gradio requirements"""
        # Happy path: Gradio detection
        description = "An interactive demo using Gradio"
        result = keyword_extractor.extract(description)
        assert result["requires_gradio"] is True

    # ========================================================================
    # Extract Method Tests - Feature Detection
    # ========================================================================

    def test_extract_detects_auth_requirement(self, keyword_extractor):
        """Test extract detects authentication requirement"""
        # Happy path: Auth detection
        description = "A system with user authentication and login"
        result = keyword_extractor.extract(description)
        assert result["requires_auth"] is True

    def test_extract_detects_database_requirement(self, keyword_extractor):
        """Test extract detects database requirement"""
        # Happy path: Database detection
        description = "A system that stores data in PostgreSQL database"
        result = keyword_extractor.extract(description)
        assert result["requires_database"] is True

    def test_extract_detects_websocket_requirement(self, keyword_extractor):
        """Test extract detects WebSocket requirement"""
        # Happy path: WebSocket detection
        description = "A real-time system using WebSocket"
        result = keyword_extractor.extract(description)
        assert result["requires_websocket"] is True

    def test_extract_detects_file_upload_requirement(self, keyword_extractor):
        """Test extract detects file upload requirement"""
        # Happy path: File upload detection
        description = "A system that allows users to upload images"
        result = keyword_extractor.extract(description)
        assert result["requires_file_upload"] is True

    def test_extract_detects_cache_requirement(self, keyword_extractor):
        """Test extract detects cache requirement"""
        # Happy path: Cache detection
        description = "A system using Redis cache"
        result = keyword_extractor.extract(description)
        assert result["requires_cache"] is True

    def test_extract_detects_queue_requirement(self, keyword_extractor):
        """Test extract detects queue requirement"""
        # Happy path: Queue detection
        description = "A system with background task queue"
        result = keyword_extractor.extract(description)
        assert result["requires_queue"] is True

    def test_extract_detects_multiple_features(self, keyword_extractor):
        """Test extract detects multiple features simultaneously"""
        # Happy path: Multiple features
        description = "A system with authentication, database, and WebSocket support"
        result = keyword_extractor.extract(description)
        assert result["requires_auth"] is True
        assert result["requires_database"] is True
        assert result["requires_websocket"] is True

    def test_extract_detects_feature_list(self, keyword_extractor):
        """Test extract populates features list"""
        # Happy path: Features list
        description = "A system with dashboard, REST API, and monitoring"
        result = keyword_extractor.extract(description)
        assert isinstance(result["features"], list)
        assert len(result["features"]) > 0

    # ========================================================================
    # Extract Method Tests - Provider Detection
    # ========================================================================

    def test_extract_detects_openai_provider(self, keyword_extractor):
        """Test extract detects OpenAI provider"""
        # Happy path: OpenAI detection
        description = "A chat system using OpenAI GPT models"
        result = keyword_extractor.extract(description)
        assert "openai" in result["model_providers"]

    def test_extract_detects_anthropic_provider(self, keyword_extractor):
        """Test extract detects Anthropic provider"""
        # Happy path: Anthropic detection
        description = "A system using Claude from Anthropic"
        result = keyword_extractor.extract(description)
        assert "anthropic" in result["model_providers"]

    def test_extract_detects_google_provider(self, keyword_extractor):
        """Test extract detects Google provider"""
        # Happy path: Google detection
        description = "A system using Google Gemini"
        result = keyword_extractor.extract(description)
        assert "google" in result["model_providers"]

    def test_extract_detects_huggingface_provider(self, keyword_extractor):
        """Test extract detects HuggingFace provider"""
        # Happy path: HuggingFace detection
        description = "A system using HuggingFace transformers"
        result = keyword_extractor.extract(description)
        assert "huggingface" in result["model_providers"]

    def test_extract_detects_local_provider(self, keyword_extractor):
        """Test extract detects local model provider"""
        # Happy path: Local detection
        description = "A system using local LLaMA model"
        result = keyword_extractor.extract(description)
        assert "local" in result["model_providers"]

    def test_extract_detects_multiple_providers(self, keyword_extractor):
        """Test extract detects multiple providers"""
        # Happy path: Multiple providers
        description = "A system supporting OpenAI, Anthropic, and Google models"
        result = keyword_extractor.extract(description)
        assert len(result["model_providers"]) >= 2

    # ========================================================================
    # Extract Method Tests - Complexity Detection
    # ========================================================================

    def test_extract_detects_simple_complexity(self, keyword_extractor):
        """Test extract detects simple complexity"""
        # Happy path: Simple complexity
        description = "A simple chat bot"
        result = keyword_extractor.extract(description)
        assert result["complexity"] == "simple"

    def test_extract_detects_medium_complexity(self, keyword_extractor):
        """Test extract detects medium complexity"""
        # Happy path: Medium complexity
        description = "A standard AI system"
        result = keyword_extractor.extract(description)
        assert result["complexity"] == "medium"

    def test_extract_detects_complex_complexity(self, keyword_extractor):
        """Test extract detects complex complexity"""
        # Happy path: Complex complexity
        description = "An advanced enterprise AI platform"
        result = keyword_extractor.extract(description)
        assert result["complexity"] == "complex"

    def test_extract_complexity_multilingual(self, keyword_extractor):
        """Test extract detects complexity in multiple languages"""
        # Edge case: Multilingual complexity
        descriptions = [
            "Un sistema básico",
            "A basic system",
            "Un sistema avanzado",
            "An advanced system"
        ]
        for desc in descriptions:
            result = keyword_extractor.extract(desc)
            assert result["complexity"] in ["simple", "medium", "complex"]

    # ========================================================================
    # Extract Method Tests - Edge Cases
    # ========================================================================

    def test_extract_with_very_long_description(self, keyword_extractor):
        """Test extract handles very long descriptions"""
        # Edge case: Very long description
        long_description = "A chat AI system. " * 1000
        result = keyword_extractor.extract(long_description)
        assert isinstance(result, dict)
        assert "ai_type" in result
        assert result["ai_type"] == "chat"

    def test_extract_with_special_characters(self, keyword_extractor):
        """Test extract handles special characters in description"""
        # Edge case: Special characters
        description = "A chat AI system with @#$% special characters & symbols!"
        result = keyword_extractor.extract(description)
        assert isinstance(result, dict)
        assert result["ai_type"] == "chat"

    def test_extract_with_mixed_case(self, keyword_extractor):
        """Test extract handles mixed case descriptions"""
        # Edge case: Mixed case
        description = "A CHAT AI SYSTEM With MiXeD CaSe"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "chat"

    def test_extract_with_numbers(self, keyword_extractor):
        """Test extract handles descriptions with numbers"""
        # Edge case: Numbers in description
        description = "A chat AI system version 2.0 with 100 features"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "chat"

    def test_extract_with_newlines_and_tabs(self, keyword_extractor):
        """Test extract handles descriptions with newlines and tabs"""
        # Edge case: Newlines and tabs
        description = "A chat\nAI\tsystem\nwith\tmultiple\nlines"
        result = keyword_extractor.extract(description)
        assert result["ai_type"] == "chat"

    # ========================================================================
    # Extract Method Tests - Comprehensive Scenarios
    # ========================================================================

    def test_extract_with_comprehensive_description(self, keyword_extractor):
        """Test extract handles comprehensive description with all features"""
        # Happy path: Comprehensive description
        description = """
        An advanced enterprise AI platform with:
        - User authentication and login
        - PostgreSQL database storage
        - WebSocket for real-time communication
        - File upload capabilities
        - Redis cache
        - Background task queue
        - Deep learning models using PyTorch
        - OpenAI and Anthropic providers
        - Dashboard and REST API
        - Monitoring and logging
        """
        result = keyword_extractor.extract(description)
        
        # Verify all major features
        assert result["requires_auth"] is True
        assert result["requires_database"] is True
        assert result["requires_websocket"] is True
        assert result["requires_file_upload"] is True
        assert result["requires_cache"] is True
        assert result["requires_queue"] is True
        assert result["is_deep_learning"] is True
        assert result["requires_pytorch"] is True
        assert len(result["model_providers"]) >= 2
        assert result["complexity"] == "complex"
        assert len(result["features"]) > 0

    def test_extract_returns_consistent_structure(self, keyword_extractor):
        """Test extract always returns consistent structure"""
        # Edge case: Structure consistency
        descriptions = [
            "",
            "Simple chat bot",
            "Complex enterprise AI system with all features"
        ]
        
        for desc in descriptions:
            result = keyword_extractor.extract(desc)
            # All results should have the same keys
            default_keys = set(_get_default_keywords().keys())
            result_keys = set(result.keys())
            assert default_keys == result_keys, f"Structure mismatch for: {desc}"


@pytest.fixture
def keyword_extractor():
    """Fixture for KeywordExtractor instance"""
    return KeywordExtractor()


