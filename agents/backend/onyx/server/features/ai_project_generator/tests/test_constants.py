"""
Tests for Constants

This test suite covers:
- Constant values
- Enum types
- Framework types
- Project complexity
- AI types
- Model architectures

Test Generation Principles:
- Unique: Each test covers a distinct scenario
- Diverse: Tests cover happy paths, edge cases, and validations
- Intuitive: Clear names and assertions express intent
"""

import pytest
from ..core.constants import (
    DEFAULT_PROJECT_VERSION,
    DEFAULT_AUTHOR,
    DEFAULT_BACKEND_PORT,
    DEFAULT_FRONTEND_PORT,
    MAX_PROJECT_NAME_LENGTH,
    FrameworkType,
    ProjectComplexity,
    AIType,
    ModelArchitecture,
    SUPPORTED_BACKEND_FRAMEWORKS,
    SUPPORTED_FRONTEND_FRAMEWORKS,
    COMMON_FEATURES
)


class TestConstants:
    """Test suite for constant values"""

    def test_default_project_version_is_string(self):
        """Test DEFAULT_PROJECT_VERSION is a string"""
        # Happy path: Type check
        assert isinstance(DEFAULT_PROJECT_VERSION, str)
        assert DEFAULT_PROJECT_VERSION == "1.0.0"

    def test_default_author_is_string(self):
        """Test DEFAULT_AUTHOR is a string"""
        # Happy path: Type check
        assert isinstance(DEFAULT_AUTHOR, str)
        assert DEFAULT_AUTHOR == "Blatam Academy"

    def test_default_backend_port_is_integer(self):
        """Test DEFAULT_BACKEND_PORT is an integer"""
        # Happy path: Type check
        assert isinstance(DEFAULT_BACKEND_PORT, int)
        assert DEFAULT_BACKEND_PORT == 8000

    def test_default_frontend_port_is_integer(self):
        """Test DEFAULT_FRONTEND_PORT is an integer"""
        # Happy path: Type check
        assert isinstance(DEFAULT_FRONTEND_PORT, int)
        assert DEFAULT_FRONTEND_PORT == 3000

    def test_max_project_name_length_is_integer(self):
        """Test MAX_PROJECT_NAME_LENGTH is an integer"""
        # Happy path: Type check
        assert isinstance(MAX_PROJECT_NAME_LENGTH, int)
        assert MAX_PROJECT_NAME_LENGTH == 50

    def test_max_project_name_length_is_positive(self):
        """Test MAX_PROJECT_NAME_LENGTH is positive"""
        # Edge case: Positive value
        assert MAX_PROJECT_NAME_LENGTH > 0


class TestFrameworkType:
    """Test suite for FrameworkType enum"""

    def test_framework_type_fastapi_value(self):
        """Test FrameworkType.FASTAPI has correct value"""
        # Happy path: FastAPI value
        assert FrameworkType.FASTAPI == "fastapi"
        assert FrameworkType.FASTAPI.value == "fastapi"

    def test_framework_type_flask_value(self):
        """Test FrameworkType.FLASK has correct value"""
        # Happy path: Flask value
        assert FrameworkType.FLASK == "flask"
        assert FrameworkType.FLASK.value == "flask"

    def test_framework_type_django_value(self):
        """Test FrameworkType.DJANGO has correct value"""
        # Happy path: Django value
        assert FrameworkType.DJANGO == "django"
        assert FrameworkType.DJANGO.value == "django"

    def test_framework_type_react_value(self):
        """Test FrameworkType.REACT has correct value"""
        # Happy path: React value
        assert FrameworkType.REACT == "react"
        assert FrameworkType.REACT.value == "react"

    def test_framework_type_vue_value(self):
        """Test FrameworkType.VUE has correct value"""
        # Happy path: Vue value
        assert FrameworkType.VUE == "vue"
        assert FrameworkType.VUE.value == "vue"

    def test_framework_type_nextjs_value(self):
        """Test FrameworkType.NEXTJS has correct value"""
        # Happy path: NextJS value
        assert FrameworkType.NEXTJS == "nextjs"
        assert FrameworkType.NEXTJS.value == "nextjs"

    def test_framework_type_all_backend_frameworks(self):
        """Test all backend frameworks are defined"""
        # Happy path: Backend frameworks
        backend_frameworks = [FrameworkType.FASTAPI, FrameworkType.FLASK, FrameworkType.DJANGO]
        assert len(backend_frameworks) == 3

    def test_framework_type_all_frontend_frameworks(self):
        """Test all frontend frameworks are defined"""
        # Happy path: Frontend frameworks
        frontend_frameworks = [FrameworkType.REACT, FrameworkType.VUE, FrameworkType.NEXTJS]
        assert len(frontend_frameworks) == 3

    def test_framework_type_comparison_with_string(self):
        """Test FrameworkType can be compared with string"""
        # Edge case: String comparison
        assert FrameworkType.FASTAPI == "fastapi"
        assert FrameworkType.REACT == "react"

    def test_framework_type_membership_check(self):
        """Test FrameworkType membership check"""
        # Edge case: Membership
        assert "fastapi" in [f.value for f in FrameworkType]
        assert "react" in [f.value for f in FrameworkType]


class TestProjectComplexity:
    """Test suite for ProjectComplexity enum"""

    def test_project_complexity_simple_value(self):
        """Test ProjectComplexity.SIMPLE has correct value"""
        # Happy path: Simple value
        assert ProjectComplexity.SIMPLE == "simple"
        assert ProjectComplexity.SIMPLE.value == "simple"

    def test_project_complexity_medium_value(self):
        """Test ProjectComplexity.MEDIUM has correct value"""
        # Happy path: Medium value
        assert ProjectComplexity.MEDIUM == "medium"
        assert ProjectComplexity.MEDIUM.value == "medium"

    def test_project_complexity_complex_value(self):
        """Test ProjectComplexity.COMPLEX has correct value"""
        # Happy path: Complex value
        assert ProjectComplexity.COMPLEX == "complex"
        assert ProjectComplexity.COMPLEX.value == "complex"

    def test_project_complexity_all_levels(self):
        """Test all complexity levels are defined"""
        # Happy path: All levels
        complexities = [ProjectComplexity.SIMPLE, ProjectComplexity.MEDIUM, ProjectComplexity.COMPLEX]
        assert len(complexities) == 3

    def test_project_complexity_comparison_with_string(self):
        """Test ProjectComplexity can be compared with string"""
        # Edge case: String comparison
        assert ProjectComplexity.SIMPLE == "simple"
        assert ProjectComplexity.COMPLEX == "complex"


class TestAIType:
    """Test suite for AIType enum"""

    def test_ai_type_general_value(self):
        """Test AIType.GENERAL has correct value"""
        # Happy path: General value
        assert AIType.GENERAL == "general"
        assert AIType.GENERAL.value == "general"

    def test_ai_type_chat_value(self):
        """Test AIType.CHAT has correct value"""
        # Happy path: Chat value
        assert AIType.CHAT == "chat"
        assert AIType.CHAT.value == "chat"

    def test_ai_type_vision_value(self):
        """Test AIType.VISION has correct value"""
        # Happy path: Vision value
        assert AIType.VISION == "vision"
        assert AIType.VISION.value == "vision"

    def test_ai_type_audio_value(self):
        """Test AIType.AUDIO has correct value"""
        # Happy path: Audio value
        assert AIType.AUDIO == "audio"
        assert AIType.AUDIO.value == "audio"

    def test_ai_type_nlp_value(self):
        """Test AIType.NLP has correct value"""
        # Happy path: NLP value
        assert AIType.NLP == "nlp"
        assert AIType.NLP.value == "nlp"

    def test_ai_type_video_value(self):
        """Test AIType.VIDEO has correct value"""
        # Happy path: Video value
        assert AIType.VIDEO == "video"
        assert AIType.VIDEO.value == "video"

    def test_ai_type_recommendation_value(self):
        """Test AIType.RECOMMENDATION has correct value"""
        # Happy path: Recommendation value
        assert AIType.RECOMMENDATION == "recommendation"
        assert AIType.RECOMMENDATION.value == "recommendation"

    def test_ai_type_all_types(self):
        """Test all AI types are defined"""
        # Happy path: All types
        ai_types = [
            AIType.GENERAL, AIType.CHAT, AIType.VISION, AIType.AUDIO,
            AIType.NLP, AIType.VIDEO, AIType.RECOMMENDATION,
            AIType.ANALYTICS, AIType.GENERATION, AIType.CLASSIFICATION,
            AIType.SUMMARIZATION, AIType.QA
        ]
        assert len(ai_types) == 12

    def test_ai_type_comparison_with_string(self):
        """Test AIType can be compared with string"""
        # Edge case: String comparison
        assert AIType.CHAT == "chat"
        assert AIType.VISION == "vision"


class TestModelArchitecture:
    """Test suite for ModelArchitecture enum"""

    def test_model_architecture_transformer_value(self):
        """Test ModelArchitecture.TRANSFORMER has correct value"""
        # Happy path: Transformer value
        assert ModelArchitecture.TRANSFORMER == "transformer"
        assert ModelArchitecture.TRANSFORMER.value == "transformer"

    def test_model_architecture_diffusion_value(self):
        """Test ModelArchitecture.DIFFUSION has correct value"""
        # Happy path: Diffusion value
        assert ModelArchitecture.DIFFUSION == "diffusion"
        assert ModelArchitecture.DIFFUSION.value == "diffusion"

    def test_model_architecture_llm_value(self):
        """Test ModelArchitecture.LLM has correct value"""
        # Happy path: LLM value
        assert ModelArchitecture.LLM == "llm"
        assert ModelArchitecture.LLM.value == "llm"

    def test_model_architecture_cnn_value(self):
        """Test ModelArchitecture.CNN has correct value"""
        # Happy path: CNN value
        assert ModelArchitecture.CNN == "cnn"
        assert ModelArchitecture.CNN.value == "cnn"

    def test_model_architecture_rnn_value(self):
        """Test ModelArchitecture.RNN has correct value"""
        # Happy path: RNN value
        assert ModelArchitecture.RNN == "rnn"
        assert ModelArchitecture.RNN.value == "rnn"

    def test_model_architecture_custom_value(self):
        """Test ModelArchitecture.CUSTOM has correct value"""
        # Happy path: Custom value
        assert ModelArchitecture.CUSTOM == "custom"
        assert ModelArchitecture.CUSTOM.value == "custom"

    def test_model_architecture_all_architectures(self):
        """Test all model architectures are defined"""
        # Happy path: All architectures
        architectures = [
            ModelArchitecture.TRANSFORMER, ModelArchitecture.DIFFUSION,
            ModelArchitecture.LLM, ModelArchitecture.CNN,
            ModelArchitecture.RNN, ModelArchitecture.CUSTOM
        ]
        assert len(architectures) == 6


class TestSupportedFrameworks:
    """Test suite for supported frameworks lists"""

    def test_supported_backend_frameworks_is_list(self):
        """Test SUPPORTED_BACKEND_FRAMEWORKS is a list"""
        # Happy path: Type check
        assert isinstance(SUPPORTED_BACKEND_FRAMEWORKS, list)

    def test_supported_backend_frameworks_contains_fastapi(self):
        """Test SUPPORTED_BACKEND_FRAMEWORKS contains FastAPI"""
        # Happy path: FastAPI in list
        assert FrameworkType.FASTAPI.value in SUPPORTED_BACKEND_FRAMEWORKS

    def test_supported_backend_frameworks_not_empty(self):
        """Test SUPPORTED_BACKEND_FRAMEWORKS is not empty"""
        # Edge case: Not empty
        assert len(SUPPORTED_BACKEND_FRAMEWORKS) > 0

    def test_supported_frontend_frameworks_is_list(self):
        """Test SUPPORTED_FRONTEND_FRAMEWORKS is a list"""
        # Happy path: Type check
        assert isinstance(SUPPORTED_FRONTEND_FRAMEWORKS, list)

    def test_supported_frontend_frameworks_contains_react(self):
        """Test SUPPORTED_FRONTEND_FRAMEWORKS contains React"""
        # Happy path: React in list
        assert FrameworkType.REACT.value in SUPPORTED_FRONTEND_FRAMEWORKS

    def test_supported_frontend_frameworks_not_empty(self):
        """Test SUPPORTED_FRONTEND_FRAMEWORKS is not empty"""
        # Edge case: Not empty
        assert len(SUPPORTED_FRONTEND_FRAMEWORKS) > 0


class TestCommonFeatures:
    """Test suite for COMMON_FEATURES list"""

    def test_common_features_is_list(self):
        """Test COMMON_FEATURES is a list"""
        # Happy path: Type check
        assert isinstance(COMMON_FEATURES, list)

    def test_common_features_not_empty(self):
        """Test COMMON_FEATURES is not empty"""
        # Edge case: Not empty
        assert len(COMMON_FEATURES) > 0

    def test_common_features_contains_dashboard(self):
        """Test COMMON_FEATURES contains dashboard"""
        # Happy path: Dashboard in list
        assert "dashboard" in COMMON_FEATURES

    def test_common_features_contains_rest_api(self):
        """Test COMMON_FEATURES contains rest_api"""
        # Happy path: REST API in list
        assert "rest_api" in COMMON_FEATURES

    def test_common_features_contains_monitoring(self):
        """Test COMMON_FEATURES contains monitoring"""
        # Happy path: Monitoring in list
        assert "monitoring" in COMMON_FEATURES

    def test_common_features_contains_testing(self):
        """Test COMMON_FEATURES contains testing"""
        # Happy path: Testing in list
        assert "testing" in COMMON_FEATURES

    def test_common_features_all_strings(self):
        """Test COMMON_FEATURES contains only strings"""
        # Edge case: All strings
        assert all(isinstance(feature, str) for feature in COMMON_FEATURES)

    def test_common_features_no_duplicates(self):
        """Test COMMON_FEATURES has no duplicates"""
        # Edge case: No duplicates
        assert len(COMMON_FEATURES) == len(set(COMMON_FEATURES))


