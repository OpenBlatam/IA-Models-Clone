import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from backend.unified_ai_model.core.llm_service import LLMService
    print("✅ Successfully imported LLMService")
except ImportError as e:
    print(f"❌ Failed to import LLMService: {e}")
