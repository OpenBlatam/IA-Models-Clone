#!/usr/bin/env python3
"""
Quick start script for Unified AI Model API
Run with: python run.py

Or set your API key directly:
    OPENROUTER_API_KEY=sk-or-v1-xxx python run.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_env_file():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        
        # Try loading from current directory
        current_env = Path(__file__).parent / ".env"
        if current_env.exists():
            print(f"Loading environment from {current_env}")
            load_dotenv(current_env)
            
        # Try loading from project root (../../.env)
        root_env = Path(__file__).parent.parent.parent / ".env"
        if root_env.exists():
            print(f"Loading environment from {root_env}")
            load_dotenv(root_env)
            return True
            
        return False
    except ImportError:
        print("[WARNING] python-dotenv not installed, skipping .env loading")
        return False


def check_api_key():
    """Check if API key is configured (DeepSeek or OpenRouter)."""
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    
    # Check DeepSeek key first
    if deepseek_key and deepseek_key != "sk-your-api-key-here":
        print("[OK] Using DeepSeek API")
        return True
    
    # Check OpenRouter key
    if openrouter_key and openrouter_key != "sk-or-v1-your-api-key-here":
        print("[OK] Using OpenRouter API")
        return True
    
    print("\n" + "="*60)
    print("[WARNING] No API key configured!")
    print("="*60)
    print("\nTo fix this, set one of these environment variables:")
    print("\n  DeepSeek (recommended):")
    print("     Windows: set DEEPSEEK_API_KEY=sk-xxx")
    print("     Linux/Mac: export DEEPSEEK_API_KEY=sk-xxx")
    print("\n  OpenRouter:")
    print("     Windows: set OPENROUTER_API_KEY=sk-or-v1-xxx")
    print("     Linux/Mac: export OPENROUTER_API_KEY=sk-or-v1-xxx")
    print("\n" + "="*60 + "\n")
    return False


def print_startup_banner():
    """Print startup banner with configuration info."""
    host = os.environ.get('UNIFIED_AI_HOST', '0.0.0.0')
    port = os.environ.get('UNIFIED_AI_PORT', '8050')
    model = os.environ.get('UNIFIED_AI_DEFAULT_MODEL', 'deepseek-chat')
    
    # Detect provider
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        provider = "DeepSeek API (direct)"
    else:
        provider = "OpenRouter"
    
    print("\n" + "="*60)
    print("UNIFIED AI MODEL - API Server")
    print("="*60)
    print(f"\n  Provider: {provider}")
    print(f"  Default Model: {model}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"\n  API Documentation: http://localhost:{port}/docs")
    print(f"  Chat: POST http://localhost:{port}/api/v1/chat")
    print(f"  Agents: POST http://localhost:{port}/api/v1/agents")
    print(f"  Health: GET http://localhost:{port}/api/v1/health")
    print("\n" + "-"*60)
    print("Frontend endpoints ready for consumption!")
    print("-"*60 + "\n")


def main():
    """Main entry point."""
    # Load .env file
    load_env_file()
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    # Print startup info
    print_startup_banner()
    
    # Import and run
    from unified_ai_model.main import run
    run()


if __name__ == "__main__":
    main()



