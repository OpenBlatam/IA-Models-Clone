import sys
sys.path.insert(0, r'c:\blatam-academy\TruthGPT-main\optimization_core')

from agents.engines import AnthropicProvider, engine_registry

# Test 1: Anthropic model name
p = AnthropicProvider()
print(f"Anthropic model: {p.model}")
print(f"API key present: {bool(p.api_key)}")
if p.api_key:
    print(f"API key prefix: {p.api_key[:15]}...")

# Test 2: DeepSeek engine
eng = engine_registry.get_engine('deepseek')
print(f"\nDeepSeek engine available: {eng is not None}")
if eng:
    print(f"DeepSeek model: {getattr(eng, 'model_name', '?')}")

# Test 3: Preferred engine (claude) with auto-fallback
eng2 = engine_registry.get_engine('claude')
print(f"\nClaude engine available: {eng2 is not None}")
if eng2:
    print(f"Claude model: {getattr(eng2, 'model_name', '?')}")

# Test 4: Quick live API test with DeepSeek
import asyncio
async def test_live():
    if eng:
        try:
            result = await eng("Say 'hello' in one word. Respond with just the word.")
            print(f"\nLive DeepSeek test: {result[:100]}")
        except Exception as e:
            print(f"\nLive DeepSeek test FAILED: {e}")
    else:
        print("\nNo engine to test")

asyncio.run(test_live())
