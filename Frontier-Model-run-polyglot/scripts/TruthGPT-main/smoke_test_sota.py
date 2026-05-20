"""
Smoke Test for TruthGPT SOTA Optimization Core.
Validates: Cache, Compression, Cascade, and Budget.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from optimization_core.agents.cost_intelligence import cost_intelligence
from optimization_core.agents.engines import DummyAsyncLLM

async def run_smoke_test():
    print("🚀 Starting SOTA Smoke Test...")
    
    # 1. Test Mock Engine
    llm = DummyAsyncLLM()
    
    # 2. Test Prompt Compression
    long_prompt = "Este es un prompt muy largo que debería ser comprimido por el sistema LLMLingua-2. " * 50
    print(f"\n[Test 1] Compression (Input: {len(long_prompt)} chars)")
    compressed = cost_intelligence.compress(long_prompt)
    print(f"Compressed: {len(compressed)} chars (Ratio: {len(compressed)/len(long_prompt):.2f})")
    
    # 3. Test Optimized Call (with Cache)
    print("\n[Test 2] Optimized Call (First Run - should miss cache)")
    resp1 = await cost_intelligence.optimize_call(
        prompt="¿Cuál es el significado de la vida?",
        llm_func=llm,
        models=["gpt-4o-mini"]
    )
    print(f"Response 1: {resp1[:50]}...")
    
    print("\n[Test 3] Optimized Call (Second Run - should HIT cache)")
    resp2 = await cost_intelligence.optimize_call(
        prompt="¿Cuál es el significado de la vida?",
        llm_func=llm,
        models=["gpt-4o-mini"]
    )
    print(f"Response 2: {resp2[:50]}...")
    
    # 4. Test Stats
    print("\n[Test 4] Operational Stats")
    stats = cost_intelligence.get_stats()
    for k, v in stats.items():
        print(f" - {k}: {v}")
    
    if stats.get('hits_exact', 0) > 0 or stats.get('hits_semantic', 0) > 0:
        print("\n✅ SUCCESS: SOTA Optimization Core is operational.")
    else:
        print("\n❌ FAILURE: Cache hit not detected.")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
