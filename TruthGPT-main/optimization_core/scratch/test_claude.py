import sys
sys.path.insert(0, r'c:\blatam-academy\TruthGPT-main\optimization_core')

import asyncio
from agents.engines import engine_registry

async def main():
    eng = engine_registry.get_engine('claude')
    print(f"Claude engine resolved: {eng is not None}")
    if eng:
        try:
            res = await eng("Hello! Responda solo 'Hola' en español.")
            print(f"Response: {res}")
        except Exception as e:
            print(f"Error testing Claude: {e}")

if __name__ == '__main__':
    asyncio.run(main())
