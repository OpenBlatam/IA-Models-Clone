import time
import sys
from pathlib import Path

start = time.time()
print(f"[{time.time()-start:.4f}s] Starting profile...")

import asyncio
print(f"[{time.time()-start:.4f}s] asyncio imported")

# Add the path
p = r"c:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run-polyglot\scripts\TruthGPT-main\optimization_core"
if p not in sys.path:
    sys.path.insert(0, p)

import interface.core
print(f"[{time.time()-start:.4f}s] interface.core imported")

import interface.interactive_dashboard
print(f"[{time.time()-start:.4f}s] interface.interactive_dashboard imported")

import agents.engines
print(f"[{time.time()-start:.4f}s] agents.engines imported")

import agents.registry
print(f"[{time.time()-start:.4f}s] agents.registry imported")
