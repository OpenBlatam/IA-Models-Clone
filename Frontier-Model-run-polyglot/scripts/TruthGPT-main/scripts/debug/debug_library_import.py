
import sys
import os

sys.path.append(os.getcwd())

print("Attempting to import optimization_core.optimizers.library.library_optimizer...")
try:
    from optimization_core.optimizers.library.library_optimizer import LibraryOptimizer
    print(f"Success 1: {LibraryOptimizer}")
except ImportError as e:
    print(f"Failure 1: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting lazy import via optimization_core.optimizers...")
try:
    from optimization_core.optimizers import LibraryOptimizer
    print(f"Success 2: {LibraryOptimizer}")
except ImportError as e:
    print(f"Failure 2: {e}")
    import traceback
    traceback.print_exc()
