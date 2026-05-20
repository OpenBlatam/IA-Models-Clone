
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.getcwd())

try:
    print("Testing lazy import for generic_compatibility...")
    from optimization_core.optimizers.generic_compatibility import UltraSpeedOptimizer
    print(f"Success! UltraSpeedOptimizer: {UltraSpeedOptimizer}")
except Exception as e:
    print(f"Failed! Error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nTesting lazy import for LibraryOptimizer...")
    from optimization_core.optimizers.LibraryOptimizer import LibraryOptimizer
    print(f"Success! LibraryOptimizer: {LibraryOptimizer}")
except Exception as e:
    print(f"Failed! Error: {e}")
    import traceback
    traceback.print_exc()
