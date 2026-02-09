"""
Optimization Core - Structure Verification Tests
=================================================
Verifies that all __init__.py exports and lazy import systems resolve correctly.
Uses extensive mocking to bypass missing heavy dependencies (torch, numpy, etc.).
"""

import sys
import unittest
from unittest.mock import MagicMock
import os

# ── Project root on sys.path ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Mock heavy external dependencies ─────────────────────────────────
# We need __spec__ set on top-level mocks so importlib.util.find_spec works
import types

def _make_mock_module(name: str) -> MagicMock:
    """Create a MagicMock that has a proper __spec__ for importlib compat."""
    m = MagicMock()
    m.__spec__ = types.ModuleType(name).__spec__  # None, but attr exists
    m.__name__ = name
    m.__path__ = []
    m.__file__ = f"<mock:{name}>"
    return m

_MOCK_MODULES = [
    # PyTorch – every submodule any file might import
    "torch", "torch.nn", "torch.nn.functional", "torch.nn.utils",
    "torch.nn.init", "torch.nn.parameter",
    "torch.optim", "torch.optim.lr_scheduler",
    "torch.utils", "torch.utils.data", "torch.utils.checkpoint",
    "torch.cuda", "torch.cuda.amp", "torch.distributed",
    "torch.jit", "torch.autograd",
    "torch.fx", "torch.profiler", "torch.onnx",
    "torch.quantization", "torch.ao", "torch.ao.quantization",
    "torch.compile", "torch.export",
    # NumPy / SciPy
    "numpy", "numpy.linalg", "scipy", "scipy.optimize",
    # TensorFlow
    "tensorflow", "tensorflow.keras",
    # Diffusers / HuggingFace
    "diffusers", "diffusers.utils", "diffusers.pipelines",
    # Others
    "triton", "triton.language",
    "transformers", "datasets", "tokenizers",
    "onnxruntime", "tensorrt",
    "pycuda", "pycuda.driver",
    "cupy",
    "accelerate",
    "bitsandbytes",
    "safetensors",
    "wandb",
    "mlflow",
    "gradio",
    "fastapi", "uvicorn", "pydantic",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _make_mock_module(mod_name)


class TestOptimizersPackage(unittest.TestCase):
    """Tests for the `optimizers` package."""

    def test_import_optimizers(self):
        """Top-level import of optimizers should succeed."""
        import optimizers  # noqa: F811
        self.assertTrue(hasattr(optimizers, "__all__"))
        print("\n[OK] import optimizers")

    def test_factory_function(self):
        """create_truthgpt_optimizer should be callable."""
        from optimizers import create_truthgpt_optimizer
        self.assertTrue(callable(create_truthgpt_optimizer))
        print("[OK] create_truthgpt_optimizer is callable")

    def test_lazy_core_submodule(self):
        """Lazily accessing optimizers.core should return a module."""
        import optimizers
        core = optimizers.core
        self.assertIsNotNone(core)
        print(f"[OK] optimizers.core = {core}")

    def test_optimization_cores_factory(self):
        """create_optimization_core factory should be importable."""
        from optimizers.optimization_cores import create_optimization_core
        self.assertTrue(callable(create_optimization_core))
        print("[OK] create_optimization_core is callable")

    def test_optimization_cores_list(self):
        """list_available_cores should return a non-empty list."""
        from optimizers.optimization_cores import list_available_cores
        cores = list_available_cores()
        self.assertIsInstance(cores, list)
        self.assertGreater(len(cores), 0)
        print(f"[OK] Available cores: {cores}")

    def test_backward_compat_enhanced(self):
        """EnhancedOptimizationCore should be importable from optimizers (shim)."""
        from optimizers import EnhancedOptimizationCore
        self.assertIsNotNone(EnhancedOptimizationCore)
        print("[OK] EnhancedOptimizationCore backward compat")

    def test_generic_optimizer(self):
        """Generic optimizer should be importable."""
        from optimizers import create_generic_optimizer
        self.assertTrue(callable(create_generic_optimizer))
        print("[OK] create_generic_optimizer is callable")


class TestOptimizationCoreTopLevel(unittest.TestCase):
    """Tests for the top-level `optimization_core` package (this directory)."""

    def test_import_top_level(self):
        """Top-level __init__.py should load without error."""
        # Add parent dir so 'optimization_core' resolves as a real package
        parent_dir = os.path.dirname(PROJECT_ROOT)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        try:
            import optimization_core  # noqa: F811
            self.assertTrue(hasattr(optimization_core, '__version__'))
            print("\n[OK] import optimization_core")
        except Exception as e:
            self.fail(f"Failed to import optimization_core: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
