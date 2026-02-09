from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import unittest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os
from official_docs_reference import (
        import shutil
import torch
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
        import shutil
import torch
from torch.cuda.amp import autocast, GradScaler
    import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Test Suite for Official Documentation Reference System
====================================================

This module provides comprehensive tests for the official documentation
reference system, ensuring all functionality works correctly.
"""


# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    OfficialDocsReference,
    LibraryInfo,
    APIRef,
    BestPractice
)


class TestLibraryInfo(unittest.TestCase):
    """Test cases for LibraryInfo dataclass."""
    
    def test_library_info_creation(self) -> Any:
        """Test creating a LibraryInfo instance."""
        lib_info = LibraryInfo(
            name="Test Library",
            current_version="1.0.0",
            min_supported_version="0.9.0",
            documentation_url="https://test.com/docs",
            github_url="https://github.com/test/lib",
            pip_package="test-lib"
        )
        
        self.assertEqual(lib_info.name, "Test Library")
        self.assertEqual(lib_info.current_version, "1.0.0")
        self.assertEqual(lib_info.min_supported_version, "0.9.0")
        self.assertEqual(lib_info.documentation_url, "https://test.com/docs")
        self.assertEqual(lib_info.github_url, "https://github.com/test/lib")
        self.assertEqual(lib_info.pip_package, "test-lib")
        self.assertIsNone(lib_info.conda_package)
    
    def test_library_info_with_conda(self) -> Any:
        """Test creating a LibraryInfo instance with conda package."""
        lib_info = LibraryInfo(
            name="Test Library",
            current_version="1.0.0",
            min_supported_version="0.9.0",
            documentation_url="https://test.com/docs",
            github_url="https://github.com/test/lib",
            pip_package="test-lib",
            conda_package="test-lib"
        )
        
        self.assertEqual(lib_info.conda_package, "test-lib")


class TestAPIRef(unittest.TestCase):
    """Test cases for APIRef dataclass."""
    
    async def test_api_ref_creation(self) -> Any:
        """Test creating an APIRef instance."""
        api_ref = APIRef(
            name="test_api",
            description="Test API description",
            official_docs_url="https://test.com/api",
            code_example="print('test')",
            best_practices=["Practice 1", "Practice 2"],
            performance_tips=["Tip 1", "Tip 2"]
        )
        
        self.assertEqual(api_ref.name, "test_api")
        self.assertEqual(api_ref.description, "Test API description")
        self.assertEqual(api_ref.official_docs_url, "https://test.com/api")
        self.assertEqual(api_ref.code_example, "print('test')")
        self.assertEqual(api_ref.best_practices, ["Practice 1", "Practice 2"])
        self.assertEqual(api_ref.performance_tips, ["Tip 1", "Tip 2"])
        self.assertIsNone(api_ref.deprecation_warning)
        self.assertIsNone(api_ref.migration_guide)
    
    async def test_api_ref_with_deprecation(self) -> Any:
        """Test creating an APIRef instance with deprecation warning."""
        api_ref = APIRef(
            name="deprecated_api",
            description="Deprecated API",
            official_docs_url="https://test.com/deprecated",
            code_example="print('deprecated')",
            best_practices=["Use new API"],
            deprecation_warning="This API is deprecated",
            migration_guide="https://test.com/migration"
        )
        
        self.assertEqual(api_ref.deprecation_warning, "This API is deprecated")
        self.assertEqual(api_ref.migration_guide, "https://test.com/migration")


class TestBestPractice(unittest.TestCase):
    """Test cases for BestPractice dataclass."""
    
    def test_best_practice_creation(self) -> Any:
        """Test creating a BestPractice instance."""
        practice = BestPractice(
            title="Test Practice",
            description="Test practice description",
            source="Official Docs",
            code_example="print('practice')",
            category="performance",
            importance="important"
        )
        
        self.assertEqual(practice.title, "Test Practice")
        self.assertEqual(practice.description, "Test practice description")
        self.assertEqual(practice.source, "Official Docs")
        self.assertEqual(practice.code_example, "print('practice')")
        self.assertEqual(practice.category, "performance")
        self.assertEqual(practice.importance, "important")


class TestOfficialDocsReference(unittest.TestCase):
    """Test cases for OfficialDocsReference class."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ref_system = OfficialDocsReference(cache_dir=self.temp_dir)
    
    def tearDown(self) -> Any:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self) -> Any:
        """Test initialization of OfficialDocsReference."""
        self.assertIsNotNone(self.ref_system.libraries)
        self.assertIn("pytorch", self.ref_system.libraries)
        self.assertIn("transformers", self.ref_system.libraries)
        self.assertIn("diffusers", self.ref_system.libraries)
        self.assertIn("gradio", self.ref_system.libraries)
        
        # Check cache directory creation
        self.assertTrue(Path(self.temp_dir).exists())
    
    def test_get_library_info(self) -> Optional[Dict[str, Any]]:
        """Test getting library information."""
        # Test existing library
        pytorch_info = self.ref_system.get_library_info("pytorch")
        self.assertIsNotNone(pytorch_info)
        self.assertEqual(pytorch_info.name, "PyTorch")
        self.assertEqual(pytorch_info.current_version, "2.1.0")
        
        # Test non-existing library
        unknown_info = self.ref_system.get_library_info("unknown_lib")
        self.assertIsNone(unknown_info)
        
        # Test case insensitive
        pytorch_info_lower = self.ref_system.get_library_info("PyTorch")
        self.assertIsNotNone(pytorch_info_lower)
        self.assertEqual(pytorch_info_lower.name, "PyTorch")
    
    async def test_get_api_reference(self) -> Optional[Dict[str, Any]]:
        """Test getting API references."""
        # Test PyTorch API reference
        amp_ref = self.ref_system.get_api_reference("pytorch", "mixed_precision")
        self.assertIsNotNone(amp_ref)
        self.assertEqual(amp_ref.name, "torch.cuda.amp")
        self.assertIn("Automatic Mixed Precision", amp_ref.description)
        
        # Test non-existing API
        unknown_ref = self.ref_system.get_api_reference("pytorch", "unknown_api")
        self.assertIsNone(unknown_ref)
        
        # Test non-existing library
        unknown_lib_ref = self.ref_system.get_api_reference("unknown_lib", "mixed_precision")
        self.assertIsNone(unknown_lib_ref)
    
    def test_get_best_practices(self) -> Optional[Dict[str, Any]]:
        """Test getting best practices."""
        # Test PyTorch best practices
        practices = self.ref_system.get_best_practices("pytorch")
        self.assertIsInstance(practices, list)
        self.assertGreater(len(practices), 0)
        
        # Test with category filter
        performance_practices = self.ref_system.get_best_practices("pytorch", "performance")
        self.assertIsInstance(performance_practices, list)
        
        # Test non-existing library
        unknown_practices = self.ref_system.get_best_practices("unknown_lib")
        self.assertEqual(unknown_practices, [])
        
        # Test non-existing category
        unknown_category_practices = self.ref_system.get_best_practices("pytorch", "unknown_category")
        self.assertEqual(unknown_category_practices, [])
    
    def test_check_version_compatibility(self) -> Any:
        """Test version compatibility checking."""
        # Test compatible version
        compat = self.ref_system.check_version_compatibility("pytorch", "2.1.0")
        self.assertTrue(compat["compatible"])
        self.assertEqual(compat["current_version"], "2.1.0")
        self.assertEqual(compat["requested_version"], "2.1.0")
        
        # Test newer version
        compat_newer = self.ref_system.check_version_compatibility("pytorch", "2.2.0")
        self.assertTrue(compat_newer["compatible"])
        
        # Test older version
        compat_older = self.ref_system.check_version_compatibility("pytorch", "1.12.0")
        self.assertFalse(compat_older["compatible"])
        self.assertIn("Upgrade", compat_older["recommendation"])
        
        # Test non-existing library
        unknown_compat = self.ref_system.check_version_compatibility("unknown_lib", "1.0.0")
        self.assertFalse(unknown_compat["compatible"])
        self.assertIn("Unknown library", unknown_compat["error"])
    
    def test_generate_migration_guide(self) -> Any:
        """Test migration guide generation."""
        guide = self.ref_system.generate_migration_guide("pytorch", "1.13.0", "2.1.0")
        
        self.assertEqual(guide["library"], "pytorch")
        self.assertEqual(guide["from_version"], "1.13.0")
        self.assertEqual(guide["to_version"], "2.1.0")
        self.assertIn("migration_steps", guide)
        self.assertIsInstance(guide["migration_steps"], list)
        self.assertGreater(len(guide["migration_steps"]), 0)
    
    def test_export_references_json(self) -> Any:
        """Test exporting references to JSON."""
        output_file = Path(self.temp_dir) / "references.json"
        
        self.ref_system.export_references(str(output_file), "json")
        
        self.assertTrue(output_file.exists())
        
        # Verify JSON content
        with open(output_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data = json.load(f)
        
        self.assertIn("libraries", data)
        self.assertIn("api_references", data)
        self.assertIn("pytorch", data["libraries"])
        self.assertIn("pytorch", data["api_references"])
    
    def test_export_references_yaml(self) -> Any:
        """Test exporting references to YAML."""
        output_file = Path(self.temp_dir) / "references.yaml"
        
        self.ref_system.export_references(str(output_file), "yaml")
        
        self.assertTrue(output_file.exists())
        
        # Verify YAML content
        with open(output_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data = yaml.safe_load(f)
        
        self.assertIn("libraries", data)
        self.assertIn("api_references", data)
        self.assertIn("pytorch", data["libraries"])
        self.assertIn("pytorch", data["api_references"])
    
    def test_get_performance_recommendations(self) -> Optional[Dict[str, Any]]:
        """Test getting performance recommendations."""
        # Test PyTorch recommendations
        pytorch_recs = self.ref_system.get_performance_recommendations("pytorch")
        self.assertIsInstance(pytorch_recs, list)
        self.assertGreater(len(pytorch_recs), 0)
        
        # Test Transformers recommendations
        transformers_recs = self.ref_system.get_performance_recommendations("transformers")
        self.assertIsInstance(transformers_recs, list)
        self.assertGreater(len(transformers_recs), 0)
        
        # Test Diffusers recommendations
        diffusers_recs = self.ref_system.get_performance_recommendations("diffusers")
        self.assertIsInstance(diffusers_recs, list)
        self.assertGreater(len(diffusers_recs), 0)
        
        # Test Gradio recommendations
        gradio_recs = self.ref_system.get_performance_recommendations("gradio")
        self.assertIsInstance(gradio_recs, list)
        self.assertGreater(len(gradio_recs), 0)
        
        # Test non-existing library
        unknown_recs = self.ref_system.get_performance_recommendations("unknown_lib")
        self.assertEqual(unknown_recs, [])
    
    def test_validate_code_snippet(self) -> bool:
        """Test code snippet validation."""
        # Test valid PyTorch code
        valid_code = """

dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)
"""
        result = self.ref_system.validate_code_snippet(valid_code, "pytorch")
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)
        self.assertGreater(len(result["recommendations"]), 0)
        
        # Test code with issues
        problematic_code = """

dataloader = DataLoader(dataset)  # Missing num_workers
"""
        result = self.ref_system.validate_code_snippet(problematic_code, "pytorch")
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["issues"]), 0)
        self.assertGreater(len(result["recommendations"]), 0)
        
        # Test non-existing library
        unknown_result = self.ref_system.validate_code_snippet("print('test')", "unknown_lib")
        self.assertTrue(unknown_result["valid"])
        self.assertEqual(len(unknown_result["issues"]), 0)
        self.assertEqual(len(unknown_result["recommendations"]), 0)
    
    def test_pytorch_references(self) -> Any:
        """Test PyTorch-specific references."""
        self.assertIsNotNone(self.ref_system.pytorch_refs)
        self.assertIn("mixed_precision", self.ref_system.pytorch_refs)
        self.assertIn("data_loading", self.ref_system.pytorch_refs)
        self.assertIn("model_checkpointing", self.ref_system.pytorch_refs)
        self.assertIn("distributed_training", self.ref_system.pytorch_refs)
        
        # Test mixed precision reference
        amp_ref = self.ref_system.pytorch_refs["mixed_precision"]
        self.assertEqual(amp_ref.name, "torch.cuda.amp")
        self.assertIn("Automatic Mixed Precision", amp_ref.description)
        self.assertIn("GradScaler", amp_ref.best_practices[0])
    
    def test_transformers_references(self) -> Any:
        """Test Transformers-specific references."""
        self.assertIsNotNone(self.ref_system.transformers_refs)
        self.assertIn("model_loading", self.ref_system.transformers_refs)
        self.assertIn("tokenization", self.ref_system.transformers_refs)
        self.assertIn("training", self.ref_system.transformers_refs)
        
        # Test model loading reference
        model_ref = self.ref_system.transformers_refs["model_loading"]
        self.assertEqual(model_ref.name, "Model Loading")
        self.assertIn("AutoModel", model_ref.code_example)
    
    def test_diffusers_references(self) -> Any:
        """Test Diffusers-specific references."""
        self.assertIsNotNone(self.ref_system.diffusers_refs)
        self.assertIn("pipeline_usage", self.ref_system.diffusers_refs)
        self.assertIn("custom_training", self.ref_system.diffusers_refs)
        self.assertIn("memory_optimization", self.ref_system.diffusers_refs)
        
        # Test pipeline usage reference
        pipeline_ref = self.ref_system.diffusers_refs["pipeline_usage"]
        self.assertEqual(pipeline_ref.name, "Diffusion Pipeline")
        self.assertIn("DiffusionPipeline", pipeline_ref.code_example)
    
    def test_gradio_references(self) -> Any:
        """Test Gradio-specific references."""
        self.assertIsNotNone(self.ref_system.gradio_refs)
        self.assertIn("interface_creation", self.ref_system.gradio_refs)
        self.assertIn("advanced_components", self.ref_system.gradio_refs)
        self.assertIn("deployment", self.ref_system.gradio_refs)
        
        # Test interface creation reference
        interface_ref = self.ref_system.gradio_refs["interface_creation"]
        self.assertEqual(interface_ref.name, "Interface Creation")
        self.assertIn("gr.Interface", interface_ref.code_example)
    
    def test_library_versions(self) -> Any:
        """Test library version information."""
        pytorch_info = self.ref_system.libraries["pytorch"]
        self.assertEqual(pytorch_info.current_version, "2.1.0")
        self.assertEqual(pytorch_info.min_supported_version, "1.13.0")
        
        transformers_info = self.ref_system.libraries["transformers"]
        self.assertEqual(transformers_info.current_version, "4.35.0")
        self.assertEqual(transformers_info.min_supported_version, "4.20.0")
        
        diffusers_info = self.ref_system.libraries["diffusers"]
        self.assertEqual(diffusers_info.current_version, "0.24.0")
        self.assertEqual(diffusers_info.min_supported_version, "0.18.0")
        
        gradio_info = self.ref_system.libraries["gradio"]
        self.assertEqual(gradio_info.current_version, "4.0.0")
        self.assertEqual(gradio_info.min_supported_version, "3.50.0")
    
    def test_documentation_urls(self) -> Any:
        """Test documentation URLs."""
        pytorch_info = self.ref_system.libraries["pytorch"]
        self.assertIn("pytorch.org", pytorch_info.documentation_url)
        
        transformers_info = self.ref_system.libraries["transformers"]
        self.assertIn("huggingface.co", transformers_info.documentation_url)
        
        diffusers_info = self.ref_system.libraries["diffusers"]
        self.assertIn("huggingface.co", diffusers_info.documentation_url)
        
        gradio_info = self.ref_system.libraries["gradio"]
        self.assertIn("gradio.app", gradio_info.documentation_url)
    
    def test_github_urls(self) -> Any:
        """Test GitHub URLs."""
        pytorch_info = self.ref_system.libraries["pytorch"]
        self.assertIn("github.com/pytorch", pytorch_info.github_url)
        
        transformers_info = self.ref_system.libraries["transformers"]
        self.assertIn("github.com/huggingface", transformers_info.github_url)
        
        diffusers_info = self.ref_system.libraries["diffusers"]
        self.assertIn("github.com/huggingface", diffusers_info.github_url)
        
        gradio_info = self.ref_system.libraries["gradio"]
        self.assertIn("github.com/gradio-app", gradio_info.github_url)
    
    def test_pip_packages(self) -> Any:
        """Test pip package names."""
        pytorch_info = self.ref_system.libraries["pytorch"]
        self.assertEqual(pytorch_info.pip_package, "torch")
        
        transformers_info = self.ref_system.libraries["transformers"]
        self.assertEqual(transformers_info.pip_package, "transformers")
        
        diffusers_info = self.ref_system.libraries["diffusers"]
        self.assertEqual(diffusers_info.pip_package, "diffusers")
        
        gradio_info = self.ref_system.libraries["gradio"]
        self.assertEqual(gradio_info.pip_package, "gradio")
    
    def test_conda_packages(self) -> Any:
        """Test conda package names."""
        pytorch_info = self.ref_system.libraries["pytorch"]
        self.assertEqual(pytorch_info.conda_package, "pytorch")
        
        # Other libraries don't have conda packages specified
        transformers_info = self.ref_system.libraries["transformers"]
        self.assertIsNone(transformers_info.conda_package)
        
        diffusers_info = self.ref_system.libraries["diffusers"]
        self.assertIsNone(diffusers_info.conda_package)
        
        gradio_info = self.ref_system.libraries["gradio"]
        self.assertIsNone(gradio_info.conda_package)


class TestOfficialDocsReferenceIntegration(unittest.TestCase):
    """Integration tests for OfficialDocsReference."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ref_system = OfficialDocsReference(cache_dir=self.temp_dir)
    
    def tearDown(self) -> Any:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self) -> Any:
        """Test a complete workflow using the reference system."""
        # 1. Get library information
        pytorch_info = self.ref_system.get_library_info("pytorch")
        self.assertIsNotNone(pytorch_info)
        
        # 2. Check version compatibility
        compat = self.ref_system.check_version_compatibility("pytorch", "2.0.0")
        self.assertTrue(compat["compatible"])
        
        # 3. Get API reference
        amp_ref = self.ref_system.get_api_reference("pytorch", "mixed_precision")
        self.assertIsNotNone(amp_ref)
        
        # 4. Get best practices
        practices = self.ref_system.get_best_practices("pytorch", "performance")
        self.assertGreater(len(practices), 0)
        
        # 5. Get performance recommendations
        recommendations = self.ref_system.get_performance_recommendations("pytorch")
        self.assertGreater(len(recommendations), 0)
        
        # 6. Validate code snippet
        code = """

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = loss_fn(output, target)
"""
        validation = self.ref_system.validate_code_snippet(code, "pytorch")
        self.assertTrue(validation["valid"])
        
        # 7. Export references
        output_file = Path(self.temp_dir) / "workflow_test.json"
        self.ref_system.export_references(str(output_file))
        self.assertTrue(output_file.exists())
    
    def test_multi_library_workflow(self) -> Any:
        """Test workflow across multiple libraries."""
        libraries = ["pytorch", "transformers", "diffusers", "gradio"]
        
        for lib in libraries:
            # Get library info
            info = self.ref_system.get_library_info(lib)
            self.assertIsNotNone(info)
            
            # Get best practices
            practices = self.ref_system.get_best_practices(lib)
            self.assertGreater(len(practices), 0)
            
            # Get performance recommendations
            recommendations = self.ref_system.get_performance_recommendations(lib)
            self.assertGreater(len(recommendations), 0)
    
    def test_error_handling(self) -> Any:
        """Test error handling for invalid inputs."""
        # Test invalid library name
        unknown_info = self.ref_system.get_library_info("")
        self.assertIsNone(unknown_info)
        
        # Test invalid API name
        unknown_api = self.ref_system.get_api_reference("pytorch", "")
        self.assertIsNone(unknown_api)
        
        # Test invalid version format
        invalid_compat = self.ref_system.check_version_compatibility("pytorch", "invalid_version")
        self.assertFalse(invalid_compat["compatible"])
        
        # Test empty code snippet
        empty_validation = self.ref_system.validate_code_snippet("", "pytorch")
        self.assertTrue(empty_validation["valid"])
    
    def test_cache_directory_handling(self) -> Any:
        """Test cache directory handling."""
        # Test with None cache directory
        ref_system_no_cache = OfficialDocsReference(cache_dir=None)
        self.assertIsNotNone(ref_system_no_cache.cache_dir)
        
        # Test with existing directory
        existing_dir = Path(self.temp_dir) / "existing_cache"
        existing_dir.mkdir()
        ref_system_existing = OfficialDocsReference(cache_dir=str(existing_dir))
        self.assertEqual(ref_system_existing.cache_dir, existing_dir)
        
        # Test with non-existent directory
        non_existent_dir = Path(self.temp_dir) / "non_existent"
        ref_system_new = OfficialDocsReference(cache_dir=str(non_existent_dir))
        self.assertTrue(non_existent_dir.exists())


def run_performance_tests():
    """Run performance tests for the reference system."""
    
    print("Running performance tests...")
    
    # Initialize system
    start_time = time.time()
    ref_system = OfficialDocsReference()
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.4f} seconds")
    
    # Test library info retrieval
    start_time = time.time()
    for _ in range(1000):
        ref_system.get_library_info("pytorch")
    lib_info_time = time.time() - start_time
    print(f"Library info retrieval (1000 calls): {lib_info_time:.4f} seconds")
    
    # Test API reference retrieval
    start_time = time.time()
    for _ in range(1000):
        ref_system.get_api_reference("pytorch", "mixed_precision")
    api_ref_time = time.time() - start_time
    print(f"API reference retrieval (1000 calls): {api_ref_time:.4f} seconds")
    
    # Test best practices retrieval
    start_time = time.time()
    for _ in range(1000):
        ref_system.get_best_practices("pytorch")
    practices_time = time.time() - start_time
    print(f"Best practices retrieval (1000 calls): {practices_time:.4f} seconds")
    
    # Test export
    start_time = time.time()
    ref_system.export_references("performance_test.json")
    export_time = time.time() - start_time
    print(f"Export time: {export_time:.4f} seconds")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    run_performance_tests() 