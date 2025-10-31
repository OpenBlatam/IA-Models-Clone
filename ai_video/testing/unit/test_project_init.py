from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import sys
from pathlib import Path
import tempfile
import shutil
    from project_init import ProblemDefinition, DatasetAnalyzer, ProjectInitializer
        import pandas as pd
        import numpy as np
            import pandas as pd
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Test Script for Project Initialization System
=============================================

This script demonstrates the basic functionality of the project initialization system.
Run this to test the system and see how it works.

Usage:
    python test_project_init.py
"""


# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    print("✅ Successfully imported project initialization modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)


def test_problem_definition():
    """Test ProblemDefinition class."""
    print("\n🧪 Testing ProblemDefinition...")
    
    try:
        problem_def = ProblemDefinition(
            project_name="test_project",
            problem_type="classification",
            business_objective="Test objective",
            success_metrics=["accuracy", "precision"],
            constraints=["memory_limit"],
            assumptions=["data_quality"],
            stakeholders=["test_user"],
            timeline="1 week"
        )
        
        # Test serialization
        problem_dict = problem_def.to_dict()
        assert "project_name" in problem_dict
        assert problem_dict["project_name"] == "test_project"
        
        # Test save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            problem_def.save(f.name)
            loaded_problem = ProblemDefinition.load(f.name)
            assert loaded_problem.project_name == problem_def.project_name
        
        print("✅ ProblemDefinition tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ProblemDefinition test failed: {e}")
        return False


def test_dataset_analyzer():
    """Test DatasetAnalyzer class."""
    print("\n🧪 Testing DatasetAnalyzer...")
    
    try:
        # Create temporary test data
        
        test_data = pd.DataFrame({
            'id': range(10),
            'value': np.random.randn(10),
            'category': ['A', 'B'] * 5
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_file = temp_path / "test_data.csv"
            test_data.to_csv(data_file, index=False)
            
            # Test analyzer
            analyzer = DatasetAnalyzer(data_file, temp_path / "analysis")
            dataset_info = analyzer.analyze_dataset()
            
            assert dataset_info.size == 10
            assert len(dataset_info.features) == 3
            assert "id" in dataset_info.features
            
            print("✅ DatasetAnalyzer tests passed")
            return True
            
    except Exception as e:
        print(f"❌ DatasetAnalyzer test failed: {e}")
        return False


def test_project_initializer():
    """Test ProjectInitializer class."""
    print("\n🧪 Testing ProjectInitializer...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test problem definition
            problem_def = ProblemDefinition(
                project_name="test_project",
                problem_type="classification",
                business_objective="Test objective",
                success_metrics=["accuracy"],
                constraints=["memory_limit"],
                assumptions=["data_quality"],
                stakeholders=["test_user"],
                timeline="1 week"
            )
            
            # Create test data
            test_data = pd.DataFrame({'id': range(5), 'value': [1, 2, 3, 4, 5]})
            data_file = temp_path / "test_data.csv"
            test_data.to_csv(data_file, index=False)
            
            # Test initializer
            initializer = ProjectInitializer("test_project", temp_path / "project")
            summary = initializer.initialize_project(
                problem_def=problem_def,
                data_path=data_file,
                target_column="value",
                enable_tracking=False  # Disable tracking for test
            )
            
            assert summary["project_name"] == "test_project"
            assert "problem_definition" in summary
            assert "dataset_info" in summary
            
            # Check that project structure was created
            project_dir = temp_path / "project"
            assert (project_dir / "data").exists()
            assert (project_dir / "models").exists()
            assert (project_dir / "configs").exists()
            
            print("✅ ProjectInitializer tests passed")
            return True
            
    except Exception as e:
        print(f"❌ ProjectInitializer test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Project Initialization System")
    print("=" * 50)
    
    tests = [
        test_problem_definition,
        test_dataset_analyzer,
        test_project_initializer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project initialization system is working correctly.")
        print("\n📚 Next steps:")
        print("   1. Run the example: python examples/project_init_example.py")
        print("   2. Read the documentation: docs/PROJECT_INITIALIZATION.md")
        print("   3. Start your own project!")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 