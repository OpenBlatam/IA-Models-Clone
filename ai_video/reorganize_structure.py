from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import shutil
import glob
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
AI Video Feature Directory Reorganization Script

This script reorganizes the ai_video directory into a more structured layout
with clear separation of concerns and better maintainability.
"""


def create_directory_structure():
    """Create the new directory structure"""
    
    # Define the new structure
    structure = {
        'core/': {
            'models/': ['pydantic_schemas.py', 'pydantic_validation.py'],
            'middleware/': ['error_middleware.py', 'middleware_patterns.py'],
            'exceptions/': ['http_exceptions.py', 'error_handling.py'],
            'utils/': ['functional_utils.py', 'lazy_loading_system.py']
        },
        'api/': {
            'endpoints/': ['functional_api.py'],
            'dependencies/': ['dependencies.py'],
            'routers/': []
        },
        'performance/': {
            'optimization/': [
                'performance_optimization.py', 
                'advanced_performance_optimization.py',
                'async_io_optimization.py',
                'pydantic_serialization_optimization.py',
                'enhanced_caching_system.py'
            ],
            'benchmarking/': ['performance_benchmark.py', 'optimized_pipeline.py']
        },
        'async_io/': {
            'patterns/': ['async_sync_patterns.py', 'lifespan_patterns.py'],
            'examples/': ['async_conversion_examples.py', 'async_sync_examples.py', 'lifespan_examples.py']
        },
        'validation/': {
            'pydantic/': ['pydantic_examples.py', 'pydantic_serialization_examples.py'],
            'guards/': ['guard_clauses.py', 'early_validation.py', 'early_returns.py']
        },
        'caching/': {
            'systems/': [],
            'examples/': ['caching_integration_example.py']
        },
        'serialization/': {
            'examples/': ['serialization_integration_example.py']
        },
        'functional/': {
            'patterns/': ['functional_pipeline.py', 'functional_training.py'],
            'examples/': ['functional_examples.py']
        },
        'gradio/': {
            'interface/': ['gradio_interface.py', 'gradio_error_handling.py'],
            'launcher/': ['gradio_launcher.py', 'gradio_app_example.py']
        },
        'testing/': {
            'unit/': ['test_pydantic_validation.py', 'test_project_init.py'],
            'integration/': []
        },
        'examples/': {
            'usage/': ['usage_examples.py'],
            'performance/': ['performance_examples.py', 'performance_integration_example.py'],
            'middleware/': ['middleware_examples.py', 'error_middleware_examples.py'],
            'http/': ['http_exception_examples.py'],
            'validation/': ['guard_examples.py', 'early_return_examples.py'],
            'error/': ['error_examples.py', 'edge_case_handler.py']
        },
        'docs/': {
            'guides/': [
                'FASTAPI_BEST_PRACTICES_GUIDE.md',
                'PYDANTIC_SERIALIZATION_GUIDE.md', 
                'CACHING_IMPLEMENTATION_GUIDE.md',
                'ASYNC_IO_CONVERSION_GUIDE.md',
                'PERFORMANCE_OPTIMIZATION_GUIDE.md',
                'README_MIDDLEWARE.md',
                'README_LIFESPAN.md',
                'README_ASYNC_SYNC.md',
                'README_EARLY_RETURNS.md',
                'README_GUARD_CLAUSES.md',
                'README_ERROR_HANDLING.md',
                'README_PYDANTIC_VALIDATION.md',
                'README_HTTP_EXCEPTIONS.md',
                'README_PERFORMANCE.md',
                'functional_README.md'
            ],
            'summaries/': [
                'FASTAPI_SUMMARY.md',
                'ASYNC_IO_SUMMARY.md',
                'REFACTOR_COMPLETE_SUMMARY.md',
                'REFACTORED_ARCHITECTURE.md',
                'SISTEMA_MODULAR_RESUMEN.md',
                'README_LATEST.md'
            ]
        },
        'scripts/': {
            'setup/': ['install_latest.py', 'project_init.py', 'organize_modular_structure.py'],
            'quickstart/': ['quick_start.py']
        },
        'main/': {
            'entry_points/': ['main.py', 'onyx_main.py', '__init__.py']
        }
    }
    
    return structure

def move_files_to_structure(base_path, structure) -> Any:
    """Move files to their new locations according to the structure"""
    
    moved_files = []
    errors = []
    
    for category, subcategories in structure.items():
        category_path = os.path.join(base_path, category)
        os.makedirs(category_path, exist_ok=True)
        
        for subcategory, files in subcategories.items():
            subcategory_path = os.path.join(category_path, subcategory)
            os.makedirs(subcategory_path, exist_ok=True)
            
            for file_name in files:
                source_path = os.path.join(base_path, file_name)
                dest_path = os.path.join(subcategory_path, file_name)
                
                if os.path.exists(source_path):
                    try:
                        shutil.move(source_path, dest_path)
                        moved_files.append(f"Moved: {file_name} -> {category}/{subcategory}/")
                        print(f"✓ Moved: {file_name} -> {category}/{subcategory}/")
                    except Exception as e:
                        error_msg = f"Error moving {file_name}: {str(e)}"
                        errors.append(error_msg)
                        print(f"✗ {error_msg}")
                else:
                    print(f"⚠ File not found: {file_name}")
    
    return moved_files, errors

def create_readme_files(base_path) -> Any:
    """Create README files for each directory to explain its purpose"""
    
    readme_content = {
        'core/': """# Core Module

This directory contains the core components of the AI Video feature:

- **models/**: Pydantic schemas and validation logic
- **middleware/**: Custom middleware implementations
- **exceptions/**: HTTP exceptions and error handling
- **utils/**: Utility functions and lazy loading systems
""",
        'api/': """# API Module

This directory contains API-related components:

- **endpoints/**: API endpoint implementations
- **dependencies/**: FastAPI dependencies
- **routers/**: API router configurations
""",
        'performance/': """# Performance Module

This directory contains performance optimization components:

- **optimization/**: Performance optimization implementations
- **benchmarking/**: Performance benchmarking tools
""",
        'async_io/': """# Async IO Module

This directory contains async/await patterns and examples:

- **patterns/**: Async pattern implementations
- **examples/**: Async usage examples
""",
        'validation/': """# Validation Module

This directory contains validation logic:

- **pydantic/**: Pydantic validation examples
- **guards/**: Guard clause implementations
""",
        'caching/': """# Caching Module

This directory contains caching implementations:

- **systems/**: Caching system implementations
- **examples/**: Caching usage examples
""",
        'serialization/': """# Serialization Module

This directory contains serialization examples and implementations.
""",
        'functional/': """# Functional Programming Module

This directory contains functional programming patterns:

- **patterns/**: Functional pattern implementations
- **examples/**: Functional programming examples
""",
        'gradio/': """# Gradio Module

This directory contains Gradio interface components:

- **interface/**: Gradio interface implementations
- **launcher/**: Gradio launcher scripts
""",
        'testing/': """# Testing Module

This directory contains test files:

- **unit/**: Unit tests
- **integration/**: Integration tests
""",
        'examples/': """# Examples Module

This directory contains usage examples organized by category:

- **usage/**: General usage examples
- **performance/**: Performance-related examples
- **middleware/**: Middleware examples
- **http/**: HTTP exception examples
- **validation/**: Validation examples
- **error/**: Error handling examples
""",
        'docs/': """# Documentation Module

This directory contains documentation:

- **guides/**: Implementation guides
- **summaries/**: Project summaries and architecture docs
""",
        'scripts/': """# Scripts Module

This directory contains utility scripts:

- **setup/**: Setup and initialization scripts
- **quickstart/**: Quick start scripts
""",
        'main/': """# Main Module

This directory contains main entry points:

- **entry_points/**: Main application entry points
"""
    }
    
    for directory, content in readme_content.items():
        readme_path = os.path.join(base_path, directory, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")

def create_main_readme(base_path) -> Any:
    """Create the main README for the ai_video directory"""
    
    main_readme = """# AI Video Feature

This directory contains the AI Video processing feature implementation with a well-organized modular structure.

## Directory Structure

### Core Components
- **core/**: Core models, middleware, exceptions, and utilities
- **api/**: API endpoints, dependencies, and routers
- **main/**: Main entry points and initialization

### Performance & Optimization
- **performance/**: Performance optimization and benchmarking
- **async_io/**: Async/await patterns and examples
- **caching/**: Caching implementations
- **serialization/**: Serialization examples

### Validation & Functional Programming
- **validation/**: Pydantic validation and guard clauses
- **functional/**: Functional programming patterns

### User Interfaces
- **gradio/**: Gradio interface implementations

### Development & Testing
- **testing/**: Unit and integration tests
- **examples/**: Comprehensive usage examples
- **docs/**: Documentation and guides
- **scripts/**: Setup and utility scripts

## Quick Start

1. Check the `scripts/quickstart/` directory for quick start guides
2. Review `docs/guides/` for implementation guides
3. Explore `examples/` for usage examples
4. Run tests in `testing/` directory

## Architecture

The feature follows a modular architecture with clear separation of concerns:
- **Core**: Business logic and models
- **API**: HTTP interface layer
- **Performance**: Optimization and caching
- **Validation**: Data validation and guards
- **Testing**: Comprehensive test coverage

## Contributing

When adding new features:
1. Place core logic in appropriate `core/` subdirectory
2. Add API endpoints in `api/endpoints/`
3. Include examples in `examples/`
4. Add tests in `testing/`
5. Update documentation in `docs/`
"""
    
    readme_path = os.path.join(base_path, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.write(main_readme)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")

def cleanup_empty_directories(base_path) -> Any:
    """Remove empty directories after reorganization"""
    
    for root, dirs, files in os.walk(base_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
            except OSError:
                pass  # Directory not empty or cannot be removed

def main():
    """Main reorganization function"""
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🚀 Starting AI Video Feature Directory Reorganization...")
    print(f"Working directory: {current_dir}")
    
    # Create the new structure
    structure = create_directory_structure()
    
    # Move files to their new locations
    print("\n📁 Moving files to new structure...")
    moved_files, errors = move_files_to_structure(current_dir, structure)
    
    # Create README files
    print("\n📝 Creating README files...")
    create_readme_files(current_dir)
    create_main_readme(current_dir)
    
    # Cleanup empty directories
    print("\n🧹 Cleaning up empty directories...")
    cleanup_empty_directories(current_dir)
    
    # Summary
    print(f"\n✅ Reorganization complete!")
    print(f"📊 Files moved: {len(moved_files)}")
    print(f"❌ Errors: {len(errors)}")
    
    if errors:
        print("\n⚠️  Errors encountered:")
        for error in errors:
            print(f"  - {error}")
    
    print(f"\n📚 Check the new README.md file for the complete directory structure overview.")

match __name__:
    case "__main__":
    main() 