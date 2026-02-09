"""
Unified Project Management System for the ads feature.

This module consolidates all project management functionality from the scattered implementations:
- project_initializer.py (comprehensive project initialization)

The new structure follows Clean Architecture principles with clear separation of concerns.
"""

import os
import json
import yaml
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import logging
from enum import Enum
import hashlib
import shutil
from collections import defaultdict
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from ...config import get_settings


class ProjectType(Enum):
    """Types of ML projects supported."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    DIFFUSION = "diffusion"
    FINE_TUNING = "fine_tuning"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


class DatasetType(Enum):
    """Types of datasets supported."""
    TEXT = "text"
    IMAGE = "image"
    TABULAR = "tabular"
    MULTIMODAL = "multimodal"
    TIME_SERIES = "time_series"
    AUDIO = "audio"
    VIDEO = "video"


class ProblemComplexity(Enum):
    """Problem complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


@dataclass
class DatasetInfo:
    """Information about the dataset."""
    name: str
    type: DatasetType
    path: str
    size: int = 0
    features: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    validation_split: float = 0.2
    test_split: float = 0.1
    description: str = ""
    source: str = ""
    license: str = ""
    last_updated: Optional[datetime] = None
    
    def __post_init__(self) -> Any:
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class ProblemDefinition:
    """Clear definition of the ML problem."""
    title: str
    description: str
    project_type: ProjectType
    complexity: ProblemComplexity
    objectives: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    business_value: str = ""
    timeline: str = ""
    budget: Optional[float] = None


@dataclass
class ProjectStructure:
    """Project directory structure."""
    root_dir: Path
    data_dir: Path
    models_dir: Path
    configs_dir: Path
    logs_dir: Path
    docs_dir: Path
    tests_dir: Path
    notebooks_dir: Path
    scripts_dir: Path
    requirements_file: Path
    readme_file: Path
    gitignore_file: Path


@dataclass
class ProjectConfig:
    """Project configuration."""
    name: str
    version: str
    description: str
    author: str
    project_type: ProjectType
    problem_definition: ProblemDefinition
    dataset_info: DatasetInfo
    dependencies: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ProjectManager:
    """Comprehensive project management for ML projects."""
    
    def __init__(self, project_root: str = "."):
        """Initialize the project manager."""
        self.project_root = Path(project_root).resolve()
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        
        # Ensure project root exists
        self.project_root.mkdir(parents=True, exist_ok=True)
    
    def create_project_structure(self, config: ProjectConfig) -> ProjectStructure:
        """Create the complete project directory structure."""
        try:
            structure = ProjectStructure(
                root_dir=self.project_root,
                data_dir=self.project_root / "data",
                models_dir=self.project_root / "models",
                configs_dir=self.project_root / "configs",
                logs_dir=self.project_root / "logs",
                docs_dir=self.project_root / "docs",
                tests_dir=self.project_root / "tests",
                notebooks_dir=self.project_root / "notebooks",
                scripts_dir=self.project_root / "scripts",
                requirements_file=self.project_root / "requirements.txt",
                readme_file=self.project_root / "README.md",
                gitignore_file=self.project_root / ".gitignore"
            )
            
            # Create directories
            for dir_path in [
                structure.data_dir, structure.models_dir, structure.configs_dir,
                structure.logs_dir, structure.docs_dir, structure.tests_dir,
                structure.notebooks_dir, structure.scripts_dir
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create files
            self._create_requirements_file(structure.requirements_file, config)
            self._create_readme_file(structure.readme_file, config)
            self._create_gitignore_file(structure.gitignore_file)
            self._create_project_config(structure.configs_dir, config)
            
            self.logger.info(f"Created project structure in {self.project_root}")
            return structure
            
        except Exception as e:
            self.logger.error(f"Failed to create project structure: {e}")
            raise
    
    def _create_requirements_file(self, requirements_file: Path, config: ProjectConfig) -> None:
        """Create requirements.txt file."""
        requirements = [
            "# Core ML libraries",
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "diffusers>=0.20.0",
            "accelerate>=0.20.0",
            "optimum>=1.12.0",
            "",
            "# Data processing",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "",
            "# Configuration and utilities",
            "pydantic>=2.0.0",
            "pydantic-settings>=2.0.0",
            "pyyaml>=6.0",
            "",
            "# API and web framework",
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "",
            "# Database and caching",
            "sqlalchemy>=2.0.0",
            "aioredis>=2.0.0",
            "",
            "# Monitoring and logging",
            "structlog>=23.0.0",
            "prometheus-client>=0.17.0",
            "",
            "# Testing",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            ""
        ]
        
        requirements_file.write_text("\n".join(requirements))
    
    def _create_readme_file(self, readme_file: Path, config: ProjectConfig) -> None:
        """Create README.md file."""
        readme_content = f"""# {config.name}

{config.description}

## Project Information

- **Type**: {config.project_type.value}
- **Complexity**: {config.problem_definition.complexity.value}
- **Author**: {config.author}
- **Version**: {config.version}
- **Created**: {config.created_at.strftime('%Y-%m-%d %H:%M:%S')}

## Problem Definition

**Title**: {config.problem_definition.title}

**Description**: {config.problem_definition.description}

**Objectives**:
{chr(10).join(f"- {obj}" for obj in config.problem_definition.objectives)}

**Success Metrics**:
{chr(10).join(f"- {metric}" for metric in config.problem_definition.success_metrics)}

## Dataset Information

- **Name**: {config.dataset_info.name}
- **Type**: {config.dataset_info.type.value}
- **Size**: {config.dataset_info.size:,} samples
- **Features**: {len(config.dataset_info.features)} features
- **Source**: {config.dataset_info.source}

## Project Structure

```
{config.name}/
├── data/           # Dataset files
├── models/         # Trained models
├── configs/        # Configuration files
├── logs/           # Training and application logs
├── docs/           # Documentation
├── tests/          # Test files
├── notebooks/      # Jupyter notebooks
├── scripts/        # Utility scripts
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the project**:
   ```bash
   cp configs/config.example.yaml configs/config.yaml
   # Edit configs/config.yaml with your settings
   ```

3. **Run the application**:
   ```bash
   python -m {config.name}.main
   ```

## Configuration

Edit `configs/config.yaml` to configure:
- Model parameters
- Training settings
- Data paths
- API settings
- Logging configuration

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

{config.dataset_info.license}
"""
        
        readme_file.write_text(readme_content)
    
    def _create_gitignore_file(self, gitignore_file: Path) -> None:
        """Create .gitignore file."""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local
.env.production

# Logs
logs/
*.log

# Models and data
models/
data/
*.pkl
*.pth
*.onnx
*.h5

# Temporary files
tmp/
temp/
*.tmp

# OS
.DS_Store
Thumbs.db

# MLflow
mlruns/

# Weights & Biases
wandb/

# TensorBoard
runs/
"""
        
        gitignore_file.write_text(gitignore_content)
    
    def _create_project_config(self, configs_dir: Path, config: ProjectConfig) -> None:
        """Create project configuration file."""
        config_data = {
            "project": {
                "name": config.name,
                "version": config.version,
                "description": config.description,
                "author": config.author,
                "type": config.project_type.value,
                "created_at": config.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat()
            },
            "problem_definition": {
                "title": config.problem_definition.title,
                "description": config.problem_definition.description,
                "complexity": config.problem_definition.complexity.value,
                "objectives": config.problem_definition.objectives,
                "constraints": config.problem_definition.constraints,
                "success_metrics": config.problem_definition.success_metrics,
                "business_value": config.problem_definition.business_value,
                "timeline": config.problem_definition.timeline,
                "budget": config.problem_definition.budget
            },
            "dataset": {
                "name": config.dataset_info.name,
                "type": config.dataset_info.type.value,
                "path": config.dataset_info.path,
                "size": config.dataset_info.size,
                "features": config.dataset_info.features,
                "target_column": config.dataset_info.target_column,
                "validation_split": config.dataset_info.validation_split,
                "test_split": config.dataset_info.test_split,
                "description": config.dataset_info.description,
                "source": config.dataset_info.source,
                "license": config.dataset_info.license,
                "last_updated": config.dataset_info.last_updated.isoformat()
            },
            "dependencies": config.dependencies,
            "environment": config.environment
        }
        
        config_file = configs_dir / "project.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def analyze_dataset(self, dataset_path: str) -> DatasetInfo:
        """Analyze a dataset and extract information."""
        try:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
            # Determine dataset type based on file extension
            if dataset_path.suffix.lower() in ['.csv', '.tsv']:
                dataset_type = DatasetType.TABULAR
                df = pd.read_csv(dataset_path)
                size = len(df)
                features = list(df.columns)
                target_column = None  # User should specify
            elif dataset_path.suffix.lower() in ['.json', '.jsonl']:
                dataset_type = DatasetType.TEXT
                with open(dataset_path, 'r') as f:
                    data = [json.loads(line) for line in f]
                size = len(data)
                features = list(data[0].keys()) if data else []
                target_column = None
            elif dataset_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                dataset_type = DatasetType.IMAGE
                # For single image, count as 1
                size = 1
                features = ["image"]
                target_column = None
            elif dataset_path.is_dir():
                # Directory of images
                image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(dataset_path.glob(f"*{ext}"))
                size = len(image_files)
                dataset_type = DatasetType.IMAGE
                features = ["image"]
                target_column = None
            else:
                dataset_type = DatasetType.CUSTOM
                size = 0
                features = []
                target_column = None
            
            return DatasetInfo(
                name=dataset_path.stem,
                type=dataset_type,
                path=str(dataset_path),
                size=size,
                features=features,
                target_column=target_column,
                description=f"Dataset from {dataset_path}",
                source="local",
                license="unknown"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze dataset {dataset_path}: {e}")
            raise
    
    def validate_project_config(self, config: ProjectConfig) -> List[str]:
        """Validate project configuration and return any issues."""
        issues = []
        
        # Check required fields
        if not config.name or not config.name.strip():
            issues.append("Project name is required")
        
        if not config.description or not config.description.strip():
            issues.append("Project description is required")
        
        if not config.author or not config.author.strip():
            issues.append("Project author is required")
        
        # Check problem definition
        if not config.problem_definition.title:
            issues.append("Problem title is required")
        
        if not config.problem_definition.description:
            issues.append("Problem description is required")
        
        if not config.problem_definition.objectives:
            issues.append("At least one objective is required")
        
        if not config.problem_definition.success_metrics:
            issues.append("At least one success metric is required")
        
        # Check dataset info
        if not config.dataset_info.name:
            issues.append("Dataset name is required")
        
        if not config.dataset_info.path:
            issues.append("Dataset path is required")
        
        # Check splits
        if config.dataset_info.validation_split + config.dataset_info.test_split >= 1.0:
            issues.append("Validation + test split must be less than 1.0")
        
        return issues
    
    def get_project_summary(self, config: ProjectConfig) -> Dict[str, Any]:
        """Get a comprehensive project summary."""
        return {
            "project_info": {
                "name": config.name,
                "version": config.version,
                "type": config.project_type.value,
                "complexity": config.problem_definition.complexity.value,
                "created_at": config.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat()
            },
            "problem_summary": {
                "title": config.problem_definition.title,
                "objectives_count": len(config.problem_definition.objectives),
                "metrics_count": len(config.problem_definition.success_metrics),
                "timeline": config.problem_definition.timeline,
                "budget": config.problem_definition.budget
            },
            "dataset_summary": {
                "name": config.dataset_info.name,
                "type": config.dataset_info.type.value,
                "size": config.dataset_info.size,
                "features_count": len(config.dataset_info.features),
                "source": config.dataset_info.source
            },
            "dependencies_count": len(config.dependencies),
            "environment_variables_count": len(config.environment)
        }


class ProjectInitializer:
    """High-level project initialization service."""
    
    def __init__(self, project_root: str = "."):
        """Initialize the project initializer."""
        self.manager = ProjectManager(project_root)
        self.logger = logging.getLogger(__name__)
    
    async def initialize_project(
        self,
        name: str,
        description: str,
        project_type: ProjectType,
        complexity: ProblemComplexity,
        dataset_path: str,
        author: str = "Unknown",
        version: str = "1.0.0"
    ) -> ProjectConfig:
        """Initialize a complete ML project."""
        try:
            # Analyze dataset
            dataset_info = self.manager.analyze_dataset(dataset_path)
            
            # Create problem definition
            problem_definition = ProblemDefinition(
                title=f"{name} ML Project",
                description=description,
                project_type=project_type,
                complexity=complexity,
                objectives=["Improve model performance", "Reduce training time", "Increase accuracy"],
                constraints=["Limited computational resources", "Time constraints"],
                success_metrics=["Accuracy", "F1 Score", "Training time"],
                business_value="Improve business outcomes through better ML models",
                timeline="3 months",
                budget=10000.0
            )
            
            # Create project config
            config = ProjectConfig(
                name=name,
                version=version,
                description=description,
                author=author,
                project_type=project_type,
                problem_definition=problem_definition,
                dataset_info=dataset_info,
                dependencies={
                    "torch": ">=2.0.0",
                    "transformers": ">=4.30.0",
                    "fastapi": ">=0.100.0"
                },
                environment={
                    "PYTHONPATH": ".",
                    "LOG_LEVEL": "INFO"
                }
            )
            
            # Validate config
            issues = self.manager.validate_project_config(config)
            if issues:
                raise ValueError(f"Configuration validation failed: {', '.join(issues)}")
            
            # Create project structure
            structure = self.manager.create_project_structure(config)
            
            self.logger.info(f"Successfully initialized project '{name}' in {self.manager.project_root}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to initialize project: {e}")
            raise
    
    async def get_project_status(self, project_path: str = None) -> Dict[str, Any]:
        """Get the status of an existing project."""
        try:
            if project_path is None:
                project_path = str(self.manager.project_root)
            
            project_path = Path(project_path)
            if not project_path.exists():
                return {"status": "not_found", "path": str(project_path)}
            
            # Check for key project files
            has_requirements = (project_path / "requirements.txt").exists()
            has_readme = (project_path / "README.md").exists()
            has_configs = (project_path / "configs").exists()
            has_data = (project_path / "data").exists()
            has_models = (project_path / "models").exists()
            
            return {
                "status": "found",
                "path": str(project_path),
                "has_requirements": has_requirements,
                "has_readme": has_readme,
                "has_configs": has_configs,
                "has_data": has_data,
                "has_models": has_models,
                "is_complete": all([has_requirements, has_readme, has_configs, has_data, has_models])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get project status: {e}")
            return {"status": "error", "message": str(e)}


# Global utility functions
def get_project_manager(project_root: str = ".") -> ProjectManager:
    """Get a global project manager instance."""
    return ProjectManager(project_root)


def get_project_initializer(project_root: str = ".") -> ProjectInitializer:
    """Get a global project initializer instance."""
    return ProjectInitializer(project_root)
