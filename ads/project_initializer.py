from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
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
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.optimized_config import settings
            from PIL import Image
            import glob
from typing import Dict, Any
from dataclasses import dataclass
from pydantic import BaseSettings
from typing import Any, List, Dict, Optional
"""
Project Initializer for Onyx Ads Backend

This module provides a comprehensive framework for initializing ML projects with:
- Clear problem definition and scope
- Dataset analysis and validation
- Project structure setup
- Configuration management
- Documentation generation
"""


logger = setup_logger()

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
    business_objective: str
    success_metrics: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'description': self.description,
            'project_type': self.project_type.value,
            'complexity': self.complexity.value,
            'business_objective': self.business_objective,
            'success_metrics': self.success_metrics,
            'constraints': self.constraints,
            'assumptions': self.assumptions,
            'risks': self.risks,
            'stakeholders': self.stakeholders
        }

class DatasetAnalyzer:
    """Comprehensive dataset analysis and validation."""
    
    def __init__(self, dataset_info: DatasetInfo):
        
    """__init__ function."""
self.dataset_info = dataset_info
        self.logger = logger
        self.analysis_results = {}
    
    async def analyze_dataset(self) -> Dict[str, Any]:
        """Perform comprehensive dataset analysis."""
        self.logger.info(f"Starting analysis of dataset: {self.dataset_info.name}")
        
        analysis = {
            'basic_info': await self._analyze_basic_info(),
            'data_quality': await self._analyze_data_quality(),
            'statistics': await self._analyze_statistics(),
            'distributions': await self._analyze_distributions(),
            'correlations': await self._analyze_correlations(),
            'missing_data': await self._analyze_missing_data(),
            'outliers': await self._analyze_outliers(),
            'recommendations': []
        }
        
        # Generate recommendations
        analysis['recommendations'] = await self._generate_recommendations(analysis)
        
        self.analysis_results = analysis
        return analysis
    
    async def _analyze_basic_info(self) -> Dict[str, Any]:
        """Analyze basic dataset information."""
        try:
            if self.dataset_info.type == DatasetType.TABULAR:
                df = pd.read_csv(self.dataset_info.path)
                return {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'file_size': os.path.getsize(self.dataset_info.path)
                }
            elif self.dataset_info.type == DatasetType.TEXT:
                # Analyze text dataset
                return await self._analyze_text_dataset()
            elif self.dataset_info.type == DatasetType.IMAGE:
                # Analyze image dataset
                return await self._analyze_image_dataset()
            else:
                return {'error': f'Dataset type {self.dataset_info.type} not yet supported'}
        except Exception as e:
            self.logger.error(f"Error analyzing basic info: {e}")
            return {'error': str(e)}
    
    async def _analyze_text_dataset(self) -> Dict[str, Any]:
        """Analyze text dataset characteristics."""
        try:
            with open(self.dataset_info.path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                lines = f.readlines()
            
            # Basic text statistics
            total_chars = sum(len(line) for line in lines)
            total_words = sum(len(line.split()) for line in lines)
            avg_line_length = total_chars / len(lines) if lines else 0
            avg_words_per_line = total_words / len(lines) if lines else 0
            
            return {
                'total_lines': len(lines),
                'total_characters': total_chars,
                'total_words': total_words,
                'avg_line_length': avg_line_length,
                'avg_words_per_line': avg_words_per_line,
                'file_size': os.path.getsize(self.dataset_info.path)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing text dataset: {e}")
            return {'error': str(e)}
    
    async def _analyze_image_dataset(self) -> Dict[str, Any]:
        """Analyze image dataset characteristics."""
        try:
            
            image_files = glob.glob(os.path.join(self.dataset_info.path, "*.jpg")) + \
                         glob.glob(os.path.join(self.dataset_info.path, "*.png")) + \
                         glob.glob(os.path.join(self.dataset_info.path, "*.jpeg"))
            
            if not image_files:
                return {'error': 'No image files found'}
            
            # Analyze first few images for statistics
            sample_images = image_files[:min(100, len(image_files))]
            sizes = []
            formats = []
            
            for img_path in sample_images:
                try:
                    with Image.open(img_path) as img:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        sizes.append(img.size)
                        formats.append(img.format)
                except Exception as e:
                    self.logger.warning(f"Could not analyze image {img_path}: {e}")
            
            return {
                'total_images': len(image_files),
                'sample_analyzed': len(sample_images),
                'common_formats': list(set(formats)),
                'size_range': {
                    'min_width': min(s[0] for s in sizes) if sizes else 0,
                    'max_width': max(s[0] for s in sizes) if sizes else 0,
                    'min_height': min(s[1] for s in sizes) if sizes else 0,
                    'max_height': max(s[1] for s in sizes) if sizes else 0
                }
            }
        except Exception as e:
            self.logger.error(f"Error analyzing image dataset: {e}")
            return {'error': str(e)}
    
    async def _analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality issues."""
        try:
            if self.dataset_info.type == DatasetType.TABULAR:
                df = pd.read_csv(self.dataset_info.path)
                
                quality_issues = {
                    'duplicates': df.duplicated().sum(),
                    'null_counts': df.isnull().sum().to_dict(),
                    'unique_counts': df.nunique().to_dict(),
                    'data_types': df.dtypes.to_dict()
                }
                
                # Check for potential data quality issues
                issues = []
                if quality_issues['duplicates'] > 0:
                    issues.append(f"Found {quality_issues['duplicates']} duplicate rows")
                
                null_columns = [col for col, count in quality_issues['null_counts'].items() if count > 0]
                if null_columns:
                    issues.append(f"Missing values in columns: {null_columns}")
                
                quality_issues['issues'] = issues
                return quality_issues
            else:
                return {'message': f'Data quality analysis for {self.dataset_info.type} not implemented'}
        except Exception as e:
            self.logger.error(f"Error analyzing data quality: {e}")
            return {'error': str(e)}
    
    async def _analyze_statistics(self) -> Dict[str, Any]:
        """Analyze statistical properties of the dataset."""
        try:
            if self.dataset_info.type == DatasetType.TABULAR:
                df = pd.read_csv(self.dataset_info.path)
                
                # Numeric columns statistics
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                stats = {}
                
                for col in numeric_cols:
                    stats[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'median': df[col].median(),
                        'q25': df[col].quantile(0.25),
                        'q75': df[col].quantile(0.75)
                    }
                
                return {'numeric_statistics': stats}
            else:
                return {'message': f'Statistical analysis for {self.dataset_info.type} not implemented'}
        except Exception as e:
            self.logger.error(f"Error analyzing statistics: {e}")
            return {'error': str(e)}
    
    async def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze data distributions."""
        try:
            if self.dataset_info.type == DatasetType.TABULAR:
                df = pd.read_csv(self.dataset_info.path)
                
                distributions = {}
                for col in df.columns:
                    if df[col].dtype in ['object', 'category']:
                        # Categorical distribution
                        distributions[col] = df[col].value_counts().to_dict()
                    elif df[col].dtype in ['int64', 'float64']:
                        # Numeric distribution (binned)
                        distributions[col] = df[col].value_counts(bins=10).to_dict()
                
                return distributions
            else:
                return {'message': f'Distribution analysis for {self.dataset_info.type} not implemented'}
        except Exception as e:
            self.logger.error(f"Error analyzing distributions: {e}")
            return {'error': str(e)}
    
    async def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between features."""
        try:
            if self.dataset_info.type == DatasetType.TABULAR:
                df = pd.read_csv(self.dataset_info.path)
                numeric_df = df.select_dtypes(include=[np.number])
                
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    return {
                        'correlation_matrix': corr_matrix.to_dict(),
                        'high_correlations': self._find_high_correlations(corr_matrix)
                    }
                else:
                    return {'message': 'Not enough numeric columns for correlation analysis'}
            else:
                return {'message': f'Correlation analysis for {self.dataset_info.type} not implemented'}
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return {'error': str(e)}
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find highly correlated feature pairs."""
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        return high_corr
    
    async def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        try:
            if self.dataset_info.type == DatasetType.TABULAR:
                df = pd.read_csv(self.dataset_info.path)
                
                missing_analysis = {
                    'total_missing': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                    'missing_by_column': df.isnull().sum().to_dict(),
                    'missing_patterns': self._analyze_missing_patterns(df)
                }
                
                return missing_analysis
            else:
                return {'message': f'Missing data analysis for {self.dataset_info.type} not implemented'}
        except Exception as e:
            self.logger.error(f"Error analyzing missing data: {e}")
            return {'error': str(e)}
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data."""
        missing_cols = df.columns[df.isnull().any()].tolist()
        patterns = {}
        
        for col in missing_cols:
            # Check if missing values are related to other columns
            missing_mask = df[col].isnull()
            related_patterns = {}
            
            for other_col in df.columns:
                if other_col != col:
                    # Check if missing values in this column correlate with other columns
                    if df[other_col].dtype in ['object', 'category']:
                        pattern = df[other_col][missing_mask].value_counts().to_dict()
                        if pattern:
                            related_patterns[other_col] = pattern
            
            if related_patterns:
                patterns[col] = related_patterns
        
        return patterns
    
    async def _analyze_outliers(self) -> Dict[str, Any]:
        """Analyze outliers in the dataset."""
        try:
            if self.dataset_info.type == DatasetType.TABULAR:
                df = pd.read_csv(self.dataset_info.path)
                numeric_df = df.select_dtypes(include=[np.number])
                
                outliers = {}
                for col in numeric_df.columns:
                    Q1 = numeric_df[col].quantile(0.25)
                    Q3 = numeric_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_count = ((numeric_df[col] < lower_bound) | 
                                    (numeric_df[col] > upper_bound)).sum()
                    
                    outliers[col] = {
                        'count': outliers_count,
                        'percentage': (outliers_count / len(numeric_df)) * 100,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                
                return outliers
            else:
                return {'message': f'Outlier analysis for {self.dataset_info.type} not implemented'}
        except Exception as e:
            self.logger.error(f"Error analyzing outliers: {e}")
            return {'error': str(e)}
    
    async def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Data quality recommendations
        if 'data_quality' in analysis:
            quality = analysis['data_quality']
            if 'issues' in quality:
                for issue in quality['issues']:
                    if 'duplicate' in issue.lower():
                        recommendations.append("Consider removing duplicate rows or investigating their source")
                    if 'missing' in issue.lower():
                        recommendations.append("Implement missing data handling strategy (imputation, removal, or modeling)")
        
        # Correlation recommendations
        if 'correlations' in analysis and 'high_correlations' in analysis['correlations']:
            high_corr = analysis['correlations']['high_correlations']
            if high_corr:
                recommendations.append(f"Consider feature selection to remove highly correlated features ({len(high_corr)} pairs found)")
        
        # Outlier recommendations
        if 'outliers' in analysis:
            outliers = analysis['outliers']
            high_outlier_cols = [col for col, data in outliers.items() if data['percentage'] > 5]
            if high_outlier_cols:
                recommendations.append(f"Investigate outliers in columns: {high_outlier_cols}")
        
        # Dataset size recommendations
        if 'basic_info' in analysis and 'shape' in analysis['basic_info']:
            shape = analysis['basic_info']['shape']
            if shape[0] < 1000:
                recommendations.append("Dataset is small - consider data augmentation or collecting more data")
            elif shape[0] > 100000:
                recommendations.append("Large dataset detected - consider sampling for initial exploration")
        
        return recommendations

class ProjectInitializer:
    """Main project initialization orchestrator."""
    
    def __init__(self, project_name: str, base_path: str = "./projects"):
        
    """__init__ function."""
self.project_name = project_name
        self.base_path = Path(base_path)
        self.project_path = self.base_path / project_name
        self.logger = logger
        
        # Initialize components
        self.problem_definition = None
        self.dataset_info = None
        self.dataset_analyzer = None
        
    async def initialize_project(self, 
                               problem_def: ProblemDefinition,
                               dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Initialize a complete ML project."""
        self.logger.info(f"Initializing project: {self.project_name}")
        
        self.problem_definition = problem_def
        self.dataset_info = dataset_info
        
        # Create project structure
        await self._create_project_structure()
        
        # Analyze dataset
        self.dataset_analyzer = DatasetAnalyzer(dataset_info)
        analysis_results = await self.dataset_analyzer.analyze_dataset()
        
        # Generate configuration
        config = await self._generate_project_config()
        
        # Create documentation
        await self._create_documentation(analysis_results)
        
        # Validate project setup
        validation_results = await self._validate_project_setup()
        
        return {
            'project_name': self.project_name,
            'project_path': str(self.project_path),
            'problem_definition': problem_def.to_dict(),
            'dataset_analysis': analysis_results,
            'configuration': config,
            'validation': validation_results,
            'status': 'initialized'
        }
    
    async def _create_project_structure(self) -> Any:
        """Create the project directory structure."""
        try:
            # Create main project directory
            self.project_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            directories = [
                'data',
                'models',
                'notebooks',
                'src',
                'tests',
                'configs',
                'logs',
                'results',
                'docs',
                'scripts'
            ]
            
            for directory in directories:
                (self.project_path / directory).mkdir(exist_ok=True)
            
            # Create data subdirectories
            data_dirs = ['raw', 'processed', 'interim', 'external']
            for data_dir in data_dirs:
                (self.project_path / 'data' / data_dir).mkdir(exist_ok=True)
            
            # Create src subdirectories
            src_dirs = ['data', 'models', 'features', 'utils', 'api']
            for src_dir in src_dirs:
                (self.project_path / 'src' / src_dir).mkdir(exist_ok=True)
            
            self.logger.info(f"Project structure created at: {self.project_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating project structure: {e}")
            raise
    
    async def _generate_project_config(self) -> Dict[str, Any]:
        """Generate project configuration files."""
        try:
            config = {
                'project': {
                    'name': self.project_name,
                    'type': self.problem_definition.project_type.value,
                    'complexity': self.problem_definition.complexity.value,
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0.0'
                },
                'problem_definition': self.problem_definition.to_dict(),
                'dataset': {
                    'name': self.dataset_info.name,
                    'type': self.dataset_info.type.value,
                    'path': self.dataset_info.path,
                    'size': self.dataset_info.size,
                    'features': self.dataset_info.features,
                    'target_column': self.dataset_info.target_column
                },
                'training': {
                    'validation_split': self.dataset_info.validation_split,
                    'test_split': self.dataset_info.test_split,
                    'random_state': 42,
                    'batch_size': 32,
                    'learning_rate': 1e-4,
                    'epochs': 100
                },
                'optimization': {
                    'enable_mixed_precision': True,
                    'enable_gradient_checkpointing': True,
                    'enable_model_compilation': True,
                    'num_workers': 4
                }
            }
            
            # Save configuration files
            config_path = self.project_path / 'configs' / 'project_config.yaml'
            with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(config, f, default_flow_style=False)
            
            # Create Python config
            python_config_path = self.project_path / 'configs' / 'config.py'
            await self._create_python_config(python_config_path, config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error generating project config: {e}")
            raise
    
    async def _create_python_config(self, config_path: Path, config: Dict[str, Any]):
        """Create Python configuration file."""
        config_content = f'''"""
Project configuration for {self.project_name}
"""

@dataclass
class ProjectConfig:
    """Project configuration."""
    name: str = "{config['project']['name']}"
    type: str = "{config['project']['type']}"
    complexity: str = "{config['project']['complexity']}"
    version: str = "{config['project']['version']}"

class Settings(BaseSettings):
    """Application settings."""
    
    # Project settings
    project_name: str = "{config['project']['name']}"
    project_type: str = "{config['project']['type']}"
    
    # Dataset settings
    dataset_path: str = "{config['dataset']['path']}"
    dataset_type: str = "{config['dataset']['type']}"
    target_column: str = "{config['dataset']['target_column'] or 'None'}"
    
    # Training settings
    validation_split: float = {config['training']['validation_split']}
    test_split: float = {config['training']['test_split']}
    batch_size: int = {config['training']['batch_size']}
    learning_rate: float = {config['training']['learning_rate']}
    epochs: int = {config['training']['epochs']}
    
    # Optimization settings
    enable_mixed_precision: bool = {config['optimization']['enable_mixed_precision']}
    enable_gradient_checkpointing: bool = {config['optimization']['enable_gradient_checkpointing']}
    enable_model_compilation: bool = {config['optimization']['enable_model_compilation']}
    num_workers: int = {config['optimization']['num_workers']}
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()
'''
        
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(config_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    async def _create_documentation(self, analysis_results: Dict[str, Any]):
        """Create project documentation."""
        try:
            # Create README
            readme_path = self.project_path / 'README.md'
            await self._create_readme(readme_path, analysis_results)
            
            # Create problem definition document
            problem_doc_path = self.project_path / 'docs' / 'problem_definition.md'
            await self._create_problem_documentation(problem_doc_path)
            
            # Create dataset documentation
            dataset_doc_path = self.project_path / 'docs' / 'dataset_analysis.md'
            await self._create_dataset_documentation(dataset_doc_path, analysis_results)
            
            # Create project plan
            plan_path = self.project_path / 'docs' / 'project_plan.md'
            await self._create_project_plan(plan_path)
            
        except Exception as e:
            self.logger.error(f"Error creating documentation: {e}")
            raise
    
    async def _create_readme(self, readme_path: Path, analysis_results: Dict[str, Any]):
        """Create project README."""
        readme_content = f'''# {self.project_name}

## Project Overview

**Problem Type:** {self.problem_definition.project_type.value}  
**Complexity:** {self.problem_definition.complexity.value}  
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Problem Definition

{self.problem_definition.description}

### Business Objective
{self.problem_definition.business_objective}

### Success Metrics
{chr(10).join(f"- {metric}" for metric in self.problem_definition.success_metrics)}

## Dataset Information

- **Name:** {self.dataset_info.name}
- **Type:** {self.dataset_info.type.value}
- **Size:** {analysis_results.get('basic_info', {}).get('shape', 'Unknown')}
- **Features:** {len(self.dataset_info.features)} columns

## Project Structure

```
{self.project_name}/
├── data/           # Dataset files
├── models/         # Trained models
├── notebooks/      # Jupyter notebooks
├── src/           # Source code
├── tests/         # Test files
├── configs/       # Configuration files
├── logs/          # Log files
├── results/       # Results and outputs
├── docs/          # Documentation
└── scripts/       # Utility scripts
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the project:**
   ```bash
   cp configs/project_config.yaml.example configs/project_config.yaml
   # Edit configuration as needed
   ```

3. **Run the project:**
   ```bash
   python src/main.py
   ```

## Key Findings

### Dataset Analysis
{chr(10).join(f"- {rec}" for rec in analysis_results.get('recommendations', []))}

## Documentation

- [Problem Definition](docs/problem_definition.md)
- [Dataset Analysis](docs/dataset_analysis.md)
- [Project Plan](docs/project_plan.md)

## Contributing

1. Follow the established code conventions
2. Add tests for new features
3. Update documentation as needed
4. Use the provided logging and monitoring tools
'''
        
        with open(readme_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(readme_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    async def _create_problem_documentation(self, doc_path: Path):
        """Create detailed problem definition documentation."""
        doc_content = f'''# Problem Definition

## Overview

**Title:** {self.problem_definition.title}  
**Type:** {self.problem_definition.project_type.value}  
**Complexity:** {self.problem_definition.complexity.value}

## Description

{self.problem_definition.description}

## Business Objective

{self.problem_definition.business_objective}

## Success Metrics

{chr(10).join(f"1. {metric}" for metric in self.problem_definition.success_metrics)}

## Constraints

{chr(10).join(f"- {constraint}" for constraint in self.problem_definition.constraints)}

## Assumptions

{chr(10).join(f"- {assumption}" for assumption in self.problem_definition.assumptions)}

## Risks

{chr(10).join(f"- {risk}" for risk in self.problem_definition.risks)}

## Stakeholders

{chr(10).join(f"- {stakeholder}" for stakeholder in self.problem_definition.stakeholders)}

## Technical Requirements

### Data Requirements
- Dataset type: {self.dataset_info.type.value}
- Expected size: {self.dataset_info.size} samples
- Features: {len(self.dataset_info.features)} columns

### Model Requirements
- Performance: Meet success metrics
- Scalability: Handle production load
- Interpretability: Provide explainable results

### Infrastructure Requirements
- Compute: GPU support for training
- Storage: Sufficient space for models and data
- Monitoring: Track model performance and drift
'''
        
        with open(doc_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(doc_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    async def _create_dataset_documentation(self, doc_path: Path, analysis_results: Dict[str, Any]):
        """Create dataset analysis documentation."""
        doc_content = f'''# Dataset Analysis

## Dataset Information

- **Name:** {self.dataset_info.name}
- **Type:** {self.dataset_info.type.value}
- **Path:** {self.dataset_info.path}
- **Size:** {analysis_results.get('basic_info', {}).get('shape', 'Unknown')}
- **Description:** {self.dataset_info.description}
- **Source:** {self.dataset_info.source}
- **License:** {self.dataset_info.license}

## Analysis Results

### Basic Information
```yaml
{json.dumps(analysis_results.get('basic_info', {}), indent=2)}
```

### Data Quality
```yaml
{json.dumps(analysis_results.get('data_quality', {}), indent=2)}
```

### Statistics
```yaml
{json.dumps(analysis_results.get('statistics', {}), indent=2)}
```

### Missing Data Analysis
```yaml
{json.dumps(analysis_results.get('missing_data', {}), indent=2)}
```

### Outlier Analysis
```yaml
{json.dumps(analysis_results.get('outliers', {}), indent=2)}
```

## Recommendations

{chr(10).join(f"1. {rec}" for rec in analysis_results.get('recommendations', []))}

## Data Preprocessing Plan

Based on the analysis, the following preprocessing steps are recommended:

1. **Data Cleaning**
   - Handle missing values
   - Remove duplicates
   - Address outliers

2. **Feature Engineering**
   - Create relevant features
   - Handle categorical variables
   - Scale numerical features

3. **Data Splitting**
   - Training: {1 - self.dataset_info.validation_split - self.dataset_info.test_split:.1%}
   - Validation: {self.dataset_info.validation_split:.1%}
   - Test: {self.dataset_info.test_split:.1%}
'''
        
        with open(doc_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(doc_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    async def _create_project_plan(self, plan_path: Path):
        """Create project execution plan."""
        plan_content = f'''# Project Plan

## Phase 1: Data Preparation (Week 1)

### Tasks
- [ ] Data cleaning and preprocessing
- [ ] Feature engineering
- [ ] Data splitting (train/validation/test)
- [ ] Data validation and quality checks

### Deliverables
- Cleaned dataset
- Feature engineering pipeline
- Data validation report

## Phase 2: Model Development (Week 2-3)

### Tasks
- [ ] Baseline model implementation
- [ ] Model architecture design
- [ ] Training pipeline setup
- [ ] Hyperparameter optimization

### Deliverables
- Baseline model
- Training pipeline
- Model performance metrics

## Phase 3: Model Optimization (Week 4)

### Tasks
- [ ] Model performance analysis
- [ ] Feature selection/importance
- [ ] Model tuning and optimization
- [ ] Ensemble methods (if applicable)

### Deliverables
- Optimized model
- Feature importance analysis
- Performance comparison report

## Phase 4: Evaluation and Deployment (Week 5)

### Tasks
- [ ] Model evaluation on test set
- [ ] Performance monitoring setup
- [ ] Model deployment preparation
- [ ] Documentation finalization

### Deliverables
- Final model
- Deployment pipeline
- Complete documentation

## Success Criteria

{chr(10).join(f"- {metric}" for metric in self.problem_definition.success_metrics)}

## Risk Mitigation

{chr(10).join(f"- {risk}" for risk in self.problem_definition.risks)}
'''
        
        with open(plan_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(plan_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    async def _validate_project_setup(self) -> Dict[str, Any]:
        """Validate the project setup."""
        validation_results = {
            'structure': await self._validate_project_structure(),
            'configuration': await self._validate_configuration(),
            'dataset': await self._validate_dataset(),
            'dependencies': await self._validate_dependencies()
        }
        
        # Overall validation status
        all_valid = all(
            result.get('status', 'failed') == 'valid' 
            for result in validation_results.values()
        )
        
        validation_results['overall_status'] = 'valid' if all_valid else 'failed'
        
        return validation_results
    
    async def _validate_project_structure(self) -> Dict[str, Any]:
        """Validate project directory structure."""
        required_dirs = [
            'data', 'models', 'notebooks', 'src', 'tests', 
            'configs', 'logs', 'results', 'docs', 'scripts'
        ]
        
        missing_dirs = []
        for directory in required_dirs:
            if not (self.project_path / directory).exists():
                missing_dirs.append(directory)
        
        return {
            'status': 'valid' if not missing_dirs else 'failed',
            'missing_directories': missing_dirs,
            'message': f"Project structure validation {'passed' if not missing_dirs else 'failed'}"
        }
    
    async def _validate_configuration(self) -> Dict[str, Any]:
        """Validate project configuration."""
        config_file = self.project_path / 'configs' / 'project_config.yaml'
        
        if not config_file.exists():
            return {
                'status': 'failed',
                'message': 'Configuration file not found'
            }
        
        try:
            with open(config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config = yaml.safe_load(f)
            
            required_keys = ['project', 'problem_definition', 'dataset', 'training']
            missing_keys = [key for key in required_keys if key not in config]
            
            return {
                'status': 'valid' if not missing_keys else 'failed',
                'missing_keys': missing_keys,
                'message': f"Configuration validation {'passed' if not missing_keys else 'failed'}"
            }
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error reading configuration: {e}'
            }
    
    async def _validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset accessibility."""
        try:
            if not os.path.exists(self.dataset_info.path):
                return {
                    'status': 'failed',
                    'message': f'Dataset not found at: {self.dataset_info.path}'
                }
            
            # Check if dataset can be read
            if self.dataset_info.type == DatasetType.TABULAR:
                df = pd.read_csv(self.dataset_info.path)
                return {
                    'status': 'valid',
                    'message': f'Dataset loaded successfully: {df.shape}',
                    'shape': df.shape
                }
            else:
                return {
                    'status': 'valid',
                    'message': f'Dataset exists at: {self.dataset_info.path}'
                }
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error validating dataset: {e}'
            }
    
    async def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate required dependencies."""
        required_packages = [
            'torch', 'transformers', 'pandas', 'numpy', 
            'scikit-learn', 'matplotlib', 'seaborn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        return {
            'status': 'valid' if not missing_packages else 'failed',
            'missing_packages': missing_packages,
            'message': f"Dependencies validation {'passed' if not missing_packages else 'failed'}"
        }

# Utility functions for project initialization
async def create_project(project_name: str,
                        problem_definition: ProblemDefinition,
                        dataset_info: DatasetInfo,
                        base_path: str = "./projects") -> Dict[str, Any]:
    """Create a new ML project with comprehensive initialization."""
    
    initializer = ProjectInitializer(project_name, base_path)
    return await initializer.initialize_project(problem_definition, dataset_info)

async def analyze_dataset(dataset_info: DatasetInfo) -> Dict[str, Any]:
    """Analyze a dataset and return comprehensive analysis results."""
    
    analyzer = DatasetAnalyzer(dataset_info)
    return await analyzer.analyze_dataset()

def create_problem_definition(title: str,
                            description: str,
                            project_type: ProjectType,
                            complexity: ProblemComplexity,
                            business_objective: str,
                            success_metrics: List[str] = None,
                            constraints: List[str] = None,
                            assumptions: List[str] = None,
                            risks: List[str] = None,
                            stakeholders: List[str] = None) -> ProblemDefinition:
    """Create a problem definition with all required components."""
    
    return ProblemDefinition(
        title=title,
        description=description,
        project_type=project_type,
        complexity=complexity,
        business_objective=business_objective,
        success_metrics=success_metrics or [],
        constraints=constraints or [],
        assumptions=assumptions or [],
        risks=risks or [],
        stakeholders=stakeholders or []
    )

def create_dataset_info(name: str,
                       type: DatasetType,
                       path: str,
                       features: List[str] = None,
                       target_column: str = None,
                       description: str = "",
                       source: str = "",
                       license: str = "") -> DatasetInfo:
    """Create dataset information with all required components."""
    
    return DatasetInfo(
        name=name,
        type=type,
        path=path,
        features=features or [],
        target_column=target_column,
        description=description,
        source=source,
        license=license
    )

# Example usage
if __name__ == "__main__":
    # Example: Create a text classification project
    async def example_project():
        
    """example_project function."""
# Define the problem
        problem_def = create_problem_definition(
            title="Ad Content Classification",
            description="Classify ad content into different categories for better targeting",
            project_type=ProjectType.CLASSIFICATION,
            complexity=ProblemComplexity.MODERATE,
            business_objective="Improve ad targeting accuracy by 20%",
            success_metrics=["Accuracy > 85%", "F1-score > 0.8", "Processing time < 1s"],
            constraints=["Must work with real-time data", "Model size < 100MB"],
            assumptions=["Text data is clean", "Categories are well-defined"],
            risks=["Data quality issues", "Model bias", "Performance degradation"],
            stakeholders=["Marketing team", "Data science team", "Product team"]
        )
        
        # Define the dataset
        dataset_info = create_dataset_info(
            name="Ad Content Dataset",
            type=DatasetType.TEXT,
            path="./data/ad_content.csv",
            features=["text", "category", "engagement_rate"],
            target_column="category",
            description="Dataset containing ad content and their categories",
            source="Internal marketing data",
            license="Proprietary"
        )
        
        # Create the project
        result = await create_project(
            project_name="ad_classification",
            problem_definition=problem_def,
            dataset_info=dataset_info
        )
        
        print(f"Project created successfully: {result}")
    
    # Run the example
    asyncio.run(example_project()) 