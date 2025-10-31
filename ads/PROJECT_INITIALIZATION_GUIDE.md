# Project Initialization Guide

A comprehensive guide for initializing ML projects with clear problem definition and dataset analysis in the Onyx Ads Backend system.

## 🎯 Overview

This guide covers the systematic approach to starting ML projects with:
- **Clear problem definition** - Structured approach to defining ML problems
- **Dataset analysis** - Comprehensive dataset exploration and validation
- **Project structure** - Standardized project organization
- **Configuration management** - Automated configuration generation
- **Documentation** - Auto-generated project documentation

## 🚀 Quick Start

### 1. Basic Project Initialization

```python
from onyx.server.features.ads.project_initializer import (
    create_project, create_problem_definition, create_dataset_info,
    ProjectType, DatasetType, ProblemComplexity
)

# Define the problem
problem_def = create_problem_definition(
    title="Ad Content Classification",
    description="Classify ad content into categories for better targeting",
    project_type=ProjectType.CLASSIFICATION,
    complexity=ProblemComplexity.MODERATE,
    business_objective="Improve ad targeting accuracy by 20%",
    success_metrics=["Accuracy > 85%", "F1-score > 0.8"],
    constraints=["Real-time processing", "Model size < 100MB"],
    assumptions=["Clean text data", "Well-defined categories"],
    risks=["Data quality issues", "Model bias"],
    stakeholders=["Marketing team", "Data science team"]
)

# Define the dataset
dataset_info = create_dataset_info(
    name="Ad Content Dataset",
    type=DatasetType.TEXT,
    path="./data/ad_content.csv",
    features=["text", "category", "engagement_rate"],
    target_column="category",
    description="Dataset containing ad content and categories",
    source="Internal marketing data"
)

# Create the project
result = await create_project(
    project_name="ad_classification",
    problem_definition=problem_def,
    dataset_info=dataset_info
)
```

### 2. Dataset Analysis Only

```python
from onyx.server.features.ads.project_initializer import analyze_dataset

# Analyze existing dataset
analysis = await analyze_dataset(dataset_info)
print(f"Dataset analysis: {analysis}")
```

## 📋 Problem Definition Framework

### Problem Definition Components

Every ML project must have a clear problem definition with these components:

#### 1. **Title and Description**
```python
title = "Ad Content Classification"
description = """
Classify ad content into predefined categories to improve targeting accuracy.
The system should analyze text content and assign appropriate category labels
for better ad placement and audience targeting.
"""
```

#### 2. **Project Type and Complexity**
```python
from onyx.server.features.ads.project_initializer import ProjectType, ProblemComplexity

project_type = ProjectType.CLASSIFICATION  # or REGRESSION, GENERATION, etc.
complexity = ProblemComplexity.MODERATE     # or SIMPLE, COMPLEX, ADVANCED
```

#### 3. **Business Objective**
```python
business_objective = """
Improve ad targeting accuracy by 20% through automated content classification,
reducing manual review time by 60% and increasing click-through rates by 15%.
"""
```

#### 4. **Success Metrics**
```python
success_metrics = [
    "Classification accuracy > 85%",
    "F1-score > 0.8",
    "Processing time < 1 second per ad",
    "Model size < 100MB",
    "Reduction in manual review time by 60%"
]
```

#### 5. **Constraints and Assumptions**
```python
constraints = [
    "Must process ads in real-time",
    "Model size limited to 100MB",
    "Must work with existing infrastructure",
    "Compliance with data privacy regulations"
]

assumptions = [
    "Text data is relatively clean",
    "Categories are well-defined and stable",
    "Sufficient training data available",
    "Domain experts available for validation"
]
```

#### 6. **Risks and Stakeholders**
```python
risks = [
    "Data quality issues affecting model performance",
    "Model bias towards certain categories",
    "Performance degradation with new content types",
    "Regulatory changes affecting data usage"
]

stakeholders = [
    "Marketing team - End users",
    "Data science team - Model development",
    "Product team - Integration requirements",
    "Legal team - Compliance review"
]
```

## 🔍 Dataset Analysis Framework

### Comprehensive Dataset Analysis

The system automatically performs these analyses:

#### 1. **Basic Information**
- Dataset shape and size
- Column names and data types
- Memory usage
- File size

#### 2. **Data Quality**
- Duplicate detection
- Missing value analysis
- Data type validation
- Unique value counts

#### 3. **Statistical Analysis**
- Descriptive statistics
- Distribution analysis
- Correlation analysis
- Outlier detection

#### 4. **Recommendations**
- Data preprocessing suggestions
- Feature engineering ideas
- Model selection guidance
- Risk mitigation strategies

### Example Analysis Output

```python
{
    "basic_info": {
        "shape": (10000, 5),
        "columns": ["text", "category", "engagement_rate", "timestamp", "user_id"],
        "dtypes": {"text": "object", "category": "object", "engagement_rate": "float64"},
        "memory_usage": 2048576,
        "file_size": 1048576
    },
    "data_quality": {
        "duplicates": 0,
        "null_counts": {"text": 0, "category": 0, "engagement_rate": 150},
        "unique_counts": {"text": 9500, "category": 10, "engagement_rate": 9850},
        "issues": ["Missing values in engagement_rate column"]
    },
    "statistics": {
        "numeric_statistics": {
            "engagement_rate": {
                "mean": 0.045,
                "std": 0.023,
                "min": 0.001,
                "max": 0.156,
                "median": 0.042
            }
        }
    },
    "recommendations": [
        "Handle missing values in engagement_rate column",
        "Consider feature engineering for text length and complexity",
        "Investigate class imbalance in category distribution"
    ]
}
```

## 📁 Project Structure

### Generated Directory Structure

```
project_name/
├── data/                    # Dataset files
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned and processed data
│   ├── interim/           # Intermediate data files
│   └── external/          # External data sources
├── models/                 # Trained models
│   ├── checkpoints/       # Model checkpoints
│   ├── final/            # Final trained models
│   └── experiments/      # Experimental models
├── notebooks/             # Jupyter notebooks
│   ├── exploration/      # Data exploration notebooks
│   ├── experiments/      # Experiment notebooks
│   └── analysis/         # Analysis notebooks
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── models/          # Model definitions
│   ├── features/        # Feature engineering
│   ├── utils/           # Utility functions
│   └── api/             # API endpoints
├── tests/               # Test files
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── data/           # Data validation tests
├── configs/             # Configuration files
│   ├── project_config.yaml
│   └── config.py
├── logs/                # Log files
├── results/             # Results and outputs
│   ├── figures/        # Generated figures
│   ├── reports/        # Analysis reports
│   └── predictions/    # Model predictions
├── docs/               # Documentation
│   ├── problem_definition.md
│   ├── dataset_analysis.md
│   └── project_plan.md
├── scripts/            # Utility scripts
└── README.md          # Project overview
```

## ⚙️ Configuration Management

### Automated Configuration Generation

The system generates two types of configuration files:

#### 1. **YAML Configuration**
```yaml
project:
  name: "ad_classification"
  type: "classification"
  complexity: "moderate"
  created_at: "2024-01-15T10:30:00"
  version: "1.0.0"

problem_definition:
  title: "Ad Content Classification"
  description: "Classify ad content into categories..."
  business_objective: "Improve ad targeting accuracy by 20%"
  success_metrics:
    - "Accuracy > 85%"
    - "F1-score > 0.8"

dataset:
  name: "Ad Content Dataset"
  type: "text"
  path: "./data/ad_content.csv"
  target_column: "category"

training:
  validation_split: 0.2
  test_split: 0.1
  batch_size: 32
  learning_rate: 0.0001
  epochs: 100

optimization:
  enable_mixed_precision: true
  enable_gradient_checkpointing: true
  enable_model_compilation: true
  num_workers: 4
```

#### 2. **Python Configuration**
```python
@dataclass
class ProjectConfig:
    name: str = "ad_classification"
    type: str = "classification"
    complexity: str = "moderate"

class Settings(BaseSettings):
    project_name: str = "ad_classification"
    dataset_path: str = "./data/ad_content.csv"
    batch_size: int = 32
    learning_rate: float = 0.0001
    enable_mixed_precision: bool = True
```

## 📚 Documentation Generation

### Auto-Generated Documentation

The system creates comprehensive documentation:

#### 1. **README.md**
- Project overview
- Problem definition summary
- Dataset information
- Quick start guide
- Key findings and recommendations

#### 2. **Problem Definition Document**
- Detailed problem description
- Business objectives
- Success metrics
- Constraints and assumptions
- Risks and stakeholders
- Technical requirements

#### 3. **Dataset Analysis Document**
- Dataset information
- Analysis results
- Data quality assessment
- Statistical summaries
- Recommendations
- Preprocessing plan

#### 4. **Project Plan**
- Phase-by-phase execution plan
- Task breakdown
- Deliverables
- Success criteria
- Risk mitigation

## 🔧 Advanced Usage

### Custom Project Types

```python
# Custom project type
class CustomProjectType(ProjectType):
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES_FORECASTING = "time_series_forecasting"

# Use custom project type
problem_def = create_problem_definition(
    title="Ad Recommendation System",
    project_type=CustomProjectType.RECOMMENDATION,
    # ... other parameters
)
```

### Custom Dataset Analysis

```python
class CustomDatasetAnalyzer(DatasetAnalyzer):
    async def _analyze_custom_metrics(self) -> Dict[str, Any]:
        """Custom analysis for specific dataset type."""
        # Implement custom analysis logic
        return {"custom_metric": "value"}
    
    async def analyze_dataset(self) -> Dict[str, Any]:
        """Override with custom analysis."""
        analysis = await super().analyze_dataset()
        analysis['custom_analysis'] = await self._analyze_custom_metrics()
        return analysis
```

### Batch Project Creation

```python
async def create_multiple_projects(project_configs: List[Dict[str, Any]]):
    """Create multiple projects from configuration."""
    results = []
    
    for config in project_configs:
        problem_def = create_problem_definition(**config['problem'])
        dataset_info = create_dataset_info(**config['dataset'])
        
        result = await create_project(
            project_name=config['name'],
            problem_definition=problem_def,
            dataset_info=dataset_info
        )
        results.append(result)
    
    return results
```

## 🧪 Testing and Validation

### Project Validation

The system automatically validates:

#### 1. **Structure Validation**
- Required directories exist
- Configuration files present
- Documentation files created

#### 2. **Configuration Validation**
- Required configuration keys present
- Valid data types and values
- Consistent settings

#### 3. **Dataset Validation**
- Dataset file exists
- Dataset can be loaded
- Basic data integrity checks

#### 4. **Dependencies Validation**
- Required packages installed
- Version compatibility
- Import functionality

### Example Validation Output

```python
{
    "structure": {
        "status": "valid",
        "missing_directories": [],
        "message": "Project structure validation passed"
    },
    "configuration": {
        "status": "valid",
        "missing_keys": [],
        "message": "Configuration validation passed"
    },
    "dataset": {
        "status": "valid",
        "message": "Dataset loaded successfully: (10000, 5)",
        "shape": (10000, 5)
    },
    "dependencies": {
        "status": "valid",
        "missing_packages": [],
        "message": "Dependencies validation passed"
    },
    "overall_status": "valid"
}
```

## 🚀 Integration with Existing Systems

### Integration with Optimization Systems

```python
from onyx.server.features.ads.project_initializer import ProjectInitializer
from onyx.server.features.ads.torch_optimizer import TorchOptimizationConfig

# Initialize project with optimization settings
initializer = ProjectInitializer("optimized_project")

# Create optimization-aware configuration
optimization_config = TorchOptimizationConfig(
    enable_mixed_precision=True,
    enable_gradient_checkpointing=True,
    enable_model_compilation=True
)

# Initialize project with optimization
result = await initializer.initialize_project(
    problem_definition=problem_def,
    dataset_info=dataset_info
)
```

### Integration with Profiling Systems

```python
from onyx.server.features.ads.profiling_optimizer import ProfilingOptimizer

# Add profiling to project initialization
profiler = ProfilingOptimizer()

async def initialize_project_with_profiling(project_name: str, ...):
    # Initialize project
    result = await create_project(project_name, ...)
    
    # Add profiling configuration
    profiler_config = profiler.create_config()
    result['profiling_config'] = profiler_config
    
    return result
```

## 📊 Best Practices

### 1. **Problem Definition**
- Be specific about business objectives
- Define measurable success metrics
- Identify all constraints early
- Document assumptions clearly
- Assess risks comprehensively

### 2. **Dataset Analysis**
- Always analyze data quality first
- Check for data leakage
- Validate data distributions
- Assess class imbalance
- Document data lineage

### 3. **Project Structure**
- Follow consistent naming conventions
- Separate raw and processed data
- Version control everything
- Document data transformations
- Maintain reproducible workflows

### 4. **Configuration Management**
- Use environment-specific configs
- Validate configuration at startup
- Document all parameters
- Use type hints and validation
- Version configuration files

### 5. **Documentation**
- Keep documentation up-to-date
- Include examples and use cases
- Document decisions and rationale
- Provide troubleshooting guides
- Maintain change logs

## 🔍 Troubleshooting

### Common Issues

#### 1. **Dataset Loading Errors**
```python
# Check dataset path and format
dataset_info = create_dataset_info(
    name="My Dataset",
    type=DatasetType.TABULAR,  # Ensure correct type
    path="./data/my_dataset.csv",  # Verify path exists
    # ...
)
```

#### 2. **Configuration Validation Failures**
```python
# Ensure all required fields are provided
problem_def = create_problem_definition(
    title="My Project",  # Required
    description="Project description",  # Required
    project_type=ProjectType.CLASSIFICATION,  # Required
    complexity=ProblemComplexity.MODERATE,  # Required
    business_objective="Clear objective",  # Required
    # Optional fields can be empty lists
    success_metrics=[],
    constraints=[],
    # ...
)
```

#### 3. **Permission Errors**
```python
# Ensure write permissions for project directory
import os
os.makedirs("./projects", exist_ok=True)
os.chmod("./projects", 0o755)
```

## 📈 Performance Considerations

### Large Dataset Handling

```python
# For large datasets, use sampling for analysis
class LargeDatasetAnalyzer(DatasetAnalyzer):
    async def _analyze_basic_info(self) -> Dict[str, Any]:
        """Analyze large dataset with sampling."""
        if self.dataset_info.type == DatasetType.TABULAR:
            # Read sample for analysis
            df_sample = pd.read_csv(self.dataset_info.path, nrows=10000)
            return {
                'shape': df_sample.shape,
                'columns': list(df_sample.columns),
                'sample_size': len(df_sample),
                'estimated_total_size': len(df_sample) * 10  # Estimate
            }
```

### Memory Optimization

```python
# Use memory-efficient analysis for large datasets
async def analyze_large_dataset(dataset_info: DatasetInfo):
    """Memory-efficient dataset analysis."""
    analyzer = DatasetAnalyzer(dataset_info)
    
    # Process in chunks
    chunk_size = 10000
    analysis_results = {}
    
    for chunk in pd.read_csv(dataset_info.path, chunksize=chunk_size):
        # Process chunk
        chunk_analysis = await analyzer._analyze_chunk(chunk)
        analysis_results = merge_analysis_results(analysis_results, chunk_analysis)
    
    return analysis_results
```

This comprehensive project initialization system ensures that every ML project starts with clear problem definition and thorough dataset analysis, following established conventions and best practices. 