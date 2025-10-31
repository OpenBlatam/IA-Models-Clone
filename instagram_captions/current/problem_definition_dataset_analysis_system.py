"""
Problem Definition and Dataset Analysis System
Follows key convention: Begin projects with clear problem definition and dataset analysis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
from collections import Counter, defaultdict
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProblemDefinition:
    """Clear problem definition for NLP projects"""
    task_name: str
    task_type: str  # classification, generation, translation, etc.
    input_format: str
    output_format: str
    performance_metrics: List[str]
    constraints: List[str]
    success_criteria: Dict[str, Any]
    business_objectives: List[str]
    technical_requirements: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_name': self.task_name,
            'task_type': self.task_type,
            'input_format': self.input_format,
            'output_format': self.output_format,
            'performance_metrics': self.performance_metrics,
            'constraints': self.constraints,
            'success_criteria': self.success_criteria,
            'business_objectives': self.business_objectives,
            'technical_requirements': self.technical_requirements
        }
    
    def save_to_file(self, filepath: str):
        """Save problem definition to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Problem definition saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProblemDefinition':
        """Load problem definition from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

@dataclass
class DatasetMetadata:
    """Metadata about the dataset"""
    name: str
    source: str
    size: int
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    features: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    text_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'source': self.source,
            'size': self.size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'features': self.features,
            'target_column': self.target_column,
            'text_columns': self.text_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }

class DatasetAnalyzer:
    """Comprehensive dataset analysis system"""
    
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.dataset = None
        self.metadata = None
        self.analysis_results = {}
        
    def load_dataset(self, dataset_path: str, file_type: str = 'auto') -> Any:
        """Load dataset from various file formats"""
        try:
            if file_type == 'auto':
                file_type = Path(dataset_path).suffix.lower()
            
            if file_type in ['.csv', '.tsv']:
                self.dataset = pd.read_csv(dataset_path)
            elif file_type == '.json':
                self.dataset = pd.read_json(dataset_path)
            elif file_type == '.parquet':
                self.dataset = pd.read_parquet(dataset_path)
            elif file_type == '.pkl':
                self.dataset = pd.read_pickle(dataset_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.dataset_path = dataset_path
            logger.info(f"Dataset loaded successfully: {self.dataset.shape}")
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def analyze_dataset(self, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive dataset analysis"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        logger.info("Starting comprehensive dataset analysis...")
        
        # Basic information
        self.analysis_results['basic_info'] = self._analyze_basic_info()
        
        # Data quality analysis
        self.analysis_results['data_quality'] = self._analyze_data_quality()
        
        # Text analysis (if applicable)
        if self._has_text_columns():
            self.analysis_results['text_analysis'] = self._analyze_text_data()
        
        # Target analysis
        if target_column and target_column in self.dataset.columns:
            self.analysis_results['target_analysis'] = self._analyze_target(target_column)
        
        # Statistical analysis
        self.analysis_results['statistical_analysis'] = self._analyze_statistics()
        
        # Missing data analysis
        self.analysis_results['missing_data'] = self._analyze_missing_data()
        
        # Data distribution analysis
        self.analysis_results['distributions'] = self._analyze_distributions()
        
        logger.info("Dataset analysis completed successfully")
        return self.analysis_results
    
    def _analyze_basic_info(self) -> Dict[str, Any]:
        """Analyze basic dataset information"""
        return {
            'shape': self.dataset.shape,
            'columns': list(self.dataset.columns),
            'dtypes': self.dataset.dtypes.to_dict(),
            'memory_usage': self.dataset.memory_usage(deep=True).sum(),
            'sample_data': self.dataset.head().to_dict()
        }
    
    def _analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality issues"""
        quality_issues = {}
        
        # Duplicate rows
        quality_issues['duplicate_rows'] = self.dataset.duplicated().sum()
        
        # Column data types
        quality_issues['data_types'] = {
            'object': len(self.dataset.select_dtypes(include=['object']).columns),
            'numeric': len(self.dataset.select_dtypes(include=['number']).columns),
            'datetime': len(self.dataset.select_dtypes(include=['datetime']).columns),
            'categorical': len(self.dataset.select_dtypes(include=['category']).columns)
        }
        
        # Unique values per column
        quality_issues['unique_values'] = {
            col: self.dataset[col].nunique() 
            for col in self.dataset.columns
        }
        
        return quality_issues
    
    def _has_text_columns(self) -> bool:
        """Check if dataset has text columns"""
        text_columns = self.dataset.select_dtypes(include=['object']).columns
        return len(text_columns) > 0
    
    def _analyze_text_data(self) -> Dict[str, Any]:
        """Analyze text data characteristics"""
        text_columns = self.dataset.select_dtypes(include=['object']).columns
        text_analysis = {}
        
        for col in text_columns:
            col_data = self.dataset[col].dropna()
            if len(col_data) > 0:
                text_analysis[col] = {
                    'total_texts': len(col_data),
                    'avg_length': col_data.str.len().mean(),
                    'min_length': col_data.str.len().min(),
                    'max_length': col_data.str.len().max(),
                    'length_distribution': col_data.str.len().describe().to_dict(),
                    'unique_texts': col_data.nunique(),
                    'sample_texts': col_data.head(3).tolist()
                }
        
        return text_analysis
    
    def _analyze_target(self, target_column: str) -> Dict[str, Any]:
        """Analyze target variable"""
        target_data = self.dataset[target_column].dropna()
        target_analysis = {}
        
        if target_data.dtype in ['object', 'category']:
            # Categorical target
            value_counts = target_data.value_counts()
            target_analysis = {
                'type': 'categorical',
                'num_classes': len(value_counts),
                'class_distribution': value_counts.to_dict(),
                'class_imbalance': self._calculate_class_imbalance(value_counts),
                'most_common': value_counts.index[0],
                'least_common': value_counts.index[-1]
            }
        else:
            # Numerical target
            target_analysis = {
                'type': 'numerical',
                'min': target_data.min(),
                'max': target_data.max(),
                'mean': target_data.mean(),
                'median': target_data.median(),
                'std': target_data.std(),
                'distribution': target_data.describe().to_dict()
            }
        
        return target_analysis
    
    def _calculate_class_imbalance(self, value_counts: pd.Series) -> float:
        """Calculate class imbalance ratio"""
        max_count = value_counts.max()
        min_count = value_counts.min()
        return max_count / min_count if min_count > 0 else float('inf')
    
    def _analyze_statistics(self) -> Dict[str, Any]:
        """Analyze statistical properties"""
        numeric_columns = self.dataset.select_dtypes(include=['number']).columns
        
        if len(numeric_columns) == 0:
            return {}
        
        stats = {}
        for col in numeric_columns:
            col_data = self.dataset[col].dropna()
            if len(col_data) > 0:
                stats[col] = {
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    '25%': col_data.quantile(0.25),
                    '50%': col_data.quantile(0.50),
                    '75%': col_data.quantile(0.75),
                    'max': col_data.max(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                }
        
        return stats
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_data = {}
        
        # Missing values per column
        missing_counts = self.dataset.isnull().sum()
        missing_percentages = (missing_counts / len(self.dataset)) * 100
        
        missing_data['missing_counts'] = missing_counts.to_dict()
        missing_data['missing_percentages'] = missing_percentages.to_dict()
        missing_data['total_missing'] = missing_counts.sum()
        missing_data['total_missing_percentage'] = (missing_counts.sum() / (len(self.dataset) * len(self.dataset.columns))) * 100
        
        # Columns with missing data
        columns_with_missing = missing_counts[missing_counts > 0]
        missing_data['columns_with_missing'] = columns_with_missing.index.tolist()
        
        return missing_data
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze data distributions"""
        distributions = {}
        
        # Numerical columns
        numeric_columns = self.dataset.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            col_data = self.dataset[col].dropna()
            if len(col_data) > 0:
                distributions[col] = {
                    'histogram': np.histogram(col_data, bins=20),
                    'percentiles': col_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
                }
        
        # Categorical columns
        categorical_columns = self.dataset.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            col_data = self.dataset[col].dropna()
            if len(col_data) > 0:
                value_counts = col_data.value_counts()
                distributions[col] = {
                    'value_counts': value_counts.head(10).to_dict(),
                    'top_categories': value_counts.head(5).index.tolist()
                }
        
        return distributions
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive analysis report"""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_dataset() first.")
        
        report = []
        report.append("=" * 80)
        report.append("DATASET ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic Information
        report.append("1. BASIC INFORMATION")
        report.append("-" * 40)
        basic_info = self.analysis_results['basic_info']
        report.append(f"Dataset Shape: {basic_info['shape']}")
        report.append(f"Total Columns: {len(basic_info['columns'])}")
        report.append(f"Memory Usage: {basic_info['memory_usage'] / 1024 / 1024:.2f} MB")
        report.append("")
        
        # Data Quality
        report.append("2. DATA QUALITY")
        report.append("-" * 40)
        quality = self.analysis_results['data_quality']
        report.append(f"Duplicate Rows: {quality['duplicate_rows']}")
        report.append(f"Data Types: {quality['data_types']}")
        report.append("")
        
        # Missing Data
        report.append("3. MISSING DATA ANALYSIS")
        report.append("-" * 40)
        missing = self.analysis_results['missing_data']
        report.append(f"Total Missing Values: {missing['total_missing']}")
        report.append(f"Missing Data Percentage: {missing['total_missing_percentage']:.2f}%")
        if missing['columns_with_missing']:
            report.append(f"Columns with Missing Data: {', '.join(missing['columns_with_missing'])}")
        report.append("")
        
        # Text Analysis
        if 'text_analysis' in self.analysis_results:
            report.append("4. TEXT DATA ANALYSIS")
            report.append("-" * 40)
            text_analysis = self.analysis_results['text_analysis']
            for col, analysis in text_analysis.items():
                report.append(f"Column: {col}")
                report.append(f"  Total Texts: {analysis['total_texts']}")
                report.append(f"  Average Length: {analysis['avg_length']:.2f}")
                report.append(f"  Unique Texts: {analysis['unique_texts']}")
                report.append("")
        
        # Target Analysis
        if 'target_analysis' in self.analysis_results:
            report.append("5. TARGET VARIABLE ANALYSIS")
            report.append("-" * 40)
            target_analysis = self.analysis_results['target_analysis']
            report.append(f"Target Type: {target_analysis['type']}")
            if target_analysis['type'] == 'categorical':
                report.append(f"Number of Classes: {target_analysis['num_classes']}")
                report.append(f"Class Imbalance Ratio: {target_analysis['class_imbalance']:.2f}")
            report.append("")
        
        # Recommendations
        report.append("6. RECOMMENDATIONS")
        report.append("-" * 40)
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data-driven recommendations"""
        recommendations = []
        
        # Missing data recommendations
        missing = self.analysis_results['missing_data']
        if missing['total_missing_percentage'] > 20:
            recommendations.append("High percentage of missing data detected. Consider data imputation strategies or investigate data collection issues.")
        
        # Class imbalance recommendations
        if 'target_analysis' in self.analysis_results:
            target = self.analysis_results['target_analysis']
            if target['type'] == 'categorical' and target['class_imbalance'] > 10:
                recommendations.append("Severe class imbalance detected. Consider techniques like oversampling, undersampling, or weighted loss functions.")
        
        # Data quality recommendations
        quality = self.analysis_results['data_quality']
        if quality['duplicate_rows'] > 0:
            recommendations.append("Duplicate rows found. Consider removing duplicates to improve data quality.")
        
        # Text data recommendations
        if 'text_analysis' in self.analysis_results:
            text_analysis = self.analysis_results['text_analysis']
            for col, analysis in text_analysis.items():
                if analysis['avg_length'] > 1000:
                    recommendations.append(f"Long text sequences in column '{col}'. Consider truncation or chunking strategies.")
                if analysis['unique_texts'] / analysis['total_texts'] < 0.1:
                    recommendations.append(f"Low text diversity in column '{col}'. Consider data augmentation techniques.")
        
        if not recommendations:
            recommendations.append("Dataset appears to be well-structured. Proceed with standard preprocessing steps.")
        
        return recommendations
    
    def create_visualizations(self, output_dir: str):
        """Create visualization plots for the analysis"""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_dataset() first.")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Missing data heatmap
        if 'missing_data' in self.analysis_results:
            self._plot_missing_data_heatmap(output_dir)
        
        # Target distribution
        if 'target_analysis' in self.analysis_results:
            self._plot_target_distribution(output_dir)
        
        # Text length distributions
        if 'text_analysis' in self.analysis_results:
            self._plot_text_length_distributions(output_dir)
        
        # Numerical feature distributions
        if 'statistical_analysis' in self.analysis_results:
            self._plot_numerical_distributions(output_dir)
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _plot_missing_data_heatmap(self, output_dir: str):
        """Plot missing data heatmap"""
        missing_data = self.dataset.isnull()
        plt.figure(figsize=(12, 8))
        sns.heatmap(missing_data, cbar=True, yticklabels=False)
        plt.title('Missing Data Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/missing_data_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_target_distribution(self, output_dir: str):
        """Plot target variable distribution"""
        target_analysis = self.analysis_results['target_analysis']
        
        if target_analysis['type'] == 'categorical':
            plt.figure(figsize=(10, 6))
            target_counts = pd.Series(target_analysis['class_distribution'])
            target_counts.plot(kind='bar')
            plt.title('Target Variable Distribution')
            plt.xlabel('Classes')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/target_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.figure(figsize=(10, 6))
            target_col = [col for col in self.dataset.columns if col in self.analysis_results.get('statistical_analysis', {})][0]
            self.dataset[target_col].hist(bins=30)
            plt.title(f'Target Variable Distribution: {target_col}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/target_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_text_length_distributions(self, output_dir: str):
        """Plot text length distributions"""
        text_analysis = self.analysis_results['text_analysis']
        
        if len(text_analysis) > 1:
            fig, axes = plt.subplots(1, len(text_analysis), figsize=(5*len(text_analysis), 5))
            if len(text_analysis) == 1:
                axes = [axes]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        
        for i, (col, analysis) in enumerate(text_analysis.items()):
            col_data = self.dataset[col].dropna()
            lengths = col_data.str.len()
            axes[i].hist(lengths, bins=30, alpha=0.7)
            axes[i].set_title(f'Text Length Distribution: {col}')
            axes[i].set_xlabel('Length')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/text_length_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_numerical_distributions(self, output_dir: str):
        """Plot numerical feature distributions"""
        statistical_analysis = self.analysis_results['statistical_analysis']
        
        if not statistical_analysis:
            return
        
        num_features = len(statistical_analysis)
        cols = min(3, num_features)
        rows = (num_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (col, analysis) in enumerate(statistical_analysis.items()):
            row = i // cols
            col_idx = i % cols
            ax = axes[row, col_idx] if rows > 1 else axes[col_idx]
            
            col_data = self.dataset[col].dropna()
            ax.hist(col_data, bins=30, alpha=0.7)
            ax.set_title(f'Distribution: {col}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(num_features, rows * cols):
            row = i // cols
            col_idx = i % cols
            ax = axes[row, col_idx] if rows > 1 else axes[col_idx]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/numerical_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

class ProjectInitializer:
    """Initialize NLP projects with problem definition and dataset analysis"""
    
    def __init__(self, project_name: str, project_dir: str):
        self.project_name = project_name
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project structure
        self._create_project_structure()
        
        # Initialize logging
        self._setup_logging()
        
        logger.info(f"Project '{project_name}' initialized in {project_dir}")
    
    def _create_project_structure(self):
        """Create standard project directory structure"""
        directories = [
            'data',
            'models',
            'training',
            'evaluation',
            'utils',
            'configs',
            'experiments',
            'logs',
            'reports',
            'visualizations'
        ]
        
        for dir_name in directories:
            (self.project_dir / dir_name).mkdir(exist_ok=True)
        
        # Create __init__.py files
        for dir_name in ['models', 'training', 'evaluation', 'utils']:
            init_file = self.project_dir / dir_name / '__init__.py'
            if not init_file.exists():
                init_file.touch()
    
    def _setup_logging(self):
        """Setup project logging"""
        log_file = self.project_dir / 'logs' / 'project.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def create_problem_definition(self, **kwargs) -> ProblemDefinition:
        """Create problem definition for the project"""
        problem_def = ProblemDefinition(**kwargs)
        
        # Save to project directory
        problem_file = self.project_dir / 'configs' / 'problem_definition.json'
        problem_def.save_to_file(str(problem_file))
        
        return problem_def
    
    def analyze_dataset(self, dataset_path: str, target_column: Optional[str] = None) -> DatasetAnalyzer:
        """Analyze dataset for the project"""
        analyzer = DatasetAnalyzer()
        analyzer.load_dataset(dataset_path)
        analyzer.analyze_dataset(target_column)
        
        # Generate and save report
        report_file = self.project_dir / 'reports' / 'dataset_analysis_report.txt'
        analyzer.generate_report(str(report_file))
        
        # Create visualizations
        viz_dir = self.project_dir / 'visualizations'
        analyzer.create_visualizations(str(viz_dir))
        
        # Save analysis results
        results_file = self.project_dir / 'reports' / 'dataset_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(analyzer.analysis_results, f, indent=2, default=str)
        
        return analyzer
    
    def create_project_summary(self) -> str:
        """Create project summary document"""
        summary = []
        summary.append(f"# {self.project_name} - Project Summary")
        summary.append("")
        summary.append("## Project Structure")
        summary.append("```")
        summary.append(self._get_directory_tree())
        summary.append("```")
        summary.append("")
        summary.append("## Next Steps")
        summary.append("1. Review problem definition in `configs/problem_definition.json`")
        summary.append("2. Review dataset analysis in `reports/dataset_analysis_report.txt`")
        summary.append("3. Review visualizations in `visualizations/` directory")
        summary.append("4. Begin model development in `models/` directory")
        summary.append("5. Set up training pipeline in `training/` directory")
        
        summary_text = "\n".join(summary)
        
        # Save summary
        summary_file = self.project_dir / 'README.md'
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        return summary_text
    
    def _get_directory_tree(self) -> str:
        """Get directory tree structure"""
        tree = []
        
        def add_to_tree(path: Path, prefix: str = ""):
            if path.is_file():
                tree.append(f"{prefix}├── {path.name}")
            elif path.is_dir():
                tree.append(f"{prefix}├── {path.name}/")
                children = sorted(path.iterdir())
                for i, child in enumerate(children):
                    if i == len(children) - 1:
                        add_to_tree(child, prefix + "    ")
                    else:
                        add_to_tree(child, prefix + "│   ")
        
        add_to_tree(self.project_dir)
        return "\n".join(tree)

def main():
    """Example usage of the problem definition and dataset analysis system"""
    
    # Initialize project
    project_name = "instagram_captions_nlp"
    project_dir = "./instagram_captions_project"
    
    initializer = ProjectInitializer(project_name, project_dir)
    
    # Create problem definition
    problem_def = initializer.create_problem_definition(
        task_name="Instagram Caption Generation",
        task_type="text_generation",
        input_format="image_description",
        output_format="engaging_caption",
        performance_metrics=["BLEU", "ROUGE", "Perplexity", "Human_Evaluation"],
        constraints=["max_length_100", "response_time_2s", "memory_usage_4GB"],
        success_criteria={
            "BLEU_score": 0.7,
            "ROUGE_score": 0.6,
            "human_rating": 4.0
        },
        business_objectives=[
            "Increase user engagement",
            "Improve caption quality",
            "Reduce manual caption writing"
        ],
        technical_requirements=[
            "Real-time inference",
            "Multi-language support",
            "Scalable architecture"
        ]
    )
    
    print("✅ Problem definition created successfully!")
    print(f"📁 Project initialized in: {project_dir}")
    print(f"📋 Problem definition saved to: {project_dir}/configs/problem_definition.json")
    
    # Example dataset analysis (if dataset is available)
    # dataset_path = "path/to/your/dataset.csv"
    # if Path(dataset_path).exists():
    #     analyzer = initializer.analyze_dataset(dataset_path, target_column="caption")
    #     print("✅ Dataset analysis completed!")
    #     print(f"📊 Report saved to: {project_dir}/reports/dataset_analysis_report.txt")
    #     print(f"📈 Visualizations saved to: {project_dir}/visualizations/")
    
    # Create project summary
    summary = initializer.create_project_summary()
    print("✅ Project summary created!")
    print(f"📖 README saved to: {project_dir}/README.md")
    
    print("\n🎯 Project initialization complete!")
    print("Next steps:")
    print("1. Review the problem definition")
    print("2. Analyze your dataset")
    print("3. Begin model development")

if __name__ == "__main__":
    main()


