from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from abc import ABC, abstractmethod
import functools
from collections import defaultdict, Counter
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno
from scipy import stats
import warnings
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Problem Definition and Dataset Analysis
Comprehensive system for problem definition, dataset analysis, and data exploration for deep learning projects.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class ProblemDefinitionConfig:
    """Configuration for problem definition and dataset analysis."""
    # Analysis parameters
    enable_data_exploration: bool = True
    enable_statistical_analysis: bool = True
    enable_visualization: bool = True
    enable_feature_analysis: bool = True
    enable_data_quality_assessment: bool = True
    
    # Dataset parameters
    target_column: str = None
    feature_columns: List[str] = None
    categorical_columns: List[str] = None
    numerical_columns: List[str] = None
    datetime_columns: List[str] = None
    
    # Analysis thresholds
    missing_data_threshold: float = 0.5
    correlation_threshold: float = 0.8
    outlier_threshold: float = 3.0
    class_imbalance_threshold: float = 0.1
    
    # Output parameters
    enable_detailed_reports: bool = True
    enable_export_analysis: bool = True
    report_format: str = "json"  # json, html, pdf
    
    # Advanced parameters
    enable_automated_insights: bool = True
    enable_data_recommendations: bool = True
    enable_problem_formulation: bool = True


class ProblemDefinition:
    """Comprehensive problem definition and analysis system."""
    
    def __init__(self, config: ProblemDefinitionConfig):
        
    """__init__ function."""
self.config = config
        self.analysis_results = {}
        self.dataset_insights = {}
        self.problem_formulation = {}
        self.recommendations = []
        
    def define_problem(self, dataset: Union[pd.DataFrame, torch.utils.data.Dataset], 
                      problem_type: str = None) -> Dict[str, Any]:
        """Define the problem and analyze the dataset."""
        logger.info("Starting problem definition and dataset analysis")
        
        # Convert dataset to DataFrame if needed
        if isinstance(dataset, torch.utils.data.Dataset):
            df = self._dataset_to_dataframe(dataset)
        else:
            df = dataset.copy()
        
        # Basic dataset information
        basic_info = self._analyze_basic_info(df)
        self.analysis_results['basic_info'] = basic_info
        
        # Data quality assessment
        if self.config.enable_data_quality_assessment:
            quality_assessment = self._assess_data_quality(df)
            self.analysis_results['data_quality'] = quality_assessment
        
        # Statistical analysis
        if self.config.enable_statistical_analysis:
            statistical_analysis = self._perform_statistical_analysis(df)
            self.analysis_results['statistical_analysis'] = statistical_analysis
        
        # Feature analysis
        if self.config.enable_feature_analysis:
            feature_analysis = self._analyze_features(df)
            self.analysis_results['feature_analysis'] = feature_analysis
        
        # Data exploration
        if self.config.enable_data_exploration:
            exploration_results = self._explore_data(df)
            self.analysis_results['data_exploration'] = exploration_results
        
        # Problem formulation
        if self.config.enable_problem_formulation:
            problem_formulation = self._formulate_problem(df, problem_type)
            self.problem_formulation = problem_formulation
        
        # Generate insights and recommendations
        if self.config.enable_automated_insights:
            insights = self._generate_insights()
            self.dataset_insights = insights
        
        if self.config.enable_data_recommendations:
            recommendations = self._generate_recommendations()
            self.recommendations = recommendations
        
        return self.analysis_results
    
    def _dataset_to_dataframe(self, dataset: torch.utils.data.Dataset) -> pd.DataFrame:
        """Convert PyTorch dataset to pandas DataFrame."""
        data_list = []
        
        for i in range(min(len(dataset), 1000)):  # Sample first 1000 items
            try:
                item = dataset[i]
                if isinstance(item, (tuple, list)):
                    data_list.append(item[0].flatten().numpy())
                else:
                    data_list.append(item.flatten().numpy())
            except Exception as e:
                logger.warning(f"Error processing dataset item {i}: {e}")
                continue
        
        if data_list:
            return pd.DataFrame(data_list)
        else:
            return pd.DataFrame()
    
    def _analyze_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic dataset information."""
        basic_info = {
            'dataset_shape': df.shape,
            'total_samples': len(df),
            'total_features': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'data_types': df.dtypes.to_dict(),
            'columns': list(df.columns),
            'sample_data': df.head().to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Identify column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        basic_info['numerical_columns'] = numerical_cols
        basic_info['categorical_columns'] = categorical_cols
        basic_info['datetime_columns'] = datetime_cols
        
        return basic_info
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality issues."""
        quality_assessment = {
            'missing_data': {},
            'duplicates': {},
            'outliers': {},
            'data_consistency': {},
            'data_completeness': {}
        }
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        quality_assessment['missing_data'] = {
            'missing_counts': missing_data.to_dict(),
            'missing_percentages': missing_percentage.to_dict(),
            'total_missing': missing_data.sum(),
            'total_missing_percentage': (missing_data.sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Duplicate analysis
        duplicates = df.duplicated().sum()
        quality_assessment['duplicates'] = {
            'duplicate_count': duplicates,
            'duplicate_percentage': (duplicates / len(df)) * 100
        }
        
        # Outlier analysis for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        quality_assessment['outliers'] = outlier_info
        
        # Data consistency check
        consistency_issues = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_values = df[col].nunique()
                consistency_issues[col] = {
                    'unique_values': unique_values,
                    'cardinality': 'high' if unique_values > 50 else 'medium' if unique_values > 10 else 'low'
                }
        
        quality_assessment['data_consistency'] = consistency_issues
        
        return quality_assessment
    
    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        statistical_analysis = {
            'descriptive_statistics': {},
            'correlation_analysis': {},
            'distribution_analysis': {},
            'normality_tests': {},
            'skewness_kurtosis': {}
        }
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Descriptive statistics
        descriptive_stats = df.describe()
        statistical_analysis['descriptive_statistics'] = descriptive_stats.to_dict()
        
        # Correlation analysis
        if len(numerical_cols) > 1:
            correlation_matrix = df[numerical_cols].corr()
            statistical_analysis['correlation_analysis'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': self._find_high_correlations(correlation_matrix)
            }
        
        # Distribution analysis
        distribution_info = {}
        for col in numerical_cols:
            distribution_info[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'variance': df[col].var(),
                'min': df[col].min(),
                'max': df[col].max(),
                'range': df[col].max() - df[col].min(),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
            }
        
        statistical_analysis['distribution_analysis'] = distribution_info
        
        # Normality tests
        normality_tests = {}
        for col in numerical_cols:
            if len(df[col].dropna()) > 3:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(df[col].dropna())
                    normality_tests[col] = {
                        'shapiro_statistic': shapiro_stat,
                        'shapiro_p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                except Exception as e:
                    normality_tests[col] = {'error': str(e)}
        
        statistical_analysis['normality_tests'] = normality_tests
        
        # Skewness and kurtosis
        skewness_kurtosis = {}
        for col in numerical_cols:
            skewness_kurtosis[col] = {
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
        
        statistical_analysis['skewness_kurtosis'] = skewness_kurtosis
        
        return statistical_analysis
    
    def _find_high_correlations(self, correlation_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find highly correlated feature pairs."""
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > self.config.correlation_threshold:
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        return high_correlations
    
    def _analyze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual features."""
        feature_analysis = {
            'feature_importance': {},
            'feature_distributions': {},
            'feature_relationships': {},
            'feature_recommendations': {}
        }
        
        # Analyze each feature
        for col in df.columns:
            feature_info = {}
            
            if df[col].dtype in ['int64', 'float64']:
                feature_info = self._analyze_numerical_feature(df, col)
            elif df[col].dtype == 'object':
                feature_info = self._analyze_categorical_feature(df, col)
            
            feature_analysis['feature_distributions'][col] = feature_info
        
        # Feature importance (if target is specified)
        if self.config.target_column and self.config.target_column in df.columns:
            feature_importance = self._calculate_feature_importance(df)
            feature_analysis['feature_importance'] = feature_importance
        
        return feature_analysis
    
    def _analyze_numerical_feature(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Analyze a numerical feature."""
        feature_info = {
            'type': 'numerical',
            'unique_values': df[col].nunique(),
            'missing_values': df[col].isnull().sum(),
            'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'quartiles': df[col].quantile([0.25, 0.5, 0.75]).to_dict(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
        
        # Check for outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        
        feature_info['outliers'] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100
        }
        
        return feature_info
    
    def _analyze_categorical_feature(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Analyze a categorical feature."""
        value_counts = df[col].value_counts()
        
        feature_info = {
            'type': 'categorical',
            'unique_values': df[col].nunique(),
            'missing_values': df[col].isnull().sum(),
            'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_common': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_common_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'value_counts': value_counts.head(10).to_dict(),
            'cardinality': 'high' if df[col].nunique() > 50 else 'medium' if df[col].nunique() > 10 else 'low'
        }
        
        return feature_info
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance using correlation with target."""
        if self.config.target_column not in df.columns:
            return {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.config.target_column]
        
        feature_importance = {}
        for col in numerical_cols:
            correlation = df[col].corr(df[self.config.target_column])
            feature_importance[col] = abs(correlation)
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def _explore_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Explore data patterns and relationships."""
        exploration_results = {
            'data_patterns': {},
            'relationships': {},
            'anomalies': {},
            'trends': {}
        }
        
        # Data patterns
        patterns = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                patterns[col] = self._identify_numerical_patterns(df, col)
            elif df[col].dtype == 'object':
                patterns[col] = self._identify_categorical_patterns(df, col)
        
        exploration_results['data_patterns'] = patterns
        
        # Feature relationships
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            relationships = self._analyze_feature_relationships(df)
            exploration_results['relationships'] = relationships
        
        # Anomalies detection
        anomalies = self._detect_anomalies(df)
        exploration_results['anomalies'] = anomalies
        
        return exploration_results
    
    def _identify_numerical_patterns(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Identify patterns in numerical data."""
        patterns = {
            'distribution_type': 'unknown',
            'has_trends': False,
            'seasonality': False,
            'clusters': False
        }
        
        # Check distribution type
        skewness = df[col].skew()
        if abs(skewness) < 0.5:
            patterns['distribution_type'] = 'normal'
        elif skewness > 0.5:
            patterns['distribution_type'] = 'right_skewed'
        else:
            patterns['distribution_type'] = 'left_skewed'
        
        # Check for trends (simple linear trend)
        if len(df) > 10:
            x = np.arange(len(df))
            y = df[col].values
            slope = np.polyfit(x, y, 1)[0]
            patterns['has_trends'] = abs(slope) > 0.01
        
        return patterns
    
    def _identify_categorical_patterns(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Identify patterns in categorical data."""
        patterns = {
            'distribution_type': 'unknown',
            'dominant_categories': False,
            'balanced': False
        }
        
        value_counts = df[col].value_counts()
        total_count = len(df)
        
        # Check if there are dominant categories
        max_count = value_counts.iloc[0]
        if max_count / total_count > 0.5:
            patterns['dominant_categories'] = True
            patterns['distribution_type'] = 'imbalanced'
        else:
            patterns['balanced'] = True
            patterns['distribution_type'] = 'balanced'
        
        return patterns
    
    def _analyze_feature_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between features."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        relationships = {
            'correlations': {},
            'interactions': {}
        }
        
        # Calculate correlations
        correlation_matrix = df[numerical_cols].corr()
        relationships['correlations'] = correlation_matrix.to_dict()
        
        # Find strong interactions
        strong_interactions = []
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    strong_interactions.append({
                        'feature1': numerical_cols[i],
                        'feature2': numerical_cols[j],
                        'correlation': corr
                    })
        
        relationships['interactions'] = strong_interactions
        
        return relationships
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the dataset."""
        anomalies = {
            'outliers': {},
            'missing_patterns': {},
            'inconsistencies': {}
        }
        
        # Detect outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                anomalies['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'indices': outliers.index.tolist()
                }
        
        # Detect missing data patterns
        missing_matrix = df.isnull()
        if missing_matrix.any().any():
            anomalies['missing_patterns'] = {
                'total_missing': missing_matrix.sum().sum(),
                'missing_by_column': missing_matrix.sum().to_dict(),
                'missing_patterns': missing_matrix.sum(axis=1).value_counts().to_dict()
            }
        
        return anomalies
    
    def _formulate_problem(self, df: pd.DataFrame, problem_type: str = None) -> Dict[str, Any]:
        """Formulate the machine learning problem."""
        problem_formulation = {
            'problem_type': problem_type,
            'target_variable': self.config.target_column,
            'features': list(df.columns) if self.config.target_column is None else 
                       [col for col in df.columns if col != self.config.target_column],
            'data_characteristics': {},
            'problem_complexity': 'unknown',
            'recommended_approaches': []
        }
        
        # Determine problem type if not specified
        if problem_type is None and self.config.target_column:
            problem_type = self._determine_problem_type(df, self.config.target_column)
            problem_formulation['problem_type'] = problem_type
        
        # Analyze data characteristics
        if self.config.target_column and self.config.target_column in df.columns:
            characteristics = self._analyze_target_characteristics(df, self.config.target_column)
            problem_formulation['data_characteristics'] = characteristics
        
        # Assess problem complexity
        complexity = self._assess_problem_complexity(df, problem_formulation)
        problem_formulation['problem_complexity'] = complexity
        
        # Recommend approaches
        recommendations = self._recommend_approaches(problem_formulation)
        problem_formulation['recommended_approaches'] = recommendations
        
        return problem_formulation
    
    def _determine_problem_type(self, df: pd.DataFrame, target_col: str) -> str:
        """Determine the type of machine learning problem."""
        target_dtype = df[target_col].dtype
        
        if target_dtype in ['int64', 'float64']:
            # Check if it's classification or regression
            unique_values = df[target_col].nunique()
            total_samples = len(df)
            
            if unique_values <= 20 and unique_values / total_samples < 0.1:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'
    
    def _analyze_target_characteristics(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze characteristics of the target variable."""
        characteristics = {
            'data_type': str(df[target_col].dtype),
            'missing_values': df[target_col].isnull().sum(),
            'unique_values': df[target_col].nunique()
        }
        
        if df[target_col].dtype in ['int64', 'float64']:
            characteristics.update({
                'mean': df[target_col].mean(),
                'std': df[target_col].std(),
                'min': df[target_col].min(),
                'max': df[target_col].max(),
                'distribution': 'normal' if abs(df[target_col].skew()) < 1 else 'skewed'
            })
        else:
            value_counts = df[target_col].value_counts()
            characteristics.update({
                'class_distribution': value_counts.to_dict(),
                'class_imbalance': self._check_class_imbalance(value_counts),
                'most_common_class': value_counts.index[0] if len(value_counts) > 0 else None
            })
        
        return characteristics
    
    def _check_class_imbalance(self, value_counts: pd.Series) -> Dict[str, Any]:
        """Check for class imbalance in classification problems."""
        total_samples = value_counts.sum()
        class_ratios = value_counts / total_samples
        
        imbalance_info = {
            'is_imbalanced': False,
            'imbalance_ratio': 0.0,
            'minority_classes': [],
            'majority_classes': []
        }
        
        min_ratio = class_ratios.min()
        max_ratio = class_ratios.max()
        imbalance_ratio = min_ratio / max_ratio
        
        imbalance_info['imbalance_ratio'] = imbalance_ratio
        
        if imbalance_ratio < self.config.class_imbalance_threshold:
            imbalance_info['is_imbalanced'] = True
            imbalance_info['minority_classes'] = class_ratios[class_ratios < 0.1].index.tolist()
            imbalance_info['majority_classes'] = class_ratios[class_ratios > 0.5].index.tolist()
        
        return imbalance_info
    
    def _assess_problem_complexity(self, df: pd.DataFrame, problem_formulation: Dict[str, Any]) -> str:
        """Assess the complexity of the machine learning problem."""
        complexity_score = 0
        
        # Feature complexity
        num_features = len(problem_formulation['features'])
        if num_features < 10:
            complexity_score += 1
        elif num_features < 50:
            complexity_score += 2
        else:
            complexity_score += 3
        
        # Sample size complexity
        num_samples = len(df)
        if num_samples < 1000:
            complexity_score += 1
        elif num_samples < 10000:
            complexity_score += 2
        else:
            complexity_score += 3
        
        # Data quality complexity
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_percentage > 0.1:
            complexity_score += 2
        elif missing_percentage > 0.05:
            complexity_score += 1
        
        # Problem type complexity
        if problem_formulation['problem_type'] == 'classification':
            if problem_formulation['data_characteristics'].get('class_imbalance', {}).get('is_imbalanced', False):
                complexity_score += 2
        
        # Determine complexity level
        if complexity_score <= 3:
            return 'low'
        elif complexity_score <= 6:
            return 'medium'
        else:
            return 'high'
    
    def _recommend_approaches(self, problem_formulation: Dict[str, Any]) -> List[str]:
        """Recommend machine learning approaches based on problem characteristics."""
        recommendations = []
        
        problem_type = problem_formulation['problem_type']
        complexity = problem_formulation['problem_complexity']
        
        if problem_type == 'classification':
            if complexity == 'low':
                recommendations.extend(['Logistic Regression', 'Random Forest', 'SVM'])
            elif complexity == 'medium':
                recommendations.extend(['Random Forest', 'XGBoost', 'Neural Networks'])
            else:
                recommendations.extend(['Deep Learning', 'Ensemble Methods', 'Advanced ML'])
        
        elif problem_type == 'regression':
            if complexity == 'low':
                recommendations.extend(['Linear Regression', 'Random Forest', 'SVR'])
            elif complexity == 'medium':
                recommendations.extend(['Random Forest', 'XGBoost', 'Neural Networks'])
            else:
                recommendations.extend(['Deep Learning', 'Ensemble Methods', 'Advanced ML'])
        
        # Add general recommendations
        recommendations.extend(['Cross-validation', 'Hyperparameter tuning', 'Feature engineering'])
        
        return recommendations
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate automated insights from the analysis."""
        insights = {
            'key_findings': [],
            'data_quality_issues': [],
            'feature_insights': [],
            'recommendations': []
        }
        
        # Key findings
        if 'basic_info' in self.analysis_results:
            basic_info = self.analysis_results['basic_info']
            insights['key_findings'].append(f"Dataset contains {basic_info['total_samples']} samples with {basic_info['total_features']} features")
        
        # Data quality issues
        if 'data_quality' in self.analysis_results:
            quality = self.analysis_results['data_quality']
            
            if quality['missing_data']['total_missing_percentage'] > 10:
                insights['data_quality_issues'].append("High percentage of missing data detected")
            
            if quality['duplicates']['duplicate_percentage'] > 5:
                insights['data_quality_issues'].append("Significant number of duplicate records found")
        
        # Feature insights
        if 'feature_analysis' in self.analysis_results:
            feature_analysis = self.analysis_results['feature_analysis']
            if 'feature_importance' in feature_analysis:
                important_features = list(feature_analysis['feature_importance'].keys())[:5]
                insights['feature_insights'].append(f"Top important features: {', '.join(important_features)}")
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data quality recommendations
        if 'data_quality' in self.analysis_results:
            quality = self.analysis_results['data_quality']
            
            if quality['missing_data']['total_missing_percentage'] > 10:
                recommendations.append("Implement missing data imputation strategies")
            
            if quality['duplicates']['duplicate_percentage'] > 5:
                recommendations.append("Remove or handle duplicate records")
        
        # Feature engineering recommendations
        if 'feature_analysis' in self.analysis_results:
            feature_analysis = self.analysis_results['feature_analysis']
            if 'feature_importance' in feature_analysis:
                recommendations.append("Focus on feature engineering for important features")
        
        # Model recommendations
        if self.problem_formulation:
            recommendations.extend(self.problem_formulation.get('recommended_approaches', []))
        
        return recommendations


class DatasetAnalyzer:
    """Specialized dataset analysis and visualization."""
    
    def __init__(self, config: ProblemDefinitionConfig):
        
    """__init__ function."""
self.config = config
        
    def create_visualization_report(self, df: pd.DataFrame, save_path: str = None) -> Dict[str, Any]:
        """Create comprehensive visualization report."""
        if not self.config.enable_visualization:
            return {}
        
        report = {
            'missing_data_plot': self._plot_missing_data(df),
            'correlation_heatmap': self._plot_correlation_heatmap(df),
            'feature_distributions': self._plot_feature_distributions(df),
            'target_analysis': self._plot_target_analysis(df),
            'outlier_analysis': self._plot_outlier_analysis(df)
        }
        
        if save_path:
            self._save_visualizations(report, save_path)
        
        return report
    
    def _plot_missing_data(self, df: pd.DataFrame):
        """Plot missing data patterns."""
        try:
            plt.figure(figsize=(12, 6))
            msno.matrix(df)
            plt.title('Missing Data Pattern')
            return plt.gcf()
        except Exception as e:
            logger.warning(f"Error creating missing data plot: {e}")
            return None
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap."""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                correlation_matrix = df[numerical_cols].corr()
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Feature Correlation Heatmap')
                return plt.gcf()
        except Exception as e:
            logger.warning(f"Error creating correlation heatmap: {e}")
            return None
    
    def _plot_feature_distributions(self, df: pd.DataFrame):
        """Plot feature distributions."""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) > 0:
                n_cols = min(3, len(numerical_cols))
                n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
                
                for i, col in enumerate(numerical_cols):
                    if i < len(axes):
                        axes[i].hist(df[col].dropna(), bins=30, alpha=0.7)
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                
                # Hide empty subplots
                for i in range(len(numerical_cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                return fig
        except Exception as e:
            logger.warning(f"Error creating feature distributions plot: {e}")
            return None
    
    def _plot_target_analysis(self, df: pd.DataFrame):
        """Plot target variable analysis."""
        if not self.config.target_column or self.config.target_column not in df.columns:
            return None
        
        try:
            target_col = self.config.target_column
            
            if df[target_col].dtype in ['int64', 'float64']:
                # Regression target
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                axes[0].hist(df[target_col].dropna(), bins=30, alpha=0.7)
                axes[0].set_title(f'Distribution of {target_col}')
                axes[0].set_xlabel(target_col)
                axes[0].set_ylabel('Frequency')
                
                axes[1].boxplot(df[target_col].dropna())
                axes[1].set_title(f'Boxplot of {target_col}')
                axes[1].set_ylabel(target_col)
                
                plt.tight_layout()
                return fig
            else:
                # Classification target
                value_counts = df[target_col].value_counts()
                
                plt.figure(figsize=(10, 6))
                value_counts.plot(kind='bar')
                plt.title(f'Class Distribution of {target_col}')
                plt.xlabel('Classes')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                return plt.gcf()
        except Exception as e:
            logger.warning(f"Error creating target analysis plot: {e}")
            return None
    
    def _plot_outlier_analysis(self, df: pd.DataFrame):
        """Plot outlier analysis."""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) > 0:
                n_cols = min(3, len(numerical_cols))
                n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
                
                for i, col in enumerate(numerical_cols):
                    if i < len(axes):
                        axes[i].boxplot(df[col].dropna())
                        axes[i].set_title(f'Outlier Analysis - {col}')
                        axes[i].set_ylabel(col)
                
                # Hide empty subplots
                for i in range(len(numerical_cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                return fig
        except Exception as e:
            logger.warning(f"Error creating outlier analysis plot: {e}")
            return None
    
    def _save_visualizations(self, report: Dict[str, Any], save_path: str):
        """Save visualization plots."""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            for name, fig in report.items():
                if fig is not None:
                    fig.savefig(os.path.join(save_path, f"{name}.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig)
        except Exception as e:
            logger.warning(f"Error saving visualizations: {e}")


# Utility functions
def analyze_dataset(dataset: Union[pd.DataFrame, torch.utils.data.Dataset], 
                   config: ProblemDefinitionConfig = None) -> Dict[str, Any]:
    """Analyze a dataset comprehensively."""
    if config is None:
        config = ProblemDefinitionConfig()
    
    analyzer = ProblemDefinition(config)
    return analyzer.define_problem(dataset)


def create_dataset_report(dataset: Union[pd.DataFrame, torch.utils.data.Dataset], 
                         config: ProblemDefinitionConfig = None,
                         save_path: str = None) -> Dict[str, Any]:
    """Create a comprehensive dataset analysis report."""
    if config is None:
        config = ProblemDefinitionConfig()
    
    # Perform analysis
    analyzer = ProblemDefinition(config)
    analysis_results = analyzer.define_problem(dataset)
    
    # Create visualizations
    if isinstance(dataset, pd.DataFrame):
        df = dataset
    else:
        df = analyzer._dataset_to_dataframe(dataset)
    
    visualizer = DatasetAnalyzer(config)
    visualization_report = visualizer.create_visualization_report(df, save_path)
    
    # Combine results
    complete_report = {
        'analysis_results': analysis_results,
        'visualization_report': visualization_report,
        'insights': analyzer.dataset_insights,
        'recommendations': analyzer.recommendations,
        'problem_formulation': analyzer.problem_formulation,
        'timestamp': datetime.now().isoformat()
    }
    
    return complete_report


def export_analysis_report(report: Dict[str, Any], filepath: str, format: str = "json"):
    """Export analysis report to file."""
    try:
        if format == "json":
            with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(report, f, indent=2, default=str)
        elif format == "html":
            # Convert to HTML format
            html_content = f"""
            <html>
            <head><title>Dataset Analysis Report</title></head>
            <body>
                <h1>Dataset Analysis Report</h1>
                <p>Generated on: {report.get('timestamp', 'Unknown')}</p>
                <h2>Key Insights</h2>
                <ul>
                    {''.join([f'<li>{insight}</li>' for insight in report.get('insights', {}).get('key_findings', [])])}
                </ul>
                <h2>Recommendations</h2>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in report.get('recommendations', [])])}
                </ul>
            </body>
            </html>
            """
            with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(html_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Analysis report exported to {filepath}")
    except Exception as e:
        logger.error(f"Error exporting report: {e}")


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = ProblemDefinitionConfig(
        enable_data_exploration=True,
        enable_statistical_analysis=True,
        enable_visualization=True,
        enable_feature_analysis=True,
        enable_data_quality_assessment=True,
        target_column='target',  # Specify your target column
        enable_automated_insights=True,
        enable_data_recommendations=True,
        enable_problem_formulation=True
    )
    
    # Example dataset
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    }
    df = pd.DataFrame(data)
    
    # Analyze dataset
    report = create_dataset_report(df, config, save_path="./analysis_plots")
    
    # Export report
    export_analysis_report(report, "dataset_analysis_report.json", "json")
    
    print("Dataset analysis completed!")
    print(f"Found {len(report['insights']['key_findings'])} key insights")
    print(f"Generated {len(report['recommendations'])} recommendations") 