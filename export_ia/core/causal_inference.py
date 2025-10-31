"""
Causal Inference Engine for Export IA
Advanced causal inference with causal discovery, effect estimation, and counterfactual reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import json
import random
from pathlib import Path
from collections import defaultdict, deque
import copy
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests
import pgmpy
from pgmpy.models import BayesianNetwork, MarkovNetwork
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import PC, HillClimbSearch, BicScore
import causalml
from causalml.inference.meta import LRSRegressor, XGBTRegressor, MLPTRegressor
from causalml.inference.meta import LRSClassifier, XGBTClassifier, MLPTClassifier
from causalml.dataset import synthetic_data
from causalml.propensity import ElasticNetPropensityModel
from causalml.match import NearestNeighborMatch, PropensityScoreMatch
import dowhy
from dowhy import CausalModel
from dowhy.causal_estimators import PropensityScoreEstimator, InstrumentalVariableEstimator
from dowhy.causal_estimators import RegressionDiscontinuityEstimator, DifferenceInDifferencesEstimator
import econml
from econml.dml import DML, DMLCateEstimator
from econml.dr import DRLearner
from econml.metalearners import TLearner, SLearner, XLearner
from econml.sklearn_extensions.linear_model import WeightedLasso, WeightedElasticNet
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CausalInferenceConfig:
    """Configuration for causal inference"""
    # Causal discovery methods
    discovery_method: str = "pc"  # pc, gs, ges, lingam, custom
    
    # PC algorithm parameters
    pc_alpha: float = 0.05
    pc_ci_test: str = "pearsonr"  # pearsonr, spearmanr, kendalltau, gsq, chi_square
    pc_stable: bool = True
    pc_uc_rule: int = 0  # 0: original, 1: conservative, 2: majority
    
    # GES algorithm parameters
    ges_score: str = "bic"  # bic, aic, bdeu, k2
    ges_max_indegree: int = 4
    ges_tabu_length: int = 100
    
    # LiNGAM parameters
    lingam_algorithm: str = "ica"  # ica, direct
    lingam_random_state: int = 42
    
    # Causal effect estimation
    effect_estimation_method: str = "propensity_score"  # propensity_score, instrumental_variable, regression_discontinuity, difference_in_differences
    
    # Propensity score parameters
    ps_method: str = "logistic"  # logistic, random_forest, neural_network
    ps_calibration: bool = True
    ps_matching_method: str = "nearest_neighbor"  # nearest_neighbor, caliper, optimal
    
    # Instrumental variable parameters
    iv_method: str = "two_stage_least_squares"  # two_stage_least_squares, limited_information_maximum_likelihood
    iv_robust: bool = True
    
    # Regression discontinuity parameters
    rd_method: str = "local_linear"  # local_linear, local_quadratic, global_polynomial
    rd_bandwidth: str = "imse"  # imse, mse, rule_of_thumb
    
    # Difference in differences parameters
    did_method: str = "two_way_fixed_effects"  # two_way_fixed_effects, synthetic_control
    did_cluster: bool = True
    
    # Counterfactual reasoning
    counterfactual_method: str = "neural_network"  # neural_network, random_forest, linear_model
    counterfactual_samples: int = 1000
    counterfactual_confidence: float = 0.95
    
    # Neural network parameters
    nn_hidden_layers: List[int] = None  # [64, 32, 16]
    nn_dropout: float = 0.2
    nn_activation: str = "relu"  # relu, tanh, sigmoid
    nn_optimizer: str = "adam"  # adam, sgd, rmsprop
    nn_learning_rate: float = 0.001
    nn_epochs: int = 100
    nn_batch_size: int = 32
    
    # Evaluation parameters
    evaluation_metrics: List[str] = None  # ate, att, atc, cate, policy_value
    evaluation_cv_folds: int = 5
    evaluation_bootstrap_samples: int = 1000
    
    # Data preprocessing
    enable_preprocessing: bool = True
    handle_missing_values: str = "drop"  # drop, impute, forward_fill
    feature_selection: bool = True
    feature_scaling: bool = True
    
    # Performance parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    enable_caching: bool = True

class CausalDiscovery:
    """Causal discovery algorithms"""
    
    def __init__(self, config: CausalInferenceConfig):
        self.config = config
        
    def discover_causal_graph(self, data: pd.DataFrame) -> nx.DiGraph:
        """Discover causal graph from data"""
        
        if self.config.discovery_method == "pc":
            return self._pc_algorithm(data)
        elif self.config.discovery_method == "ges":
            return self._ges_algorithm(data)
        elif self.config.discovery_method == "lingam":
            return self._lingam_algorithm(data)
        else:
            raise ValueError(f"Unsupported discovery method: {self.config.discovery_method}")
            
    def _pc_algorithm(self, data: pd.DataFrame) -> nx.DiGraph:
        """PC algorithm for causal discovery"""
        
        # Create PC estimator
        pc_estimator = PC(data)
        
        # Run PC algorithm
        estimated_cpdag = pc_estimator.estimate(
            variant=self.config.pc_stable,
            ci_test=self.config.pc_ci_test,
            max_cond_vars=len(data.columns),
            significance_level=self.config.pc_alpha
        )
        
        # Convert to NetworkX graph
        graph = nx.DiGraph()
        
        # Add nodes
        for node in data.columns:
            graph.add_node(node)
            
        # Add edges
        for edge in estimated_cpdag.edges():
            graph.add_edge(edge[0], edge[1])
            
        return graph
        
    def _ges_algorithm(self, data: pd.DataFrame) -> nx.DiGraph:
        """GES algorithm for causal discovery"""
        
        # Create GES estimator
        ges_estimator = HillClimbSearch(data)
        
        # Run GES algorithm
        estimated_model = ges_estimator.estimate(
            scoring_method=BicScore(data),
            max_indegree=self.config.ges_max_indegree,
            tabu_length=self.config.ges_tabu_length
        )
        
        # Convert to NetworkX graph
        graph = nx.DiGraph()
        
        # Add nodes
        for node in data.columns:
            graph.add_node(node)
            
        # Add edges
        for edge in estimated_model.edges():
            graph.add_edge(edge[0], edge[1])
            
        return graph
        
    def _lingam_algorithm(self, data: pd.DataFrame) -> nx.DiGraph:
        """LiNGAM algorithm for causal discovery"""
        
        # Simplified LiNGAM implementation
        # In practice, you'd use the actual LiNGAM library
        
        # Create correlation matrix
        corr_matrix = data.corr().values
        
        # Find causal ordering using ICA-like approach
        causal_order = self._find_causal_order(corr_matrix)
        
        # Create graph
        graph = nx.DiGraph()
        
        # Add nodes
        for node in data.columns:
            graph.add_node(node)
            
        # Add edges based on causal order
        for i in range(len(causal_order)):
            for j in range(i + 1, len(causal_order)):
                if abs(corr_matrix[causal_order[i], causal_order[j]]) > 0.1:
                    graph.add_edge(
                        data.columns[causal_order[i]], 
                        data.columns[causal_order[j]]
                    )
                    
        return graph
        
    def _find_causal_order(self, corr_matrix: np.ndarray) -> List[int]:
        """Find causal ordering from correlation matrix"""
        
        # Simple heuristic: order by variance
        variances = np.var(corr_matrix, axis=1)
        return np.argsort(variances)

class CausalEffectEstimator:
    """Estimate causal effects"""
    
    def __init__(self, config: CausalInferenceConfig):
        self.config = config
        
    def estimate_causal_effect(self, data: pd.DataFrame, treatment: str, 
                             outcome: str, confounders: List[str] = None) -> Dict[str, Any]:
        """Estimate causal effect"""
        
        if self.config.effect_estimation_method == "propensity_score":
            return self._propensity_score_estimation(data, treatment, outcome, confounders)
        elif self.config.effect_estimation_method == "instrumental_variable":
            return self._instrumental_variable_estimation(data, treatment, outcome, confounders)
        elif self.config.effect_estimation_method == "regression_discontinuity":
            return self._regression_discontinuity_estimation(data, treatment, outcome, confounders)
        elif self.config.effect_estimation_method == "difference_in_differences":
            return self._difference_in_differences_estimation(data, treatment, outcome, confounders)
        else:
            raise ValueError(f"Unsupported effect estimation method: {self.config.effect_estimation_method}")
            
    def _propensity_score_estimation(self, data: pd.DataFrame, treatment: str, 
                                   outcome: str, confounders: List[str] = None) -> Dict[str, Any]:
        """Propensity score estimation"""
        
        if confounders is None:
            confounders = [col for col in data.columns if col not in [treatment, outcome]]
            
        # Prepare data
        X = data[confounders]
        y = data[outcome]
        t = data[treatment]
        
        # Estimate propensity scores
        if self.config.ps_method == "logistic":
            ps_model = LogisticRegression()
        elif self.config.ps_method == "random_forest":
            ps_model = RandomForestClassifier()
        else:
            ps_model = LogisticRegression()
            
        ps_model.fit(X, t)
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        
        # Create causal model
        causal_model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders
        )
        
        # Estimate causal effect
        causal_estimator = PropensityScoreEstimator(causal_model)
        causal_effect = causal_estimator.estimate_effect(
            method_name="backdoor.propensity_score_matching"
        )
        
        return {
            'causal_effect': causal_effect.value,
            'confidence_interval': causal_effect.confidence_interval,
            'propensity_scores': propensity_scores,
            'method': 'propensity_score'
        }
        
    def _instrumental_variable_estimation(self, data: pd.DataFrame, treatment: str, 
                                        outcome: str, confounders: List[str] = None) -> Dict[str, Any]:
        """Instrumental variable estimation"""
        
        if confounders is None:
            confounders = [col for col in data.columns if col not in [treatment, outcome]]
            
        # Create causal model
        causal_model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders
        )
        
        # Estimate causal effect
        causal_estimator = InstrumentalVariableEstimator(causal_model)
        causal_effect = causal_estimator.estimate_effect(
            method_name="iv.instrumental_variable"
        )
        
        return {
            'causal_effect': causal_effect.value,
            'confidence_interval': causal_effect.confidence_interval,
            'method': 'instrumental_variable'
        }
        
    def _regression_discontinuity_estimation(self, data: pd.DataFrame, treatment: str, 
                                           outcome: str, confounders: List[str] = None) -> Dict[str, Any]:
        """Regression discontinuity estimation"""
        
        if confounders is None:
            confounders = [col for col in data.columns if col not in [treatment, outcome]]
            
        # Create causal model
        causal_model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders
        )
        
        # Estimate causal effect
        causal_estimator = RegressionDiscontinuityEstimator(causal_model)
        causal_effect = causal_estimator.estimate_effect(
            method_name="iv.regression_discontinuity"
        )
        
        return {
            'causal_effect': causal_effect.value,
            'confidence_interval': causal_effect.confidence_interval,
            'method': 'regression_discontinuity'
        }
        
    def _difference_in_differences_estimation(self, data: pd.DataFrame, treatment: str, 
                                            outcome: str, confounders: List[str] = None) -> Dict[str, Any]:
        """Difference in differences estimation"""
        
        if confounders is None:
            confounders = [col for col in data.columns if col not in [treatment, outcome]]
            
        # Create causal model
        causal_model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders
        )
        
        # Estimate causal effect
        causal_estimator = DifferenceInDifferencesEstimator(causal_model)
        causal_effect = causal_estimator.estimate_effect(
            method_name="iv.difference_in_differences"
        )
        
        return {
            'causal_effect': causal_effect.value,
            'confidence_interval': causal_effect.confidence_interval,
            'method': 'difference_in_differences'
        }

class CounterfactualReasoner:
    """Counterfactual reasoning"""
    
    def __init__(self, config: CausalInferenceConfig):
        self.config = config
        
    def generate_counterfactuals(self, data: pd.DataFrame, treatment: str, 
                               outcome: str, confounders: List[str] = None) -> Dict[str, Any]:
        """Generate counterfactual scenarios"""
        
        if confounders is None:
            confounders = [col for col in data.columns if col not in [treatment, outcome]]
            
        # Prepare data
        X = data[confounders]
        y = data[outcome]
        t = data[treatment]
        
        # Train counterfactual model
        if self.config.counterfactual_method == "neural_network":
            counterfactual_model = self._train_neural_network(X, y, t)
        elif self.config.counterfactual_method == "random_forest":
            counterfactual_model = self._train_random_forest(X, y, t)
        else:
            counterfactual_model = self._train_linear_model(X, y, t)
            
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactual_scenarios(
            counterfactual_model, X, y, t
        )
        
        return {
            'counterfactuals': counterfactuals,
            'model': counterfactual_model,
            'method': self.config.counterfactual_method
        }
        
    def _train_neural_network(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> nn.Module:
        """Train neural network for counterfactual reasoning"""
        
        class CounterfactualNN(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(self.config.nn_dropout))
                    prev_dim = hidden_dim
                    
                layers.append(nn.Linear(prev_dim, output_dim))
                
                self.network = nn.Sequential(*layers)
                
            def forward(self, x):
                return self.network(x)
                
        # Create model
        model = CounterfactualNN(
            input_dim=X.shape[1],
            hidden_dims=self.config.nn_hidden_layers or [64, 32, 16],
            output_dim=1
        )
        
        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.nn_learning_rate)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
        
        for epoch in range(self.config.nn_epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        return model
        
    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> Any:
        """Train random forest for counterfactual reasoning"""
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
        
    def _train_linear_model(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> Any:
        """Train linear model for counterfactual reasoning"""
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model
        
    def _generate_counterfactual_scenarios(self, model: Any, X: pd.DataFrame, 
                                         y: pd.Series, t: pd.Series) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios"""
        
        counterfactuals = []
        
        # Generate counterfactual scenarios
        for i in range(self.config.counterfactual_samples):
            # Sample random treatment value
            counterfactual_treatment = np.random.choice([0, 1])
            
            # Create counterfactual data
            counterfactual_X = X.copy()
            counterfactual_X['treatment'] = counterfactual_treatment
            
            # Predict counterfactual outcome
            if isinstance(model, nn.Module):
                with torch.no_grad():
                    counterfactual_X_tensor = torch.FloatTensor(counterfactual_X.values)
                    counterfactual_outcome = model(counterfactual_X_tensor).item()
            else:
                counterfactual_outcome = model.predict(counterfactual_X)[0]
                
            counterfactuals.append({
                'treatment': counterfactual_treatment,
                'outcome': counterfactual_outcome,
                'confidence': self.config.counterfactual_confidence
            })
            
        return counterfactuals

class CausalInferenceEngine:
    """Main Causal Inference Engine"""
    
    def __init__(self, config: CausalInferenceConfig):
        self.config = config
        self.causal_discovery = CausalDiscovery(config)
        self.effect_estimator = CausalEffectEstimator(config)
        self.counterfactual_reasoner = CounterfactualReasoner(config)
        
        # Results storage
        self.results = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
    def discover_causal_structure(self, data: pd.DataFrame) -> nx.DiGraph:
        """Discover causal structure from data"""
        
        # Preprocess data
        if self.config.enable_preprocessing:
            data = self._preprocess_data(data)
            
        # Discover causal graph
        causal_graph = self.causal_discovery.discover_causal_graph(data)
        
        # Store results
        self.results['causal_discovery'].append({
            'graph': causal_graph,
            'nodes': list(causal_graph.nodes()),
            'edges': list(causal_graph.edges())
        })
        
        return causal_graph
        
    def estimate_causal_effects(self, data: pd.DataFrame, treatment: str, 
                              outcome: str, confounders: List[str] = None) -> Dict[str, Any]:
        """Estimate causal effects"""
        
        # Preprocess data
        if self.config.enable_preprocessing:
            data = self._preprocess_data(data)
            
        # Estimate causal effects
        causal_effects = self.effect_estimator.estimate_causal_effect(
            data, treatment, outcome, confounders
        )
        
        # Store results
        self.results['causal_effects'].append(causal_effects)
        
        return causal_effects
        
    def perform_counterfactual_analysis(self, data: pd.DataFrame, treatment: str, 
                                      outcome: str, confounders: List[str] = None) -> Dict[str, Any]:
        """Perform counterfactual analysis"""
        
        # Preprocess data
        if self.config.enable_preprocessing:
            data = self._preprocess_data(data)
            
        # Generate counterfactuals
        counterfactuals = self.counterfactual_reasoner.generate_counterfactuals(
            data, treatment, outcome, confounders
        )
        
        # Store results
        self.results['counterfactuals'].append(counterfactuals)
        
        return counterfactuals
        
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data"""
        
        # Handle missing values
        if self.config.handle_missing_values == "drop":
            data = data.dropna()
        elif self.config.handle_missing_values == "impute":
            data = data.fillna(data.mean())
        elif self.config.handle_missing_values == "forward_fill":
            data = data.fillna(method='ffill')
            
        # Feature scaling
        if self.config.feature_scaling:
            scaler = StandardScaler()
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            
        return data
        
    def evaluate_causal_models(self, data: pd.DataFrame, treatment: str, 
                             outcome: str, confounders: List[str] = None) -> Dict[str, Any]:
        """Evaluate causal models"""
        
        # Preprocess data
        if self.config.enable_preprocessing:
            data = self._preprocess_data(data)
            
        # Estimate causal effects
        causal_effects = self.estimate_causal_effects(data, treatment, outcome, confounders)
        
        # Perform counterfactual analysis
        counterfactuals = self.perform_counterfactual_analysis(data, treatment, outcome, confounders)
        
        # Calculate evaluation metrics
        evaluation_metrics = self._calculate_evaluation_metrics(
            causal_effects, counterfactuals
        )
        
        return {
            'causal_effects': causal_effects,
            'counterfactuals': counterfactuals,
            'evaluation_metrics': evaluation_metrics
        }
        
    def _calculate_evaluation_metrics(self, causal_effects: Dict[str, Any], 
                                    counterfactuals: Dict[str, Any]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        metrics = {}
        
        # Causal effect metrics
        if 'causal_effect' in causal_effects:
            metrics['causal_effect'] = causal_effects['causal_effect']
            
        if 'confidence_interval' in causal_effects:
            ci = causal_effects['confidence_interval']
            metrics['confidence_interval_width'] = ci[1] - ci[0]
            
        # Counterfactual metrics
        if 'counterfactuals' in counterfactuals:
            counterfactual_outcomes = [cf['outcome'] for cf in counterfactuals['counterfactuals']]
            metrics['counterfactual_mean'] = np.mean(counterfactual_outcomes)
            metrics['counterfactual_std'] = np.std(counterfactual_outcomes)
            
        return metrics
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        metrics = {
            'discovery_method': self.config.discovery_method,
            'effect_estimation_method': self.config.effect_estimation_method,
            'counterfactual_method': self.config.counterfactual_method,
            'total_discoveries': len(self.results.get('causal_discovery', [])),
            'total_effect_estimations': len(self.results.get('causal_effects', [])),
            'total_counterfactual_analyses': len(self.results.get('counterfactuals', []))
        }
        
        return metrics
        
    def save_results(self, filepath: str):
        """Save results to file"""
        
        results_data = {
            'results': dict(self.results),
            'performance_metrics': self.get_performance_metrics(),
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, default=str)
            
    def load_results(self, filepath: str):
        """Load results from file"""
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
            
        self.results = defaultdict(list, results_data['results'])

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test causal inference engine
    print("Testing Causal Inference Engine...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic causal data
    X1 = np.random.normal(0, 1, n_samples)
    X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
    X3 = 0.3 * X1 + 0.4 * X2 + np.random.normal(0, 0.3, n_samples)
    treatment = (X1 + X2 + np.random.normal(0, 0.2, n_samples) > 0).astype(int)
    outcome = 2 * treatment + 0.5 * X1 + 0.3 * X2 + 0.2 * X3 + np.random.normal(0, 0.1, n_samples)
    
    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'treatment': treatment,
        'outcome': outcome
    })
    
    # Create config
    config = CausalInferenceConfig(
        discovery_method="pc",
        effect_estimation_method="propensity_score",
        counterfactual_method="neural_network",
        pc_alpha=0.05,
        ps_method="logistic",
        nn_hidden_layers=[64, 32, 16],
        nn_epochs=50,
        evaluation_metrics=["ate", "att", "atc"]
    )
    
    # Create engine
    causal_engine = CausalInferenceEngine(config)
    
    # Test causal discovery
    print("Testing causal discovery...")
    causal_graph = causal_engine.discover_causal_structure(data)
    print(f"Causal graph discovered: {len(causal_graph.nodes())} nodes, {len(causal_graph.edges())} edges")
    
    # Test causal effect estimation
    print("Testing causal effect estimation...")
    causal_effects = causal_engine.estimate_causal_effects(data, 'treatment', 'outcome', ['X1', 'X2', 'X3'])
    print(f"Causal effect estimated: {causal_effects['causal_effect']:.4f}")
    
    # Test counterfactual analysis
    print("Testing counterfactual analysis...")
    counterfactuals = causal_engine.perform_counterfactual_analysis(data, 'treatment', 'outcome', ['X1', 'X2', 'X3'])
    print(f"Counterfactuals generated: {len(counterfactuals['counterfactuals'])} scenarios")
    
    # Test evaluation
    print("Testing evaluation...")
    evaluation = causal_engine.evaluate_causal_models(data, 'treatment', 'outcome', ['X1', 'X2', 'X3'])
    print(f"Evaluation completed: {len(evaluation['evaluation_metrics'])} metrics")
    
    # Get performance metrics
    metrics = causal_engine.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    print("\nCausal inference engine initialized successfully!")
























