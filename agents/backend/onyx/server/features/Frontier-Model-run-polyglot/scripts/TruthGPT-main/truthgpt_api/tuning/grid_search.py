"""
Grid Search for TruthGPT API
============================

TensorFlow-like grid search hyperparameter tuning implementation.
"""

import itertools
from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
import time


class GridSearch:
    """
    Grid search hyperparameter tuning.
    
    Similar to tf.keras.tuner.GridSearch, this class
    implements grid search for hyperparameter optimization.
    """
    
    def __init__(self, 
                 model_builder: Callable,
                 param_grid: Dict[str, List[Any]],
                 scoring: str = 'accuracy',
                 cv: int = 3,
                 n_jobs: int = 1,
                 verbose: int = 1,
                 name: Optional[str] = None):
        """
        Initialize GridSearch.
        
        Args:
            model_builder: Function that builds a model given parameters
            param_grid: Dictionary of parameters and their values
            scoring: Scoring metric
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            name: Optional name for the tuner
        """
        self.model_builder = model_builder
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.name = name or "grid_search"
        
        # Results storage
        self.results = []
        self.best_params = None
        self.best_score = None
        self.best_model = None
    
    def search(self, 
               x_train: np.ndarray,
               y_train: np.ndarray,
               x_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None,
               epochs: int = 10,
               batch_size: int = 32) -> Dict[str, Any]:
        """
        Perform grid search.
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Best parameters and results
        """
        print(f"🔍 Starting Grid Search...")
        print(f"   Parameter grid: {self.param_grid}")
        print(f"   Scoring: {self.scoring}")
        print(f"   CV folds: {self.cv}")
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)
        
        print(f"   Total combinations: {total_combinations}")
        
        # Search through parameter combinations
        for i, params in enumerate(param_combinations):
            if self.verbose > 0:
                print(f"\n🔧 Testing combination {i+1}/{total_combinations}")
                print(f"   Parameters: {params}")
            
            # Build and train model
            start_time = time.time()
            
            try:
                # Create model
                model = self.model_builder(**params)
                
                # Train model
                history = model.fit(
                    x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val) if x_val is not None else None,
                    verbose=0
                )
                
                # Evaluate model
                if x_val is not None and y_val is not None:
                    val_loss, val_score = model.evaluate(x_val, y_val, verbose=0)
                else:
                    # Use training score as fallback
                    val_score = history['accuracy'][-1] if 'accuracy' in history else history['loss'][-1]
                
                training_time = time.time() - start_time
                
                # Store results
                result = {
                    'params': params,
                    'score': val_score,
                    'training_time': training_time,
                    'history': history,
                    'model': model
                }
                self.results.append(result)
                
                if self.verbose > 0:
                    print(f"   Score: {val_score:.4f}")
                    print(f"   Training time: {training_time:.2f}s")
                
                # Update best results
                if self.best_score is None or val_score > self.best_score:
                    self.best_score = val_score
                    self.best_params = params
                    self.best_model = model
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"   ❌ Error: {e}")
                continue
        
        # Print summary
        self._print_summary()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': self.best_model,
            'results': self.results
        }
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _print_summary(self):
        """Print search summary."""
        print(f"\n📊 Grid Search Summary")
        print(f"=" * 50)
        print(f"Total combinations tested: {len(self.results)}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        
        print(f"\n🏆 Top 5 Results:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"   {i+1}. Score: {result['score']:.4f}, Params: {result['params']}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all search results."""
        return self.results
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters."""
        return self.best_params
    
    def get_best_score(self) -> float:
        """Get best score."""
        return self.best_score
    
    def get_best_model(self) -> Any:
        """Get best model."""
        return self.best_model
    
    def __repr__(self):
        return f"GridSearch(param_grid={self.param_grid}, scoring={self.scoring})"









