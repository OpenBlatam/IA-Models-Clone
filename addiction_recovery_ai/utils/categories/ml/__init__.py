"""
Machine Learning utilities
"""

from utils.categories import register_utility

try:
    from utils.model_utils import ModelUtils
    from utils.model_serving import ModelServing
    from utils.model_versioning import ModelVersioning
    from utils.model_interpretability import ModelInterpretability
    from utils.distributed_inference import DistributedInference
    from utils.automl import AutoML
    from utils.hyperparameter_optimization import HyperparameterOptimization
    from utils.continuous_learning import ContinuousLearning
    from utils.experiment_tracking import ExperimentTracking
    
    def register_utilities():
        register_utility("ml", "model_utils", ModelUtils)
        register_utility("ml", "model_serving", ModelServing)
        register_utility("ml", "model_versioning", ModelVersioning)
        register_utility("ml", "model_interpretability", ModelInterpretability)
        register_utility("ml", "distributed_inference", DistributedInference)
        register_utility("ml", "automl", AutoML)
        register_utility("ml", "hyperparameter_optimization", HyperparameterOptimization)
        register_utility("ml", "continuous_learning", ContinuousLearning)
        register_utility("ml", "experiment_tracking", ExperimentTracking)
except ImportError:
    pass



