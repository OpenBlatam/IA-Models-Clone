"""
ML Utils - Utilidades de Machine Learning y Deep Learning
=========================================================

Utilidades para entrenamiento, evaluación y fine-tuning de modelos.
"""

from .training_utils import (
    Trainer,
    TrainingConfig,
    EarlyStopping,
    LearningRateScheduler
)
from .data_utils import (
    DataProcessor,
    DatasetBuilder,
    DataLoaderBuilder
)
from .evaluation_utils import (
    ModelEvaluator,
    MetricsCalculator
)
from .fine_tuning_utils import (
    LoRATrainer,
    FineTuningConfig
)
from .model_utils import (
    ModelBuilder,
    ModelCheckpointer,
    load_pretrained_model
)
from .diffusion_utils import (
    DiffusionPipelineManager,
    NoiseScheduler
)
from .experiment_tracking import (
    ExperimentTracker,
    ExperimentManager
)
from .gradio_utils import (
    GradioInterfaceBuilder,
    create_model_demo,
    create_comparison_demo
)
from .optimization_utils import (
    MixedPrecisionManager,
    GradientAccumulator,
    ModelOptimizer,
    MultiGPUTrainer,
    MemoryOptimizer
)
from .augmentation_utils import (
    TextAugmenter,
    ImageAugmenter,
    TorchAugmenter,
    MixUpAugmenter,
    CutMixAugmenter
)
from .loss_utils import (
    FocalLoss,
    DiceLoss,
    IoULoss,
    LabelSmoothingLoss,
    TripletLoss,
    ContrastiveLoss,
    HuberLoss,
    KLDivergenceLoss,
    CombinedLoss
)
from .cv_utils import (
    CrossValidator,
    TimeSeriesCrossValidator,
    GroupKFold,
    k_fold_cv,
    stratified_k_fold_cv
)
from .tokenization_utils import (
    TextPreprocessor,
    AdvancedTokenizer,
    DynamicPadding,
    TokenizerWrapper,
    create_tokenizer
)
from .interpretability_utils import (
    AttentionVisualizer,
    GradientAnalyzer,
    FeatureImportance,
    CaptumWrapper
)
from .ensemble_utils import (
    ModelEnsemble,
    StackingEnsemble,
    BaggingEnsemble,
    create_ensemble
)
from .inference_utils import (
    BatchInferenceManager,
    ONNXExporter,
    ONNXRuntimeInference,
    InferenceOptimizer,
    TorchServeExporter,
    InferenceBenchmark
)
from .optimizer_utils import (
    Lookahead,
    RAdam,
    AdaBound,
    create_optimizer
)
from .architecture_utils import (
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    TransformerEncoder,
    ResNetBlock,
    LSTMEncoder,
    CNNEncoder,
    create_transformer,
    create_resnet
)
from .regularization_utils import (
    DropBlock,
    SpectralNorm,
    WeightDecayRegularizer,
    LabelSmoothingRegularizer,
    GradientPenalty,
    MixupRegularizer,
    apply_spectral_norm,
    apply_dropblock
)
from .distillation_utils import (
    DistillationLoss,
    DistillationTrainer,
    FeatureDistillation
)
from .compression_utils import (
    ModelPruner,
    ModelQuantizer,
    ModelCompressor
)
from .multitask_utils import (
    MultiTaskHead,
    MultiTaskModel,
    MultiTaskLoss,
    MultiTaskTrainer
)
from .hyperparameter_utils import (
    HyperparameterConfig,
    GridSearch,
    RandomSearch,
    OptunaOptimizer,
    HyperparameterTuner
)
from .pipeline_utils import (
    DataPipeline,
    ParallelDataLoader,
    StreamingDataset,
    CachedDataset,
    BatchProcessor,
    DataPrefetcher,
    DataBalancer
)
from .monitoring_utils import (
    TrainingMetrics,
    TrainingMonitor,
    GradientMonitor,
    ModelHealthMonitor,
    PerformanceMonitor
)
from .transfer_learning_utils import (
    FeatureExtractor,
    TransferLearningModel,
    ProgressiveUnfreezing,
    DomainAdaptation,
    load_pretrained_backbone,
    create_transfer_model
)
from .validation_utils import (
    ValidationResult,
    DataValidator,
    TensorValidator,
    ModelOutputValidator,
    DatasetValidator,
    validate_input
)
from .registry_utils import (
    ModelMetadata,
    ModelRegistry
)
from .active_learning_utils import (
    UncertaintySampler,
    DiversitySampler,
    QueryByCommittee,
    ActiveLearningLoop
)
from .fewshot_utils import (
    PrototypicalNetwork,
    MAML,
    FewShotDataset
)
from .adversarial_utils import (
    FGSMAttack,
    PGDAttack,
    AdversarialTrainer,
    AdversarialRobustness
)
from .debugging_utils import (
    GradientChecker,
    ActivationMonitor,
    ModelDebugger,
    detect_anomaly
)
from .reproducibility_utils import (
    ReproducibilityConfig,
    ReproducibilityManager,
    ExperimentSnapshot,
    set_seed,
    make_deterministic,
    save_experiment_state,
    load_experiment_state
)
from .continual_learning_utils import (
    EWC,
    ReplayBuffer,
    ContinualLearningTrainer
)
from .comparison_utils import (
    ModelComparison,
    ModelComparator
)
from .data_quality_utils import (
    DataQualityReport,
    DataQualityChecker,
    DataCleaner
)
from .explainability_utils import (
    SHAPExplainer,
    LIMEExplainer,
    FeatureImportanceAnalyzer
)
from .production_utils import (
    ModelVersion,
    ModelServer,
    ABTestManager,
    ModelCache
)
from .nas_utils import (
    ArchitectureConfig,
    ArchitectureSearch,
    SuperNet,
    WeightSharing
)
from .distributed_utils import (
    DistributedTrainer,
    GradientSynchronizer,
    DistributedDataLoader,
    setup_distributed,
    cleanup_distributed,
    all_reduce_mean,
    broadcast_tensor
)
from .feature_engineering_utils import (
    FeatureScaler,
    FeatureSelector,
    FeatureTransformer,
    PolynomialFeatures,
    InteractionFeatures,
    create_feature_pipeline
)
from .automl_utils import (
    AutoMLConfig,
    AutoMLPipeline,
    AutoFeatureEngineering,
    AutoHyperparameterTuning,
    create_automl_pipeline
)
from .timeseries_utils import (
    LSTMForecaster,
    GRUForecaster,
    TransformerForecaster,
    TimeSeriesDataset,
    create_sliding_windows
)
from .gnn_utils import (
    GCNLayer,
    GATLayer,
    GraphSAGELayer,
    GraphNeuralNetwork
)
from .rl_utils import (
    ReplayBuffer,
    DQN,
    PolicyNetwork,
    ActorCritic,
    EpsilonGreedy
)
from .serving_utils import (
    ServingConfig,
    RESTModelServer,
    BatchPredictor,
    ModelVersionManager
)
from .initialization_utils import (
    WeightInitializer,
    LayerInitializer
)
from .normalization_utils import (
    LayerNorm,
    GroupNorm,
    InstanceNorm,
    RMSNorm,
    AdaptiveNorm
)
from .attention_utils import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    SparseAttention
)
from .visualization_utils import (
    TrainingVisualizer,
    ModelArchitectureVisualizer,
    MetricsVisualizer
)
from .config_utils import (
    TrainingConfig,
    ModelConfig,
    ConfigManager,
    create_default_config
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "EarlyStopping",
    "LearningRateScheduler",
    "DataProcessor",
    "DatasetBuilder",
    "DataLoaderBuilder",
    "ModelEvaluator",
    "MetricsCalculator",
    "LoRATrainer",
    "FineTuningConfig",
    "ModelBuilder",
    "ModelCheckpointer",
    "load_pretrained_model",
    "DiffusionPipelineManager",
    "NoiseScheduler",
    "ExperimentTracker",
    "ExperimentManager",
    "GradioInterfaceBuilder",
    "create_model_demo",
    "create_comparison_demo",
    "MixedPrecisionManager",
    "GradientAccumulator",
    "ModelOptimizer",
    "MultiGPUTrainer",
    "MemoryOptimizer",
    "TextAugmenter",
    "ImageAugmenter",
    "TorchAugmenter",
    "MixUpAugmenter",
    "CutMixAugmenter",
    "FocalLoss",
    "DiceLoss",
    "IoULoss",
    "LabelSmoothingLoss",
    "TripletLoss",
    "ContrastiveLoss",
    "HuberLoss",
    "KLDivergenceLoss",
    "CombinedLoss",
    "CrossValidator",
    "TimeSeriesCrossValidator",
    "GroupKFold",
    "k_fold_cv",
    "stratified_k_fold_cv",
    "TextPreprocessor",
    "AdvancedTokenizer",
    "DynamicPadding",
    "TokenizerWrapper",
    "create_tokenizer",
    "AttentionVisualizer",
    "GradientAnalyzer",
    "FeatureImportance",
    "CaptumWrapper",
    "ModelEnsemble",
    "StackingEnsemble",
    "BaggingEnsemble",
    "create_ensemble",
    "BatchInferenceManager",
    "ONNXExporter",
    "ONNXRuntimeInference",
    "InferenceOptimizer",
    "TorchServeExporter",
    "InferenceBenchmark",
    "Lookahead",
    "RAdam",
    "AdaBound",
    "create_optimizer",
    "PositionalEncoding",
    "MultiHeadAttention",
    "TransformerBlock",
    "TransformerEncoder",
    "ResNetBlock",
    "LSTMEncoder",
    "CNNEncoder",
    "create_transformer",
    "create_resnet",
    "DropBlock",
    "SpectralNorm",
    "WeightDecayRegularizer",
    "LabelSmoothingRegularizer",
    "GradientPenalty",
    "MixupRegularizer",
    "apply_spectral_norm",
    "apply_dropblock",
    "DistillationLoss",
    "DistillationTrainer",
    "FeatureDistillation",
    "ModelPruner",
    "ModelQuantizer",
    "ModelCompressor",
    "MultiTaskHead",
    "MultiTaskModel",
    "MultiTaskLoss",
    "MultiTaskTrainer",
    "HyperparameterConfig",
    "GridSearch",
    "RandomSearch",
    "OptunaOptimizer",
    "HyperparameterTuner",
    "DataPipeline",
    "ParallelDataLoader",
    "StreamingDataset",
    "CachedDataset",
    "BatchProcessor",
    "DataPrefetcher",
    "DataBalancer",
    "TrainingMetrics",
    "TrainingMonitor",
    "GradientMonitor",
    "ModelHealthMonitor",
    "PerformanceMonitor",
    "FeatureExtractor",
    "TransferLearningModel",
    "ProgressiveUnfreezing",
    "DomainAdaptation",
    "load_pretrained_backbone",
    "create_transfer_model",
    "ValidationResult",
    "DataValidator",
    "TensorValidator",
    "ModelOutputValidator",
    "DatasetValidator",
    "validate_input",
    "ModelMetadata",
    "ModelRegistry",
    "UncertaintySampler",
    "DiversitySampler",
    "QueryByCommittee",
    "ActiveLearningLoop",
    "PrototypicalNetwork",
    "MAML",
    "FewShotDataset",
    "FGSMAttack",
    "PGDAttack",
    "AdversarialTrainer",
    "AdversarialRobustness",
    "GradientChecker",
    "ActivationMonitor",
    "ModelDebugger",
    "detect_anomaly",
    "ReproducibilityConfig",
    "ReproducibilityManager",
    "ExperimentSnapshot",
    "set_seed",
    "make_deterministic",
    "save_experiment_state",
    "load_experiment_state",
    "EWC",
    "ReplayBuffer",
    "ContinualLearningTrainer",
    "ModelComparison",
    "ModelComparator",
    "DataQualityReport",
    "DataQualityChecker",
    "DataCleaner",
    "SHAPExplainer",
    "LIMEExplainer",
    "FeatureImportanceAnalyzer",
    "ModelVersion",
    "ModelServer",
    "ABTestManager",
    "ModelCache",
    "ArchitectureConfig",
    "ArchitectureSearch",
    "SuperNet",
    "WeightSharing",
    "DistributedTrainer",
    "GradientSynchronizer",
    "DistributedDataLoader",
    "setup_distributed",
    "cleanup_distributed",
    "all_reduce_mean",
    "broadcast_tensor",
    "FeatureScaler",
    "FeatureSelector",
    "FeatureTransformer",
    "PolynomialFeatures",
    "InteractionFeatures",
    "create_feature_pipeline",
    "AutoMLConfig",
    "AutoMLPipeline",
    "AutoFeatureEngineering",
    "AutoHyperparameterTuning",
    "create_automl_pipeline",
    "LSTMForecaster",
    "GRUForecaster",
    "TransformerForecaster",
    "TimeSeriesDataset",
    "create_sliding_windows",
    "GCNLayer",
    "GATLayer",
    "GraphSAGELayer",
    "GraphNeuralNetwork",
    "ReplayBuffer",
    "DQN",
    "PolicyNetwork",
    "ActorCritic",
    "EpsilonGreedy",
    "ServingConfig",
    "RESTModelServer",
    "BatchPredictor",
    "ModelVersionManager",
    "WeightInitializer",
    "LayerInitializer",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm",
    "RMSNorm",
    "AdaptiveNorm",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "SelfAttention",
    "CrossAttention",
    "SparseAttention",
    "TrainingVisualizer",
    "ModelArchitectureVisualizer",
    "MetricsVisualizer",
    "TrainingConfig",
    "ModelConfig",
    "ConfigManager",
    "create_default_config",
]

