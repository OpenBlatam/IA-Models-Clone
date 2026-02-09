"""
Generator Configuration - Configuración de generadores
======================================================

Define la configuración centralizada de todos los generadores de Deep Learning.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

from typing import Dict, Tuple, List, Final

GeneratorConfig = Dict[str, Tuple[str, str, str]]

GENERATOR_MAP: Final[GeneratorConfig] = {
    "model": ("model_generator", "ModelGenerator", "models"),
    "training": ("training_generator", "TrainingGenerator", "utils"),
    "data": ("data_generator", "DataGenerator", "utils"),
    "evaluation": ("evaluation_generator", "EvaluationGenerator", "utils"),
    "performance": ("performance_generator", "PerformanceGenerator", "utils"),
    "utils": ("utils_generator", "UtilsGenerator", "utils"),
    "deployment": ("deployment_generator", "DeploymentGenerator", "utils"),
    "testing": ("testing_generator", "TestingGenerator", "utils"),
    "conversion": ("conversion_generator", "ConversionGenerator", "utils"),
    "monitoring": ("monitoring_generator", "MonitoringGenerator", "utils"),
    "analysis": ("analysis_generator", "AnalysisGenerator", "utils"),
    "experimentation": ("experimentation_generator", "ExperimentationGenerator", "utils"),
    "serialization": ("serialization_generator", "SerializationGenerator", "utils"),
    "validation": ("validation_generator", "ValidationGenerator", "utils"),
    "security": ("security_generator", "SecurityGenerator", "utils"),
    "data_io": ("data_io_generator", "DataIOGenerator", "utils"),
    "reporting": ("reporting_generator", "ReportingGenerator", "utils"),
    "preprocessing": ("preprocessing_generator", "PreprocessingGenerator", "utils"),
    "postprocessing": ("postprocessing_generator", "PostprocessingGenerator", "utils"),
    "interface": ("interface_generator", "InterfaceGenerator", "services"),
    "config": ("config_generator", "ConfigGenerator", "config"),
}

TRAINING_UTILS: Final[List[str]] = [
    "training", "data", "evaluation", "performance", "utils", "deployment",
    "testing", "conversion", "monitoring", "analysis", "experimentation",
    "serialization", "validation", "security", "data_io", "reporting",
    "preprocessing", "postprocessing"
]

GENERATION_GROUPS: Final[Dict[str, List[str]]] = {
    "core": ["model"],
    "training": TRAINING_UTILS,
    "interfaces": ["interface"],
    "config": ["config"],
}
