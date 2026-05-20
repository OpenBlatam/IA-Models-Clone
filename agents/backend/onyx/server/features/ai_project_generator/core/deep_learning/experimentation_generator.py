"""
Experimentation Generator - Generador de utilidades de experimentación
======================================================================

Genera utilidades para experimentación avanzada:
- Hyperparameter tuning
- Experiment tracking avanzado
- A/B testing de modelos
- Experiment management
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ExperimentationGenerator:
    """Generador de utilidades de experimentación"""
    
    def __init__(self):
        """Inicializa el generador de experimentación"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de experimentación.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        exp_dir = utils_dir / "experimentation"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_hyperparameter_tuning(exp_dir, keywords, project_info)
        self._generate_experiment_manager(exp_dir, keywords, project_info)
        self._generate_experimentation_init(exp_dir, keywords)
    
    def _generate_experimentation_init(
        self,
        exp_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de experimentación"""
        
        init_content = '''"""
Experimentation Utilities Module
=================================

Utilidades para experimentación avanzada y hyperparameter tuning.
"""

from .hyperparameter_tuning import (
    HyperparameterTuner,
    create_tuner,
    run_hyperparameter_search,
)
from .experiment_manager import (
    ExperimentManager,
    create_experiment,
    compare_experiments,
)

__all__ = [
    "HyperparameterTuner",
    "create_tuner",
    "run_hyperparameter_search",
    "ExperimentManager",
    "create_experiment",
    "compare_experiments",
]
'''
        
        (exp_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_hyperparameter_tuning(
        self,
        exp_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de hyperparameter tuning"""
        
        tuning_content = '''"""
Hyperparameter Tuning - Optimización de hyperparámetros
=======================================================

Utilidades para buscar los mejores hyperparámetros usando Optuna.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Optional, List
import logging

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna no disponible. Instala con: pip install optuna")

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Tuner de hyperparámetros usando Optuna.
    
    Permite buscar automáticamente los mejores hyperparámetros.
    """
    
    def __init__(
        self,
        study_name: str = "hyperparameter_tuning",
        direction: str = "minimize",
        n_trials: int = 100,
    ):
        """
        Inicializa el tuner.
        
        Args:
            study_name: Nombre del estudio
            direction: Dirección de optimización (minimize/maximize)
            n_trials: Número de trials
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna no está disponible. Instala con: pip install optuna")
        
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
        )
    
    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sugiere hyperparámetros según el espacio de búsqueda.
        
        Args:
            trial: Trial de Optuna
            search_space: Espacio de búsqueda
        
        Returns:
            Diccionario con hyperparámetros sugeridos
        """
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config.get("type", "float")
            
            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )
        
        return params
    
    def optimize(
        self,
        objective_function: Callable[[optuna.Trial], float],
        search_space: Optional[Dict[str, Any]] = None,
    ) -> optuna.Study:
        """
        Optimiza hyperparámetros.
        
        Args:
            objective_function: Función objetivo que retorna métrica
            search_space: Espacio de búsqueda (opcional)
        
        Returns:
            Estudio de Optuna
        """
        def wrapped_objective(trial):
            if search_space:
                params = self.suggest_hyperparameters(trial, search_space)
                trial.set_user_attr("params", params)
            return objective_function(trial)
        
        self.study.optimize(wrapped_objective, n_trials=self.n_trials)
        
        logger.info(f"Mejores hyperparámetros: {self.study.best_params}")
        logger.info(f"Mejor valor: {self.study.best_value}")
        
        return self.study
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Obtiene mejores hyperparámetros.
        
        Returns:
            Diccionario con mejores parámetros
        """
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """
        Obtiene mejor valor de métrica.
        
        Returns:
            Mejor valor
        """
        return self.study.best_value


def create_tuner(
    study_name: str = "hyperparameter_tuning",
    direction: str = "minimize",
    n_trials: int = 100,
) -> HyperparameterTuner:
    """
    Factory function para crear tuner.
    
    Args:
        study_name: Nombre del estudio
        direction: Dirección de optimización
        n_trials: Número de trials
    
    Returns:
        HyperparameterTuner configurado
    """
    return HyperparameterTuner(study_name, direction, n_trials)


def run_hyperparameter_search(
    model_factory: Callable[[Dict[str, Any]], nn.Module],
    train_function: Callable[[nn.Module, Dict[str, Any]], float],
    search_space: Dict[str, Any],
    n_trials: int = 100,
) -> Dict[str, Any]:
    """
    Ejecuta búsqueda de hyperparámetros.
    
    Args:
        model_factory: Función que crea modelo con parámetros
        train_function: Función que entrena y retorna métrica
        search_space: Espacio de búsqueda
        n_trials: Número de trials
    
    Returns:
        Diccionario con mejores parámetros y valor
    """
    tuner = create_tuner(n_trials=n_trials)
    
    def objective(trial):
        params = tuner.suggest_hyperparameters(trial, search_space)
        model = model_factory(params)
        metric = train_function(model, params)
        return metric
    
    study = tuner.optimize(objective, search_space)
    
    return {
        "best_params": tuner.get_best_params(),
        "best_value": tuner.get_best_value(),
        "study": study,
    }
'''
        
        (exp_dir / "hyperparameter_tuning.py").write_text(tuning_content, encoding="utf-8")
    
    def _generate_experiment_manager(
        self,
        exp_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera gestor de experimentos"""
        
        manager_content = '''"""
Experiment Manager - Gestor de experimentos
============================================

Sistema para gestionar y comparar múltiples experimentos.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Gestor de experimentos.
    
    Permite guardar, cargar y comparar experimentos.
    """
    
    def __init__(self, experiments_dir: Path):
        """
        Inicializa el gestor.
        
        Args:
            experiments_dir: Directorio donde guardar experimentos
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.experiments_dir / "experiments.json"
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> Dict[str, Any]:
        """Carga experimentos guardados"""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando experimentos: {e}")
                return {}
        return {}
    
    def _save_experiments(self) -> None:
        """Guarda experimentos"""
        try:
            with open(self.experiments_file, "w") as f:
                json.dump(self.experiments, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando experimentos: {e}")
    
    def create_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        model_path: Optional[Path] = None,
    ) -> str:
        """
        Crea un nuevo experimento.
        
        Args:
            experiment_name: Nombre del experimento
            config: Configuración del experimento
            model_path: Ruta al modelo (opcional)
        
        Returns:
            ID del experimento
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment = {
            "id": experiment_id,
            "name": experiment_name,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path) if model_path else None,
            "metrics": {},
            "status": "created",
        }
        
        self.experiments[experiment_id] = experiment
        self._save_experiments()
        
        logger.info(f"Experimento creado: {experiment_id}")
        return experiment_id
    
    def update_experiment(
        self,
        experiment_id: str,
        metrics: Optional[Dict[str, float]] = None,
        status: Optional[str] = None,
    ) -> None:
        """
        Actualiza un experimento.
        
        Args:
            experiment_id: ID del experimento
            metrics: Métricas a actualizar (opcional)
            status: Estado a actualizar (opcional)
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experimento {experiment_id} no encontrado")
        
        if metrics:
            self.experiments[experiment_id]["metrics"].update(metrics)
        
        if status:
            self.experiments[experiment_id]["status"] = status
        
        self.experiments[experiment_id]["updated_at"] = datetime.now().isoformat()
        self._save_experiments()
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Obtiene un experimento.
        
        Args:
            experiment_id: ID del experimento
        
        Returns:
            Diccionario con información del experimento
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experimento {experiment_id} no encontrado")
        
        return self.experiments[experiment_id]
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        Lista todos los experimentos.
        
        Returns:
            Lista de experimentos
        """
        return list(self.experiments.values())
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric_name: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Compara múltiples experimentos.
        
        Args:
            experiment_ids: IDs de experimentos a comparar
            metric_name: Nombre de métrica a comparar
        
        Returns:
            Diccionario con comparación
        """
        comparison = {
            "metric": metric_name,
            "experiments": [],
        }
        
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            metric_value = exp["metrics"].get(metric_name, None)
            
            comparison["experiments"].append({
                "id": exp_id,
                "name": exp["name"],
                "metric_value": metric_value,
                "config": exp["config"],
            })
        
        # Ordenar por métrica
        comparison["experiments"].sort(
            key=lambda x: x["metric_value"] if x["metric_value"] is not None else float('-inf'),
            reverse=True,
        )
        
        return comparison


def create_experiment(
    experiments_dir: Path,
    experiment_name: str,
    config: Dict[str, Any],
) -> str:
    """
    Función helper para crear experimento.
    
    Args:
        experiments_dir: Directorio de experimentos
        experiment_name: Nombre del experimento
        config: Configuración
    
    Returns:
        ID del experimento
    """
    manager = ExperimentManager(experiments_dir)
    return manager.create_experiment(experiment_name, config)


def compare_experiments(
    experiments_dir: Path,
    experiment_ids: List[str],
    metric_name: str = "accuracy",
) -> Dict[str, Any]:
    """
    Función helper para comparar experimentos.
    
    Args:
        experiments_dir: Directorio de experimentos
        experiment_ids: IDs de experimentos
        metric_name: Nombre de métrica
    
    Returns:
        Diccionario con comparación
    """
    manager = ExperimentManager(experiments_dir)
    return manager.compare_experiments(experiment_ids, metric_name)
'''
        
        (exp_dir / "experiment_manager.py").write_text(manager_content, encoding="utf-8")

