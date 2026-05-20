"""
Router Registry - Sistema de registro centralizado de routers
=============================================================

Gestiona la carga y registro de todos los routers de la aplicación
de manera organizada y eficiente.
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable
from fastapi import APIRouter

logger = logging.getLogger(__name__)


class RouterRegistry:
    """Registro centralizado de routers"""
    
    def __init__(self):
        self._routers: Dict[str, Tuple[APIRouter, bool]] = {}
        self._loaded_routers: Dict[str, APIRouter] = {}
        self._failed_imports: List[str] = []
    
    def register(
        self, 
        name: str, 
        router: APIRouter, 
        required: bool = True
    ) -> None:
        """
        Registrar un router
        
        Args:
            name: Nombre identificador del router
            router: Instancia del router
            required: Si es True, falla si no se puede cargar
        """
        self._routers[name] = (router, required)
        self._loaded_routers[name] = router
    
    def register_lazy(
        self, 
        name: str, 
        import_path: str, 
        router_name: str = "router",
        required: bool = False
    ) -> None:
        """
        Registrar un router para carga diferida
        
        Args:
            name: Nombre identificador del router
            import_path: Ruta de importación (ej: "api.routes")
            router_name: Nombre del atributo router (default: "router")
            required: Si es True, falla si no se puede cargar
        """
        self._routers[name] = ((import_path, router_name), required)
    
    def load_lazy_router(self, name: str) -> Optional[APIRouter]:
        """
        Cargar un router registrado de forma diferida
        
        Args:
            name: Nombre del router
            
        Returns:
            Router cargado o None si falla
        """
        if name in self._loaded_routers:
            return self._loaded_routers[name]
        
        if name not in self._routers:
            logger.warning(f"Router '{name}' no registrado")
            return None
        
        router_info, required = self._routers[name]
        
        # Si ya es un router, retornarlo
        if isinstance(router_info, APIRouter):
            return router_info
        
        # Si es una tupla (import_path, router_name), cargarlo
        import_path, router_name = router_info
        
        try:
            module = __import__(import_path, fromlist=[router_name])
            router = getattr(module, router_name)
            self._loaded_routers[name] = router
            logger.info(f"Router '{name}' cargado exitosamente")
            return router
        except ImportError as e:
            error_msg = f"No se pudo importar router '{name}' desde {import_path}: {e}"
            self._failed_imports.append(name)
            
            if required:
                logger.error(error_msg)
                raise ImportError(error_msg)
            else:
                logger.warning(f"{error_msg} (opcional, continuando...)")
                return None
        except AttributeError as e:
            error_msg = f"Router '{name}' no tiene atributo '{router_name}': {e}"
            self._failed_imports.append(name)
            
            if required:
                logger.error(error_msg)
                raise AttributeError(error_msg)
            else:
                logger.warning(f"{error_msg} (opcional, continuando...)")
                return None
    
    def load_all_routers(self, force_reload: bool = False) -> List[APIRouter]:
        """
        Cargar todos los routers registrados
        
        Args:
            force_reload: Si es True, recarga todos los routers incluso si ya están cargados
        
        Returns:
            Lista de routers cargados exitosamente
        """
        if not force_reload and self._loaded_routers:
            # Si ya están cargados y no se fuerza recarga, retornar los existentes
            return list(self._loaded_routers.values())
        
        loaded = []
        
        for name in self._routers.keys():
            router = self.load_lazy_router(name)
            if router is not None:
                loaded.append(router)
        
        logger.info(
            f"Cargados {len(loaded)} routers. "
            f"Fallos: {len(self._failed_imports)} ({', '.join(self._failed_imports)})"
        )
        
        return loaded
    
    def get_router(self, name: str) -> Optional[APIRouter]:
        """Obtener un router cargado"""
        return self._loaded_routers.get(name)
    
    def get_failed_imports(self) -> List[str]:
        """Obtener lista de imports que fallaron"""
        return self._failed_imports.copy()
    
    def get_loaded_count(self) -> int:
        """Obtener número de routers cargados exitosamente"""
        return len(self._loaded_routers)
    
    def get_total_registered(self) -> int:
        """Obtener número total de routers registrados"""
        return len(self._routers)


# Instancia global del registro
_registry = RouterRegistry()


def get_registry() -> RouterRegistry:
    """Obtener la instancia global del registro"""
    return _registry


def register_core_routers(registry: RouterRegistry) -> None:
    """Registrar routers principales (requeridos)"""
    
    # Routers principales - requeridos
    core_routers = [
        ("main", "api.routes", "router"),
        ("metrics", "api.metrics_routes", "router"),
        ("batch", "api.batch_routes", "router"),
        ("advanced", "api.advanced_routes", "router"),
        ("validator", "api.validator_routes", "router"),
        ("trends", "api.trends_routes", "router"),
        ("summary", "api.summary_routes", "router"),
        ("ocr", "api.ocr_routes", "router"),
        ("templates", "api.templates_routes", "router"),
        ("sentiment", "api.sentiment_routes", "router"),
        ("search", "api.search_routes", "router"),
        ("workflow", "api.workflow_routes", "router"),
        ("anomaly", "api.anomaly_routes", "router"),
        ("predictive", "api.predictive_routes", "router"),
        ("vector_db", "api.vector_db_routes", "router"),
        ("image", "api.image_routes", "router"),
        ("alerts", "api.alerts_routes", "router"),
        ("audit", "api.audit_routes", "router"),
        ("websocket", "api.websocket_routes", "router"),
        ("streaming", "api.streaming_routes", "router"),
        ("dashboard", "api.dashboard_routes", "router"),
    ]
    
    for name, import_path, router_name in core_routers:
        registry.register_lazy(name, import_path, router_name, required=True)


def register_optional_routers(registry: RouterRegistry) -> None:
    """Registrar routers opcionales (pueden fallar sin problemas)"""
    
    # Routers opcionales - pueden no estar disponibles
    optional_routers = [
        ("tenant", "api.tenant_routes", "router"),
        ("versioning", "api.versioning_routes", "router"),
        ("pipeline", "api.pipeline_routes", "router"),
        ("profiler", "api.profiler_routes", "router"),
        ("scaling", "api.scaling_routes", "router"),
        ("testing", "api.testing_routes", "router"),
        ("analytics", "api.analytics_routes", "router"),
        ("backup", "api.backup_routes", "router"),
        ("recommendation", "api.recommendation_routes", "router"),
        ("gateway", "api.gateway_routes", "router"),
        ("cloud", "api.cloud_routes", "router"),
        ("resource", "api.resource_routes", "router"),
        ("advanced_health", "api.advanced_health_routes", "router"),
        ("federated", "api.federated_routes", "router"),
        ("automl", "api.automl_routes", "router"),
        ("nlp", "api.nlp_routes", "router"),
        ("orchestrator", "api.orchestrator_routes", "router"),
        ("database", "api.database_routes", "router"),
        ("cache_distributed", "api.cache_routes", "router"),
        ("edge", "api.edge_routes", "router"),
        ("knowledge_graph", "api.knowledge_graph_routes", "router"),
        ("quantum", "api.quantum_routes", "router"),
        ("blockchain", "api.blockchain_routes", "router"),
        ("agent", "api.agent_routes", "router"),
        ("multimodal", "api.multimodal_routes", "router"),
        ("rl", "api.rl_routes", "router"),
        ("computer_vision", "api.computer_vision_routes", "router"),
        ("video", "api.video_routes", "router"),
        ("audio", "api.audio_routes", "router"),
        ("transfer", "api.transfer_routes", "router"),
        ("nas", "api.nas_routes", "router"),
        ("xai", "api.xai_routes", "router"),
        ("adversarial", "api.adversarial_routes", "router"),
        ("continual", "api.continual_routes", "router"),
        ("fewshot", "api.fewshot_routes", "router"),
        ("meta", "api.meta_routes", "router"),
        ("active", "api.active_routes", "router"),
        ("ssl", "api.ssl_routes", "router"),
        ("contrastive", "api.contrastive_routes", "router"),
        ("generative", "api.generative_routes", "router"),
        ("prompt", "api.prompt_routes", "router"),
        ("time_series", "api.time_series_routes", "router"),
        ("gnn", "api.gnn_routes", "router"),
        ("causal", "api.causal_routes", "router"),
        ("online", "api.online_routes", "router"),
        ("multitask", "api.multitask_routes", "router"),
        ("hyperparameter", "api.hyperparameter_routes", "router"),
        ("feature", "api.feature_routes", "router"),
        ("ensemble", "api.ensemble_routes", "router"),
        ("augmentation", "api.augmentation_routes", "router"),
        ("compression", "api.compression_routes", "router"),
        ("nlg", "api.nlg_routes", "router"),
        ("interpretability", "api.interpretability_routes", "router"),
        ("serving", "api.serving_routes", "router"),
        ("abtest", "api.abtest_routes", "router"),
        ("mlops", "api.mlops_routes", "router"),
        ("automl_advanced", "api.automl_advanced_routes", "router"),
        ("rag", "api.rag_routes", "router"),
        ("evaluation", "api.evaluation_routes", "router"),
        ("bias", "api.bias_routes", "router"),
        ("privacy", "api.privacy_routes", "router"),
        ("prompt_opt", "api.prompt_opt_routes", "router"),
        ("federation", "api.federation_routes", "router"),
        ("imitation", "api.imitation_routes", "router"),
        ("drift", "api.drift_routes", "router"),
        ("memory", "api.memory_routes", "router"),
        ("cost", "api.cost_routes", "router"),
        ("marl", "api.marl_routes", "router"),
        ("resource_ml", "api.resource_ml_routes", "router"),
        ("adversarial_detection", "api.adversarial_detection_routes", "router"),
        ("transfer_advanced", "api.transfer_advanced_routes", "router"),
        ("uncertainty", "api.uncertainty_routes", "router"),
        ("compression_advanced", "api.compression_advanced_routes", "router"),
        ("federated_advanced", "api.federated_advanced_routes", "router"),
        ("nas_advanced", "api.nas_advanced_routes", "router"),
        ("interpretability_advanced", "api.interpretability_advanced_routes", "router"),
        ("deployment", "api.deployment_routes", "router"),
        ("hpo_advanced", "api.hpo_advanced_routes", "router"),
        ("feature_store", "api.feature_store_routes", "router"),
        ("monitoring_advanced", "api.monitoring_advanced_routes", "router"),
        ("experiment_tracking", "api.experiment_tracking_routes", "router"),
        ("governance", "api.governance_routes", "router"),
        ("data_versioning", "api.data_versioning_routes", "router"),
        ("model_registry", "api.model_registry_routes", "router"),
        ("feature_engineering_advanced", "api.feature_engineering_advanced_routes", "router"),
        ("serving_advanced", "api.serving_advanced_routes", "router"),
        ("pipeline_orchestration", "api.pipeline_orchestration_routes", "router"),
    ]
    
    for name, import_path, router_name in optional_routers:
        registry.register_lazy(name, import_path, router_name, required=False)

