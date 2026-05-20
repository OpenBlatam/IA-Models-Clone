"""
Bulk Operations Improvements - Mejoras y Optimizaciones Avanzadas
===================================================================

Clases adicionales para mejorar el rendimiento y robustez del sistema bulk.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class BulkPerformanceMonitor:
    """Monitor avanzado de performance para operaciones bulk."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts: List[Dict[str, Any]] = []
    
    def record_operation(
        self,
        operation_name: str,
        duration: float,
        items_processed: int,
        success_rate: float
    ):
        """Registrar métrica de operación."""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = []
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "items_processed": items_processed,
            "success_rate": success_rate,
            "throughput": items_processed / duration if duration > 0 else 0
        }
        
        self.metrics[operation_name].append(metric)
        
        # Mantener solo últimos 1000 registros
        if len(self.metrics[operation_name]) > 1000:
            self.metrics[operation_name] = self.metrics[operation_name][-1000:]
        
        # Verificar alertas
        self._check_alerts(operation_name, metric)
    
    def _check_alerts(self, operation_name: str, metric: Dict[str, Any]):
        """Verificar y generar alertas."""
        # Alertas de throughput bajo
        if metric["throughput"] < 10 and metric["items_processed"] > 100:
            self.alerts.append({
                "type": "low_throughput",
                "operation": operation_name,
                "message": f"Low throughput detected: {metric['throughput']:.2f} items/sec",
                "timestamp": datetime.now().isoformat()
            })
        
        # Alertas de duración alta
        if metric["duration"] > 300:  # Más de 5 minutos
            self.alerts.append({
                "type": "high_duration",
                "operation": operation_name,
                "message": f"Long operation detected: {metric['duration']:.2f}s",
                "timestamp": datetime.now().isoformat()
            })
    
    def get_performance_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas de performance."""
        if operation_name:
            if operation_name not in self.metrics:
                return {}
            
            metrics = self.metrics[operation_name]
            if not metrics:
                return {}
            
            durations = [m["duration"] for m in metrics]
            throughputs = [m["throughput"] for m in metrics]
            success_rates = [m["success_rate"] for m in metrics]
            
            return {
                "operation": operation_name,
                "total_operations": len(metrics),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "avg_throughput": sum(throughputs) / len(throughputs),
                "avg_success_rate": sum(success_rates) / len(success_rates),
                "recent_alerts": len([a for a in self.alerts if a["operation"] == operation_name])
            }
        else:
            # Estadísticas globales
            all_stats = {}
            for op_name in self.metrics.keys():
                all_stats[op_name] = self.get_performance_stats(op_name)
            
            return {
                "operations": all_stats,
                "total_alerts": len(self.alerts),
                "recent_alerts": self.alerts[-10:]  # Últimos 10
            }
    
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtener alertas recientes."""
        return self.alerts[-limit:]


class BulkAdaptiveOptimizer:
    """Optimizador adaptativo que ajusta parámetros automáticamente."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {
            "max_workers": 10,
            "batch_size": 100,
            "retry_max": 3,
            "timeout": 60.0
        }
        self.performance_history: List[Dict[str, Any]] = []
    
    def optimize_config(
        self,
        operation_type: str,
        current_performance: Dict[str, Any],
        target_throughput: Optional[float] = None
    ) -> Dict[str, Any]:
        """Optimizar configuración basado en performance."""
        # Registrar performance actual
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "performance": current_performance,
            "config": self.config.copy()
        })
        
        # Mantener solo últimos 100 registros
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Ajustar workers basado en throughput
        current_throughput = current_performance.get("throughput", 0)
        if target_throughput and current_throughput < target_throughput * 0.8:
            # Aumentar workers si el throughput es bajo
            self.config["max_workers"] = min(
                self.config["max_workers"] + 2,
                50  # Límite máximo
            )
        
        # Ajustar batch size basado en memoria y errores
        error_rate = current_performance.get("error_rate", 0)
        if error_rate > 0.1:  # Más del 10% de errores
            # Reducir batch size si hay muchos errores
            self.config["batch_size"] = max(
                int(self.config["batch_size"] * 0.8),
                10  # Mínimo
            )
        
        return self.config.copy()
    
    def get_recommended_config(self, operation_type: str) -> Dict[str, Any]:
        """Obtener configuración recomendada."""
        # Analizar historial para esta operación
        relevant_history = [
            h for h in self.performance_history
            if h["operation_type"] == operation_type
        ]
        
        if not relevant_history:
            return self.config.copy()
        
        # Encontrar mejor configuración
        best_perf = max(
            relevant_history,
            key=lambda h: h["performance"].get("throughput", 0) * (1 - h["performance"].get("error_rate", 0))
        )
        
        return best_perf["config"]


class BulkSmartBatching:
    """Sistema inteligente de batching adaptativo."""
    
    def __init__(self):
        self.batch_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def calculate_dynamic_batch_size(
        self,
        operation_name: str,
        total_items: int,
        previous_batch_performance: Optional[Dict[str, Any]] = None
    ) -> int:
        """Calcular tamaño de batch dinámico basado en performance."""
        # Batch inicial
        if operation_name not in self.batch_history:
            return min(100, total_items)
        
        history = self.batch_history[operation_name]
        if not history:
            return min(100, total_items)
        
        # Analizar performance previa
        if previous_batch_performance:
            avg_duration = previous_batch_performance.get("avg_duration", 0)
            error_rate = previous_batch_performance.get("error_rate", 0)
            
            # Si es rápido y sin errores, aumentar batch
            if avg_duration < 5.0 and error_rate < 0.05:
                current_batch = history[-1].get("batch_size", 100)
                return min(int(current_batch * 1.2), 1000, total_items)
            
            # Si es lento o con errores, reducir batch
            if avg_duration > 30.0 or error_rate > 0.1:
                current_batch = history[-1].get("batch_size", 100)
                return max(int(current_batch * 0.8), 10)
        
        # Usar último batch exitoso
        last_batch = history[-1].get("batch_size", 100)
        return min(last_batch, total_items)
    
    def record_batch_performance(
        self,
        operation_name: str,
        batch_size: int,
        duration: float,
        success_count: int,
        total_count: int
    ):
        """Registrar performance de batch."""
        if operation_name not in self.batch_history:
            self.batch_history[operation_name] = []
        
        self.batch_history[operation_name].append({
            "timestamp": datetime.now().isoformat(),
            "batch_size": batch_size,
            "duration": duration,
            "success_count": success_count,
            "total_count": total_count,
            "success_rate": success_count / total_count if total_count > 0 else 0
        })
        
        # Mantener solo últimos 50 registros
        if len(self.batch_history[operation_name]) > 50:
            self.batch_history[operation_name] = self.batch_history[operation_name][-50:]


class BulkIntelligentRetry:
    """Sistema inteligente de retry con análisis de errores."""
    
    def __init__(self):
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.retry_strategies: Dict[str, Callable] = {}
    
    def analyze_error(
        self,
        error: Exception,
        operation: str,
        attempt: int
    ) -> Dict[str, Any]:
        """Analizar error y determinar si debe reintentarse."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Patrones de errores que no deben reintentarse
        non_retryable_patterns = [
            "permission denied",
            "authentication failed",
            "invalid input",
            "not found"
        ]
        
        should_retry = True
        retry_delay = 1.0 * (2 ** attempt)  # Backoff exponencial
        
        for pattern in non_retryable_patterns:
            if pattern.lower() in error_msg.lower():
                should_retry = False
                break
        
        # Errores temporales que deben reintentarse
        retryable_patterns = [
            "timeout",
            "connection",
            "temporary",
            "rate limit",
            "server error"
        ]
        
        for pattern in retryable_patterns:
            if pattern.lower() in error_msg.lower():
                should_retry = True
                retry_delay = min(retry_delay * 1.5, 60.0)  # Max 60s
                break
        
        # Registrar patrón de error
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = {
                "count": 0,
                "last_seen": None,
                "retry_success_rate": 0.0
            }
        
        self.error_patterns[error_type]["count"] += 1
        self.error_patterns[error_type]["last_seen"] = datetime.now().isoformat()
        
        return {
            "should_retry": should_retry,
            "retry_delay": retry_delay,
            "error_type": error_type,
            "pattern": self.error_patterns[error_type]
        }
    
    def get_retry_strategy(self, error_type: str) -> Optional[Callable]:
        """Obtener estrategia de retry personalizada."""
        return self.retry_strategies.get(error_type)
    
    def register_retry_strategy(self, error_type: str, strategy: Callable):
        """Registrar estrategia de retry personalizada."""
        self.retry_strategies[error_type] = strategy


class BulkPredictiveScaling:
    """Sistema de escalado predictivo basado en patrones."""
    
    def __init__(self):
        self.demand_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.scaling_history: List[Dict[str, Any]] = []
    
    def predict_demand(
        self,
        operation_name: str,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Predecir demanda futura basado en patrones históricos."""
        if current_time is None:
            current_time = datetime.now()
        
        if operation_name not in self.demand_patterns:
            return {
                "predicted_throughput": 10.0,
                "confidence": 0.0,
                "recommended_workers": 5
            }
        
        patterns = self.demand_patterns[operation_name]
        
        # Analizar patrones por hora del día
        current_hour = current_time.hour
        hour_patterns = [
            p for p in patterns
            if datetime.fromisoformat(p["timestamp"]).hour == current_hour
        ]
        
        if hour_patterns:
            avg_throughput = sum(
                p["throughput"] for p in hour_patterns
            ) / len(hour_patterns)
            
            recommended_workers = max(5, int(avg_throughput / 10))
            
            return {
                "predicted_throughput": avg_throughput,
                "confidence": min(len(hour_patterns) / 10, 1.0),
                "recommended_workers": recommended_workers
            }
        
        return {
            "predicted_throughput": 10.0,
            "confidence": 0.0,
            "recommended_workers": 5
        }
    
    def record_demand(
        self,
        operation_name: str,
        throughput: float,
        workers: int
    ):
        """Registrar demanda actual."""
        if operation_name not in self.demand_patterns:
            self.demand_patterns[operation_name] = []
        
        self.demand_patterns[operation_name].append({
            "timestamp": datetime.now().isoformat(),
            "throughput": throughput,
            "workers": workers
        })
        
        # Mantener solo últimos 1000 registros
        if len(self.demand_patterns[operation_name]) > 1000:
            self.demand_patterns[operation_name] = self.demand_patterns[operation_name][-1000:]
    
    def get_scaling_recommendation(
        self,
        operation_name: str
    ) -> Dict[str, Any]:
        """Obtener recomendación de escalado."""
        prediction = self.predict_demand(operation_name)
        
        return {
            "operation": operation_name,
            "recommended_workers": prediction["recommended_workers"],
            "predicted_throughput": prediction["predicted_throughput"],
            "confidence": prediction["confidence"]
        }


class BulkMLOptimizer:
    """Optimizador basado en Machine Learning."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.training_data: Dict[str, List[Dict[str, Any]]] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
    
    def train_model(
        self,
        operation_name: str,
        features: List[str],
        target: str = "throughput"
    ):
        """Entrenar modelo ML para optimización."""
        if operation_name not in self.training_data:
            logger.warning(f"No training data for {operation_name}")
            return
        
        data = self.training_data[operation_name]
        
        # Feature importance simple (correlación)
        importance = {}
        for feature in features:
            if feature in data[0] if data else {}:
                # Calcular correlación simple
                values = [d.get(feature, 0) for d in data]
                targets = [d.get(target, 0) for d in data]
                
                # Correlación simple
                if len(values) > 1:
                    importance[feature] = abs(sum(
                        (v - sum(values)/len(values)) * (t - sum(targets)/len(targets))
                        for v, t in zip(values, targets)
                    ) / len(values))
        
        self.feature_importance[operation_name] = importance
        logger.info(f"Model trained for {operation_name}")
    
    def predict_optimal_config(
        self,
        operation_name: str,
        current_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predecir configuración óptima usando ML."""
        if operation_name not in self.feature_importance:
            return {"max_workers": 10, "batch_size": 100}
        
        importance = self.feature_importance[operation_name]
        
        # Usar importancia de features para ajustar
        recommended_workers = 10
        recommended_batch = 100
        
        # Ajustar basado en features importantes
        if "throughput" in importance:
            if current_features.get("throughput", 0) < 10:
                recommended_workers = min(50, recommended_workers + 10)
        
        if "error_rate" in importance:
            if current_features.get("error_rate", 0) > 0.1:
                recommended_batch = max(10, int(recommended_batch * 0.7))
        
        return {
            "max_workers": recommended_workers,
            "batch_size": recommended_batch
        }
    
    def add_training_data(
        self,
        operation_name: str,
        features: Dict[str, Any],
        performance: Dict[str, Any]
    ):
        """Agregar datos de entrenamiento."""
        if operation_name not in self.training_data:
            self.training_data[operation_name] = []
        
        data_point = {**features, **performance}
        self.training_data[operation_name].append(data_point)
        
        # Mantener solo últimos 10000 puntos
        if len(self.training_data[operation_name]) > 10000:
            self.training_data[operation_name] = self.training_data[operation_name][-10000:]


class BulkDistributedProcessor:
    """Procesador distribuido para operaciones bulk masivas."""
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.load_balancer: Optional[Callable] = None
    
    def register_node(
        self,
        node_id: str,
        capacity: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """Registrar nodo en el cluster."""
        self.nodes[node_id] = {
            "capacity": capacity,
            "available": capacity,
            "used": 0,
            "config": config or {},
            "status": "active",
            "last_heartbeat": datetime.now()
        }
    
    async def distribute_operation(
        self,
        items: List[Any],
        operation: Callable,
        strategy: str = "round_robin"
    ) -> List[Any]:
        """Distribuir operación entre nodos."""
        if not self.nodes:
            logger.warning("No nodes available")
            return []
        
        active_nodes = [
            nid for nid, node in self.nodes.items()
            if node["status"] == "active"
        ]
        
        if not active_nodes:
            logger.error("No active nodes")
            return []
        
        # Dividir items entre nodos
        if strategy == "round_robin":
            node_items = self._round_robin_distribution(items, active_nodes)
        elif strategy == "capacity_based":
            node_items = self._capacity_based_distribution(items, active_nodes)
        else:
            node_items = self._round_robin_distribution(items, active_nodes)
        
        # Ejecutar en paralelo
        tasks = []
        for node_id, node_item_list in node_items.items():
            if node_item_list:
                task = self._execute_on_node(node_id, node_item_list, operation)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combinar resultados
        all_results = []
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in distributed execution: {result}")
        
        return all_results
    
    def _round_robin_distribution(
        self,
        items: List[Any],
        nodes: List[str]
    ) -> Dict[str, List[Any]]:
        """Distribución round-robin."""
        distribution = {node: [] for node in nodes}
        
        for i, item in enumerate(items):
            node = nodes[i % len(nodes)]
            distribution[node].append(item)
        
        return distribution
    
    def _capacity_based_distribution(
        self,
        items: List[Any],
        nodes: List[str]
    ) -> Dict[str, List[Any]]:
        """Distribución basada en capacidad."""
        total_capacity = sum(self.nodes[n]["capacity"] for n in nodes)
        distribution = {node: [] for node in nodes}
        
        for item in items:
            # Seleccionar nodo con más capacidad disponible
            best_node = max(
                nodes,
                key=lambda n: self.nodes[n]["available"] / self.nodes[n]["capacity"]
            )
            distribution[best_node].append(item)
        
        return distribution
    
    async def _execute_on_node(
        self,
        node_id: str,
        items: List[Any],
        operation: Callable
    ) -> List[Any]:
        """Ejecutar operación en nodo específico."""
        # En una implementación real, esto se comunicaría con el nodo
        # Por ahora, simulamos ejecución local
        results = []
        for item in items:
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(item)
                else:
                    result = operation(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing on node {node_id}: {e}")
                results.append(None)
        
        return results
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Obtener estado del cluster."""
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len([n for n in self.nodes.values() if n["status"] == "active"]),
            "total_capacity": sum(n["capacity"] for n in self.nodes.values()),
            "used_capacity": sum(n["used"] for n in self.nodes.values()),
            "nodes": {
                nid: {
                    "status": node["status"],
                    "capacity": node["capacity"],
                    "available": node["available"],
                    "used": node["used"]
                }
                for nid, node in self.nodes.items()
            }
        }


class BulkPatternAnalyzer:
    """Analizador de patrones para operaciones bulk."""
    
    def __init__(self):
        self.patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.anomalies: List[Dict[str, Any]] = []
    
    def analyze_pattern(
        self,
        operation_name: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analizar patrón en métricas."""
        if operation_name not in self.patterns:
            self.patterns[operation_name] = []
        
        pattern_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.patterns[operation_name].append(pattern_data)
        
        # Mantener solo últimos 1000 patrones
        if len(self.patterns[operation_name]) > 1000:
            self.patterns[operation_name] = self.patterns[operation_name][-1000:]
        
        # Detectar anomalías
        anomaly = self._detect_anomaly(operation_name, metrics)
        if anomaly:
            self.anomalies.append(anomaly)
        
        return {
            "pattern_detected": True,
            "anomaly": anomaly is not None,
            "recommendations": self._generate_recommendations(operation_name, metrics)
        }
    
    def _detect_anomaly(
        self,
        operation_name: str,
        current_metrics: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detectar anomalías en métricas."""
        if operation_name not in self.patterns or len(self.patterns[operation_name]) < 10:
            return None
        
        # Calcular promedio histórico
        historical = self.patterns[operation_name][-100:]  # Últimos 100
        
        if not historical:
            return None
        
        # Comparar throughput
        historical_throughput = [
            h["metrics"].get("throughput", 0) for h in historical
        ]
        
        if historical_throughput:
            avg_throughput = sum(historical_throughput) / len(historical_throughput)
            current_throughput = current_metrics.get("throughput", 0)
            
            # Si el throughput actual es 50% menor que el promedio
            if current_throughput < avg_throughput * 0.5:
                return {
                    "type": "low_throughput",
                    "operation": operation_name,
                    "message": f"Throughput dropped to {current_throughput:.2f} from avg {avg_throughput:.2f}",
                    "severity": "high",
                    "timestamp": datetime.now().isoformat()
                }
        
        return None
    
    def _generate_recommendations(
        self,
        operation_name: str,
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones basadas en patrones."""
        recommendations = []
        
        throughput = metrics.get("throughput", 0)
        error_rate = metrics.get("error_rate", 0)
        duration = metrics.get("duration", 0)
        
        if throughput < 10:
            recommendations.append("Consider increasing max_workers or batch_size")
        
        if error_rate > 0.1:
            recommendations.append("High error rate detected - check error patterns")
        
        if duration > 300:
            recommendations.append("Long operation duration - consider breaking into smaller batches")
        
        return recommendations
    
    def get_pattern_summary(self, operation_name: str) -> Dict[str, Any]:
        """Obtener resumen de patrones."""
        if operation_name not in self.patterns:
            return {}
        
        patterns = self.patterns[operation_name]
        
        if not patterns:
            return {}
        
        # Analizar tendencias
        throughputs = [p["metrics"].get("throughput", 0) for p in patterns]
        error_rates = [p["metrics"].get("error_rate", 0) for p in patterns]
        
        return {
            "operation": operation_name,
            "total_patterns": len(patterns),
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "avg_error_rate": sum(error_rates) / len(error_rates) if error_rates else 0,
            "trend": "increasing" if len(throughputs) > 1 and throughputs[-1] > throughputs[0] else "decreasing",
            "recent_anomalies": len([
                a for a in self.anomalies
                if a.get("operation") == operation_name
            ])
        }


class BulkAutoTuner:
    """Auto-tuner avanzado que optimiza todos los parámetros."""
    
    def __init__(self):
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_configs: Dict[str, Dict[str, Any]] = {}
    
    async def auto_tune(
        self,
        operation_name: str,
        test_operation: Callable,
        parameter_space: Dict[str, List[Any]],
        target_metric: str = "throughput",
        max_iterations: int = 20
    ) -> Dict[str, Any]:
        """Auto-tuning usando búsqueda en espacio de parámetros."""
        best_config = None
        best_score = 0
        
        # Generar configuraciones a probar
        configs = self._generate_configs(parameter_space, max_iterations)
        
        for config in configs:
            try:
                # Ejecutar operación de prueba
                if asyncio.iscoroutinefunction(test_operation):
                    result = await test_operation(config)
                else:
                    result = test_operation(config)
                
                # Evaluar métrica objetivo
                score = result.get(target_metric, 0)
                
                if score > best_score:
                    best_score = score
                    best_config = config
                
                # Registrar en historial
                self.optimization_history.append({
                    "operation": operation_name,
                    "config": config.copy(),
                    "score": score,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error in auto-tuning iteration: {e}")
        
        if best_config:
            self.best_configs[operation_name] = best_config
        
        return {
            "operation": operation_name,
            "best_config": best_config,
            "best_score": best_score,
            "iterations": len(configs)
        }
    
    def _generate_configs(
        self,
        parameter_space: Dict[str, List[Any]],
        max_configs: int
    ) -> List[Dict[str, Any]]:
        """Generar configuraciones del espacio de parámetros."""
        configs = []
        
        # Generar combinaciones
        import itertools
        
        keys = list(parameter_space.keys())
        values = [parameter_space[k] for k in keys]
        
        # Generar todas las combinaciones (limitado)
        all_combinations = list(itertools.product(*values))
        
        # Tomar muestras si hay demasiadas
        if len(all_combinations) > max_configs:
            import random
            configs_tuples = random.sample(all_combinations, max_configs)
        else:
            configs_tuples = all_combinations
        
        # Convertir a diccionarios
        for combo in configs_tuples:
            config = {keys[i]: combo[i] for i in range(len(keys))}
            configs.append(config)
        
        return configs
    
    def get_best_config(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Obtener mejor configuración encontrada."""
        return self.best_configs.get(operation_name)


class BulkLoadBalancer:
    """Balanceador de carga avanzado para operaciones bulk."""
    
    def __init__(self):
        self.backends: Dict[str, Dict[str, Any]] = {}
        self.strategy: str = "round_robin"
        self.current_index: Dict[str, int] = {}
    
    def register_backend(
        self,
        backend_id: str,
        weight: int = 1,
        health_check: Optional[Callable] = None
    ):
        """Registrar backend para balanceo."""
        self.backends[backend_id] = {
            "weight": weight,
            "health_check": health_check,
            "status": "healthy",
            "requests": 0,
            "errors": 0,
            "last_used": None
        }
        self.current_index[backend_id] = 0
    
    def select_backend(self, operation_name: str = "default") -> Optional[str]:
        """Seleccionar backend según estrategia."""
        available_backends = [
            bid for bid, backend in self.backends.items()
            if backend["status"] == "healthy"
        ]
        
        if not available_backends:
            return None
        
        if self.strategy == "round_robin":
            if operation_name not in self.current_index:
                self.current_index[operation_name] = 0
            
            idx = self.current_index[operation_name] % len(available_backends)
            selected = available_backends[idx]
            self.current_index[operation_name] = (idx + 1) % len(available_backends)
            
        elif self.strategy == "least_connections":
            selected = min(
                available_backends,
                key=lambda bid: self.backends[bid]["requests"]
            )
        
        elif self.strategy == "weighted":
            # Weighted round-robin
            total_weight = sum(self.backends[bid]["weight"] for bid in available_backends)
            if operation_name not in self.current_index:
                self.current_index[operation_name] = 0
            
            idx = self.current_index[operation_name] % total_weight
            cumulative = 0
            for bid in available_backends:
                cumulative += self.backends[bid]["weight"]
                if idx < cumulative:
                    selected = bid
                    break
            self.current_index[operation_name] = (idx + 1) % total_weight
        
        else:
            selected = available_backends[0]
        
        # Actualizar estadísticas
        if selected:
            self.backends[selected]["requests"] += 1
            self.backends[selected]["last_used"] = datetime.now()
        
        return selected
    
    def record_error(self, backend_id: str):
        """Registrar error en backend."""
        if backend_id in self.backends:
            self.backends[backend_id]["errors"] += 1
            
            # Marcar como unhealthy si hay muchos errores
            if self.backends[backend_id]["errors"] > 10:
                self.backends[backend_id]["status"] = "unhealthy"
    
    def record_success(self, backend_id: str):
        """Registrar éxito en backend."""
        if backend_id in self.backends:
            # Resetear errores si hay éxito
            if self.backends[backend_id]["errors"] > 0:
                self.backends[backend_id]["errors"] = max(0, self.backends[backend_id]["errors"] - 1)
            
            if self.backends[backend_id]["status"] == "unhealthy" and self.backends[backend_id]["errors"] == 0:
                self.backends[backend_id]["status"] = "healthy"
    
    def get_backend_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de backends."""
        return {
            "total_backends": len(self.backends),
            "healthy_backends": len([b for b in self.backends.values() if b["status"] == "healthy"]),
            "backends": {
                bid: {
                    "status": backend["status"],
                    "weight": backend["weight"],
                    "requests": backend["requests"],
                    "errors": backend["errors"],
                    "last_used": backend["last_used"].isoformat() if backend["last_used"] else None
                }
                for bid, backend in self.backends.items()
            }
        }


class BulkFailoverManager:
    """Gestor de failover y redundancia para operaciones bulk."""
    
    def __init__(self):
        self.primary_operations: Dict[str, Callable] = {}
        self.fallback_operations: Dict[str, List[Callable]] = {}
        self.failover_stats: Dict[str, Dict[str, Any]] = {}
    
    def register_operation(
        self,
        operation_name: str,
        primary: Callable,
        fallbacks: Optional[List[Callable]] = None
    ):
        """Registrar operación con fallbacks."""
        self.primary_operations[operation_name] = primary
        self.fallback_operations[operation_name] = fallbacks or []
        self.failover_stats[operation_name] = {
            "primary_calls": 0,
            "fallback_calls": 0,
            "failures": 0,
            "last_failover": None
        }
    
    async def execute_with_failover(
        self,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Ejecutar operación con failover automático."""
        if operation_name not in self.primary_operations:
            raise ValueError(f"Operation {operation_name} not registered")
        
        stats = self.failover_stats[operation_name]
        
        # Intentar operación primaria
        try:
            primary = self.primary_operations[operation_name]
            if asyncio.iscoroutinefunction(primary):
                result = await primary(*args, **kwargs)
            else:
                result = primary(*args, **kwargs)
            
            stats["primary_calls"] += 1
            return result
        
        except Exception as e:
            logger.warning(f"Primary operation {operation_name} failed: {e}, trying fallbacks")
            stats["failures"] += 1
            
            # Intentar fallbacks
            fallbacks = self.fallback_operations.get(operation_name, [])
            for i, fallback in enumerate(fallbacks):
                try:
                    if asyncio.iscoroutinefunction(fallback):
                        result = await fallback(*args, **kwargs)
                    else:
                        result = fallback(*args, **kwargs)
                    
                    stats["fallback_calls"] += 1
                    stats["last_failover"] = datetime.now().isoformat()
                    logger.info(f"Fallback {i+1} for {operation_name} succeeded")
                    return result
                
                except Exception as fallback_error:
                    logger.error(f"Fallback {i+1} for {operation_name} failed: {fallback_error}")
                    continue
            
            # Todos los fallbacks fallaron
            raise Exception(f"All operations failed for {operation_name}")
    
    def get_failover_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas de failover."""
        if operation_name:
            return self.failover_stats.get(operation_name, {})
        else:
            return {
                "operations": self.failover_stats,
                "total_operations": len(self.failover_stats)
            }


class BulkAdvancedMetrics:
    """Sistema avanzado de métricas con análisis estadístico."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.aggregations: Dict[str, Dict[str, Any]] = {}
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Registrar métrica."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {}
        })
        
        # Mantener solo últimos 10000 registros
        if len(self.metrics[metric_name]) > 10000:
            self.metrics[metric_name] = self.metrics[metric_name][-10000:]
        
        # Actualizar agregaciones
        self._update_aggregations(metric_name)
    
    def _update_aggregations(self, metric_name: str):
        """Actualizar agregaciones estadísticas."""
        if metric_name not in self.metrics:
            return
        
        values = [m["value"] for m in self.metrics[metric_name]]
        
        if not values:
            return
        
        self.aggregations[metric_name] = {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "median": sorted(values)[len(values) // 2] if values else 0,
            "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 0 else 0,
            "p99": sorted(values)[int(len(values) * 0.99)] if len(values) > 0 else 0
        }
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """Obtener estadísticas de métrica."""
        if metric_name not in self.aggregations:
            return {}
        
        return {
            "metric": metric_name,
            "stats": self.aggregations[metric_name],
            "recent_values": [
                m["value"] for m in self.metrics[metric_name][-100:]
            ]
        }
    
    def get_percentile(self, metric_name: str, percentile: float) -> float:
        """Obtener percentil de métrica."""
        if metric_name not in self.metrics:
            return 0.0
        
        values = sorted([m["value"] for m in self.metrics[metric_name]])
        if not values:
            return 0.0
        
        idx = int(len(values) * percentile / 100)
        return values[min(idx, len(values) - 1)]


class BulkSynchronizer:
    """Sistema de sincronización para operaciones bulk distribuidas."""
    
    def __init__(self):
        self.locks: Dict[str, asyncio.Lock] = {}
        self.sync_points: Dict[str, List[asyncio.Event]] = {}
        self.barriers: Dict[str, asyncio.Barrier] = {}
    
    async def acquire_lock(self, resource_id: str, timeout: Optional[float] = None) -> bool:
        """Adquirir lock para recurso."""
        if resource_id not in self.locks:
            self.locks[resource_id] = asyncio.Lock()
        
        try:
            if timeout:
                await asyncio.wait_for(self.locks[resource_id].acquire(), timeout=timeout)
            else:
                await self.locks[resource_id].acquire()
            return True
        except asyncio.TimeoutError:
            return False
    
    async def release_lock(self, resource_id: str):
        """Liberar lock."""
        if resource_id in self.locks:
            self.locks[resource_id].release()
    
    async def wait_at_barrier(
        self,
        barrier_id: str,
        parties: int,
        timeout: Optional[float] = None
    ) -> bool:
        """Esperar en barrier hasta que todos los participantes lleguen."""
        if barrier_id not in self.barriers:
            self.barriers[barrier_id] = asyncio.Barrier(parties)
        
        try:
            if timeout:
                await asyncio.wait_for(
                    self.barriers[barrier_id].wait(),
                    timeout=timeout
                )
            else:
                await self.barriers[barrier_id].wait()
            return True
        except asyncio.TimeoutError:
            return False
    
    def create_sync_point(self, sync_id: str, count: int):
        """Crear punto de sincronización."""
        self.sync_points[sync_id] = [asyncio.Event() for _ in range(count)]
    
    async def signal_sync_point(self, sync_id: str, index: int):
        """Señalar punto de sincronización."""
        if sync_id in self.sync_points and index < len(self.sync_points[sync_id]):
            self.sync_points[sync_id][index].set()
    
    async def wait_for_sync_point(self, sync_id: str, index: int, timeout: Optional[float] = None):
        """Esperar punto de sincronización."""
        if sync_id in self.sync_points and index < len(self.sync_points[sync_id]):
            try:
                if timeout:
                    await asyncio.wait_for(
                        self.sync_points[sync_id][index].wait(),
                        timeout=timeout
                    )
                else:
                    await self.sync_points[sync_id][index].wait()
                return True
            except asyncio.TimeoutError:
                return False
        return False


class BulkReplicator:
    """Sistema de replicación para operaciones bulk."""
    
    def __init__(self):
        self.replicas: Dict[str, List[Callable]] = {}
        self.replication_stats: Dict[str, Dict[str, Any]] = {}
    
    def register_replica(
        self,
        operation_name: str,
        replica_func: Callable,
        priority: int = 1
    ):
        """Registrar función de replicación."""
        if operation_name not in self.replicas:
            self.replicas[operation_name] = []
            self.replication_stats[operation_name] = {
                "total_replications": 0,
                "successful": 0,
                "failed": 0
            }
        
        self.replicas[operation_name].append({
            "func": replica_func,
            "priority": priority
        })
        
        # Ordenar por prioridad
        self.replicas[operation_name].sort(key=lambda x: x["priority"], reverse=True)
    
    async def replicate_operation(
        self,
        operation_name: str,
        data: Any,
        wait_for_all: bool = False
    ) -> List[Any]:
        """Replicar operación en todas las réplicas."""
        if operation_name not in self.replicas:
            return []
        
        replicas = self.replicas[operation_name]
        stats = self.replication_stats[operation_name]
        stats["total_replications"] += 1
        
        results = []
        
        async def replicate_one(replica_info: Dict[str, Any]):
            try:
                func = replica_info["func"]
                if asyncio.iscoroutinefunction(func):
                    result = await func(data)
                else:
                    result = func(data)
                stats["successful"] += 1
                return result
            except Exception as e:
                logger.error(f"Replication failed for {operation_name}: {e}")
                stats["failed"] += 1
                return None
        
        if wait_for_all:
            # Esperar todas las réplicas
            tasks = [replicate_one(r) for r in replicas]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Ejecutar en paralelo sin esperar
            for replica in replicas:
                asyncio.create_task(replicate_one(replica))
        
        return [r for r in results if r is not None and not isinstance(r, Exception)]
    
    def get_replication_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas de replicación."""
        if operation_name:
            return self.replication_stats.get(operation_name, {})
        else:
            return {
                "operations": self.replication_stats,
                "total_operations": len(self.replication_stats)
            }


class BulkAdvancedAnalytics:
    """Sistema avanzado de analytics para operaciones bulk."""
    
    def __init__(self):
        self.analytics_data: Dict[str, List[Dict[str, Any]]] = {}
        self.insights: Dict[str, Dict[str, Any]] = {}
    
    def record_operation_data(
        self,
        operation_name: str,
        data: Dict[str, Any]
    ):
        """Registrar datos de operación."""
        if operation_name not in self.analytics_data:
            self.analytics_data[operation_name] = []
        
        self.analytics_data[operation_name].append({
            **data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mantener solo últimos 5000 registros
        if len(self.analytics_data[operation_name]) > 5000:
            self.analytics_data[operation_name] = self.analytics_data[operation_name][-5000:]
        
        # Generar insights automáticamente
        self._generate_insights(operation_name)
    
    def _generate_insights(self, operation_name: str):
        """Generar insights automáticos."""
        if operation_name not in self.analytics_data:
            return
        
        data = self.analytics_data[operation_name]
        
        if not data:
            return
        
        # Calcular insights
        throughputs = [d.get("throughput", 0) for d in data if "throughput" in d]
        error_rates = [d.get("error_rate", 0) for d in data if "error_rate" in d]
        durations = [d.get("duration", 0) for d in data if "duration" in d]
        
        insights = {
            "operation": operation_name,
            "total_operations": len(data),
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "avg_error_rate": sum(error_rates) / len(error_rates) if error_rates else 0,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "trend": self._calculate_trend(data),
            "recommendations": self._generate_recommendations(data)
        }
        
        self.insights[operation_name] = insights
    
    def _calculate_trend(self, data: List[Dict[str, Any]]) -> str:
        """Calcular tendencia."""
        if len(data) < 2:
            return "stable"
        
        # Comparar primeros y últimos valores
        recent_throughputs = [d.get("throughput", 0) for d in data[-10:] if "throughput" in d]
        older_throughputs = [d.get("throughput", 0) for d in data[:10] if "throughput" in d]
        
        if not recent_throughputs or not older_throughputs:
            return "stable"
        
        recent_avg = sum(recent_throughputs) / len(recent_throughputs)
        older_avg = sum(older_throughputs) / len(older_throughputs)
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "degrading"
        else:
            return "stable"
    
    def _generate_recommendations(self, data: List[Dict[str, Any]]) -> List[str]:
        """Generar recomendaciones."""
        recommendations = []
        
        if not data:
            return recommendations
        
        # Analizar últimos datos
        recent = data[-100:] if len(data) > 100 else data
        
        avg_error_rate = sum(d.get("error_rate", 0) for d in recent) / len(recent)
        avg_throughput = sum(d.get("throughput", 0) for d in recent) / len(recent)
        
        if avg_error_rate > 0.1:
            recommendations.append("High error rate detected - investigate error patterns")
        
        if avg_throughput < 10:
            recommendations.append("Low throughput - consider increasing workers or batch size")
        
        return recommendations
    
    def get_insights(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Obtener insights."""
        if operation_name:
            return self.insights.get(operation_name, {})
        else:
            return {
                "operations": self.insights,
                "total_operations": len(self.insights)
            }


class BulkOptimizationEngine:
    """Motor de optimización avanzado para operaciones bulk."""
    
    def __init__(self):
        self.optimization_rules: Dict[str, List[Callable]] = {}
        self.optimization_history: List[Dict[str, Any]] = {}
    
    def register_optimization_rule(
        self,
        rule_name: str,
        rule_func: Callable,
        priority: int = 1
    ):
        """Registrar regla de optimización."""
        if rule_name not in self.optimization_rules:
            self.optimization_rules[rule_name] = []
        
        self.optimization_rules[rule_name].append({
            "func": rule_func,
            "priority": priority
        })
        
        # Ordenar por prioridad
        self.optimization_rules[rule_name].sort(key=lambda x: x["priority"], reverse=True)
    
    async def optimize_operation(
        self,
        operation_name: str,
        current_config: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimizar operación aplicando reglas."""
        if operation_name not in self.optimization_rules:
            return current_config
        
        optimized_config = current_config.copy()
        applied_rules = []
        
        # Aplicar reglas en orden de prioridad
        for rule_info in self.optimization_rules[operation_name]:
            try:
                rule_func = rule_info["func"]
                if asyncio.iscoroutinefunction(rule_func):
                    result = await rule_func(optimized_config, performance_data)
                else:
                    result = rule_func(optimized_config, performance_data)
                
                if result:
                    optimized_config.update(result)
                    applied_rules.append(rule_info["func"].__name__ if hasattr(rule_info["func"], '__name__') else "unknown")
            
            except Exception as e:
                logger.error(f"Error applying optimization rule: {e}")
        
        # Registrar optimización
        self.optimization_history.append({
            "operation": operation_name,
            "original_config": current_config,
            "optimized_config": optimized_config,
            "applied_rules": applied_rules,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mantener solo últimos 1000 optimizaciones
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
        
        return optimized_config
    
    def get_optimization_history(self, operation_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtener historial de optimizaciones."""
        if operation_name:
            return [
                h for h in self.optimization_history
                if h["operation"] == operation_name
            ]
        else:
            return self.optimization_history


class BulkEventBus:
    """Bus de eventos para operaciones bulk."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict[str, Any]] = []
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable,
        priority: int = 1
    ):
        """Suscribirse a evento."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append({
            "handler": handler,
            "priority": priority
        })
        
        # Ordenar por prioridad
        self.subscribers[event_type].sort(key=lambda x: x["priority"], reverse=True)
    
    async def publish(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Publicar evento."""
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Guardar en historial
        self.event_history.append(event)
        
        # Mantener solo últimos 10000 eventos
        if len(self.event_history) > 10000:
            self.event_history = self.event_history[-10000:]
        
        # Notificar suscriptores
        if event_type in self.subscribers:
            tasks = []
            for subscriber_info in self.subscribers[event_type]:
                handler = subscriber_info["handler"]
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event_data))
                    else:
                        handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener historial de eventos."""
        if event_type:
            events = [
                e for e in self.event_history
                if e["type"] == event_type
            ]
        else:
            events = self.event_history
        
        return events[-limit:]


class BulkDataValidator:
    """Validador avanzado de datos para operaciones bulk."""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.validation_cache: Dict[str, bool] = {}
    
    def register_validation_rule(
        self,
        data_type: str,
        rule_func: Callable,
        priority: int = 1
    ):
        """Registrar regla de validación."""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = []
        
        self.validation_rules[data_type].append({
            "func": rule_func,
            "priority": priority
        })
        
        # Ordenar por prioridad
        self.validation_rules[data_type].sort(key=lambda x: x["priority"], reverse=True)
    
    async def validate_data(
        self,
        data_type: str,
        data: Any,
        use_cache: bool = True
    ) -> Tuple[bool, List[str]]:
        """Validar datos aplicando todas las reglas."""
        # Verificar cache
        cache_key = f"{data_type}:{str(data)}"
        if use_cache and cache_key in self.validation_cache:
            return self.validation_cache[cache_key], []
        
        if data_type not in self.validation_rules:
            return True, []  # Sin reglas, asumir válido
        
        errors = []
        
        # Aplicar reglas en orden de prioridad
        for rule_info in self.validation_rules[data_type]:
            try:
                rule_func = rule_info["func"]
                if asyncio.iscoroutinefunction(rule_func):
                    result = await rule_func(data)
                else:
                    result = rule_func(data)
                
                if isinstance(result, tuple):
                    is_valid, error_msg = result
                    if not is_valid:
                        errors.append(error_msg)
                elif not result:
                    errors.append(f"Validation failed for {data_type}")
            
            except Exception as e:
                logger.error(f"Error in validation rule: {e}")
                errors.append(f"Validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        
        # Guardar en cache
        if use_cache:
            self.validation_cache[cache_key] = (is_valid, errors)
            # Limpiar cache si es muy grande
            if len(self.validation_cache) > 10000:
                # Eliminar 20% más antiguos
                keys_to_delete = list(self.validation_cache.keys())[:2000]
                for key in keys_to_delete:
                    del self.validation_cache[key]
        
        return is_valid, errors
    
    def clear_cache(self):
        """Limpiar cache de validación."""
        self.validation_cache.clear()


class BulkDataTransformer:
    """Transformador avanzado de datos para operaciones bulk."""
    
    def __init__(self):
        self.transformers: Dict[str, List[Callable]] = {}
        self.transformation_cache: Dict[str, Any] = {}
    
    def register_transformer(
        self,
        transformation_name: str,
        transformer_func: Callable,
        priority: int = 1
    ):
        """Registrar transformador."""
        if transformation_name not in self.transformers:
            self.transformers[transformation_name] = []
        
        self.transformers[transformation_name].append({
            "func": transformer_func,
            "priority": priority
        })
        
        # Ordenar por prioridad
        self.transformers[transformation_name].sort(key=lambda x: x["priority"], reverse=True)
    
    async def transform_data(
        self,
        transformation_name: str,
        data: Any,
        use_cache: bool = True
    ) -> Any:
        """Transformar datos aplicando transformadores."""
        # Verificar cache
        cache_key = f"{transformation_name}:{str(data)}"
        if use_cache and cache_key in self.transformation_cache:
            return self.transformation_cache[cache_key]
        
        if transformation_name not in self.transformers:
            return data  # Sin transformadores, retornar original
        
        transformed_data = data
        
        # Aplicar transformadores en orden de prioridad
        for transformer_info in self.transformers[transformation_name]:
            try:
                transformer_func = transformer_info["func"]
                if asyncio.iscoroutinefunction(transformer_func):
                    transformed_data = await transformer_func(transformed_data)
                else:
                    transformed_data = transformer_func(transformed_data)
            
            except Exception as e:
                logger.error(f"Error in transformer: {e}")
                break  # Detener transformación si hay error
        
        # Guardar en cache
        if use_cache:
            self.transformation_cache[cache_key] = transformed_data
            # Limpiar cache si es muy grande
            if len(self.transformation_cache) > 10000:
                keys_to_delete = list(self.transformation_cache.keys())[:2000]
                for key in keys_to_delete:
                    del self.transformation_cache[key]
        
        return transformed_data
    
    def clear_cache(self):
        """Limpiar cache de transformaciones."""
        self.transformation_cache.clear()


class BulkRouter:
    """Enrutador inteligente para operaciones bulk."""
    
    def __init__(self):
        self.routes: Dict[str, Callable] = {}
        self.routing_stats: Dict[str, Dict[str, Any]] = {}
        self.routing_rules: List[Dict[str, Any]] = []
    
    def register_route(
        self,
        route_pattern: str,
        handler: Callable,
        priority: int = 1
    ):
        """Registrar ruta."""
        self.routes[route_pattern] = {
            "handler": handler,
            "priority": priority
        }
        
        if route_pattern not in self.routing_stats:
            self.routing_stats[route_pattern] = {
                "requests": 0,
                "success": 0,
                "errors": 0,
                "avg_duration": 0.0
            }
    
    def add_routing_rule(
        self,
        condition: Callable,
        target_route: str,
        priority: int = 1
    ):
        """Agregar regla de enrutamiento."""
        self.routing_rules.append({
            "condition": condition,
            "target": target_route,
            "priority": priority
        })
        
        # Ordenar por prioridad
        self.routing_rules.sort(key=lambda x: x["priority"], reverse=True)
    
    async def route(
        self,
        operation_name: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Enrutar operación según reglas."""
        context = context or {}
        
        # Buscar regla de enrutamiento que coincida
        target_route = None
        for rule in self.routing_rules:
            try:
                condition = rule["condition"]
                if asyncio.iscoroutinefunction(condition):
                    matches = await condition(operation_name, data, context)
                else:
                    matches = condition(operation_name, data, context)
                
                if matches:
                    target_route = rule["target"]
                    break
            except Exception as e:
                logger.error(f"Error evaluating routing rule: {e}")
        
        # Si no hay regla, usar ruta por defecto
        if not target_route:
            target_route = operation_name
        
        # Obtener handler
        if target_route not in self.routes:
            raise ValueError(f"No route found for {target_route}")
        
        handler_info = self.routes[target_route]
        handler = handler_info["handler"]
        stats = self.routing_stats[target_route]
        
        # Ejecutar handler
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(data, context)
            else:
                result = handler(data, context)
            
            stats["success"] += 1
            stats["requests"] += 1
            
            duration = time.time() - start_time
            stats["avg_duration"] = (
                (stats["avg_duration"] * (stats["requests"] - 1) + duration) / stats["requests"]
            )
            
            return result
        
        except Exception as e:
            stats["errors"] += 1
            stats["requests"] += 1
            logger.error(f"Error in route {target_route}: {e}")
            raise
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de enrutamiento."""
        return {
            "routes": self.routing_stats,
            "total_routes": len(self.routes),
            "total_rules": len(self.routing_rules)
        }


class BulkCompressionAdvanced:
    """Sistema avanzado de compresión para operaciones bulk."""
    
    def __init__(self):
        self.compression_methods: Dict[str, Callable] = {
            "gzip": self._compress_gzip,
            "lzma": self._compress_lzma,
            "bzip2": self._compress_bzip2
        }
        self.compression_stats: Dict[str, Dict[str, Any]] = {}
    
    def _compress_gzip(self, data: bytes) -> bytes:
        """Comprimir con gzip."""
        import gzip
        return gzip.compress(data)
    
    def _compress_lzma(self, data: bytes) -> bytes:
        """Comprimir con lzma."""
        import lzma
        return lzma.compress(data)
    
    def _compress_bzip2(self, data: bytes) -> bytes:
        """Comprimir con bzip2."""
        import bz2
        return bz2.compress(data)
    
    async def compress(
        self,
        data: Any,
        method: str = "gzip",
        format: str = "bytes"
    ) -> Any:
        """Comprimir datos."""
        # Convertir a bytes si es necesario
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict) or isinstance(data, list):
            import json
            data_bytes = json.dumps(data).encode('utf-8')
        else:
            data_bytes = bytes(data)
        
        # Comprimir
        if method not in self.compression_methods:
            method = "gzip"
        
        compressor = self.compression_methods[method]
        compressed = compressor(data_bytes)
        
        # Registrar estadísticas
        if method not in self.compression_stats:
            self.compression_stats[method] = {
                "compressions": 0,
                "total_original_size": 0,
                "total_compressed_size": 0
            }
        
        stats = self.compression_stats[method]
        stats["compressions"] += 1
        stats["total_original_size"] += len(data_bytes)
        stats["total_compressed_size"] += len(compressed)
        
        if format == "base64":
            import base64
            return base64.b64encode(compressed).decode('utf-8')
        
        return compressed
    
    async def decompress(
        self,
        compressed_data: Any,
        method: str = "gzip",
        format: str = "bytes"
    ) -> bytes:
        """Descomprimir datos."""
        # Convertir desde base64 si es necesario
        if format == "base64":
            import base64
            compressed_data = base64.b64decode(compressed_data)
        
        # Descomprimir
        if method == "gzip":
            import gzip
            return gzip.decompress(compressed_data)
        elif method == "lzma":
            import lzma
            return lzma.decompress(compressed_data)
        elif method == "bzip2":
            import bz2
            return bz2.decompress(compressed_data)
        else:
            return compressed_data
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de compresión."""
        stats_summary = {}
        for method, stats in self.compression_stats.items():
            if stats["compressions"] > 0:
                avg_ratio = (
                    (1 - stats["total_compressed_size"] / stats["total_original_size"]) * 100
                    if stats["total_original_size"] > 0 else 0
                )
                stats_summary[method] = {
                    **stats,
                    "avg_compression_ratio": avg_ratio
                }
            else:
                stats_summary[method] = stats
        
        return stats_summary


class BulkSecurityAdvanced:
    """Sistema avanzado de seguridad para operaciones bulk."""
    
    def __init__(self):
        self.encryption_keys: Dict[str, bytes] = {}
        self.access_control: Dict[str, List[str]] = {}
        self.audit_log: List[Dict[str, Any]] = []
    
    def generate_key(self, key_id: str) -> bytes:
        """Generar clave de encriptación."""
        import secrets
        key = secrets.token_bytes(32)
        self.encryption_keys[key_id] = key
        return key
    
    async def encrypt_data(
        self,
        data: bytes,
        key_id: str
    ) -> bytes:
        """Encriptar datos."""
        if key_id not in self.encryption_keys:
            raise ValueError(f"Key {key_id} not found")
        
        key = self.encryption_keys[key_id]
        
        # Usar Fernet para encriptación simétrica simple
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'bulk_chat_salt',
                iterations=100000,
            )
            key_derived = base64.urlsafe_b64encode(kdf.derive(key))
            fernet = Fernet(key_derived)
            return fernet.encrypt(data)
        except ImportError:
            # Fallback simple si no hay cryptography
            logger.warning("cryptography not available, using simple XOR")
            return bytes(a ^ b for a, b in zip(data, key * (len(data) // len(key) + 1)))
    
    async def decrypt_data(
        self,
        encrypted_data: bytes,
        key_id: str
    ) -> bytes:
        """Desencriptar datos."""
        if key_id not in self.encryption_keys:
            raise ValueError(f"Key {key_id} not found")
        
        key = self.encryption_keys[key_id]
        
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'bulk_chat_salt',
                iterations=100000,
            )
            key_derived = base64.urlsafe_b64encode(kdf.derive(key))
            fernet = Fernet(key_derived)
            return fernet.decrypt(encrypted_data)
        except ImportError:
            # Fallback simple
            logger.warning("cryptography not available, using simple XOR")
            return bytes(a ^ b for a, b in zip(encrypted_data, key * (len(encrypted_data) // len(key) + 1)))
    
    def grant_access(self, user_id: str, resource: str):
        """Otorgar acceso a recurso."""
        if user_id not in self.access_control:
            self.access_control[user_id] = []
        
        if resource not in self.access_control[user_id]:
            self.access_control[user_id].append(resource)
    
    def revoke_access(self, user_id: str, resource: str):
        """Revocar acceso a recurso."""
        if user_id in self.access_control:
            if resource in self.access_control[user_id]:
                self.access_control[user_id].remove(resource)
    
    def check_access(self, user_id: str, resource: str) -> bool:
        """Verificar acceso a recurso."""
        if user_id in self.access_control:
            return "*" in self.access_control[user_id] or resource in self.access_control[user_id]
        return False
    
    def audit_log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool
    ):
        """Registrar acción en audit log."""
        self.audit_log.append({
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mantener solo últimos 50000 registros
        if len(self.audit_log) > 50000:
            self.audit_log = self.audit_log[-50000:]
    
    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Obtener audit log."""
        if user_id:
            logs = [log for log in self.audit_log if log["user_id"] == user_id]
        else:
            logs = self.audit_log
        
        return logs[-limit:]

