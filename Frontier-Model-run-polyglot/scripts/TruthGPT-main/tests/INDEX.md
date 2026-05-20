# Índice de Archivos - TruthGPT Test Suite

Este documento proporciona un índice completo de todos los archivos organizados en la suite de tests.

## 📁 Estructura Completa

### Core Tests (`core/`)

#### Unit Tests (`core/unit/`)
- `test_core.py` - Tests de componentes core
- `test_models.py` - Tests de gestión de modelos
- `test_training.py` - Tests de sistema de entrenamiento
- `test_inference.py` - Tests de motor de inferencia
- `test_optimization.py` - Tests de optimización
- `test_optimizer.py` - Tests de optimizador

#### Integration Tests (`core/integration/`)
- `test_integration.py` - Tests de integración end-to-end
- `test_api.py` - Tests de API

#### Fixtures (`core/fixtures/`)
- `test_fixtures.py` - Fixtures de pytest
- `test_helpers.py` - Helpers y decoradores
- `test_utils.py` - Utilidades compartidas

#### Otros Tests Core
- `test_assertions.py` - Tests de aserciones
- `test_backup.py` - Tests de backup
- `test_benchmarks.py` - Tests de benchmarks
- `test_cache.py` - Tests de caché
- `test_clustering.py` - Tests de clustering
- `test_comparator.py` - Tests de comparación
- `test_compatibility.py` - Tests de compatibilidad
- `test_coverage.py` - Tests de cobertura
- `test_data_generators.py` - Generadores de datos de prueba
- `test_database.py` - Tests de base de datos
- `test_dependency_analyzer.py` - Tests de análisis de dependencias
- `test_diff.py` - Tests de diferencias
- `test_edge_cases.py` - Tests de casos límite
- `test_flakiness_detector.py` - Tests de detección de flakiness
- `test_history.py` - Tests de historial
- `test_insights.py` - Tests de insights
- `test_metrics.py` - Tests de métricas
- `test_monitoring.py` - Tests de monitoreo
- `test_notifier.py` - Tests de notificaciones
- `test_performance.py` - Tests de rendimiento
- `test_profiler.py` - Tests de profiling
- `test_regression.py` - Tests de regresión
- `test_scheduler.py` - Tests de scheduler
- `test_search.py` - Tests de búsqueda
- `test_security.py` - Tests de seguridad
- `test_tagging.py` - Tests de etiquetado
- `test_timeline.py` - Tests de timeline
- `test_validation.py` - Tests de validación

### Analyzers (`analyzers/`)

#### General (`analyzers/general/`)
- `complete_analyzer.py` - Analizador completo
- `enhanced_analyzer.py` - Analizador mejorado
- `pattern_analyzer.py` - Analizador de patrones

#### Cost Analysis (`analyzers/cost/`)
- `cost_analyzer_advanced.py` - Analizador avanzado de costos
- `cost_analyzer_enhanced.py` - Analizador mejorado de costos
- `cost_calculator.py` - Calculadora de costos

#### Performance Analysis (`analyzers/performance/`)
- `efficiency_analyzer.py` - Analizador de eficiencia
- `performance_analyzer_advanced.py` - Analizador avanzado de rendimiento
- `performance_analyzer_enhanced.py` - Analizador mejorado de rendimiento
- `performance_analyzer.py` - Analizador de rendimiento base
- `performance_regression_detector.py` - Detector de regresiones de rendimiento
- `resource_efficiency_analyzer.py` - Analizador de eficiencia de recursos
- `resource_optimizer.py` - Optimizador de recursos

#### Quality Analysis (`analyzers/quality/`)
- `quality_analyzer_advanced.py` - Analizador avanzado de calidad
- `quality_analyzer_enhanced.py` - Analizador mejorado de calidad
- `quality_assessor.py` - Evaluador de calidad
- `quality_gates_advanced.py` - Quality gates avanzados
- `quality_gates.py` - Quality gates base

#### Security Analysis (`analyzers/security/`)
- `security_analyzer_enhanced.py` - Analizador mejorado de seguridad
- `security_analyzer.py` - Analizador de seguridad base

#### Compliance (`analyzers/compliance/`)
- `compliance_checker_enhanced.py` - Verificador mejorado de cumplimiento
- `compliance_checker.py` - Verificador de cumplimiento base

#### Coverage (`analyzers/coverage/`)
- `coverage_analyzer_advanced.py` - Analizador avanzado de cobertura
- `coverage_analyzer_enhanced.py` - Analizador mejorado de cobertura

#### Dependency (`analyzers/dependency/`)
- `dependency_analyzer_advanced.py` - Analizador avanzado de dependencias
- `dependency_analyzer_enhanced.py` - Analizador mejorado de dependencias
- `dependency_visualizer.py` - Visualizador de dependencias

#### Trend Analysis (`analyzers/trend/`)
- `trend_analyzer_advanced.py` - Analizador avanzado de tendencias
- `trend_analyzer_enhanced.py` - Analizador mejorado de tendencias
- `trend_forecaster.py` - Pronosticador de tendencias
- `trend_predictor_advanced.py` - Predictor avanzado de tendencias

#### Flakiness (`analyzers/flakiness/`)
- `flakiness_analyzer_advanced.py` - Analizador avanzado de flakiness

#### Regression (`analyzers/regression/`)
- `regression_analyzer_advanced.py` - Analizador avanzado de regresión

#### Optimization (`analyzers/optimization/`)
- `advanced_optimizer.py` - Optimizador avanzado
- `change_impact_analyzer.py` - Analizador de impacto de cambios
- `impact_analyzer.py` - Analizador de impacto
- `optimization_analyzer_enhanced.py` - Analizador mejorado de optimización

### Systems (`systems/`)
- `anomaly_detector.py` - Detector de anomalías
- `audit_system.py` - Sistema de auditoría
- `benchmark_comparator.py` - Comparador de benchmarks
- `benchmarking.py` - Sistema de benchmarking
- `business_metrics_enhanced.py` - Métricas de negocio mejoradas
- `business_metrics.py` - Métricas de negocio base
- `consistency_checker.py` - Verificador de consistencia
- `correlation_analyzer_advanced.py` - Analizador avanzado de correlación
- `correlation_analyzer.py` - Analizador de correlación base
- `failure_pattern_analyzer_advanced.py` - Analizador avanzado de patrones de fallo
- `failure_predictor_advanced.py` - Predictor avanzado de fallos
- `health_scorer.py` - Evaluador de salud
- `improved_predictor.py` - Predictor mejorado
- `maturity_model.py` - Modelo de madurez
- `metrics_analyzer_advanced.py` - Analizador avanzado de métricas
- `metrics_system.py` - Sistema de métricas
- `prediction_system_advanced.py` - Sistema avanzado de predicción
- `realtime_metrics.py` - Métricas en tiempo real
- `recommendation_engine_advanced.py` - Motor avanzado de recomendaciones
- `recommendation_engine.py` - Motor de recomendaciones base
- `recommender_system.py` - Sistema de recomendación
- `reliability_monitor.py` - Monitor de confiabilidad
- `risk_assessor.py` - Evaluador de riesgos
- `roi_calculator.py` - Calculadora de ROI
- `stability_analyzer.py` - Analizador de estabilidad
- `test_predictor.py` - Tests de predictor

### Reporters (`reporters/`)
- `comparative_reporter.py` - Reportero comparativo
- `comprehensive_reporter.py` - Reportero comprehensivo
- `executive_dashboard.py` - Dashboard ejecutivo
- `executive_report_advanced.py` - Reporte ejecutivo avanzado
- `executive_report.py` - Reporte ejecutivo base
- `html_report_generator.py` - Generador de reportes HTML
- `summary_generator.py` - Generador de resúmenes

### Exporters (`exporters/`)
- `advanced_exporter.py` - Exportador avanzado
- `junit_exporter.py` - Exportador JUnit
- `pdf_exporter.py` - Exportador PDF
- `test_exporter.py` - Tests de exportador
- `universal_exporter.py` - Exportador universal

### Utilities (`utilities/`)

#### Integration (`utilities/integration/`)
- `advanced_alerts.py` - Alertas avanzadas
- `intelligent_alerts.py` - Alertas inteligentes
- `realtime_dashboard.py` - Dashboard en tiempo real
- `slack_integration.py` - Integración con Slack
- `test_alerts.py` - Tests de alertas
- `test_dashboard.py` - Tests de dashboard

#### Results (`utilities/results/`)
- `advanced_comparer.py` - Comparador avanzado
- `advanced_statistics.py` - Estadísticas avanzadas
- `enhanced_validator.py` - Validador mejorado
- `intelligent_merger.py` - Fusionador inteligente
- `result_aggregator.py` - Agregador de resultados
- `result_deduplicator.py` - Desduplicador de resultados
- `result_filter.py` - Filtro de resultados
- `result_merger.py` - Fusionador de resultados
- `result_normalizer.py` - Normalizador de resultados
- `result_sampler.py` - Muestreador de resultados
- `result_sorter.py` - Ordenador de resultados
- `result_splitter.py` - Divisor de resultados
- `result_transformer.py` - Transformador de resultados
- `result_validator.py` - Validador de resultados
- `statistics_aggregator.py` - Agregador de estadísticas
- `version_comparator.py` - Comparador de versiones

## 📊 Estadísticas

- **Total de archivos**: 150+
- **Tests core**: 40+
- **Analyzers**: 50+
- **Systems**: 25+
- **Reporters**: 7
- **Exporters**: 5
- **Utilities**: 24+

## 🔍 Búsqueda Rápida

### Por Tipo
- **Tests**: Buscar en `core/`
- **Analizadores**: Buscar en `analyzers/`
- **Sistemas**: Buscar en `systems/`
- **Reportes**: Buscar en `reporters/`
- **Exportación**: Buscar en `exporters/`

### Por Funcionalidad
- **Costos**: `analyzers/cost/`
- **Rendimiento**: `analyzers/performance/`
- **Calidad**: `analyzers/quality/`
- **Seguridad**: `analyzers/security/`
- **Predicción**: `systems/` (prediction_system*)
- **Métricas**: `systems/` (metrics*, business_metrics*)

## 📝 Notas

- Todos los archivos tienen `__init__.py` para ser paquetes Python válidos
- Los tests pueden ejecutarse individualmente o por categoría usando `run_tests.py`
- Ver `README.md` para más información sobre uso y mejores prácticas

