"""
Advanced System Demo for AI History Comparison
Demostración completa del sistema avanzado de análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import os

# Agregar el directorio padre al path para importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar todos los sistemas avanzados
from ai_optimizer import AIOptimizer, ModelType, OptimizationGoal
from emotion_analyzer import AdvancedEmotionAnalyzer, EmotionType
from temporal_analyzer import AdvancedTemporalAnalyzer, TrendType
from content_quality_analyzer import AdvancedContentQualityAnalyzer, ContentType, QualityLevel
from behavior_pattern_analyzer import AdvancedBehaviorPatternAnalyzer, BehaviorType
from performance_optimizer import AdvancedPerformanceOptimizer, PerformanceLevel
from security_analyzer import AdvancedSecurityAnalyzer, SecurityLevel
from advanced_orchestrator import AdvancedOrchestrator, AnalysisType, IntegrationLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSystemDemo:
    """
    Demostración completa del sistema avanzado
    """
    
    def __init__(self):
        self.orchestrator = AdvancedOrchestrator(
            enable_parallel_processing=True,
            enable_auto_optimization=True,
            enable_real_time_monitoring=True,
            max_concurrent_analyses=3
        )
        
        # Datos de ejemplo
        self.sample_documents = self._create_sample_documents()
        self.sample_temporal_data = self._create_sample_temporal_data()
        self.sample_behavior_data = self._create_sample_behavior_data()
    
    def _create_sample_documents(self) -> List[Dict[str, Any]]:
        """Crear documentos de ejemplo"""
        return [
            {
                "id": "doc_001",
                "text": """
                # Análisis de Tendencias en IA 2024
                
                La inteligencia artificial ha experimentado un crecimiento exponencial en los últimos años. 
                Los modelos de lenguaje como GPT-4 y Claude han revolucionado la forma en que interactuamos 
                con la tecnología.
                
                ## Principales Avances
                
                - **Procesamiento de lenguaje natural**: Mejoras significativas en comprensión y generación
                - **Visión por computadora**: Detección de objetos más precisa
                - **Robótica**: Mayor autonomía en tareas complejas
                
                ## Impacto en la Sociedad
                
                La IA está transformando múltiples industrias, desde la salud hasta la educación. 
                Sin embargo, también plantea desafíos éticos importantes que debemos abordar.
                
                ### Consideraciones Éticas
                
                1. Privacidad de datos
                2. Sesgos algorítmicos
                3. Transparencia en decisiones automatizadas
                
                En conclusión, la IA representa tanto oportunidades como desafíos que requieren 
                una aproximación cuidadosa y responsable.
                """,
                "metadata": {
                    "author": "Dr. Jane Smith",
                    "date": "2024-01-15",
                    "category": "technology",
                    "word_count": 150
                }
            },
            {
                "id": "doc_002",
                "text": """
                # Reporte de Rendimiento del Sistema
                
                ## Resumen Ejecutivo
                
                El sistema ha mostrado un rendimiento excepcional durante el último trimestre. 
                Los indicadores clave han superado las expectativas establecidas.
                
                ### Métricas Principales
                
                - **Tiempo de respuesta**: 1.2 segundos (objetivo: <2s) ✅
                - **Disponibilidad**: 99.9% (objetivo: >99.5%) ✅
                - **Satisfacción del usuario**: 4.8/5 (objetivo: >4.5) ✅
                
                ## Análisis Detallado
                
                ### Fortalezas Identificadas
                
                1. **Escalabilidad**: El sistema maneja eficientemente picos de tráfico
                2. **Confiabilidad**: Mínimos tiempos de inactividad
                3. **Usabilidad**: Interfaz intuitiva y fácil de usar
                
                ### Áreas de Mejora
                
                1. **Optimización de consultas**: Reducir latencia en operaciones complejas
                2. **Caché inteligente**: Implementar estrategias de caché más sofisticadas
                3. **Monitoreo proactivo**: Mejorar detección temprana de problemas
                
                ## Recomendaciones
                
                Para mantener el alto rendimiento, recomendamos:
                
                - Implementar análisis predictivo
                - Expandir capacidades de monitoreo
                - Invertir en optimización de algoritmos
                
                El futuro se ve prometedor con estas mejoras planificadas.
                """,
                "metadata": {
                    "author": "Ing. Carlos Rodriguez",
                    "date": "2024-01-20",
                    "category": "performance",
                    "word_count": 200
                }
            },
            {
                "id": "doc_003",
                "text": """
                # Investigación sobre Patrones de Comportamiento
                
                ## Introducción
                
                Este estudio examina los patrones de comportamiento de usuarios en plataformas digitales. 
                Los resultados son fascinantes y reveladores.
                
                ## Metodología
                
                Utilizamos técnicas avanzadas de análisis de datos y machine learning para identificar 
                patrones complejos en el comportamiento de los usuarios.
                
                ### Datos Analizados
                
                - 1,000,000 de interacciones de usuarios
                - 50,000 usuarios únicos
                - Período de estudio: 6 meses
                
                ## Hallazgos Principales
                
                ### Patrones Identificados
                
                1. **Comportamiento Circadiano**: Los usuarios muestran patrones claros de actividad
                2. **Influencia Social**: Las interacciones grupales afectan el comportamiento individual
                3. **Adaptación Temporal**: Los usuarios se adaptan a nuevas funcionalidades rápidamente
                
                ### Implicaciones
                
                Estos hallazgos tienen implicaciones significativas para:
                
                - Diseño de interfaces de usuario
                - Estrategias de engagement
                - Personalización de experiencias
                
                ## Conclusiones
                
                El comportamiento humano en entornos digitales sigue patrones predecibles pero complejos. 
                Comprender estos patrones es crucial para crear experiencias más efectivas.
                
                ### Próximos Pasos
                
                1. Validar hallazgos con estudios adicionales
                2. Desarrollar modelos predictivos
                3. Implementar mejoras basadas en insights
                """,
                "metadata": {
                    "author": "Dra. Maria Gonzalez",
                    "date": "2024-01-25",
                    "category": "research",
                    "word_count": 180
                }
            }
        ]
    
    def _create_sample_temporal_data(self) -> Dict[str, List]:
        """Crear datos temporales de ejemplo"""
        from temporal_analyzer import TemporalPoint
        
        # Generar datos de ejemplo para diferentes métricas
        base_time = datetime.now() - timedelta(days=30)
        
        # Métrica 1: Calidad de contenido
        quality_data = []
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            # Simular tendencia creciente con variación
            value = 0.6 + (i * 0.01) + np.random.normal(0, 0.05)
            quality_data.append(TemporalPoint(
                timestamp=timestamp,
                value=max(0, min(1, value)),
                confidence=0.9
            ))
        
        # Métrica 2: Tiempo de respuesta
        response_time_data = []
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            # Simular tendencia decreciente (mejora)
            value = 3.0 - (i * 0.05) + np.random.normal(0, 0.2)
            response_time_data.append(TemporalPoint(
                timestamp=timestamp,
                value=max(0.1, value),
                confidence=0.8
            ))
        
        # Métrica 3: Satisfacción del usuario
        satisfaction_data = []
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            # Simular estabilidad con pequeñas variaciones
            value = 4.5 + np.random.normal(0, 0.3)
            satisfaction_data.append(TemporalPoint(
                timestamp=timestamp,
                value=max(1, min(5, value)),
                confidence=0.85
            ))
        
        return {
            "content_quality": quality_data,
            "response_time": response_time_data,
            "user_satisfaction": satisfaction_data
        }
    
    def _create_sample_behavior_data(self) -> Dict[str, List]:
        """Crear datos de comportamiento de ejemplo"""
        from behavior_pattern_analyzer import BehaviorMetric
        
        # Generar datos de comportamiento para diferentes entidades
        base_time = datetime.now() - timedelta(hours=24)
        
        # Entidad 1: Usuario tipo A
        user_a_data = []
        for i in range(50):
            timestamp = base_time + timedelta(minutes=i*30)
            # Simular comportamiento consistente
            value = 0.7 + np.random.normal(0, 0.1)
            user_a_data.append(BehaviorMetric(
                name="engagement_level",
                value=max(0, min(1, value)),
                timestamp=timestamp,
                context={"user_type": "A", "session_id": f"session_{i}"}
            ))
        
        # Entidad 2: Usuario tipo B
        user_b_data = []
        for i in range(50):
            timestamp = base_time + timedelta(minutes=i*30)
            # Simular comportamiento más variable
            value = 0.5 + np.random.normal(0, 0.2)
            user_b_data.append(BehaviorMetric(
                name="engagement_level",
                value=max(0, min(1, value)),
                timestamp=timestamp,
                context={"user_type": "B", "session_id": f"session_{i}"}
            ))
        
        return {
            "user_type_A": user_a_data,
            "user_type_B": user_b_data
        }
    
    async def run_comprehensive_demo(self):
        """Ejecutar demostración comprensiva"""
        try:
            logger.info("🚀 Iniciando demostración comprensiva del sistema avanzado")
            
            # 1. Análisis comprensivo
            logger.info("\n📊 1. Análisis Comprensivo")
            await self._demo_comprehensive_analysis()
            
            # 2. Análisis enfocado en calidad
            logger.info("\n🎯 2. Análisis Enfocado en Calidad")
            await self._demo_quality_focused_analysis()
            
            # 3. Análisis enfocado en rendimiento
            logger.info("\n⚡ 3. Análisis Enfocado en Rendimiento")
            await self._demo_performance_focused_analysis()
            
            # 4. Análisis enfocado en seguridad
            logger.info("\n🔒 4. Análisis Enfocado en Seguridad")
            await self._demo_security_focused_analysis()
            
            # 5. Análisis enfocado en emociones
            logger.info("\n😊 5. Análisis Enfocado en Emociones")
            await self._demo_emotion_focused_analysis()
            
            # 6. Análisis temporal
            logger.info("\n📈 6. Análisis Temporal")
            await self._demo_temporal_analysis()
            
            # 7. Análisis de comportamiento
            logger.info("\n🧠 7. Análisis de Comportamiento")
            await self._demo_behavior_analysis()
            
            # 8. Optimización de IA
            logger.info("\n🤖 8. Optimización de IA")
            await self._demo_ai_optimization()
            
            # 9. Resumen final
            logger.info("\n📋 9. Resumen Final")
            await self._demo_final_summary()
            
            logger.info("\n✅ Demostración completada exitosamente!")
            
        except Exception as e:
            logger.error(f"❌ Error en la demostración: {e}")
            raise
    
    async def _demo_comprehensive_analysis(self):
        """Demostrar análisis comprensivo"""
        try:
            result = await self.orchestrator.analyze_documents(
                documents=self.sample_documents,
                analysis_type=AnalysisType.COMPREHENSIVE,
                integration_level=IntegrationLevel.EXPERT
            )
            
            logger.info(f"✅ Análisis comprensivo completado en {result.execution_time:.2f} segundos")
            logger.info(f"📊 Componentes analizados: {len(result.results)}")
            logger.info(f"💡 Insights generados: {len(result.insights)}")
            logger.info(f"🎯 Recomendaciones: {len(result.recommendations)}")
            
            # Mostrar insights principales
            for insight in result.insights[:3]:
                logger.info(f"   • {insight['title']}: {insight['description']}")
            
        except Exception as e:
            logger.error(f"❌ Error en análisis comprensivo: {e}")
    
    async def _demo_quality_focused_analysis(self):
        """Demostrar análisis enfocado en calidad"""
        try:
            result = await self.orchestrator.analyze_documents(
                documents=self.sample_documents,
                analysis_type=AnalysisType.QUALITY_FOCUSED,
                integration_level=IntegrationLevel.ADVANCED
            )
            
            logger.info(f"✅ Análisis de calidad completado en {result.execution_time:.2f} segundos")
            
            if "content_quality" in result.results:
                quality_analyses = result.results["content_quality"]
                logger.info(f"📝 Documentos analizados: {len(quality_analyses)}")
                
                for i, analysis in enumerate(quality_analyses):
                    logger.info(f"   Documento {i+1}:")
                    logger.info(f"     • Score general: {analysis.overall_score:.2f}")
                    logger.info(f"     • Nivel de calidad: {analysis.quality_level.value}")
                    logger.info(f"     • Fortalezas: {len(analysis.strengths)}")
                    logger.info(f"     • Debilidades: {len(analysis.weaknesses)}")
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de calidad: {e}")
    
    async def _demo_performance_focused_analysis(self):
        """Demostrar análisis enfocado en rendimiento"""
        try:
            result = await self.orchestrator.analyze_documents(
                documents=self.sample_documents,
                analysis_type=AnalysisType.PERFORMANCE_FOCUSED,
                integration_level=IntegrationLevel.ADVANCED
            )
            
            logger.info(f"✅ Análisis de rendimiento completado en {result.execution_time:.2f} segundos")
            
            if "system_performance" in result.results:
                performance = result.results["system_performance"]
                logger.info(f"📊 Métricas de rendimiento:")
                logger.info(f"   • Alertas activas: {performance.get('active_alerts', 0)}")
                logger.info(f"   • Total de alertas: {performance.get('total_alerts', 0)}")
                logger.info(f"   • Recomendaciones: {performance.get('total_recommendations', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de rendimiento: {e}")
    
    async def _demo_security_focused_analysis(self):
        """Demostrar análisis enfocado en seguridad"""
        try:
            result = await self.orchestrator.analyze_documents(
                documents=self.sample_documents,
                analysis_type=AnalysisType.SECURITY_FOCUSED,
                integration_level=IntegrationLevel.ADVANCED
            )
            
            logger.info(f"✅ Análisis de seguridad completado en {result.execution_time:.2f} segundos")
            
            if "security_issues" in result.results:
                issues = result.results["security_issues"]
                logger.info(f"🔒 Problemas de seguridad detectados: {len(issues)}")
                
                for issue in issues[:3]:
                    logger.info(f"   • {issue.issue_type}: {issue.description}")
            
            if "privacy_analyses" in result.results:
                privacy_analyses = result.results["privacy_analyses"]
                logger.info(f"🔐 Análisis de privacidad: {len(privacy_analyses)}")
                
                for analysis in privacy_analyses:
                    logger.info(f"   • PII detectado: {len(analysis.pii_detected)}")
                    logger.info(f"   • Score de riesgo: {analysis.risk_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de seguridad: {e}")
    
    async def _demo_emotion_focused_analysis(self):
        """Demostrar análisis enfocado en emociones"""
        try:
            result = await self.orchestrator.analyze_documents(
                documents=self.sample_documents,
                analysis_type=AnalysisType.EMOTION_FOCUSED,
                integration_level=IntegrationLevel.ADVANCED
            )
            
            logger.info(f"✅ Análisis emocional completado en {result.execution_time:.2f} segundos")
            
            if "emotion_analyses" in result.results:
                emotion_analyses = result.results["emotion_analyses"]
                logger.info(f"😊 Análisis emocionales: {len(emotion_analyses)}")
                
                for i, analysis in enumerate(emotion_analyses):
                    logger.info(f"   Documento {i+1}:")
                    logger.info(f"     • Emoción dominante: {analysis.dominant_emotion.value}")
                    logger.info(f"     • Tono emocional: {analysis.emotional_tone.value}")
                    logger.info(f"     • Intensidad: {analysis.intensity.value}")
                    logger.info(f"     • Confianza: {analysis.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error en análisis emocional: {e}")
    
    async def _demo_temporal_analysis(self):
        """Demostrar análisis temporal"""
        try:
            # Agregar datos temporales
            for metric_name, data_points in self.sample_temporal_data.items():
                await self.orchestrator.temporal_analyzer.add_temporal_data(metric_name, data_points)
            
            # Analizar tendencias
            trend_analyses = []
            for metric_name in self.sample_temporal_data.keys():
                analysis = await self.orchestrator.temporal_analyzer.analyze_trends(metric_name)
                trend_analyses.append(analysis)
            
            logger.info(f"✅ Análisis temporal completado")
            logger.info(f"📈 Métricas analizadas: {len(trend_analyses)}")
            
            for analysis in trend_analyses:
                logger.info(f"   • {analysis.metric_name}:")
                logger.info(f"     - Tipo de tendencia: {analysis.trend_type.value}")
                logger.info(f"     - Patrón: {analysis.pattern_type.value}")
                logger.info(f"     - R²: {analysis.r_squared:.3f}")
                logger.info(f"     - Anomalías: {len(analysis.anomalies)}")
            
            # Comparar métricas
            if len(trend_analyses) > 1:
                comparison = await self.orchestrator.temporal_analyzer.compare_temporal_metrics(
                    list(self.sample_temporal_data.keys())
                )
                logger.info(f"🔄 Comparación temporal completada")
                logger.info(f"   • Correlaciones: {len(comparison.get('correlations', {}))}")
                logger.info(f"   • Diferencias: {len(comparison.get('significant_differences', []))}")
            
        except Exception as e:
            logger.error(f"❌ Error en análisis temporal: {e}")
    
    async def _demo_behavior_analysis(self):
        """Demostrar análisis de comportamiento"""
        try:
            # Agregar datos de comportamiento
            for entity_id, metrics in self.sample_behavior_data.items():
                await self.orchestrator.behavior_analyzer.add_behavior_metrics(entity_id, metrics)
            
            # Analizar patrones de comportamiento
            all_patterns = []
            for entity_id in self.sample_behavior_data.keys():
                patterns = await self.orchestrator.behavior_analyzer.analyze_behavior_patterns(entity_id)
                all_patterns.extend(patterns)
            
            logger.info(f"✅ Análisis de comportamiento completado")
            logger.info(f"🧠 Patrones identificados: {len(all_patterns)}")
            
            for pattern in all_patterns:
                logger.info(f"   • {pattern.id}:")
                logger.info(f"     - Tipo: {pattern.pattern_type.value}")
                logger.info(f"     - Complejidad: {pattern.complexity.value}")
                logger.info(f"     - Fuerza: {pattern.strength:.2f}")
                logger.info(f"     - Confianza: {pattern.confidence:.2f}")
            
            # Comparar patrones
            if len(self.sample_behavior_data) > 1:
                comparison = await self.orchestrator.behavior_analyzer.compare_behavior_patterns(
                    list(self.sample_behavior_data.keys())
                )
                logger.info(f"🔄 Comparación de comportamiento completada")
                logger.info(f"   • Similitudes: {len(comparison.get('similarities', {}))}")
                logger.info(f"   • Diferencias: {len(comparison.get('significant_differences', []))}")
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de comportamiento: {e}")
    
    async def _demo_ai_optimization(self):
        """Demostrar optimización de IA"""
        try:
            # Crear datos de ejemplo para optimización
            sample_data = pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100),
                'feature3': np.random.randn(100),
                'target': np.random.randn(100)
            })
            
            # Preparar datos de entrenamiento
            await self.orchestrator.ai_optimizer.prepare_training_data(
                data=sample_data,
                feature_columns=['feature1', 'feature2', 'feature3'],
                target_columns=['target']
            )
            
            # Entrenar modelos
            models_to_train = [
                ModelType.LINEAR_REGRESSION,
                ModelType.RANDOM_FOREST,
                ModelType.XGBOOST
            ]
            
            trained_models = []
            for model_type in models_to_train:
                performance = await self.orchestrator.ai_optimizer.train_model(model_type)
                trained_models.append(performance)
                logger.info(f"   • {model_type.value}: R² = {performance.r_squared:.3f}")
            
            # Optimizar modelos
            optimization_result = await self.orchestrator.ai_optimizer.optimize_models(
                OptimizationGoal.MAXIMIZE_QUALITY
            )
            
            logger.info(f"✅ Optimización de IA completada")
            logger.info(f"🤖 Mejor modelo: {optimization_result.best_model}")
            logger.info(f"📈 Mejora de rendimiento: {optimization_result.performance_improvement:.1%}")
            logger.info(f"💰 Reducción de costo: {optimization_result.cost_reduction:.1%}")
            logger.info(f"⚡ Mejora de velocidad: {optimization_result.speed_improvement:.1%}")
            
            # Generar insights de aprendizaje
            insights = await self.orchestrator.ai_optimizer.generate_learning_insights()
            logger.info(f"💡 Insights de aprendizaje: {len(insights)}")
            
            for insight in insights[:3]:
                logger.info(f"   • {insight.description}")
            
        except Exception as e:
            logger.error(f"❌ Error en optimización de IA: {e}")
    
    async def _demo_final_summary(self):
        """Demostrar resumen final"""
        try:
            # Obtener resumen del orquestador
            summary = await self.orchestrator.get_orchestrator_summary()
            
            logger.info("📋 RESUMEN FINAL DEL SISTEMA")
            logger.info("=" * 50)
            logger.info(f"📊 Total de solicitudes: {summary['total_requests']}")
            logger.info(f"✅ Análisis exitosos: {summary['successful_analyses']}")
            logger.info(f"❌ Análisis fallidos: {summary['failed_analyses']}")
            logger.info(f"⏱️ Tiempo promedio de ejecución: {summary['average_execution_time']:.2f}s")
            
            # Estado de los sistemas
            logger.info("\n🔧 ESTADO DE LOS SISTEMAS:")
            system_status = summary['system_status']
            for system_name, status in system_status.items():
                logger.info(f"   • {status.system_name}: {status.status} (Health: {status.health_score:.1f})")
            
            # Configuración
            logger.info("\n⚙️ CONFIGURACIÓN:")
            config = summary['configuration']
            logger.info(f"   • Procesamiento paralelo: {config['parallel_processing']}")
            logger.info(f"   • Optimización automática: {config['auto_optimization']}")
            logger.info(f"   • Monitoreo en tiempo real: {config['real_time_monitoring']}")
            logger.info(f"   • Análisis concurrentes máximos: {config['max_concurrent_analyses']}")
            
            # Exportar datos
            logger.info("\n💾 EXPORTANDO DATOS...")
            export_path = await self.orchestrator.export_orchestrator_data()
            logger.info(f"✅ Datos exportados a: {export_path}")
            
            # Exportar datos de sistemas individuales
            systems_to_export = [
                ("emotion_analyzer", self.orchestrator.emotion_analyzer),
                ("temporal_analyzer", self.orchestrator.temporal_analyzer),
                ("content_quality_analyzer", self.orchestrator.content_quality_analyzer),
                ("behavior_analyzer", self.orchestrator.behavior_analyzer),
                ("performance_optimizer", self.orchestrator.performance_optimizer),
                ("security_analyzer", self.orchestrator.security_analyzer),
                ("ai_optimizer", self.orchestrator.ai_optimizer)
            ]
            
            for system_name, system in systems_to_export:
                try:
                    if hasattr(system, 'export_emotion_analysis'):
                        export_path = await system.export_emotion_analysis()
                    elif hasattr(system, 'export_temporal_analysis'):
                        export_path = await system.export_temporal_analysis()
                    elif hasattr(system, 'export_quality_analysis'):
                        export_path = await system.export_quality_analysis()
                    elif hasattr(system, 'export_behavior_analysis'):
                        export_path = await system.export_behavior_analysis()
                    elif hasattr(system, 'export_performance_data'):
                        export_path = await system.export_performance_data()
                    elif hasattr(system, 'export_security_data'):
                        export_path = await system.export_security_data()
                    elif hasattr(system, 'export_optimization_data'):
                        export_path = await system.export_optimization_data()
                    else:
                        continue
                    
                    logger.info(f"   • {system_name}: {export_path}")
                except Exception as e:
                    logger.warning(f"   • {system_name}: Error en exportación - {e}")
            
            logger.info("\n🎉 DEMOSTRACIÓN COMPLETADA EXITOSAMENTE!")
            logger.info("El sistema avanzado de análisis de historial de IA está funcionando correctamente.")
            
        except Exception as e:
            logger.error(f"❌ Error en resumen final: {e}")

async def main():
    """Función principal"""
    try:
        demo = AdvancedSystemDemo()
        await demo.run_comprehensive_demo()
    except Exception as e:
        logger.error(f"❌ Error en la demostración: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
























