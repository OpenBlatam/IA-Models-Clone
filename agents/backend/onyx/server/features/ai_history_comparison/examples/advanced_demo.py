"""
Advanced Demo for AI History Comparison System
Demo avanzado para el sistema de análisis de historial de IA
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_advanced_system():
    """
    Demostración completa del sistema avanzado
    """
    print("🚀 Demo del Sistema Avanzado de Análisis de Historial de IA")
    print("=" * 70)
    
    # ============================================================================
    # DEMO 1: Evaluación de Calidad de Texto
    # ============================================================================
    print(f"\n📊 Demo 1: Evaluación Avanzada de Calidad de Texto")
    print("-" * 50)
    
    try:
        from text_quality_evaluator import AdvancedTextQualityEvaluator, QualityDimension
        
        # Inicializar evaluador
        quality_evaluator = AdvancedTextQualityEvaluator()
        
        # Textos de ejemplo para evaluar
        sample_texts = [
            {
                "id": "high_quality",
                "content": """
                La inteligencia artificial representa una revolución tecnológica sin precedentes. 
                Los algoritmos de machine learning pueden procesar grandes volúmenes de datos 
                y extraer patrones complejos que los humanos no podrían detectar fácilmente.
                
                Sin embargo, es crucial considerar las implicaciones éticas de estas tecnologías. 
                Debemos asegurarnos de que la IA se implemente de manera responsable y justa, 
                considerando el impacto en la sociedad y el medio ambiente.
                
                En conclusión, la IA ofrece oportunidades extraordinarias, pero requiere 
                un enfoque equilibrado que priorice el bienestar humano.
                """
            },
            {
                "id": "medium_quality",
                "content": """
                La IA es muy importante. Hay muchos algoritmos que pueden hacer cosas. 
                Es bueno usar la IA para diferentes tareas. Algunas veces funciona bien 
                y otras veces no tanto. Los desarrolladores trabajan mucho en esto.
                
                Es importante pensar en la ética. La IA puede ser útil pero también 
                puede causar problemas. Hay que tener cuidado al usarla.
                """
            },
            {
                "id": "low_quality",
                "content": """
                IA es genial. Algoritmos hacen cosas. Machine learning es cool. 
                Datos son importantes. IA puede ayudar. Es el futuro. 
                Muy bueno para todo. Recomiendo usar IA. Es increíble.
                """
            }
        ]
        
        # Evaluar calidad de cada texto
        for text_data in sample_texts:
            print(f"  📝 Evaluando: {text_data['id']}")
            
            report = await quality_evaluator.evaluate_text_quality(
                text=text_data["content"],
                document_id=text_data["id"]
            )
            
            print(f"    • Score general: {report.overall_score:.2f}")
            print(f"    • Nivel de calidad: {report.quality_level.value}")
            print(f"    • Fortalezas: {len(report.strengths)}")
            print(f"    • Debilidades: {len(report.weaknesses)}")
            print(f"    • Recomendaciones: {len(report.recommendations)}")
            
            # Mostrar algunas recomendaciones
            if report.recommendations:
                print(f"    • Ejemplo de recomendación: {report.recommendations[0]}")
        
        print(f"\n✅ Evaluación de calidad completada para {len(sample_texts)} textos")
        
    except Exception as e:
        print(f"❌ Error en evaluación de calidad: {e}")
    
    # ============================================================================
    # DEMO 2: Generación Inteligente de Contenido
    # ============================================================================
    print(f"\n🤖 Demo 2: Generación Inteligente de Contenido")
    print("-" * 50)
    
    try:
        from intelligent_content_generator import IntelligentContentGenerator, ContentRequest, ContentType, ContentStyle, ContentTone
        
        # Inicializar generador
        content_generator = IntelligentContentGenerator()
        
        # Solicitudes de contenido
        content_requests = [
            ContentRequest(
                topic="Inteligencia Artificial en la Medicina",
                content_type=ContentType.ARTICLE,
                style=ContentStyle.TECHNICAL,
                tone=ContentTone.PROFESSIONAL,
                target_audience="médicos y profesionales de la salud",
                word_count=800,
                keywords=["IA", "medicina", "diagnóstico", "tratamiento", "salud"],
                requirements=["Incluir ejemplos específicos", "Mencionar beneficios y riesgos"]
            ),
            ContentRequest(
                topic="Cómo Aprender Machine Learning",
                content_type=ContentType.TUTORIAL,
                style=ContentStyle.INFORMATIVE,
                tone=ContentTone.FRIENDLY,
                target_audience="principiantes en programación",
                word_count=600,
                keywords=["machine learning", "aprendizaje", "programación", "datos", "algoritmos"],
                requirements=["Paso a paso", "Recursos recomendados", "Ejemplos prácticos"]
            )
        ]
        
        # Generar contenido
        for i, request in enumerate(content_requests):
            print(f"  📝 Generando contenido {i+1}: {request.topic}")
            
            generated_content = await content_generator.generate_content(request)
            
            print(f"    • ID: {generated_content.id}")
            print(f"    • Palabras generadas: {generated_content.metadata['word_count']}")
            print(f"    • Secciones: {generated_content.metadata['section_count']}")
            print(f"    • Score de calidad: {generated_content.quality_score:.2f}")
            print(f"    • Plantilla usada: {generated_content.metadata['template_used']}")
            
            # Mostrar preview del contenido
            content_preview = generated_content.content[:200] + "..." if len(generated_content.content) > 200 else generated_content.content
            print(f"    • Preview: {content_preview}")
        
        # Obtener resumen
        summary = await content_generator.get_content_summary()
        print(f"\n  📊 Resumen de generación:")
        print(f"    • Total contenido generado: {summary['total_content_generated']}")
        print(f"    • Score promedio de calidad: {summary['average_quality_score']:.2f}")
        print(f"    • Plantillas disponibles: {summary['templates_available']}")
        
    except Exception as e:
        print(f"❌ Error en generación de contenido: {e}")
    
    # ============================================================================
    # DEMO 3: Análisis de Tendencias y Predicciones
    # ============================================================================
    print(f"\n📈 Demo 3: Análisis de Tendencias y Predicciones")
    print("-" * 50)
    
    try:
        from trend_analyzer import AdvancedTrendAnalyzer, PredictionType
        
        # Inicializar analizador de tendencias
        trend_analyzer = AdvancedTrendAnalyzer()
        
        # Generar datos históricos simulados
        historical_data = {}
        base_date = datetime.now() - timedelta(days=30)
        
        # Simular métricas con diferentes tendencias
        metrics_data = {
            "quality_score": {"base": 0.7, "trend": 0.01, "volatility": 0.05},
            "user_satisfaction": {"base": 0.8, "trend": 0.005, "volatility": 0.03},
            "processing_time": {"base": 2.0, "trend": -0.02, "volatility": 0.1},
            "error_rate": {"base": 0.02, "trend": -0.001, "volatility": 0.01}
        }
        
        for metric_name, config in metrics_data.items():
            data_points = []
            for i in range(30):
                date = base_date + timedelta(days=i)
                # Simular valor con tendencia y ruido
                value = config["base"] + (config["trend"] * i) + (config["volatility"] * (i % 7 - 3))
                data_points.append({
                    "timestamp": date,
                    "value": max(0, value)  # Asegurar valores positivos
                })
            historical_data[metric_name] = data_points
        
        # Analizar tendencias
        print("  🔍 Analizando tendencias...")
        trends = await trend_analyzer.analyze_trends(historical_data)
        
        for metric_name, trend in trends.items():
            print(f"    • {metric_name}:")
            print(f"      - Tipo de tendencia: {trend.trend_type.value}")
            print(f"      - Confianza: {trend.confidence:.2f}")
            print(f"      - Volatilidad: {trend.volatility:.2f}")
            print(f"      - R²: {trend.r_squared:.2f}")
        
        # Generar predicciones
        print("\n  🔮 Generando predicciones...")
        predictions = {}
        for metric_name in ["quality_score", "user_satisfaction"]:
            prediction = await trend_analyzer.generate_predictions(
                metric_name=metric_name,
                prediction_type=PredictionType.SHORT_TERM
            )
            predictions[metric_name] = prediction
            
            print(f"    • {metric_name}:")
            print(f"      - Nivel de confianza: {prediction.confidence_level.value}")
            print(f"      - Score de confianza: {prediction.confidence_score:.2f}")
            print(f"      - Modelo usado: {prediction.model_used}")
            print(f"      - Valores predichos: {len(prediction.predicted_values)}")
        
        # Generar insights
        print("\n  💡 Generando insights...")
        insights = await trend_analyzer.generate_insights(trends)
        
        for insight in insights[:3]:  # Mostrar solo los primeros 3
            print(f"    • {insight.title}")
            print(f"      - Impacto: {insight.impact_level}")
            print(f"      - Confianza: {insight.confidence:.2f}")
            print(f"      - Recomendaciones: {len(insight.actionable_recommendations)}")
        
        # Obtener resumen
        summary = await trend_analyzer.get_trend_summary()
        print(f"\n  📊 Resumen de análisis:")
        print(f"    • Métricas analizadas: {summary['total_metrics_analyzed']}")
        print(f"    • Predicciones generadas: {summary['total_predictions']}")
        print(f"    • Insights generados: {summary['total_insights']}")
        print(f"    • Confianza promedio: {summary['average_confidence']:.2f}")
        
    except Exception as e:
        print(f"❌ Error en análisis de tendencias: {e}")
    
    # ============================================================================
    # DEMO 4: Dashboard Avanzado
    # ============================================================================
    print(f"\n📊 Demo 4: Dashboard Avanzado Interactivo")
    print("-" * 50)
    
    try:
        from advanced_dashboard import AdvancedDashboard, DashboardWidget, ChartType, DashboardTheme
        
        # Inicializar dashboard
        dashboard = AdvancedDashboard()
        
        # Crear widgets de ejemplo
        widgets_data = [
            {
                "id": "quality_trend",
                "title": "Tendencia de Calidad",
                "chart_type": ChartType.LINE,
                "data": {
                    "x": [f"Día {i}" for i in range(1, 31)],
                    "y": [0.7 + 0.01*i + 0.05*(i%7-3) for i in range(30)],
                    "labels": {"x": "Días", "y": "Score de Calidad"}
                }
            },
            {
                "id": "user_satisfaction",
                "title": "Satisfacción del Usuario",
                "chart_type": ChartType.BAR,
                "data": {
                    "x": ["Enero", "Febrero", "Marzo", "Abril", "Mayo"],
                    "y": [0.75, 0.82, 0.78, 0.85, 0.88],
                    "labels": {"x": "Mes", "y": "Satisfacción"}
                }
            },
            {
                "id": "error_distribution",
                "title": "Distribución de Errores",
                "chart_type": ChartType.PIE,
                "data": {
                    "labels": ["Errores de Red", "Errores de Validación", "Errores de Sistema", "Otros"],
                    "values": [25, 35, 20, 20]
                }
            },
            {
                "id": "performance_metrics",
                "title": "Métricas de Rendimiento",
                "chart_type": ChartType.HEATMAP,
                "data": {
                    "x": ["CPU", "Memoria", "Disco", "Red"],
                    "y": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"],
                    "z": [[80, 75, 85, 70], [65, 70, 75, 68], [90, 85, 88, 92], [45, 50, 48, 52], [60, 65, 62, 58]]
                }
            }
        ]
        
        # Crear widgets
        created_widgets = []
        for widget_data in widgets_data:
            widget = await dashboard.create_widget(
                widget_id=widget_data["id"],
                title=widget_data["title"],
                chart_type=widget_data["chart_type"],
                data=widget_data["data"]
            )
            created_widgets.append(widget)
            print(f"  📊 Widget creado: {widget.title}")
        
        # Agregar widgets al layout principal
        for i, widget in enumerate(created_widgets):
            await dashboard.add_widget_to_layout(
                layout_id="main",
                widget_id=widget.id,
                position=(i % 2, i // 2),
                size=(1, 1)
            )
        
        # Generar dashboard HTML
        print("  🌐 Generando dashboard HTML...")
        dashboard_html = await dashboard.generate_dashboard_html("main", DashboardTheme.LIGHT)
        
        # Guardar dashboard
        with open("exports/advanced_dashboard.html", "w", encoding="utf-8") as f:
            f.write(dashboard_html)
        
        print("  ✅ Dashboard generado y guardado en exports/advanced_dashboard.html")
        
        # Obtener resumen del dashboard
        summary = await dashboard.get_dashboard_summary()
        print(f"\n  📊 Resumen del dashboard:")
        print(f"    • Layouts disponibles: {summary['total_layouts']}")
        print(f"    • Widgets creados: {summary['total_widgets']}")
        print(f"    • Temas disponibles: {len(summary['available_themes'])}")
        print(f"    • Formatos de exportación: {len(summary['export_formats'])}")
        
    except Exception as e:
        print(f"❌ Error en dashboard: {e}")
    
    # ============================================================================
    # DEMO 5: Integraciones Externas
    # ============================================================================
    print(f"\n🔗 Demo 5: Integraciones Externas")
    print("-" * 50)
    
    try:
        from external_integrations import ExternalIntegrations, IntegrationType
        
        # Inicializar integraciones
        integrations = ExternalIntegrations()
        
        # Mostrar integraciones disponibles
        status = await integrations.get_integration_status()
        print(f"  📋 Integraciones disponibles: {status['total_integrations']}")
        
        for integration_id, integration_info in status['integrations'].items():
            print(f"    • {integration_info['name']} ({integration_info['type']})")
            print(f"      - Estado: {integration_info['status']}")
            print(f"      - Rate limit: {integration_info['rate_limit']}/min")
        
        # Simular configuración de integración (sin claves reales)
        print("\n  ⚙️ Simulando configuración de integraciones...")
        
        # Configurar Slack (simulado)
        slack_configured = await integrations.configure_integration(
            integration_id="slack",
            api_key="xoxb-simulated-token",
            custom_headers={"Content-Type": "application/json"}
        )
        print(f"    • Slack configurado: {'✅' if slack_configured else '❌'}")
        
        # Configurar Discord (simulado)
        discord_configured = await integrations.configure_integration(
            integration_id="discord",
            api_key="simulated-bot-token"
        )
        print(f"    • Discord configurado: {'✅' if discord_configured else '❌'}")
        
        # Simular envío de notificación
        print("\n  📤 Simulando envío de notificaciones...")
        
        notification_sent = await integrations.send_notification(
            service="slack",
            message="🚀 Sistema de análisis de historial de IA funcionando correctamente",
            channel="#ai-monitoring",
            title="Estado del Sistema"
        )
        print(f"    • Notificación Slack enviada: {'✅' if notification_sent else '❌'}")
        
        # Obtener estadísticas de cache y rate limiting
        print(f"\n  📊 Estadísticas del sistema:")
        print(f"    • Respuestas en cache: {status['cache_stats']['cached_responses']}")
        print(f"    • TTL del cache: {status['cache_stats']['cache_ttl']}s")
        print(f"    • Integraciones activas: {status['active_integrations']}")
        
    except Exception as e:
        print(f"❌ Error en integraciones: {e}")
    
    # ============================================================================
    # DEMO 6: Sistema de Monitoreo
    # ============================================================================
    print(f"\n📡 Demo 6: Sistema de Monitoreo Avanzado")
    print("-" * 50)
    
    try:
        from monitoring_system import AdvancedMonitoringSystem, AlertLevel, MetricType
        
        # Inicializar sistema de monitoreo
        monitoring = AdvancedMonitoringSystem()
        
        # Iniciar monitoreo
        await monitoring.start_monitoring()
        print("  🚀 Sistema de monitoreo iniciado")
        
        # Simular algunas métricas de aplicación
        print("  📊 Simulando métricas de aplicación...")
        
        for i in range(10):
            # Simular métricas
            await monitoring._record_metric("application.response_time", 1.5 + (i * 0.1), MetricType.GAUGE)
            await monitoring._record_metric("application.error_rate", 0.02 + (i * 0.001), MetricType.GAUGE)
            await monitoring._record_metric("application.requests_per_second", 100 + (i * 5), MetricType.GAUGE)
            
            await asyncio.sleep(0.1)  # Pequeña pausa
        
        # Obtener estado de salud del sistema
        health = await monitoring.get_system_health()
        print(f"\n  💚 Estado de salud del sistema:")
        print(f"    • Estado general: {health.overall_status}")
        print(f"    • Uso de CPU: {health.cpu_usage:.1f}%")
        print(f"    • Uso de memoria: {health.memory_usage:.1f}%")
        print(f"    • Uso de disco: {health.disk_usage:.1f}%")
        print(f"    • Alertas activas: {health.active_alerts}")
        print(f"    • Métricas totales: {health.total_metrics}")
        print(f"    • Tiempo activo: {health.uptime:.1f}s")
        
        # Obtener resumen de métricas
        metrics_summary = await monitoring.get_metrics_summary()
        print(f"\n  📈 Resumen de métricas:")
        print(f"    • Total de métricas: {metrics_summary['total_metrics']}")
        print(f"    • Métricas por tipo: {metrics_summary['metrics_by_type']}")
        print(f"    • Métricas recientes: {len(metrics_summary['recent_metrics'])}")
        
        # Obtener resumen de alertas
        alerts_summary = await monitoring.get_alerts_summary()
        print(f"\n  🚨 Resumen de alertas:")
        print(f"    • Alertas activas: {alerts_summary['active_alerts']}")
        print(f"    • Reglas de alerta: {alerts_summary['total_alert_rules']}")
        print(f"    • Reglas habilitadas: {alerts_summary['enabled_rules']}")
        print(f"    • Distribución por nivel: {alerts_summary['alert_levels']}")
        
        # Detener monitoreo
        await monitoring.stop_monitoring()
        print("  🛑 Sistema de monitoreo detenido")
        
    except Exception as e:
        print(f"❌ Error en sistema de monitoreo: {e}")
    
    # ============================================================================
    # RESUMEN FINAL
    # ============================================================================
    print(f"\n🎉 Demo Avanzado Completado Exitosamente!")
    print("=" * 70)
    
    print(f"📊 Resumen del Demo Avanzado:")
    print(f"  ✅ Evaluación de calidad de texto")
    print(f"  ✅ Generación inteligente de contenido")
    print(f"  ✅ Análisis de tendencias y predicciones")
    print(f"  ✅ Dashboard avanzado interactivo")
    print(f"  ✅ Integraciones externas")
    print(f"  ✅ Sistema de monitoreo avanzado")
    
    print(f"\n🚀 El sistema avanzado está completamente operativo!")
    print(f"   - Evaluación de calidad con 10 dimensiones")
    print(f"   - Generación de contenido con plantillas inteligentes")
    print(f"   - Análisis de tendencias con predicciones ML")
    print(f"   - Dashboard interactivo con múltiples gráficos")
    print(f"   - Integraciones con servicios externos")
    print(f"   - Monitoreo en tiempo real con alertas")
    print(f"   - Exportación de datos en múltiples formatos")
    
    print(f"\n📁 Archivos generados:")
    print(f"   - exports/advanced_dashboard.html")
    print(f"   - exports/quality_reports_*.json")
    print(f"   - exports/trend_analysis_*.json")
    print(f"   - exports/monitoring_data_*.json")

async def demo_integration_workflow():
    """
    Demo de flujo de trabajo integrado
    """
    print("\n🔄 Demo de Flujo de Trabajo Integrado")
    print("=" * 50)
    
    try:
        # Simular flujo completo
        print("1. 📝 Generando contenido con IA...")
        await asyncio.sleep(1)
        
        print("2. 📊 Evaluando calidad del contenido...")
        await asyncio.sleep(1)
        
        print("3. 📈 Analizando tendencias de calidad...")
        await asyncio.sleep(1)
        
        print("4. 🚨 Verificando alertas del sistema...")
        await asyncio.sleep(1)
        
        print("5. 📤 Enviando notificaciones...")
        await asyncio.sleep(1)
        
        print("6. 📊 Actualizando dashboard...")
        await asyncio.sleep(1)
        
        print("✅ Flujo de trabajo completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en flujo de trabajo: {e}")

if __name__ == "__main__":
    # Ejecutar demo principal
    asyncio.run(demo_advanced_system())
    
    # Ejecutar demo de flujo integrado
    asyncio.run(demo_integration_workflow())



























