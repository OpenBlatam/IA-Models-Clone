"""
Demo del Sistema de Análisis de Historial de IA
Ejemplo completo de uso del sistema de comparación de historial de IA
"""

import asyncio
import json
from datetime import datetime, timedelta
from ai_history_analyzer import AIHistoryAnalyzer, DocumentHistory
from ml_optimizer import MLOptimizer
from realtime_analyzer import RealtimeAnalyzer, RealtimeMetric, MetricType
from notification_system import NotificationSystem, NotificationChannel, NotificationPriority, NotificationRecipient
from export_tools import AdvancedExportTools, ExportConfig, ExportFormat, ExportType

async def demo_ai_history_analysis():
    """
    Demostración completa del sistema de análisis de historial de IA
    """
    print("🚀 Iniciando Demo del Sistema de Análisis de Historial de IA")
    print("=" * 60)
    
    # Inicializar los componentes del sistema
    analyzer = AIHistoryAnalyzer(
        ai_provider="openai",  # Cambiar a "anthropic" si prefieres
        api_key=None,  # Configurar con tu API key
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Inicializar optimizador ML
    ml_optimizer = MLOptimizer(
        model_storage_path="models/",
        cache_path="cache/",
        enable_auto_optimization=True
    )
    
    # Inicializar analizador en tiempo real
    realtime_analyzer = RealtimeAnalyzer(
        redis_url="redis://localhost:6379",
        analysis_interval=10,
        websocket_port=8765
    )
    
    # Iniciar analizador en tiempo real en background
    asyncio.create_task(realtime_analyzer.start())
    
    # Datos de ejemplo - simulando historial de una IA que genera documentos
    sample_documents = [
        {
            "content": """
            # Guía Completa de Marketing Digital
            
            El marketing digital es una estrategia fundamental para las empresas modernas. 
            En este artículo, exploraremos las mejores prácticas y estrategias efectivas.
            
            ## 1. SEO y Optimización
            
            El SEO es crucial para mejorar la visibilidad en línea. Las técnicas incluyen:
            - Investigación de palabras clave
            - Optimización de contenido
            - Construcción de enlaces
            
            ## 2. Marketing en Redes Sociales
            
            Las redes sociales ofrecen oportunidades únicas para conectar con audiencias.
            Plataformas principales:
            - Facebook: Ideal para B2C
            - LinkedIn: Perfecto para B2B
            - Instagram: Excelente para visuales
            
            ## Conclusión
            
            El marketing digital requiere una estrategia integral que combine múltiples canales
            y técnicas para maximizar el impacto y los resultados.
            """,
            "query": "Escribe una guía completa sobre marketing digital para empresas",
            "metadata": {"category": "marketing", "target_audience": "empresas"},
            "user_feedback": {"rating": 4, "comments": "Muy útil y bien estructurado"}
        },
        {
            "content": """
            Marketing digital es importante para empresas. SEO ayuda con visibilidad.
            Redes sociales conectan con audiencias. Facebook es bueno para B2C.
            LinkedIn funciona bien para B2B. Instagram es visual.
            Necesitas estrategia integral para buenos resultados.
            """,
            "query": "Escribe una guía completa sobre marketing digital para empresas",
            "metadata": {"category": "marketing", "target_audience": "empresas"},
            "user_feedback": {"rating": 2, "comments": "Muy básico, falta estructura"}
        },
        {
            "content": """
            # Estrategias Avanzadas de Marketing Digital
            
            ## Introducción
            
            En el panorama competitivo actual, el marketing digital se ha convertido en 
            una disciplina compleja que requiere conocimientos especializados y enfoques 
            estratégicos sofisticados.
            
            ## Análisis de Mercado y Segmentación
            
            ### Investigación de Audiencia
            - Análisis demográfico detallado
            - Comportamiento de compra
            - Preferencias de contenido
            - Canales de comunicación preferidos
            
            ### Segmentación Avanzada
            1. Segmentación psicográfica
            2. Segmentación por comportamiento
            3. Segmentación por valor de cliente
            
            ## Estrategias de Contenido
            
            ### Content Marketing
            - Storytelling efectivo
            - Contenido educativo
            - Casos de estudio
            - Tutoriales paso a paso
            
            ### SEO Técnico
            - Optimización de velocidad
            - Estructura de datos
            - Core Web Vitals
            - Mobile-first indexing
            
            ## Automatización y Personalización
            
            ### Marketing Automation
            - Email marketing automatizado
            - Nurturing de leads
            - Scoring de prospectos
            - Retargeting inteligente
            
            ## Medición y Optimización
            
            ### KPIs Clave
            - ROI del marketing digital
            - Costo de adquisición de cliente (CAC)
            - Valor de vida del cliente (LTV)
            - Tasa de conversión
            
            ## Conclusión
            
            El marketing digital exitoso requiere una combinación de creatividad, 
            análisis de datos y ejecución estratégica. Las empresas que invierten 
            en capacidades digitales avanzadas obtienen ventajas competitivas 
            significativas en el mercado actual.
            """,
            "query": "Escribe una guía completa sobre marketing digital para empresas",
            "metadata": {"category": "marketing", "target_audience": "empresas"},
            "user_feedback": {"rating": 5, "comments": "Excelente, muy detallado y profesional"}
        },
        {
            "content": """
            # Inteligencia Artificial en el Marketing
            
            La IA está revolucionando el marketing digital. Los algoritmos pueden:
            - Analizar grandes volúmenes de datos
            - Personalizar experiencias
            - Optimizar campañas automáticamente
            - Predecir comportamientos de compra
            
            ## Aplicaciones Prácticas
            
            ### Chatbots Inteligentes
            Proporcionan atención al cliente 24/7 y pueden:
            - Responder preguntas frecuentes
            - Guiar a los usuarios
            - Recopilar información de leads
            
            ### Análisis Predictivo
            - Identificar clientes en riesgo de abandono
            - Predecir tendencias de mercado
            - Optimizar precios dinámicamente
            
            ## Beneficios Clave
            
            1. **Eficiencia Operativa**: Automatización de tareas repetitivas
            2. **Personalización**: Experiencias únicas para cada usuario
            3. **Insights Accionables**: Datos procesables para decisiones
            4. **Escalabilidad**: Manejo de grandes volúmenes de datos
            
            ## Implementación
            
            Para implementar IA en marketing:
            - Comienza con objetivos claros
            - Selecciona las herramientas adecuadas
            - Capacita a tu equipo
            - Mide y optimiza continuamente
            
            La IA no reemplaza la creatividad humana, sino que la potencia.
            """,
            "query": "Escribe sobre inteligencia artificial en marketing",
            "metadata": {"category": "ai", "target_audience": "marketers"},
            "user_feedback": {"rating": 4, "comments": "Buen contenido, bien estructurado"}
        },
        {
            "content": """
            IA en marketing es importante. Chatbots ayudan con atención al cliente.
            Análisis predictivo identifica tendencias. Automatización mejora eficiencia.
            Personalización crea mejores experiencias. Implementación requiere planificación.
            """,
            "query": "Escribe sobre inteligencia artificial en marketing",
            "metadata": {"category": "ai", "target_audience": "marketers"},
            "user_feedback": {"rating": 2, "comments": "Muy superficial, necesita más detalle"}
        }
    ]
    
    print("📝 Agregando documentos al historial...")
    
    # Agregar documentos al historial
    document_ids = []
    for i, doc_data in enumerate(sample_documents):
        doc_id = await analyzer.add_document(
            content=doc_data["content"],
            query=doc_data["query"],
            metadata=doc_data["metadata"],
            user_feedback=doc_data["user_feedback"]
        )
        document_ids.append(doc_id)
        print(f"  ✅ Documento {i+1} agregado: {doc_id[:8]}...")
    
    print(f"\n📊 Estadísticas del Historial:")
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n🔍 Generando Insights del Historial...")
    insights = await analyzer.generate_history_insights()
    
    for i, insight in enumerate(insights, 1):
        print(f"\n  💡 Insight {i}: {insight.title}")
        print(f"     Tipo: {insight.insight_type}")
        print(f"     Confianza: {insight.confidence:.2f}")
        print(f"     Impacto: {insight.impact_score:.2f}")
        print(f"     Descripción: {insight.description}")
        print(f"     Recomendaciones:")
        for rec in insight.recommendations:
            print(f"       • {rec}")
    
    print(f"\n🎯 Generando Recomendaciones de Mejora...")
    recommendations = await analyzer.generate_improvement_recommendations()
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  📋 Recomendación {i}: {rec.title}")
        print(f"     Categoría: {rec.category.value}")
        print(f"     Prioridad: {rec.priority}")
        print(f"     Mejora Esperada: {rec.expected_improvement:.1%}")
        print(f"     Descripción: {rec.description}")
        print(f"     Tips de Implementación:")
        for tip in rec.implementation_tips:
            print(f"       • {tip}")
    
    print(f"\n🔄 Comparando Documentos...")
    
    # Comparar documentos similares
    if len(document_ids) >= 2:
        comparison = await analyzer.compare_documents(document_ids[0], document_ids[1])
        print(f"\n  📊 Comparación entre documentos {document_ids[0][:8]}... y {document_ids[1][:8]}...")
        print(f"     Similitud: {comparison.similarity_score:.2%}")
        print(f"     Diferencia de Calidad: {comparison.quality_difference:.2f}")
        print(f"     Áreas de Mejora: {[area.value for area in comparison.improvement_areas]}")
        print(f"     Mejores Prácticas:")
        for practice in comparison.best_practices:
            print(f"       • {practice}")
    
    print(f"\n🏆 Mejores Documentos:")
    best_docs = analyzer.get_best_documents(3)
    for i, doc in enumerate(best_docs, 1):
        print(f"  {i}. ID: {doc.id[:8]}... | Calidad: {doc.quality_score:.2f} | Query: {doc.query[:50]}...")
    
    print(f"\n⚠️  Documentos que Necesitan Mejora:")
    worst_docs = analyzer.get_worst_documents(3)
    for i, doc in enumerate(worst_docs, 1):
        print(f"  {i}. ID: {doc.id[:8]}... | Calidad: {doc.quality_score:.2f} | Query: {doc.query[:50]}...")
    
    print(f"\n📄 Exportando Reporte Completo...")
    report = await analyzer.export_analysis_report()
    
    # Guardar reporte en archivo
    report_filename = f"ai_history_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"  ✅ Reporte guardado en: {report_filename}")
    
    print(f"\n🤖 Demostrando Optimización de ML...")
    
    # Demostrar optimización de ML
    if len(analyzer.document_history) >= 10:
        print("  📊 Preparando datos para optimización de ML...")
        
        # Preparar datos de entrenamiento
        documents_data = []
        for doc in analyzer.document_history.values():
            documents_data.append({
                "content": doc.content,
                "query": doc.query,
                "quality_score": doc.quality_score,
                "readability_score": doc.readability_score,
                "originality_score": doc.originality_score,
                "word_count": doc.word_count,
                "processing_time": doc.processing_time or 0.0,
                "timestamp": doc.timestamp.isoformat(),
                "metadata": doc.metadata
            })
        
        print(f"  🔧 Optimizando modelos de ML...")
        X, y = await ml_optimizer.prepare_training_data(documents_data)
        optimization_result = await ml_optimizer.optimize_models(X, y)
        
        print(f"  ✅ Mejor modelo: {optimization_result.best_model}")
        print(f"  📈 Puntuación R²: {optimization_result.best_score:.4f}")
        print(f"  📊 Mejora: {optimization_result.improvement:.4f}")
        print(f"  💡 Recomendaciones:")
        for rec in optimization_result.recommendations:
            print(f"    • {rec}")
        
        # Demostrar predicción de calidad
        print(f"\n  🔮 Prediciendo calidad de nuevo documento...")
        test_content = "Este es un documento de prueba para demostrar la predicción de calidad."
        test_query = "Escribe un documento de prueba"
        
        test_doc_data = [{
            "content": test_content,
            "query": test_query,
            "quality_score": 0.0,
            "readability_score": 0.0,
            "originality_score": 0.0,
            "word_count": len(test_content.split()),
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat(),
            "metadata": {}
        }]
        
        X_test, _ = await ml_optimizer.prepare_training_data(test_doc_data)
        predicted_quality = await ml_optimizer.predict_quality(X_test[0])
        
        print(f"  🎯 Calidad predicha: {predicted_quality:.3f}")
    else:
        print("  ⚠️  Se requieren al menos 10 documentos para optimización de ML")
    
    print(f"\n⚡ Demostrando Análisis en Tiempo Real...")
    
    # Simular métricas en tiempo real
    print("  📊 Enviando métricas al analizador en tiempo real...")
    
    for i, doc in enumerate(list(analyzer.document_history.values())[:5]):
        await realtime_analyzer.add_metric(RealtimeMetric(
            metric_type=MetricType.QUALITY_SCORE,
            value=doc.quality_score,
            timestamp=doc.timestamp,
            document_id=doc.id,
            query=doc.query,
            metadata={"word_count": doc.word_count}
        ))
        
        await realtime_analyzer.add_metric(RealtimeMetric(
            metric_type=MetricType.READABILITY,
            value=doc.readability_score,
            timestamp=doc.timestamp,
            document_id=doc.id,
            query=doc.query
        ))
        
        await realtime_analyzer.add_metric(RealtimeMetric(
            metric_type=MetricType.ORIGINALITY,
            value=doc.originality_score,
            timestamp=doc.timestamp,
            document_id=doc.id,
            query=doc.query
        ))
        
        print(f"    ✅ Métricas enviadas para documento {i+1}")
    
    # Esperar un poco para que se procesen las métricas
    await asyncio.sleep(2)
    
    # Obtener estado del sistema en tiempo real
    status = realtime_analyzer.get_system_status()
    print(f"  📈 Estado del sistema:")
    print(f"    • Alertas activas: {status['active_alerts']}")
    print(f"    • Métricas procesadas: {status['metrics_processed']}")
    print(f"    • Clientes WebSocket: {status['websocket_clients']}")
    print(f"    • Tiempo de actividad: {status['uptime']:.1f}s")
    
    # Mostrar tendencias
    if realtime_analyzer.trend_analysis:
        print(f"  📊 Tendencias detectadas:")
        for metric_type, trend in realtime_analyzer.trend_analysis.items():
            print(f"    • {metric_type.value}: {trend.trend_direction} (confianza: {trend.confidence:.2f})")
    
    # Mostrar alertas activas
    if realtime_analyzer.active_alerts:
        print(f"  🚨 Alertas activas:")
        for alert in realtime_analyzer.active_alerts.values():
            print(f"    • {alert.level.value.upper()}: {alert.title}")
    
    print(f"\n📧 Demostrando Sistema de Notificaciones...")
    # Inicializar sistema de notificaciones
    notification_system = NotificationSystem(
        redis_url="redis://localhost:6379"
    )
    
    # Agregar destinatario de prueba
    recipient = NotificationRecipient(
        id="demo_user",
        name="Usuario Demo",
        email="demo@example.com",
        preferences={"email_notifications": True}
    )
    await notification_system.add_recipient(recipient)
    print(f"  👤 Destinatario agregado: {recipient.name}")
    
    # Enviar notificación de alerta crítica
    notification_ids = await notification_system.send_notification(
        template_id="critical_alert",
        recipient_id="demo_user",
        variables={
            "alert_type": "Calidad Baja",
            "message": "Se detectó un documento con calidad inferior al umbral",
            "timestamp": datetime.now().isoformat(),
            "current_value": 0.3,
            "threshold": 0.5,
            "document_id": "doc_001"
        }
    )
    print(f"  📧 Notificación enviada: {len(notification_ids)} canales")
    
    # Obtener estadísticas de notificaciones
    notif_stats = notification_system.get_notification_stats()
    print(f"  📊 Estadísticas de notificaciones: {notif_stats['total_notifications']} enviadas")
    
    print(f"\n📤 Demostrando Herramientas de Exportación...")
    # Inicializar herramientas de exportación
    export_tools = AdvancedExportTools(
        output_directory="exports/",
        templates_directory="templates/",
        charts_directory="charts/"
    )
    
    # Recopilar datos para exportación
    export_data = {
        "statistics": analyzer.get_statistics(),
        "insights": [insight.__dict__ for insight in analyzer.insights],
        "recommendations": [rec.__dict__ for rec in analyzer.recommendations],
        "performance_data": await realtime_analyzer.get_performance_data(),
        "ml_results": ml_optimizer.get_model_performance(),
        "alerts": await realtime_analyzer.get_active_alerts(),
        "trends": await realtime_analyzer.get_trend_analysis()
    }
    
    # Exportar a JSON
    json_config = ExportConfig(
        format=ExportFormat.JSON,
        export_type=ExportType.FULL_REPORT,
        include_charts=True,
        filename_prefix="demo_export"
    )
    json_result = await export_tools.export_data(export_data, json_config)
    print(f"  📄 Exportación JSON: {'✅' if json_result.success else '❌'} - {json_result.file_path}")
    
    # Exportar a Excel
    excel_config = ExportConfig(
        format=ExportFormat.EXCEL,
        export_type=ExportType.STATISTICS,
        include_charts=True,
        filename_prefix="demo_stats"
    )
    excel_result = await export_tools.export_data(export_data, excel_config)
    print(f"  📊 Exportación Excel: {'✅' if excel_result.success else '❌'} - {excel_result.file_path}")
    
    # Exportar a PDF
    pdf_config = ExportConfig(
        format=ExportFormat.PDF,
        export_type=ExportType.FULL_REPORT,
        include_charts=True,
        filename_prefix="demo_report"
    )
    pdf_result = await export_tools.export_data(export_data, pdf_config)
    print(f"  📋 Exportación PDF: {'✅' if pdf_result.success else '❌'} - {pdf_result.file_path}")
    
    # Obtener historial de exportaciones
    export_history = export_tools.get_export_history()
    print(f"  📚 Historial de exportaciones: {len(export_history)} archivos")
    
    print(f"\n🎉 Demo Completado Exitosamente!")
    print("=" * 60)
    
    # Resumen final
    print(f"\n📈 Resumen del Análisis:")
    print(f"  • Total de documentos analizados: {stats['total_documents']}")
    print(f"  • Calidad promedio: {stats['average_quality']:.2f}")
    print(f"  • Insights generados: {len(insights)}")
    print(f"  • Recomendaciones generadas: {len(recommendations)}")
    print(f"  • Queries únicas: {stats['unique_queries']}")
    print(f"  • Modelos ML optimizados: {len(ml_optimizer.trained_models)}")
    print(f"  • Métricas en tiempo real: {sum(len(metrics) for metrics in realtime_analyzer.metrics_history.values())}")
    print(f"  • Notificaciones enviadas: {notif_stats['total_notifications']}")
    print(f"  • Archivos exportados: {len(export_history)}")
    
    return analyzer, report

async def demo_api_usage():
    """
    Demostración de uso de la API
    """
    print("\n🌐 Demo de Uso de API")
    print("=" * 40)
    
    import httpx
    
    base_url = "http://localhost:8000"
    
    # Ejemplo de uso de la API
    async with httpx.AsyncClient() as client:
        try:
            # Health check
            response = await client.get(f"{base_url}/health")
            print(f"✅ Health check: {response.status_code}")
            
            # Agregar documento
            doc_data = {
                "content": "Este es un documento de prueba para demostrar la API.",
                "query": "Escribe un documento de prueba",
                "metadata": {"demo": True}
            }
            
            response = await client.post(f"{base_url}/documents", json=doc_data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Documento agregado: {result['id']}")
                print(f"   Calidad: {result['quality_score']:.2f}")
            
            # Obtener estadísticas
            response = await client.get(f"{base_url}/statistics")
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Estadísticas obtenidas: {stats['total_documents']} documentos")
            
        except Exception as e:
            print(f"❌ Error en demo de API: {e}")
            print("   Asegúrate de que el servidor esté ejecutándose en localhost:8000")

if __name__ == "__main__":
    # Ejecutar demo principal
    asyncio.run(demo_ai_history_analysis())
    
    # Ejecutar demo de API (opcional)
    # asyncio.run(demo_api_usage())
