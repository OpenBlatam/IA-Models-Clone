"""
📊 Ejemplos de uso del Módulo de Advanced Analytics
Demuestra las capacidades avanzadas de análisis de datos del sistema Blaze AI
"""

import asyncio
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Importar el módulo de Advanced Analytics
from ..modules.advanced_analytics import (
    create_advanced_analytics_module,
    AnalyticsType,
    DataSourceType,
    VisualizationType
)


async def ejemplo_analisis_predictivo():
    """Ejemplo de análisis predictivo con machine learning"""
    print("🔮 Ejemplo de Análisis Predictivo")
    print("=" * 50)
    
    # Crear módulo de Advanced Analytics
    analytics_module = create_advanced_analytics_module(
        enabled_analytics=[AnalyticsType.PREDICTIVE],
        ml_models_enabled=True,
        max_data_size=500000
    )
    
    # Inicializar módulo
    await analytics_module.initialize()
    
    # Añadir fuente de datos
    data_source_id = await analytics_module.add_data_source(
        name="dataset_ventas",
        source_type=DataSourceType.DATABASE,
        connection_string="postgresql://localhost:5432/ventas_db"
    )
    
    print(f"✅ Fuente de datos añadida: {data_source_id}")
    
    # Ejecutar trabajo de análisis predictivo
    job_id = await analytics_module.execute_analytics_job(
        analytics_type=AnalyticsType.PREDICTIVE,
        data_source_id=data_source_id,
        parameters={
            "model_type": "regression",
            "target_variable": "ventas",
            "features": ["precio", "promocion", "estacion", "competencia"],
            "validation_split": 0.2
        }
    )
    
    print(f"✅ Trabajo de análisis predictivo iniciado: {job_id}")
    
    # Esperar y obtener resultados
    await asyncio.sleep(2)
    
    # Obtener estado del trabajo
    status = await analytics_module.get_job_status(job_id)
    print(f"📊 Estado del trabajo: {status}")
    
    # Obtener resultados
    result = await analytics_module.get_job_result(job_id)
    if result:
        print("🎯 Resultado de análisis predictivo:")
        print(f"  - Tipo de análisis: {result['analysis_type']}")
        print(f"  - Modelo entrenado: {result['model_id']}")
        print(f"  - Predicción: {result['prediction']}")
        print(f"  - Datos procesados: {result['data_processed']}")
    
    # Obtener métricas
    metrics = await analytics_module.get_metrics()
    print(f"📈 Métricas del módulo: {metrics}")
    
    # Apagar módulo
    await analytics_module.shutdown()
    print("✅ Análisis predictivo completado\n")


async def ejemplo_analisis_series_temporales():
    """Ejemplo de análisis de series temporales y forecasting"""
    print("⏰ Ejemplo de Análisis de Series Temporales")
    print("=" * 55)
    
    # Crear módulo con análisis de series temporales
    analytics_module = create_advanced_analytics_module(
        enabled_analytics=[AnalyticsType.TIME_SERIES],
        real_time_processing=True
    )
    
    await analytics_module.initialize()
    
    # Añadir fuente de datos temporal
    data_source_id = await analytics_module.add_data_source(
        name="datos_tiempo_real",
        source_type=DataSourceType.STREAM,
        connection_string="kafka://localhost:9092/topic_ventas"
    )
    
    print(f"✅ Fuente de datos temporal añadida: {data_source_id}")
    
    # Ejecutar análisis de series temporales
    job_id = await analytics_module.execute_analytics_job(
        analytics_type=AnalyticsType.TIME_SERIES,
        data_source_id=data_source_id,
        parameters={
            "forecast_periods": 24,
            "seasonality_detection": True,
            "trend_analysis": True,
            "confidence_intervals": True
        }
    )
    
    print(f"✅ Trabajo de series temporales iniciado: {job_id}")
    
    # Esperar resultados
    await asyncio.sleep(1.5)
    
    # Obtener resultados
    result = await analytics_module.get_job_result(job_id)
    if result:
        print("🎯 Resultado de análisis de series temporales:")
        print(f"  - Tipo de análisis: {result['analysis_type']}")
        print(f"  - Análisis de tendencia: {result['trend_analysis']}")
        print(f"  - Forecasting: {result['forecast']}")
        print(f"  - Datos procesados: {result['data_processed']}")
        
        # Mostrar detalles del forecasting
        forecast = result['forecast']
        print(f"  - Períodos de predicción: {forecast['forecast_periods']}")
        print(f"  - Precisión del modelo: {forecast['model_accuracy']:.3f}")
        print(f"  - Valores predichos: {forecast['forecast_values'][:5]}...")
    
    await analytics_module.shutdown()
    print("✅ Análisis de series temporales completado\n")


async def ejemplo_analisis_sentimientos():
    """Ejemplo de análisis de sentimientos y NLP"""
    print("😊 Ejemplo de Análisis de Sentimientos")
    print("=" * 50)
    
    # Crear módulo con análisis de sentimientos
    analytics_module = create_advanced_analytics_module(
        enabled_analytics=[AnalyticsType.SENTIMENT],
        privacy_preserving=True
    )
    
    await analytics_module.initialize()
    
    # Añadir fuente de datos de texto
    data_source_id = await analytics_module.add_data_source(
        name="redes_sociales",
        source_type=DataSourceType.API,
        connection_string="https://api.twitter.com/v2/tweets"
    )
    
    print(f"✅ Fuente de datos de texto añadida: {data_source_id}")
    
    # Ejecutar análisis de sentimientos
    job_id = await analytics_module.execute_analytics_job(
        analytics_type=AnalyticsType.SENTIMENT,
        data_source_id=data_source_id,
        parameters={
            "language": "es",
            "emotion_detection": True,
            "entity_extraction": True,
            "sentiment_threshold": 0.3
        }
    )
    
    print(f"✅ Trabajo de análisis de sentimientos iniciado: {job_id}")
    
    # Esperar resultados
    await asyncio.sleep(1.2)
    
    # Obtener resultados
    result = await analytics_module.get_job_result(job_id)
    if result:
        print("🎯 Resultado de análisis de sentimientos:")
        print(f"  - Tipo de análisis: {result['analysis_type']}")
        print(f"  - Datos procesados: {result['data_processed']}")
        
        # Mostrar análisis de sentimientos
        sentiment_analysis = result['sentiment_analysis']
        print(f"  - Análisis de sentimientos: {len(sentiment_analysis)} textos analizados")
        
        for i, sentiment in enumerate(sentiment_analysis[:3]):
            print(f"    Texto {i+1}: {sentiment['text'][:50]}...")
            print(f"      Sentimiento: {sentiment['sentiment']} (score: {sentiment['sentiment_score']:.3f})")
            print(f"      Confianza: {sentiment['confidence']:.3f}")
        
        # Mostrar extracción de entidades
        entity_extraction = result['entity_extraction']
        print(f"  - Extracción de entidades: {len(entity_extraction)} textos procesados")
        
        for i, entities in enumerate(entity_extraction[:2]):
            print(f"    Texto {i+1}: {entities['total_entities']} entidades encontradas")
            for entity in entities['entities'][:3]:
                print(f"      - {entity['text']} ({entity['type']}, confianza: {entity['confidence']:.3f})")
    
    await analytics_module.shutdown()
    print("✅ Análisis de sentimientos completado\n")


async def ejemplo_analisis_grafos():
    """Ejemplo de análisis de grafos y redes complejas"""
    print("🕸️ Ejemplo de Análisis de Grafos")
    print("=" * 45)
    
    # Crear módulo con análisis de grafos
    analytics_module = create_advanced_analytics_module(
        enabled_analytics=[AnalyticsType.GRAPH],
        max_data_size=100000
    )
    
    await analytics_module.initialize()
    
    # Añadir fuente de datos de red
    data_source_id = await analytics_module.add_data_source(
        name="red_social",
        source_type=DataSourceType.DATABASE,
        connection_string="neo4j://localhost:7687/social_network"
    )
    
    print(f"✅ Fuente de datos de red añadida: {data_source_id}")
    
    # Ejecutar análisis de grafos
    job_id = await analytics_module.execute_analytics_job(
        analytics_type=AnalyticsType.GRAPH,
        data_source_id=data_source_id,
        parameters={
            "centrality_analysis": True,
            "community_detection": True,
            "influence_analysis": True,
            "path_analysis": True
        }
    )
    
    print(f"✅ Trabajo de análisis de grafos iniciado: {job_id}")
    
    # Esperar resultados
    await asyncio.sleep(1.8)
    
    # Obtener resultados
    result = await analytics_module.get_job_result(job_id)
    if result:
        print("🎯 Resultado de análisis de grafos:")
        print(f"  - Tipo de análisis: {result['analysis_type']}")
        print(f"  - ID del grafo: {result['graph_id']}")
        print(f"  - Datos procesados: {result['data_processed']}")
        
        # Mostrar análisis de centralidad
        centrality = result['centrality_analysis']
        print(f"  - Análisis de centralidad: {len(centrality['centrality_metrics'])} nodos analizados")
        
        # Mostrar algunas métricas de centralidad
        centrality_metrics = centrality['centrality_metrics']
        sample_nodes = list(centrality_metrics.keys())[:3]
        for node in sample_nodes:
            metrics = centrality_metrics[node]
            print(f"    Nodo {node}:")
            print(f"      - Centralidad de grado: {metrics['degree_centrality']:.3f}")
            print(f"      - Centralidad de intermediación: {metrics['betweenness_centrality']:.3f}")
            print(f"      - Centralidad de cercanía: {metrics['closeness_centrality']:.3f}")
        
        # Mostrar detección de comunidades
        communities = result['community_detection']
        print(f"  - Detección de comunidades: {communities['total_communities']} comunidades encontradas")
        print(f"    Score de modularidad: {communities['modularity_score']:.3f}")
        
        for community in communities['communities'][:3]:
            print(f"    Comunidad {community['community_id']}: {community['size']} nodos")
            print(f"      Modularidad: {community['modularity']:.3f}")
    
    await analytics_module.shutdown()
    print("✅ Análisis de grafos completado\n")


async def ejemplo_analisis_geoespacial():
    """Ejemplo de análisis geoespacial y mapeo"""
    print("🗺️ Ejemplo de Análisis Geoespacial")
    print("=" * 50)
    
    # Crear módulo con análisis geoespacial
    analytics_module = create_advanced_analytics_module(
        enabled_analytics=[AnalyticsType.GEOSPATIAL],
        visualization_types=[VisualizationType.GEO_MAP, VisualizationType.HEATMAP]
    )
    
    await analytics_module.initialize()
    
    # Añadir fuente de datos geoespacial
    data_source_id = await analytics_module.add_data_source(
        name="datos_ubicacion",
        source_type=DataSourceType.API,
        connection_string="https://api.location.com/v1/coordinates"
    )
    
    print(f"✅ Fuente de datos geoespacial añadida: {data_source_id}")
    
    # Ejecutar análisis geoespacial
    job_id = await analytics_module.execute_analytics_job(
        analytics_type=AnalyticsType.GEOSPATIAL,
        data_source_id=data_source_id,
        parameters={
            "spatial_analysis": True,
            "hotspot_detection": True,
            "distance_calculation": True,
            "clustering_analysis": True
        }
    )
    
    print(f"✅ Trabajo de análisis geoespacial iniciado: {job_id}")
    
    # Esperar resultados
    await asyncio.sleep(1.5)
    
    # Obtener resultados
    result = await analytics_module.get_job_result(job_id)
    if result:
        print("🎯 Resultado de análisis geoespacial:")
        print(f"  - Tipo de análisis: {result['analysis_type']}")
        print(f"  - Datos procesados: {result['data_processed']}")
        
        # Mostrar patrones espaciales
        spatial_patterns = result['spatial_patterns']
        print(f"  - Análisis de patrones espaciales:")
        print(f"    Total de puntos: {spatial_patterns['total_points']}")
        
        metrics = spatial_patterns['spatial_metrics']
        print(f"    - Autocorrelación espacial: {metrics['spatial_autocorrelation']:.3f}")
        print(f"    - Score de hotspot: {metrics['hotspot_score']:.3f}")
        print(f"    - Índice de clustering: {metrics['clustering_index']:.3f}")
        print(f"    - Índice de dispersión: {metrics['dispersion_index']:.3f}")
        
        # Mostrar hotspots detectados
        hotspots = spatial_patterns['hotspots']
        print(f"    - Hotspots detectados: {len(hotspots)}")
        for i, hotspot in enumerate(hotspots[:3]):
            print(f"      Hotspot {i+1}: Centro {hotspot['center']}, Radio: {hotspot['radius']:.2f}")
            print(f"        Intensidad: {hotspot['intensity']:.3f}")
        
        # Mostrar análisis de distancias
        distances = result['distance_analysis']
        print(f"  - Análisis de distancias:")
        print(f"    Total de puntos: {distances['total_points']}")
        print(f"    Distancia mínima: {distances['min_distance']:.2f}")
        print(f"    Distancia máxima: {distances['max_distance']:.2f}")
        print(f"    Distancia promedio: {distances['average_distance']:.2f}")
    
    await analytics_module.shutdown()
    print("✅ Análisis geoespacial completado\n")


async def ejemplo_analisis_big_data():
    """Ejemplo de análisis de big data con procesamiento distribuido"""
    print("📊 Ejemplo de Análisis de Big Data")
    print("=" * 50)
    
    # Crear módulo con capacidades de big data
    analytics_module = create_advanced_analytics_module(
        enabled_analytics=[AnalyticsType.BIG_DATA],
        max_data_size=10000000,  # 10M registros
        batch_size=50000
    )
    
    await analytics_module.initialize()
    
    # Añadir fuente de datos de big data
    data_source_id = await analytics_module.add_data_source(
        name="dataset_masivo",
        source_type=DataSourceType.CLOUD,
        connection_string="s3://bucket-analytics/large-dataset/"
    )
    
    print(f"✅ Fuente de datos de big data añadida: {data_source_id}")
    
    # Ejecutar análisis de big data
    job_id = await analytics_module.execute_analytics_job(
        analytics_type=AnalyticsType.BIG_DATA,
        data_source_id=data_source_id,
        parameters={
            "processing_type": "distributed",
            "aggregation_functions": ["sum", "avg", "count", "max", "min"],
            "group_by_columns": ["region", "category", "date"],
            "chunk_size": 50000
        }
    )
    
    print(f"✅ Trabajo de big data iniciado: {job_id}")
    
    # Esperar resultados
    await asyncio.sleep(2.2)
    
    # Obtener resultados
    result = await analytics_module.get_job_result(job_id)
    if result:
        print("🎯 Resultado de análisis de big data:")
        print(f"  - Tipo de análisis: {result['analysis_type']}")
        print(f"  - Datos procesados: {result['data_processed']}")
        
        # Mostrar resultado de procesamiento
        processing_result = result['processing_result']
        print(f"  - Resultado de procesamiento:")
        print(f"    ID del trabajo: {processing_result['job_id']}")
        print(f"    Operación: {processing_result['operation']}")
        print(f"    Total de registros: {processing_result['total_records']}")
        print(f"    Chunks procesados: {processing_result['processed_chunks']}")
        print(f"    Tiempo de procesamiento: {processing_result['processing_time']:.2f}s")
        
        # Mostrar resultado de agregación
        aggregation_result = result['aggregation_result']
        print(f"  - Resultado de agregación:")
        print(f"    Columnas de agrupación: {aggregation_result['group_by']}")
        print(f"    Funciones de agregación: {aggregation_result['aggregations']}")
        print(f"    Total de grupos: {aggregation_result['total_groups']}")
        
        # Mostrar algunos resultados agregados
        aggregated_results = aggregation_result['aggregated_results']
        for i, group_result in enumerate(aggregated_results[:3]):
            print(f"    Grupo {i+1}:")
            print(f"      Valores de grupo: {group_result['group_values']}")
            print(f"      Valores agregados: {group_result['aggregated_values']}")
    
    await analytics_module.shutdown()
    print("✅ Análisis de big data completado\n")


async def ejemplo_visualizacion_avanzada():
    """Ejemplo de visualización avanzada e interactiva"""
    print("🎨 Ejemplo de Visualización Avanzada")
    print("=" * 50)
    
    # Crear módulo con capacidades de visualización
    analytics_module = create_advanced_analytics_module(
        visualization_types=[
            VisualizationType.LINE_CHART,
            VisualizationType.BAR_CHART,
            VisualizationType.SCATTER_PLOT,
            VisualizationType.HEATMAP,
            VisualizationType.DASHBOARD
        ]
    )
    
    await analytics_module.initialize()
    
    # Crear diferentes tipos de gráficos
    print("🎨 Creando gráficos interactivos...")
    
    # Gráfico de líneas
    line_chart_data = {
        "x": list(range(1, 11)),
        "y": [np.random.uniform(10, 100) for _ in range(10)],
        "title": "Tendencia de Ventas",
        "x_label": "Mes",
        "y_label": "Ventas (Miles)"
    }
    
    line_chart_id = await analytics_module.visualization.create_chart(
        chart_type=VisualizationType.LINE_CHART,
        data=line_chart_data,
        chart_name="tendencia_ventas"
    )
    
    print(f"✅ Gráfico de líneas creado: {line_chart_id}")
    
    # Gráfico de barras
    bar_chart_data = {
        "categories": ["Q1", "Q2", "Q3", "Q4"],
        "values": [np.random.uniform(50, 200) for _ in range(4)],
        "title": "Ventas por Trimestre",
        "x_label": "Trimestre",
        "y_label": "Ventas (Miles)"
    }
    
    bar_chart_id = await analytics_module.visualization.create_chart(
        chart_type=VisualizationType.BAR_CHART,
        data=bar_chart_data,
        chart_name="ventas_trimestre"
    )
    
    print(f"✅ Gráfico de barras creado: {bar_chart_id}")
    
    # Gráfico de dispersión
    scatter_data = {
        "x": [np.random.uniform(0, 100) for _ in range(20)],
        "y": [np.random.uniform(0, 100) for _ in range(20)],
        "title": "Correlación Precio vs. Demanda",
        "x_label": "Precio",
        "y_label": "Demanda"
    }
    
    scatter_chart_id = await analytics_module.visualization.create_chart(
        chart_type=VisualizationType.SCATTER_PLOT,
        data=scatter_data,
        chart_name="correlacion_precio_demanda"
    )
    
    print(f"✅ Gráfico de dispersión creado: {scatter_chart_id}")
    
    # Crear dashboard
    dashboard_layout = {
        "title": "Dashboard de Ventas",
        "grid": "2x2",
        "charts": [line_chart_id, bar_chart_id, scatter_chart_id],
        "refresh_rate": 60,
        "theme": "light"
    }
    
    dashboard_id = await analytics_module.visualization.create_dashboard(
        charts=[line_chart_id, bar_chart_id, scatter_chart_id],
        layout=dashboard_layout
    )
    
    print(f"✅ Dashboard creado: {dashboard_id}")
    
    # Mostrar información de los gráficos
    charts = analytics_module.visualization.charts
    print(f"\n📊 Información de gráficos creados:")
    print(f"  - Total de gráficos: {len(charts)}")
    
    for chart_id, chart_info in charts.items():
        print(f"    - {chart_info['name']}: {chart_info['type']}")
        print(f"      Interactivo: {chart_info['interactive']}")
        print(f"      Responsivo: {chart_info['responsive']}")
        print(f"      Creado: {chart_info['created_at']}")
    
    await analytics_module.shutdown()
    print("✅ Visualización avanzada completada\n")


async def ejemplo_integracion_completa():
    """Ejemplo de integración completa con múltiples tipos de análisis"""
    print("🔗 Ejemplo de Integración Completa")
    print("=" * 50)
    
    # Crear módulo con todas las capacidades
    analytics_module = create_advanced_analytics_module(
        enabled_analytics=[
            AnalyticsType.PREDICTIVE,
            AnalyticsType.TIME_SERIES,
            AnalyticsType.SENTIMENT,
            AnalyticsType.GRAPH,
            AnalyticsType.GEOSPATIAL,
            AnalyticsType.BIG_DATA
        ],
        data_sources=[
            DataSourceType.DATABASE,
            DataSourceType.API,
            DataSourceType.STREAM,
            DataSourceType.CLOUD
        ],
        visualization_types=[
            VisualizationType.LINE_CHART,
            VisualizationType.BAR_CHART,
            VisualizationType.SCATTER_PLOT,
            VisualizationType.HEATMAP,
            VisualizationType.NETWORK_GRAPH,
            VisualizationType.GEO_MAP,
            VisualizationType.DASHBOARD
        ]
    )
    
    await analytics_module.initialize()
    
    # Añadir múltiples fuentes de datos
    print("🔗 Configurando múltiples fuentes de datos...")
    
    sources = {
        "ventas_db": await analytics_module.add_data_source(
            "Base de Datos de Ventas",
            DataSourceType.DATABASE,
            "postgresql://localhost:5432/ventas_db"
        ),
        "redes_sociales": await analytics_module.add_data_source(
            "API de Redes Sociales",
            DataSourceType.API,
            "https://api.social.com/v2/posts"
        ),
        "sensores_iot": await analytics_module.add_data_source(
            "Stream de Sensores IoT",
            DataSourceType.STREAM,
            "kafka://localhost:9092/sensors"
        ),
        "datos_cloud": await analytics_module.add_data_source(
            "Almacenamiento en la Nube",
            DataSourceType.CLOUD,
            "s3://analytics-bucket/raw-data/"
        )
    }
    
    print(f"✅ {len(sources)} fuentes de datos configuradas")
    
    # Ejecutar múltiples trabajos de análisis
    print("🚀 Ejecutando análisis integrado...")
    
    jobs = {}
    
    # Análisis predictivo
    jobs["predictive"] = await analytics_module.execute_analytics_job(
        AnalyticsType.PREDICTIVE,
        sources["ventas_db"],
        {"model_type": "regression", "target": "ventas"}
    )
    
    # Análisis de series temporales
    jobs["time_series"] = await analytics_module.execute_analytics_job(
        AnalyticsType.TIME_SERIES,
        sources["sensores_iot"],
        {"forecast_periods": 48, "seasonality": True}
    )
    
    # Análisis de sentimientos
    jobs["sentiment"] = await analytics_module.execute_analytics_job(
        AnalyticsType.SENTIMENT,
        sources["redes_sociales"],
        {"language": "es", "emotions": True}
    )
    
    # Análisis de grafos
    jobs["graph"] = await analytics_module.execute_analytics_job(
        AnalyticsType.GRAPH,
        sources["ventas_db"],
        {"centrality": True, "communities": True}
    )
    
    # Análisis geoespacial
    jobs["geospatial"] = await analytics_module.execute_analytics_job(
        AnalyticsType.GEOSPATIAL,
        sources["datos_cloud"],
        {"hotspots": True, "clustering": True}
    )
    
    # Análisis de big data
    jobs["big_data"] = await analytics_module.execute_analytics_job(
        AnalyticsType.BIG_DATA,
        sources["datos_cloud"],
        {"aggregation": True, "distributed": True}
    )
    
    print(f"✅ {len(jobs)} trabajos de análisis iniciados")
    
    # Esperar y recopilar resultados
    print("\n⏳ Esperando resultados...")
    await asyncio.sleep(4)
    
    # Mostrar resultados integrados
    print("\n🎯 Resultados de Análisis Integrado:")
    print("-" * 40)
    
    for analysis_type, job_id in jobs.items():
        status = await analytics_module.get_job_status(job_id)
        result = await analytics_module.get_job_result(job_id)
        
        print(f"\n🔬 {analysis_type.upper()}:")
        print(f"  - Estado: {status['status']}")
        print(f"  - Tiempo: {status.get('execution_time', 0):.3f}s")
        print(f"  - Datos procesados: {status['data_size_processed']}")
        
        if result and 'error' not in result:
            print(f"  - Tipo: {result['analysis_type']}")
            print(f"  - Datos procesados: {result['data_processed']}")
    
    # Métricas finales
    metrics = await analytics_module.get_metrics()
    print(f"\n📊 Métricas Finales del Sistema:")
    print(f"  - Total de trabajos: {metrics.total_jobs}")
    print(f"  - Trabajos completados: {metrics.completed_jobs}")
    print(f"  - Trabajos fallidos: {metrics.failed_jobs}")
    print(f"  - Total de datos procesados: {metrics.total_data_processed}")
    print(f"  - Tiempo promedio: {metrics.average_execution_time:.3f}s")
    print(f"  - Modelos ML entrenados: {metrics.ml_models_trained}")
    print(f"  - Predicciones realizadas: {metrics.predictions_made}")
    
    # Estado de salud del módulo
    health = await analytics_module.get_health_status()
    print(f"\n🏥 Estado de Salud del Sistema:")
    print(f"  - Estado: {health['status']}")
    print(f"  - Fuentes de datos: {health['data_sources_count']}")
    print(f"  - Trabajos activos: {health['active_jobs']}")
    print(f"  - Scikit-learn: {'Disponible' if health['sklearn_available'] else 'Simulado'}")
    print(f"  - Plotly: {'Disponible' if health['plotly_available'] else 'Simulado'}")
    print(f"  - NetworkX: {'Disponible' if health['networkx_available'] else 'Simulado'}")
    
    await analytics_module.shutdown()
    print("\n✅ Integración completa completada\n")


async def main():
    """Función principal que ejecuta todos los ejemplos"""
    print("📊 SISTEMA BLAZE AI - MÓDULO DE ADVANCED ANALYTICS")
    print("=" * 65)
    print("Ejecutando ejemplos de análisis avanzado de datos...\n")
    
    try:
        # Ejecutar ejemplos secuencialmente
        await ejemplo_analisis_predictivo()
        await ejemplo_analisis_series_temporales()
        await ejemplo_analisis_sentimientos()
        await ejemplo_analisis_grafos()
        await ejemplo_analisis_geoespacial()
        await ejemplo_analisis_big_data()
        await ejemplo_visualizacion_avanzada()
        await ejemplo_integracion_completa()
        
        print("🎉 ¡Todos los ejemplos de Advanced Analytics ejecutados exitosamente!")
        print("🚀 El sistema Blaze AI ahora tiene capacidades avanzadas de análisis de datos")
        
    except Exception as e:
        print(f"❌ Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ejecutar ejemplos
    asyncio.run(main())

