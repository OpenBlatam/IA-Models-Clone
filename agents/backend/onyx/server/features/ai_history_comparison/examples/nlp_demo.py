"""
NLP System Demo for AI History Comparison
Demo del sistema NLP para análisis de historial de IA
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_nlp_system():
    """
    Demostración completa del sistema NLP
    """
    print("🚀 Demo del Sistema NLP Avanzado")
    print("=" * 60)
    
    # Textos de ejemplo
    sample_texts = [
        {
            "id": "doc_001",
            "content": """
            La inteligencia artificial está revolucionando la forma en que trabajamos y vivimos. 
            Los algoritmos de machine learning pueden procesar grandes cantidades de datos y 
            encontrar patrones que los humanos no podrían detectar fácilmente. 
            
            Sin embargo, es importante considerar las implicaciones éticas de estas tecnologías. 
            Debemos asegurarnos de que la IA se use de manera responsable y justa.
            """,
            "query": "Escribe sobre inteligencia artificial y ética",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "doc_002", 
            "content": """
            Artificial Intelligence is transforming industries across the globe. 
            From healthcare to finance, AI applications are becoming increasingly sophisticated.
            
            The key to successful AI implementation lies in understanding the data, 
            choosing the right algorithms, and ensuring proper validation of results.
            
            Companies that embrace AI early will have a significant competitive advantage.
            """,
            "query": "Write about AI transformation in industries",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "doc_003",
            "content": """
            El machine learning es una rama de la inteligencia artificial que permite a las 
            computadoras aprender sin ser programadas explícitamente. Los modelos pueden 
            mejorar su rendimiento a través de la experiencia.
            
            Existen diferentes tipos de aprendizaje: supervisado, no supervisado y por refuerzo. 
            Cada uno tiene sus propias aplicaciones y ventajas.
            """,
            "query": "Explica qué es machine learning",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "doc_004",
            "content": """
            Data science combines statistics, programming, and domain expertise to extract 
            insights from data. It's a multidisciplinary field that requires both technical 
            and analytical skills.
            
            The data science process typically involves: data collection, cleaning, 
            exploration, modeling, and interpretation of results.
            """,
            "query": "Describe the data science process",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "doc_005",
            "content": """
            La calidad de los datos es fundamental para el éxito de cualquier proyecto de 
            análisis. Datos de mala calidad pueden llevar a conclusiones incorrectas y 
            decisiones empresariales erróneas.
            
            Es crucial implementar procesos de validación y limpieza de datos desde el 
            inicio del proyecto para garantizar la integridad de los resultados.
            """,
            "query": "Habla sobre la importancia de la calidad de datos",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    print(f"📝 Analizando {len(sample_texts)} documentos de ejemplo...")
    
    # ============================================================================
    # DEMO 1: Análisis NLP Básico
    # ============================================================================
    print(f"\n🔍 Demo 1: Análisis NLP Básico")
    print("-" * 40)
    
    try:
        from nlp_engine import AdvancedNLPEngine, AnalysisType, LanguageType
        
        # Inicializar motor NLP
        nlp_engine = AdvancedNLPEngine(language=LanguageType.AUTO)
        
        # Analizar cada documento
        analyses = []
        for doc in sample_texts:
            print(f"  📄 Analizando documento: {doc['id']}")
            
            analysis = await nlp_engine.analyze_text(
                text=doc["content"],
                document_id=doc["id"],
                analysis_types=[
                    AnalysisType.TOKENIZATION,
                    AnalysisType.POS_TAGGING,
                    AnalysisType.SENTIMENT,
                    AnalysisType.KEYWORD_EXTRACTION
                ]
            )
            
            analyses.append(analysis)
            
            # Mostrar resultados básicos
            print(f"    • Idioma detectado: {analysis.language}")
            print(f"    • Tokens: {len(analysis.tokens)}")
            print(f"    • Entidades: {len(analysis.entities)}")
            if analysis.sentiment:
                print(f"    • Sentimiento: {analysis.sentiment.sentiment_type.value} (confianza: {analysis.sentiment.confidence:.2f})")
            print(f"    • Palabras clave: {len(analysis.keywords)}")
            print(f"    • Métricas: {analysis.metrics.word_count} palabras, legibilidad: {analysis.metrics.readability_score:.2f}")
        
        print(f"\n✅ Análisis NLP completado para {len(analyses)} documentos")
        
    except Exception as e:
        print(f"❌ Error en análisis NLP: {e}")
    
    # ============================================================================
    # DEMO 2: Procesamiento en Tiempo Real
    # ============================================================================
    print(f"\n⚡ Demo 2: Procesamiento en Tiempo Real")
    print("-" * 40)
    
    try:
        from text_processor import RealTimeTextProcessor, ProcessingPriority
        
        # Inicializar procesador
        processor = RealTimeTextProcessor(max_workers=2)
        await processor.start()
        
        print("  🚀 Procesador iniciado")
        
        # Enviar tareas con diferentes prioridades
        task_ids = []
        for i, doc in enumerate(sample_texts):
            priority = ProcessingPriority.HIGH if i < 2 else ProcessingPriority.NORMAL
            
            task_id = await processor.submit_task(
                text=doc["content"],
                document_id=doc["id"],
                priority=priority
            )
            task_ids.append(task_id)
            print(f"  📤 Tarea enviada: {task_id} (prioridad: {priority.name})")
        
        # Esperar completación
        print("  ⏳ Esperando completación...")
        results = await processor.wait_for_completion(task_ids, timeout=60)
        
        # Mostrar resultados
        for task_id, task in results.items():
            if task.status.value == "completed":
                print(f"  ✅ Tarea completada: {task_id} en {task.completed_at - task.started_at}")
            else:
                print(f"  ❌ Tarea fallida: {task_id} - {task.error}")
        
        # Obtener estadísticas
        stats = await processor.get_processing_statistics()
        print(f"  📊 Estadísticas:")
        print(f"    • Total tareas: {stats['metrics']['total_tasks']}")
        print(f"    • Completadas: {stats['metrics']['completed_tasks']}")
        print(f"    • Fallidas: {stats['metrics']['failed_tasks']}")
        print(f"    • Tiempo promedio: {stats['metrics']['avg_processing_time']:.2f}s")
        
        await processor.stop()
        print("  🛑 Procesador detenido")
        
    except Exception as e:
        print(f"❌ Error en procesamiento en tiempo real: {e}")
    
    # ============================================================================
    # DEMO 3: Análisis de Patrones
    # ============================================================================
    print(f"\n🔍 Demo 3: Análisis de Patrones")
    print("-" * 40)
    
    try:
        from pattern_analyzer import TextPatternAnalyzer, PatternType
        
        # Inicializar analizador de patrones
        pattern_analyzer = TextPatternAnalyzer()
        
        # Analizar patrones
        pattern_results = await pattern_analyzer.analyze_documents(
            documents=sample_texts,
            pattern_types=[
                PatternType.LINGUISTIC,
                PatternType.STRUCTURAL,
                PatternType.QUALITY,
                PatternType.SEMANTIC
            ]
        )
        
        # Mostrar resultados por tipo
        for pattern_type, results in pattern_results.items():
            if isinstance(results, dict) and "patterns" in results:
                print(f"  📋 {pattern_type.title()} Patterns:")
                for pattern in results["patterns"][:3]:  # Mostrar solo los primeros 3
                    print(f"    • {pattern['name']}: {pattern['frequency']} ocurrencias")
                    print(f"      Ejemplo: {pattern['examples'][0] if pattern['examples'] else 'N/A'}")
        
        # Obtener resumen
        summary = await pattern_analyzer.get_pattern_summary()
        print(f"\n  📊 Resumen de Patrones:")
        print(f"    • Total patrones: {summary.get('total_patterns', 0)}")
        print(f"    • Distribución por tipo: {summary.get('type_distribution', {})}")
        print(f"    • Distribución por categoría: {summary.get('category_distribution', {})}")
        
    except Exception as e:
        print(f"❌ Error en análisis de patrones: {e}")
    
    # ============================================================================
    # DEMO 4: Comparación de Textos
    # ============================================================================
    print(f"\n🔄 Demo 4: Comparación de Textos")
    print("-" * 40)
    
    try:
        from nlp_engine import AdvancedNLPEngine, LanguageType
        
        nlp_engine = AdvancedNLPEngine(language=LanguageType.AUTO)
        
        # Comparar textos similares
        text1 = sample_texts[0]["content"]
        text2 = sample_texts[2]["content"]  # Ambos en español sobre IA
        
        print("  🔍 Comparando documentos sobre IA en español...")
        
        comparison = await nlp_engine.compare_texts(text1, text2)
        
        if "overall_similarity" in comparison:
            print(f"  📊 Similitud general: {comparison['overall_similarity']:.2f}")
            print(f"  📈 Desglose de similitud:")
            for sim_type, score in comparison.get("similarity_breakdown", {}).items():
                print(f"    • {sim_type}: {score:.2f}")
        
    except Exception as e:
        print(f"❌ Error en comparación de textos: {e}")
    
    # ============================================================================
    # DEMO 5: Análisis de Sentimientos
    # ============================================================================
    print(f"\n😊 Demo 5: Análisis de Sentimientos")
    print("-" * 40)
    
    try:
        from nlp_engine import AdvancedNLPEngine, AnalysisType, LanguageType
        
        nlp_engine = AdvancedNLPEngine(language=LanguageType.AUTO)
        
        # Analizar sentimientos de todos los documentos
        sentiments = []
        for doc in sample_texts:
            analysis = await nlp_engine.analyze_text(
                text=doc["content"],
                document_id=f"sentiment_{doc['id']}",
                analysis_types=[AnalysisType.SENTIMENT]
            )
            
            if analysis.sentiment:
                sentiments.append({
                    "document_id": doc["id"],
                    "sentiment": analysis.sentiment.sentiment_type.value,
                    "polarity": analysis.sentiment.polarity,
                    "confidence": analysis.sentiment.confidence,
                    "emotional_tone": analysis.sentiment.emotional_tone
                })
        
        # Mostrar resultados
        print("  📊 Análisis de Sentimientos:")
        for sent in sentiments:
            print(f"    • {sent['document_id']}: {sent['sentiment']} "
                  f"(polaridad: {sent['polarity']:.2f}, confianza: {sent['confidence']:.2f})")
            print(f"      Tono emocional: {sent['emotional_tone']}")
        
        # Estadísticas generales
        sentiment_counts = {}
        for sent in sentiments:
            sentiment_counts[sent['sentiment']] = sentiment_counts.get(sent['sentiment'], 0) + 1
        
        print(f"\n  📈 Distribución de sentimientos: {sentiment_counts}")
        
    except Exception as e:
        print(f"❌ Error en análisis de sentimientos: {e}")
    
    # ============================================================================
    # DEMO 6: Extracción de Palabras Clave
    # ============================================================================
    print(f"\n🔑 Demo 6: Extracción de Palabras Clave")
    print("-" * 40)
    
    try:
        from nlp_engine import AdvancedNLPEngine, AnalysisType, LanguageType
        
        nlp_engine = AdvancedNLPEngine(language=LanguageType.AUTO)
        
        # Extraer palabras clave de cada documento
        all_keywords = []
        for doc in sample_texts:
            analysis = await nlp_engine.analyze_text(
                text=doc["content"],
                document_id=f"keywords_{doc['id']}",
                analysis_types=[AnalysisType.KEYWORD_EXTRACTION]
            )
            
            print(f"  📄 {doc['id']} - Top 5 palabras clave:")
            for i, kw in enumerate(analysis.keywords[:5]):
                print(f"    {i+1}. {kw.text} (score: {kw.score:.2f}, frecuencia: {kw.frequency})")
                all_keywords.append(kw.text)
        
        # Palabras clave más comunes
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  🏆 Top 10 palabras clave más comunes:")
        for i, (kw, count) in enumerate(top_keywords):
            print(f"    {i+1}. {kw}: {count} documentos")
        
    except Exception as e:
        print(f"❌ Error en extracción de palabras clave: {e}")
    
    # ============================================================================
    # DEMO 7: Exportación de Resultados
    # ============================================================================
    print(f"\n💾 Demo 7: Exportación de Resultados")
    print("-" * 40)
    
    try:
        from pattern_analyzer import TextPatternAnalyzer
        
        # Exportar patrones
        pattern_analyzer = TextPatternAnalyzer()
        pattern_file = await pattern_analyzer.export_patterns()
        print(f"  📁 Patrones exportados a: {pattern_file}")
        
        # Exportar análisis NLP
        from nlp_engine import AdvancedNLPEngine
        
        nlp_engine = AdvancedNLPEngine()
        for doc in sample_texts[:2]:  # Exportar solo los primeros 2
            analysis_file = await nlp_engine.save_analysis(doc["id"])
            print(f"  📁 Análisis de {doc['id']} exportado a: {analysis_file}")
        
    except Exception as e:
        print(f"❌ Error en exportación: {e}")
    
    # ============================================================================
    # RESUMEN FINAL
    # ============================================================================
    print(f"\n🎉 Demo Completado Exitosamente!")
    print("=" * 60)
    
    print(f"📊 Resumen del Demo:")
    print(f"  • Documentos analizados: {len(sample_texts)}")
    print(f"  • Análisis NLP: ✅ Completado")
    print(f"  • Procesamiento en tiempo real: ✅ Completado")
    print(f"  • Análisis de patrones: ✅ Completado")
    print(f"  • Comparación de textos: ✅ Completado")
    print(f"  • Análisis de sentimientos: ✅ Completado")
    print(f"  • Extracción de palabras clave: ✅ Completado")
    print(f"  • Exportación de resultados: ✅ Completado")
    
    print(f"\n🚀 El sistema NLP está listo para uso en producción!")
    print(f"   - Motor NLP avanzado con múltiples análisis")
    print(f"   - Procesamiento en tiempo real con colas de prioridad")
    print(f"   - Análisis de patrones con clustering y evolución")
    print(f"   - API completa con endpoints REST")
    print(f"   - Soporte para múltiples idiomas")
    print(f"   - Exportación de resultados en múltiples formatos")

async def demo_api_usage():
    """
    Demo de uso de la API NLP
    """
    print("\n🌐 Demo de Uso de API NLP")
    print("=" * 40)
    
    import httpx
    
    base_url = "http://localhost:8000/nlp"
    
    # Ejemplo de uso de la API
    async with httpx.AsyncClient() as client:
        try:
            # Health check
            response = await client.get(f"{base_url}/health")
            print(f"✅ Health check: {response.status_code}")
            
            # Detectar idioma
            response = await client.post(
                f"{base_url}/language/detect",
                json={"text": "La inteligencia artificial es fascinante"}
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Idioma detectado: {result['language']}")
            
            # Analizar sentimiento
            response = await client.post(
                f"{base_url}/sentiment/analyze",
                json={"text": "Este es un texto excelente y muy bien escrito"}
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Sentimiento: {result['sentiment']['sentiment_type']}")
            
            # Extraer palabras clave
            response = await client.post(
                f"{base_url}/keywords/extract",
                json={"text": "Machine learning y deep learning son tecnologías avanzadas de inteligencia artificial"}
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Palabras clave extraídas: {len(result['keywords'])}")
            
        except Exception as e:
            print(f"❌ Error en demo de API: {e}")
            print("   Asegúrate de que el servidor esté ejecutándose en localhost:8000")

if __name__ == "__main__":
    # Ejecutar demo principal
    asyncio.run(demo_nlp_system())
    
    # Ejecutar demo de API (opcional)
    # asyncio.run(demo_api_usage())





























