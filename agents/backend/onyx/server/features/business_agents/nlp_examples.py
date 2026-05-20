"""
NLP System Examples
==================

Ejemplos de uso del sistema NLP avanzado para diferentes casos de negocio.
"""

import asyncio
import json
from typing import Dict, List, Any
from datetime import datetime

# Import the NLP systems
from .nlp_system import nlp_system
from .advanced_nlp_system import advanced_nlp_system
from .nlp_config import nlp_config

class NLPExamples:
    """Ejemplos de uso del sistema NLP."""
    
    def __init__(self):
        self.nlp = nlp_system
        self.advanced_nlp = advanced_nlp_system
    
    async def example_sentiment_analysis(self):
        """Ejemplo de análisis de sentimientos."""
        print("=== Análisis de Sentimientos ===")
        
        texts = [
            "I love this product! It's amazing and works perfectly.",
            "This is terrible. I hate it and want my money back.",
            "The product is okay, nothing special but it works.",
            "Excelente producto, muy recomendado para todos.",
            "Ce produit est fantastique, je le recommande vivement."
        ]
        
        for text in texts:
            print(f"\nTexto: {text}")
            
            # Análisis básico
            sentiment = await self.nlp.analyze_sentiment(text)
            print(f"Sentimiento: {sentiment}")
            
            # Análisis avanzado
            advanced_result = await self.advanced_nlp.analyze_text_advanced(text)
            ensemble_sentiment = advanced_result.get('sentiment', {}).get('ensemble', {})
            print(f"Sentimiento Ensemble: {ensemble_sentiment}")
    
    async def example_entity_extraction(self):
        """Ejemplo de extracción de entidades."""
        print("\n=== Extracción de Entidades ===")
        
        text = """
        Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
        The company is headquartered in Cupertino, California, United States.
        Apple's current CEO is Tim Cook, and the company is worth over $3 trillion.
        Contact us at info@apple.com or call (408) 996-1010.
        Visit our website at https://www.apple.com for more information.
        """
        
        print(f"Texto: {text}")
        
        # Extracción básica
        entities = await self.nlp.extract_entities(text)
        print(f"\nEntidades básicas: {entities}")
        
        # Extracción avanzada
        advanced_result = await self.advanced_nlp.analyze_text_advanced(text)
        advanced_entities = advanced_result.get('entities', [])
        print(f"\nEntidades avanzadas: {advanced_entities}")
    
    async def example_keyword_extraction(self):
        """Ejemplo de extracción de palabras clave."""
        print("\n=== Extracción de Palabras Clave ===")
        
        business_text = """
        Our company specializes in artificial intelligence solutions for healthcare.
        We develop machine learning algorithms that help doctors diagnose diseases.
        Our AI-powered diagnostic tools have improved patient outcomes by 40%.
        We work with hospitals, clinics, and medical research institutions.
        Our technology uses deep learning and natural language processing.
        """
        
        print(f"Texto: {business_text}")
        
        # Extracción básica
        keywords = await self.nlp.extract_keywords(business_text, top_k=10)
        print(f"\nPalabras clave básicas: {keywords}")
        
        # Extracción avanzada
        advanced_result = await self.advanced_nlp.analyze_text_advanced(business_text)
        advanced_keywords = advanced_result.get('keywords', [])
        print(f"Palabras clave avanzadas: {advanced_keywords}")
    
    async def example_topic_modeling(self):
        """Ejemplo de modelado de temas."""
        print("\n=== Modelado de Temas ===")
        
        documents = [
            "Machine learning is revolutionizing healthcare with AI-powered diagnostics.",
            "Deep learning algorithms can detect cancer in medical images with high accuracy.",
            "Natural language processing helps doctors analyze patient records efficiently.",
            "Computer vision technology enables automated medical image analysis.",
            "Artificial intelligence is transforming the future of medicine.",
            "Neural networks are being used to predict patient outcomes.",
            "Healthcare AI systems are improving patient care and reducing costs.",
            "Medical AI applications include drug discovery and personalized treatment."
        ]
        
        print("Documentos:")
        for i, doc in enumerate(documents):
            print(f"{i+1}. {doc}")
        
        # Modelado básico
        topics = await self.nlp.extract_topics(documents, n_topics=3)
        print(f"\nTemas básicos: {topics}")
        
        # Modelado avanzado
        advanced_result = await self.advanced_nlp.analyze_text_advanced(documents[0])
        advanced_topics = advanced_result.get('topics', [])
        print(f"Temas avanzados: {advanced_topics}")
    
    async def example_readability_analysis(self):
        """Ejemplo de análisis de legibilidad."""
        print("\n=== Análisis de Legibilidad ===")
        
        texts = [
            "The cat sat on the mat.",
            "The implementation of sophisticated machine learning algorithms requires comprehensive understanding of statistical methodologies and computational complexity theory.",
            "Our company leverages cutting-edge artificial intelligence technologies to deliver innovative solutions that drive digital transformation and enhance operational efficiency across diverse industry verticals."
        ]
        
        for text in texts:
            print(f"\nTexto: {text}")
            
            # Legibilidad básica
            readability = await self.nlp.calculate_readability(text)
            print(f"Puntuación de legibilidad básica: {readability}")
            
            # Legibilidad avanzada
            advanced_result = await self.advanced_nlp.analyze_text_advanced(text)
            advanced_readability = advanced_result.get('readability', {})
            print(f"Análisis de legibilidad avanzado: {advanced_readability}")
    
    async def example_language_detection(self):
        """Ejemplo de detección de idioma."""
        print("\n=== Detección de Idioma ===")
        
        texts = [
            "Hello, how are you today?",
            "Hola, ¿cómo estás hoy?",
            "Bonjour, comment allez-vous aujourd'hui?",
            "Hallo, wie geht es dir heute?",
            "Ciao, come stai oggi?",
            "你好，你今天怎么样？",
            "こんにちは、今日はどうですか？",
            "안녕하세요, 오늘 어떠세요?",
            "Привет, как дела сегодня?",
            "مرحبا، كيف حالك اليوم؟"
        ]
        
        for text in texts:
            print(f"\nTexto: {text}")
            
            # Detección básica
            language = await self.nlp.detect_language(text)
            print(f"Idioma detectado básico: {language}")
            
            # Detección avanzada
            advanced_result = await self.advanced_nlp.analyze_text_advanced(text)
            advanced_language = advanced_result.get('language', 'unknown')
            print(f"Idioma detectado avanzado: {advanced_language}")
    
    async def example_business_document_analysis(self):
        """Ejemplo de análisis de documentos de negocio."""
        print("\n=== Análisis de Documentos de Negocio ===")
        
        business_document = """
        Executive Summary
        
        Our company, TechCorp Solutions, has experienced significant growth in the past year.
        Revenue increased by 25% to $50 million, driven by strong demand for our AI solutions.
        We expanded our team from 50 to 75 employees and opened new offices in San Francisco and London.
        
        Market Analysis
        
        The artificial intelligence market is projected to reach $1.8 trillion by 2030.
        Our main competitors include Google, Microsoft, and Amazon Web Services.
        We have a competitive advantage in healthcare AI applications.
        
        Financial Performance
        
        Q1 2024 Results:
        - Revenue: $12.5 million (up 30% YoY)
        - Net Profit: $2.1 million (up 45% YoY)
        - Customer Acquisition Cost: $500 (down 20% YoY)
        
        Future Outlook
        
        We plan to launch three new AI products in 2024.
        Target revenue for 2024: $75 million
        We are seeking $10 million in Series B funding.
        """
        
        print(f"Documento: {business_document}")
        
        # Análisis completo
        result = await self.advanced_nlp.analyze_text_advanced(business_document)
        
        print(f"\n=== Resultados del Análisis ===")
        print(f"Idioma: {result.get('language', 'unknown')}")
        print(f"Estadísticas: {result.get('statistics', {})}")
        print(f"Sentimiento: {result.get('sentiment', {}).get('ensemble', {})}")
        print(f"Entidades: {len(result.get('entities', []))} encontradas")
        print(f"Palabras clave: {result.get('keywords', [])[:10]}")
        print(f"Legibilidad: {result.get('readability', {}).get('overall_level', 'unknown')}")
        print(f"Tiempo de procesamiento: {result.get('processing_time', 0):.2f} segundos")
    
    async def example_content_optimization(self):
        """Ejemplo de optimización de contenido."""
        print("\n=== Optimización de Contenido ===")
        
        original_content = """
        Our company is very good at making software that is very useful for businesses.
        We have been in business for a very long time and we know a lot about technology.
        Our products are very popular and many customers like them very much.
        We think you should buy our software because it is very good and very reliable.
        """
        
        print(f"Contenido original: {original_content}")
        
        # Análisis del contenido original
        original_analysis = await self.advanced_nlp.analyze_text_advanced(original_content)
        
        # Contenido optimizado (simulado)
        optimized_content = """
        Our company excels at developing innovative software solutions for businesses.
        With decades of experience in technology, we deliver products that drive results.
        Our software is trusted by thousands of satisfied customers worldwide.
        Discover how our reliable, high-performance solutions can transform your business.
        """
        
        print(f"\nContenido optimizado: {optimized_content}")
        
        # Análisis del contenido optimizado
        optimized_analysis = await self.advanced_nlp.analyze_text_advanced(optimized_content)
        
        # Comparación
        print(f"\n=== Comparación ===")
        print(f"Legibilidad original: {original_analysis.get('readability', {}).get('average_score', 0):.1f}")
        print(f"Legibilidad optimizada: {optimized_analysis.get('readability', {}).get('average_score', 0):.1f}")
        print(f"Palabras clave originales: {len(original_analysis.get('keywords', []))}")
        print(f"Palabras clave optimizadas: {len(optimized_analysis.get('keywords', []))}")
    
    async def example_multilingual_processing(self):
        """Ejemplo de procesamiento multilingüe."""
        print("\n=== Procesamiento Multilingüe ===")
        
        multilingual_texts = {
            "English": "This is an excellent product with great features and amazing performance.",
            "Spanish": "Este es un producto excelente con características geniales y rendimiento increíble.",
            "French": "C'est un excellent produit avec de grandes fonctionnalités et des performances incroyables.",
            "German": "Dies ist ein ausgezeichnetes Produkt mit großartigen Funktionen und erstaunlicher Leistung."
        }
        
        for language, text in multilingual_texts.items():
            print(f"\n{language}: {text}")
            
            # Análisis por idioma
            result = await self.advanced_nlp.analyze_text_advanced(text)
            
            print(f"Idioma detectado: {result.get('language', 'unknown')}")
            print(f"Sentimiento: {result.get('sentiment', {}).get('ensemble', {}).get('sentiment', 'unknown')}")
            print(f"Palabras clave: {result.get('keywords', [])[:5]}")
    
    async def run_all_examples(self):
        """Ejecutar todos los ejemplos."""
        print("🚀 Iniciando ejemplos del sistema NLP...")
        
        try:
            # Inicializar sistemas
            await self.nlp.initialize()
            await self.advanced_nlp.initialize()
            
            # Ejecutar ejemplos
            await self.example_sentiment_analysis()
            await self.example_entity_extraction()
            await self.example_keyword_extraction()
            await self.example_topic_modeling()
            await self.example_readability_analysis()
            await self.example_language_detection()
            await self.example_business_document_analysis()
            await self.example_content_optimization()
            await self.example_multilingual_processing()
            
            print("\n✅ Todos los ejemplos completados exitosamente!")
            
        except Exception as e:
            print(f"\n❌ Error ejecutando ejemplos: {e}")
    
    async def benchmark_performance(self):
        """Benchmark de rendimiento del sistema."""
        print("\n=== Benchmark de Rendimiento ===")
        
        test_texts = [
            "Short text for testing.",
            "This is a medium length text that contains several sentences and should provide a good test case for the NLP system performance analysis.",
            "This is a much longer text that contains many sentences and should provide a comprehensive test case for the NLP system performance analysis. It includes various types of content such as technical terms, business language, and general vocabulary to test the system's ability to handle different types of text content effectively."
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\nTexto {i+1} (longitud: {len(text)} caracteres)")
            
            # Benchmark análisis básico
            start_time = datetime.now()
            basic_result = await self.nlp.analyze_text(text)
            basic_time = (datetime.now() - start_time).total_seconds()
            
            # Benchmark análisis avanzado
            start_time = datetime.now()
            advanced_result = await self.advanced_nlp.analyze_text_advanced(text)
            advanced_time = (datetime.now() - start_time).total_seconds()
            
            print(f"Análisis básico: {basic_time:.3f} segundos")
            print(f"Análisis avanzado: {advanced_time:.3f} segundos")
            print(f"Mejora de tiempo: {((advanced_time - basic_time) / basic_time * 100):.1f}%")

# Función principal para ejecutar ejemplos
async def main():
    """Función principal para ejecutar ejemplos."""
    examples = NLPExamples()
    await examples.run_all_examples()
    await examples.benchmark_performance()

if __name__ == "__main__":
    asyncio.run(main())












