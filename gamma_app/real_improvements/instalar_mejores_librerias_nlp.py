#!/usr/bin/env python3
"""
🚀 INSTALADOR AUTOMÁTICO DE MEJORES LIBRERÍAS NLP
Script para instalar las mejores librerías de procesamiento de lenguaje natural
"""

import subprocess
import sys
import os
from typing import List, Dict, Any
import time

def ejecutar_comando(comando: str, descripcion: str = "") -> bool:
    """Ejecutar comando y mostrar resultado"""
    try:
        print(f"🔄 {descripcion}...")
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
        
        if resultado.returncode == 0:
            print(f"✅ {descripcion} completado")
            return True
        else:
            print(f"❌ Error en {descripcion}: {resultado.stderr}")
            return False
    except Exception as e:
        print(f"❌ Excepción en {descripcion}: {str(e)}")
        return False

def instalar_librerias_core():
    """Instalar librerías core de NLP"""
    print("\n🧠 INSTALANDO LIBRERÍAS CORE NLP")
    print("=" * 50)
    
    librerias_core = [
        "spacy>=3.7.0",
        "nltk>=3.8.1", 
        "textblob>=0.17.1"
    ]
    
    for libreria in librerias_core:
        comando = f"pip install {libreria}"
        ejecutar_comando(comando, f"Instalando {libreria}")

def instalar_librerias_deep_learning():
    """Instalar librerías de deep learning"""
    print("\n🔬 INSTALANDO LIBRERÍAS DEEP LEARNING")
    print("=" * 50)
    
    librerias_dl = [
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "tensorflow>=2.13.0",
        "flair>=0.12.0"
    ]
    
    for libreria in librerias_dl:
        comando = f"pip install {libreria}"
        ejecutar_comando(comando, f"Instalando {libreria}")

def instalar_librerias_ml():
    """Instalar librerías de machine learning"""
    print("\n📊 INSTALANDO LIBRERÍAS MACHINE LEARNING")
    print("=" * 50)
    
    librerias_ml = [
        "scikit-learn>=1.3.0",
        "gensim>=4.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0"
    ]
    
    for libreria in librerias_ml:
        comando = f"pip install {libreria}"
        ejecutar_comando(comando, f"Instalando {libreria}")

def instalar_librerias_visualizacion():
    """Instalar librerías de visualización"""
    print("\n📈 INSTALANDO LIBRERÍAS VISUALIZACIÓN")
    print("=" * 50)
    
    librerias_viz = [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "wordcloud>=1.9.2",
        "plotly>=5.15.0"
    ]
    
    for libreria in librerias_viz:
        comando = f"pip install {libreria}"
        ejecutar_comando(comando, f"Instalando {libreria}")

def instalar_librerias_utilidades():
    """Instalar librerías de utilidades"""
    print("\n🔧 INSTALANDO LIBRERÍAS UTILIDADES")
    print("=" * 50)
    
    librerias_util = [
        "regex>=2023.8.8",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "googletrans>=4.0.0",
        "translate>=3.6.1"
    ]
    
    for libreria in librerias_util:
        comando = f"pip install {libreria}"
        ejecutar_comando(comando, f"Instalando {libreria}")

def instalar_librerias_rendimiento():
    """Instalar librerías de rendimiento"""
    print("\n⚡ INSTALANDO LIBRERÍAS RENDIMIENTO")
    print("=" * 50)
    
    librerias_perf = [
        "uvloop>=0.17.0",
        "httptools>=0.6.0",
        "cython>=0.29.0",
        "numba>=0.57.0"
    ]
    
    for libreria in librerias_perf:
        comando = f"pip install {libreria}"
        ejecutar_comando(comando, f"Instalando {libreria}")

def instalar_librerias_especializadas():
    """Instalar librerías especializadas"""
    print("\n🎯 INSTALANDO LIBRERÍAS ESPECIALIZADAS")
    print("=" * 50)
    
    librerias_esp = [
        "vaderSentiment>=3.3.2",
        "afinn>=0.1",
        "sentence-transformers>=2.2.2",
        "langdetect>=1.0.9",
        "textstat>=0.7.3"
    ]
    
    for libreria in librerias_esp:
        comando = f"pip install {libreria}"
        ejecutar_comando(comando, f"Instalando {libreria}")

def instalar_librerias_desarrollo():
    """Instalar librerías de desarrollo"""
    print("\n🛠️ INSTALANDO LIBRERÍAS DESARROLLO")
    print("=" * 50)
    
    librerias_dev = [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "loguru>=0.7.0",
        "structlog>=23.1.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0"
    ]
    
    for libreria in librerias_dev:
        comando = f"pip install {libreria}"
        ejecutar_comando(comando, f"Instalando {libreria}")

def descargar_modelos_spacy():
    """Descargar modelos de spaCy"""
    print("\n🧠 DESCARGANDO MODELOS SPACY")
    print("=" * 50)
    
    modelos = [
        "es_core_news_sm",
        "en_core_web_sm"
    ]
    
    for modelo in modelos:
        comando = f"python -m spacy download {modelo}"
        ejecutar_comando(comando, f"Descargando modelo {modelo}")

def descargar_recursos_nltk():
    """Descargar recursos de NLTK"""
    print("\n📚 DESCARGANDO RECURSOS NLTK")
    print("=" * 50)
    
    recursos = [
        "punkt",
        "stopwords", 
        "vader_lexicon",
        "averaged_perceptron_tagger",
        "wordnet",
        "omw-1.4"
    ]
    
    for recurso in recursos:
        comando = f"python -c \"import nltk; nltk.download('{recurso}')\""
        ejecutar_comando(comando, f"Descargando recurso {recurso}")

def verificar_instalacion():
    """Verificar que todas las librerías estén instaladas"""
    print("\n🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 50)
    
    librerias_verificar = [
        "spacy",
        "nltk", 
        "textblob",
        "transformers",
        "torch",
        "tensorflow",
        "flair",
        "sklearn",
        "gensim",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "wordcloud",
        "regex",
        "requests"
    ]
    
    exitosas = 0
    fallidas = 0
    
    for libreria in librerias_verificar:
        try:
            comando = f"python -c \"import {libreria}; print('✅ {libreria} instalada correctamente')\""
            resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
            
            if resultado.returncode == 0:
                print(f"✅ {libreria}")
                exitosas += 1
            else:
                print(f"❌ {libreria}")
                fallidas += 1
        except:
            print(f"❌ {libreria}")
            fallidas += 1
    
    print(f"\n📊 RESUMEN DE VERIFICACIÓN")
    print(f"   ✅ Librerías exitosas: {exitosas}")
    print(f"   ❌ Librerías fallidas: {fallidas}")
    print(f"   📈 Porcentaje de éxito: {(exitosas/(exitosas+fallidas)*100):.1f}%")
    
    return exitosas, fallidas

def crear_script_verificacion():
    """Crear script de verificación"""
    print("\n📝 CREANDO SCRIPT DE VERIFICACIÓN")
    print("=" * 50)
    
    script_content = '''#!/usr/bin/env python3
"""
🔍 SCRIPT DE VERIFICACIÓN DE LIBRERÍAS NLP
Verificar que todas las librerías estén instaladas correctamente
"""

def verificar_librerias():
    """Verificar instalación de librerías"""
    librerias = [
        "spacy", "nltk", "textblob", "transformers", "torch", "tensorflow",
        "flair", "sklearn", "gensim", "pandas", "numpy", "matplotlib",
        "seaborn", "wordcloud", "regex", "requests", "googletrans"
    ]
    
    print("🔍 VERIFICANDO LIBRERÍAS NLP")
    print("=" * 40)
    
    exitosas = 0
    for libreria in librerias:
        try:
            __import__(libreria)
            print(f"✅ {libreria}")
            exitosas += 1
        except ImportError:
            print(f"❌ {libreria}")
    
    print(f"\\n📊 RESULTADO: {exitosas}/{len(librerias)} librerías instaladas")
    return exitosas == len(librerias)

if __name__ == "__main__":
    verificar_librerias()
'''
    
    with open("verificar_nlp.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ Script de verificación creado: verificar_nlp.py")

def crear_ejemplo_uso():
    """Crear ejemplo de uso de las librerías"""
    print("\n📝 CREANDO EJEMPLO DE USO")
    print("=" * 50)
    
    ejemplo_content = '''#!/usr/bin/env python3
"""
🚀 EJEMPLO DE USO DE LIBRERÍAS NLP
Demostración de las mejores librerías NLP instaladas
"""

import spacy
import nltk
from textblob import TextBlob
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def ejemplo_spacy():
    """Ejemplo con spaCy"""
    print("🧠 EJEMPLO SPACY")
    print("-" * 20)
    
    try:
        nlp = spacy.load('es_core_news_sm')
        doc = nlp("Juan Pérez trabaja en Google en Madrid")
        
        print("Entidades encontradas:")
        for ent in doc.ents:
            print(f"  {ent.text} -> {ent.label_}")
    except Exception as e:
        print(f"Error: {e}")

def ejemplo_nltk():
    """Ejemplo con NLTK"""
    print("\\n📚 EJEMPLO NLTK")
    print("-" * 20)
    
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores("Este texto es muy positivo")
        print(f"Puntuaciones de sentimiento: {scores}")
    except Exception as e:
        print(f"Error: {e}")

def ejemplo_textblob():
    """Ejemplo con TextBlob"""
    print("\\n💭 EJEMPLO TEXTBLOB")
    print("-" * 20)
    
    try:
        blob = TextBlob("Este texto es muy positivo")
        print(f"Polaridad: {blob.sentiment.polarity}")
        print(f"Subjetividad: {blob.sentiment.subjectivity}")
    except Exception as e:
        print(f"Error: {e}")

def ejemplo_transformers():
    """Ejemplo con Transformers"""
    print("\\n🤖 EJEMPLO TRANSFORMERS")
    print("-" * 20)
    
    try:
        classifier = pipeline("sentiment-analysis")
        result = classifier("This text is very positive")
        print(f"Resultado: {result}")
    except Exception as e:
        print(f"Error: {e}")

def ejemplo_pandas():
    """Ejemplo con pandas"""
    print("\\n📊 EJEMPLO PANDAS")
    print("-" * 20)
    
    try:
        df = pd.DataFrame({
            'texto': ['Texto positivo', 'Texto negativo', 'Texto neutral'],
            'sentimiento': ['positivo', 'negativo', 'neutral']
        })
        print(df)
    except Exception as e:
        print(f"Error: {e}")

def ejemplo_wordcloud():
    """Ejemplo con WordCloud"""
    print("\\n☁️ EJEMPLO WORDCLOUD")
    print("-" * 20)
    
    try:
        text = "python programación datos análisis machine learning"
        wordcloud = WordCloud().generate(text)
        print("WordCloud generado correctamente")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Función principal"""
    print("🚀 DEMO DE LIBRERÍAS NLP")
    print("=" * 50)
    
    ejemplo_spacy()
    ejemplo_nltk()
    ejemplo_textblob()
    ejemplo_transformers()
    ejemplo_pandas()
    ejemplo_wordcloud()
    
    print("\\n🎉 ¡DEMO COMPLETADO!")

if __name__ == "__main__":
    main()
'''
    
    with open("ejemplo_nlp.py", "w", encoding="utf-8") as f:
        f.write(ejemplo_content)
    
    print("✅ Ejemplo de uso creado: ejemplo_nlp.py")

def main():
    """Función principal de instalación"""
    print("🚀 INSTALADOR AUTOMÁTICO DE MEJORES LIBRERÍAS NLP")
    print("=" * 80)
    print("Este script instalará las mejores librerías de procesamiento de lenguaje natural")
    print("Incluye: spaCy, NLTK, Transformers, PyTorch, TensorFlow, scikit-learn, y más")
    print("=" * 80)
    
    # Confirmar instalación
    respuesta = input("\n¿Deseas continuar con la instalación? (s/n): ")
    if respuesta.lower() != 's':
        print("❌ Instalación cancelada")
        return
    
    # Instalar librerías por categorías
    instalar_librerias_core()
    instalar_librerias_deep_learning()
    instalar_librerias_ml()
    instalar_librerias_visualizacion()
    instalar_librerias_utilidades()
    instalar_librerias_rendimiento()
    instalar_librerias_especializadas()
    instalar_librerias_desarrollo()
    
    # Descargar modelos y recursos
    descargar_modelos_spacy()
    descargar_recursos_nltk()
    
    # Verificar instalación
    exitosas, fallidas = verificar_instalacion()
    
    # Crear scripts auxiliares
    crear_script_verificacion()
    crear_ejemplo_uso()
    
    # Resumen final
    print("\n🎉 ¡INSTALACIÓN COMPLETADA!")
    print("=" * 50)
    print(f"✅ Librerías instaladas exitosamente: {exitosas}")
    print(f"❌ Librerías con errores: {fallidas}")
    print(f"📈 Porcentaje de éxito: {(exitosas/(exitosas+fallidas)*100):.1f}%")
    
    print("\n📝 ARCHIVOS CREADOS:")
    print("   • verificar_nlp.py - Script de verificación")
    print("   • ejemplo_nlp.py - Ejemplo de uso")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("   1. Ejecutar: python verificar_nlp.py")
    print("   2. Probar: python ejemplo_nlp.py")
    print("   3. Comenzar a usar las librerías NLP")
    
    print("\n💡 COMANDOS ÚTILES:")
    print("   • Verificar instalación: python verificar_nlp.py")
    print("   • Ejecutar demo: python ejemplo_nlp.py")
    print("   • Ver versión de Python: python --version")
    print("   • Ver pip instalado: pip --version")

if __name__ == "__main__":
    main()




