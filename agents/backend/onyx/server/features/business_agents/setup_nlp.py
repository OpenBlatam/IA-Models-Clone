"""
NLP System Setup
================

Script de configuración e instalación del sistema NLP avanzado.
Incluye descarga de modelos, configuración de dependencias y verificación del sistema.
"""

import os
import sys
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPSetup:
    """Configurador del sistema NLP."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.cache_dir = self.base_dir / "cache"
        self.config_file = self.base_dir / "nlp_config.json"
        
        # Crear directorios necesarios
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
    
    async def install_dependencies(self):
        """Instalar dependencias del sistema NLP."""
        print("🔧 Instalando dependencias del sistema NLP...")
        
        try:
            # Instalar requirements.txt
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(self.base_dir / "requirements.txt")
            ], check=True)
            
            print("✅ Dependencias instaladas correctamente")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando dependencias: {e}")
            return False
        
        return True
    
    async def download_spacy_models(self):
        """Descargar modelos de spaCy."""
        print("📥 Descargando modelos de spaCy...")
        
        spacy_models = [
            "en_core_web_sm",
            "es_core_news_sm", 
            "fr_core_news_sm",
            "de_core_news_sm",
            "it_core_news_sm",
            "pt_core_news_sm",
            "zh_core_web_sm",
            "ja_core_news_sm",
            "ko_core_news_sm",
            "ru_core_news_sm",
            "ar_core_news_sm"
        ]
        
        for model in spacy_models:
            try:
                print(f"  Descargando {model}...")
                subprocess.run([
                    sys.executable, "-m", "spacy", "download", model
                ], check=True, capture_output=True)
                print(f"  ✅ {model} descargado")
                
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  No se pudo descargar {model}: {e}")
                continue
    
    async def download_nltk_data(self):
        """Descargar datos de NLTK."""
        print("📥 Descargando datos de NLTK...")
        
        nltk_data = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'vader_lexicon', 'wordnet', 'omw-1.4', 'maxent_ne_chunker',
            'words', 'brown', 'reuters', 'gutenberg', 'punkt_tab',
            'averaged_perceptron_tagger', 'universal_tagset'
        ]
        
        try:
            import nltk
            for data in nltk_data:
                try:
                    nltk.download(data, quiet=True)
                    print(f"  ✅ {data} descargado")
                except Exception as e:
                    print(f"  ⚠️  Error descargando {data}: {e}")
                    
        except ImportError:
            print("  ⚠️  NLTK no está instalado")
    
    async def download_huggingface_models(self):
        """Descargar modelos de Hugging Face."""
        print("📥 Descargando modelos de Hugging Face...")
        
        models = {
            "sentiment": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "nlptown/bert-base-multilingual-uncased-sentiment",
                "distilbert-base-uncased-finetuned-sst-2-english"
            ],
            "ner": [
                "dbmdz/bert-large-cased-finetuned-conll03-english",
                "xlm-roberta-large-finetuned-conll03-english"
            ],
            "classification": [
                "microsoft/DialoGPT-medium",
                "distilbert-base-uncased"
            ],
            "summarization": [
                "facebook/bart-large-cnn",
                "google/pegasus-xsum"
            ],
            "translation": [
                "Helsinki-NLP/opus-mt-en-es",
                "Helsinki-NLP/opus-mt-en-fr",
                "Helsinki-NLP/opus-mt-en-de"
            ],
            "embeddings": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2"
            ]
        }
        
        for category, model_list in models.items():
            print(f"  Descargando modelos de {category}...")
            for model in model_list:
                try:
                    # Pre-cargar modelo para descargarlo
                    from transformers import AutoTokenizer, AutoModel
                    tokenizer = AutoTokenizer.from_pretrained(model)
                    model_obj = AutoModel.from_pretrained(model)
                    print(f"    ✅ {model}")
                except Exception as e:
                    print(f"    ⚠️  Error con {model}: {e}")
    
    async def download_flair_models(self):
        """Descargar modelos de Flair."""
        print("📥 Descargando modelos de Flair...")
        
        try:
            import flair
            from flair.models import SequenceTagger
            
            # Descargar modelo NER
            tagger = SequenceTagger.load('ner')
            print("  ✅ Flair NER model")
            
        except Exception as e:
            print(f"  ⚠️  Error descargando modelos de Flair: {e}")
    
    async def download_stanza_models(self):
        """Descargar modelos de Stanza."""
        print("📥 Descargando modelos de Stanza...")
        
        try:
            import stanza
            # Descargar pipeline en inglés
            stanza.download('en')
            print("  ✅ Stanza English model")
            
        except Exception as e:
            print(f"  ⚠️  Error descargando modelos de Stanza: {e}")
    
    async def create_config_file(self):
        """Crear archivo de configuración."""
        print("⚙️  Creando archivo de configuración...")
        
        config = {
            "nlp_system": {
                "default_language": "en",
                "model_provider": "huggingface",
                "use_gpu": True,
                "cache_models": True,
                "max_text_length": 512,
                "batch_size": 32
            },
            "models": {
                "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "ner": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "classification": "microsoft/DialoGPT-medium",
                "summarization": "facebook/bart-large-cnn",
                "translation": "Helsinki-NLP/opus-mt-en-es",
                "embeddings": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "performance": {
                "use_gpu": True,
                "batch_size": 32,
                "max_length": 512,
                "cache_models": True
            },
            "languages": [
                "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", "ar"
            ],
            "directories": {
                "models": str(self.models_dir),
                "cache": str(self.cache_dir)
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configuración guardada en {self.config_file}")
    
    async def test_system(self):
        """Probar el sistema NLP."""
        print("🧪 Probando el sistema NLP...")
        
        try:
            # Importar sistemas
            from .nlp_system import nlp_system
            from .advanced_nlp_system import advanced_nlp_system
            
            # Inicializar sistemas
            await nlp_system.initialize()
            await advanced_nlp_system.initialize()
            
            # Probar análisis básico
            test_text = "This is a test of the NLP system. It should work correctly."
            result = await nlp_system.analyze_text(test_text)
            
            print("  ✅ Sistema básico funcionando")
            
            # Probar análisis avanzado
            advanced_result = await advanced_nlp_system.analyze_text_advanced(test_text)
            
            print("  ✅ Sistema avanzado funcionando")
            
            # Verificar salud del sistema
            health = await advanced_nlp_system.get_system_health()
            print(f"  ✅ Salud del sistema: {health.get('initialized', False)}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error probando el sistema: {e}")
            return False
    
    async def create_environment_file(self):
        """Crear archivo .env para configuración."""
        print("📝 Creando archivo de entorno...")
        
        env_content = """
# NLP System Configuration
NLP_DEFAULT_LANGUAGE=en
NLP_MODEL_PROVIDER=huggingface
NLP_USE_GPU=true
NLP_CACHE_MODELS=true
NLP_MAX_TEXT_LENGTH=512
NLP_BATCH_SIZE=32

# Model Paths
NLP_MODELS_DIR=models/
NLP_CACHE_DIR=cache/

# API Keys (opcional)
HUGGINGFACE_TOKEN=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# Performance Settings
NLP_MAX_CONCURRENT_REQUESTS=10
NLP_API_TIMEOUT=30
NLP_RATE_LIMIT=100

# Logging
NLP_LOG_LEVEL=INFO
NLP_ENABLE_METRICS=true
"""
        
        env_file = self.base_dir / ".env"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content.strip())
        
        print(f"✅ Archivo de entorno creado en {env_file}")
    
    async def run_setup(self):
        """Ejecutar configuración completa."""
        print("🚀 Iniciando configuración del sistema NLP...")
        
        steps = [
            ("Instalando dependencias", self.install_dependencies),
            ("Descargando modelos spaCy", self.download_spacy_models),
            ("Descargando datos NLTK", self.download_nltk_data),
            ("Descargando modelos Hugging Face", self.download_huggingface_models),
            ("Descargando modelos Flair", self.download_flair_models),
            ("Descargando modelos Stanza", self.download_stanza_models),
            ("Creando configuración", self.create_config_file),
            ("Creando archivo de entorno", self.create_environment_file),
            ("Probando sistema", self.test_system)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            try:
                success = await step_func()
                if success is False:
                    print(f"⚠️  {step_name} falló, pero continuando...")
            except Exception as e:
                print(f"❌ Error en {step_name}: {e}")
                continue
        
        print("\n🎉 Configuración del sistema NLP completada!")
        print("\n📚 Próximos pasos:")
        print("1. Revisar el archivo .env y configurar las API keys si es necesario")
        print("2. Ejecutar: python nlp_examples.py para probar el sistema")
        print("3. Iniciar el servidor: python main.py")
        print("4. Acceder a la documentación API en: http://localhost:8000/docs")

async def main():
    """Función principal."""
    setup = NLPSetup()
    await setup.run_setup()

if __name__ == "__main__":
    asyncio.run(main())












