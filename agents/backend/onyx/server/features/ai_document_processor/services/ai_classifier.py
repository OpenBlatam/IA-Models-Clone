"""
Clasificador AI para documentos
==============================

Sistema de clasificación inteligente que identifica el área y tipo de documento
usando IA y análisis de patrones.
"""

import os
import logging
import re
from typing import Dict, List, Optional, Tuple
import asyncio
from dataclasses import dataclass

# Importaciones de IA
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from models.document_models import (
    DocumentAnalysis, DocumentType, DocumentArea, DocumentCategory,
    ClassificationResult
)

logger = logging.getLogger(__name__)

@dataclass
class ClassificationPattern:
    """Patrón de clasificación basado en palabras clave"""
    keywords: List[str]
    area: DocumentArea
    category: DocumentCategory
    weight: float = 1.0

class AIClassifier:
    """Clasificador AI para documentos"""
    
    def __init__(self):
        self.openai_client = None
        self.patterns = self._load_classification_patterns()
        self.ml_model = None
        self.vectorizer = None
        
    async def initialize(self):
        """Inicializa el clasificador"""
        logger.info("Inicializando clasificador AI...")
        
        # Configurar OpenAI si está disponible
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = openai
            logger.info("✅ OpenAI configurado para clasificación")
        
        # Cargar modelo ML si está disponible
        if SKLEARN_AVAILABLE:
            await self._load_ml_model()
        
        logger.info("Clasificador AI inicializado")
    
    def _load_classification_patterns(self) -> List[ClassificationPattern]:
        """Carga patrones de clasificación basados en palabras clave"""
        return [
            # Business
            ClassificationPattern(
                keywords=["negocio", "empresa", "comercial", "ventas", "marketing", "estrategia", 
                         "plan de negocio", "consultoría", "análisis financiero", "presupuesto",
                         "business", "company", "sales", "strategy", "consulting"],
                area=DocumentArea.BUSINESS,
                category=DocumentCategory.BUSINESS_PLAN,
                weight=1.0
            ),
            ClassificationPattern(
                keywords=["consultoría", "asesoría", "recomendaciones", "diagnóstico", 
                         "consulting", "advisory", "recommendations", "diagnosis"],
                area=DocumentArea.BUSINESS,
                category=DocumentCategory.CONSULTANCY_REPORT,
                weight=1.2
            ),
            
            # Technology
            ClassificationPattern(
                keywords=["software", "programación", "código", "sistema", "aplicación",
                         "desarrollo", "arquitectura", "API", "base de datos", "tecnología",
                         "programming", "code", "system", "application", "development"],
                area=DocumentArea.TECHNOLOGY,
                category=DocumentCategory.TECHNICAL_DOCUMENTATION,
                weight=1.0
            ),
            ClassificationPattern(
                keywords=["especificación", "requisitos", "funcionalidades", "casos de uso",
                         "specification", "requirements", "features", "use cases"],
                area=DocumentArea.TECHNOLOGY,
                category=DocumentCategory.SOFTWARE_SPECIFICATION,
                weight=1.1
            ),
            
            # Academic
            ClassificationPattern(
                keywords=["investigación", "estudio", "análisis", "metodología", "hipótesis",
                         "conclusión", "bibliografía", "referencias", "académico", "universidad",
                         "research", "study", "analysis", "methodology", "hypothesis"],
                area=DocumentArea.ACADEMIC,
                category=DocumentCategory.RESEARCH_PAPER,
                weight=1.0
            ),
            ClassificationPattern(
                keywords=["tesis", "disertación", "doctorado", "maestría", "grado",
                         "thesis", "dissertation", "doctoral", "master", "degree"],
                area=DocumentArea.ACADEMIC,
                category=DocumentCategory.THESIS,
                weight=1.3
            ),
            
            # Legal
            ClassificationPattern(
                keywords=["contrato", "acuerdo", "términos", "condiciones", "legal",
                         "jurídico", "ley", "normativa", "compliance", "regulación",
                         "contract", "agreement", "terms", "legal", "law", "regulation"],
                area=DocumentArea.LEGAL,
                category=DocumentCategory.CONTRACT,
                weight=1.0
            ),
            ClassificationPattern(
                keywords=["política", "procedimiento", "protocolo", "norma", "estándar",
                         "policy", "procedure", "protocol", "standard"],
                area=DocumentArea.LEGAL,
                category=DocumentCategory.POLICY_DOCUMENT,
                weight=1.1
            ),
            
            # Medical
            ClassificationPattern(
                keywords=["médico", "salud", "paciente", "diagnóstico", "tratamiento",
                         "enfermedad", "síntomas", "medicina", "clínico", "hospital",
                         "medical", "health", "patient", "diagnosis", "treatment"],
                area=DocumentArea.MEDICAL,
                category=DocumentCategory.MANUAL,
                weight=1.0
            ),
            
            # Finance
            ClassificationPattern(
                keywords=["financiero", "inversión", "presupuesto", "contabilidad",
                         "auditoría", "riesgo", "capital", "activos", "pasivos",
                         "financial", "investment", "budget", "accounting", "audit"],
                area=DocumentArea.FINANCE,
                category=DocumentCategory.FINANCIAL_ANALYSIS,
                weight=1.0
            ),
            
            # Marketing
            ClassificationPattern(
                keywords=["marketing", "publicidad", "promoción", "campaña", "branding",
                         "mercado", "consumidor", "producto", "precio", "distribución",
                         "advertising", "promotion", "campaign", "market", "consumer"],
                area=DocumentArea.MARKETING,
                category=DocumentCategory.MARKETING_STRATEGY,
                weight=1.0
            ),
            
            # Education
            ClassificationPattern(
                keywords=["educación", "enseñanza", "aprendizaje", "curso", "lección",
                         "estudiante", "profesor", "pedagogía", "currículo", "evaluación",
                         "education", "teaching", "learning", "course", "student"],
                area=DocumentArea.EDUCATION,
                category=DocumentCategory.MANUAL,
                weight=1.0
            )
        ]
    
    async def _load_ml_model(self):
        """Carga o entrena un modelo de machine learning"""
        try:
            model_path = "models/classification_model.joblib"
            if os.path.exists(model_path):
                self.ml_model = joblib.load(model_path)
                logger.info("Modelo ML cargado desde archivo")
            else:
                # Crear modelo básico
                self.ml_model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                    ('classifier', MultinomialNB())
                ])
                logger.info("Modelo ML básico creado")
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo ML: {e}")
            self.ml_model = None
    
    async def classify_document(self, text: str) -> DocumentAnalysis:
        """Clasifica un documento usando múltiples métodos"""
        try:
            # Limpiar y preprocesar texto
            cleaned_text = self._preprocess_text(text)
            
            # Obtener clasificaciones de diferentes métodos
            pattern_result = self._classify_by_patterns(cleaned_text)
            ai_result = await self._classify_with_ai(cleaned_text)
            ml_result = self._classify_with_ml(cleaned_text)
            
            # Combinar resultados
            final_classification = self._combine_classifications(
                pattern_result, ai_result, ml_result
            )
            
            # Crear análisis completo
            analysis = DocumentAnalysis(
                filename="",  # Se llenará en el servicio principal
                document_type=self._detect_document_type(text),
                area=final_classification.area,
                category=final_classification.category,
                confidence=final_classification.confidence,
                language=self._detect_language(text),
                word_count=len(text.split()),
                key_topics=self._extract_key_topics(cleaned_text),
                summary=self._generate_summary(cleaned_text),
                metadata={
                    'classification_methods': {
                        'pattern': pattern_result.confidence if pattern_result else 0,
                        'ai': ai_result.confidence if ai_result else 0,
                        'ml': ml_result.confidence if ml_result else 0
                    }
                }
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            # Retornar clasificación por defecto
            return DocumentAnalysis(
                filename="",
                document_type=DocumentType.UNKNOWN,
                area=DocumentArea.GENERAL,
                category=DocumentCategory.MANUAL,
                confidence=0.1,
                word_count=len(text.split()),
                summary="Error en clasificación"
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocesa el texto para análisis"""
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover caracteres especiales pero mantener espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remover espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _classify_by_patterns(self, text: str) -> Optional[ClassificationResult]:
        """Clasifica usando patrones de palabras clave"""
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns:
            score = 0.0
            matches = 0
            
            for keyword in pattern.keywords:
                if keyword in text:
                    matches += 1
                    score += pattern.weight
            
            # Normalizar score
            if len(pattern.keywords) > 0:
                normalized_score = (score / len(pattern.keywords)) * (matches / len(pattern.keywords))
                
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = ClassificationResult(
                        document_type=DocumentType.TEXT,  # Se determinará después
                        area=pattern.area,
                        category=pattern.category,
                        confidence=min(normalized_score, 1.0),
                        reasoning=f"Patrón de palabras clave: {matches}/{len(pattern.keywords)} coincidencias"
                    )
        
        return best_match
    
    async def _classify_with_ai(self, text: str) -> Optional[ClassificationResult]:
        """Clasifica usando IA (OpenAI)"""
        if not self.openai_client:
            return None
        
        try:
            # Truncar texto si es muy largo
            max_length = 2000
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            prompt = f"""
            Analiza el siguiente texto y clasifícalo en una de estas categorías:
            
            ÁREAS:
            - business: Negocios, empresas, consultoría, marketing
            - technology: Tecnología, software, sistemas, programación
            - academic: Académico, investigación, educación, tesis
            - legal: Legal, contratos, políticas, regulaciones
            - medical: Médico, salud, clínico
            - finance: Financiero, inversiones, contabilidad
            - marketing: Marketing, publicidad, ventas
            - education: Educación, enseñanza, aprendizaje
            - general: General, no específico
            
            CATEGORÍAS:
            - business_plan: Plan de negocio
            - consultancy_report: Reporte de consultoría
            - technical_documentation: Documentación técnica
            - research_paper: Artículo de investigación
            - contract: Contrato
            - manual: Manual o guía
            - proposal: Propuesta
            - presentation: Presentación
            
            TEXTO: {text}
            
            Responde en formato JSON:
            {{
                "area": "área_detectada",
                "category": "categoría_detectada",
                "confidence": 0.0-1.0,
                "reasoning": "explicación_breve"
            }}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parsear respuesta JSON
            import json
            result = json.loads(result_text)
            
            return ClassificationResult(
                document_type=DocumentType.TEXT,
                area=DocumentArea(result.get("area", "general")),
                category=DocumentCategory(result.get("category", "manual")),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", "Clasificación por IA")
            )
            
        except Exception as e:
            logger.warning(f"Error en clasificación AI: {e}")
            return None
    
    def _classify_with_ml(self, text: str) -> Optional[ClassificationResult]:
        """Clasifica usando machine learning"""
        if not self.ml_model:
            return None
        
        try:
            # Para un modelo básico, usar clasificación por patrones como fallback
            # En una implementación completa, aquí se entrenaría un modelo real
            return None
            
        except Exception as e:
            logger.warning(f"Error en clasificación ML: {e}")
            return None
    
    def _combine_classifications(self, *results) -> ClassificationResult:
        """Combina múltiples clasificaciones en un resultado final"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return ClassificationResult(
                document_type=DocumentType.TEXT,
                area=DocumentArea.GENERAL,
                category=DocumentCategory.MANUAL,
                confidence=0.1,
                reasoning="Sin clasificaciones válidas"
            )
        
        # Si solo hay un resultado, usarlo
        if len(valid_results) == 1:
            return valid_results[0]
        
        # Combinar resultados por área más común
        area_votes = {}
        category_votes = {}
        total_confidence = 0
        
        for result in valid_results:
            area = result.area.value
            category = result.category.value
            
            area_votes[area] = area_votes.get(area, 0) + result.confidence
            category_votes[category] = category_votes.get(category, 0) + result.confidence
            total_confidence += result.confidence
        
        # Seleccionar área y categoría con mayor puntuación
        best_area = max(area_votes, key=area_votes.get)
        best_category = max(category_votes, key=category_votes.get)
        
        # Calcular confianza promedio
        avg_confidence = total_confidence / len(valid_results)
        
        return ClassificationResult(
            document_type=DocumentType.TEXT,
            area=DocumentArea(best_area),
            category=DocumentCategory(best_category),
            confidence=min(avg_confidence, 1.0),
            reasoning=f"Combinación de {len(valid_results)} métodos de clasificación"
        )
    
    def _detect_document_type(self, text: str) -> DocumentType:
        """Detecta el tipo de documento basado en el contenido"""
        text_lower = text.lower()
        
        if any(marker in text_lower for marker in ['#', '##', '###', '**', '*']):
            return DocumentType.MARKDOWN
        elif any(marker in text_lower for marker in ['<html', '<body', '<div']):
            return DocumentType.TEXT  # HTML
        else:
            return DocumentType.TEXT
    
    def _detect_language(self, text: str) -> str:
        """Detecta el idioma del texto"""
        # Análisis simple basado en palabras comunes
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'más', 'muy', 'ya', 'todo', 'esta', 'ser', 'tiene', 'también', 'fue', 'había', 'me', 'si', 'sin', 'sobre', 'este', 'entre', 'cuando', 'muy', 'sin', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'sólo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'otros', 'aunque', 'esa', 'esos', 'estas', 'le', 'ha', 'me', 'sus', 'ya', 'están', 'como', 'está', 'sí', 'pero', 'sus', 'más', 'muy', 'ya', 'todo', 'esta', 'ser', 'tiene', 'también', 'fue', 'había', 'me', 'si', 'sin', 'sobre', 'este', 'entre', 'cuando', 'muy', 'sin', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'sólo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'otros', 'aunque', 'esa', 'esos', 'estas', 'le', 'ha', 'me', 'sus', 'ya', 'están']
        
        english_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us']
        
        words = text.lower().split()
        spanish_count = sum(1 for word in words if word in spanish_words)
        english_count = sum(1 for word in words if word in english_words)
        
        if spanish_count > english_count:
            return 'es'
        elif english_count > spanish_count:
            return 'en'
        else:
            return 'es'  # Por defecto español
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extrae temas clave del texto"""
        # Implementación simple: palabras más frecuentes
        words = text.split()
        word_freq = {}
        
        # Filtrar palabras comunes
        stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'más', 'muy', 'ya', 'todo', 'esta', 'ser', 'tiene', 'también', 'fue', 'había', 'me', 'si', 'sin', 'sobre', 'este', 'entre', 'cuando', 'muy', 'sin', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'sólo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'otros', 'aunque', 'esa', 'esos', 'estas', 'le', 'ha', 'me', 'sus', 'ya', 'están', 'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'}
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Retornar las 10 palabras más frecuentes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _generate_summary(self, text: str) -> str:
        """Genera un resumen del texto"""
        sentences = text.split('.')
        if len(sentences) <= 3:
            return text
        
        # Tomar las primeras 2-3 oraciones como resumen
        summary_sentences = sentences[:3]
        return '. '.join(summary_sentences) + '.'


