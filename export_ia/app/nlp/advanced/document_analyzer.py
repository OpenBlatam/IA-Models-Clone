"""
Document Analyzer - Sistema avanzado de análisis de documentos
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from ..models import Language, SentimentType
from .embeddings import EmbeddingManager
from .transformer_models import TransformerModelManager

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Tipos de documentos."""
    TEXT = "text"
    PDF = "pdf"
    WORD = "word"
    HTML = "html"
    MARKDOWN = "markdown"
    EMAIL = "email"
    REPORT = "report"
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS = "news"
    ACADEMIC_PAPER = "academic_paper"
    LEGAL_DOCUMENT = "legal_document"
    TECHNICAL_DOCUMENT = "technical_document"


class DocumentStructure(Enum):
    """Estructura de documentos."""
    LINEAR = "linear"
    HIERARCHICAL = "hierarchical"
    TABULAR = "tabular"
    NARRATIVE = "narrative"
    TECHNICAL = "technical"
    ACADEMIC = "academic"


@dataclass
class DocumentMetadata:
    """Metadatos de un documento."""
    document_id: str
    title: str
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    language: Language = Language.ENGLISH
    document_type: DocumentType = DocumentType.TEXT
    file_size: int = 0
    word_count: int = 0
    character_count: int = 0
    page_count: int = 0
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


@dataclass
class DocumentSection:
    """Sección de un documento."""
    section_id: str
    title: str
    content: str
    level: int = 1
    start_position: int = 0
    end_position: int = 0
    word_count: int = 0
    topics: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Optional[SentimentType] = None


@dataclass
class DocumentAnalysis:
    """Análisis completo de un documento."""
    document_id: str
    metadata: DocumentMetadata
    structure: DocumentStructure
    sections: List[DocumentSection]
    summary: str
    key_points: List[str]
    main_topics: List[str]
    entities: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, Any]
    readability_score: float
    quality_score: float
    recommendations: List[str]
    analyzed_at: datetime = field(default_factory=datetime.now)


class DocumentAnalyzer:
    """
    Sistema avanzado de análisis de documentos.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager, transformer_manager: TransformerModelManager):
        """Inicializar analizador de documentos."""
        self.embedding_manager = embedding_manager
        self.transformer_manager = transformer_manager
        
        # Almacenamiento de documentos
        self.documents: Dict[str, DocumentAnalysis] = {}
        self.document_cache = {}
        
        # Configuración
        self.max_document_size = 1000000  # 1MB
        self.cache_ttl = 3600  # 1 hora
        
        # Patrones de estructura
        self.structure_patterns = {
            DocumentStructure.HIERARCHICAL: [
                r'^#+\s+',  # Headers markdown
                r'^\d+\.\s+',  # Numbered lists
                r'^[A-Z][A-Z\s]+$'  # All caps headers
            ],
            DocumentStructure.TABULAR: [
                r'\|\s*.*\s*\|',  # Markdown tables
                r'\t.*\t',  # Tab-separated
                r',.*,.*,'  # CSV-like
            ],
            DocumentStructure.ACADEMIC: [
                r'Abstract:',  # Academic abstracts
                r'Introduction:',  # Academic sections
                r'References:',  # References
                r'Bibliography:'  # Bibliography
            ]
        }
        
        # Patrones de tipos de documento
        self.document_type_patterns = {
            DocumentType.EMAIL: [r'From:', r'To:', r'Subject:', r'Date:'],
            DocumentType.REPORT: [r'Executive Summary:', r'Conclusion:', r'Recommendations:'],
            DocumentType.ARTICLE: [r'By\s+\w+', r'Published:', r'Updated:'],
            DocumentType.ACADEMIC_PAPER: [r'Abstract:', r'Keywords:', r'References:'],
            DocumentType.LEGAL_DOCUMENT: [r'WHEREAS', r'THEREFORE', r'IN WITNESS WHEREOF']
        }
        
        logger.info("DocumentAnalyzer inicializado")
    
    async def initialize(self):
        """Inicializar el analizador de documentos."""
        try:
            logger.info("DocumentAnalyzer inicializado exitosamente")
        except Exception as e:
            logger.error(f"Error al inicializar DocumentAnalyzer: {e}")
            raise
    
    async def analyze_document(
        self,
        content: str,
        title: str = "Untitled Document",
        document_type: Optional[DocumentType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentAnalysis:
        """
        Analizar un documento completo.
        
        Args:
            content: Contenido del documento
            title: Título del documento
            document_type: Tipo de documento
            metadata: Metadatos adicionales
            
        Returns:
            Análisis completo del documento
        """
        try:
            # Generar ID único
            document_id = str(uuid.uuid4())
            
            # Verificar cache
            cache_key = self._generate_cache_key(content, title)
            if cache_key in self.document_cache:
                cached_analysis = self.document_cache[cache_key]
                if datetime.now() - cached_analysis['timestamp'] < timedelta(seconds=self.cache_ttl):
                    return cached_analysis['analysis']
            
            # Detectar tipo de documento si no se especifica
            if not document_type:
                document_type = await self._detect_document_type(content)
            
            # Crear metadatos
            doc_metadata = await self._create_document_metadata(
                document_id, title, content, document_type, metadata
            )
            
            # Detectar estructura del documento
            structure = await self._detect_document_structure(content)
            
            # Extraer secciones
            sections = await self._extract_document_sections(content, document_id)
            
            # Generar resumen
            summary = await self._generate_document_summary(content)
            
            # Extraer puntos clave
            key_points = await self._extract_key_points(content)
            
            # Extraer temas principales
            main_topics = await self._extract_main_topics(content)
            
            # Extraer entidades
            entities = await self._extract_document_entities(content)
            
            # Análisis de sentimiento
            sentiment_analysis = await self._analyze_document_sentiment(content)
            
            # Calcular puntuación de legibilidad
            readability_score = await self._calculate_document_readability(content)
            
            # Calcular puntuación de calidad
            quality_score = await self._calculate_document_quality(content, structure, sections)
            
            # Generar recomendaciones
            recommendations = await self._generate_document_recommendations(
                content, structure, readability_score, quality_score
            )
            
            # Crear análisis completo
            analysis = DocumentAnalysis(
                document_id=document_id,
                metadata=doc_metadata,
                structure=structure,
                sections=sections,
                summary=summary,
                key_points=key_points,
                main_topics=main_topics,
                entities=entities,
                sentiment_analysis=sentiment_analysis,
                readability_score=readability_score,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
            # Almacenar análisis
            self.documents[document_id] = analysis
            
            # Guardar en cache
            self.document_cache[cache_key] = {
                'analysis': analysis,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Documento {document_id} analizado exitosamente")
            return analysis
            
        except Exception as e:
            logger.error(f"Error al analizar documento: {e}")
            raise
    
    async def _detect_document_type(self, content: str) -> DocumentType:
        """Detectar tipo de documento basado en patrones."""
        try:
            content_lower = content.lower()
            
            # Verificar patrones específicos
            for doc_type, patterns in self.document_type_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return doc_type
            
            # Análisis de estructura para determinar tipo
            if re.search(r'<html|<body|<div', content, re.IGNORECASE):
                return DocumentType.HTML
            elif re.search(r'^#+\s+', content, re.MULTILINE):
                return DocumentType.MARKDOWN
            elif len(content.split('\n')) > 50 and re.search(r'^\d+\.\s+', content, re.MULTILINE):
                return DocumentType.REPORT
            elif len(content.split()) < 1000:
                return DocumentType.EMAIL
            else:
                return DocumentType.ARTICLE
                
        except Exception as e:
            logger.error(f"Error al detectar tipo de documento: {e}")
            return DocumentType.TEXT
    
    async def _detect_document_structure(self, content: str) -> DocumentStructure:
        """Detectar estructura del documento."""
        try:
            # Verificar patrones de estructura
            for structure, patterns in self.structure_patterns.items():
                pattern_matches = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    pattern_matches += len(matches)
                
                if pattern_matches > 3:  # Umbral para considerar la estructura
                    return structure
            
            # Análisis adicional
            lines = content.split('\n')
            header_count = sum(1 for line in lines if re.match(r'^#+\s+', line))
            list_count = sum(1 for line in lines if re.match(r'^\d+\.\s+', line))
            
            if header_count > 5:
                return DocumentStructure.HIERARCHICAL
            elif list_count > 10:
                return DocumentStructure.TABULAR
            else:
                return DocumentStructure.LINEAR
                
        except Exception as e:
            logger.error(f"Error al detectar estructura del documento: {e}")
            return DocumentStructure.LINEAR
    
    async def _extract_document_sections(self, content: str, document_id: str) -> List[DocumentSection]:
        """Extraer secciones del documento."""
        try:
            sections = []
            
            # Dividir por headers
            header_pattern = r'^(#+\s+.*?)$'
            parts = re.split(header_pattern, content, flags=re.MULTILINE)
            
            current_position = 0
            section_id = 0
            
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    header = parts[i].strip()
                    section_content = parts[i + 1].strip()
                    
                    if header and section_content:
                        # Determinar nivel del header
                        level = len(header) - len(header.lstrip('#'))
                        
                        # Crear sección
                        section = DocumentSection(
                            section_id=f"{document_id}_section_{section_id}",
                            title=header.lstrip('# '),
                            content=section_content,
                            level=level,
                            start_position=current_position,
                            end_position=current_position + len(section_content),
                            word_count=len(section_content.split())
                        )
                        
                        # Analizar sección
                        section.topics = await self._extract_section_topics(section_content)
                        section.entities = await self._extract_section_entities(section_content)
                        section.sentiment = await self._analyze_section_sentiment(section_content)
                        
                        sections.append(section)
                        current_position += len(section_content)
                        section_id += 1
            
            # Si no se encontraron headers, crear una sección única
            if not sections:
                section = DocumentSection(
                    section_id=f"{document_id}_section_0",
                    title="Main Content",
                    content=content,
                    level=1,
                    start_position=0,
                    end_position=len(content),
                    word_count=len(content.split())
                )
                
                section.topics = await self._extract_section_topics(content)
                section.entities = await self._extract_section_entities(content)
                section.sentiment = await self._analyze_section_sentiment(content)
                
                sections.append(section)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error al extraer secciones del documento: {e}")
            return []
    
    async def _generate_document_summary(self, content: str) -> str:
        """Generar resumen del documento."""
        try:
            # Usar transformer para resumización si está disponible
            if hasattr(self.transformer_manager, 'summarize_advanced'):
                summary_result = await self.transformer_manager.summarize_advanced(content)
                return summary_result.get('summary', '')
            
            # Resumización básica
            sentences = content.split('.')
            if len(sentences) <= 3:
                return content
            
            # Tomar las primeras y últimas oraciones
            summary_sentences = sentences[:2] + sentences[-1:]
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Error al generar resumen del documento: {e}")
            return "No se pudo generar resumen del documento."
    
    async def _extract_key_points(self, content: str) -> List[str]:
        """Extraer puntos clave del documento."""
        try:
            # Buscar patrones de puntos clave
            key_point_patterns = [
                r'•\s+(.+?)(?=\n|$)',
                r'-\s+(.+?)(?=\n|$)',
                r'\d+\.\s+(.+?)(?=\n|$)',
                r'Key\s+point[s]?:\s*(.+?)(?=\n|$)',
                r'Important:\s*(.+?)(?=\n|$)'
            ]
            
            key_points = []
            for pattern in key_point_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                key_points.extend(matches)
            
            # Limpiar y limitar puntos clave
            key_points = [point.strip() for point in key_points if len(point.strip()) > 10]
            return key_points[:10]  # Máximo 10 puntos clave
            
        except Exception as e:
            logger.error(f"Error al extraer puntos clave: {e}")
            return []
    
    async def _extract_main_topics(self, content: str) -> List[str]:
        """Extraer temas principales del documento."""
        try:
            # Usar embeddings para clustering de temas
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            
            if len(sentences) < 3:
                return []
            
            # Obtener embeddings de oraciones
            sentence_embeddings = []
            for sentence in sentences[:20]:  # Limitar para rendimiento
                embedding = await self.embedding_manager.get_embedding(sentence)
                sentence_embeddings.append(embedding)
            
            # Clustering básico de temas
            topics = await self._cluster_sentences_by_topic(sentences[:20], sentence_embeddings)
            
            return topics[:5]  # Máximo 5 temas principales
            
        except Exception as e:
            logger.error(f"Error al extraer temas principales: {e}")
            return []
    
    async def _extract_document_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extraer entidades del documento."""
        try:
            # Usar transformer para NER si está disponible
            if hasattr(self.transformer_manager, 'extract_entities_advanced'):
                entities_result = await self.transformer_manager.extract_entities_advanced(content)
                return entities_result.get('entities', [])
            
            # Extracción básica de entidades
            entities = []
            
            # Patrones para diferentes tipos de entidades
            patterns = {
                'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                'ORGANIZATION': r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'PHONE': r'\b\d{3}-\d{3}-\d{4}\b',
                'DATE': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                'MONEY': r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b'
            }
            
            for entity_type, pattern in patterns.items():
                matches = re.findall(pattern, content)
                for match in matches:
                    entities.append({
                        'text': match,
                        'type': entity_type,
                        'start': content.find(match),
                        'end': content.find(match) + len(match)
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error al extraer entidades del documento: {e}")
            return []
    
    async def _analyze_document_sentiment(self, content: str) -> Dict[str, Any]:
        """Analizar sentimiento del documento."""
        try:
            # Usar transformer para análisis de sentimiento si está disponible
            if hasattr(self.transformer_manager, 'analyze_sentiment_advanced'):
                sentiment_result = await self.transformer_manager.analyze_sentiment_advanced(content)
                return sentiment_result
            
            # Análisis básico de sentimiento
            positive_words = ['good', 'excellent', 'great', 'amazing', 'wonderful', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor']
            
            content_lower = content.lower()
            positive_count = sum(1 for word in positive_words if word in content_lower)
            negative_count = sum(1 for word in negative_words if word in content_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                confidence = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
            elif negative_count > positive_count:
                sentiment = "negative"
                confidence = negative_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            return {
                "overall_sentiment": sentiment,
                "confidence": confidence,
                "positive_score": positive_count,
                "negative_score": negative_count
            }
            
        except Exception as e:
            logger.error(f"Error al analizar sentimiento del documento: {e}")
            return {"overall_sentiment": "neutral", "confidence": 0.0}
    
    async def _calculate_document_readability(self, content: str) -> float:
        """Calcular legibilidad del documento."""
        try:
            # Implementación simplificada de Flesch Reading Ease
            sentences = [s for s in content.split('.') if s.strip()]
            words = content.split()
            
            if not sentences or not words:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error al calcular legibilidad del documento: {e}")
            return 0.0
    
    async def _calculate_document_quality(
        self,
        content: str,
        structure: DocumentStructure,
        sections: List[DocumentSection]
    ) -> float:
        """Calcular calidad del documento."""
        try:
            # Factores de calidad
            readability_score = await self._calculate_document_readability(content)
            structure_score = self._calculate_structure_score(structure, sections)
            completeness_score = self._calculate_completeness_score(content, sections)
            coherence_score = await self._calculate_coherence_score(content)
            
            # Ponderación de factores
            quality_score = (
                readability_score * 0.3 +
                structure_score * 0.25 +
                completeness_score * 0.25 +
                coherence_score * 0.2
            )
            
            return min(100.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error al calcular calidad del documento: {e}")
            return 0.0
    
    async def _generate_document_recommendations(
        self,
        content: str,
        structure: DocumentStructure,
        readability_score: float,
        quality_score: float
    ) -> List[str]:
        """Generar recomendaciones para el documento."""
        try:
            recommendations = []
            
            # Recomendaciones de legibilidad
            if readability_score < 30:
                recommendations.append("Considera simplificar el lenguaje para mejorar la legibilidad")
            elif readability_score > 80:
                recommendations.append("El documento es muy simple, considera agregar más complejidad")
            
            # Recomendaciones de estructura
            if structure == DocumentStructure.LINEAR and len(content.split()) > 1000:
                recommendations.append("Considera dividir el documento en secciones con headers")
            
            # Recomendaciones de calidad
            if quality_score < 60:
                recommendations.append("El documento necesita mejoras en estructura y coherencia")
            
            # Recomendaciones de longitud
            word_count = len(content.split())
            if word_count < 300:
                recommendations.append("El documento es muy corto, considera expandirlo")
            elif word_count > 5000:
                recommendations.append("El documento es muy largo, considera dividirlo en partes")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error al generar recomendaciones del documento: {e}")
            return []
    
    # Métodos auxiliares
    async def _create_document_metadata(
        self,
        document_id: str,
        title: str,
        content: str,
        document_type: DocumentType,
        metadata: Optional[Dict[str, Any]]
    ) -> DocumentMetadata:
        """Crear metadatos del documento."""
        return DocumentMetadata(
            document_id=document_id,
            title=title,
            language=Language.ENGLISH,  # Detectar idioma en producción
            document_type=document_type,
            word_count=len(content.split()),
            character_count=len(content),
            tags=metadata.get('tags', []) if metadata else [],
            categories=metadata.get('categories', []) if metadata else []
        )
    
    async def _extract_section_topics(self, content: str) -> List[str]:
        """Extraer temas de una sección."""
        # Implementación básica
        words = content.lower().split()
        word_freq = Counter(words)
        return [word for word, freq in word_freq.most_common(3) if len(word) > 3]
    
    async def _extract_section_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extraer entidades de una sección."""
        # Implementación básica
        return []
    
    async def _analyze_section_sentiment(self, content: str) -> SentimentType:
        """Analizar sentimiento de una sección."""
        # Implementación básica
        return SentimentType.NEUTRAL
    
    async def _cluster_sentences_by_topic(self, sentences: List[str], embeddings: List[Any]) -> List[str]:
        """Agrupar oraciones por tema."""
        # Implementación básica - en producción usar clustering real
        return sentences[:3]
    
    def _calculate_structure_score(self, structure: DocumentStructure, sections: List[DocumentSection]) -> float:
        """Calcular puntuación de estructura."""
        if structure == DocumentStructure.HIERARCHICAL and len(sections) > 3:
            return 90.0
        elif structure == DocumentStructure.LINEAR and len(sections) == 1:
            return 60.0
        else:
            return 70.0
    
    def _calculate_completeness_score(self, content: str, sections: List[DocumentSection]) -> float:
        """Calcular puntuación de completitud."""
        word_count = len(content.split())
        if word_count > 500:
            return 80.0
        elif word_count > 200:
            return 60.0
        else:
            return 40.0
    
    async def _calculate_coherence_score(self, content: str) -> float:
        """Calcular puntuación de coherencia."""
        # Implementación básica
        return 75.0
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas en una palabra."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _generate_cache_key(self, content: str, title: str) -> str:
        """Generar clave de cache."""
        import hashlib
        return hashlib.md5(f"{title}:{content}".encode()).hexdigest()
    
    async def get_document_analysis(self, document_id: str) -> Optional[DocumentAnalysis]:
        """Obtener análisis de documento."""
        return self.documents.get(document_id)
    
    async def get_document_summary(self, document_id: str) -> Optional[str]:
        """Obtener resumen de documento."""
        analysis = self.documents.get(document_id)
        return analysis.summary if analysis else None
    
    async def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Buscar documentos por consulta."""
        try:
            results = []
            query_embedding = await self.embedding_manager.get_embedding(query)
            
            for doc_id, analysis in self.documents.items():
                # Calcular similitud con el resumen
                summary_embedding = await self.embedding_manager.get_embedding(analysis.summary)
                similarity = await self.embedding_manager.calculate_similarity(query, analysis.summary)
                
                if similarity > 0.3:  # Umbral de similitud
                    results.append({
                        "document_id": doc_id,
                        "title": analysis.metadata.title,
                        "summary": analysis.summary,
                        "similarity": similarity,
                        "main_topics": analysis.main_topics,
                        "quality_score": analysis.quality_score
                    })
            
            # Ordenar por similitud
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error al buscar documentos: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del analizador de documentos."""
        try:
            return {
                "status": "healthy",
                "documents_analyzed": len(self.documents),
                "cache_size": len(self.document_cache),
                "embedding_manager_status": await self.embedding_manager.health_check(),
                "transformer_manager_status": await self.transformer_manager.health_check(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error en health check de DocumentAnalyzer: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




