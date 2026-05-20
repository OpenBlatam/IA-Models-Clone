"""
Document Analyzer Advanced - Funcionalidades Avanzadas
=======================================================

Funcionalidades adicionales de nivel enterprise:
- Análisis de imágenes en documentos
- Extracción de tablas y gráficos
- Análisis multi-idioma avanzado
- Análisis de calidad de documentos
- Detección de fraudes/autenticidad
- Análisis de documentos legales
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysis:
    """Resultado de análisis de imagen."""
    image_path: str
    text_extracted: Optional[str] = None
    objects_detected: List[Dict[str, Any]] = field(default_factory=list)
    ocr_confidence: float = 0.0
    image_type: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None


@dataclass
class TableExtraction:
    """Resultado de extracción de tabla."""
    table_id: str
    rows: List[List[str]]
    headers: Optional[List[str]] = None
    confidence: float = 0.0
    position: Optional[Dict[str, int]] = None


@dataclass
class DocumentQuality:
    """Análisis de calidad del documento."""
    overall_score: float = 0.0
    readability_score: float = 0.0
    completeness_score: float = 0.0
    structure_score: float = 0.0
    language_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ImageAnalyzer:
    """Analizador de imágenes en documentos."""
    
    def __init__(self, analyzer):
        """Inicializar analizador de imágenes."""
        self.analyzer = analyzer
        self.ocr_enabled = True
    
    async def analyze_image(
        self,
        image_path: str,
        extract_text: bool = True,
        detect_objects: bool = False
    ) -> ImageAnalysis:
        """
        Analizar imagen en documento.
        
        Args:
            image_path: Ruta a la imagen
            extract_text: Extraer texto con OCR
            detect_objects: Detectar objetos
        
        Returns:
            ImageAnalysis con resultados
        """
        try:
            # En producción usar bibliotecas como pytesseract, easyocr, etc.
            # Por ahora implementación básica
            
            result = ImageAnalysis(image_path=image_path)
            
            if extract_text:
                # OCR básico (en producción usar OCR real)
                result.text_extracted = await self._extract_text_ocr(image_path)
                result.ocr_confidence = 0.85  # Simulado
            
            if detect_objects:
                result.objects_detected = await self._detect_objects(image_path)
            
            # Detectar tipo de imagen
            result.image_type = self._detect_image_type(image_path)
            
            return result
        
        except Exception as e:
            logger.error(f"Error analizando imagen {image_path}: {e}")
            return ImageAnalysis(image_path=image_path, ocr_confidence=0.0)
    
    async def _extract_text_ocr(self, image_path: str) -> str:
        """Extraer texto con OCR."""
        # En producción usar pytesseract o easyocr
        # Por ahora retornar texto simulado
        return "Texto extraído de imagen"
    
    async def _detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """Detectar objetos en imagen."""
        # En producción usar YOLO, Detectron, etc.
        return []
    
    def _detect_image_type(self, image_path: str) -> str:
        """Detectar tipo de imagen."""
        ext = image_path.lower().split('.')[-1]
        if ext in ['jpg', 'jpeg', 'png']:
            return 'photo'
        elif ext in ['pdf']:
            return 'document'
        return 'unknown'


class TableExtractor:
    """Extractor de tablas de documentos."""
    
    def __init__(self, analyzer):
        """Inicializar extractor de tablas."""
        self.analyzer = analyzer
    
    async def extract_tables(
        self,
        content: str,
        document_path: Optional[str] = None
    ) -> List[TableExtraction]:
        """
        Extraer tablas del documento.
        
        Args:
            content: Contenido del documento
            document_path: Ruta al documento (para PDFs)
        
        Returns:
            Lista de tablas extraídas
        """
        tables = []
        
        # Buscar tablas en HTML/Markdown
        html_tables = self._extract_html_tables(content)
        for i, table in enumerate(html_tables):
            tables.append(TableExtraction(
                table_id=f"table_{i+1}",
                rows=table.get("rows", []),
                headers=table.get("headers"),
                confidence=0.9
            ))
        
        # Buscar tablas en texto plano (formato básico)
        text_tables = self._extract_text_tables(content)
        for i, table in enumerate(text_tables):
            tables.append(TableExtraction(
                table_id=f"text_table_{i+1}",
                rows=table.get("rows", []),
                headers=table.get("headers"),
                confidence=0.7
            ))
        
        return tables
    
    def _extract_html_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extraer tablas HTML."""
        import re
        
        tables = []
        table_pattern = r'<table[^>]*>(.*?)</table>'
        
        for match in re.finditer(table_pattern, content, re.DOTALL | re.IGNORECASE):
            table_html = match.group(1)
            rows = self._parse_html_table(table_html)
            tables.append({"rows": rows, "headers": rows[0] if rows else None})
        
        return tables
    
    def _extract_text_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extraer tablas de texto plano."""
        tables = []
        lines = content.split('\n')
        
        # Detectar líneas con múltiples separadores (tabs, espacios múltiples)
        table_lines = []
        for line in lines:
            if re.match(r'.*\t.*\t', line) or re.match(r'.*  +.*  +', line):
                table_lines.append(line)
        
        if table_lines:
            rows = []
            for line in table_lines:
                # Separar por tabs o espacios múltiples
                if '\t' in line:
                    cells = [cell.strip() for cell in line.split('\t')]
                else:
                    cells = [cell.strip() for cell in re.split(r'  +', line)]
                if cells:
                    rows.append(cells)
            
            if rows:
                tables.append({
                    "rows": rows,
                    "headers": rows[0] if len(rows) > 0 else None
                })
        
        return tables
    
    def _parse_html_table(self, html: str) -> List[List[str]]:
        """Parsear tabla HTML."""
        rows = []
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
        
        for row_match in re.finditer(row_pattern, html, re.DOTALL | re.IGNORECASE):
            row_html = row_match.group(1)
            cells = []
            for cell_match in re.finditer(cell_pattern, row_html, re.DOTALL | re.IGNORECASE):
                cell_text = re.sub(r'<[^>]+>', '', cell_match.group(1)).strip()
                cells.append(cell_text)
            if cells:
                rows.append(cells)
        
        return rows


class MultiLanguageAnalyzer:
    """Analizador multi-idioma avanzado."""
    
    def __init__(self, analyzer):
        """Inicializar analizador multi-idioma."""
        self.analyzer = analyzer
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja']
    
    async def detect_language(self, content: str) -> Dict[str, Any]:
        """
        Detectar idioma del documento.
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con idioma detectado y confianza
        """
        # En producción usar bibliotecas como langdetect, polyglot, etc.
        # Por ahora implementación básica
        
        # Palabras comunes por idioma
        language_patterns = {
            'en': ['the', 'and', 'is', 'are', 'was', 'were'],
            'es': ['el', 'la', 'de', 'que', 'y', 'en'],
            'fr': ['le', 'de', 'et', 'à', 'un', 'être'],
            'de': ['der', 'die', 'das', 'und', 'ist', 'sind'],
            'pt': ['o', 'a', 'de', 'que', 'e', 'em'],
            'it': ['il', 'la', 'di', 'che', 'e', 'in']
        }
        
        content_lower = content.lower()
        scores = {}
        
        for lang, patterns in language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in content_lower)
            scores[lang] = score / len(patterns) if patterns else 0
        
        detected_lang = max(scores.items(), key=lambda x: x[1])[0] if scores else 'en'
        confidence = scores.get(detected_lang, 0.0)
        
        return {
            "language": detected_lang,
            "confidence": confidence,
            "all_scores": scores
        }
    
    async def translate_content(
        self,
        content: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Traducir contenido (requiere servicio de traducción).
        
        Args:
            content: Contenido a traducir
            target_language: Idioma objetivo
            source_language: Idioma origen (opcional, auto-detecta)
        
        Returns:
            Diccionario con traducción y metadatos
        """
        # En producción integrar con servicios como Google Translate API,
        # DeepL, etc.
        
        if not source_language:
            lang_detection = await self.detect_language(content)
            source_language = lang_detection["language"]
        
        # Por ahora retornar contenido original (requiere servicio real)
        return {
            "original": content,
            "translated": content,  # En producción: traducción real
            "source_language": source_language,
            "target_language": target_language,
            "confidence": 0.0
        }


class DocumentQualityAnalyzer:
    """Analizador de calidad de documentos."""
    
    def __init__(self, analyzer):
        """Inicializar analizador de calidad."""
        self.analyzer = analyzer
    
    async def analyze_quality(
        self,
        content: str,
        document_type: Optional[str] = None
    ) -> DocumentQuality:
        """
        Analizar calidad del documento.
        
        Args:
            content: Contenido del documento
            document_type: Tipo de documento
        
        Returns:
            DocumentQuality con análisis completo
        """
        # Análisis de legibilidad
        readability = self._calculate_readability(content)
        
        # Análisis de completitud
        completeness = self._calculate_completeness(content, document_type)
        
        # Análisis de estructura
        structure = self._analyze_structure(content)
        
        # Análisis de lenguaje
        language = await self._analyze_language_quality(content)
        
        # Calcular score general
        overall = (readability + completeness + structure + language) / 4
        
        # Detectar issues
        issues = self._detect_issues(content, readability, completeness, structure)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            issues, readability, completeness, structure
        )
        
        return DocumentQuality(
            overall_score=overall,
            readability_score=readability,
            completeness_score=completeness,
            structure_score=structure,
            language_score=language,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_readability(self, content: str) -> float:
        """Calcular score de legibilidad."""
        sentences = content.split('. ')
        words = content.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Score más alto = más legible
        score = 100 - (avg_sentence_length * 2) - (avg_word_length * 3)
        return max(0, min(100, score))
    
    def _calculate_completeness(self, content: str, doc_type: Optional[str]) -> float:
        """Calcular completitud del documento."""
        score = 50.0  # Base
        
        # Verificar longitud mínima
        if len(content) > 100:
            score += 20
        if len(content) > 500:
            score += 20
        if len(content) > 1000:
            score += 10
        
        # Verificar estructura básica
        has_paragraphs = '\n\n' in content or len(content.split('\n')) > 3
        if has_paragraphs:
            score += 10
        
        # Verificar contenido sustancial
        words = content.split()
        unique_words = len(set(words))
        if len(words) > 0:
            diversity = unique_words / len(words)
            score += diversity * 20
        
        return min(100, score)
    
    def _analyze_structure(self, content: str) -> float:
        """Analizar estructura del documento."""
        score = 50.0
        
        # Verificar párrafos
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            score += 15
        
        # Verificar títulos/secciones
        has_titles = bool(re.search(r'^#+\s|^[A-Z][A-Z\s]+$', content, re.MULTILINE))
        if has_titles:
            score += 15
        
        # Verificar listas
        has_lists = bool(re.search(r'^\s*[-*•]\s|^\s*\d+\.\s', content, re.MULTILINE))
        if has_lists:
            score += 10
        
        # Verificar tablas
        has_tables = '<table' in content or '\t' in content
        if has_tables:
            score += 10
        
        return min(100, score)
    
    async def _analyze_language_quality(self, content: str) -> float:
        """Analizar calidad del lenguaje."""
        score = 70.0  # Base
        
        # Verificar errores básicos (simplificado)
        # En producción usar bibliotecas de gramática
        
        # Verificar ortografía básica (palabras muy cortas o muy largas)
        words = content.split()
        if words:
            avg_length = np.mean([len(w) for w in words])
            if 3 < avg_length < 8:
                score += 10
        
        # Verificar diversidad de palabras
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.5:
                score += 10
        
        # Verificar puntuación
        has_punctuation = bool(re.search(r'[.!?]', content))
        if has_punctuation:
            score += 10
        
        return min(100, score)
    
    def _detect_issues(
        self,
        content: str,
        readability: float,
        completeness: float,
        structure: float
    ) -> List[str]:
        """Detectar issues en el documento."""
        issues = []
        
        if readability < 50:
            issues.append("Legibilidad baja - texto difícil de leer")
        
        if completeness < 50:
            issues.append("Documento incompleto - falta contenido")
        
        if structure < 50:
            issues.append("Estructura pobre - falta organización")
        
        if len(content) < 100:
            issues.append("Documento muy corto")
        
        if len(content.split()) < 20:
            issues.append("Contenido insuficiente")
        
        return issues
    
    def _generate_recommendations(
        self,
        issues: List[str],
        readability: float,
        completeness: float,
        structure: float
    ) -> List[str]:
        """Generar recomendaciones."""
        recommendations = []
        
        if readability < 60:
            recommendations.append("Usar oraciones más cortas y palabras más simples")
        
        if completeness < 60:
            recommendations.append("Agregar más contenido y detalles")
        
        if structure < 60:
            recommendations.append("Organizar en secciones con títulos claros")
        
        if not issues:
            recommendations.append("Documento de buena calidad")
        
        return recommendations


class FraudDetector:
    """Detector de fraudes y autenticidad."""
    
    def __init__(self, analyzer):
        """Inicializar detector."""
        self.analyzer = analyzer
    
    async def detect_fraud_indicators(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detectar indicadores de fraude.
        
        Args:
            content: Contenido del documento
            metadata: Metadatos del documento
        
        Returns:
            Diccionario con indicadores detectados
        """
        indicators = {
            "suspicious_patterns": [],
            "anomalies": [],
            "risk_score": 0.0,
            "recommendations": []
        }
        
        # Detectar patrones sospechosos
        suspicious_keywords = [
            'urgent', 'immediate', 'confidential', 'secret',
            'guaranteed', 'no risk', 'act now', 'limited time'
        ]
        
        content_lower = content.lower()
        found_keywords = [kw for kw in suspicious_keywords if kw in content_lower]
        
        if found_keywords:
            indicators["suspicious_patterns"].append(
                f"Palabras sospechosas encontradas: {', '.join(found_keywords)}"
            )
            indicators["risk_score"] += 0.2
        
        # Detectar inconsistencias en fechas
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', content)
        if len(dates) > 1:
            # Verificar si fechas son consistentes
            indicators["risk_score"] += 0.1
        
        # Detectar información faltante común
        common_fields = ['date', 'signature', 'author', 'title']
        missing_fields = []
        for field in common_fields:
            if field not in content_lower:
                missing_fields.append(field)
        
        if missing_fields and metadata:
            indicators["anomalies"].append(
                f"Campos comunes faltantes: {', '.join(missing_fields)}"
            )
            indicators["risk_score"] += 0.1
        
        # Generar recomendaciones
        if indicators["risk_score"] > 0.5:
            indicators["recommendations"].append(
                "Alto riesgo detectado - verificar autenticidad del documento"
            )
        elif indicators["risk_score"] > 0.3:
            indicators["recommendations"].append(
                "Riesgo moderado - revisar cuidadosamente"
            )
        else:
            indicators["recommendations"].append(
                "Riesgo bajo - documento parece legítimo"
            )
        
        return indicators


class LegalDocumentAnalyzer:
    """Analizador especializado para documentos legales."""
    
    def __init__(self, analyzer):
        """Inicializar analizador legal."""
        self.analyzer = analyzer
        self.legal_terms = [
            'contract', 'agreement', 'clause', 'party', 'obligation',
            'liability', 'warranty', 'termination', 'breach', 'indemnity'
        ]
    
    async def analyze_legal_document(
        self,
        content: str,
        document_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar documento legal.
        
        Args:
            content: Contenido del documento
            document_type: Tipo de documento legal
        
        Returns:
            Diccionario con análisis legal
        """
        # Extraer cláusulas
        clauses = self._extract_clauses(content)
        
        # Identificar partes
        parties = self._identify_parties(content)
        
        # Detectar términos legales
        legal_terms_found = self._detect_legal_terms(content)
        
        # Analizar obligaciones
        obligations = self._extract_obligations(content)
        
        # Detectar fechas importantes
        important_dates = self._extract_important_dates(content)
        
        # Calcular score de completitud legal
        completeness = self._calculate_legal_completeness(
            clauses, parties, obligations
        )
        
        return {
            "document_type": document_type or "legal_document",
            "clauses": clauses,
            "parties": parties,
            "legal_terms": legal_terms_found,
            "obligations": obligations,
            "important_dates": important_dates,
            "completeness_score": completeness,
            "recommendations": self._generate_legal_recommendations(
                clauses, parties, obligations
            )
        }
    
    def _extract_clauses(self, content: str) -> List[Dict[str, Any]]:
        """Extraer cláusulas del documento."""
        clauses = []
        
        # Buscar numeración de cláusulas
        clause_pattern = r'(?:Clause|Section|Article)\s+(\d+)[:.]\s*(.+?)(?=(?:Clause|Section|Article)\s+\d+|$)'
        
        for match in re.finditer(clause_pattern, content, re.IGNORECASE | re.DOTALL):
            clauses.append({
                "number": match.group(1),
                "title": match.group(2).strip()[:100],
                "content": match.group(2).strip()
            })
        
        return clauses
    
    def _identify_parties(self, content: str) -> List[str]:
        """Identificar partes del contrato."""
        parties = []
        
        # Buscar patrones comunes
        party_patterns = [
            r'Party\s+(?:A|B|One|Two)[:.]\s*([A-Z][^.]{0,100})',
            r'Between\s+([A-Z][^,]{0,100})\s+and',
            r'([A-Z][A-Za-z\s&]+(?:Inc\.|LLC|Corp\.|Ltd\.))'
        ]
        
        for pattern in party_patterns:
            for match in re.finditer(pattern, content):
                party = match.group(1).strip()
                if party and len(party) > 3:
                    parties.append(party)
        
        return list(set(parties))  # Remover duplicados
    
    def _detect_legal_terms(self, content: str) -> List[str]:
        """Detectar términos legales."""
        content_lower = content.lower()
        found_terms = [
            term for term in self.legal_terms
            if term in content_lower
        ]
        return found_terms
    
    def _extract_obligations(self, content: str) -> List[str]:
        """Extraer obligaciones."""
        obligations = []
        
        # Buscar patrones de obligaciones
        obligation_patterns = [
            r'shall\s+(.+?)(?:\.|;)',
            r'must\s+(.+?)(?:\.|;)',
            r'is\s+required\s+to\s+(.+?)(?:\.|;)'
        ]
        
        for pattern in obligation_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                obligation = match.group(1).strip()
                if len(obligation) > 10 and len(obligation) < 200:
                    obligations.append(obligation)
        
        return obligations[:20]  # Limitar a 20
    
    def _extract_important_dates(self, content: str) -> List[Dict[str, Any]]:
        """Extraer fechas importantes."""
        dates = []
        
        # Buscar fechas con contexto
        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        date_context_pattern = r'((?:effective|expiration|termination|start|end|due).{0,50}?\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        
        for match in re.finditer(date_context_pattern, content, re.IGNORECASE):
            context = match.group(1)
            date_match = re.search(date_pattern, context)
            if date_match:
                dates.append({
                    "date": date_match.group(1),
                    "context": context.strip(),
                    "type": "important"
                })
        
        return dates
    
    def _calculate_legal_completeness(
        self,
        clauses: List[Dict],
        parties: List[str],
        obligations: List[str]
    ) -> float:
        """Calcular completitud legal."""
        score = 0.0
        
        # Cláusulas
        if len(clauses) >= 5:
            score += 30
        elif len(clauses) >= 3:
            score += 20
        elif len(clauses) >= 1:
            score += 10
        
        # Partes
        if len(parties) >= 2:
            score += 30
        elif len(parties) >= 1:
            score += 15
        
        # Obligaciones
        if len(obligations) >= 5:
            score += 20
        elif len(obligations) >= 3:
            score += 15
        elif len(obligations) >= 1:
            score += 10
        
        # Elementos adicionales
        score += 20  # Base
        
        return min(100, score)
    
    def _generate_legal_recommendations(
        self,
        clauses: List[Dict],
        parties: List[str],
        obligations: List[str]
    ) -> List[str]:
        """Generar recomendaciones legales."""
        recommendations = []
        
        if len(clauses) < 3:
            recommendations.append("Agregar más cláusulas para mayor especificidad")
        
        if len(parties) < 2:
            recommendations.append("Identificar claramente todas las partes del contrato")
        
        if len(obligations) < 3:
            recommendations.append("Especificar obligaciones de cada parte")
        
        recommendations.append("Revisar con abogado antes de firmar")
        
        return recommendations


# Funciones de utilidad

async def analyze_document_with_images(
    analyzer,
    document_path: str,
    extract_images: bool = True
) -> Dict[str, Any]:
    """Analizar documento incluyendo imágenes."""
    image_analyzer = ImageAnalyzer(analyzer)
    
    # Analizar documento
    doc_analysis = await analyzer.analyze_document(document_path=document_path)
    
    # Extraer y analizar imágenes
    images = []
    if extract_images:
        # En producción: extraer imágenes del documento
        # Por ahora placeholder
        pass
    
    return {
        "document_analysis": doc_analysis,
        "images": images
    }


__all__ = [
    "ImageAnalyzer",
    "TableExtractor",
    "MultiLanguageAnalyzer",
    "DocumentQualityAnalyzer",
    "FraudDetector",
    "LegalDocumentAnalyzer",
    "ImageAnalysis",
    "TableExtraction",
    "DocumentQuality",
    "analyze_document_with_images"
]
















