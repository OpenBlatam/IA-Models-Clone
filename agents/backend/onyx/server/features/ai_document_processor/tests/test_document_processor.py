"""
Pruebas para el procesador de documentos AI
==========================================
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from services.document_processor import DocumentProcessor
from services.ai_classifier import AIClassifier
from services.professional_transformer import ProfessionalTransformer
from models.document_models import (
    DocumentProcessingRequest, ProfessionalFormat, DocumentType,
    DocumentArea, DocumentCategory
)

class TestDocumentProcessor:
    """Pruebas para el procesador principal"""
    
    @pytest.fixture
    async def processor(self):
        """Fixture para el procesador"""
        processor = DocumentProcessor()
        await processor.initialize()
        return processor
    
    @pytest.fixture
    def sample_markdown_content(self):
        """Contenido de ejemplo en Markdown"""
        return """
        # Análisis de Mercado Digital
        
        ## Resumen Ejecutivo
        
        El mercado digital ha experimentado un crecimiento exponencial.
        
        ## Recomendaciones
        
        1. Implementar estrategia digital
        2. Capacitar personal
        3. Invertir en tecnología
        """
    
    @pytest.fixture
    def sample_text_file(self, sample_markdown_content):
        """Archivo de texto temporal"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_markdown_content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    async def test_processor_initialization(self, processor):
        """Prueba la inicialización del procesador"""
        assert processor is not None
        assert processor.file_handler_factory is not None
        assert processor.ai_classifier is not None
        assert processor.professional_transformer is not None
    
    async def test_extract_text_from_markdown(self, processor, sample_text_file):
        """Prueba la extracción de texto de Markdown"""
        text = await processor.extract_text(sample_text_file, "test.md")
        
        assert text is not None
        assert len(text) > 0
        assert "Análisis de Mercado Digital" in text
        assert "Recomendaciones" in text
    
    async def test_process_document_consultancy(self, processor, sample_text_file):
        """Prueba el procesamiento completo a formato consultoría"""
        request = DocumentProcessingRequest(
            filename="test.md",
            target_format=ProfessionalFormat.CONSULTANCY,
            language="es",
            include_analysis=True
        )
        
        result = await processor.process_document(sample_text_file, request)
        
        assert result.success is True
        assert result.analysis is not None
        assert result.professional_document is not None
        assert result.professional_document.format == ProfessionalFormat.CONSULTANCY
        assert len(result.professional_document.content) > 0
    
    async def test_process_document_technical(self, processor, sample_text_file):
        """Prueba el procesamiento a formato técnico"""
        request = DocumentProcessingRequest(
            filename="test.md",
            target_format=ProfessionalFormat.TECHNICAL,
            language="es",
            include_analysis=True
        )
        
        result = await processor.process_document(sample_text_file, request)
        
        assert result.success is True
        assert result.professional_document.format == ProfessionalFormat.TECHNICAL
    
    async def test_get_supported_formats(self, processor):
        """Prueba la obtención de formatos soportados"""
        formats = await processor.get_supported_formats()
        
        assert "input_formats" in formats
        assert "output_formats" in formats
        assert len(formats["input_formats"]) > 0
        assert len(formats["output_formats"]) > 0
    
    async def test_validate_file(self, processor, sample_text_file):
        """Prueba la validación de archivos"""
        validation = await processor.validate_file(sample_text_file)
        
        assert validation["valid"] is True
        assert "size" in validation
        assert "extension" in validation
        assert validation["extension"] == ".md"
    
    async def test_validate_nonexistent_file(self, processor):
        """Prueba la validación de archivo inexistente"""
        validation = await processor.validate_file("nonexistent.md")
        
        assert validation["valid"] is False
        assert "error" in validation

class TestAIClassifier:
    """Pruebas para el clasificador AI"""
    
    @pytest.fixture
    async def classifier(self):
        """Fixture para el clasificador"""
        classifier = AIClassifier()
        await classifier.initialize()
        return classifier
    
    def test_classification_patterns_loaded(self, classifier):
        """Prueba que los patrones de clasificación estén cargados"""
        assert len(classifier.patterns) > 0
        
        # Verificar que hay patrones para diferentes áreas
        areas = {pattern.area for pattern in classifier.patterns}
        assert DocumentArea.BUSINESS in areas
        assert DocumentArea.TECHNOLOGY in areas
        assert DocumentArea.ACADEMIC in areas
    
    async def test_classify_business_document(self, classifier):
        """Prueba la clasificación de documento de negocios"""
        business_text = """
        Plan de negocio para nueva empresa tecnológica.
        Análisis de mercado, estrategia comercial, proyecciones financieras.
        Recomendaciones para inversión y crecimiento.
        """
        
        analysis = await classifier.classify_document(business_text)
        
        assert analysis is not None
        assert analysis.area in [DocumentArea.BUSINESS, DocumentArea.GENERAL]
        assert analysis.word_count > 0
        assert len(analysis.key_topics) > 0
    
    async def test_classify_technical_document(self, classifier):
        """Prueba la clasificación de documento técnico"""
        technical_text = """
        Especificaciones técnicas del sistema.
        Arquitectura de microservicios, base de datos PostgreSQL.
        API REST, autenticación JWT, documentación técnica.
        """
        
        analysis = await classifier.classify_document(technical_text)
        
        assert analysis is not None
        assert analysis.area in [DocumentArea.TECHNOLOGY, DocumentArea.GENERAL]
        assert analysis.word_count > 0
    
    def test_preprocess_text(self, classifier):
        """Prueba el preprocesamiento de texto"""
        text = "Este es un TEXTO con CARACTERES especiales!!!"
        processed = classifier._preprocess_text(text)
        
        assert processed == "este es un texto con caracteres especiales"
    
    def test_detect_language_spanish(self, classifier):
        """Prueba la detección de idioma español"""
        spanish_text = "Este es un documento en español con palabras comunes como el, la, de, que, y, a, en, un, es, se, no, te, lo, le, da, su, por, son, con, para, al, del, los, las, una, como, pero, sus, más, muy, ya, todo, esta, ser, tiene, también, fue, había, me, si, sin, sobre, este, entre, cuando, muy, sin, hasta, desde, está, mi, porque, qué, sólo, han, yo, hay, vez, puede, todos, así, nos, ni, parte, tiene, él, uno, donde, bien, tiempo, mismo, ese, ahora, cada, e, vida, otro, después, te, otros, aunque, esa, esos, estas, le, ha, me, sus, ya, están"
        
        language = classifier._detect_language(spanish_text)
        assert language == "es"
    
    def test_detect_language_english(self, classifier):
        """Prueba la detección de idioma inglés"""
        english_text = "This is an English document with common words like the, be, to, of, and, a, in, that, have, i, it, for, not, on, with, he, as, you, do, at, this, but, his, by, from, they, we, say, her, she, or, an, will, my, one, all, would, there, their, what, so, up, out, if, about, who, get, which, go, me, when, make, can, like, time, no, just, him, know, take, people, into, year, your, good, some, could, them, see, other, than, then, now, look, only, come, its, over, think, also, back, after, use, two, how, our, work, first, well, way, even, new, want, because, any, these, give, day, most, us"
        
        language = classifier._detect_language(english_text)
        assert language == "en"

class TestProfessionalTransformer:
    """Pruebas para el transformador profesional"""
    
    @pytest.fixture
    async def transformer(self):
        """Fixture para el transformador"""
        transformer = ProfessionalTransformer()
        await transformer.initialize()
        return transformer
    
    def test_templates_loaded(self, transformer):
        """Prueba que las plantillas estén cargadas"""
        assert len(transformer.templates) > 0
        assert ProfessionalFormat.CONSULTANCY in transformer.templates
        assert ProfessionalFormat.TECHNICAL in transformer.templates
        assert ProfessionalFormat.ACADEMIC in transformer.templates
    
    async def test_transform_to_consultancy(self, transformer):
        """Prueba la transformación a formato consultoría"""
        text = "Análisis de mercado digital. Recomendaciones para implementar estrategia digital."
        
        doc = await transformer.transform_to_professional(
            text, None, ProfessionalFormat.CONSULTANCY, "es"
        )
        
        assert doc is not None
        assert doc.format == ProfessionalFormat.CONSULTANCY
        assert doc.language == "es"
        assert len(doc.content) > 0
        assert len(doc.sections) > 0
    
    async def test_transform_to_technical(self, transformer):
        """Prueba la transformación a formato técnico"""
        text = "Sistema de gestión de inventarios. Base de datos PostgreSQL. API REST."
        
        doc = await transformer.transform_to_professional(
            text, None, ProfessionalFormat.TECHNICAL, "es"
        )
        
        assert doc is not None
        assert doc.format == ProfessionalFormat.TECHNICAL
        assert doc.language == "es"
        assert len(doc.content) > 0
    
    def test_get_format_instructions(self, transformer):
        """Prueba la obtención de instrucciones de formato"""
        instructions = transformer._get_format_instructions(
            ProfessionalFormat.CONSULTANCY, "es"
        )
        
        assert len(instructions) > 0
        assert "consultoría" in instructions.lower()
    
    def test_create_document_structure(self, transformer):
        """Prueba la creación de estructura de documento"""
        template = transformer.templates[ProfessionalFormat.CONSULTANCY]
        structure = transformer._create_document_structure(template, None)
        
        assert "header" in structure
        assert "subtitle" in structure
        assert "footer" in structure
        assert "date" in structure

# Pruebas de integración
class TestIntegration:
    """Pruebas de integración del sistema completo"""
    
    @pytest.fixture
    async def full_system(self):
        """Sistema completo para pruebas de integración"""
        processor = DocumentProcessor()
        await processor.initialize()
        return processor
    
    async def test_end_to_end_processing(self, full_system):
        """Prueba completa de extremo a extremo"""
        # Crear archivo de prueba
        content = """
        # Propuesta de Negocio Digital
        
        ## Resumen
        Propuesta para digitalizar procesos empresariales.
        
        ## Beneficios
        - Mayor eficiencia
        - Reducción de costos
        - Mejor experiencia del cliente
        
        ## Implementación
        Fase 1: Análisis y planificación
        Fase 2: Desarrollo e implementación
        Fase 3: Pruebas y optimización
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            file_path = f.name
        
        try:
            request = DocumentProcessingRequest(
                filename="propuesta_negocio.md",
                target_format=ProfessionalFormat.CONSULTANCY,
                language="es",
                include_analysis=True
            )
            
            result = await full_system.process_document(file_path, request)
            
            # Verificar resultado completo
            assert result.success is True
            assert result.analysis is not None
            assert result.professional_document is not None
            assert result.processing_time > 0
            
            # Verificar análisis
            analysis = result.analysis
            assert analysis.filename == "propuesta_negocio.md"
            assert analysis.word_count > 0
            assert len(analysis.key_topics) > 0
            
            # Verificar documento profesional
            doc = result.professional_document
            assert doc.format == ProfessionalFormat.CONSULTANCY
            assert doc.language == "es"
            assert len(doc.content) > 0
            assert len(doc.sections) > 0
            
        finally:
            os.unlink(file_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


