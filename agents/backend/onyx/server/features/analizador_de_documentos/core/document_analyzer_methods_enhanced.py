"""
Métodos Mejorados para DocumentAnalyzer
========================================

Métodos adicionales que se pueden agregar al DocumentAnalyzer
para integrar las características avanzadas.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable


def add_enhanced_methods_to_analyzer(analyzer_class):
    """
    Agregar métodos mejorados a la clase DocumentAnalyzer.
    
    Usage:
        from .document_analyzer import DocumentAnalyzer
        from .document_analyzer_methods_enhanced import add_enhanced_methods_to_analyzer
        
        add_enhanced_methods_to_analyzer(DocumentAnalyzer)
    """
    
    async def compare_documents(
        self,
        doc1_content: str,
        doc2_content: str,
        doc1_id: Optional[str] = None,
        doc2_id: Optional[str] = None
    ):
        """
        Comparar dos documentos (requiere características avanzadas).
        
        Args:
            doc1_content: Contenido del primer documento
            doc2_content: Contenido del segundo documento
            doc1_id: ID del primer documento
            doc2_id: ID del segundo documento
        
        Returns:
            DocumentSimilarity con resultados
        """
        if not hasattr(self, 'comparator') or not self.comparator:
            raise RuntimeError("Características avanzadas no disponibles. Instale document_analyzer_enhanced.")
        
        return await self.comparator.compare_documents(
            doc1_content, doc2_content, doc1_id, doc2_id
        )
    
    async def process_batch(
        self,
        documents: List[Dict[str, Any]],
        tasks: Optional[List] = None,
        max_workers: int = 10,
        on_progress: Optional[Callable[[int, int], None]] = None
    ):
        """
        Procesar múltiples documentos en batch (requiere características avanzadas).
        
        Args:
            documents: Lista de documentos a procesar
            tasks: Tareas de análisis
            max_workers: Número máximo de workers paralelos
            on_progress: Callback de progreso
        
        Returns:
            BatchAnalysisResult con todos los resultados
        """
        if not hasattr(self, 'batch_processor') or not self.batch_processor:
            from .document_analyzer_enhanced import BatchDocumentProcessor
            self.batch_processor = BatchDocumentProcessor(self, max_workers=max_workers)
        
        if max_workers != self.batch_processor.max_workers:
            from .document_analyzer_enhanced import BatchDocumentProcessor
            self.batch_processor = BatchDocumentProcessor(self, max_workers=max_workers)
        
        return await self.batch_processor.process_batch(
            documents, tasks=tasks, on_progress=on_progress
        )
    
    async def extract_structured_data(
        self,
        content: str,
        schema: Dict[str, Any]
    ):
        """
        Extraer información estructurada (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
            schema: Schema de extracción
        
        Returns:
            Diccionario con datos extraídos
        """
        if not hasattr(self, 'info_extractor') or not self.info_extractor:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.info_extractor.extract_structured_data(content, schema)
    
    async def analyze_writing_style(
        self,
        content: str
    ):
        """
        Analizar estilo de escritura (requiere características avanzadas).
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis de estilo
        """
        if not hasattr(self, 'language_analyzer') or not self.language_analyzer:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.language_analyzer.analyze_writing_style(content)
    
    async def find_similar_documents(
        self,
        target_doc: str,
        document_corpus: List[Tuple[str, str]],
        threshold: float = 0.7,
        top_k: int = 10
    ):
        """
        Encontrar documentos similares (requiere características avanzadas).
        
        Args:
            target_doc: Documento objetivo
            document_corpus: Lista de (doc_id, content)
            threshold: Umbral de similitud
            top_k: Número de resultados
        
        Returns:
            Lista de DocumentSimilarity
        """
        if not hasattr(self, 'comparator') or not self.comparator:
            raise RuntimeError("Características avanzadas no disponibles.")
        
        return await self.comparator.find_similar_documents(
            target_doc, document_corpus, threshold, top_k
        )
    
    # Agregar métodos a la clase
    analyzer_class.compare_documents = compare_documents
    analyzer_class.process_batch = process_batch
    analyzer_class.extract_structured_data = extract_structured_data
    analyzer_class.analyze_writing_style = analyze_writing_style
    analyzer_class.find_similar_documents = find_similar_documents
    
    return analyzer_class
















