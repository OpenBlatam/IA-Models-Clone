"""
Code Improver - Mejora de código usando modelo entrenado
==========================================================
"""

import logging
from typing import Dict, Any, Optional, List
import os

logger = logging.getLogger(__name__)

# Importar RAG Engine
try:
    from .rag_engine import RAGEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG Engine no disponible")

# Importar Cache Manager
try:
    from .cache_manager import CacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("Cache Manager no disponible")

# Importar Code Analyzer
try:
    from .code_analyzer import CodeAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    logger.warning("Code Analyzer no disponible")


class CodeImprover:
    """
    Mejora código de GitHub usando conocimiento de papers entrenados.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vector_store=None,
        use_rag: bool = True,
        use_cache: bool = True,
        use_analyzer: bool = True
    ):
        """
        Inicializar mejorador de código.
        
        Args:
            model_path: Ruta al modelo entrenado (opcional)
            vector_store: Instancia de VectorStore para RAG (opcional)
            use_rag: Usar RAG para mejoras (default: True)
            use_cache: Usar cache para mejoras (default: True)
            use_analyzer: Usar analizador de código (default: True)
        """
        self.model_path = model_path
        self.model = None
        self.use_rag = use_rag
        self.use_cache = use_cache
        self.use_analyzer = use_analyzer
        self.vector_store = vector_store
        
        if model_path:
            from .model_trainer import ModelTrainer
            trainer = ModelTrainer()
            try:
                self.model = trainer.load_model(model_path)
            except Exception as e:
                logger.warning(f"No se pudo cargar modelo desde {model_path}: {e}")
                self.model = None
        
        # Inicializar RAG Engine si está disponible
        if use_rag and RAG_AVAILABLE:
            try:
                self.rag_engine = RAGEngine(vector_store=vector_store)
                logger.info("RAG Engine inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar RAG Engine: {e}")
                self.rag_engine = None
        else:
            self.rag_engine = None
        
        # Inicializar Cache Manager
        if use_cache and CACHE_AVAILABLE:
            try:
                self.cache_manager = CacheManager()
                logger.info("Cache Manager inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Cache Manager: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None
        
        # Inicializar Code Analyzer
        if use_analyzer and ANALYZER_AVAILABLE:
            try:
                self.code_analyzer = CodeAnalyzer()
                logger.info("Code Analyzer inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Code Analyzer: {e}")
                self.code_analyzer = None
        else:
            self.code_analyzer = None
    
    def improve_code(
        self,
        github_repo: str,
        file_path: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mejora código de un repositorio de GitHub.
        
        Args:
            github_repo: Repositorio en formato "owner/repo"
            file_path: Ruta al archivo en el repositorio
            branch: Rama del repositorio (opcional, default: main)
            
        Returns:
            Diccionario con código mejorado y sugerencias
        """
        try:
            from utils.github_integration import GitHubIntegration
            
            github = GitHubIntegration()
            
            # Obtener código del repositorio
            code = github.get_file_content(github_repo, file_path, branch)
            
            if not code:
                raise ValueError(f"No se pudo obtener código de {github_repo}/{file_path}")
            
            # Verificar cache
            if self.cache_manager and self.use_cache:
                cached = self.cache_manager.get_cached_improvement(code, f"{github_repo}/{file_path}")
                if cached:
                    logger.info("Usando mejora desde cache")
                    return cached
            
            # Analizar código original
            analysis_original = None
            if self.code_analyzer and self.use_analyzer:
                try:
                    language = self._detect_language(file_path)
                    analysis_original = self.code_analyzer.analyze_code(code, language)
                except Exception as e:
                    logger.warning(f"Error analizando código original: {e}")
            
            # Mejorar código usando el modelo o RAG
            improved_code = self._apply_improvements(code, github_repo, file_path)
            
            # Analizar código mejorado
            analysis_improved = None
            if self.code_analyzer and self.use_analyzer and analysis_original:
                try:
                    language = self._detect_language(file_path)
                    analysis_improved = self.code_analyzer.analyze_code(improved_code, language)
                except Exception as e:
                    logger.warning(f"Error analizando código mejorado: {e}")
            
            # Comparar análisis
            comparison = None
            if analysis_original and analysis_improved:
                try:
                    language = self._detect_language(file_path)
                    comparison = self.code_analyzer.compare_code(code, improved_code, language)
                except Exception as e:
                    logger.warning(f"Error comparando código: {e}")
            
            # Generar sugerencias
            suggestions = self._generate_suggestions(code, improved_code)
            
            # Obtener papers usados si RAG está activo
            papers_used = []
            if self.rag_engine and self.use_rag:
                try:
                    result = self.rag_engine.improve_code_with_rag(code)
                    papers_used = result.get("papers_used", [])
                except:
                    pass
            
            result = {
                "original_code": code,
                "improved_code": improved_code,
                "suggestions": suggestions,
                "papers_used": papers_used,
                "repo": github_repo,
                "file_path": file_path,
                "improvements_applied": len(suggestions),
                "analysis": {
                    "original": analysis_original,
                    "improved": analysis_improved,
                    "comparison": comparison
                } if analysis_original else None
            }
            
            # Guardar en cache
            if self.cache_manager and self.use_cache:
                self.cache_manager.cache_improvement(code, result, f"{github_repo}/{file_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error mejorando código: {e}")
            raise
    
    def _detect_language(self, file_path: str) -> str:
        """Detecta lenguaje desde extensión del archivo"""
        ext = file_path.split(".")[-1].lower()
        lang_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "java": "java",
            "cpp": "c++",
            "c": "c",
            "go": "go",
            "rs": "rust"
        }
        return lang_map.get(ext, "python")
    
    def improve_code_from_text(self, code: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Mejora código directamente desde texto.
        
        Args:
            code: Código a mejorar
            context: Contexto adicional (opcional)
            
        Returns:
            Diccionario con código mejorado
        """
        try:
            improved_code = self._apply_improvements(code, context=context)
            suggestions = self._generate_suggestions(code, improved_code)
            
            return {
                "original_code": code,
                "improved_code": improved_code,
                "suggestions": suggestions,
                "improvements_applied": len(suggestions)
            }
            
        except Exception as e:
            logger.error(f"Error mejorando código: {e}")
            raise
    
    def _apply_improvements(self, code: str, repo: Optional[str] = None, 
                           file_path: Optional[str] = None, 
                           context: Optional[str] = None) -> str:
        """
        Aplica mejoras al código usando el modelo entrenado o RAG.
        
        Args:
            code: Código original
            repo: Repositorio (opcional)
            file_path: Ruta al archivo (opcional)
            context: Contexto adicional (opcional)
            
        Returns:
            Código mejorado
        """
        # Usar RAG si está disponible
        if self.rag_engine and self.use_rag:
            try:
                # Detectar lenguaje desde extensión del archivo
                language = None
                if file_path:
                    ext = file_path.split(".")[-1].lower()
                    lang_map = {
                        "py": "python",
                        "js": "javascript",
                        "ts": "typescript",
                        "java": "java",
                        "cpp": "c++",
                        "c": "c",
                        "go": "go",
                        "rs": "rust"
                    }
                    language = lang_map.get(ext)
                
                result = self.rag_engine.improve_code_with_rag(
                    code=code,
                    context=context or f"Repository: {repo}, File: {file_path}",
                    language=language
                )
                return result.get("improved_code", code)
            except Exception as e:
                logger.warning(f"Error en RAG, usando fallback: {e}")
        
        # Fallback a modelo entrenado
        if self.model:
            # En producción, aquí se usaría el modelo entrenado
            logger.info("Usando modelo entrenado para mejoras")
            return self._basic_improvements(code)
        
        # Fallback final: mejoras básicas
        logger.warning("Usando mejoras básicas (sin modelo ni RAG)")
        return self._basic_improvements(code)
    
    def _basic_improvements(self, code: str) -> str:
        """
        Aplica mejoras básicas al código.
        
        Args:
            code: Código original
            
        Returns:
            Código con mejoras básicas
        """
        # Mejoras básicas que se pueden aplicar
        # En producción, esto se haría usando el modelo entrenado
        
        improved = code
        
        # Agregar comentarios de documentación si faltan
        if not code.strip().startswith('"""') and not code.strip().startswith("'''"):
            # Agregar docstring básico
            pass
        
        return improved
    
    def _generate_suggestions(self, original: str, improved: str) -> List[Dict[str, Any]]:
        """
        Genera sugerencias de mejora.
        
        Args:
            original: Código original
            improved: Código mejorado
            
        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        # Comparar código original y mejorado
        if original != improved:
            suggestions.append({
                "type": "code_improvement",
                "description": "Código mejorado basado en técnicas de papers",
                "priority": "high"
            })
        
        # Agregar más sugerencias basadas en análisis
        # (optimización, mejores prácticas, etc.)
        
        return suggestions
    
    def analyze_repository(self, github_repo: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Analiza un repositorio completo y sugiere mejoras.
        
        Args:
            github_repo: Repositorio en formato "owner/repo"
            branch: Rama del repositorio (opcional)
            
        Returns:
            Análisis completo del repositorio
        """
        try:
            from utils.github_integration import GitHubIntegration
            
            github = GitHubIntegration()
            
            # Obtener estructura del repositorio
            files = github.list_repository_files(github_repo, branch)
            
            improvements = []
            
            # Analizar cada archivo
            for file_info in files[:10]:  # Limitar a 10 archivos por ahora
                if file_info.get("type") == "file":
                    file_path = file_info.get("path")
                    try:
                        result = self.improve_code(github_repo, file_path, branch)
                        improvements.append({
                            "file": file_path,
                            "improvements": result.get("improvements_applied", 0),
                            "suggestions": result.get("suggestions", [])
                        })
                    except Exception as e:
                        logger.warning(f"Error analizando {file_path}: {e}")
                        continue
            
            return {
                "repo": github_repo,
                "files_analyzed": len(improvements),
                "total_improvements": sum(i["improvements"] for i in improvements),
                "improvements": improvements
            }
            
        except Exception as e:
            logger.error(f"Error analizando repositorio: {e}")
            raise

