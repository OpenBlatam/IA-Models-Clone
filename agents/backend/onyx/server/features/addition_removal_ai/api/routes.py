"""
API Routes - Endpoints REST para el sistema
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import datetime

from ..core.editor import ContentEditor

router = APIRouter()

# Instancia global del editor (puede mejorarse con dependency injection)
editor = ContentEditor()


class AddRequest(BaseModel):
    """Request model para agregar contenido"""
    content: str
    addition: str
    position: str = "end"
    context: Optional[Dict[str, Any]] = None


class RemoveRequest(BaseModel):
    """Request model para eliminar contenido"""
    content: str
    pattern: Optional[str] = None
    selector: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class AddResponse(BaseModel):
    """Response model para operación de adición"""
    success: bool
    content: str
    validation: Dict[str, Any]
    change_id: Optional[str] = None
    error: Optional[str] = None


class RemoveResponse(BaseModel):
    """Response model para operación de eliminación"""
    success: bool
    content: str
    validation: Dict[str, Any]
    change_id: Optional[str] = None
    error: Optional[str] = None


@router.post("/add", response_model=AddResponse)
async def add_content(request: AddRequest):
    """
    Agregar contenido al texto original.

    Args:
        request: Request con contenido y adición

    Returns:
        Response con el resultado
    """
    try:
        result = await editor.add(
            content=request.content,
            addition=request.addition,
            position=request.position,
            context=request.context
        )
        return AddResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/remove", response_model=RemoveResponse)
async def remove_content(request: RemoveRequest):
    """
    Eliminar contenido del texto original.

    Args:
        request: Request con contenido y patrón a eliminar

    Returns:
        Response con el resultado
    """
    try:
        result = await editor.remove(
            content=request.content,
            pattern=request.pattern,
            selector=request.selector,
            context=request.context
        )
        return RemoveResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_history(limit: int = 10):
    """
    Obtener historial de cambios.

    Args:
        limit: Número máximo de cambios a retornar

    Returns:
        Lista de cambios recientes
    """
    try:
        history = editor.get_history(limit=limit)
        return {"success": True, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BatchAddRequest(BaseModel):
    """Request model para operación batch de adición"""
    content: str
    additions: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]] = None


class BatchRemoveRequest(BaseModel):
    """Request model para operación batch de eliminación"""
    content: str
    patterns: List[str]
    context: Optional[Dict[str, Any]] = None


@router.post("/batch/add")
async def batch_add_content(request: BatchAddRequest):
    """
    Agregar múltiples elementos en una operación batch.

    Args:
        request: Request con contenido y lista de adiciones

    Returns:
        Response con el resultado
    """
    try:
        result = await editor.batch_add(
            content=request.content,
            additions=request.additions,
            context=request.context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/remove")
async def batch_remove_content(request: BatchRemoveRequest):
    """
    Eliminar múltiples elementos en una operación batch.

    Args:
        request: Request con contenido y lista de patrones

    Returns:
        Response con el resultado
    """
    try:
        result = await editor.batch_remove(
            content=request.content,
            patterns=request.patterns,
            context=request.context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_content(content: str, context: Optional[Dict[str, Any]] = None):
    """
    Analizar contenido sin modificarlo.

    Args:
        content: Contenido a analizar
        context: Contexto adicional

    Returns:
        Análisis del contenido
    """
    try:
        analysis = await editor.analyzer.analyze(content, context)
        return {"success": True, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DiffRequest(BaseModel):
    """Request model para comparación de contenido"""
    original: str
    modified: str
    format: Optional[str] = "json"


@router.post("/diff")
async def compute_diff(request: DiffRequest):
    """
    Calcular diferencias entre dos versiones de contenido.

    Args:
        request: Request con contenido original y modificado

    Returns:
        Diferencias calculadas
    """
    try:
        diff_result = editor.diff.compute_diff(request.original, request.modified)
        
        if request.format == "html":
            highlighted = editor.diff.highlight_changes(request.original, request.modified, "html")
            diff_result["highlighted_html"] = highlighted
        elif request.format == "markdown":
            highlighted = editor.diff.highlight_changes(request.original, request.modified, "markdown")
            diff_result["highlighted_markdown"] = highlighted
        
        return {"success": True, "diff": diff_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/undo")
async def undo_operation(current_content: str):
    """
    Deshacer la última operación.

    Args:
        current_content: Contenido actual

    Returns:
        Estado anterior
    """
    try:
        previous_state = editor.undo_redo.undo(current_content)
        if not previous_state:
            return {"success": False, "message": "No hay operaciones para deshacer"}
        return {"success": True, "state": previous_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/redo")
async def redo_operation(current_content: str):
    """
    Rehacer la última operación deshecha.

    Args:
        current_content: Contenido actual

    Returns:
        Estado siguiente
    """
    try:
        next_state = editor.undo_redo.redo(current_content)
        if not next_state:
            return {"success": False, "message": "No hay operaciones para rehacer"}
        return {"success": True, "state": next_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """
    Obtener métricas del sistema.

    Returns:
        Métricas y estadísticas
    """
    try:
        stats = editor.metrics.get_stats()
        return {"success": True, "metrics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{operation_type}")
async def get_operation_metrics(operation_type: str):
    """
    Obtener métricas de un tipo de operación específico.

    Args:
        operation_type: Tipo de operación

    Returns:
        Métricas del tipo de operación
    """
    try:
        stats = editor.metrics.get_operation_stats(operation_type)
        return {"success": True, "metrics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plugins")
async def get_plugins():
    """
    Obtener lista de plugins registrados.

    Returns:
        Lista de plugins
    """
    try:
        if editor.plugin_manager:
            plugins = editor.plugin_manager.get_plugins()
            return {"success": True, "plugins": plugins}
        return {"success": True, "plugins": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check del sistema.

    Returns:
        Estado del sistema
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "editor": "operational",
            "ai_engine": "operational" if editor.ai_engine.openai_client or editor.ai_engine.langchain_llm else "limited",
            "cache": "operational",
            "metrics": "operational",
            "database": "operational" if editor.database else "unavailable",
            "versioning": "operational" if editor.versioning else "unavailable",
            "backups": "operational" if editor.backup_manager else "unavailable"
        }
    }


@router.get("/versions/{content_id}")
async def get_content_versions(content_id: str, limit: int = 10):
    """
    Obtener versiones de un contenido.

    Args:
        content_id: ID del contenido
        limit: Número máximo de versiones

    Returns:
        Lista de versiones
    """
    try:
        if not editor.versioning:
            raise HTTPException(status_code=503, detail="Versionado no disponible")
        
        versions = editor.versioning.get_versions(content_id, limit)
        return {"success": True, "versions": versions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/versions/{content_id}/restore")
async def restore_version(content_id: str, version_number: int):
    """
    Restaurar una versión anterior.

    Args:
        content_id: ID del contenido
        version_number: Número de versión a restaurar

    Returns:
        Versión restaurada
    """
    try:
        if not editor.versioning:
            raise HTTPException(status_code=503, detail="Versionado no disponible")
        
        restored = editor.versioning.rollback_to_version(content_id, version_number)
        if not restored:
            raise HTTPException(status_code=404, detail="Versión no encontrada")
        
        return {"success": True, "version": restored}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backups")
async def list_backups():
    """
    Listar backups disponibles.

    Returns:
        Lista de backups
    """
    try:
        if not editor.backup_manager:
            raise HTTPException(status_code=503, detail="Backups no disponibles")
        
        backups = editor.backup_manager.list_backups()
        return {"success": True, "backups": backups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backups/create")
async def create_backup():
    """
    Crear un backup manual.

    Returns:
        Información del backup creado
    """
    try:
        if not editor.backup_manager:
            raise HTTPException(status_code=503, detail="Backups no disponibles")
        
        # Recopilar datos para backup
        backup_data = {
            "history": editor.get_history(limit=1000),
            "metrics": editor.metrics.get_stats(),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        backup_path = editor.backup_manager.create_backup(backup_data, "manual")
        return {"success": True, "backup_path": backup_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/login")
async def login(username: str, password: str):
    """
    Autenticar usuario.

    Args:
        username: Nombre de usuario
        password: Contraseña

    Returns:
        Token de autenticación
    """
    try:
        from ..core.auth import AuthManager
        auth_manager = AuthManager()
        
        result = auth_manager.authenticate(username, password)
        if not result:
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/register")
async def register(username: str, password: str, email: Optional[str] = None):
    """
    Registrar nuevo usuario.

    Args:
        username: Nombre de usuario
        password: Contraseña
        email: Email (opcional)

    Returns:
        Información del usuario creado
    """
    try:
        from ..core.auth import AuthManager
        auth_manager = AuthManager()
        
        user = auth_manager.create_user(username, password, email)
        return {"success": True, "user": user}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notifications")
async def get_notifications(
    notification_type: Optional[str] = None,
    unread_only: bool = False,
    limit: int = 50
):
    """
    Obtener notificaciones.

    Args:
        notification_type: Filtrar por tipo
        unread_only: Solo no leídas
        limit: Límite de resultados

    Returns:
        Lista de notificaciones
    """
    try:
        if not editor.notifications:
            raise HTTPException(status_code=503, detail="Notificaciones no disponibles")
        
        from ..core.notifications import NotificationType
        ntype = NotificationType[notification_type.upper()] if notification_type else None
        
        notifications = editor.notifications.get_notifications(ntype, unread_only, limit)
        return {"success": True, "notifications": notifications}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """
    Marcar notificación como leída.

    Args:
        notification_id: ID de la notificación

    Returns:
        Resultado
    """
    try:
        if not editor.notifications:
            raise HTTPException(status_code=503, detail="Notificaciones no disponibles")
        
        editor.notifications.mark_as_read(notification_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integrations/translate")
async def translate_content(content: str, target_language: str = "en"):
    """
    Traducir contenido usando servicio externo.

    Args:
        content: Contenido a traducir
        target_language: Idioma objetivo

    Returns:
        Resultado de la traducción
    """
    try:
        if not editor.integrations:
            raise HTTPException(status_code=503, detail="Integraciones no disponibles")
        
        result = await editor.integrations.translate(content, target_language)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integrations/spellcheck")
async def spell_check_content(content: str):
    """
    Corregir ortografía del contenido.

    Args:
        content: Contenido a corregir

    Returns:
        Resultado de la corrección
    """
    try:
        if not editor.integrations:
            raise HTTPException(status_code=503, detail="Integraciones no disponibles")
        
        result = await editor.integrations.spell_check(content)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/integrations/services")
async def get_available_services():
    """
    Obtener servicios de integración disponibles.

    Returns:
        Lista de servicios
    """
    try:
        if not editor.integrations:
            return {"success": True, "services": []}
        
        services = editor.integrations.get_available_services()
        return {"success": True, "services": services}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduler/tasks")
async def get_scheduled_tasks():
    """
    Obtener tareas programadas.

    Returns:
        Lista de tareas
    """
    try:
        if not editor.scheduler:
            raise HTTPException(status_code=503, detail="Scheduler no disponible")
        
        tasks = editor.scheduler.get_tasks()
        return {"success": True, "tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/history")
async def export_history(format: str = "json"):
    """
    Exportar historial.

    Args:
        format: Formato (json, csv)

    Returns:
        Datos exportados
    """
    try:
        if not editor.export_manager:
            raise HTTPException(status_code=503, detail="Export no disponible")
        
        history = editor.get_history(limit=1000)
        exported = editor.export_manager.export_history(history, format)
        return {"success": True, "data": exported, "format": format}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/metrics")
async def export_metrics(format: str = "json"):
    """
    Exportar métricas.

    Args:
        format: Formato (json)

    Returns:
        Métricas exportadas
    """
    try:
        if not editor.export_manager:
            raise HTTPException(status_code=503, detail="Export no disponible")
        
        metrics = editor.metrics.get_stats()
        exported = editor.export_manager.export_metrics(metrics, format)
        return {"success": True, "data": exported, "format": format}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_content(content: str, query: str, case_sensitive: bool = False, whole_words: bool = False):
    """
    Buscar en contenido.

    Args:
        content: Contenido donde buscar
        query: Consulta de búsqueda
        case_sensitive: Si es sensible a mayúsculas
        whole_words: Si busca palabras completas

    Returns:
        Resultados de búsqueda
    """
    try:
        if not editor.search_engine:
            raise HTTPException(status_code=503, detail="Búsqueda no disponible")
        
        matches = editor.search_engine.search_content(content, query, case_sensitive, whole_words)
        return {"success": True, "matches": matches, "count": len(matches)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/highlight")
async def search_with_highlight(content: str, query: str, highlight_tag: str = "mark"):
    """
    Buscar y resaltar coincidencias.

    Args:
        content: Contenido
        query: Consulta
        highlight_tag: Tag HTML para resaltar

    Returns:
        Contenido con resaltado
    """
    try:
        if not editor.search_engine:
            raise HTTPException(status_code=503, detail="Búsqueda no disponible")
        
        result = editor.search_engine.search_with_highlight(content, query, highlight_tag)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_templates():
    """
    Listar plantillas disponibles.

    Returns:
        Lista de plantillas
    """
    try:
        if not editor.templates:
            raise HTTPException(status_code=503, detail="Plantillas no disponibles")
        
        templates = editor.templates.list_templates()
        return {"success": True, "templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/{template_name}/render")
async def render_template(template_name: str, context: Dict[str, Any]):
    """
    Renderizar una plantilla.

    Args:
        template_name: Nombre de la plantilla
        context: Contexto con valores

    Returns:
        Contenido renderizado
    """
    try:
        if not editor.templates:
            raise HTTPException(status_code=503, detail="Plantillas no disponibles")
        
        rendered = editor.templates.render_template(template_name, context)
        if not rendered:
            raise HTTPException(status_code=404, detail="Plantilla no encontrada")
        
        return {"success": True, "content": rendered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates")
async def create_template(name: str, content: str):
    """
    Crear una nueva plantilla.

    Args:
        name: Nombre de la plantilla
        content: Contenido de la plantilla

    Returns:
        Plantilla creada
    """
    try:
        if not editor.templates:
            raise HTTPException(status_code=503, detail="Plantillas no disponibles")
        
        template = editor.templates.register_template(name, content)
        return {
            "success": True,
            "template": {
                "name": template.name,
                "variables": template.variables
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_events(event_type: Optional[str] = None, limit: int = 50):
    """
    Obtener historial de eventos.

    Args:
        event_type: Filtrar por tipo
        limit: Límite de resultados

    Returns:
        Lista de eventos
    """
    try:
        if not editor.event_bus:
            raise HTTPException(status_code=503, detail="Eventos no disponibles")
        
        from ..core.events import EventType
        etype = EventType[event_type.upper()] if event_type else None
        
        events = editor.event_bus.get_event_history(etype, limit)
        return {"success": True, "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transform")
async def transform_content(content: str, transformations: List[Dict[str, Any]]):
    """
    Aplicar transformaciones al contenido.

    Args:
        content: Contenido a transformar
        transformations: Lista de transformaciones a aplicar

    Returns:
        Contenido transformado
    """
    try:
        if not editor.transformation_pipeline:
            raise HTTPException(status_code=503, detail="Transformaciones no disponibles")
        
        # Limpiar pipeline
        editor.transformation_pipeline.clear()
        
        # Agregar transformaciones
        for trans in transformations:
            trans_type = trans.get("type")
            kwargs = trans.get("params", {})
            
            if trans_type == "case":
                editor.transformation_pipeline.add_transformation(
                    editor.CaseTransformation(), **kwargs
                )
            elif trans_type == "whitespace":
                editor.transformation_pipeline.add_transformation(
                    editor.WhitespaceTransformation(), **kwargs
                )
            elif trans_type == "lines":
                editor.transformation_pipeline.add_transformation(
                    editor.LineTransformation(), **kwargs
                )
        
        # Aplicar transformaciones
        transformed = editor.transformation_pipeline.apply(content)
        
        return {"success": True, "content": transformed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/readability")
async def analyze_readability(content: str):
    """
    Analizar legibilidad del contenido.

    Args:
        content: Contenido a analizar

    Returns:
        Análisis de legibilidad
    """
    try:
        if not editor.advanced_analyzer:
            raise HTTPException(status_code=503, detail="Analizador avanzado no disponible")
        
        analysis = editor.advanced_analyzer.analyze_readability(content)
        return {"success": True, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/sentiment")
async def analyze_sentiment(content: str):
    """
    Analizar sentimiento del contenido.

    Args:
        content: Contenido a analizar

    Returns:
        Análisis de sentimiento
    """
    try:
        if not editor.advanced_analyzer:
            raise HTTPException(status_code=503, detail="Analizador avanzado no disponible")
        
        analysis = editor.advanced_analyzer.analyze_sentiment_basic(content)
        return {"success": True, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/keywords")
async def extract_keywords(content: str, top_n: int = 10):
    """
    Extraer palabras clave del contenido.

    Args:
        content: Contenido
        top_n: Número de palabras clave

    Returns:
        Lista de palabras clave
    """
    try:
        if not editor.advanced_analyzer:
            raise HTTPException(status_code=503, detail="Analizador avanzado no disponible")
        
        keywords = editor.advanced_analyzer.extract_keywords(content, top_n)
        return {"success": True, "keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/structure")
async def analyze_structure(content: str):
    """
    Analizar estructura del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de estructura
    """
    try:
        if not editor.advanced_analyzer:
            raise HTTPException(status_code=503, detail="Analizador avanzado no disponible")
        
        analysis = editor.advanced_analyzer.analyze_structure(content)
        return {"success": True, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/schema")
async def validate_with_schema(
    content: str,
    schema_name: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None
):
    """
    Validar contenido contra un esquema.

    Args:
        content: Contenido a validar
        schema_name: Nombre del esquema registrado
        schema: Esquema directo

    Returns:
        Resultado de la validación
    """
    try:
        if not editor.schema_validator:
            raise HTTPException(status_code=503, detail="Validador de esquemas no disponible")
        
        from ..core.schema_validator import ContentSchema
        
        validation_schema = None
        if schema:
            validation_schema = ContentSchema(**schema)
        
        result = editor.schema_validator.validate_content(
            content,
            schema_name=schema_name,
            schema=validation_schema
        )
        
        return {"success": True, "validation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/status")
async def get_queue_status():
    """
    Obtener estado de la cola de tareas.

    Returns:
        Estado de la cola
    """
    try:
        if not editor.task_queue:
            raise HTTPException(status_code=503, detail="Cola de tareas no disponible")
        
        return {
            "success": True,
            "queue_size": editor.task_queue.get_queue_size(),
            "running": editor.task_queue.running,
            "max_workers": editor.task_queue.max_workers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collaboration/sessions")
async def create_collaboration_session(content_id: str, user_id: str):
    """
    Crear sesión de colaboración.

    Args:
        content_id: ID del contenido
        user_id: ID del usuario

    Returns:
        Sesión creada
    """
    try:
        if not editor.collaboration:
            raise HTTPException(status_code=503, detail="Colaboración no disponible")
        
        session = editor.collaboration.create_session(content_id, user_id)
        return {
            "success": True,
            "session": {
                "id": session.id,
                "content_id": session.content_id,
                "participants": session.participants
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collaboration/comments")
async def add_comment(
    content_id: str,
    user_id: str,
    text: str,
    position: Optional[int] = None,
    parent_comment_id: Optional[str] = None
):
    """
    Agregar comentario.

    Args:
        content_id: ID del contenido
        user_id: ID del usuario
        text: Texto del comentario
        position: Posición (opcional)
        parent_comment_id: ID del comentario padre (opcional)

    Returns:
        Comentario creado
    """
    try:
        if not editor.collaboration:
            raise HTTPException(status_code=503, detail="Colaboración no disponible")
        
        comment = editor.collaboration.add_comment(
            content_id, user_id, text, position, parent_comment_id
        )
        return {"success": True, "comment": {"id": comment.id, "text": comment.text}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collaboration/comments/{content_id}")
async def get_comments(content_id: str):
    """
    Obtener comentarios de un contenido.

    Args:
        content_id: ID del contenido

    Returns:
        Lista de comentarios
    """
    try:
        if not editor.collaboration:
            raise HTTPException(status_code=503, detail="Colaboración no disponible")
        
        comments = editor.collaboration.get_comments(content_id)
        return {"success": True, "comments": comments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tags")
async def create_tag(name: str, color: Optional[str] = None, description: Optional[str] = None):
    """
    Crear etiqueta.

    Args:
        name: Nombre de la etiqueta
        color: Color (opcional)
        description: Descripción (opcional)

    Returns:
        Etiqueta creada
    """
    try:
        if not editor.tags:
            raise HTTPException(status_code=503, detail="Etiquetas no disponibles")
        
        tag = editor.tags.create_tag(name, color, description)
        return {"success": True, "tag": {"id": tag.id, "name": tag.name}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tags/{content_id}/tag")
async def tag_content(content_id: str, tag_id: str):
    """
    Etiquetar contenido.

    Args:
        content_id: ID del contenido
        tag_id: ID de la etiqueta

    Returns:
        Resultado
    """
    try:
        if not editor.tags:
            raise HTTPException(status_code=503, detail="Etiquetas no disponibles")
        
        editor.tags.tag_content(content_id, tag_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tags/{content_id}")
async def get_content_tags(content_id: str):
    """
    Obtener etiquetas de un contenido.

    Args:
        content_id: ID del contenido

    Returns:
        Lista de etiquetas
    """
    try:
        if not editor.tags:
            raise HTTPException(status_code=503, detail="Etiquetas no disponibles")
        
        tags = editor.tags.get_content_tags(content_id)
        return {"success": True, "tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tags/popular")
async def get_popular_tags(limit: int = 10):
    """
    Obtener etiquetas populares.

    Args:
        limit: Número de etiquetas

    Returns:
        Lista de etiquetas
    """
    try:
        if not editor.tags:
            raise HTTPException(status_code=503, detail="Etiquetas no disponibles")
        
        tags = editor.tags.get_popular_tags(limit)
        return {"success": True, "tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/{content_id}/transition")
async def transition_workflow(content_id: str, to_state: str, user_id: Optional[str] = None):
    """
    Transicionar workflow.

    Args:
        content_id: ID del contenido
        to_state: Estado destino
        user_id: ID del usuario

    Returns:
        Resultado
    """
    try:
        if not editor.workflow:
            raise HTTPException(status_code=503, detail="Workflow no disponible")
        
        from ..core.workflow import WorkflowState
        state = WorkflowState[to_state.upper()]
        
        success = editor.workflow.transition(content_id, state, user_id)
        if not success:
            raise HTTPException(status_code=400, detail="Transición inválida")
        
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/{content_id}")
async def get_workflow(content_id: str):
    """
    Obtener workflow de un contenido.

    Args:
        content_id: ID del contenido

    Returns:
        Información del workflow
    """
    try:
        if not editor.workflow:
            raise HTTPException(status_code=503, detail="Workflow no disponible")
        
        workflow = editor.workflow.get_workflow(content_id)
        if not workflow:
            return {"success": False, "message": "Workflow no encontrado"}
        
        available = editor.workflow.get_available_transitions(content_id)
        
        return {
            "success": True,
            "workflow": {
                "id": workflow.id,
                "current_state": workflow.current_state.value,
                "available_transitions": [s.value for s in available],
                "history": workflow.history
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/summary")
async def get_statistics_summary():
    """
    Obtener resumen de estadísticas.

    Returns:
        Resumen de estadísticas
    """
    try:
        if not editor.statistics:
            raise HTTPException(status_code=503, detail="Estadísticas no disponibles")
        
        summary = editor.statistics.get_summary()
        return {"success": True, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/daily")
async def get_daily_statistics(days: int = 7):
    """
    Obtener estadísticas diarias.

    Args:
        days: Número de días

    Returns:
        Estadísticas diarias
    """
    try:
        if not editor.statistics:
            raise HTTPException(status_code=503, detail="Estadísticas no disponibles")
        
        stats = editor.statistics.get_daily_statistics(days)
        return {"success": True, "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/users/top")
async def get_top_users(limit: int = 10):
    """
    Obtener usuarios más activos.

    Args:
        limit: Número de usuarios

    Returns:
        Lista de usuarios
    """
    try:
        if not editor.statistics:
            raise HTTPException(status_code=503, detail="Estadísticas no disponibles")
        
        users = editor.statistics.get_top_users(limit)
        return {"success": True, "users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/operations")
async def generate_operations_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    Generar reporte de operaciones.

    Args:
        start_date: Fecha de inicio (ISO format)
        end_date: Fecha de fin (ISO format)
        user_id: ID del usuario

    Returns:
        Reporte generado
    """
    try:
        if not editor.report_generator:
            raise HTTPException(status_code=503, detail="Reportes no disponibles")
        
        from datetime import datetime
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        report = editor.report_generator.generate_operations_report(start, end, user_id)
        return {"success": True, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/performance")
async def generate_performance_report(days: int = 7):
    """
    Generar reporte de rendimiento.

    Args:
        days: Número de días

    Returns:
        Reporte de rendimiento
    """
    try:
        if not editor.report_generator:
            raise HTTPException(status_code=503, detail="Reportes no disponibles")
        
        report = editor.report_generator.generate_performance_report(days)
        return {"success": True, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/quality")
async def generate_quality_report(content_id: Optional[str] = None):
    """
    Generar reporte de calidad.

    Args:
        content_id: ID del contenido

    Returns:
        Reporte de calidad
    """
    try:
        if not editor.report_generator:
            raise HTTPException(status_code=503, detail="Reportes no disponibles")
        
        report = editor.report_generator.generate_quality_report(content_id)
        return {"success": True, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    unresolved_only: bool = False,
    limit: int = 50
):
    """
    Obtener alertas.

    Args:
        alert_type: Filtrar por tipo
        severity: Filtrar por severidad
        unresolved_only: Solo no resueltas
        limit: Límite de resultados

    Returns:
        Lista de alertas
    """
    try:
        if not editor.alerts:
            raise HTTPException(status_code=503, detail="Alertas no disponibles")
        
        from ..core.alerts import AlertType, AlertSeverity
        atype = AlertType[alert_type.upper()] if alert_type else None
        sev = AlertSeverity[severity.upper()] if severity else None
        
        alerts = editor.alerts.get_alerts(atype, sev, unresolved_only, limit)
        return {"success": True, "alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """
    Reconocer una alerta.

    Args:
        alert_id: ID de la alerta

    Returns:
        Resultado
    """
    try:
        if not editor.alerts:
            raise HTTPException(status_code=503, detail="Alertas no disponibles")
        
        editor.alerts.acknowledge_alert(alert_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """
    Resolver una alerta.

    Args:
        alert_id: ID de la alerta

    Returns:
        Resultado
    """
    try:
        if not editor.alerts:
            raise HTTPException(status_code=503, detail="Alertas no disponibles")
        
        editor.alerts.resolve_alert(alert_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_content(content: str):
    """
    Optimizar contenido automáticamente.

    Args:
        content: Contenido a optimizar

    Returns:
        Contenido optimizado
    """
    try:
        if not editor.optimization_engine:
            raise HTTPException(status_code=503, detail="Motor de optimización no disponible")
        
        result = await editor.optimization_engine.optimize_content(content)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/stats")
async def get_optimization_stats():
    """
    Obtener estadísticas de optimizaciones.

    Returns:
        Estadísticas de optimizaciones
    """
    try:
        if not editor.optimization_engine:
            raise HTTPException(status_code=503, detail="Motor de optimización no disponible")
        
        stats = editor.optimization_engine.get_optimization_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/position")
async def recommend_position(content: str, addition: str, user_id: Optional[str] = None):
    """
    Recomendar posición para agregar contenido.

    Args:
        content: Contenido original
        addition: Contenido a agregar
        user_id: ID del usuario

    Returns:
        Recomendación de posición
    """
    try:
        if not editor.recommendations:
            raise HTTPException(status_code=503, detail="Recomendaciones no disponibles")
        
        recommendation = editor.recommendations.recommend_position(content, addition, user_id)
        return {"success": True, "recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{user_id}")
async def get_user_recommendations(user_id: str, limit: int = 5):
    """
    Obtener recomendaciones para un usuario.

    Args:
        user_id: ID del usuario
        limit: Número de recomendaciones

    Returns:
        Lista de recomendaciones
    """
    try:
        if not editor.recommendations:
            raise HTTPException(status_code=503, detail="Recomendaciones no disponibles")
        
        recommendations = editor.recommendations.get_user_recommendations(user_id, limit)
        return {"success": True, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clustering/assign")
async def assign_to_cluster(content_id: str, content: str, similarity_threshold: float = 0.3):
    """
    Asignar contenido a un cluster.

    Args:
        content_id: ID del contenido
        content: Contenido
        similarity_threshold: Umbral de similitud

    Returns:
        ID del cluster asignado
    """
    try:
        if not editor.clustering:
            raise HTTPException(status_code=503, detail="Clustering no disponible")
        
        cluster_id = editor.clustering.assign_to_cluster(content_id, content, similarity_threshold)
        return {"success": True, "cluster_id": cluster_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clustering/clusters")
async def get_all_clusters():
    """
    Obtener todos los clusters.

    Returns:
        Lista de clusters
    """
    try:
        if not editor.clustering:
            raise HTTPException(status_code=503, detail="Clustering no disponible")
        
        clusters = editor.clustering.get_all_clusters()
        return {"success": True, "clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clustering/{content_id}")
async def get_content_cluster(content_id: str):
    """
    Obtener cluster de un contenido.

    Args:
        content_id: ID del contenido

    Returns:
        Información del cluster
    """
    try:
        if not editor.clustering:
            raise HTTPException(status_code=503, detail="Clustering no disponible")
        
        cluster = editor.clustering.get_content_cluster(content_id)
        if not cluster:
            return {"success": False, "message": "Cluster no encontrado"}
        
        return {"success": True, "cluster": cluster}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/trend")
async def predict_trend(metric: str, days_ahead: int = 7):
    """
    Predecir tendencia de una métrica.

    Args:
        metric: Nombre de la métrica
        days_ahead: Días a predecir

    Returns:
        Predicción de tendencia
    """
    try:
        if not editor.predictive:
            raise HTTPException(status_code=503, detail="Análisis predictivo no disponible")
        
        prediction = editor.predictive.predict_trend(metric, days_ahead)
        return {"success": True, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/anomalies")
async def detect_anomalies(metric: str, threshold_std: float = 2.0):
    """
    Detectar anomalías en una métrica.

    Args:
        metric: Nombre de la métrica
        threshold_std: Umbral en desviaciones estándar

    Returns:
        Lista de anomalías
    """
    try:
        if not editor.predictive:
            raise HTTPException(status_code=503, detail="Análisis predictivo no disponible")
        
        anomalies = editor.predictive.detect_anomalies(metric, threshold_std)
        return {"success": True, "anomalies": anomalies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/forecast")
async def forecast_demand(operation_type: str, days_ahead: int = 30):
    """
    Pronosticar demanda.

    Args:
        operation_type: Tipo de operación
        days_ahead: Días a pronosticar

    Returns:
        Pronóstico de demanda
    """
    try:
        if not editor.predictive:
            raise HTTPException(status_code=503, detail="Análisis predictivo no disponible")
        
        forecast = editor.predictive.forecast_demand(operation_type, days_ahead)
        return {"success": True, "forecast": forecast}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/learn")
async def learn_from_example(
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    success: bool
):
    """
    Aprender de un ejemplo.

    Args:
        input_data: Datos de entrada
        output_data: Datos de salida
        success: Si fue exitoso

    Returns:
        Resultado
    """
    try:
        if not editor.ml_learning:
            raise HTTPException(status_code=503, detail="Aprendizaje ML no disponible")
        
        editor.ml_learning.record_training_example(input_data, output_data, success)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/predict")
async def predict_operation_success(
    content: str,
    operation: str,
    **kwargs
):
    """
    Predecir éxito de una operación.

    Args:
        content: Contenido
        operation: Tipo de operación
        **kwargs: Argumentos adicionales

    Returns:
        Predicción
    """
    try:
        if not editor.ml_learning:
            raise HTTPException(status_code=503, detail="Aprendizaje ML no disponible")
        
        prediction = editor.ml_learning.predict_success(content, operation, **kwargs)
        return {"success": True, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/stats")
async def get_ml_stats():
    """
    Obtener estadísticas de aprendizaje ML.

    Returns:
        Estadísticas
    """
    try:
        if not editor.ml_learning:
            raise HTTPException(status_code=503, detail="Aprendizaje ML no disponible")
        
        stats = editor.ml_learning.get_learning_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/queue")
async def queue_sync_operation(
    content_id: str,
    operation_type: str,
    data: Dict[str, Any]
):
    """
    Encolar operación de sincronización.

    Args:
        content_id: ID del contenido
        operation_type: Tipo de operación
        data: Datos de la operación

    Returns:
        ID de la operación
    """
    try:
        if not editor.sync_manager:
            raise HTTPException(status_code=503, detail="Sincronización no disponible")
        
        operation_id = editor.sync_manager.queue_sync_operation(content_id, operation_type, data)
        return {"success": True, "operation_id": operation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/execute")
async def execute_sync(target_system: Optional[str] = None):
    """
    Ejecutar sincronización.

    Args:
        target_system: Sistema destino

    Returns:
        Resultado de la sincronización
    """
    try:
        if not editor.sync_manager:
            raise HTTPException(status_code=503, detail="Sincronización no disponible")
        
        result = await editor.sync_manager.sync_operations(target_system)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/detect-conflicts")
async def detect_conflicts(local_content: str, remote_content: str):
    """
    Detectar conflictos entre versiones.

    Args:
        local_content: Contenido local
        remote_content: Contenido remoto

    Returns:
        Lista de conflictos
    """
    try:
        if not editor.sync_manager:
            raise HTTPException(status_code=503, detail="Sincronización no disponible")
        
        conflicts = editor.sync_manager.detect_conflicts(local_content, remote_content)
        return {"success": True, "conflicts": conflicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/business-rules/validate")
async def validate_business_rules(
    content: str,
    operation: str,
    context: Optional[Dict[str, Any]] = None
):
    """
    Validar contra reglas de negocio.

    Args:
        content: Contenido
        operation: Tipo de operación
        context: Contexto adicional

    Returns:
        Resultado de la validación
    """
    try:
        if not editor.business_rules:
            raise HTTPException(status_code=503, detail="Reglas de negocio no disponibles")
        
        result = editor.business_rules.validate(content, operation, context)
        return {"success": True, "validation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business-rules/violations")
async def get_business_rule_violations(severity: Optional[str] = None, limit: int = 50):
    """
    Obtener violaciones de reglas de negocio.

    Args:
        severity: Filtrar por severidad
        limit: Límite de resultados

    Returns:
        Lista de violaciones
    """
    try:
        if not editor.business_rules:
            raise HTTPException(status_code=503, detail="Reglas de negocio no disponibles")
        
        from ..core.business_rules import RuleSeverity
        sev = RuleSeverity[severity.upper()] if severity else None
        
        violations = editor.business_rules.get_violations(sev, limit)
        return {"success": True, "violations": violations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/logs")
async def get_audit_logs(
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    limit: int = 100
):
    """
    Obtener logs de auditoría.

    Args:
        event_type: Filtrar por tipo
        user_id: Filtrar por usuario
        resource_id: Filtrar por recurso
        limit: Límite de resultados

    Returns:
        Lista de logs
    """
    try:
        if not editor.audit:
            raise HTTPException(status_code=503, detail="Auditoría no disponible")
        
        from ..core.audit import AuditEventType
        etype = AuditEventType[event_type.upper()] if event_type else None
        
        logs = editor.audit.query_logs(etype, user_id, resource_id, limit=limit)
        return {"success": True, "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit/report")
async def generate_audit_report(start_date: str, end_date: str):
    """
    Generar reporte de auditoría.

    Args:
        start_date: Fecha de inicio (ISO format)
        end_date: Fecha de fin (ISO format)

    Returns:
        Reporte de auditoría
    """
    try:
        if not editor.audit:
            raise HTTPException(status_code=503, detail="Auditoría no disponible")
        
        from datetime import datetime
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        report = editor.audit.generate_audit_report(start, end)
        return {"success": True, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare/versions")
async def compare_versions_detailed(
    version1: Dict[str, Any],
    version2: Dict[str, Any]
):
    """
    Comparar dos versiones en detalle.

    Args:
        version1: Versión 1
        version2: Versión 2

    Returns:
        Comparación detallada
    """
    try:
        if not editor.advanced_comparison:
            raise HTTPException(status_code=503, detail="Comparación avanzada no disponible")
        
        comparison = editor.advanced_comparison.compare_versions_detailed(version1, version2)
        return {"success": True, "comparison": comparison}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare/multiple")
async def compare_multiple_versions(versions: List[Dict[str, Any]]):
    """
    Comparar múltiples versiones.

    Args:
        versions: Lista de versiones

    Returns:
        Comparación múltiple
    """
    try:
        if not editor.advanced_comparison:
            raise HTTPException(status_code=503, detail="Comparación avanzada no disponible")
        
        comparison = editor.advanced_comparison.compare_multiple_versions(versions)
        return {"success": True, "comparison": comparison}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality/analyze")
async def analyze_quality(content: str, content_type: Optional[str] = None):
    """
    Analizar calidad del contenido.

    Args:
        content: Contenido a analizar
        content_type: Tipo de contenido

    Returns:
        Análisis de calidad
    """
    try:
        if not editor.quality_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de calidad no disponible")
        
        score = editor.quality_analyzer.analyze_quality(content, content_type)
        return {
            "success": True,
            "overall_score": score.overall,
            "level": score.level.value,
            "scores": {
                "readability": score.readability,
                "structure": score.structure,
                "grammar": score.grammar,
                "coherence": score.coherence,
                "completeness": score.completeness
            },
            "issues": score.issues,
            "suggestions": score.suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality/report")
async def get_quality_report(content: str):
    """
    Generar reporte de calidad.

    Args:
        content: Contenido

    Returns:
        Reporte de calidad
    """
    try:
        if not editor.quality_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de calidad no disponible")
        
        report = editor.quality_analyzer.get_quality_report(content)
        return {"success": True, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize")
async def summarize_content(
    content: str,
    max_sentences: int = 3,
    method: str = "extractive"
):
    """
    Generar resumen del contenido.

    Args:
        content: Contenido a resumir
        max_sentences: Número máximo de oraciones
        method: Método (extractive, abstractive)

    Returns:
        Resumen
    """
    try:
        if not editor.summarizer:
            raise HTTPException(status_code=503, detail="Generador de resúmenes no disponible")
        
        summary = editor.summarizer.summarize(content, max_sentences, method)
        return {"success": True, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize/sections")
async def summarize_by_sections(content: str):
    """
    Resumir por secciones.

    Args:
        content: Contenido con secciones

    Returns:
        Resumen por secciones
    """
    try:
        if not editor.summarizer:
            raise HTTPException(status_code=503, detail="Generador de resúmenes no disponible")
        
        summary = editor.summarizer.summarize_by_sections(content)
        return {"success": True, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize/key-points")
async def generate_key_points(content: str, max_points: int = 5):
    """
    Generar puntos clave.

    Args:
        content: Contenido
        max_points: Número máximo de puntos

    Returns:
        Lista de puntos clave
    """
    try:
        if not editor.summarizer:
            raise HTTPException(status_code=503, detail="Generador de resúmenes no disponible")
        
        points = editor.summarizer.generate_key_points(content, max_points)
        return {"success": True, "key_points": points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/index")
async def index_document(doc_id: str, content: str):
    """
    Indexar un documento para búsqueda.

    Args:
        doc_id: ID del documento
        content: Contenido del documento

    Returns:
        Resultado
    """
    try:
        if not editor.semantic_search:
            raise HTTPException(status_code=503, detail="Búsqueda semántica no disponible")
        
        editor.semantic_search.index_document(doc_id, content)
        return {"success": True, "message": f"Documento {doc_id} indexado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_documents(query: str, top_k: int = 10, threshold: float = 0.1):
    """
    Buscar documentos relevantes.

    Args:
        query: Consulta de búsqueda
        top_k: Número de resultados
        threshold: Umbral de relevancia

    Returns:
        Lista de resultados
    """
    try:
        if not editor.semantic_search:
            raise HTTPException(status_code=503, detail="Búsqueda semántica no disponible")
        
        results = editor.semantic_search.search(query, top_k, threshold)
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/similar/{doc_id}")
async def find_similar_documents(doc_id: str, top_k: int = 5):
    """
    Encontrar documentos similares.

    Args:
        doc_id: ID del documento
        top_k: Número de resultados

    Returns:
        Lista de documentos similares
    """
    try:
        if not editor.semantic_search:
            raise HTTPException(status_code=503, detail="Búsqueda semántica no disponible")
        
        results = editor.semantic_search.find_similar(doc_id, top_k)
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate")
async def translate_text(
    text: str,
    target_language: str,
    source_language: Optional[str] = None
):
    """
    Traducir texto.

    Args:
        text: Texto a traducir
        target_language: Idioma destino
        source_language: Idioma origen

    Returns:
        Traducción
    """
    try:
        if not editor.translator:
            raise HTTPException(status_code=503, detail="Traductor no disponible")
        
        translation = editor.translator.translate(text, target_language, source_language)
        return {"success": True, "translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate/batch")
async def translate_batch(
    texts: List[str],
    target_language: str,
    source_language: Optional[str] = None
):
    """
    Traducir múltiples textos.

    Args:
        texts: Lista de textos
        target_language: Idioma destino
        source_language: Idioma origen

    Returns:
        Lista de traducciones
    """
    try:
        if not editor.translator:
            raise HTTPException(status_code=503, detail="Traductor no disponible")
        
        translations = editor.translator.translate_batch(texts, target_language, source_language)
        return {"success": True, "translations": translations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/translate/languages")
async def get_supported_languages():
    """
    Obtener idiomas soportados.

    Returns:
        Lista de idiomas
    """
    try:
        if not editor.translator:
            raise HTTPException(status_code=503, detail="Traductor no disponible")
        
        languages = editor.translator.get_supported_languages()
        return {"success": True, "languages": languages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spell/check")
async def check_spelling(text: str, language: Optional[str] = None):
    """
    Verificar ortografía.

    Args:
        text: Texto a verificar
        language: Idioma

    Returns:
        Resultado de verificación
    """
    try:
        if not editor.spell_checker:
            raise HTTPException(status_code=503, detail="Corrector ortográfico no disponible")
        
        result = editor.spell_checker.check(text, language)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spell/correct")
async def correct_spelling(text: str, language: Optional[str] = None):
    """
    Corregir texto.

    Args:
        text: Texto a corregir
        language: Idioma

    Returns:
        Texto corregido
    """
    try:
        if not editor.spell_checker:
            raise HTTPException(status_code=503, detail="Corrector ortográfico no disponible")
        
        result = editor.spell_checker.correct(text, language)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spell/add-word")
async def add_word_to_dictionary(word: str):
    """
    Agregar palabra al diccionario.

    Args:
        word: Palabra a agregar

    Returns:
        Resultado
    """
    try:
        if not editor.spell_checker:
            raise HTTPException(status_code=503, detail="Corrector ortográfico no disponible")
        
        editor.spell_checker.add_to_dictionary(word)
        return {"success": True, "message": f"Palabra '{word}' agregada al diccionario"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/content")
async def validate_content(
    content: str,
    context: Optional[Dict[str, Any]] = None
):
    """
    Validar contenido.

    Args:
        content: Contenido a validar
        context: Contexto adicional

    Returns:
        Resultado de validación
    """
    try:
        if not editor.content_validator:
            raise HTTPException(status_code=503, detail="Validador de contenido no disponible")
        
        result = editor.content_validator.validate(content, context)
        return {"success": True, "validation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/set-level")
async def set_validation_level(level: str):
    """
    Establecer nivel de validación.

    Args:
        level: Nivel (strict, moderate, lenient)

    Returns:
        Resultado
    """
    try:
        if not editor.content_validator:
            raise HTTPException(status_code=503, detail="Validador de contenido no disponible")
        
        from ..core.content_validator import ValidationLevel
        validation_level = ValidationLevel[level.upper()]
        editor.content_validator.set_validation_level(validation_level)
        return {"success": True, "level": level}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment/analyze")
async def analyze_sentiment(text: str, detailed: bool = False):
    """
    Analizar sentimiento del texto.

    Args:
        text: Texto a analizar
        detailed: Si incluir detalles

    Returns:
        Análisis de sentimiento
    """
    try:
        if not editor.sentiment_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de sentimientos no disponible")
        
        result = editor.sentiment_analyzer.analyze(text, detailed)
        return {"success": True, "sentiment": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment/by-sentences")
async def analyze_sentiment_by_sentences(text: str):
    """
    Analizar sentimiento por oraciones.

    Args:
        text: Texto

    Returns:
        Análisis por oraciones
    """
    try:
        if not editor.sentiment_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de sentimientos no disponible")
        
        result = editor.sentiment_analyzer.analyze_by_sentences(text)
        return {"success": True, "sentiment": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/extract")
async def extract_entities(text: str):
    """
    Extraer entidades del texto.

    Args:
        text: Texto

    Returns:
        Lista de entidades
    """
    try:
        if not editor.entity_extractor:
            raise HTTPException(status_code=503, detail="Extractor de entidades no disponible")
        
        entities = editor.entity_extractor.extract(text)
        return {
            "success": True,
            "entities": [
                {
                    "text": e.text,
                    "type": e.type.value,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence
                }
                for e in entities
            ],
            "count": len(entities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/summary")
async def get_entities_summary(text: str):
    """
    Obtener resumen de entidades.

    Args:
        text: Texto

    Returns:
        Resumen de entidades
    """
    try:
        if not editor.entity_extractor:
            raise HTTPException(status_code=503, detail="Extractor de entidades no disponible")
        
        summary = editor.entity_extractor.get_entities_summary(text)
        return {"success": True, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/by-type")
async def extract_entities_by_type(text: str, entity_type: str):
    """
    Extraer entidades de un tipo específico.

    Args:
        text: Texto
        entity_type: Tipo de entidad

    Returns:
        Lista de entidades
    """
    try:
        if not editor.entity_extractor:
            raise HTTPException(status_code=503, detail="Extractor de entidades no disponible")
        
        from ..core.entity_extractor import EntityType
        etype = EntityType[entity_type.upper()]
        entities = editor.entity_extractor.extract_by_type(text, etype)
        
        return {
            "success": True,
            "entities": [
                {
                    "text": e.text,
                    "type": e.type.value,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence
                }
                for e in entities
            ],
            "count": len(entities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plagiarism/add-reference")
async def add_plagiarism_reference(doc_id: str, content: str):
    """
    Agregar documento de referencia para detección de plagio.

    Args:
        doc_id: ID del documento
        content: Contenido

    Returns:
        Resultado
    """
    try:
        if not editor.plagiarism_detector:
            raise HTTPException(status_code=503, detail="Detector de plagio no disponible")
        
        editor.plagiarism_detector.add_reference(doc_id, content)
        return {"success": True, "message": f"Documento {doc_id} agregado como referencia"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plagiarism/check")
async def check_plagiarism(
    content: str,
    min_similarity: float = 0.5,
    min_phrase_length: int = 5
):
    """
    Verificar plagio.

    Args:
        content: Contenido a verificar
        min_similarity: Similitud mínima
        min_phrase_length: Longitud mínima de frase

    Returns:
        Resultado de verificación
    """
    try:
        if not editor.plagiarism_detector:
            raise HTTPException(status_code=503, detail="Detector de plagio no disponible")
        
        result = editor.plagiarism_detector.check(content, min_similarity, min_phrase_length)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plagiarism/compare")
async def compare_documents_for_plagiarism(doc1_id: str, doc2_id: str):
    """
    Comparar dos documentos para detectar plagio.

    Args:
        doc1_id: ID documento 1
        doc2_id: ID documento 2

    Returns:
        Comparación
    """
    try:
        if not editor.plagiarism_detector:
            raise HTTPException(status_code=503, detail="Detector de plagio no disponible")
        
        result = editor.plagiarism_detector.compare_documents(doc1_id, doc2_id)
        return {"success": True, "comparison": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics/extract")
async def extract_topics(
    content: str,
    num_topics: int = 5,
    min_word_length: int = 4
):
    """
    Extraer temas del contenido.

    Args:
        content: Contenido
        num_topics: Número de temas
        min_word_length: Longitud mínima de palabra

    Returns:
        Temas extraídos
    """
    try:
        if not editor.topic_modeler:
            raise HTTPException(status_code=503, detail="Modelador de temas no disponible")
        
        topics = editor.topic_modeler.extract_topics(content, num_topics, min_word_length)
        return {"success": True, "topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics/by-sections")
async def extract_topics_by_sections(
    content: str,
    num_topics_per_section: int = 3
):
    """
    Extraer temas por secciones.

    Args:
        content: Contenido con secciones
        num_topics_per_section: Temas por sección

    Returns:
        Temas por sección
    """
    try:
        if not editor.topic_modeler:
            raise HTTPException(status_code=503, detail="Modelador de temas no disponible")
        
        topics = editor.topic_modeler.extract_topics_by_sections(content, num_topics_per_section)
        return {"success": True, "topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics/keywords")
async def get_topic_keywords(
    content: str,
    topic: str,
    num_keywords: int = 10
):
    """
    Obtener palabras clave relacionadas con un tema.

    Args:
        content: Contenido
        topic: Tema
        num_keywords: Número de palabras clave

    Returns:
        Lista de palabras clave
    """
    try:
        if not editor.topic_modeler:
            raise HTTPException(status_code=503, detail="Modelador de temas no disponible")
        
        keywords = editor.topic_modeler.get_topic_keywords(content, topic, num_keywords)
        return {"success": True, "keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complexity/analyze")
async def analyze_complexity(content: str, detailed: bool = False):
    """
    Analizar complejidad del contenido.

    Args:
        content: Contenido a analizar
        detailed: Si incluir detalles

    Returns:
        Análisis de complejidad
    """
    try:
        if not editor.complexity_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de complejidad no disponible")
        
        result = editor.complexity_analyzer.analyze(content, detailed)
        return {"success": True, "complexity": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/introduction")
async def generate_introduction(topic: str, length: str = "medium"):
    """
    Generar introducción.

    Args:
        topic: Tema
        length: Longitud (short, medium, long)

    Returns:
        Introducción generada
    """
    try:
        if not editor.content_generator:
            raise HTTPException(status_code=503, detail="Generador de contenido no disponible")
        
        introduction = editor.content_generator.generate_introduction(topic, length)
        return {"success": True, "introduction": introduction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/conclusion")
async def generate_conclusion(summary: str, length: str = "medium"):
    """
    Generar conclusión.

    Args:
        summary: Resumen
        length: Longitud

    Returns:
        Conclusión generada
    """
    try:
        if not editor.content_generator:
            raise HTTPException(status_code=503, detail="Generador de contenido no disponible")
        
        conclusion = editor.content_generator.generate_conclusion(summary, length)
        return {"success": True, "conclusion": conclusion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/expand")
async def expand_content(
    content: str,
    target_length: int,
    style: str = "formal"
):
    """
    Expandir contenido.

    Args:
        content: Contenido original
        target_length: Longitud objetivo
        style: Estilo (formal, informal, technical)

    Returns:
        Contenido expandido
    """
    try:
        if not editor.content_generator:
            raise HTTPException(status_code=503, detail="Generador de contenido no disponible")
        
        expanded = editor.content_generator.expand_content(content, target_length, style)
        return {"success": True, "expanded_content": expanded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/bullet-points")
async def generate_bullet_points(topic: str, num_points: int = 5):
    """
    Generar puntos de lista.

    Args:
        topic: Tema
        num_points: Número de puntos

    Returns:
        Lista de puntos
    """
    try:
        if not editor.content_generator:
            raise HTTPException(status_code=503, detail="Generador de contenido no disponible")
        
        points = editor.content_generator.generate_bullet_points(topic, num_points)
        return {"success": True, "bullet_points": points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/suggestions")
async def get_content_suggestions(content: str):
    """
    Sugerir mejoras al contenido.

    Args:
        content: Contenido

    Returns:
        Lista de sugerencias
    """
    try:
        if not editor.content_generator:
            raise HTTPException(status_code=503, detail="Generador de contenido no disponible")
        
        suggestions = editor.content_generator.suggest_improvements(content)
        return {"success": True, "suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/redundancy/analyze")
async def analyze_redundancy(content: str, threshold: float = 0.3):
    """
    Analizar redundancia del contenido.

    Args:
        content: Contenido a analizar
        threshold: Umbral de redundancia

    Returns:
        Análisis de redundancia
    """
    try:
        if not editor.redundancy_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de redundancia no disponible")
        
        result = editor.redundancy_analyzer.analyze(content, threshold)
        return {"success": True, "redundancy": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/redundancy/find-sections")
async def find_redundant_sections(content: str):
    """
    Encontrar secciones redundantes.

    Args:
        content: Contenido

    Returns:
        Lista de secciones redundantes
    """
    try:
        if not editor.redundancy_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de redundancia no disponible")
        
        sections = editor.redundancy_analyzer.find_redundant_sections(content)
        return {"success": True, "redundant_sections": sections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/structure/analyze")
async def analyze_structure(content: str):
    """
    Analizar estructura del documento.

    Args:
        content: Contenido

    Returns:
        Análisis de estructura
    """
    try:
        if not editor.structure_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de estructura no disponible")
        
        result = editor.structure_analyzer.analyze(content)
        return {"success": True, "structure": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tone/analyze")
async def analyze_tone(content: str):
    """
    Analizar tono del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de tono
    """
    try:
        if not editor.tone_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de tono no disponible")
        
        result = editor.tone_analyzer.analyze(content)
        return {"success": True, "tone": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tone/by-sections")
async def analyze_tone_by_sections(content: str):
    """
    Analizar tono por secciones.

    Args:
        content: Contenido

    Returns:
        Análisis por secciones
    """
    try:
        if not editor.tone_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de tono no disponible")
        
        result = editor.tone_analyzer.analyze_by_sections(content)
        return {"success": True, "tone": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/coherence/analyze")
async def analyze_coherence(content: str):
    """
    Analizar coherencia del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de coherencia
    """
    try:
        if not editor.coherence_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de coherencia no disponible")
        
        result = editor.coherence_analyzer.analyze(content)
        return {"success": True, "coherence": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/coherence/paragraphs")
async def analyze_paragraph_coherence(content: str):
    """
    Analizar coherencia entre párrafos.

    Args:
        content: Contenido

    Returns:
        Análisis de coherencia entre párrafos
    """
    try:
        if not editor.coherence_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de coherencia no disponible")
        
        result = editor.coherence_analyzer.analyze_paragraph_coherence(content)
        return {"success": True, "coherence": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/accessibility/analyze")
async def analyze_accessibility(content: str):
    """
    Analizar accesibilidad del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de accesibilidad
    """
    try:
        if not editor.accessibility_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de accesibilidad no disponible")
        
        result = editor.accessibility_analyzer.analyze(content)
        return {"success": True, "accessibility": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seo/analyze")
async def analyze_seo(
    content: str,
    target_keywords: Optional[List[str]] = None
):
    """
    Analizar SEO del contenido.

    Args:
        content: Contenido
        target_keywords: Palabras clave objetivo (opcional)

    Returns:
        Análisis SEO
    """
    try:
        if not editor.seo_analyzer:
            raise HTTPException(status_code=503, detail="Analizador SEO no disponible")
        
        result = editor.seo_analyzer.analyze(content, target_keywords)
        return {"success": True, "seo": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/readability/advanced")
async def analyze_readability_advanced(content: str):
    """
    Analizar legibilidad avanzada del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de legibilidad avanzada
    """
    try:
        if not editor.readability_advanced:
            raise HTTPException(status_code=503, detail="Analizador de legibilidad avanzado no disponible")
        
        result = editor.readability_advanced.analyze(content)
        return {"success": True, "readability": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fluency/analyze")
async def analyze_fluency(content: str):
    """
    Analizar fluidez del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de fluidez
    """
    try:
        if not editor.fluency_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de fluidez no disponible")
        
        result = editor.fluency_analyzer.analyze(content)
        return {"success": True, "fluency": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vocabulary/analyze")
async def analyze_vocabulary(content: str):
    """
    Analizar vocabulario del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de vocabulario
    """
    try:
        if not editor.vocabulary_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de vocabulario no disponible")
        
        result = editor.vocabulary_analyzer.analyze(content)
        return {"success": True, "vocabulary": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/format/analyze")
async def analyze_format(content: str):
    """
    Analizar formato del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de formato
    """
    try:
        if not editor.format_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de formato no disponible")
        
        result = editor.format_analyzer.analyze(content)
        return {"success": True, "format": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/length/analyze")
async def analyze_length(content: str, content_type: str = "article"):
    """
    Analizar longitud del contenido.

    Args:
        content: Contenido
        content_type: Tipo de contenido

    Returns:
        Análisis de longitud
    """
    try:
        if not editor.length_optimizer:
            raise HTTPException(status_code=503, detail="Optimizador de longitud no disponible")
        
        result = editor.length_optimizer.analyze(content, content_type)
        return {"success": True, "length": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/length/optimize")
async def optimize_length(
    content: str,
    target_length: int,
    method: str = "expand"
):
    """
    Optimizar longitud del contenido.

    Args:
        content: Contenido
        target_length: Longitud objetivo
        method: Método (expand, reduce, maintain)

    Returns:
        Contenido optimizado
    """
    try:
        if not editor.length_optimizer:
            raise HTTPException(status_code=503, detail="Optimizador de longitud no disponible")
        
        result = editor.length_optimizer.optimize_length(content, target_length, method)
        return {"success": True, "optimization": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/generate")
async def generate_recommendations(
    content: str,
    analysis_results: Optional[Dict[str, Any]] = None
):
    """
    Generar recomendaciones de mejora.

    Args:
        content: Contenido
        analysis_results: Resultados de análisis previos (opcional)

    Returns:
        Recomendaciones
    """
    try:
        if not editor.improvement_recommender:
            raise HTTPException(status_code=503, detail="Recomendador de mejoras no disponible")
        
        result = editor.improvement_recommender.generate_recommendations(content, analysis_results)
        return {"success": True, "recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engagement/analyze")
async def analyze_engagement(content: str):
    """
    Analizar engagement del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de engagement
    """
    try:
        if not editor.engagement_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de engagement no disponible")
        
        result = editor.engagement_analyzer.analyze(content)
        return {"success": True, "engagement": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/calculate")
async def calculate_content_metrics(content: str):
    """
    Calcular métricas completas del contenido.

    Args:
        content: Contenido

    Returns:
        Métricas completas
    """
    try:
        if not editor.content_metrics:
            raise HTTPException(status_code=503, detail="Métricas de contenido no disponibles")
        
        result = editor.content_metrics.calculate_comprehensive_metrics(content)
        return {"success": True, "metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/compare")
async def compare_content_metrics(
    metrics1: Dict[str, Any],
    metrics2: Dict[str, Any]
):
    """
    Comparar dos conjuntos de métricas.

    Args:
        metrics1: Métricas 1
        metrics2: Métricas 2

    Returns:
        Comparación
    """
    try:
        if not editor.content_metrics:
            raise HTTPException(status_code=503, detail="Métricas de contenido no disponibles")
        
        result = editor.content_metrics.compare_metrics(metrics1, metrics2)
        return {"success": True, "comparison": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/analyze")
async def analyze_performance(operation_name: Optional[str] = None, limit: int = 100):
    """
    Analizar performance.

    Args:
        operation_name: Filtrar por nombre de operación
        limit: Límite de métricas

    Returns:
        Análisis de performance
    """
    try:
        if not editor.performance_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de performance no disponible")
        
        result = editor.performance_analyzer.analyze_performance(operation_name, limit)
        return {"success": True, "performance": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/stats")
async def get_performance_stats():
    """
    Obtener estadísticas de performance.

    Returns:
        Estadísticas
    """
    try:
        if not editor.performance_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de performance no disponible")
        
        result = editor.performance_analyzer.get_performance_stats()
        return {"success": True, "stats": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance/clear")
async def clear_performance_metrics():
    """
    Limpiar métricas de performance.

    Returns:
        Resultado
    """
    try:
        if not editor.performance_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de performance no disponible")
        
        editor.performance_analyzer.clear_metrics()
        return {"success": True, "message": "Métricas de performance limpiadas"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trends/record")
async def record_content_for_trends(
    content_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Registrar contenido para análisis de tendencias.

    Args:
        content_id: ID del contenido
        content: Contenido
        metadata: Metadatos adicionales

    Returns:
        Resultado
    """
    try:
        if not editor.trend_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de tendencias no disponible")
        
        editor.trend_analyzer.record_content(content_id, content, metadata)
        return {"success": True, "message": f"Contenido {content_id} registrado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/analyze")
async def analyze_trends(period_days: int = 30, metric: str = "word_count"):
    """
    Analizar tendencias.

    Args:
        period_days: Período en días
        metric: Métrica a analizar

    Returns:
        Análisis de tendencias
    """
    try:
        if not editor.trend_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de tendencias no disponible")
        
        result = editor.trend_analyzer.analyze_trends(period_days, metric)
        return {"success": True, "trends": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/keywords")
async def analyze_keyword_trends(period_days: int = 30, top_n: int = 10):
    """
    Analizar tendencias de keywords.

    Args:
        period_days: Período en días
        top_n: Número de keywords top

    Returns:
        Análisis de tendencias de keywords
    """
    try:
        if not editor.trend_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de tendencias no disponible")
        
        result = editor.trend_analyzer.analyze_keyword_trends(period_days, top_n)
        return {"success": True, "keyword_trends": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/predict")
async def predict_future_trend(metric: str = "word_count", days_ahead: int = 7):
    """
    Predecir tendencia futura.

    Args:
        metric: Métrica a predecir
        days_ahead: Días a predecir

    Returns:
        Predicción
    """
    try:
        if not editor.trend_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de tendencias no disponible")
        
        result = editor.trend_analyzer.predict_future_trend(metric, days_ahead)
        return {"success": True, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitors/add")
async def add_competitor(
    competitor_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Agregar contenido de competidor.

    Args:
        competitor_id: ID del competidor
        content: Contenido
        metadata: Metadatos adicionales

    Returns:
        Resultado
    """
    try:
        if not editor.competitor_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de competencia no disponible")
        
        editor.competitor_analyzer.add_competitor(competitor_id, content, metadata)
        return {"success": True, "message": f"Competidor {competitor_id} agregado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitors/compare")
async def compare_with_competitors(content: str, metric: str = "comprehensive"):
    """
    Comparar contenido con competidores.

    Args:
        content: Contenido a comparar
        metric: Métrica de comparación

    Returns:
        Comparación con competidores
    """
    try:
        if not editor.competitor_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de competencia no disponible")
        
        result = editor.competitor_analyzer.compare_with_competitors(content, metric)
        return {"success": True, "comparison": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitors/insights")
async def get_competitive_insights(content: str):
    """
    Obtener insights competitivos.

    Args:
        content: Contenido

    Returns:
        Insights competitivos
    """
    try:
        if not editor.competitor_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de competencia no disponible")
        
        result = editor.competitor_analyzer.get_competitive_insights(content)
        return {"success": True, "insights": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/roi/record-investment")
async def record_investment(
    content_id: str,
    cost: float,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Registrar inversión en contenido.

    Args:
        content_id: ID del contenido
        cost: Costo de producción
        metadata: Metadatos adicionales

    Returns:
        Resultado
    """
    try:
        if not editor.roi_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de ROI no disponible")
        
        editor.roi_analyzer.record_investment(content_id, cost, metadata)
        return {"success": True, "message": f"Inversión registrada para {content_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/roi/record-revenue")
async def record_revenue(
    content_id: str,
    revenue: float,
    metric_name: str = "revenue"
):
    """
    Registrar ingresos de contenido.

    Args:
        content_id: ID del contenido
        revenue: Ingresos generados
        metric_name: Nombre de la métrica

    Returns:
        Resultado
    """
    try:
        if not editor.roi_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de ROI no disponible")
        
        editor.roi_analyzer.record_revenue(content_id, revenue, metric_name)
        return {"success": True, "message": f"Ingresos registrados para {content_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roi/calculate/{content_id}")
async def calculate_roi(content_id: str):
    """
    Calcular ROI de un contenido.

    Args:
        content_id: ID del contenido

    Returns:
        Análisis de ROI
    """
    try:
        if not editor.roi_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de ROI no disponible")
        
        result = editor.roi_analyzer.calculate_roi(content_id)
        return {"success": True, "roi": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roi/portfolio")
async def analyze_portfolio_roi():
    """
    Analizar ROI del portafolio completo.

    Returns:
        Análisis de ROI del portafolio
    """
    try:
        if not editor.roi_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de ROI no disponible")
        
        result = editor.roi_analyzer.analyze_portfolio_roi()
        return {"success": True, "portfolio": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roi/recommendations")
async def get_roi_recommendations():
    """
    Obtener recomendaciones basadas en ROI.

    Returns:
        Recomendaciones
    """
    try:
        if not editor.roi_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de ROI no disponible")
        
        result = editor.roi_analyzer.get_roi_recommendations()
        return {"success": True, "recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audience/analyze")
async def analyze_audience_fit(
    content: str,
    target_audience: str = "general"
):
    """
    Analizar si el contenido se ajusta a la audiencia objetivo.

    Args:
        content: Contenido
        target_audience: Audiencia objetivo (beginner, intermediate, advanced, expert, general)

    Returns:
        Análisis de ajuste a audiencia
    """
    try:
        if not editor.audience_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de audiencia no disponible")
        
        result = editor.audience_analyzer.analyze_audience_fit(content, target_audience)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversion/analyze")
async def analyze_conversion_potential(content: str):
    """
    Analizar potencial de conversión del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de potencial de conversión
    """
    try:
        if not editor.conversion_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de conversión no disponible")
        
        result = editor.conversion_analyzer.analyze_conversion_potential(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-testing/create")
async def create_ab_test(
    name: str,
    description: str,
    variants: List[Dict[str, str]],
    duration_days: Optional[int] = None
):
    """
    Crear prueba A/B.

    Args:
        name: Nombre de la prueba
        description: Descripción
        variants: Lista de variantes (cada una con 'name' y 'content')
        duration_days: Duración en días (opcional)

    Returns:
        ID de la prueba
    """
    try:
        if not editor.ab_testing:
            raise HTTPException(status_code=503, detail="Gestor de A/B testing no disponible")
        
        test_id = editor.ab_testing.create_test(name, description, variants, duration_days)
        return {"success": True, "test_id": test_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-testing/impression")
async def record_impression(test_id: str, variant_id: str):
    """
    Registrar impresión de variante.

    Args:
        test_id: ID de la prueba
        variant_id: ID de la variante

    Returns:
        Resultado
    """
    try:
        if not editor.ab_testing:
            raise HTTPException(status_code=503, detail="Gestor de A/B testing no disponible")
        
        editor.ab_testing.record_impression(test_id, variant_id)
        return {"success": True, "message": "Impresión registrada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-testing/conversion")
async def record_conversion(test_id: str, variant_id: str):
    """
    Registrar conversión de variante.

    Args:
        test_id: ID de la prueba
        variant_id: ID de la variante

    Returns:
        Resultado
    """
    try:
        if not editor.ab_testing:
            raise HTTPException(status_code=503, detail="Gestor de A/B testing no disponible")
        
        editor.ab_testing.record_conversion(test_id, variant_id)
        return {"success": True, "message": "Conversión registrada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-testing/results/{test_id}")
async def get_ab_test_results(test_id: str):
    """
    Obtener resultados de prueba A/B.

    Args:
        test_id: ID de la prueba

    Returns:
        Resultados de la prueba
    """
    try:
        if not editor.ab_testing:
            raise HTTPException(status_code=503, detail="Gestor de A/B testing no disponible")
        
        result = editor.ab_testing.get_test_results(test_id)
        return {"success": True, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-testing/tests")
async def get_all_ab_tests():
    """
    Obtener todas las pruebas A/B.

    Returns:
        Lista de pruebas
    """
    try:
        if not editor.ab_testing:
            raise HTTPException(status_code=503, detail="Gestor de A/B testing no disponible")
        
        result = editor.ab_testing.get_all_tests()
        return {"success": True, "tests": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-testing/end/{test_id}")
async def end_ab_test(test_id: str):
    """
    Finalizar prueba A/B.

    Args:
        test_id: ID de la prueba

    Returns:
        Resultados finales
    """
    try:
        if not editor.ab_testing:
            raise HTTPException(status_code=503, detail="Gestor de A/B testing no disponible")
        
        result = editor.ab_testing.end_test(test_id)
        return {"success": True, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/add")
async def add_feedback(
    content_id: str,
    feedback_text: str,
    rating: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Agregar feedback.

    Args:
        content_id: ID del contenido
        feedback_text: Texto del feedback
        rating: Calificación (opcional)
        metadata: Metadatos adicionales

    Returns:
        ID del feedback
    """
    try:
        if not editor.feedback_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de feedback no disponible")
        
        feedback_id = editor.feedback_analyzer.add_feedback(content_id, feedback_text, rating, metadata)
        return {"success": True, "feedback_id": feedback_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/analyze/{content_id}")
async def analyze_content_feedback(content_id: str):
    """
    Analizar feedback de un contenido.

    Args:
        content_id: ID del contenido

    Returns:
        Análisis de feedback
    """
    try:
        if not editor.feedback_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de feedback no disponible")
        
        result = editor.feedback_analyzer.analyze_content_feedback(content_id)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/summary")
async def get_feedback_summary():
    """
    Obtener resumen de todo el feedback.

    Returns:
        Resumen de feedback
    """
    try:
        if not editor.feedback_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de feedback no disponible")
        
        result = editor.feedback_analyzer.get_feedback_summary()
        return {"success": True, "summary": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/personalization/create-profile")
async def create_user_profile(
    user_id: str,
    initial_preferences: Optional[Dict[str, Any]] = None
):
    """
    Crear perfil de usuario.

    Args:
        user_id: ID del usuario
        initial_preferences: Preferencias iniciales

    Returns:
        Resultado
    """
    try:
        if not editor.personalization_engine:
            raise HTTPException(status_code=503, detail="Motor de personalización no disponible")
        
        editor.personalization_engine.create_user_profile(user_id, initial_preferences)
        return {"success": True, "message": f"Perfil creado para usuario {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/personalization/update-preferences")
async def update_user_preferences(
    user_id: str,
    preferences: Dict[str, Any]
):
    """
    Actualizar preferencias de usuario.

    Args:
        user_id: ID del usuario
        preferences: Nuevas preferencias

    Returns:
        Resultado
    """
    try:
        if not editor.personalization_engine:
            raise HTTPException(status_code=503, detail="Motor de personalización no disponible")
        
        editor.personalization_engine.update_user_preferences(user_id, preferences)
        return {"success": True, "message": f"Preferencias actualizadas para usuario {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/personalization/record-interaction")
async def record_interaction(
    user_id: str,
    content_id: str,
    interaction_type: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Registrar interacción de usuario.

    Args:
        user_id: ID del usuario
        content_id: ID del contenido
        interaction_type: Tipo de interacción
        metadata: Metadatos adicionales

    Returns:
        Resultado
    """
    try:
        if not editor.personalization_engine:
            raise HTTPException(status_code=503, detail="Motor de personalización no disponible")
        
        editor.personalization_engine.record_interaction(user_id, content_id, interaction_type, metadata)
        return {"success": True, "message": "Interacción registrada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/personalization/personalize")
async def personalize_content(
    user_id: str,
    content: str,
    content_id: Optional[str] = None
):
    """
    Personalizar contenido para usuario.

    Args:
        user_id: ID del usuario
        content: Contenido original
        content_id: ID del contenido (opcional)

    Returns:
        Contenido personalizado
    """
    try:
        if not editor.personalization_engine:
            raise HTTPException(status_code=503, detail="Motor de personalización no disponible")
        
        result = editor.personalization_engine.personalize_content(user_id, content, content_id)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/personalization/recommendations/{user_id}")
async def get_content_recommendations(user_id: str, limit: int = 10):
    """
    Obtener recomendaciones de contenido para usuario.

    Args:
        user_id: ID del usuario
        limit: Límite de recomendaciones

    Returns:
        Lista de recomendaciones
    """
    try:
        if not editor.personalization_engine:
            raise HTTPException(status_code=503, detail="Motor de personalización no disponible")
        
        result = editor.personalization_engine.get_content_recommendations(user_id, limit)
        return {"success": True, "recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/personalization/tag-content")
async def tag_content(content_id: str, tags: List[str]):
    """
    Etiquetar contenido.

    Args:
        content_id: ID del contenido
        tags: Lista de tags

    Returns:
        Resultado
    """
    try:
        if not editor.personalization_engine:
            raise HTTPException(status_code=503, detail="Motor de personalización no disponible")
        
        editor.personalization_engine.tag_content(content_id, tags)
        return {"success": True, "message": f"Contenido {content_id} etiquetado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/satisfaction/record")
async def record_satisfaction(
    content_id: str,
    satisfaction_score: float,
    user_id: Optional[str] = None,
    metric_type: str = "overall",
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Registrar métrica de satisfacción.

    Args:
        content_id: ID del contenido
        satisfaction_score: Score de satisfacción (0-1)
        user_id: ID del usuario (opcional)
        metric_type: Tipo de métrica
        metadata: Metadatos adicionales

    Returns:
        Resultado
    """
    try:
        if not editor.satisfaction_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de satisfacción no disponible")
        
        editor.satisfaction_analyzer.record_satisfaction(
            content_id, satisfaction_score, user_id, metric_type, metadata
        )
        return {"success": True, "message": "Métrica de satisfacción registrada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/satisfaction/analyze/{content_id}")
async def analyze_content_satisfaction(content_id: str, period_days: Optional[int] = None):
    """
    Analizar satisfacción de un contenido.

    Args:
        content_id: ID del contenido
        period_days: Período en días (opcional)

    Returns:
        Análisis de satisfacción
    """
    try:
        if not editor.satisfaction_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de satisfacción no disponible")
        
        result = editor.satisfaction_analyzer.analyze_content_satisfaction(content_id, period_days)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/satisfaction/overall")
async def get_overall_satisfaction(period_days: Optional[int] = None):
    """
    Obtener satisfacción general.

    Args:
        period_days: Período en días (opcional)

    Returns:
        Satisfacción general
    """
    try:
        if not editor.satisfaction_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de satisfacción no disponible")
        
        result = editor.satisfaction_analyzer.get_overall_satisfaction(period_days)
        return {"success": True, "satisfaction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/satisfaction/recommendations/{content_id}")
async def get_satisfaction_recommendations(content_id: str):
    """
    Obtener recomendaciones basadas en satisfacción.

    Args:
        content_id: ID del contenido

    Returns:
        Recomendaciones
    """
    try:
        if not editor.satisfaction_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de satisfacción no disponible")
        
        result = editor.satisfaction_analyzer.get_satisfaction_recommendations(content_id)
        return {"success": True, "recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/behavior/record")
async def record_behavior(
    user_id: str,
    content_id: str,
    action_type: str,
    duration: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Registrar comportamiento de usuario.

    Args:
        user_id: ID del usuario
        content_id: ID del contenido
        action_type: Tipo de acción
        duration: Duración en segundos (opcional)
        metadata: Metadatos adicionales

    Returns:
        Resultado
    """
    try:
        if not editor.behavior_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de comportamiento no disponible")
        
        editor.behavior_analyzer.record_behavior(user_id, content_id, action_type, duration, metadata)
        return {"success": True, "message": "Comportamiento registrado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/behavior/user/{user_id}")
async def analyze_user_behavior(user_id: str, period_days: Optional[int] = None):
    """
    Analizar comportamiento de un usuario.

    Args:
        user_id: ID del usuario
        period_days: Período en días (opcional)

    Returns:
        Análisis de comportamiento
    """
    try:
        if not editor.behavior_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de comportamiento no disponible")
        
        result = editor.behavior_analyzer.analyze_user_behavior(user_id, period_days)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/behavior/content/{content_id}")
async def analyze_content_behavior(content_id: str, period_days: Optional[int] = None):
    """
    Analizar comportamiento de usuarios con un contenido.

    Args:
        content_id: ID del contenido
        period_days: Período en días (opcional)

    Returns:
        Análisis de comportamiento del contenido
    """
    try:
        if not editor.behavior_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de comportamiento no disponible")
        
        result = editor.behavior_analyzer.analyze_content_behavior(content_id, period_days)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/behavior/patterns")
async def get_behavior_patterns(user_id: Optional[str] = None):
    """
    Obtener patrones de comportamiento.

    Args:
        user_id: ID del usuario (opcional)

    Returns:
        Patrones de comportamiento
    """
    try:
        if not editor.behavior_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de comportamiento no disponible")
        
        result = editor.behavior_analyzer.get_behavior_patterns(user_id)
        return {"success": True, "patterns": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retention/record-visit")
async def record_visit(
    user_id: str,
    content_id: str,
    start_time: Optional[datetime] = None
):
    """
    Registrar visita de usuario.

    Args:
        user_id: ID del usuario
        content_id: ID del contenido
        start_time: Tiempo de inicio (opcional)

    Returns:
        ID de sesión
    """
    try:
        if not editor.retention_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de retención no disponible")
        
        session_id = editor.retention_analyzer.record_visit(user_id, content_id, start_time)
        return {"success": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retention/complete-session")
async def complete_session(
    user_id: str,
    content_id: str,
    completed: bool = True,
    duration: Optional[float] = None
):
    """
    Completar sesión de usuario.

    Args:
        user_id: ID del usuario
        content_id: ID del contenido
        completed: Si se completó el contenido
        duration: Duración en segundos

    Returns:
        Resultado
    """
    try:
        if not editor.retention_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de retención no disponible")
        
        editor.retention_analyzer.complete_session(user_id, content_id, completed, duration)
        return {"success": True, "message": "Sesión completada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retention/rate")
async def calculate_retention_rate(period_days: int = 30, cohort: Optional[str] = None):
    """
    Calcular tasa de retención.

    Args:
        period_days: Período en días
        cohort: Cohorte específica (opcional)

    Returns:
        Análisis de retención
    """
    try:
        if not editor.retention_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de retención no disponible")
        
        result = editor.retention_analyzer.calculate_retention_rate(period_days, cohort)
        return {"success": True, "retention": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retention/user/{user_id}")
async def analyze_user_retention(user_id: str):
    """
    Analizar retención de un usuario específico.

    Args:
        user_id: ID del usuario

    Returns:
        Análisis de retención del usuario
    """
    try:
        if not editor.retention_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de retención no disponible")
        
        result = editor.retention_analyzer.analyze_user_retention(user_id)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retention/cohorts")
async def get_retention_cohorts(cohort_size_days: int = 7):
    """
    Obtener análisis de cohortes de retención.

    Args:
        cohort_size_days: Tamaño de cohorte en días

    Returns:
        Análisis de cohortes
    """
    try:
        if not editor.retention_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de retención no disponible")
        
        result = editor.retention_analyzer.get_retention_cohorts(cohort_size_days)
        return {"success": True, "cohorts": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/virality/record-share")
async def record_share(
    content_id: str,
    user_id: str,
    share_type: str = "social",
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Registrar compartir de contenido.

    Args:
        content_id: ID del contenido
        user_id: ID del usuario
        share_type: Tipo de compartir
        metadata: Metadatos adicionales

    Returns:
        Resultado
    """
    try:
        if not editor.virality_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de viralidad no disponible")
        
        editor.virality_analyzer.record_share(content_id, user_id, share_type, metadata)
        return {"success": True, "message": "Compartir registrado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/virality/score/{content_id}")
async def calculate_virality_score(content_id: str, period_days: Optional[int] = None):
    """
    Calcular score de viralidad de un contenido.

    Args:
        content_id: ID del contenido
        period_days: Período en días (opcional)

    Returns:
        Análisis de viralidad
    """
    try:
        if not editor.virality_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de viralidad no disponible")
        
        result = editor.virality_analyzer.calculate_virality_score(content_id, period_days)
        return {"success": True, "virality": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/virality/top")
async def analyze_viral_content(limit: int = 10, period_days: Optional[int] = None):
    """
    Analizar contenidos más virales.

    Args:
        limit: Límite de contenidos
        period_days: Período en días (opcional)

    Returns:
        Lista de contenidos virales
    """
    try:
        if not editor.virality_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de viralidad no disponible")
        
        result = editor.virality_analyzer.analyze_viral_content(limit, period_days)
        return {"success": True, "viral_content": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/virality/trends")
async def get_sharing_trends(content_id: Optional[str] = None, period_days: int = 30):
    """
    Obtener tendencias de compartir.

    Args:
        content_id: ID del contenido (opcional)
        period_days: Período en días

    Returns:
        Tendencias de compartir
    """
    try:
        if not editor.virality_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de viralidad no disponible")
        
        result = editor.virality_analyzer.get_sharing_trends(content_id, period_days)
        return {"success": True, "trends": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/virality/influencers")
async def get_influencer_analysis(limit: int = 10):
    """
    Obtener análisis de usuarios más influyentes.

    Args:
        limit: Límite de usuarios

    Returns:
        Lista de usuarios influyentes
    """
    try:
        if not editor.virality_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de viralidad no disponible")
        
        result = editor.virality_analyzer.get_influencer_analysis(limit)
        return {"success": True, "influencers": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predictive/record-data")
async def record_historical_data(
    content_id: str,
    metric_name: str,
    value: float,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Registrar datos históricos.

    Args:
        content_id: ID del contenido
        metric_name: Nombre de la métrica
        value: Valor de la métrica
        metadata: Metadatos adicionales

    Returns:
        Resultado
    """
    try:
        if not editor.predictive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador predictivo no disponible")
        
        editor.predictive_analyzer.record_historical_data(content_id, metric_name, value, metadata)
        return {"success": True, "message": "Dato histórico registrado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictive/predict/{content_id}")
async def predict_metric(
    content_id: str,
    metric_name: str,
    days_ahead: int = 7
):
    """
    Predecir métrica futura.

    Args:
        content_id: ID del contenido
        metric_name: Nombre de la métrica
        days_ahead: Días a predecir

    Returns:
        Predicción
    """
    try:
        if not editor.predictive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador predictivo no disponible")
        
        result = editor.predictive_analyzer.predict_metric(content_id, metric_name, days_ahead)
        return {"success": True, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictive/performance/{content_id}")
async def predict_content_performance(
    content_id: str,
    metrics: List[str]
):
    """
    Predecir performance general del contenido.

    Args:
        content_id: ID del contenido
        metrics: Lista de métricas a predecir

    Returns:
        Predicciones de performance
    """
    try:
        if not editor.predictive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador predictivo no disponible")
        
        result = editor.predictive_analyzer.predict_content_performance(content_id, metrics)
        return {"success": True, "performance": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictive/history")
async def get_prediction_history(
    content_id: Optional[str] = None,
    metric_name: Optional[str] = None
):
    """
    Obtener historial de predicciones.

    Args:
        content_id: ID del contenido (opcional)
        metric_name: Nombre de la métrica (opcional)

    Returns:
        Historial de predicciones
    """
    try:
        if not editor.predictive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador predictivo no disponible")
        
        result = editor.predictive_analyzer.get_prediction_history(content_id, metric_name)
        return {"success": True, "history": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multilanguage/detect")
async def detect_language(content: str):
    """
    Detectar idioma del contenido.

    Args:
        content: Contenido

    Returns:
        Detección de idioma
    """
    try:
        if not editor.multilanguage_analyzer:
            raise HTTPException(status_code=503, detail="Analizador multiidioma no disponible")
        
        result = editor.multilanguage_analyzer.detect_language(content)
        return {"success": True, "detection": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multilanguage/analyze")
async def analyze_multilanguage_content(content: str):
    """
    Analizar contenido multiidioma.

    Args:
        content: Contenido

    Returns:
        Análisis multiidioma
    """
    try:
        if not editor.multilanguage_analyzer:
            raise HTTPException(status_code=503, detail="Analizador multiidioma no disponible")
        
        result = editor.multilanguage_analyzer.analyze_multilanguage_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multilanguage/compare")
async def compare_languages(content1: str, content2: str):
    """
    Comparar contenido en diferentes idiomas.

    Args:
        content1: Contenido 1
        content2: Contenido 2

    Returns:
        Comparación de idiomas
    """
    try:
        if not editor.multilanguage_analyzer:
            raise HTTPException(status_code=503, detail="Analizador multiidioma no disponible")
        
        result = editor.multilanguage_analyzer.compare_languages(content1, content2)
        return {"success": True, "comparison": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multilanguage/statistics")
async def get_language_statistics(contents: List[Dict[str, str]]):
    """
    Obtener estadísticas de idiomas de múltiples contenidos.

    Args:
        contents: Lista de contenidos (cada uno con 'id' y 'content')

    Returns:
        Estadísticas de idiomas
    """
    try:
        if not editor.multilanguage_analyzer:
            raise HTTPException(status_code=503, detail="Analizador multiidioma no disponible")
        
        result = editor.multilanguage_analyzer.get_language_statistics(contents)
        return {"success": True, "statistics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generative/analyze")
async def analyze_generative_indicators(content: str):
    """
    Analizar indicadores de contenido generativo.

    Args:
        content: Contenido

    Returns:
        Análisis de indicadores generativos
    """
    try:
        if not editor.generative_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido generativo no disponible")
        
        result = editor.generative_analyzer.analyze_generative_indicators(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generative/compare")
async def compare_with_human_content(
    generative_content: str,
    human_content: str
):
    """
    Comparar contenido generativo con contenido humano.

    Args:
        generative_content: Contenido generativo
        human_content: Contenido humano

    Returns:
        Comparación
    """
    try:
        if not editor.generative_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido generativo no disponible")
        
        result = editor.generative_analyzer.compare_with_human_content(generative_content, human_content)
        return {"success": True, "comparison": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generative/detect-sections")
async def detect_generative_sections(content: str):
    """
    Detectar secciones que parecen generadas.

    Args:
        content: Contenido

    Returns:
        Secciones detectadas
    """
    try:
        if not editor.generative_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido generativo no disponible")
        
        result = editor.generative_analyzer.detect_generative_sections(content)
        return {"success": True, "sections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generative/suggest-improvements")
async def suggest_improvements(content: str):
    """
    Sugerir mejoras para hacer el contenido más natural.

    Args:
        content: Contenido

    Returns:
        Sugerencias de mejora
    """
    try:
        if not editor.generative_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido generativo no disponible")
        
        result = editor.generative_analyzer.suggest_improvements(content)
        return {"success": True, "suggestions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/realtime/analyze-metric")
async def analyze_realtime_metric(
    content_id: str,
    metric_name: str,
    value: float
):
    """
    Analizar métricas en tiempo real.

    Args:
        content_id: ID del contenido
        metric_name: Nombre de la métrica
        value: Valor de la métrica

    Returns:
        Análisis en tiempo real
    """
    try:
        if not editor.realtime_analyzer:
            raise HTTPException(status_code=503, detail="Analizador en tiempo real no disponible")
        
        result = await editor.realtime_analyzer.analyze_realtime_metrics(content_id, metric_name, value)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/realtime/events")
async def get_recent_events(event_type: Optional[str] = None, limit: int = 100):
    """
    Obtener eventos recientes.

    Args:
        event_type: Tipo de evento (opcional)
        limit: Límite de eventos

    Returns:
        Lista de eventos
    """
    try:
        if not editor.realtime_analyzer:
            raise HTTPException(status_code=503, detail="Analizador en tiempo real no disponible")
        
        result = editor.realtime_analyzer.get_recent_events(event_type, limit)
        return {"success": True, "events": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/realtime/stats")
async def get_realtime_stats():
    """
    Obtener estadísticas en tiempo real.

    Returns:
        Estadísticas
    """
    try:
        if not editor.realtime_analyzer:
            raise HTTPException(status_code=503, detail="Analizador en tiempo real no disponible")
        
        result = editor.realtime_analyzer.get_realtime_stats()
        return {"success": True, "stats": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimedia/analyze")
async def analyze_multimedia_content(content: str):
    """
    Analizar contenido multimedia.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido multimedia
    """
    try:
        if not editor.multimedia_analyzer:
            raise HTTPException(status_code=503, detail="Analizador multimedia no disponible")
        
        result = editor.multimedia_analyzer.analyze_multimedia_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimedia/balance")
async def analyze_multimedia_balance(content: str):
    """
    Analizar balance de contenido multimedia.

    Args:
        content: Contenido

    Returns:
        Análisis de balance
    """
    try:
        if not editor.multimedia_analyzer:
            raise HTTPException(status_code=503, detail="Analizador multimedia no disponible")
        
        result = editor.multimedia_analyzer.analyze_multimedia_balance(content)
        return {"success": True, "balance": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimedia/recommendations")
async def get_multimedia_recommendations(content: str):
    """
    Obtener recomendaciones de multimedia.

    Args:
        content: Contenido

    Returns:
        Recomendaciones
    """
    try:
        if not editor.multimedia_analyzer:
            raise HTTPException(status_code=503, detail="Analizador multimedia no disponible")
        
        result = editor.multimedia_analyzer.get_multimedia_recommendations(content)
        return {"success": True, "recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptive/add-rule")
async def add_adaptation_rule(
    condition: str,
    action: str,
    priority: int = 1,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Agregar regla de adaptación.

    Args:
        condition: Condición para la adaptación
        action: Acción a realizar
        priority: Prioridad
        metadata: Metadatos adicionales

    Returns:
        Resultado
    """
    try:
        if not editor.adaptive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador adaptativo no disponible")
        
        editor.adaptive_analyzer.add_adaptation_rule(condition, action, priority, metadata)
        return {"success": True, "message": "Regla de adaptación agregada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptive/analyze-needs")
async def analyze_adaptation_needs(
    content_id: str,
    content_metrics: Dict[str, Any],
    user_context: Optional[Dict[str, Any]] = None
):
    """
    Analizar necesidades de adaptación.

    Args:
        content_id: ID del contenido
        content_metrics: Métricas del contenido
        user_context: Contexto del usuario (opcional)

    Returns:
        Análisis de necesidades de adaptación
    """
    try:
        if not editor.adaptive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador adaptativo no disponible")
        
        result = editor.adaptive_analyzer.analyze_adaptation_needs(content_id, content_metrics, user_context)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptive/suggest-changes")
async def suggest_adaptive_changes(
    content_id: str,
    current_content: str,
    target_audience: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
):
    """
    Sugerir cambios adaptativos.

    Args:
        content_id: ID del contenido
        current_content: Contenido actual
        target_audience: Audiencia objetivo (opcional)
        context: Contexto adicional (opcional)

    Returns:
        Sugerencias de cambios adaptativos
    """
    try:
        if not editor.adaptive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador adaptativo no disponible")
        
        result = editor.adaptive_analyzer.suggest_adaptive_changes(content_id, current_content, target_audience, context)
        return {"success": True, "suggestions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptive/track-performance")
async def track_adaptation_performance(
    content_id: str,
    performance_score: float
):
    """
    Rastrear performance de adaptaciones.

    Args:
        content_id: ID del contenido
        performance_score: Score de performance (0-1)

    Returns:
        Resultado
    """
    try:
        if not editor.adaptive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador adaptativo no disponible")
        
        editor.adaptive_analyzer.track_adaptation_performance(content_id, performance_score)
        return {"success": True, "message": "Performance registrada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adaptive/effectiveness/{content_id}")
async def get_adaptation_effectiveness(content_id: str):
    """
    Obtener efectividad de adaptaciones.

    Args:
        content_id: ID del contenido

    Returns:
        Análisis de efectividad
    """
    try:
        if not editor.adaptive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador adaptativo no disponible")
        
        result = editor.adaptive_analyzer.get_adaptation_effectiveness(content_id)
        return {"success": True, "effectiveness": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interactive/analyze")
async def analyze_interactive_elements(content: str):
    """
    Analizar elementos interactivos en el contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de elementos interactivos
    """
    try:
        if not editor.interactive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido interactivo no disponible")
        
        result = editor.interactive_analyzer.analyze_interactive_elements(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interactive/engagement")
async def analyze_user_engagement_potential(content: str):
    """
    Analizar potencial de engagement del usuario.

    Args:
        content: Contenido

    Returns:
        Análisis de potencial de engagement
    """
    try:
        if not editor.interactive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido interactivo no disponible")
        
        result = editor.interactive_analyzer.analyze_user_engagement_potential(content)
        return {"success": True, "engagement": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/contextual/analyze")
async def analyze_context(
    content: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Analizar contexto del contenido.

    Args:
        content: Contenido
        metadata: Metadatos adicionales (opcional)

    Returns:
        Análisis contextual
    """
    try:
        if not editor.contextual_analyzer:
            raise HTTPException(status_code=503, detail="Analizador contextual no disponible")
        
        result = editor.contextual_analyzer.analyze_context(content, metadata)
        return {"success": True, "context": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/contextual/relevance")
async def analyze_contextual_relevance(
    content: str,
    target_context: Dict[str, Any]
):
    """
    Analizar relevancia contextual del contenido.

    Args:
        content: Contenido
        target_context: Contexto objetivo

    Returns:
        Análisis de relevancia contextual
    """
    try:
        if not editor.contextual_analyzer:
            raise HTTPException(status_code=503, detail="Analizador contextual no disponible")
        
        result = editor.contextual_analyzer.analyze_contextual_relevance(content, target_context)
        return {"success": True, "relevance": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/narrative/analyze-structure")
async def analyze_narrative_structure(content: str):
    """
    Analizar estructura narrativa del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de estructura narrativa
    """
    try:
        if not editor.narrative_analyzer:
            raise HTTPException(status_code=503, detail="Analizador narrativo no disponible")
        
        result = editor.narrative_analyzer.analyze_narrative_structure(content)
        return {"success": True, "structure": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/narrative/analyze-flow")
async def analyze_story_flow(content: str):
    """
    Analizar flujo narrativo del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de flujo narrativo
    """
    try:
        if not editor.narrative_analyzer:
            raise HTTPException(status_code=503, detail="Analizador narrativo no disponible")
        
        result = editor.narrative_analyzer.analyze_story_flow(content)
        return {"success": True, "flow": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/narrative/suggest-improvements")
async def suggest_narrative_improvements(content: str):
    """
    Sugerir mejoras narrativas.

    Args:
        content: Contenido

    Returns:
        Sugerencias de mejora narrativa
    """
    try:
        if not editor.narrative_analyzer:
            raise HTTPException(status_code=503, detail="Analizador narrativo no disponible")
        
        result = editor.narrative_analyzer.suggest_narrative_improvements(content)
        return {"success": True, "suggestions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emotional/analyze")
async def analyze_emotional_content(content: str):
    """
    Analizar contenido emocional.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido emocional
    """
    try:
        if not editor.emotional_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido emocional no disponible")
        
        result = editor.emotional_analyzer.analyze_emotional_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emotional/analyze-arc")
async def analyze_emotional_arc(content: str):
    """
    Analizar arco emocional del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de arco emocional
    """
    try:
        if not editor.emotional_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido emocional no disponible")
        
        result = editor.emotional_analyzer.analyze_emotional_arc(content)
        return {"success": True, "arc": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/persuasive/analyze-elements")
async def analyze_persuasive_elements(content: str):
    """
    Analizar elementos persuasivos en el contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de elementos persuasivos
    """
    try:
        if not editor.persuasive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido persuasivo no disponible")
        
        result = editor.persuasive_analyzer.analyze_persuasive_elements(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/persuasive/analyze-strength")
async def analyze_persuasion_strength(content: str):
    """
    Analizar fuerza persuasiva del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de fuerza persuasiva
    """
    try:
        if not editor.persuasive_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido persuasivo no disponible")
        
        result = editor.persuasive_analyzer.analyze_persuasion_strength(content)
        return {"success": True, "strength": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/educational/analyze-structure")
async def analyze_educational_structure(content: str):
    """
    Analizar estructura educativa del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de estructura educativa
    """
    try:
        if not editor.educational_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido educativo no disponible")
        
        result = editor.educational_analyzer.analyze_educational_structure(content)
        return {"success": True, "structure": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/educational/analyze-objectives")
async def analyze_learning_objectives(content: str):
    """
    Analizar objetivos de aprendizaje en el contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de objetivos de aprendizaje
    """
    try:
        if not editor.educational_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido educativo no disponible")
        
        result = editor.educational_analyzer.analyze_learning_objectives(content)
        return {"success": True, "objectives": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/educational/suggest-improvements")
async def suggest_educational_improvements(content: str):
    """
    Sugerir mejoras educativas.

    Args:
        content: Contenido

    Returns:
        Sugerencias de mejora educativa
    """
    try:
        if not editor.educational_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido educativo no disponible")
        
        result = editor.educational_analyzer.suggest_educational_improvements(content)
        return {"success": True, "suggestions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/technical/analyze")
async def analyze_technical_content(content: str):
    """
    Analizar contenido técnico.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido técnico
    """
    try:
        if not editor.technical_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido técnico no disponible")
        
        result = editor.technical_analyzer.analyze_technical_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/technical/analyze-complexity")
async def analyze_technical_complexity(content: str):
    """
    Analizar complejidad técnica del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de complejidad técnica
    """
    try:
        if not editor.technical_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido técnico no disponible")
        
        result = editor.technical_analyzer.analyze_technical_complexity(content)
        return {"success": True, "complexity": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/creative/analyze-elements")
async def analyze_creative_elements(content: str):
    """
    Analizar elementos creativos en el contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de elementos creativos
    """
    try:
        if not editor.creative_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido creativo no disponible")
        
        result = editor.creative_analyzer.analyze_creative_elements(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/creative/analyze-level")
async def analyze_creativity_level(content: str):
    """
    Analizar nivel de creatividad del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de nivel de creatividad
    """
    try:
        if not editor.creative_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido creativo no disponible")
        
        result = editor.creative_analyzer.analyze_creativity_level(content)
        return {"success": True, "creativity": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scientific/analyze")
async def analyze_scientific_content(content: str):
    """
    Analizar contenido científico.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido científico
    """
    try:
        if not editor.scientific_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido científico no disponible")
        
        result = editor.scientific_analyzer.analyze_scientific_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scientific/analyze-rigor")
async def analyze_scientific_rigor(content: str):
    """
    Analizar rigor científico del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de rigor científico
    """
    try:
        if not editor.scientific_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido científico no disponible")
        
        result = editor.scientific_analyzer.analyze_scientific_rigor(content)
        return {"success": True, "rigor": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/legal/analyze")
async def analyze_legal_content(content: str):
    """
    Analizar contenido legal.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido legal
    """
    try:
        if not editor.legal_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido legal no disponible")
        
        result = editor.legal_analyzer.analyze_legal_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/legal/analyze-structure")
async def analyze_legal_structure(content: str):
    """
    Analizar estructura legal del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de estructura legal
    """
    try:
        if not editor.legal_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido legal no disponible")
        
        result = editor.legal_analyzer.analyze_legal_structure(content)
        return {"success": True, "structure": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/financial/analyze")
async def analyze_financial_content(content: str):
    """
    Analizar contenido financiero.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido financiero
    """
    try:
        if not editor.financial_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido financiero no disponible")
        
        result = editor.financial_analyzer.analyze_financial_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/financial/analyze-accuracy")
async def analyze_financial_accuracy(content: str):
    """
    Analizar precisión financiera del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de precisión financiera
    """
    try:
        if not editor.financial_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido financiero no disponible")
        
        result = editor.financial_analyzer.analyze_financial_accuracy(content)
        return {"success": True, "accuracy": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/journalistic/analyze")
async def analyze_journalistic_content(content: str):
    """
    Analizar contenido periodístico.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido periodístico
    """
    try:
        if not editor.journalistic_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido periodístico no disponible")
        
        result = editor.journalistic_analyzer.analyze_journalistic_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/journalistic/analyze-quality")
async def analyze_journalistic_quality(content: str):
    """
    Analizar calidad periodística del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de calidad periodística
    """
    try:
        if not editor.journalistic_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido periodístico no disponible")
        
        result = editor.journalistic_analyzer.analyze_journalistic_quality(content)
        return {"success": True, "quality": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/medical/analyze")
async def analyze_medical_content(content: str):
    """
    Analizar contenido médico.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido médico
    """
    try:
        if not editor.medical_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido médico no disponible")
        
        result = editor.medical_analyzer.analyze_medical_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/medical/analyze-safety")
async def analyze_medical_safety(content: str):
    """
    Analizar seguridad médica del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de seguridad médica
    """
    try:
        if not editor.medical_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido médico no disponible")
        
        result = editor.medical_analyzer.analyze_medical_safety(content)
        return {"success": True, "safety": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/marketing/analyze")
async def analyze_marketing_content(content: str):
    """
    Analizar contenido de marketing.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de marketing
    """
    try:
        if not editor.marketing_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de marketing no disponible")
        
        result = editor.marketing_analyzer.analyze_marketing_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/marketing/analyze-effectiveness")
async def analyze_marketing_effectiveness(content: str):
    """
    Analizar efectividad de marketing del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de efectividad de marketing
    """
    try:
        if not editor.marketing_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de marketing no disponible")
        
        result = editor.marketing_analyzer.analyze_marketing_effectiveness(content)
        return {"success": True, "effectiveness": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sales/analyze")
async def analyze_sales_content(content: str):
    """
    Analizar contenido de ventas.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de ventas
    """
    try:
        if not editor.sales_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de ventas no disponible")
        
        result = editor.sales_analyzer.analyze_sales_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sales/analyze-potential")
async def analyze_sales_potential(content: str):
    """
    Analizar potencial de ventas del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de potencial de ventas
    """
    try:
        if not editor.sales_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de ventas no disponible")
        
        result = editor.sales_analyzer.analyze_sales_potential(content)
        return {"success": True, "potential": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hr/analyze")
async def analyze_hr_content(content: str):
    """
    Analizar contenido de recursos humanos.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de HR
    """
    try:
        if not editor.hr_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de recursos humanos no disponible")
        
        result = editor.hr_analyzer.analyze_hr_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hr/analyze-completeness")
async def analyze_hr_completeness(content: str):
    """
    Analizar completitud del contenido de HR.

    Args:
        content: Contenido

    Returns:
        Análisis de completitud de HR
    """
    try:
        if not editor.hr_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de recursos humanos no disponible")
        
        result = editor.hr_analyzer.analyze_hr_completeness(content)
        return {"success": True, "completeness": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/support/analyze")
async def analyze_support_content(content: str):
    """
    Analizar contenido de soporte técnico.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de soporte
    """
    try:
        if not editor.support_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de soporte técnico no disponible")
        
        result = editor.support_analyzer.analyze_support_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/support/analyze-quality")
async def analyze_support_quality(content: str):
    """
    Analizar calidad del contenido de soporte.

    Args:
        content: Contenido

    Returns:
        Análisis de calidad de soporte
    """
    try:
        if not editor.support_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de soporte técnico no disponible")
        
        result = editor.support_analyzer.analyze_support_quality(content)
        return {"success": True, "quality": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documentation/analyze")
async def analyze_documentation_content(content: str):
    """
    Analizar contenido de documentación técnica.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de documentación
    """
    try:
        if not editor.documentation_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de documentación técnica no disponible")
        
        result = editor.documentation_analyzer.analyze_documentation_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documentation/analyze-structure")
async def analyze_documentation_structure(content: str):
    """
    Analizar estructura de la documentación.

    Args:
        content: Contenido

    Returns:
        Análisis de estructura de documentación
    """
    try:
        if not editor.documentation_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de documentación técnica no disponible")
        
        result = editor.documentation_analyzer.analyze_documentation_structure(content)
        return {"success": True, "structure": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/blog/analyze")
async def analyze_blog_content(content: str):
    """
    Analizar contenido de blog.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de blog
    """
    try:
        if not editor.blog_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de blog no disponible")
        
        result = editor.blog_analyzer.analyze_blog_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/blog/analyze-engagement")
async def analyze_blog_engagement(content: str):
    """
    Analizar potencial de engagement del blog.

    Args:
        content: Contenido

    Returns:
        Análisis de engagement del blog
    """
    try:
        if not editor.blog_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de blog no disponible")
        
        result = editor.blog_analyzer.analyze_blog_engagement(content)
        return {"success": True, "engagement": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/email-marketing/analyze")
async def analyze_email_content(content: str):
    """
    Analizar contenido de email marketing.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de email
    """
    try:
        if not editor.email_marketing_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de email marketing no disponible")
        
        result = editor.email_marketing_analyzer.analyze_email_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/email-marketing/analyze-effectiveness")
async def analyze_email_effectiveness(content: str):
    """
    Analizar efectividad del email marketing.

    Args:
        content: Contenido

    Returns:
        Análisis de efectividad del email
    """
    try:
        if not editor.email_marketing_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de email marketing no disponible")
        
        result = editor.email_marketing_analyzer.analyze_email_effectiveness(content)
        return {"success": True, "effectiveness": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/social-media/analyze")
async def analyze_social_content(content: str):
    """
    Analizar contenido de redes sociales.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de redes sociales
    """
    try:
        if not editor.social_media_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de redes sociales no disponible")
        
        result = editor.social_media_analyzer.analyze_social_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/social-media/analyze-virality")
async def analyze_social_virality(content: str):
    """
    Analizar potencial de viralidad del contenido.

    Args:
        content: Contenido

    Returns:
        Análisis de viralidad
    """
    try:
        if not editor.social_media_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de redes sociales no disponible")
        
        result = editor.social_media_analyzer.analyze_social_virality(content)
        return {"success": True, "virality": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/elearning/analyze")
async def analyze_elearning_content(content: str):
    """
    Analizar contenido de e-learning.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de e-learning
    """
    try:
        if not editor.elearning_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de e-learning no disponible")
        
        result = editor.elearning_analyzer.analyze_elearning_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/elearning/analyze-quality")
async def analyze_elearning_quality(content: str):
    """
    Analizar calidad del contenido de e-learning.

    Args:
        content: Contenido

    Returns:
        Análisis de calidad de e-learning
    """
    try:
        if not editor.elearning_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de e-learning no disponible")
        
        result = editor.elearning_analyzer.analyze_elearning_quality(content)
        return {"success": True, "quality": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/podcast/analyze")
async def analyze_podcast_content(content: str):
    """
    Analizar contenido de podcast.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de podcast
    """
    try:
        if not editor.podcast_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de podcast no disponible")
        
        result = editor.podcast_analyzer.analyze_podcast_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/podcast/analyze-structure")
async def analyze_podcast_structure(content: str):
    """
    Analizar estructura del podcast.

    Args:
        content: Contenido

    Returns:
        Análisis de estructura del podcast
    """
    try:
        if not editor.podcast_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de podcast no disponible")
        
        result = editor.podcast_analyzer.analyze_podcast_structure(content)
        return {"success": True, "structure": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/analyze")
async def analyze_video_content(content: str):
    """
    Analizar contenido de video.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de video
    """
    try:
        if not editor.video_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de video no disponible")
        
        result = editor.video_analyzer.analyze_video_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/analyze-optimization")
async def analyze_video_optimization(content: str):
    """
    Analizar optimización del contenido de video.

    Args:
        content: Contenido

    Returns:
        Análisis de optimización del video
    """
    try:
        if not editor.video_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de video no disponible")
        
        result = editor.video_analyzer.analyze_video_optimization(content)
        return {"success": True, "optimization": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/news/analyze")
async def analyze_news_content(content: str):
    """
    Analizar contenido de noticias.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de noticias
    """
    try:
        if not editor.news_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de noticias no disponible")
        
        result = editor.news_analyzer.analyze_news_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/news/analyze-credibility")
async def analyze_news_credibility(content: str):
    """
    Analizar credibilidad del contenido de noticias.

    Args:
        content: Contenido

    Returns:
        Análisis de credibilidad
    """
    try:
        if not editor.news_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de noticias no disponible")
        
        result = editor.news_analyzer.analyze_news_credibility(content)
        return {"success": True, "credibility": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review/analyze")
async def analyze_review_content(content: str):
    """
    Analizar contenido de reseñas.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de reseñas
    """
    try:
        if not editor.review_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de reseñas no disponible")
        
        result = editor.review_analyzer.analyze_review_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review/analyze-helpfulness")
async def analyze_review_helpfulness(content: str):
    """
    Analizar utilidad de la reseña.

    Args:
        content: Contenido

    Returns:
        Análisis de utilidad de la reseña
    """
    try:
        if not editor.review_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de reseñas no disponible")
        
        result = editor.review_analyzer.analyze_review_helpfulness(content)
        return {"success": True, "helpfulness": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/landing-page/analyze")
async def analyze_landing_content(content: str):
    """
    Analizar contenido de landing page.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de landing page
    """
    try:
        if not editor.landing_page_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de landing pages no disponible")
        
        result = editor.landing_page_analyzer.analyze_landing_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/landing-page/analyze-conversion")
async def analyze_landing_conversion(content: str):
    """
    Analizar potencial de conversión de la landing page.

    Args:
        content: Contenido

    Returns:
        Análisis de conversión
    """
    try:
        if not editor.landing_page_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de landing pages no disponible")
        
        result = editor.landing_page_analyzer.analyze_landing_conversion(content)
        return {"success": True, "conversion": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/faq/analyze")
async def analyze_faq_content(content: str):
    """
    Analizar contenido de FAQ.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de FAQ
    """
    try:
        if not editor.faq_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de FAQ no disponible")
        
        result = editor.faq_analyzer.analyze_faq_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/faq/analyze-completeness")
async def analyze_faq_completeness(content: str):
    """
    Analizar completitud del FAQ.

    Args:
        content: Contenido

    Returns:
        Análisis de completitud del FAQ
    """
    try:
        if not editor.faq_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de FAQ no disponible")
        
        result = editor.faq_analyzer.analyze_faq_completeness(content)
        return {"success": True, "completeness": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/newsletter/analyze")
async def analyze_newsletter_content(content: str):
    """
    Analizar contenido de newsletter.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de newsletter
    """
    try:
        if not editor.newsletter_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de newsletters no disponible")
        
        result = editor.newsletter_analyzer.analyze_newsletter_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/newsletter/analyze-effectiveness")
async def analyze_newsletter_effectiveness(content: str):
    """
    Analizar efectividad del newsletter.

    Args:
        content: Contenido

    Returns:
        Análisis de efectividad del newsletter
    """
    try:
        if not editor.newsletter_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de newsletters no disponible")
        
        result = editor.newsletter_analyzer.analyze_newsletter_effectiveness(content)
        return {"success": True, "effectiveness": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whitepaper/analyze")
async def analyze_whitepaper_content(content: str):
    """
    Analizar contenido de whitepaper.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de whitepaper
    """
    try:
        if not editor.whitepaper_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de whitepapers no disponible")
        
        result = editor.whitepaper_analyzer.analyze_whitepaper_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whitepaper/analyze-quality")
async def analyze_whitepaper_quality(content: str):
    """
    Analizar calidad del whitepaper.

    Args:
        content: Contenido

    Returns:
        Análisis de calidad del whitepaper
    """
    try:
        if not editor.whitepaper_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de whitepapers no disponible")
        
        result = editor.whitepaper_analyzer.analyze_whitepaper_quality(content)
        return {"success": True, "quality": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/case-study/analyze")
async def analyze_case_study_content(content: str):
    """
    Analizar contenido de caso de estudio.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de caso de estudio
    """
    try:
        if not editor.case_study_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de casos de estudio no disponible")
        
        result = editor.case_study_analyzer.analyze_case_study_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/case-study/analyze-structure")
async def analyze_case_study_structure(content: str):
    """
    Analizar estructura del caso de estudio.

    Args:
        content: Contenido

    Returns:
        Análisis de estructura del caso de estudio
    """
    try:
        if not editor.case_study_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de casos de estudio no disponible")
        
        result = editor.case_study_analyzer.analyze_case_study_structure(content)
        return {"success": True, "structure": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proposal/analyze")
async def analyze_proposal_content(content: str):
    """
    Analizar contenido de propuesta.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de propuesta
    """
    try:
        if not editor.proposal_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de propuestas no disponible")
        
        result = editor.proposal_analyzer.analyze_proposal_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proposal/analyze-completeness")
async def analyze_proposal_completeness(content: str):
    """
    Analizar completitud de la propuesta.

    Args:
        content: Contenido

    Returns:
        Análisis de completitud de la propuesta
    """
    try:
        if not editor.proposal_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de propuestas no disponible")
        
        result = editor.proposal_analyzer.analyze_proposal_completeness(content)
        return {"success": True, "completeness": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/analyze")
async def analyze_report_content(content: str):
    """
    Analizar contenido de informe.

    Args:
        content: Contenido

    Returns:
        Análisis de contenido de informe
    """
    try:
        if not editor.report_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de informes no disponible")
        
        result = editor.report_analyzer.analyze_report_content(content)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/analyze-quality")
async def analyze_report_quality(content: str):
    """
    Analizar calidad del informe.

    Args:
        content: Contenido

    Returns:
        Análisis de calidad del informe
    """
    try:
        if not editor.report_analyzer:
            raise HTTPException(status_code=503, detail="Analizador de contenido de informes no disponible")
        
        result = editor.report_analyzer.analyze_report_quality(content)
        return {"success": True, "quality": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
