"""
Gradio Interface - Interfaz interactiva con Gradio
====================================================

Interfaz web interactiva usando Gradio para demostrar capacidades del agente.
Proporciona una interfaz visual para controlar el agente, ejecutar comandos,
y usar funcionalidades de IA.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
import asyncio

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None  # type: ignore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .agent import CursorAgent


class GradioInterface:
    """
    Interfaz Gradio para el agente.
    
    Proporciona una interfaz web interactiva para:
    - Controlar el agente (iniciar, detener, pausar, reanudar)
    - Ejecutar comandos
    - Procesar con IA
    - Búsqueda semántica
    - Ver estadísticas
    """
    
    def __init__(self, agent: Optional["CursorAgent"] = None) -> None:
        """
        Inicializar interfaz Gradio.
        
        Args:
            agent: Instancia del agente a controlar (opcional).
        """
        if not GRADIO_AVAILABLE:
            raise ImportError(
                "Gradio is not available. Install it with: pip install gradio"
            )
        
        self.agent: Optional["CursorAgent"] = agent
        self.interface: Optional[gr.Blocks] = None
    
    def create_interface(self) -> gr.Blocks:
        """
        Crear interfaz Gradio.
        
        Returns:
            Instancia de gr.Blocks con la interfaz configurada.
        
        Raises:
            RuntimeError: Si Gradio no está disponible.
        """
        if not GRADIO_AVAILABLE:
            raise RuntimeError("Gradio is not available")
        
        with gr.Blocks(title="Cursor Agent 24/7", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Cursor Agent 24/7 - Interfaz Interactiva")
            
            with gr.Tabs():
                # Tab 1: Control del Agente
                with gr.Tab("Control"):
                    with gr.Row():
                        status_display = gr.Textbox(
                            label="Estado del Agente",
                            value="Desconocido",
                            interactive=False
                        )
                    
                    with gr.Row():
                        start_btn = gr.Button("▶️ Iniciar", variant="primary")
                        stop_btn = gr.Button("⏸️ Detener", variant="stop")
                        pause_btn = gr.Button("⏸️ Pausar")
                        resume_btn = gr.Button("▶️ Reanudar")
                    
                    with gr.Row():
                        status_output = gr.Textbox(
                            label="Respuesta",
                            lines=3,
                            interactive=False
                        )
                    
                    def update_status() -> str:
                        """Actualizar estado del agente"""
                        try:
                            if self.agent:
                                status = self.agent.get_status()
                                return (
                                    f"Estado: {status.get('status', 'unknown')}\n"
                                    f"Tareas: {status.get('tasks_total', 0)}"
                                )
                            return "Agente no disponible"
                        except Exception as e:
                            logger.error(f"Error updating status: {e}", exc_info=True)
                            return f"Error: {str(e)}"
                    
                    start_btn.click(
                        self._start_agent,
                        outputs=status_output
                    )
                    stop_btn.click(
                        self._stop_agent,
                        outputs=status_output
                    )
                    pause_btn.click(
                        self._pause_agent,
                        outputs=status_output
                    )
                    resume_btn.click(
                        self._resume_agent,
                        outputs=status_output
                    )
                
                # Tab 2: Ejecutar Comandos
                with gr.Tab("Comandos"):
                    command_input = gr.Textbox(
                        label="Comando",
                        placeholder="Escribe tu comando aquí...",
                        lines=3
                    )
                    
                    with gr.Row():
                        execute_btn = gr.Button("🚀 Ejecutar", variant="primary")
                        clear_btn = gr.Button("🗑️ Limpiar")
                    
                    command_output = gr.Textbox(
                        label="Resultado",
                        lines=10,
                        interactive=False
                    )
                    
                    execute_btn.click(
                        self._execute_command,
                        inputs=command_input,
                        outputs=command_output
                    )
                    clear_btn.click(
                        lambda: ("", ""),
                        outputs=[command_input, command_output]
                    )
                
                # Tab 3: Procesamiento con IA
                with gr.Tab("IA"):
                    ai_input = gr.Textbox(
                        label="Texto a procesar",
                        placeholder="Escribe texto para procesar con IA...",
                        lines=5
                    )
                    
                    ai_mode = gr.Radio(
                        choices=[
                            "Procesar",
                            "Generar Código",
                            "Explicar Código",
                            "Corregir Código",
                            "Resumir"
                        ],
                        value="Procesar",
                        label="Modo"
                    )
                    
                    ai_btn = gr.Button("🤖 Procesar con IA", variant="primary")
                    ai_output = gr.Textbox(
                        label="Resultado",
                        lines=10,
                        interactive=False
                    )
                    
                    ai_btn.click(
                        self._process_with_ai,
                        inputs=[ai_input, ai_mode],
                        outputs=ai_output
                    )
                
                # Tab 4: Búsqueda Semántica
                with gr.Tab("Búsqueda"):
                    search_query = gr.Textbox(
                        label="Consulta",
                        placeholder="Buscar comandos similares...",
                        lines=2
                    )
                    
                    search_btn = gr.Button("🔍 Buscar", variant="primary")
                    search_results = gr.Dataframe(
                        label="Resultados",
                        headers=["Comando", "Similitud", "Metadata"],
                        interactive=False
                    )
                    
                    search_btn.click(
                        self._search_embeddings,
                        inputs=search_query,
                        outputs=search_results
                    )
                
                # Tab 5: Estadísticas
                with gr.Tab("Estadísticas"):
                    stats_btn = gr.Button(
                        "📊 Actualizar Estadísticas",
                        variant="primary"
                    )
                    
                    with gr.Row():
                        pattern_stats = gr.JSON(label="Estadísticas de Patrones")
                        metrics_display = gr.JSON(label="Métricas")
                    
                    stats_btn.click(
                        self._get_statistics,
                        outputs=[pattern_stats, metrics_display]
                    )
            
            # Auto-refresh status
            interface.load(
                update_status,
                outputs=status_display,
                every=2
            )
        
        self.interface = interface
        return interface
    
    def _start_agent(self) -> str:
        """
        Iniciar agente.
        
        Returns:
            Mensaje de resultado.
        """
        try:
            if not self.agent:
                return "❌ Agente no disponible"
            
            # Ejecutar de forma segura
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Si hay un loop corriendo, crear tarea
                    asyncio.create_task(self.agent.start())
                    return "✅ Agente iniciando..."
                else:
                    loop.run_until_complete(self.agent.start())
                    return "✅ Agente iniciado exitosamente"
            except RuntimeError:
                # No hay loop, crear uno nuevo
                asyncio.run(self.agent.start())
                return "✅ Agente iniciado exitosamente"
        
        except Exception as e:
            logger.error(f"Error starting agent: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def _stop_agent(self) -> str:
        """
        Detener agente.
        
        Returns:
            Mensaje de resultado.
        """
        try:
            if not self.agent:
                return "❌ Agente no disponible"
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.agent.stop())
                    return "⏹️ Agente deteniendo..."
                else:
                    loop.run_until_complete(self.agent.stop())
                    return "⏹️ Agente detenido"
            except RuntimeError:
                asyncio.run(self.agent.stop())
                return "⏹️ Agente detenido"
        
        except Exception as e:
            logger.error(f"Error stopping agent: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def _pause_agent(self) -> str:
        """
        Pausar agente.
        
        Returns:
            Mensaje de resultado.
        """
        try:
            if not self.agent:
                return "❌ Agente no disponible"
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.agent.pause())
                    return "⏸️ Agente pausando..."
                else:
                    loop.run_until_complete(self.agent.pause())
                    return "⏸️ Agente pausado"
            except RuntimeError:
                asyncio.run(self.agent.pause())
                return "⏸️ Agente pausado"
        
        except Exception as e:
            logger.error(f"Error pausing agent: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def _resume_agent(self) -> str:
        """
        Reanudar agente.
        
        Returns:
            Mensaje de resultado.
        """
        try:
            if not self.agent:
                return "❌ Agente no disponible"
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.agent.resume())
                    return "▶️ Agente reanudando..."
                else:
                    loop.run_until_complete(self.agent.resume())
                    return "▶️ Agente reanudado"
            except RuntimeError:
                asyncio.run(self.agent.resume())
                return "▶️ Agente reanudado"
        
        except Exception as e:
            logger.error(f"Error resuming agent: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def _execute_command(self, command: str) -> str:
        """
        Ejecutar comando.
        
        Args:
            command: Comando a ejecutar.
        
        Returns:
            Mensaje de resultado.
        """
        try:
            if not self.agent:
                return "❌ Agente no disponible"
            
            if not command or not command.strip():
                return "❌ Comando vacío"
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Crear tarea si hay loop corriendo
                    task = asyncio.create_task(
                        self.agent.add_task(command.strip())
                    )
                    return f"✅ Tarea agregada (verificando...)"
                else:
                    task_id = loop.run_until_complete(
                        self.agent.add_task(command.strip())
                    )
                    return f"✅ Tarea agregada: {task_id}"
            except RuntimeError:
                task_id = asyncio.run(self.agent.add_task(command.strip()))
                return f"✅ Tarea agregada: {task_id}"
        
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def _process_with_ai(
        self,
        text: str,
        mode: str
    ) -> str:
        """
        Procesar con IA.
        
        Args:
            text: Texto a procesar.
            mode: Modo de procesamiento.
        
        Returns:
            Resultado del procesamiento.
        """
        try:
            if not self.agent or not self.agent.ai_processor:
                return "❌ Procesador de IA no disponible"
            
            if not text or not text.strip():
                return "❌ Texto vacío"
            
            try:
                loop = asyncio.get_event_loop()
                run_async = loop.is_running()
            except RuntimeError:
                run_async = False
            
            if mode == "Procesar":
                try:
                    if run_async:
                        # No podemos usar asyncio.run si hay loop corriendo
                        return "⚠️ Modo async no soportado en este contexto"
                    processed = asyncio.run(
                        self.agent.ai_processor.process_command(text)
                    )
                    return (
                        f"Intención: {processed.intent.value}\n"
                        f"Confianza: {processed.confidence:.2f}\n"
                        f"Código extraído: {processed.extracted_code or 'N/A'}"
                    )
                except Exception as e:
                    logger.error(f"Error processing command: {e}", exc_info=True)
                    return f"❌ Error: {str(e)}"
            
            elif mode == "Generar Código":
                try:
                    if run_async:
                        return "⚠️ Modo async no soportado en este contexto"
                    code = asyncio.run(
                        self.agent.ai_processor.generate_code(text)
                    )
                    return code or "No se pudo generar código"
                except Exception as e:
                    logger.error(f"Error generating code: {e}", exc_info=True)
                    return f"❌ Error: {str(e)}"
            
            elif mode == "Explicar Código":
                # Acceder al modelo de forma segura
                try:
                    if hasattr(self.agent.ai_processor, '_model') and self.agent.ai_processor._model:
                        explanation = self.agent.ai_processor._model.explain_code(text)
                        return explanation
                    return "❌ Modelo no disponible"
                except Exception as e:
                    logger.error(f"Error explaining code: {e}", exc_info=True)
                    return f"❌ Error: {str(e)}"
            
            elif mode == "Corregir Código":
                try:
                    if hasattr(self.agent.ai_processor, '_model') and self.agent.ai_processor._model:
                        fixed = self.agent.ai_processor._model.fix_code(text)
                        return fixed
                    return "❌ Modelo no disponible"
                except Exception as e:
                    logger.error(f"Error fixing code: {e}", exc_info=True)
                    return f"❌ Error: {str(e)}"
            
            elif mode == "Resumir":
                try:
                    if run_async:
                        return "⚠️ Modo async no soportado en este contexto"
                    summary = asyncio.run(
                        self.agent.ai_processor.summarize_result(text)
                    )
                    return summary
                except Exception as e:
                    logger.error(f"Error summarizing: {e}", exc_info=True)
                    return f"❌ Error: {str(e)}"
            
            return "❌ Modo no reconocido"
        
        except Exception as e:
            logger.error(f"Error in AI processing: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def _search_embeddings(self, query: str) -> List[List[str]]:
        """
        Buscar en embeddings.
        
        Args:
            query: Consulta de búsqueda.
        
        Returns:
            Lista de resultados con formato [comando, similitud, metadata].
        """
        try:
            if not self.agent or not self.agent.embedding_store:
                return []
            
            if not query or not query.strip():
                return []
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # No podemos usar asyncio.run si hay loop corriendo
                    logger.warning("Cannot run async search in running event loop")
                    return []
                
                results = asyncio.run(
                    self.agent.embedding_store.search(query, top_k=10)
                )
            except RuntimeError:
                results = asyncio.run(
                    self.agent.embedding_store.search(query, top_k=10)
                )
            
            return [
                [
                    metadata.get("text", key) if isinstance(metadata, dict) else str(key),
                    f"{similarity:.3f}",
                    str(metadata)
                ]
                for key, similarity, metadata in results
            ]
        
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return []
    
    def _get_statistics(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Obtener estadísticas.
        
        Returns:
            Tupla con (estadísticas de patrones, métricas).
        """
        try:
            pattern_stats: Dict[str, Any] = {}
            metrics: Dict[str, Any] = {}
            
            if not self.agent:
                return pattern_stats, metrics
            
            # Obtener estadísticas de patrones
            if self.agent.pattern_learner:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        logger.warning("Cannot run async get_statistics in running event loop")
                    else:
                        pattern_stats = loop.run_until_complete(
                            self.agent.pattern_learner.get_statistics()
                        )
                except RuntimeError:
                    pattern_stats = asyncio.run(
                        self.agent.pattern_learner.get_statistics()
                    )
            
            # Obtener métricas
            if self.agent.metrics:
                try:
                    metrics = self.agent.metrics.get_summary()
                except Exception as e:
                    logger.error(f"Error getting metrics: {e}", exc_info=True)
            
            return pattern_stats, metrics
        
        except Exception as e:
            logger.error(f"Stats error: {e}", exc_info=True)
            return {}, {}
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860
    ) -> None:
        """
        Lanzar interfaz.
        
        Args:
            share: Si crear link público (default: False).
            server_name: Host del servidor (default: "0.0.0.0").
            server_port: Puerto del servidor (default: 7860).
        
        Raises:
            RuntimeError: Si la interfaz no está creada o Gradio no está disponible.
            ValueError: Si el puerto es inválido.
        """
        if not GRADIO_AVAILABLE:
            raise RuntimeError("Gradio is not available")
        
        if not self.interface:
            self.create_interface()
        
        if not (1 <= server_port <= 65535):
            raise ValueError(f"Invalid port: {server_port}. Must be between 1 and 65535")
        
        if self.interface:
            self.interface.launch(
                share=share,
                server_name=server_name,
                server_port=server_port
            )
