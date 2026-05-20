"""
Gradio Interface - Interfaz interactiva con Gradio
====================================================

Interfaz web interactiva usando Gradio para demostrar capacidades del agente.
"""

import logging
from typing import Optional, Dict, Any
import gradio as gr

logger = logging.getLogger(__name__)


class GradioInterface:
    """Interfaz Gradio para el agente"""
    
    def __init__(self, agent=None):
        self.agent = agent
        self.interface = None
    
    def create_interface(self) -> gr.Blocks:
        """Crear interfaz Gradio"""
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
                    
                    def update_status():
                        if self.agent:
                            status = self.agent.get_status()
                            return f"Estado: {status.get('status', 'unknown')}\nTareas: {status.get('tasks_total', 0)}"
                        return "Agente no disponible"
                    
                    start_btn.click(
                        lambda: self._start_agent(),
                        outputs=status_output
                    )
                    stop_btn.click(
                        lambda: self._stop_agent(),
                        outputs=status_output
                    )
                    pause_btn.click(
                        lambda: self._pause_agent(),
                        outputs=status_output
                    )
                    resume_btn.click(
                        lambda: self._resume_agent(),
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
                        lambda cmd: self._execute_command(cmd),
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
                        choices=["Procesar", "Generar Código", "Explicar Código", "Corregir Código", "Resumir"],
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
                        lambda text, mode: self._process_with_ai(text, mode),
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
                        lambda q: self._search_embeddings(q),
                        inputs=search_query,
                        outputs=search_results
                    )
                
                # Tab 5: Estadísticas
                with gr.Tab("Estadísticas"):
                    stats_btn = gr.Button("📊 Actualizar Estadísticas", variant="primary")
                    
                    with gr.Row():
                        pattern_stats = gr.JSON(label="Estadísticas de Patrones")
                        metrics_display = gr.JSON(label="Métricas")
                    
                    stats_btn.click(
                        lambda: self._get_statistics(),
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
        """Iniciar agente"""
        try:
            if self.agent:
                import asyncio
                asyncio.run(self.agent.start())
                return "✅ Agente iniciado exitosamente"
            return "❌ Agente no disponible"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _stop_agent(self) -> str:
        """Detener agente"""
        try:
            if self.agent:
                import asyncio
                asyncio.run(self.agent.stop())
                return "⏹️ Agente detenido"
            return "❌ Agente no disponible"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _pause_agent(self) -> str:
        """Pausar agente"""
        try:
            if self.agent:
                import asyncio
                asyncio.run(self.agent.pause())
                return "⏸️ Agente pausado"
            return "❌ Agente no disponible"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _resume_agent(self) -> str:
        """Reanudar agente"""
        try:
            if self.agent:
                import asyncio
                asyncio.run(self.agent.resume())
                return "▶️ Agente reanudado"
            return "❌ Agente no disponible"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _execute_command(self, command: str) -> str:
        """Ejecutar comando"""
        try:
            if self.agent and command:
                import asyncio
                task_id = asyncio.run(self.agent.add_task(command))
                return f"✅ Tarea agregada: {task_id}"
            return "❌ Agente no disponible o comando vacío"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _process_with_ai(self, text: str, mode: str) -> str:
        """Procesar con IA"""
        try:
            if not self.agent or not self.agent.ai_processor:
                return "❌ Procesador de IA no disponible"
            
            import asyncio
            
            if mode == "Procesar":
                processed = asyncio.run(self.agent.ai_processor.process_command(text))
                return f"Intención: {processed.intent.value}\nConfianza: {processed.confidence:.2f}\nCódigo extraído: {processed.extracted_code or 'N/A'}"
            elif mode == "Generar Código":
                code = asyncio.run(self.agent.ai_processor.generate_code(text))
                return code or "No se pudo generar código"
            elif mode == "Explicar Código":
                if self.agent.ai_processor._model:
                    explanation = self.agent.ai_processor._model.explain_code(text)
                    return explanation
                return "Modelo no disponible"
            elif mode == "Corregir Código":
                if self.agent.ai_processor._model:
                    fixed = self.agent.ai_processor._model.fix_code(text)
                    return fixed
                return "Modelo no disponible"
            elif mode == "Resumir":
                summary = asyncio.run(self.agent.ai_processor.summarize_result(text))
                return summary
            
            return "Modo no reconocido"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _search_embeddings(self, query: str):
        """Buscar en embeddings"""
        try:
            if not self.agent or not self.agent.embedding_store:
                return []
            
            import asyncio
            results = asyncio.run(
                self.agent.embedding_store.search(query, top_k=10)
            )
            
            return [
                [metadata.get("text", key), f"{similarity:.3f}", str(metadata)]
                for key, similarity, metadata in results
            ]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _get_statistics(self):
        """Obtener estadísticas"""
        try:
            pattern_stats = {}
            metrics = {}
            
            if self.agent:
                if self.agent.pattern_learner:
                    import asyncio
                    pattern_stats = asyncio.run(self.agent.pattern_learner.get_statistics())
                
                if self.agent.metrics:
                    metrics = self.agent.metrics.get_summary()
            
            return pattern_stats, metrics
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}, {}
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Lanzar interfaz"""
        if not self.interface:
            self.create_interface()
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )


