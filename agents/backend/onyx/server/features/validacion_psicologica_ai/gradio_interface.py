"""
Interfaz Gradio para Validación Psicológica
===========================================
Interfaz interactiva para demostración y uso
"""

from typing import Dict, Any, List, Optional
import structlog
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json

from .deep_learning_models import deep_learning_analyzer
from .models import PsychologicalProfile

logger = structlog.get_logger()


class GradioInterface:
    """Interfaz Gradio para el sistema"""
    
    def __init__(self):
        """Inicializar interfaz"""
        self.interface = None
        logger.info("GradioInterface initialized")
    
    def create_interface(self) -> gr.Blocks:
        """
        Crear interfaz Gradio con validación mejorada
        
        Returns:
            Interfaz Gradio
        """
        with gr.Blocks(title="Validación Psicológica AI", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🧠 Sistema de Validación Psicológica AI")
            gr.Markdown("Análisis psicológico avanzado usando Deep Learning y LLMs")
            
            with gr.Tabs():
                # Tab 1: Análisis de Texto
                with gr.Tab("📝 Análisis de Texto"):
                    with gr.Row():
                        with gr.Column():
                            text_input = gr.Textbox(
                                label="Ingresa texto para análisis",
                                placeholder="Escribe o pega texto de redes sociales...",
                                lines=5,
                                max_lines=20
                            )
                            analyze_btn = gr.Button("Analizar", variant="primary")
                    
                        with gr.Column():
                            sentiment_output = gr.JSON(label="Análisis de Sentimiento")
                            personality_output = gr.JSON(label="Rasgos de Personalidad")
                    
                    def validate_and_analyze(text):
                        """Validar y analizar texto"""
                        if not text or len(text.strip()) < 10:
                            raise gr.Error("El texto debe tener al menos 10 caracteres")
                        if len(text) > 10000:
                            raise gr.Error("El texto es demasiado largo (máximo 10,000 caracteres)")
                        return self._analyze_text(text)
                    
                    analyze_btn.click(
                        fn=validate_and_analyze,
                        inputs=[text_input],
                        outputs=[sentiment_output, personality_output]
                    )
                
                # Tab 2: Análisis de Múltiples Textos
                with gr.Tab("📊 Análisis por Lotes"):
                    with gr.Row():
                        with gr.Column():
                            batch_texts = gr.Dataframe(
                                label="Textos para análisis",
                                headers=["Texto"],
                                datatype=["str"],
                                row_count=5
                            )
                            batch_analyze_btn = gr.Button("Analizar Lote", variant="primary")
                        
                        with gr.Column():
                            batch_results = gr.JSON(label="Resultados del Análisis")
                            visualization = gr.Plot(label="Visualización")
                    
                    batch_analyze_btn.click(
                        fn=self._analyze_batch,
                        inputs=[batch_texts],
                        outputs=[batch_results, visualization]
                    )
                
                # Tab 3: Análisis de Perfil Completo
                with gr.Tab("👤 Perfil Psicológico"):
                    with gr.Row():
                        with gr.Column():
                            profile_texts = gr.Textbox(
                                label="Textos del perfil",
                                placeholder="Ingresa múltiples textos separados por líneas...",
                                lines=10
                            )
                            generate_profile_btn = gr.Button("Generar Perfil", variant="primary")
                        
                        with gr.Column():
                            profile_output = gr.JSON(label="Perfil Psicológico")
                            profile_chart = gr.Plot(label="Gráfico de Rasgos")
                    
                    generate_profile_btn.click(
                        fn=self._generate_profile,
                        inputs=[profile_texts],
                        outputs=[profile_output, profile_chart]
                    )
                
                # Tab 4: Comparación
                with gr.Tab("⚖️ Comparación"):
                    with gr.Row():
                        with gr.Column():
                            text1 = gr.Textbox(label="Texto 1", lines=3)
                            text2 = gr.Textbox(label="Texto 2", lines=3)
                            compare_btn = gr.Button("Comparar", variant="primary")
                        
                        with gr.Column():
                            comparison_output = gr.JSON(label="Comparación")
                            comparison_chart = gr.Plot(label="Gráfico Comparativo")
                    
                    compare_btn.click(
                        fn=self._compare_texts,
                        inputs=[text1, text2],
                        outputs=[comparison_output, comparison_chart]
                    )
            
            # Footer
            gr.Markdown("---")
            gr.Markdown("### ℹ️ Información")
            gr.Markdown(
                """
                - **Modelos**: Transformers, RoBERTa, DistilBERT
                - **Análisis**: Sentimientos, Personalidad (Big Five), Patrones de comportamiento
                - **Tecnología**: PyTorch, Transformers, Deep Learning
                """
            )
        
        self.interface = interface
        return interface
    
    async def _analyze_text(self, text: str) -> tuple[Dict, Dict]:
        """Analizar texto individual"""
        if not text.strip():
            return {}, {}
        
        try:
            results = await deep_learning_analyzer.analyze_comprehensive(
                texts=[text],
                include_llm=False
            )
            
            sentiment = results.get("sentiment", {})
            personality = results.get("personality", {})
            
            return sentiment, personality
        except Exception as e:
            logger.error("Error in text analysis", error=str(e))
            return {"error": str(e)}, {}
    
    async def _analyze_batch(self, dataframe: pd.DataFrame) -> tuple[Dict, go.Figure]:
        """Analizar lote de textos"""
        if dataframe is None or dataframe.empty:
            return {}, go.Figure()
        
        try:
            texts = dataframe.iloc[:, 0].tolist()
            results = await deep_learning_analyzer.analyze_comprehensive(
                texts=texts,
                include_llm=False
            )
            
            # Crear visualización
            fig = self._create_batch_visualization(results)
            
            return results, fig
        except Exception as e:
            logger.error("Error in batch analysis", error=str(e))
            return {"error": str(e)}, go.Figure()
    
    async def _generate_profile(self, texts: str) -> tuple[Dict, go.Figure]:
        """Generar perfil psicológico"""
        if not texts.strip():
            return {}, go.Figure()
        
        try:
            text_list = [t.strip() for t in texts.split('\n') if t.strip()]
            results = await deep_learning_analyzer.analyze_comprehensive(
                texts=text_list,
                include_llm=True
            )
            
            # Crear gráfico de rasgos
            personality = results.get("personality", {})
            fig = self._create_personality_chart(personality)
            
            return results, fig
        except Exception as e:
            logger.error("Error generating profile", error=str(e))
            return {"error": str(e)}, go.Figure()
    
    async def _compare_texts(self, text1: str, text2: str) -> tuple[Dict, go.Figure]:
        """Comparar dos textos"""
        if not text1.strip() or not text2.strip():
            return {}, go.Figure()
        
        try:
            results1 = await deep_learning_analyzer.analyze_comprehensive([text1], include_llm=False)
            results2 = await deep_learning_analyzer.analyze_comprehensive([text2], include_llm=False)
            
            comparison = {
                "text1": results1,
                "text2": results2,
                "differences": self._calculate_differences(results1, results2)
            }
            
            fig = self._create_comparison_chart(results1, results2)
            
            return comparison, fig
        except Exception as e:
            logger.error("Error comparing texts", error=str(e))
            return {"error": str(e)}, go.Figure()
    
    def _create_personality_chart(self, personality: Dict[str, List]) -> go.Figure:
        """Crear gráfico de personalidad"""
        fig = go.Figure()
        
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        values = [personality.get(trait, [0.5])[0] for trait in traits]
        
        fig.add_trace(go.Bar(
            x=traits,
            y=values,
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title="Rasgos de Personalidad (Big Five)",
            xaxis_title="Rasgo",
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        
        return fig
    
    def _create_batch_visualization(self, results: Dict) -> go.Figure:
        """Crear visualización de lote"""
        # Implementación simplificada
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3],
            y=[0.5, 0.6, 0.7],
            mode='lines+markers'
        ))
        return fig
    
    def _create_comparison_chart(self, results1: Dict, results2: Dict) -> go.Figure:
        """Crear gráfico comparativo"""
        fig = go.Figure()
        
        personality1 = results1.get("personality", {})
        personality2 = results2.get("personality", {})
        
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        values1 = [personality1.get(trait, [0.5])[0] for trait in traits]
        values2 = [personality2.get(trait, [0.5])[0] for trait in traits]
        
        fig.add_trace(go.Bar(
            name="Texto 1",
            x=traits,
            y=values1,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name="Texto 2",
            x=traits,
            y=values2,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Comparación de Rasgos de Personalidad",
            xaxis_title="Rasgo",
            yaxis_title="Score",
            barmode='group'
        )
        
        return fig
    
    def _calculate_differences(self, results1: Dict, results2: Dict) -> Dict:
        """Calcular diferencias entre resultados"""
        personality1 = results1.get("personality", {})
        personality2 = results2.get("personality", {})
        
        differences = {}
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            val1 = personality1.get(trait, [0.5])[0]
            val2 = personality2.get(trait, [0.5])[0]
            differences[trait] = abs(val1 - val2)
        
        return differences
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """
        Lanzar interfaz
        
        Args:
            share: Compartir públicamente
            server_name: Nombre del servidor
            server_port: Puerto del servidor
        """
        if self.interface is None:
            self.create_interface()
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )


# Instancia global de la interfaz
gradio_interface = GradioInterface()

