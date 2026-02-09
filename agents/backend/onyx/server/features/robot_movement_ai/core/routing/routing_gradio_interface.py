"""
Routing Gradio Interface
========================

Interfaz Gradio profesional para el sistema de routing.
Permite visualización interactiva y pruebas de modelos.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available. Interface features will be disabled.")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Plotting libraries not available. Visualization will be limited.")


class RoutingGradioInterface:
    """Interfaz Gradio para routing."""
    
    def __init__(self, router: Any):
        """
        Inicializar interfaz Gradio.
        
        Args:
            router: Instancia de IntelligentRouter
        """
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required for RoutingGradioInterface")
        
        self.router = router
        self.interface = None
    
    def create_interface(self) -> gr.Blocks:
        """Crear interfaz Gradio completa."""
        with gr.Blocks(title="Intelligent Routing System", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Intelligent Routing System")
            gr.Markdown("Sistema avanzado de enrutamiento con Deep Learning")
            
            with gr.Tabs():
                # Tab 1: Route Finding
                with gr.Tab("Find Route"):
                    with gr.Row():
                        with gr.Column():
                            start_node = gr.Textbox(
                                label="Start Node ID",
                                placeholder="Enter start node ID"
                            )
                            end_node = gr.Textbox(
                                label="End Node ID",
                                placeholder="Enter end node ID"
                            )
                            strategy = gr.Dropdown(
                                choices=[
                                    "shortest_path", "fastest_path", "least_cost",
                                    "load_balanced", "adaptive", "deep_learning",
                                    "transformer", "gnn_based"
                                ],
                                value="adaptive",
                                label="Routing Strategy"
                            )
                            find_btn = gr.Button("Find Route", variant="primary")
                        
                        with gr.Column():
                            route_output = gr.JSON(label="Route Result")
                            route_visualization = gr.Plot(label="Route Visualization")
                    
                    find_btn.click(
                        fn=self._find_route,
                        inputs=[start_node, end_node, strategy],
                        outputs=[route_output, route_visualization]
                    )
                
                # Tab 2: Batch Processing
                with gr.Tab("Batch Processing"):
                    with gr.Row():
                        with gr.Column():
                            batch_input = gr.Textbox(
                                label="Route Requests (JSON)",
                                placeholder='[{"start": "node1", "end": "node2"}, ...]',
                                lines=5
                            )
                            batch_strategy = gr.Dropdown(
                                choices=["adaptive", "shortest_path", "fastest_path"],
                                value="adaptive",
                                label="Strategy"
                            )
                            process_batch_btn = gr.Button("Process Batch", variant="primary")
                        
                        with gr.Column():
                            batch_output = gr.JSON(label="Batch Results")
                            batch_stats = gr.JSON(label="Statistics")
                    
                    process_batch_btn.click(
                        fn=self._process_batch,
                        inputs=[batch_input, batch_strategy],
                        outputs=[batch_output, batch_stats]
                    )
                
                # Tab 3: Model Training
                with gr.Tab("Model Training"):
                    with gr.Row():
                        with gr.Column():
                            epochs = gr.Slider(1, 100, value=10, label="Epochs")
                            batch_size = gr.Slider(8, 128, value=32, step=8, label="Batch Size")
                            learning_rate = gr.Slider(1e-5, 1e-2, value=1e-3, step=1e-4, label="Learning Rate")
                            train_btn = gr.Button("Start Training", variant="primary")
                        
                        with gr.Column():
                            training_log = gr.Textbox(label="Training Log", lines=10)
                            training_plot = gr.Plot(label="Training Curves")
                    
                    train_btn.click(
                        fn=self._train_model,
                        inputs=[epochs, batch_size, learning_rate],
                        outputs=[training_log, training_plot]
                    )
                
                # Tab 4: Statistics
                with gr.Tab("Statistics"):
                    stats_output = gr.JSON(label="Router Statistics")
                    refresh_stats_btn = gr.Button("Refresh Statistics")
                    
                    refresh_stats_btn.click(
                        fn=self._get_statistics,
                        inputs=[],
                        outputs=[stats_output]
                    )
            
            # Cargar estadísticas iniciales
            interface.load(
                fn=self._get_statistics,
                inputs=[],
                outputs=[stats_output]
            )
        
        self.interface = interface
        return interface
    
    def _find_route(
        self,
        start: str,
        end: str,
        strategy: str
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """
        Encontrar ruta con validación de inputs.
        
        Args:
            start: ID del nodo de inicio
            end: ID del nodo de destino
            strategy: Estrategia de routing
        
        Returns:
            (result_dict, visualization_figure)
        """
        # Validar inputs
        if not start or not start.strip():
            return {"error": "Start node ID is required"}, None
        
        if not end or not end.strip():
            return {"error": "End node ID is required"}, None
        
        if start == end:
            return {"error": "Start and end nodes must be different"}, None
        
        try:
            from .intelligent_routing import RoutingStrategy
            
            # Validar que los nodos existen
            if start not in self.router.nodes:
                return {"error": f"Start node '{start}' not found in graph"}, None
            
            if end not in self.router.nodes:
                return {"error": f"End node '{end}' not found in graph"}, None
            
            # Convertir estrategia
            try:
                strategy_enum = RoutingStrategy[strategy.upper()] if hasattr(RoutingStrategy, strategy.upper()) else RoutingStrategy.ADAPTIVE
            except (KeyError, AttributeError):
                strategy_enum = RoutingStrategy.ADAPTIVE
                logger.warning(f"Unknown strategy '{strategy}', using ADAPTIVE")
            
            # Encontrar ruta
            route = self.router.find_route(start.strip(), end.strip(), strategy_enum)
            
            result = {
                "route_id": route.route_id,
                "path": route.path,
                "total_distance": route.total_distance,
                "total_time": route.total_time,
                "total_cost": route.total_cost,
                "confidence_score": route.confidence_score,
                "strategy": route.strategy.value,
                "success": True
            }
            
            # Agregar métricas predichas si están disponibles
            if route.predicted_metrics:
                result["predicted_metrics"] = route.predicted_metrics
            
            if route.explanation:
                result["explanation"] = route.explanation
            
            # Visualización
            fig = self._visualize_route(route) if PLOTTING_AVAILABLE else None
            
            return result, fig
        except ValueError as e:
            logger.error(f"Validation error finding route: {e}")
            return {"error": f"Validation error: {str(e)}", "success": False}, None
        except RuntimeError as e:
            logger.error(f"Runtime error finding route: {e}")
            return {"error": f"Runtime error: {str(e)}", "success": False}, None
        except Exception as e:
            logger.error(f"Unexpected error finding route: {e}", exc_info=True)
            return {"error": f"Unexpected error: {str(e)}", "success": False}, None
    
    def _process_batch(
        self,
        batch_input: str,
        strategy: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Procesar batch de rutas con validación.
        
        Args:
            batch_input: JSON string con lista de requests
            strategy: Estrategia de routing
        
        Returns:
            (results_dict, statistics_dict)
        """
        # Validar input
        if not batch_input or not batch_input.strip():
            return {"error": "Batch input is required", "success": False}, {}
        
        try:
            # Parsear JSON
            requests = json.loads(batch_input)
            
            if not isinstance(requests, list):
                return {"error": "Batch input must be a JSON array", "success": False}, {}
            
            if len(requests) == 0:
                return {"error": "Batch input must contain at least one request", "success": False}, {}
            
            if len(requests) > 100:
                return {"error": "Batch size too large (max 100)", "success": False}, {}
            
            # Validar cada request
            validated_requests = []
            for i, req in enumerate(requests):
                if not isinstance(req, dict):
                    logger.warning(f"Request {i} is not a dict, skipping")
                    continue
                
                if 'start' not in req or 'end' not in req:
                    logger.warning(f"Request {i} missing 'start' or 'end', skipping")
                    continue
                
                start = str(req['start']).strip()
                end = str(req['end']).strip()
                
                if not start or not end:
                    logger.warning(f"Request {i} has empty start/end, skipping")
                    continue
                
                if start not in self.router.nodes:
                    logger.warning(f"Request {i}: start node '{start}' not found, skipping")
                    continue
                
                if end not in self.router.nodes:
                    logger.warning(f"Request {i}: end node '{end}' not found, skipping")
                    continue
                
                validated_requests.append({'start': start, 'end': end})
            
            if len(validated_requests) == 0:
                return {"error": "No valid requests found in batch", "success": False}, {}
            
            from .intelligent_routing import RoutingStrategy
            
            try:
                strategy_enum = RoutingStrategy[strategy.upper()] if hasattr(RoutingStrategy, strategy.upper()) else RoutingStrategy.ADAPTIVE
            except (KeyError, AttributeError):
                strategy_enum = RoutingStrategy.ADAPTIVE
            
            # Procesar batch
            routes = self.router.find_routes_batch(validated_requests, strategy_enum)
            
            results = [
                {
                    "route_id": r.route_id,
                    "path": r.path,
                    "distance": r.total_distance,
                    "time": r.total_time,
                    "cost": r.total_cost,
                    "confidence": r.confidence_score
                }
                for r in routes
            ]
            
            stats = self.router.get_statistics()
            
            return {
                "routes": results,
                "count": len(results),
                "requested": len(requests),
                "validated": len(validated_requests),
                "success": True
            }, stats
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {"error": f"Invalid JSON: {str(e)}", "success": False}, {}
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            return {"error": f"Error: {str(e)}", "success": False}, {}
    
    def _train_model(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Tuple[str, Optional[Any]]:
        """Entrenar modelo."""
        try:
            from .routing_trainer import ModelTrainer, TrainingConfig
            from .routing_config import ModelConfig
            
            # Crear configuración
            config = TrainingConfig(
                num_epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=learning_rate
            )
            
            # Aquí se necesitaría el modelo y dataloader real
            # Por ahora retornamos placeholder
            log = f"Training started:\n- Epochs: {epochs}\n- Batch Size: {batch_size}\n- Learning Rate: {learning_rate}\n\nTraining would start here..."
            
            return log, None
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return f"Error: {str(e)}", None
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        try:
            return self.router.get_statistics()
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def _visualize_route(self, route: Any) -> Optional[Any]:
        """Visualizar ruta."""
        if not PLOTTING_AVAILABLE:
            return None
        
        try:
            # Extraer posiciones de nodos
            positions = []
            for node_id in route.path:
                if node_id in self.router.nodes:
                    pos = self.router.nodes[node_id].position
                    positions.append([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
            
            if len(positions) < 2:
                return None
            
            positions = np.array(positions)
            
            # Crear visualización 3D con plotly
            fig = go.Figure()
            
            # Agregar ruta
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='lines+markers',
                name='Route',
                line=dict(color='blue', width=4),
                marker=dict(size=5, color='red')
            ))
            
            # Agregar punto de inicio
            fig.add_trace(go.Scatter3d(
                x=[positions[0, 0]],
                y=[positions[0, 1]],
                z=[positions[0, 2]],
                mode='markers',
                name='Start',
                marker=dict(size=10, color='green')
            ))
            
            # Agregar punto de fin
            fig.add_trace(go.Scatter3d(
                x=[positions[-1, 0]],
                y=[positions[-1, 1]],
                z=[positions[-1, 2]],
                mode='markers',
                name='End',
                marker=dict(size=10, color='red')
            ))
            
            fig.update_layout(
                title="Route Visualization",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z"
                )
            )
            
            return fig
        except Exception as e:
            logger.warning(f"Error visualizing route: {e}")
            return None
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Lanzar interfaz Gradio."""
        if self.interface is None:
            self.create_interface()
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )

