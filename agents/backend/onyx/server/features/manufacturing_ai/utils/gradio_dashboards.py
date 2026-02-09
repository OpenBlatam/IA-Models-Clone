"""
Gradio Dashboards for Manufacturing
===================================

Dashboards interactivos para monitoreo y control de manufactura.
"""

import logging
from typing import Dict, Any, Optional

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

logger = logging.getLogger(__name__)


class ManufacturingDashboards:
    """Dashboards de manufactura con Gradio."""
    
    def __init__(self):
        """Inicializar dashboards."""
        if not GRADIO_AVAILABLE:
            logger.warning("Gradio not available")
        self.interfaces: Dict[str, Any] = {}
    
    def create_production_dashboard(self) -> str:
        """
        Crear dashboard de producción.
        
        Returns:
            ID de interfaz
        """
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required")
        
        from ..core.production_planner import get_production_planner
        from ..core.monitoring import get_manufacturing_monitor
        
        def update_dashboard():
            """Actualizar dashboard."""
            planner = get_production_planner()
            monitor = get_manufacturing_monitor()
            
            stats = planner.get_statistics()
            monitor_stats = monitor.get_statistics()
            
            info = f"""
            ## Production Dashboard
            
            ### Orders
            - Total Orders: {stats['total_orders']}
            - Status: {stats['status_counts']}
            
            ### Resources
            - Total Resources: {stats['total_resources']}
            - Available: {stats['available_resources']}
            
            ### Equipment
            - Total Equipment: {monitor_stats['total_equipment']}
            - Status: {monitor_stats['status_counts']}
            
            ### Metrics
            - Total Metrics Recorded: {monitor_stats['total_metrics']}
            - Active Alerts: {monitor_stats['active_alerts']}
            """
            
            return info
        
        interface = gr.Interface(
            fn=update_dashboard,
            inputs=None,
            outputs=gr.Markdown(),
            title="Production Dashboard",
            description="Real-time production monitoring",
            live=True,
            refresh=5
        )
        
        interface_id = "production_dashboard"
        self.interfaces[interface_id] = interface
        
        return interface_id
    
    def create_quality_dashboard(self) -> str:
        """Crear dashboard de calidad."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required")
        
        from ..core.quality_control import get_quality_controller
        
        def analyze_quality(image, features_json: str):
            """Analizar calidad."""
            import json
            
            try:
                features = json.loads(features_json) if features_json else {}
                features_list = [features.get(f"feature_{i}", 0.0) for i in range(10)]
                
                controller = get_quality_controller()
                
                # Crear check
                check_id = controller.create_check("product_001", "visual")
                
                # Realizar inspección (simplificado)
                result = controller.perform_visual_inspection(check_id, image)
                
                return f"""
                ## Quality Analysis Result
                
                - Status: **{result.status.value.upper()}**
                - Score: {result.score:.2%}
                - Confidence: {result.confidence:.2%}
                - Defects: {len(result.defects)}
                """
            except Exception as e:
                return f"Error: {str(e)}"
        
        interface = gr.Interface(
            fn=analyze_quality,
            inputs=[
                gr.Image(label="Product Image"),
                gr.Textbox(label="Features (JSON)", value='{"feature_0": 10.5, "feature_1": 20.3}')
            ],
            outputs=gr.Markdown(),
            title="Quality Control Dashboard",
            description="Analyze product quality using AI"
        )
        
        interface_id = "quality_dashboard"
        self.interfaces[interface_id] = interface
        
        return interface_id
    
    def create_optimization_dashboard(self) -> str:
        """Crear dashboard de optimización."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required")
        
        from ..core.process_optimizer import get_process_optimizer
        
        def optimize_process(process_type: str, objective: str, params_json: str):
            """Optimizar proceso."""
            import json
            
            try:
                params = json.loads(params_json)
                
                optimizer = get_process_optimizer()
                process_id = optimizer.register_process("Process", process_type, params)
                result = optimizer.optimize_process(process_id, objective)
                
                recommendations = "\n".join([f"- {r}" for r in result.recommendations])
                
                return f"""
                ## Optimization Result
                
                - Predicted Improvement: **{result.predicted_improvement:.2%}**
                - Confidence: {result.confidence:.2%}
                
                ### Optimized Parameters
                {json.dumps(result.optimized_parameters, indent=2)}
                
                ### Recommendations
                {recommendations}
                """
            except Exception as e:
                return f"Error: {str(e)}"
        
        interface = gr.Interface(
            fn=optimize_process,
            inputs=[
                gr.Dropdown(["machining", "welding", "assembly"], label="Process Type"),
                gr.Dropdown(["efficiency", "quality", "cost"], label="Objective"),
                gr.Textbox(label="Parameters (JSON)", value='{"feed_rate": 100, "spindle_speed": 1000}')
            ],
            outputs=gr.Markdown(),
            title="Process Optimization Dashboard",
            description="Optimize manufacturing processes using AI"
        )
        
        interface_id = "optimization_dashboard"
        self.interfaces[interface_id] = interface
        
        return interface_id
    
    def launch_all(self, server_port: int = 7860):
        """Lanzar todos los dashboards."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required")
        
        # Crear tabs
        with gr.Blocks(title="Manufacturing AI Dashboards") as demo:
            with gr.Tabs():
                with gr.Tab("Production"):
                    self.create_production_dashboard()
                    iface = self.interfaces["production_dashboard"]
                    iface.render()
                
                with gr.Tab("Quality"):
                    self.create_quality_dashboard()
                    iface = self.interfaces["quality_dashboard"]
                    iface.render()
                
                with gr.Tab("Optimization"):
                    self.create_optimization_dashboard()
                    iface = self.interfaces["optimization_dashboard"]
                    iface.render()
        
        demo.launch(server_port=server_port, share=False)
        logger.info(f"Launched manufacturing dashboards on port {server_port}")


# Instancia global
_manufacturing_dashboards = None


def get_manufacturing_dashboards() -> ManufacturingDashboards:
    """Obtener instancia global."""
    global _manufacturing_dashboards
    if _manufacturing_dashboards is None:
        _manufacturing_dashboards = ManufacturingDashboards()
    return _manufacturing_dashboards

