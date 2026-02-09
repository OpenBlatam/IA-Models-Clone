"""
Gradio Interface for Robot Movement AI
=======================================

Interfaz interactiva usando Gradio para demostración y control del robot.
"""

import logging
from typing import Optional, Dict, Any, List
import gradio as gr
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class GradioRobotInterface:
    """
    Interfaz Gradio para control y visualización del robot.
    """
    
    def __init__(
        self,
        movement_engine=None,
        chat_controller=None,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None
    ):
        """
        Inicializar interfaz Gradio.
        
        Args:
            movement_engine: Motor de movimiento del robot
            chat_controller: Controlador de chat
            model: Modelo de deep learning (opcional)
            model_path: Ruta al modelo (opcional)
        """
        self.movement_engine = movement_engine
        self.chat_controller = chat_controller
        self.model = model
        self.model_path = model_path
        
        if model_path and model is None:
            self._load_model()
    
    def _load_model(self):
        """Cargar modelo desde archivo."""
        if self.model_path and Path(self.model_path).exists():
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu')
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
    
    def create_interface(self) -> gr.Blocks:
        """
        Crear interfaz Gradio completa.
        
        Returns:
            Blocks de Gradio
        """
        with gr.Blocks(title="Robot Movement AI", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Robot Movement AI Control Panel")
            gr.Markdown("Control your robot using natural language or direct commands.")
            
            with gr.Tabs():
                # Tab 1: Chat Control
                with gr.Tab("💬 Chat Control"):
                    self._create_chat_tab()
                
                # Tab 2: Direct Control
                with gr.Tab("🎮 Direct Control"):
                    self._create_direct_control_tab()
                
                # Tab 3: Trajectory Generation
                with gr.Tab("📈 Trajectory Generation"):
                    self._create_trajectory_tab()
                
                # Tab 4: Model Inference
                with gr.Tab("🧠 Model Inference"):
                    self._create_model_inference_tab()
                
                # Tab 5: Status & Monitoring
                with gr.Tab("📊 Status & Monitoring"):
                    self._create_status_tab()
        
        return interface
    
    def _create_chat_tab(self):
        """Crear tab de chat."""
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat with Robot",
                    height=400,
                    show_copy_button=True
                )
                msg = gr.Textbox(
                    label="Enter command",
                    placeholder="e.g., 'move to (0.5, 0.3, 0.2)' or 'stop'",
                    show_label=False
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")
            
            with gr.Column(scale=1):
                gr.Markdown("### Example Commands")
                gr.Examples(
                    examples=[
                        "move to (0.5, 0.3, 0.2)",
                        "move relative (0.1, 0.0, -0.05)",
                        "stop",
                        "go home",
                        "status"
                    ],
                    inputs=msg
                )
        
        def respond(message, history):
            """Responder a mensaje."""
            if not message.strip():
                return history, ""
            
            if self.chat_controller:
                try:
                    result = self.chat_controller.process_command(message)
                    response = result.get('message', 'Command processed')
                    history.append((message, response))
                except Exception as e:
                    history.append((message, f"Error: {str(e)}"))
            else:
                history.append((message, "Chat controller not available"))
            
            return history, ""
        
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
    
    def _create_direct_control_tab(self):
        """Crear tab de control directo."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Position Control")
                x_input = gr.Slider(-1.0, 1.0, value=0.0, label="X")
                y_input = gr.Slider(-1.0, 1.0, value=0.0, label="Y")
                z_input = gr.Slider(0.0, 1.0, value=0.5, label="Z")
                move_btn = gr.Button("Move to Position", variant="primary")
                
                gr.Markdown("### Relative Movement")
                dx_input = gr.Slider(-0.5, 0.5, value=0.0, label="ΔX")
                dy_input = gr.Slider(-0.5, 0.5, value=0.0, label="ΔY")
                dz_input = gr.Slider(-0.5, 0.5, value=0.0, label="ΔZ")
                move_relative_btn = gr.Button("Move Relative", variant="secondary")
            
            with gr.Column():
                gr.Markdown("### Actions")
                stop_btn = gr.Button("🛑 Stop", variant="stop")
                home_btn = gr.Button("🏠 Go Home", variant="secondary")
                status_btn = gr.Button("📊 Get Status", variant="secondary")
                
                output_text = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )
        
        def move_to_position(x, y, z):
            """Mover a posición."""
            if self.movement_engine:
                try:
                    from ..robot.inverse_kinematics import EndEffectorPose
                    import asyncio
                    
                    pose = EndEffectorPose(
                        position=np.array([x, y, z], dtype=np.float64),
                        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                    )
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        self.movement_engine.move_to_pose(pose)
                    )
                    loop.close()
                    
                    return f"Movement {'successful' if success else 'failed'}"
                except Exception as e:
                    return f"Error: {str(e)}"
            return "Movement engine not available"
        
        def move_relative(dx, dy, dz):
            """Mover relativamente."""
            return "Relative movement not implemented in this interface"
        
        def stop():
            """Detener."""
            if self.movement_engine:
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.movement_engine.stop_movement())
                    loop.close()
                    return "Robot stopped"
                except Exception as e:
                    return f"Error: {str(e)}"
            return "Movement engine not available"
        
        def get_status():
            """Obtener estado."""
            if self.movement_engine:
                status = self.movement_engine.get_status()
                return str(status)
            return "Movement engine not available"
        
        move_btn.click(move_to_position, [x_input, y_input, z_input], output_text)
        move_relative_btn.click(move_relative, [dx_input, dy_input, dz_input], output_text)
        stop_btn.click(stop, outputs=output_text)
        home_btn.click(lambda: move_to_position(0.0, 0.0, 0.5), outputs=output_text)
        status_btn.click(get_status, outputs=output_text)
    
    def _create_trajectory_tab(self):
        """Crear tab de generación de trayectorias."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Generate Trajectory")
                
                start_x = gr.Number(value=0.0, label="Start X")
                start_y = gr.Number(value=0.0, label="Start Y")
                start_z = gr.Number(value=0.5, label="Start Z")
                
                end_x = gr.Number(value=0.5, label="End X")
                end_y = gr.Number(value=0.3, label="End Y")
                end_z = gr.Number(value=0.2, label="End Z")
                
                trajectory_length = gr.Slider(10, 200, value=50, step=10, label="Trajectory Length")
                generate_btn = gr.Button("Generate Trajectory", variant="primary")
            
            with gr.Column():
                trajectory_plot = gr.Plot(label="Trajectory Visualization")
                trajectory_output = gr.Textbox(label="Trajectory Data", lines=10)
        
        def generate_trajectory(sx, sy, sz, ex, ey, ez, length):
            """Generar trayectoria."""
            try:
                # Generar trayectoria simple (lineal)
                t = np.linspace(0, 1, int(length))
                trajectory = np.array([
                    [sx + (ex - sx) * ti, sy + (ey - sy) * ti, sz + (ez - sz) * ti]
                    for ti in t
                ])
                
                # Visualización simple
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='XY Projection')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Trajectory Visualization')
                ax.legend()
                ax.grid(True)
                
                return fig, str(trajectory.tolist())
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        generate_btn.click(
            generate_trajectory,
            [start_x, start_y, start_z, end_x, end_y, end_z, trajectory_length],
            [trajectory_plot, trajectory_output]
        )
    
    def _create_model_inference_tab(self):
        """Crear tab de inferencia de modelo."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Inference")
                
                input_data = gr.Textbox(
                    label="Input Data (JSON)",
                    placeholder='{"trajectory": [[0, 0, 0], [0.1, 0.1, 0.1], ...]}',
                    lines=5
                )
                
                inference_btn = gr.Button("Run Inference", variant="primary")
            
            with gr.Column():
                output_data = gr.Textbox(
                    label="Model Output",
                    lines=10,
                    interactive=False
                )
        
        def run_inference(input_json):
            """Ejecutar inferencia."""
            if not self.model:
                return "Model not loaded"
            
            try:
                import json
                data = json.loads(input_json)
                trajectory = np.array(data.get('trajectory', []))
                
                if len(trajectory) == 0:
                    return "No trajectory data provided"
                
                # Convertir a tensor
                trajectory_tensor = torch.from_numpy(trajectory).float().unsqueeze(0)
                
                # Inferencia
                self.model.eval()
                with torch.no_grad():
                    output = self.model(trajectory_tensor)
                
                result = {
                    'input_shape': list(trajectory.shape),
                    'output_shape': list(output.shape),
                    'output': output.numpy().tolist()
                }
                
                return json.dumps(result, indent=2)
            except Exception as e:
                return f"Error: {str(e)}"
        
        inference_btn.click(run_inference, input_data, output_data)
    
    def _create_status_tab(self):
        """Crear tab de estado y monitoreo."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Robot Status")
                status_display = gr.JSON(label="Current Status")
                refresh_btn = gr.Button("🔄 Refresh Status", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Statistics")
                stats_display = gr.JSON(label="Statistics")
        
        def get_status():
            """Obtener estado."""
            status = {}
            stats = {}
            
            if self.movement_engine:
                status = self.movement_engine.get_status()
                stats = self.movement_engine.get_statistics()
            
            if self.chat_controller:
                chat_stats = self.chat_controller.get_statistics()
                stats.update(chat_stats)
            
            return status, stats
        
        refresh_btn.click(get_status, outputs=[status_display, stats_display])
        # Auto-refresh cada 5 segundos
        interface.load(get_status, outputs=[status_display, stats_display], every=5)
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False
    ):
        """
        Lanzar interfaz Gradio.
        
        Args:
            server_name: Nombre del servidor
            server_port: Puerto del servidor
            share: Compartir públicamente
        """
        interface = self.create_interface()
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        )
