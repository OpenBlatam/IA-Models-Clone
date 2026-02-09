"""
Gradio Interface for Deep Learning Models
==========================================

Interfaz interactiva usando Gradio para visualización, inferencia y demos
de modelos de deep learning para control de robots.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio no disponible, interfaz no disponible")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def create_model_demo_interface(
    model_manager=None,
    trajectory_predictor=None,
    diffusion_generator=None,
    transformer_model=None
):
    """
    Crear interfaz Gradio para demos de modelos de deep learning.
    
    Args:
        model_manager: Gestor de modelos DL
        trajectory_predictor: Modelo predictor de trayectorias
        diffusion_generator: Generador de difusión
        transformer_model: Modelo Transformer
        
    Returns:
        Interfaz de Gradio
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio no disponible")
    
    def predict_trajectory_interface(
        current_pos_x: float,
        current_pos_y: float,
        current_pos_z: float,
        velocity_x: float,
        velocity_y: float,
        velocity_z: float,
        model_type: str = "mlp"
    ) -> Tuple[str, Optional[Any]]:
        """Interfaz para predicción de trayectorias."""
        try:
            if not TORCH_AVAILABLE or trajectory_predictor is None:
                return "Modelo no disponible", None
            
            # Preparar entrada
            input_data = np.array([
                current_pos_x, current_pos_y, current_pos_z,
                velocity_x, velocity_y, velocity_z
            ]).reshape(1, -1)
            
            # Convertir a tensor
            input_tensor = torch.FloatTensor(input_data)
            
            # Predecir
            trajectory_predictor.eval()
            with torch.no_grad():
                prediction = trajectory_predictor(input_tensor)
            
            pred_np = prediction.cpu().numpy()[0]
            
            # Visualizar
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Trayectoria actual y predicha
            ax.scatter([current_pos_x], [current_pos_y], [current_pos_z], 
                      c='green', s=200, marker='o', label='Posición Actual')
            ax.scatter([pred_np[0]], [pred_np[1]], [pred_np[2]], 
                      c='red', s=200, marker='s', label='Posición Predicha')
            
            # Vector de velocidad
            ax.quiver(current_pos_x, current_pos_y, current_pos_z,
                     velocity_x, velocity_y, velocity_z,
                     color='blue', arrow_length_ratio=0.3, label='Velocidad')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Predicción de Trayectoria')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            result_text = f"""
## Predicción de Trayectoria

### Entrada:
- **Posición Actual:** ({current_pos_x:.3f}, {current_pos_y:.3f}, {current_pos_z:.3f}) m
- **Velocidad:** ({velocity_x:.3f}, {velocity_y:.3f}, {velocity_z:.3f}) m/s

### Predicción:
- **Posición Futura:** ({pred_np[0]:.3f}, {pred_np[1]:.3f}, {pred_np[2]:.3f}) m
- **Desplazamiento:** {np.linalg.norm(pred_np[:3] - np.array([current_pos_x, current_pos_y, current_pos_z])):.3f} m

### Modelo:
- **Tipo:** {model_type}
- **Parámetros:** {sum(p.numel() for p in trajectory_predictor.parameters()):,}
"""
            
            return result_text, fig
            
        except Exception as e:
            error_msg = f"Error en predicción: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, None
    
    def generate_diffusion_trajectory_interface(
        trajectory_length: int = 50,
        num_samples: int = 1
    ) -> Tuple[str, Optional[Any]]:
        """Interfaz para generación de trayectorias con difusión."""
        try:
            if not TORCH_AVAILABLE or diffusion_generator is None:
                return "Generador de difusión no disponible", None
            
            # Generar trayectorias
            diffusion_generator.eval()
            with torch.no_grad():
                trajectories = diffusion_generator.sample(
                    batch_size=num_samples,
                    device=next(diffusion_generator.parameters()).device
                )
            
            traj_np = trajectories.cpu().numpy()
            
            # Visualizar
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
            
            for i, traj in enumerate(traj_np):
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                       color=colors[i], alpha=0.7, linewidth=2,
                       label=f'Trayectoria {i+1}')
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                          color=colors[i], s=100, marker='o', 
                          edgecolors='black', linewidths=2)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                          color=colors[i], s=100, marker='s',
                          edgecolors='black', linewidths=2)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Trayectorias Generadas con Modelo de Difusión (n={num_samples})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            result_text = f"""
## Trayectorias Generadas con Difusión

### Parámetros:
- **Longitud de Trayectoria:** {trajectory_length} pasos
- **Número de Muestras:** {num_samples}

### Estadísticas:
- **Modelo:** Diffusion UNet
- **Timesteps de Difusión:** {diffusion_generator.num_timesteps}
- **Parámetros del Modelo:** {sum(p.numel() for p in diffusion_generator.parameters()):,}

### Características:
- ✅ Generación suave y natural
- ✅ Variabilidad controlada
- ✅ Sin colisiones (si se entrena apropiadamente)
"""
            
            return result_text, fig
            
        except Exception as e:
            error_msg = f"Error en generación: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, None
    
    def transformer_inference_interface(
        sequence_input: str,
        max_length: int = 10
    ) -> Tuple[str, Optional[Any]]:
        """Interfaz para inferencia con Transformer."""
        try:
            if not TORCH_AVAILABLE or transformer_model is None:
                return "Modelo Transformer no disponible", None
            
            # Parsear secuencia de entrada
            try:
                seq_data = json.loads(sequence_input)
                if isinstance(seq_data, list):
                    input_seq = np.array(seq_data)
                else:
                    return "Formato de entrada inválido. Use JSON array.", None
            except:
                # Intentar parsear como lista de números separados por comas
                try:
                    seq_data = [float(x.strip()) for x in sequence_input.split(',')]
                    input_seq = np.array(seq_data).reshape(1, -1, 3)  # Asumir 3D
                except:
                    return "Formato de entrada inválido.", None
            
            # Convertir a tensor
            input_tensor = torch.FloatTensor(input_seq)
            
            # Predecir secuencia
            transformer_model.eval()
            with torch.no_grad():
                if hasattr(transformer_model, 'predict_sequence'):
                    predictions = transformer_model.predict_sequence(
                        input_tensor, future_steps=max_length
                    )
                else:
                    predictions = transformer_model(input_tensor)
            
            pred_np = predictions.cpu().numpy()
            
            # Visualizar
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for dim, ax in enumerate(axes):
                # Entrada
                input_vals = input_seq[0, :, dim] if input_seq.ndim == 3 else input_seq[0, dim::3]
                ax.plot(range(len(input_vals)), input_vals, 'b-o', 
                       label='Entrada', linewidth=2, markersize=6)
                
                # Predicción
                pred_vals = pred_np[0, :, dim] if pred_np.ndim == 3 else pred_np[0, dim::3]
                pred_start = len(input_vals)
                ax.plot(range(pred_start, pred_start + len(pred_vals)), pred_vals,
                       'r-s', label='Predicción', linewidth=2, markersize=6)
                
                ax.set_xlabel('Timestep')
                ax.set_ylabel(f'Dimensión {dim} (m)')
                ax.set_title(f'Predicción Transformer - Dimensión {dim}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            result_text = f"""
## Predicción con Transformer

### Entrada:
- **Secuencia:** {len(input_seq[0])} timesteps
- **Dimensión:** {input_seq.shape[-1]}D

### Predicción:
- **Pasos Futuros:** {max_length}
- **Modelo:** Transformer con {transformer_model.num_layers} capas
- **Heads de Atención:** {transformer_model.num_heads}
- **Parámetros:** {sum(p.numel() for p in transformer_model.parameters()):,}

### Características:
- ✅ Atención multi-head
- ✅ Codificación posicional
- ✅ Dependencias temporales
"""
            
            return result_text, fig
            
        except Exception as e:
            error_msg = f"Error en inferencia Transformer: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, None
    
    def get_model_statistics_interface() -> str:
        """Obtener estadísticas de modelos."""
        try:
            stats_text = "## Estadísticas de Modelos de Deep Learning\n\n"
            
            if model_manager:
                stats = model_manager.get_statistics()
                stats_text += f"""
### Gestor de Modelos:
- **Total de Modelos:** {stats.get('total_models', 0)}
- **Dispositivo:** {stats.get('device', 'N/A')}
- **Total de Checkpoints:** {stats.get('total_checkpoints', 0)}

### Tipos de Modelos:
"""
                for model_type, count in stats.get('model_types', {}).items():
                    stats_text += f"- **{model_type}:** {count}\n"
            
            stats_text += "\n### Modelos Disponibles:\n"
            
            if trajectory_predictor:
                stats_text += f"""
- **Trajectory Predictor:** ✅
  - Parámetros: {sum(p.numel() for p in trajectory_predictor.parameters()):,}
  - Entrenable: {sum(p.numel() for p in trajectory_predictor.parameters() if p.requires_grad):,}
"""
            
            if diffusion_generator:
                stats_text += f"""
- **Diffusion Generator:** ✅
  - Parámetros: {sum(p.numel() for p in diffusion_generator.parameters()):,}
  - Timesteps: {diffusion_generator.num_timesteps}
"""
            
            if transformer_model:
                stats_text += f"""
- **Transformer Model:** ✅
  - Parámetros: {sum(p.numel() for p in transformer_model.parameters()):,}
  - Capas: {transformer_model.num_layers}
  - Heads: {transformer_model.num_heads}
"""
            
            return stats_text
        except Exception as e:
            return f"Error obteniendo estadísticas: {str(e)}"
    
    # Crear interfaz Gradio
    with gr.Blocks(title="Deep Learning Models Demo") as interface:
        gr.Markdown("# 🤖 Robot Movement AI - Deep Learning Models Demo")
        gr.Markdown("Interfaz interactiva para modelos de deep learning: Transformers, Difusión, y más")
        
        with gr.Tabs():
            with gr.Tab("Predicción de Trayectorias"):
                gr.Markdown("### Predictor de Trayectorias (MLP/LSTM)")
                with gr.Row():
                    with gr.Column():
                        pos_x = gr.Slider(-2.0, 2.0, value=0.5, label="Posición X (m)")
                        pos_y = gr.Slider(-2.0, 2.0, value=0.3, label="Posición Y (m)")
                        pos_z = gr.Slider(0.0, 2.0, value=0.2, label="Posición Z (m)")
                        vel_x = gr.Slider(-1.0, 1.0, value=0.1, label="Velocidad X (m/s)")
                        vel_y = gr.Slider(-1.0, 1.0, value=0.05, label="Velocidad Y (m/s)")
                        vel_z = gr.Slider(-1.0, 1.0, value=0.0, label="Velocidad Z (m/s)")
                        model_type_dropdown = gr.Dropdown(
                            choices=["mlp", "lstm", "transformer"],
                            value="mlp",
                            label="Tipo de Modelo"
                        )
                        predict_btn = gr.Button("Predecir Trayectoria", variant="primary")
                    with gr.Column():
                        prediction_output = gr.Markdown(label="Resultado")
                        prediction_plot = gr.Plot(label="Visualización")
            
            with gr.Tab("Generación con Difusión"):
                gr.Markdown("### Generador de Trayectorias con Modelo de Difusión")
                with gr.Row():
                    with gr.Column():
                        traj_length = gr.Slider(10, 100, value=50, step=10, label="Longitud de Trayectoria")
                        num_samples_slider = gr.Slider(1, 5, value=1, step=1, label="Número de Muestras")
                        generate_btn = gr.Button("Generar Trayectorias", variant="primary")
                    with gr.Column():
                        diffusion_output = gr.Markdown(label="Resultado")
                        diffusion_plot = gr.Plot(label="Visualización")
            
            with gr.Tab("Inferencia Transformer"):
                gr.Markdown("### Predicción de Secuencias con Transformer")
                with gr.Row():
                    with gr.Column():
                        sequence_input = gr.Textbox(
                            label="Secuencia de Entrada (JSON array o números separados por comas)",
                            placeholder='[[0.5, 0.3, 0.2], [0.6, 0.4, 0.3], ...] o 0.5, 0.3, 0.2, 0.6, 0.4, 0.3',
                            lines=3
                        )
                        max_length_slider = gr.Slider(5, 50, value=10, step=5, label="Pasos Futuros a Predecir")
                        transformer_btn = gr.Button("Predecir Secuencia", variant="primary")
                    with gr.Column():
                        transformer_output = gr.Markdown(label="Resultado")
                        transformer_plot = gr.Plot(label="Visualización")
            
            with gr.Tab("Estadísticas"):
                gr.Markdown("### Estadísticas de Modelos")
                stats_btn = gr.Button("Actualizar Estadísticas", variant="secondary")
                stats_output = gr.Markdown(label="Estadísticas")
        
        # Eventos
        predict_btn.click(
            fn=predict_trajectory_interface,
            inputs=[pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, model_type_dropdown],
            outputs=[prediction_output, prediction_plot]
        )
        
        generate_btn.click(
            fn=generate_diffusion_trajectory_interface,
            inputs=[traj_length, num_samples_slider],
            outputs=[diffusion_output, diffusion_plot]
        )
        
        transformer_btn.click(
            fn=transformer_inference_interface,
            inputs=[sequence_input, max_length_slider],
            outputs=[transformer_output, transformer_plot]
        )
        
        stats_btn.click(
            fn=get_model_statistics_interface,
            outputs=stats_output
        )
        
        gr.Markdown("""
        ### Características:
        - ✅ **Predicción de Trayectorias:** MLP, LSTM, Transformer
        - ✅ **Generación con Difusión:** Trayectorias suaves y naturales
        - ✅ **Inferencia Transformer:** Predicción de secuencias temporales
        - ✅ **Visualización Interactiva:** Gráficos 3D y temporales
        - ✅ **Experiment Tracking:** WandB y TensorBoard
        """)
    
    return interface


def launch_model_demo(
    model_manager=None,
    trajectory_predictor=None,
    diffusion_generator=None,
    transformer_model=None,
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7861
):
    """
    Lanzar demo de modelos.
    
    Args:
        model_manager: Gestor de modelos
        trajectory_predictor: Modelo predictor
        diffusion_generator: Generador de difusión
        transformer_model: Modelo Transformer
        share: Compartir públicamente
        server_name: Nombre del servidor
        server_port: Puerto del servidor
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio no disponible")
    
    interface = create_model_demo_interface(
        model_manager=model_manager,
        trajectory_predictor=trajectory_predictor,
        diffusion_generator=diffusion_generator,
        transformer_model=transformer_model
    )
    
    interface.launch(
        share=share,
        server_name=server_name,
        server_port=server_port
    )

