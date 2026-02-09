"""
Model Visualization - Modular Visualization
===========================================

Visualizadores modulares para modelos y resultados.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class Visualizer:
    """Clase base para visualizadores."""
    
    def visualize(self, data: Any, **kwargs) -> Any:
        """Visualizar datos."""
        raise NotImplementedError


class TrainingCurveVisualizer(Visualizer):
    """Visualizador de curvas de entrenamiento."""
    
    def visualize(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Visualizar curvas de entrenamiento.
        
        Args:
            train_losses: Pérdidas de entrenamiento
            val_losses: Pérdidas de validación (opcional)
            save_path: Ruta para guardar (opcional)
            **kwargs: Argumentos adicionales
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=kwargs.get('figsize', (10, 6)))
            plt.plot(train_losses, label='Train Loss', **kwargs)
            
            if val_losses:
                plt.plot(val_losses, label='Val Loss', **kwargs)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(kwargs.get('title', 'Training Curves'))
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=kwargs.get('dpi', 300))
                logger.info(f"Training curves saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            logger.warning("matplotlib not available")
        except Exception as e:
            logger.error(f"Error visualizing training curves: {e}")


class TrajectoryVisualizer(Visualizer):
    """Visualizador de trayectorias."""
    
    def visualize(
        self,
        trajectories: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Visualizar trayectorias.
        
        Args:
            trajectories: Trayectorias [num_trajectories, length, dim]
            labels: Etiquetas (opcional)
            save_path: Ruta para guardar (opcional)
            **kwargs: Argumentos adicionales
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=kwargs.get('figsize', (12, 8)))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, traj in enumerate(trajectories):
                label = labels[i] if labels and i < len(labels) else f'Trajectory {i+1}'
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=label, **kwargs)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(kwargs.get('title', 'Trajectories'))
            ax.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=kwargs.get('dpi', 300))
                logger.info(f"Trajectories saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            logger.warning("matplotlib not available")
        except Exception as e:
            logger.error(f"Error visualizing trajectories: {e}")


class AttentionVisualizer(Visualizer):
    """Visualizador de atención."""
    
    def visualize(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Visualizar pesos de atención.
        
        Args:
            attention_weights: Pesos de atención [heads, seq_len, seq_len]
            tokens: Tokens (opcional)
            save_path: Ruta para guardar (opcional)
            **kwargs: Argumentos adicionales
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Promediar sobre heads
            if attention_weights.dim() == 3:
                attention_weights = attention_weights.mean(dim=0)
            
            attention_np = attention_weights.detach().cpu().numpy()
            
            plt.figure(figsize=kwargs.get('figsize', (10, 8)))
            sns.heatmap(
                attention_np,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap=kwargs.get('cmap', 'Blues'),
                **kwargs
            )
            plt.title(kwargs.get('title', 'Attention Weights'))
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=kwargs.get('dpi', 300))
                logger.info(f"Attention weights saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            logger.warning("matplotlib/seaborn not available")
        except Exception as e:
            logger.error(f"Error visualizing attention: {e}")


class ModelArchitectureVisualizer(Visualizer):
    """Visualizador de arquitectura de modelo."""
    
    def visualize(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Visualizar arquitectura de modelo.
        
        Args:
            model: Modelo PyTorch
            input_shape: Forma de entrada
            save_path: Ruta para guardar (opcional)
            **kwargs: Argumentos adicionales
        """
        try:
            from torchviz import make_dot
            
            # Crear ejemplo de entrada
            example_input = torch.randn(1, *input_shape)
            
            # Forward pass
            output = model(example_input)
            
            # Visualizar
            dot = make_dot(output, params=dict(model.named_parameters()))
            
            if save_path:
                dot.render(save_path, format='png')
                logger.info(f"Model architecture saved to {save_path}")
            else:
                dot.view()
        except ImportError:
            logger.warning("torchviz not available")
        except Exception as e:
            logger.error(f"Error visualizing model architecture: {e}")


class VisualizationFactory:
    """Factory para visualizadores."""
    
    _visualizers = {
        'training_curves': TrainingCurveVisualizer,
        'trajectories': TrajectoryVisualizer,
        'attention': AttentionVisualizer,
        'architecture': ModelArchitectureVisualizer
    }
    
    @classmethod
    def get_visualizer(cls, visualization_type: str) -> Visualizer:
        """
        Obtener visualizador por tipo.
        
        Args:
            visualization_type: Tipo de visualización
            
        Returns:
            Visualizador
        """
        if visualization_type not in cls._visualizers:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
        
        return cls._visualizers[visualization_type]()
    
    @classmethod
    def register_visualizer(cls, visualization_type: str, visualizer_class: type):
        """Registrar nuevo visualizador."""
        cls._visualizers[visualization_type] = visualizer_class


def visualize(
    visualization_type: str,
    data: Any,
    **kwargs
):
    """
    Visualizar datos.
    
    Args:
        visualization_type: Tipo de visualización
        data: Datos a visualizar
        **kwargs: Argumentos adicionales
    """
    visualizer = VisualizationFactory.get_visualizer(visualization_type)
    visualizer.visualize(data, **kwargs)








