"""
Backend Generator - Generador de Backend
========================================

Genera automáticamente la estructura completa de backend para proyectos de IA.
"""

from pathlib import Path
from typing import Dict, Any, List

from .deep_learning_generator import DeepLearningGenerator
from .backend_file_generator import BackendFileGenerator
from .constants import FrameworkType
from .shared_utils import get_logger

logger = get_logger(__name__)


def _validate_project_info(project_info: Dict[str, Any]) -> None:
    """
    Validar información del proyecto (función pura).
    
    Args:
        project_info: Información del proyecto
        
    Raises:
        ValueError: Si la información es inválida
    """
    if not project_info:
        raise ValueError("project_info cannot be empty")
    
    if 'name' not in project_info:
        raise ValueError("project_info must contain 'name'")
    
    if not project_info['name']:
        raise ValueError("project name cannot be empty")


def _create_directory_structure(base_dir: Path, directories: list[str]) -> None:
    """
    Crear estructura de directorios (función pura).
    
    Args:
        base_dir: Directorio base
        directories: Lista de directorios a crear
    """
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)


def _write_file_safe(file_path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Escribir archivo de forma segura (función pura).
    
    Args:
        file_path: Ruta del archivo
        content: Contenido a escribir
        encoding: Codificación del archivo
        
    Raises:
        IOError: Si no se puede escribir el archivo
    """
    try:
        file_path.write_text(content, encoding=encoding)
    except IOError as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        raise


class BackendGenerator:
    """
    Generador de estructura de backend.
    Optimizado con funciones puras y mejor manejo de errores.
    """

    def __init__(self, framework: str = FrameworkType.FASTAPI) -> None:
        """
        Inicializa el generador de backend.

        Args:
            framework: Framework a usar (fastapi, flask, django)
            
        Raises:
            ValueError: Si el framework no es soportado
        """
        if framework not in [FrameworkType.FASTAPI, FrameworkType.FLASK, FrameworkType.DJANGO]:
            raise ValueError(f"Unsupported framework: {framework}")
        
        self.framework = framework
        self.dl_generator = DeepLearningGenerator()
        self.file_generator = BackendFileGenerator()

    async def generate(
        self,
        project_dir: Path,
        description: str,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Genera la estructura completa del backend.

        Args:
            project_dir: Directorio donde generar el backend
            description: Descripción del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto

        Returns:
            Información del backend generado
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se puede crear la estructura
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not description:
            raise ValueError("description cannot be empty")
        
        _validate_project_info(project_info)
        
        project_dir.mkdir(parents=True, exist_ok=True)

        if self.framework == FrameworkType.FASTAPI:
            return await self._generate_fastapi(project_dir, description, keywords, project_info)
        
        raise ValueError(f"Framework {self.framework} not yet supported")

    async def _generate_fastapi(
        self,
        project_dir: Path,
        description: str,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Genera estructura FastAPI"""
        logger.info("Generando estructura FastAPI...")
        
        self.file_generator.create_directory_structure(project_dir)
        
        if keywords.get("is_deep_learning") or keywords.get("requires_pytorch"):
            logger.info("Generando código especializado de Deep Learning...")
            self.dl_generator.generate_all(project_dir, keywords, project_info)
            if keywords.get("requires_training"):
                self._generate_training_script(project_dir, keywords, project_info)

        self.file_generator.generate_main_py(project_dir, project_info, description, keywords)
        self.file_generator.generate_init_files(project_dir)
        self.file_generator.generate_config_files(project_dir, keywords)
        self.file_generator.generate_api_files(project_dir, project_info, description, keywords)
        self.file_generator.generate_service_files(project_dir, keywords, project_info)
        self.file_generator.generate_requirements_txt(project_dir, keywords)
        self.file_generator.generate_env_example(project_dir, keywords)
        self.file_generator.generate_dockerfile(project_dir)
        self.file_generator.generate_readme(project_dir, project_info, description, keywords)

        return {
            "framework": "FastAPI",
            "port": 8000,
            "status": "generated",
            "structure": {
                "main": "main.py",
                "app": "app/",
                "api": "app/api/",
                "services": "app/services/",
                "models": "app/models/",
            }
        }
    
    def _generate_training_script(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera script de entrenamiento para modelos de Deep Learning"""
        
        training_dir = project_dir / "app" / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        if keywords.get("is_transformer") or keywords.get("is_llm"):
            script_content = f'''"""
Training Script - Script de entrenamiento para modelos Transformer/LLM
{'=' * 60}

Script completo para entrenar modelos Transformer siguiendo mejores prácticas.
"""

import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from app.models.transformer_model import TransformerModel, FineTuner
from app.utils.training_utils import Trainer as CustomTrainer, EarlyStopping
from app.utils.eval_utils import evaluate_model
from app.core.config import settings
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Función principal de entrenamiento"""
    
    # Configuración
    device = settings.DEVICE
    model_name = settings.MODEL_NAME
    output_dir = settings.CHECKPOINT_DIR
    batch_size = settings.BATCH_SIZE
    learning_rate = settings.LEARNING_RATE
    num_epochs = settings.NUM_EPOCHS
    
    logger.info(f"Iniciando entrenamiento en {{device}}")
    logger.info(f"Modelo: {{model_name}}")
    logger.info(f"Batch size: {{batch_size}}, Learning rate: {{learning_rate}}, Epochs: {{num_epochs}}")
    
    # Cargar modelo
    model = TransformerModel(
        model_name=model_name,
        num_labels=settings.NUM_LABELS,
        task_type=settings.TASK_TYPE,
        device=device,
    )
    
    # TODO: Cargar tus datasets aquí
    # train_dataset = load_your_dataset("train")
    # eval_dataset = load_your_dataset("eval")
    
    # Crear fine-tuner
    fine_tuner = FineTuner(
        model=model,
        training_args={{
            "output_dir": output_dir,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": settings.GRADIENT_ACCUMULATION_STEPS,
            "fp16": settings.USE_MIXED_PRECISION and device == "cuda",
            "logging_steps": 100,
            "eval_steps": 500,
            "save_steps": 1000,
            "load_best_model_at_end": True,
        }}
    )
    
    # Entrenar
    # trainer = fine_tuner.train(train_dataset, eval_dataset)
    
    # Guardar modelo final
    final_model_path = Path(output_dir) / "final_model"
    model.save(str(final_model_path))
    
    logger.info("Entrenamiento completado")
    logger.info(f"Modelo guardado en {{final_model_path}}")


if __name__ == "__main__":
    main()
'''
        else:
            script_content = f'''"""
Training Script - Script de entrenamiento para modelos de Deep Learning
{'=' * 60}

Script completo para entrenar modelos personalizados siguiendo mejores prácticas.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from app.models.custom_model import CustomModel
from app.utils.training_utils import Trainer, EarlyStopping
from app.utils.data_loader import create_dataloader
from app.utils.eval_utils import evaluate_model
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Función principal de entrenamiento"""
    
    # Configuración
    device = settings.DEVICE
    batch_size = settings.BATCH_SIZE
    learning_rate = settings.LEARNING_RATE
    num_epochs = settings.NUM_EPOCHS
    
    logger.info(f"Iniciando entrenamiento en {{device}}")
    
    # Cargar modelo
    model = CustomModel(
        input_size=768,  # Ajustar según tu caso
        num_classes=10,  # Ajustar según tu caso
    )
    
    # Optimizador y scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Criterio de pérdida
    criterion = nn.CrossEntropyLoss()
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_amp=settings.USE_MIXED_PRECISION,
        gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=settings.MAX_GRAD_NORM,
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=7, mode="min")
    
    # TODO: Cargar tus datasets
    # train_dataset = YourDataset("train")
    # eval_dataset = YourDataset("eval")
    # train_loader = create_dataloader(train_dataset, batch_size=batch_size)
    # eval_loader = create_dataloader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Loop de entrenamiento
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"Época {{epoch + 1}}/{{num_epochs}}")
        
        # Entrenamiento
        model.train()
        train_loss = 0.0
        
        # for batch in train_loader:
        #     batch = {{k: v.to(device) if isinstance(v, torch.Tensor) else v 
        #              for k, v in batch.items()}}
        #     metrics = trainer.train_step(batch)
        #     train_loss += metrics["loss"]
        
        train_loss /= len(train_loader) if 'train_loader' in locals() else 1
        
        # Evaluación
        # val_metrics = evaluate_model(model, eval_loader, device, criterion)
        # val_loss = val_metrics.get("loss", 0.0)
        
        logger.info(f"Train Loss: {{train_loss:.4f}}")
        # logger.info(f"Val Loss: {{val_loss:.4f}}")
        
        # Scheduler step
        scheduler.step()
        
        # Early stopping
        # if early_stopping(val_loss):
        #     logger.info("Early stopping activado")
        #     break
        
        # Guardar checkpoint
        if settings.SAVE_CHECKPOINTS:
            checkpoint_path = Path(settings.CHECKPOINT_DIR) / f"checkpoint_epoch_{{epoch + 1}}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(checkpoint_path, epoch, {{"train_loss": train_loss}})
    
    logger.info("Entrenamiento completado")


if __name__ == "__main__":
    main()
'''
        
        (training_dir / "train.py").write_text(script_content, encoding="utf-8")
        (training_dir / "__init__.py").write_text('"""Training scripts"""', encoding="utf-8")

