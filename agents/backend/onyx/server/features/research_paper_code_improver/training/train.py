"""
Training Script - Script principal de entrenamiento
=====================================================
"""

import logging
import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_trainer import ModelTrainer
from core.paper_extractor import PaperExtractor

logger = logging.getLogger(__name__)


def train_model_from_papers(
    paper_sources: list,
    model_name: str = "gpt-4",
    epochs: int = 3,
    output_dir: str = "data/models"
):
    """
    Script principal para entrenar modelo desde papers.
    
    Args:
        paper_sources: Lista de rutas PDF o URLs
        model_name: Nombre del modelo base
        epochs: Número de épocas
        output_dir: Directorio de salida
    """
    try:
        logger.info("Iniciando entrenamiento de modelo desde papers...")
        
        # Extraer papers
        extractor = PaperExtractor()
        papers = extractor.extract_batch(paper_sources)
        
        if not papers:
            raise ValueError("No se pudieron extraer papers de las fuentes proporcionadas")
        
        logger.info(f"Papers extraídos: {len(papers)}")
        
        # Entrenar modelo
        trainer = ModelTrainer(model_name=model_name)
        model_path = trainer.train_from_papers(papers=papers, epochs=epochs)
        
        logger.info(f"Modelo entrenado exitosamente: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        raise


if __name__ == "__main__":
    # Ejemplo de uso
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar modelo desde papers")
    parser.add_argument("--papers", nargs="+", required=True, help="Rutas PDF o URLs de papers")
    parser.add_argument("--model", default="gpt-4", help="Nombre del modelo base")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas")
    parser.add_argument("--output", default="data/models", help="Directorio de salida")
    
    args = parser.parse_args()
    
    train_model_from_papers(
        paper_sources=args.papers,
        model_name=args.model,
        epochs=args.epochs,
        output_dir=args.output
    )




