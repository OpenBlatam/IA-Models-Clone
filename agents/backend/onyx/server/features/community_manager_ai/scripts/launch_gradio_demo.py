"""
Launch Gradio Demo - Lanzar Demo Interactivo
=============================================

Script para lanzar el demo interactivo de Gradio.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from community_manager_ai.ml.gradio_demo import GradioDemo


def main():
    parser = argparse.ArgumentParser(description="Lanzar demo interactivo de Gradio")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Puerto para el servidor (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Crear enlace público compartible"
    )
    parser.add_argument(
        "--no-image-gen",
        action="store_true",
        help="Deshabilitar generación de imágenes"
    )
    
    args = parser.parse_args()
    
    demo = GradioDemo(enable_image_gen=not args.no_image_gen)
    demo.launch(
        share=args.share,
        server_port=args.port
    )


if __name__ == "__main__":
    main()




