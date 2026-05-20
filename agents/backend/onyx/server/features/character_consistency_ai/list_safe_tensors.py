"""
Listar Safe Tensors - Interfaz Interactiva
==========================================

Muestra todos los safe tensors disponibles y permite acceder a ellos fácilmente.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

# Fix encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def list_safe_tensors(output_dir="./character_embeddings"):
    """Lista todos los safe tensors disponibles."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"[!] No existe el directorio: {output_dir}")
        print(f"[*] Los safe tensors se guardaran aqui cuando los generes.")
        return []
    
    safe_tensors = []
    
    # Buscar todos los archivos .safetensors
    for tensor_file in output_path.glob("*.safetensors"):
        tensor_info = {
            "filename": tensor_file.name,
            "path": str(tensor_file.absolute()),
            "size": tensor_file.stat().st_size,
            "size_mb": tensor_file.stat().st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(tensor_file.stat().st_mtime),
            "metadata": None
        }
        
        # Cargar metadata si existe
        metadata_path = tensor_file.with_suffix(".json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    tensor_info["metadata"] = json.load(f)
            except Exception as e:
                print(f"⚠️  Error cargando metadata de {tensor_file.name}: {e}")
        
        safe_tensors.append(tensor_info)
    
    return safe_tensors


def display_safe_tensors(safe_tensors):
    """Muestra los safe tensors de forma interactiva."""
    print("\n" + "=" * 80)
    print("SAFE TENSORS DISPONIBLES")
    print("=" * 80)
    
    if not safe_tensors:
        print("\n[!] No se encontraron safe tensors.")
        print("\n[*] Para generar uno:")
        print("   python generate_safe_tensors.py generate image1.jpg --name MyCharacter")
        return
    
    print(f"\n[OK] Encontrados {len(safe_tensors)} safe tensor(s):\n")
    
    for i, tensor in enumerate(safe_tensors, 1):
        print(f"{i}. [FILE] {tensor['filename']}")
        print(f"   [PATH] Ruta: {tensor['path']}")
        print(f"   [SIZE] Tamano: {tensor['size_mb']:.2f} MB ({tensor['size']:,} bytes)")
        print(f"   [DATE] Creado: {tensor['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if tensor.get('metadata'):
            meta = tensor['metadata']
            print(f"   [CHAR] Personaje: {meta.get('character_name', 'N/A')}")
            print(f"   [IMGS] Imagenes: {meta.get('num_images', 'N/A')}")
            print(f"   [DIM]  Dimension: {meta.get('embedding_dim', 'N/A')}")
            print(f"   [MODEL] Modelo: {meta.get('model_id', 'N/A')}")
        
        print()
    
    print("=" * 80)
    print("\n[*] Para usar estos safe tensors:")
    print("   from safetensors.torch import load_file")
    print("   data = load_file('ruta/al/archivo.safetensors')")
    print("   embedding = data['character_embedding']")


def create_html_viewer(safe_tensors, output_dir="./character_embeddings"):
    """Crea un archivo HTML para ver los safe tensors en el navegador."""
    html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Safe Tensors - Character Consistency AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .tensor-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .tensor-card:hover {
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }
        .tensor-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .tensor-name {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        .tensor-size {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .tensor-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .info-item {
            display: flex;
            flex-direction: column;
        }
        .info-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }
        .info-value {
            font-size: 1.1em;
            color: #333;
            font-weight: 500;
        }
        .tensor-path {
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            word-break: break-all;
            margin-top: 10px;
        }
        .actions {
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover {
            background: #5568d3;
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .empty-state h2 {
            font-size: 2em;
            margin-bottom: 20px;
        }
        .code-block {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📦 Safe Tensors</h1>
        <p class="subtitle">Character Consistency AI - Embeddings Disponibles</p>
        
        <div id="tensors-container">
"""
    
    if not safe_tensors:
        html_content += """
            <div class="empty-state">
                <h2>📭 No hay safe tensors aún</h2>
                <p>Genera tu primer safe tensor usando:</p>
                <div class="code-block">
python generate_safe_tensors.py generate image1.jpg --name MyCharacter
                </div>
            </div>
        """
    else:
        for tensor in safe_tensors:
            meta = tensor.get('metadata', {})
            html_content += f"""
            <div class="tensor-card">
                <div class="tensor-header">
                    <div class="tensor-name">📄 {tensor['filename']}</div>
                    <div class="tensor-size">{tensor['size_mb']:.2f} MB</div>
                </div>
                
                <div class="tensor-info">
                    <div class="info-item">
                        <span class="info-label">👤 Personaje</span>
                        <span class="info-value">{meta.get('character_name', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">🖼️ Imágenes</span>
                        <span class="info-value">{meta.get('num_images', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">🔢 Dimensión</span>
                        <span class="info-value">{meta.get('embedding_dim', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">📅 Creado</span>
                        <span class="info-value">{tensor['created'].strftime('%Y-%m-%d %H:%M')}</span>
                    </div>
                </div>
                
                <div class="tensor-path">
                    📁 {tensor['path']}
                </div>
                
                <div class="actions">
                    <button class="btn btn-primary" onclick="copyPath('{tensor['path'].replace(chr(92), '/')}')">
                        📋 Copiar Ruta
                    </button>
                    <button class="btn btn-secondary" onclick="showCode('{tensor['path'].replace(chr(92), '/')}')">
                        💻 Ver Código
                    </button>
                </div>
                
                <div id="code-{tensor['filename']}" style="display:none;">
                    <div class="code-block">
from safetensors.torch import load_file<br>
import torch<br><br>
# Cargar embedding<br>
data = load_file(r"{tensor['path'].replace(chr(92), '/')}")<br>
embedding = data["character_embedding"]<br>
print(f"Shape: {{embedding.shape}}")
                    </div>
                </div>
            </div>
            """
    
    html_content += """
        </div>
    </div>
    
    <script>
        function copyPath(path) {
            navigator.clipboard.writeText(path).then(() => {
                alert('Ruta copiada al portapapeles: ' + path);
            });
        }
        
        function showCode(filename) {
            const codeId = 'code-' + filename;
            const codeDiv = document.getElementById(codeId);
            if (codeDiv.style.display === 'none') {
                codeDiv.style.display = 'block';
            } else {
                codeDiv.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""
    
    # Asegurar que el directorio existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    html_path = output_path / "safe_tensors_viewer.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return html_path


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Listar Safe Tensors")
    parser.add_argument("--output-dir", default="./character_embeddings", help="Directorio de safe tensors")
    parser.add_argument("--html", action="store_true", help="Crear vista HTML")
    parser.add_argument("--open", action="store_true", help="Abrir HTML en navegador")
    
    args = parser.parse_args()
    
    # Listar safe tensors
    safe_tensors = list_safe_tensors(args.output_dir)
    
    # Mostrar en consola
    display_safe_tensors(safe_tensors)
    
    # Crear HTML si se solicita
    if args.html or args.open:
        html_path = create_html_viewer(safe_tensors, args.output_dir)
        print(f"\n[OK] Vista HTML creada: {html_path}")
        
        if args.open:
            print(f"[*] Abriendo en navegador...")
            webbrowser.open(f"file://{html_path.absolute()}")


if __name__ == "__main__":
    main()

